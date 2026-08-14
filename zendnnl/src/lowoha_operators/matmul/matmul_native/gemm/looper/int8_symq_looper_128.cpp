/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

// ============================================================================
// Looper for the symmetric per-group INT8 microkernel.
//
// Same shape as the BF16 128-bit looper next door, for the same reasons:
// parallelise over column panels so each thread owns a disjoint slice of C and
// no reduction is needed, and pack B per panel into the layout the microkernel
// reads. K is blocked, to keep the slice of the packed panel a thread is walking
// inside L1 -- see the block-size comment below for why, and for what it is
// worth. An earlier version of this file walked K straight through and argued
// against blocking on the grounds that the microkernel already flushes once per
// weight-scale group; true, but it was answering a question about redundant work
// when the one that mattered was about footprint.
//
// A is used unpacked. It is read as dwords -- four consecutive K of one row are
// one broadcast -- so a packed copy would buy nothing but a pass over M*K bytes.
// The one exception is an M that is not a whole number of MR: the microkernel
// always reads MR rows, so the last M panel is copied into a zero-padded MR-row
// scratch and its results are taken from a scratch C tile. Without that, every
// tile of an M=1 decode falls to the scalar tail -- mr_act is 1, the vector path
// wants 4 -- and the whole shape runs at reference speed. Padding costs MR*K
// bytes of copy per call against a pass over the entire weight.
//
// B is packed once per weight rather than once per call, through the INT8
// prepacked-weight cache the BRGEMM path already uses. That cache's layout is
// this kernel's layout and not merely a compatible one: NR_PACK is 64, matching
// the panel width chosen below, INT8_VNNI_GRP is 4, and its packer lays out
// panel-major quads with columns past N and K past the end zero-filled -- byte
// for byte what pack_b_panel() writes. So a cached panel goes to the microkernel
// as it stands and the results are identical rather than close. K_padded never
// differs from K here, because this path already requires K to be a whole number
// of groups and a group to be a whole number of quads.
//
// is_weights_const is the caller's promise that the buffer will not be written
// behind us; without it, caching by pointer would serve stale weights. The gate
// matches the other loopers'.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

// Panel width in columns. Eight would let the microkernel read B in place, but
// then the packing loop runs once per eight columns and its overhead dominates;
// sixty-four amortises it over eight microkernel calls while keeping a panel's
// packed B (K * 64 bytes) small enough to stay resident while the M loop walks
// over it. It is also NR_PACK, which is what lets the shared prepacked-weight
// cache be read in place -- keep the two equal.
constexpr int kPanelW = 64;
static_assert(kPanelW == NR_PACK,
        "panel width must match the prepacked-weight cache's NR_PACK");
static_assert(SYMQ_VNNI_GRP == INT8_VNNI_GRP,
        "VNNI quad must match the prepacked-weight cache's group");

// Pack one panel of B into VNNI quads: for each group of four consecutive K,
// the panel's columns lie contiguously, four bytes each.
//
// Columns past N and K past the end are zeroed rather than skipped, so the
// microkernel can always read a full quad and a full panel width; zeros
// contribute nothing to the dot product.
void pack_b_panel(const int8_t *B, int ldb, bool transB, int jc, int nb_act,
        int N, int K, int8_t *dst) {
    const int n_quads = K / SYMQ_VNNI_GRP;
    const int b_stride = kPanelW * SYMQ_VNNI_GRP;
    for (int kq = 0; kq < n_quads; ++kq) {
        int8_t *d = dst + static_cast<size_t>(kq) * b_stride;
        for (int n = 0; n < kPanelW; ++n) {
            const int col = jc + n;
            for (int j = 0; j < SYMQ_VNNI_GRP; ++j) {
                const int k = kq * SYMQ_VNNI_GRP + j;
                int8_t v = 0;
                if (n < nb_act && col < N && k < K) {
                    v = transB ? B[static_cast<size_t>(col) * ldb + k]
                               : B[static_cast<size_t>(k) * ldb + col];
                }
                d[n * SYMQ_VNNI_GRP + j] = v;
            }
        }
    }
}

} // namespace

bool int8_symq_execute_128(int M, int N, int K, int group_size,
        const int8_t *A, int lda, const int8_t *B, int ldb, bool transB,
        float *C, int ldc, const float *wei_scale, const float *src_scale,
        int ss_row, int ss_grp, int nthreads,
        const INT8PrepackedWeight *prepacked) {

    if (M <= 0 || N <= 0 || K <= 0) return false;
    // The microkernel flushes once per group and reads whole quads, so a K that
    // is not a whole number of groups, or a group that splits a quad, has no
    // expression here. Refused rather than approximated.
    if (group_size <= 0 || group_size % SYMQ_VNNI_GRP != 0) return false;
    if (K % group_size != 0) return false;

    const int8_symq_ukernel_128_fn_t hot = select_int8_symq_ukernel_128();
    if (hot == nullptr) return false;
    // One-row tiles go to a kernel that broadcasts one row instead of four. The
    // padded path below is still correct for them and was what they used before,
    // but it computed four rows to keep one; this needs no padded A, no scratch C
    // and no copy back.
    const int8_symq_ukernel_128_fn_t hot_m1
            = select_int8_symq_ukernel_128_m1();

    const int n_panels = (N + kPanelW - 1) / kPanelW;
    const int b_stride = kPanelW * SYMQ_VNNI_GRP;
    const int n_quads = K / SYMQ_VNNI_GRP;

    // K is blocked so the slice of the packed panel in play stays in L1.
    //
    // The first version of this looper walked K straight through, on the
    // reasoning that the microkernel already flushes its accumulators once per
    // weight-scale group and a K block would only add a coarser flush over the
    // top. That reasoning was about redundant work and it was correct as far as
    // it went; what it missed is footprint. A panel is (K/4) * 256 bytes, so at
    // K=4096 a thread walks 256 KB of packed B per column panel, against 32 KB of
    // L1d on Excavator, 16 KB on Piledriver, 1 MB of L2 shared by the two cores
    // of a module, and no L3 anywhere on this family.
    //
    // Blocking K to a 32 KB slice is worth, measured on an idle A10-8770E at four
    // threads and an FX-8370E at eight, best of five:
    //
    //     shape                     A10             FX-8370E
    //     M=128  N=K=4096    36.2 -> 73.0     116.5 -> 136.3
    //     M=512  N=K=4096    44.4 -> 72.6     118.4 -> 138.4
    //     M=128  group 16    22.6 -> 63.2      89.4 -> 104.7
    //     M=1    N=K=4096    11.3 -> 14.9      36.0 -> 37.2
    //
    // On the A10 that closes the gap to the square-1024 shape entirely (71.7),
    // which is what first suggested footprint rather than arithmetic: same work
    // per byte, a quarter of the panel.
    //
    // 512 elements rather than a per-microarchitecture size on purpose. 1024 is
    // marginally better for group 32 and much worse for group 16 on Excavator
    // (39.8 against 63.2), 256 is better for group 16 on Piledriver and worse for
    // the square shape, and 2048 gives most of the gain back on both. 512 is
    // within noise of the best everywhere measured, and one number that is never
    // wrong beats two that are each right on one machine.
    //
    // ZENDNNL_SYMQ_KBLOCK overrides it for tuning; 0 restores the unblocked walk.
    constexpr int kTargetPanelBytes = 32 * 1024;
    static const int s_kblock_env = [] {
        const char *v = std::getenv("ZENDNNL_SYMQ_KBLOCK");
        return (v != nullptr && v[0] != '\0') ? std::atoi(v) : -1;
    }();
    int k_block = s_kblock_env >= 0
            ? s_kblock_env
            : (kTargetPanelBytes / b_stride) * SYMQ_VNNI_GRP;
    if (k_block <= 0 || k_block > K) {
        k_block = K;
    } else {
        // A group boundary inside a block has no expression in the microkernel.
        k_block = (k_block / group_size) * group_size;
        if (k_block <= 0) k_block = K;
    }

    int nt = nthreads > 0 ? nthreads : 1;
    nt = std::min(nt, std::max(n_panels, 1));

    // A prepacked weight must describe this problem; anything else would read
    // the wrong bytes rather than merely miss the optimisation.
    if (prepacked != nullptr
            && (prepacked->K != K || prepacked->N != N
                    || prepacked->K_padded != K)) {
        return false;
    }

    // One pass over C: the microkernel accumulates into it, so it starts at
    // zero rather than being read.
    for (int m = 0; m < M; ++m)
        std::memset(C + static_cast<size_t>(m) * ldc, 0, sizeof(float) * N);

    // Zero-padded copy of the rows the last M panel is short of, so the vector
    // kernel can be used there too. Zeros are inside the byte contract and
    // contribute nothing to a dot product, and the rows they produce are dropped
    // with the scratch C tile. Built once per call and shared read-only by the
    // threads, which each own different columns of the same rows.
    const int m_tail = M % SYMQ_MR;
    std::vector<int8_t> a_pad;
    // Row-indexed activation scales need the same padding as the rows they
    // describe: the kernel reads MR of them, and only m_tail exist. ss_row
    // elements per row covers per-token (1) and per-group (G) alike. The padding
    // is zero, which is doubly safe -- the rows it scales are zero anyway.
    std::vector<float> ss_pad;
    if (m_tail != 0) {
        a_pad.assign(static_cast<size_t>(SYMQ_MR) * K, 0);
        const int ic0 = M - m_tail;
        for (int m = 0; m < m_tail; ++m) {
            std::memcpy(a_pad.data() + static_cast<size_t>(m) * K,
                    A + static_cast<size_t>(ic0 + m) * lda, K);
        }
        if (ss_row != 0) {
            ss_pad.assign(static_cast<size_t>(SYMQ_MR) * ss_row, 0.0f);
            for (int m = 0; m < m_tail; ++m) {
                std::memcpy(ss_pad.data() + static_cast<size_t>(m) * ss_row,
                        src_scale + static_cast<size_t>(ic0 + m) * ss_row,
                        sizeof(float) * ss_row);
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
        // No strip to own when B comes from the cache: those panels are
        // read-only and shared, so the threads read them in place.
        std::vector<int8_t> b_panel;
        if (prepacked == nullptr)
            b_panel.assign(static_cast<size_t>(n_quads) * b_stride, 0);
        // Scratch C for the padded M tail: the microkernel writes MR rows, and
        // only mr_act of them belong to C. Column tails still go to the scalar
        // tail, which writes through to C directly.
        std::vector<float> c_scratch;
        if (m_tail != 0)
            c_scratch.assign(static_cast<size_t>(SYMQ_MR) * SYMQ_NR, 0.0f);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int p = 0; p < n_panels; ++p) {
            const int jc = p * kPanelW;
            const int nb_act = std::min(kPanelW, N - jc);

            const int8_t *panel_base;
            if (prepacked != nullptr) {
                panel_base = prepacked->get_panel(/*kq=*/0, p);
            } else {
                pack_b_panel(B, ldb, transB, jc, nb_act, N, K, b_panel.data());
                panel_base = b_panel.data();
            }

            // The microkernel accumulates into C, so K blocks compose with no
            // extra flush and no re-scaling; the only cost is revisiting C once
            // per block, which is M*N*4 bytes against a block's worth of
            // arithmetic.
            for (int kb = 0; kb < K; kb += k_block) {
                const int kb_act = std::min(k_block, K - kb);
            for (int ic = 0; ic < M; ic += SYMQ_MR) {
                const int mr_act = std::min(SYMQ_MR, M - ic);

                for (int jr = 0; jr < nb_act; jr += SYMQ_NR) {
                    const int nr_act = std::min(SYMQ_NR, nb_act - jr);
                    const int8_t *b_tile = panel_base + jr * SYMQ_VNNI_GRP
                            + static_cast<size_t>(kb / SYMQ_VNNI_GRP)
                                    * b_stride;
                    const int8_t *a_tile
                            = A + static_cast<size_t>(ic) * lda + kb;
                    float *c_tile
                            = C + static_cast<size_t>(ic) * ldc + jc + jr;
                    // Scales are group-major over the full N, so the column
                    // offset goes into the pointer and the stride stays N.
                    const float *ws = wei_scale + jc + jr
                            + static_cast<size_t>(kb / group_size) * N;
                    // The activation scale is indexed by row, so the M offset
                    // goes into the pointer; ss_row is 0 for a per-tensor scale
                    // and the offset then correctly does nothing.
                    const float *ss = src_scale
                            + static_cast<size_t>(ic) * ss_row
                            + static_cast<size_t>(kb / group_size) * ss_grp;

                    if (mr_act == SYMQ_MR && nr_act == SYMQ_NR) {
                        hot(a_tile, lda, b_tile, b_stride, c_tile, ldc, kb_act,
                                group_size, ws, N, ss, ss_row, ss_grp);
                    } else if (mr_act == 1 && nr_act == SYMQ_NR
                            && hot_m1 != nullptr) {
                        // The decode shape: straight at the real row, writing
                        // straight into C.
                        hot_m1(a_tile, lda, b_tile, b_stride, c_tile, ldc,
                                kb_act, group_size, ws, N, ss, ss_row, ss_grp);
                        // mr_act < MR only ever happens on the last M panel,
                        // which is exactly the rows a_pad holds.
                    } else if (nr_act == SYMQ_NR && mr_act < SYMQ_MR
                            && m_tail != 0) {
                        // Short on rows only: run the vector kernel over the
                        // zero-padded copy and keep the rows that exist. The
                        // padded rows read scales past row M-1, so the kernel is
                        // pointed at a padded scale copy too rather than off the
                        // end of the caller's buffer.
                        std::memset(c_scratch.data(), 0,
                                sizeof(float) * c_scratch.size());
                        hot(a_pad.data() + kb, K, b_tile, b_stride,
                                c_scratch.data(), SYMQ_NR, kb_act, group_size,
                                ws, N,
                                ss_row == 0 ? src_scale : ss_pad.data(), ss_row,
                                ss_grp);
                        for (int m = 0; m < mr_act; ++m) {
                            float *dst = c_tile + static_cast<size_t>(m) * ldc;
                            const float *src = c_scratch.data()
                                    + static_cast<size_t>(m) * SYMQ_NR;
                            // Accumulate, not copy: with K blocked this tile is
                            // visited once per block.
                            for (int n = 0; n < SYMQ_NR; ++n) dst[n] += src[n];
                        }
                    } else {
                        int8_symq_tail_128(a_tile, lda, b_tile, b_stride,
                                c_tile, ldc, kb_act, group_size, mr_act, nr_act,
                                ws, N, ss, ss_row, ss_grp);
                    }
                }
            }
            }
        }
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
