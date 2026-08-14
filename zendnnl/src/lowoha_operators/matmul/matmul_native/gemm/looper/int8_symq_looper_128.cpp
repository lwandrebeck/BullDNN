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
// reads. What differs is that K is not blocked. The microkernel already flushes
// its s32 accumulators to fp32 once per weight-scale group, so a K block would
// add a second, coarser flush over the top of one that has to happen anyway,
// and would need C read back and re-accumulated at every block boundary.
// Walking K straight through leaves exactly one pass over C.
//
// A is used unpacked. It is read as dwords -- four consecutive K of one row are
// one broadcast -- so a packed copy would buy nothing but a pass over M*K bytes.
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
        float *C, int ldc, const float *wei_scale, float src_scale,
        int nthreads, const INT8PrepackedWeight *prepacked) {

    if (M <= 0 || N <= 0 || K <= 0) return false;
    // The microkernel flushes once per group and reads whole quads, so a K that
    // is not a whole number of groups, or a group that splits a quad, has no
    // expression here. Refused rather than approximated.
    if (group_size <= 0 || group_size % SYMQ_VNNI_GRP != 0) return false;
    if (K % group_size != 0) return false;

    const int8_symq_ukernel_128_fn_t hot = select_int8_symq_ukernel_128();
    if (hot == nullptr) return false;

    const int n_panels = (N + kPanelW - 1) / kPanelW;
    const int b_stride = kPanelW * SYMQ_VNNI_GRP;
    const int n_quads = K / SYMQ_VNNI_GRP;

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

#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
        // No strip to own when B comes from the cache: those panels are
        // read-only and shared, so the threads read them in place.
        std::vector<int8_t> b_panel;
        if (prepacked == nullptr)
            b_panel.assign(static_cast<size_t>(n_quads) * b_stride, 0);
        // Ragged tiles are finished by the scalar tail, which writes through to
        // C directly, so no scratch tile is needed here.

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

            for (int ic = 0; ic < M; ic += SYMQ_MR) {
                const int mr_act = std::min(SYMQ_MR, M - ic);

                for (int jr = 0; jr < nb_act; jr += SYMQ_NR) {
                    const int nr_act = std::min(SYMQ_NR, nb_act - jr);
                    const int8_t *b_tile = panel_base + jr * SYMQ_VNNI_GRP;
                    const int8_t *a_tile
                            = A + static_cast<size_t>(ic) * lda;
                    float *c_tile
                            = C + static_cast<size_t>(ic) * ldc + jc + jr;
                    // Scales are group-major over the full N, so the column
                    // offset goes into the pointer and the stride stays N.
                    const float *ws = wei_scale + jc + jr;

                    if (mr_act == SYMQ_MR && nr_act == SYMQ_NR) {
                        hot(a_tile, lda, b_tile, b_stride, c_tile, ldc, K,
                                group_size, ws, N, src_scale);
                    } else {
                        int8_symq_tail_128(a_tile, lda, b_tile, b_stride,
                                c_tile, ldc, K, group_size, mr_act, nr_act, ws,
                                N, src_scale);
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
