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
// Looper for the asymmetric per-group (k-quant) INT8 microkernel.
//
// The symmetric looper next door is the template and the two agree on every
// structural decision it arrived at by measurement, because none of those
// reasons change when the weights gain a min term:
//
//   * parallelise over column panels, so threads own disjoint slices of C
//   * pack B per panel into the microkernel's VNNI layout
//   * block K to a 32 KB slice of the packed panel, which on family 15h is worth
//     roughly 2x on prompt shapes -- a panel is (K/4)*256 bytes against 32 KB of
//     L1d, 1 MB of L2 shared by a module's two cores, and no L3
//   * pad a short M into an MR-row scratch, and route a single row to the
//     one-row microkernel, so decode does not fall to the scalar tail
//
// What is new here is the row sums. The min term needs sum(A) over each group
// per row, it does not depend on the column, and it is the same for every K
// block, so it is built once per call before the panel loop -- M*K additions
// against M*N*K multiply-adds. Building it inside the panel loop would repeat it
// once per column panel, which at N=4096 is sixty-four times.
//
// B is packed per panel per call. The symmetric path shares the INT8 prepacked
// weight cache for this and the same would work here, the codes being bytes like
// any other; it is left out of this first version deliberately, exactly as it was
// there, so that a correctness failure cannot be ambiguous between the kernel and
// a cache.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_looper_128.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_kquant_ukernel_128.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

constexpr int kPanelW = 64;

// Pack one panel of unsigned codes into VNNI quads. Columns past N and K past the
// end are zeroed rather than skipped: a zero code contributes nothing to the dot
// product, and the min term for those columns is never read back.
void pack_q_panel(const uint8_t *B, int ldb, bool transB, int jc, int nb_act,
        int N, int K, uint8_t *dst) {
    const int n_quads = K / KQ_VNNI_GRP;
    const int b_stride = kPanelW * KQ_VNNI_GRP;
    for (int kq = 0; kq < n_quads; ++kq) {
        uint8_t *d = dst + static_cast<size_t>(kq) * b_stride;
        for (int n = 0; n < kPanelW; ++n) {
            const int col = jc + n;
            for (int j = 0; j < KQ_VNNI_GRP; ++j) {
                const int k = kq * KQ_VNNI_GRP + j;
                uint8_t v = 0;
                if (n < nb_act && col < N && k < K) {
                    v = transB ? B[static_cast<size_t>(col) * ldb + k]
                               : B[static_cast<size_t>(k) * ldb + col];
                }
                d[n * KQ_VNNI_GRP + j] = v;
            }
        }
    }
}

} // namespace

bool int8_kquant_execute_128(int M, int N, int K, int group_size,
        const int8_t *A, int lda, const uint8_t *B, int ldb, bool transB,
        float *C, int ldc, const float *wei_scale, const float *wei_min,
        const float *src_scale, int ss_row, int ss_grp, int nthreads) {

    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (group_size <= 0 || group_size % KQ_VNNI_GRP != 0) return false;
    if (K % group_size != 0) return false;

    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    if (hot == nullptr) return false;
    const int8_kquant_ukernel_128_fn_t hot_m1
            = select_int8_kquant_ukernel_128_m1();

    const int n_panels = (N + kPanelW - 1) / kPanelW;
    const int b_stride = kPanelW * KQ_VNNI_GRP;
    const int n_quads = K / KQ_VNNI_GRP;
    const int n_groups = K / group_size;

    // See the symmetric looper for the measurement behind 32 KB.
    constexpr int kTargetPanelBytes = 32 * 1024;
    static const int s_kblock_env = [] {
        const char *v = std::getenv("ZENDNNL_SYMQ_KBLOCK");
        return (v != nullptr && v[0] != '\0') ? std::atoi(v) : -1;
    }();
    int k_block = s_kblock_env >= 0
            ? s_kblock_env
            : (kTargetPanelBytes / b_stride) * KQ_VNNI_GRP;
    if (k_block <= 0 || k_block > K) {
        k_block = K;
    } else {
        k_block = (k_block / group_size) * group_size;
        if (k_block <= 0) k_block = K;
    }

    int nt = nthreads > 0 ? nthreads : 1;
    nt = std::min(nt, std::max(n_panels, 1));

    // One pass over C: the microkernel accumulates into it.
    for (int m = 0; m < M; ++m)
        std::memset(C + static_cast<size_t>(m) * ldc, 0, sizeof(float) * N);

    // Row sums, once per call. Independent of the column and of the K block, so
    // building them here rather than in the panel loop saves n_panels passes.
    std::vector<int32_t> row_sums(static_cast<size_t>(M) * n_groups);
    int8_kquant_row_sums(A, lda, M, K, group_size, row_sums.data());
    const int rs_stride = n_groups;

    // Zero-padded copy of the rows the last M panel is short of, so the vector
    // kernel can be used there too, with the matching row-sum and scale rows.
    const int m_tail = M % KQ_MR;
    std::vector<int8_t> a_pad;
    std::vector<int32_t> rs_pad;
    std::vector<float> ss_pad;
    if (m_tail != 0) {
        const int ic0 = M - m_tail;
        a_pad.assign(static_cast<size_t>(KQ_MR) * K, 0);
        rs_pad.assign(static_cast<size_t>(KQ_MR) * rs_stride, 0);
        for (int m = 0; m < m_tail; ++m) {
            std::memcpy(a_pad.data() + static_cast<size_t>(m) * K,
                    A + static_cast<size_t>(ic0 + m) * lda, K);
            std::memcpy(rs_pad.data() + static_cast<size_t>(m) * rs_stride,
                    row_sums.data()
                            + static_cast<size_t>(ic0 + m) * rs_stride,
                    sizeof(int32_t) * rs_stride);
        }
        if (ss_row != 0) {
            ss_pad.assign(static_cast<size_t>(KQ_MR) * ss_row, 0.0f);
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
        std::vector<uint8_t> b_panel(
                static_cast<size_t>(n_quads) * b_stride, 0);
        std::vector<float> c_scratch;
        if (m_tail != 0)
            c_scratch.assign(static_cast<size_t>(KQ_MR) * KQ_NR, 0.0f);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int p = 0; p < n_panels; ++p) {
            const int jc = p * kPanelW;
            const int nb_act = std::min(kPanelW, N - jc);

            pack_q_panel(B, ldb, transB, jc, nb_act, N, K, b_panel.data());

            for (int kb = 0; kb < K; kb += k_block) {
                const int kb_act = std::min(k_block, K - kb);
                const int g0 = kb / group_size;

                for (int ic = 0; ic < M; ic += KQ_MR) {
                    const int mr_act = std::min(KQ_MR, M - ic);

                    for (int jr = 0; jr < nb_act; jr += KQ_NR) {
                        const int nr_act = std::min(KQ_NR, nb_act - jr);
                        const uint8_t *b_tile = b_panel.data()
                                + jr * KQ_VNNI_GRP
                                + static_cast<size_t>(kb / KQ_VNNI_GRP)
                                        * b_stride;
                        const int8_t *a_tile
                                = A + static_cast<size_t>(ic) * lda + kb;
                        float *c_tile
                                = C + static_cast<size_t>(ic) * ldc + jc + jr;
                        // D and M are group-major over the full N, so the column
                        // offset goes into the pointer and the stride stays N.
                        const float *d_ptr = wei_scale + jc + jr
                                + static_cast<size_t>(g0) * N;
                        const float *m_ptr = wei_min + jc + jr
                                + static_cast<size_t>(g0) * N;
                        const int32_t *rs = row_sums.data()
                                + static_cast<size_t>(ic) * rs_stride + g0;
                        const float *ss = src_scale
                                + static_cast<size_t>(ic) * ss_row
                                + static_cast<size_t>(g0) * ss_grp;

                        if (mr_act == KQ_MR && nr_act == KQ_NR) {
                            hot(a_tile, lda, b_tile, b_stride, c_tile, ldc,
                                    kb_act, group_size, d_ptr, m_ptr, N, rs,
                                    rs_stride, ss, ss_row, ss_grp);
                        } else if (mr_act == 1 && nr_act == KQ_NR
                                && hot_m1 != nullptr) {
                            hot_m1(a_tile, lda, b_tile, b_stride, c_tile, ldc,
                                    kb_act, group_size, d_ptr, m_ptr, N, rs,
                                    rs_stride, ss, ss_row, ss_grp);
                        } else if (nr_act == KQ_NR && mr_act < KQ_MR
                                && m_tail != 0) {
                            std::memset(c_scratch.data(), 0,
                                    sizeof(float) * c_scratch.size());
                            hot(a_pad.data() + kb, K, b_tile, b_stride,
                                    c_scratch.data(), KQ_NR, kb_act, group_size,
                                    d_ptr, m_ptr, N,
                                    rs_pad.data() + g0, rs_stride,
                                    ss_row == 0 ? src_scale
                                                : ss_pad.data()
                                                    + static_cast<size_t>(g0)
                                                            * ss_grp,
                                    ss_row, ss_grp);
                            for (int m = 0; m < mr_act; ++m) {
                                float *dst
                                        = c_tile + static_cast<size_t>(m) * ldc;
                                const float *src = c_scratch.data()
                                        + static_cast<size_t>(m) * KQ_NR;
                                for (int n = 0; n < KQ_NR; ++n) dst[n] += src[n];
                            }
                        } else {
                            int8_kquant_tail_128(a_tile, lda, b_tile, b_stride,
                                    c_tile, ldc, kb_act, group_size, mr_act,
                                    nr_act, d_ptr, m_ptr, N, rs, rs_stride, ss,
                                    ss_row, ss_grp);
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
