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
// BF16 GEMM looper for hosts without avx512bf16.
//
// A separate translation unit rather than a branch inside bf16_gemm_looper.cpp,
// because that file carries a blanket
//
//     #pragma GCC target("avx512f,avx512bf16,avx512bw,avx512vl,fma")
//
// over all of its contents -- including its entry point, so on a host without
// those features merely calling into it risks an illegal instruction. Removing
// that pragma would also stop GCC auto-vectorising those 700 lines with
// AVX-512, which would cost Zen4/5 throughput for no reason. The dispatch
// therefore happens before either file, in native_matmul.cpp.
//
// Structure, chosen for the same reason the FP32 side looks like it does:
// parallelise over column panels so each thread owns a disjoint slice of C and
// no reduction is needed, then walk K blocks inside. Each thread packs its own
// B strip, so nothing is shared but the read-only inputs.
//
// Two things differ from the AVX-512 looper by necessity. B is packed on the fly
// with the scalar VNNI packer rather than taken from the prepacked cache, whose
// fill path is not portable; and A is widened to FP32 per M-panel, because the
// microkernel wants FP32 A -- widening it there would cost three operations per
// row per k to build a value one FMA consumes. B stays BF16 all the way into
// the register, which is the point: it is the operand streamed repeatedly
// across M tiles, so keeping it half size is the whole benefit of the format on
// a machine with no BF16 arithmetic.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/bf16_gemm_looper_128.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/bf16_packing.hpp"
#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/bf16/bf16_gemm_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/planner/gemm_planner.hpp"
#include "operators/matmul/matmul_config.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

inline float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even, matching _mm512_cvtneps_pbh so the two paths do
    // not drift apart on the last mantissa bit.
    const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

// Scratch owned by one thread for the duration of a call. Plain new/delete
// rather than thread_local statics: this path is young, and a leak or a stale
// buffer keyed on the wrong shape is a worse bug than an allocation per call.
struct ThreadScratch {
    std::vector<uint16_t> b_strip; // packed VNNI B for one panel and K block
    std::vector<float> a_wide;     // FP32 A for one M panel and K block
};

} // namespace

bool bf16_gemm_execute_128(const GemmDescriptor &desc, const UarchParams &uarch,
        const void *src, const void *weight, void *dst, const void *bias,
        matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    if (M <= 0 || N <= 0 || K <= 0) return true;

    const BF16GemmPlan bp = plan_bf16_gemm(desc, uarch, params);
    const BlockPlan &plan = bp.plan;

    const int MR = plan.MR, NR = plan.NR;
    const int MB = plan.MB, KB = plan.KB;
    // Panels are NR_PACK wide because that is what the VNNI packer emits and
    // what the microkernel's b_stride assumes.
    const int panel_w = NR_PACK;

    if (select_bf16_ukernel_128(MR, NR) == nullptr) {
        // No 128-bit kernel for this shape; let the caller fall back.
        return false;
    }

    const uint16_t *A = static_cast<const uint16_t *>(src);
    const uint16_t *B = static_cast<const uint16_t *>(weight);
    const int lda = desc.lda, ldb = desc.ldb, ldc = desc.ldc;
    const bool transB = desc.transB;
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool bias_is_bf16 = (desc.bias_dt == data_type_t::bf16);

    const float alpha = desc.alpha;
    // Same trick the AVX-512 looper uses: hand the kernel beta/alpha and scale
    // the whole tile by alpha afterwards, so one multiply serves both.
    const float beta_eff = (alpha != 1.0f && desc.beta != 0.0f)
            ? (desc.beta / alpha)
            : desc.beta;

    // ---- C working buffer -------------------------------------------------
    // The microkernel writes FP32. For a BF16 destination that needs a scratch,
    // seeded from the existing values when beta asks for them.
    std::vector<float> c_scratch;
    float *C = nullptr;
    int ldc_f = 0;
    if (dst_is_bf16) {
        c_scratch.assign(static_cast<size_t>(M) * N, 0.0f);
        C = c_scratch.data();
        ldc_f = N;
        if (beta_eff != 0.0f) {
            const uint16_t *d = static_cast<const uint16_t *>(dst);
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n)
                    C[m * ldc_f + n] = bf16_to_f32(d[m * ldc + n]);
        }
    } else {
        C = static_cast<float *>(dst);
        ldc_f = ldc;
    }

    // ---- bias to FP32 -----------------------------------------------------
    std::vector<float> bias_store;
    const float *bias_f = nullptr;
    if (bias != nullptr) {
        if (bias_is_bf16) {
            bias_store.resize(N);
            const uint16_t *b16 = static_cast<const uint16_t *>(bias);
            for (int n = 0; n < N; ++n) bias_store[n] = bf16_to_f32(b16[n]);
            bias_f = bias_store.data();
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    const int k_pairs_kb = (KB + 1) / 2;
    const int b_stride = BF16PrepackedWeight::stride();
    const int n_panels = (N + panel_w - 1) / panel_w;

    int nthreads = plan.num_threads > 0 ? plan.num_threads : 1;
    nthreads = std::min(nthreads, std::max(n_panels, 1));

    // Postops and bias are applied after the K loop, not fused into the
    // microkernel: correctness first, and the epilogue is O(M*N) against
    // O(M*N*K) of arithmetic. Fusing is a later refinement.
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#endif
    {
        ThreadScratch s;
        s.b_strip.assign(
                static_cast<size_t>(k_pairs_kb) * b_stride + b_stride, 0);
        s.a_wide.assign(static_cast<size_t>(MB) * KB + KB, 0.0f);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int p = 0; p < n_panels; ++p) {
            const int jc = p * panel_w;
            const int nb_act = std::min(panel_w, N - jc);

            for (int pc = 0; pc < K; pc += KB) {
                const int kb_act = std::min(KB, K - pc);
                const float beta_k = (pc == 0) ? beta_eff : 1.0f;

                pack_b_vnni_strip_scalar(B, ldb, transB, jc, nb_act, K, pc,
                        kb_act, s.b_strip.data());

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);

                    widen_bf16_panel_to_fp32(A + static_cast<size_t>(ic) * lda
                                    + pc,
                            lda, s.a_wide.data(), kb_act, mb_act, kb_act);

                    for (int ir = 0; ir < mb_act; ir += MR) {
                        const int mr_act = std::min(MR, mb_act - ir);
                        const float *a_panel
                                = s.a_wide.data() + static_cast<size_t>(ir)
                                * kb_act;

                        for (int jr = 0; jr < nb_act; jr += NR) {
                            const int nr_act = std::min(NR, nb_act - jr);
                            float *c_tile = C + static_cast<size_t>(ic + ir)
                                            * ldc_f
                                    + jc + jr;
                            // b_stride counts uint16 per k-pair row; jr columns
                            // in means jr*VNNI_PAIR uint16 in.
                            const uint16_t *b_tile
                                    = s.b_strip.data() + jr * VNNI_PAIR;

                            auto uk = (mr_act == MR && nr_act == NR)
                                    ? select_bf16_ukernel_128(MR, NR)
                                    : nullptr;
                            if (uk != nullptr) {
                                uk(a_panel, kb_act, b_tile, b_stride, c_tile,
                                        ldc_f, kb_act, beta_k, nullptr,
                                        fused_postop_t::none, nullptr, 0);
                            } else {
                                bf16_tail_kernel_128(a_panel, kb_act, b_tile,
                                        b_stride, c_tile, ldc_f, kb_act, mr_act,
                                        nr_act, beta_k, nullptr,
                                        fused_postop_t::none, nullptr, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- epilogue: alpha, bias, postops, then narrow if needed ------------
    if (alpha != 1.0f) {
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) C[m * ldc_f + n] *= alpha;
    }

    apply_postops_tile(C, ldc_f, M, N, 0, 0, bias_f, params.postop_);

    if (dst_is_bf16) {
        uint16_t *d = static_cast<uint16_t *>(dst);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                d[m * ldc + n] = f32_to_bf16_rne(C[m * ldc_f + n]);
    }

    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
