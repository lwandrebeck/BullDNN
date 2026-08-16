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
// no reduction is needed, then walk K blocks inside. Nothing is shared between
// threads but read-only inputs: the packed B panels when the weights are
// constant and so cached, or a private strip per thread when they are not.
//
// One thing differs from the AVX-512 looper by necessity. A is widened to FP32
// once per call, because the
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
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
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
    std::vector<float> c_tile;     // full MR x NR tile, for ragged edges
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

    // No early-out on a null microkernel. The tile loop below already handles
    // `hot == nullptr` by calling bf16_tail_kernel_128, which is compiled
    // outside the target pragmas and so runs anywhere -- it was simply
    // unreachable, because this returned first.
    //
    // Declining here was worse than slow. native_matmul then reports that no
    // 128-bit microkernel exists for the shape and falls back to AOCL-DLP; in a
    // build without AOCL-DLP that is not another backend but none, and the
    // destination is returned untouched. Every one of the 6306 AI GEMV cases
    // failed exactly that way on the Piledriver box. A correct slow answer is
    // the floor; no answer is not an acceptable one.

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

    // ---- packed B, once per weight rather than once per call ---------------
    // Every other looper here caches its packed weights and this one did not,
    // on a comment of mine claiming the shared cache's fill path was not
    // portable. It is not: BF16PrepackedWeightCache::get_or_prepack is plain
    // scalar C++ with no AVX-512 in it, and the layout it produces is the one
    // this file already asks for -- panels of NR_PACK columns, k-pairs at
    // BF16PrepackedWeight::stride(), which is what b_stride above is set to.
    //
    // So a cached panel can be handed to the microkernel as it stands. Panel p
    // at K offset pc begins at
    //
    //     data + p * k_pairs_total * b_stride + (pc / 2) * b_stride
    //
    // and the bytes are identical to what the on-the-fly packer writes, so
    // results are unchanged rather than merely close.
    //
    // pc/2 is only a whole k-pair when KB is even. An odd KB would put a block
    // boundary in the middle of a pair, which this layout cannot express, so
    // that case keeps packing per call. The planner has no reason to choose an
    // odd KB, but nothing forces it not to.
    //
    // is_weights_const is the framework's promise that the buffer will not be
    // written behind us; without it, caching by pointer would serve stale
    // weights. The gate matches the other loopers'.
    static const int32_t s_weight_cache
            = matmul_config_t::instance().get_weight_cache();
    const bool can_cache = desc.is_weights_const && (s_weight_cache != 0)
            && (KB % 2 == 0);
    const BF16PrepackedWeight *prepacked_b = nullptr;
    if (can_cache) {
        const PrepackedWeightKey bk {weight, K, N, ldb, transB};
        prepacked_b = BF16PrepackedWeightCache::instance().get_or_prepack(
                bk, B);
    }
    // The cache pads K up to an even number of k-pairs; panel stride follows
    // that padding, not K itself.
    const size_t panel_pairs
            = prepacked_b ? static_cast<size_t>(prepacked_b->K_padded) / 2 : 0;

    int nthreads = plan.num_threads > 0 ? plan.num_threads : 1;
    nthreads = std::min(nthreads, std::max(n_panels, 1));

    // Fuse bias and one activation into the microkernel's epilogue where that is
    // safe, rather than sweeping C afterwards.
    //
    // Measured on an A10-8770E at 4 threads, the separate sweep costs:
    //
    //   relu        2-3%     compare-and-max, cheap against M*N*K
    //   alpha != 1  1-2%     one multiply per element
    //   gelu_erf    20-22%   transcendental, and it re-reads all of C
    //
    // So the cheap cases do not justify fusing and the expensive ones clearly
    // do; the win is keeping the tile in registers instead of streaming C a
    // second time.
    //
    // Conditions, following the FP32 looper: alpha must be 1, or the ordering
    // breaks -- the kernel would apply bias before the alpha scaling instead of
    // after -- and the chain must contain at most one fusable activation and
    // nothing else, since apply_postops_tile() would otherwise re-apply what the
    // kernel already did.
    fused_postop_t fused_candidate = fused_postop_t::none;
    bool has_unfuseable = false;
    const bool chain_fully_fusable
            = scan_gemv_postops(params, &fused_candidate, &has_unfuseable);
    const bool can_fuse = chain_fully_fusable && (alpha == 1.0f);
    const fused_postop_t fused_op
            = can_fuse ? fused_candidate : fused_postop_t::none;

    // ---- widen A once ------------------------------------------------------
    // The microkernel wants FP32 A. Widening it per column panel, which is where
    // this started, repeats the work n_panels times: sixteen at N=1024 with
    // 64-wide panels, or about 96 MB of traffic against a 55 ms GEMM. Doing it
    // once costs M*K floats of scratch and turns that into one pass.
    //
    // Rows are padded up to a whole multiple of MR and left zeroed, so the
    // vector kernel can always read MR rows even in the last M panel. Zeros
    // contribute nothing to the tiles that are kept.
    const int m_padded = ((M + MR - 1) / MR) * MR;
    std::vector<float> a_wide(
            static_cast<size_t>(m_padded) * static_cast<size_t>(K), 0.0f);
    widen_bf16_panel_to_fp32(A, lda, a_wide.data(), K, M, K);
    const int a_stride = K;
#if defined(_OPENMP)
#pragma omp parallel num_threads(nthreads)
#endif
    {
        ThreadScratch s;
        // No strip to own when B comes from the cache: the panels are read-only
        // and shared, so the threads read them in place.
        if (prepacked_b == nullptr) {
            s.b_strip.assign(
                    static_cast<size_t>(k_pairs_kb) * b_stride + b_stride, 0);
        }
        s.c_tile.assign(static_cast<size_t>(MR) * NR, 0.0f);

        const bf16_ukernel_128_fn_t hot = select_bf16_ukernel_128(MR, NR);

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int p = 0; p < n_panels; ++p) {
            const int jc = p * panel_w;
            const int nb_act = std::min(panel_w, N - jc);

            for (int pc = 0; pc < K; pc += KB) {
                const int kb_act = std::min(KB, K - pc);
                const float beta_k = (pc == 0) ? beta_eff : 1.0f;
                const bool is_last_k = (pc + kb_act >= K);

                const uint16_t *b_strip;
                if (prepacked_b != nullptr) {
                    b_strip = prepacked_b->data
                            + (static_cast<size_t>(p) * panel_pairs
                                      + static_cast<size_t>(pc) / 2)
                                    * b_stride;
                } else {
                    pack_b_vnni_strip_scalar(B, ldb, transB, jc, nb_act, K, pc,
                            kb_act, s.b_strip.data());
                    b_strip = s.b_strip.data();
                }

                for (int ic = 0; ic < M; ic += MB) {
                    const int mb_act = std::min(MB, M - ic);

                    for (int ir = 0; ir < mb_act; ir += MR) {
                        const int mr_act = std::min(MR, mb_act - ir);
                        // Straight into the pre-widened A, at this row and this
                        // K block. The padding rows past M are already zero.
                        const float *a_panel = a_wide.data()
                                + static_cast<size_t>(ic + ir) * a_stride + pc;

                        for (int jr = 0; jr < nb_act; jr += NR) {
                            const int nr_act = std::min(NR, nb_act - jr);
                            float *c_tile = C + static_cast<size_t>(ic + ir)
                                            * ldc_f
                                    + jc + jr;
                            // b_stride counts uint16 per k-pair row; jr columns
                            // in means jr*VNNI_PAIR uint16 in.
                            const uint16_t *b_tile
                                    = b_strip + jr * VNNI_PAIR;

                            // Bias and the activation ride along on the last K
                            // block, where the tile holds its final value.
                            const bool fuse_now = can_fuse && is_last_k;
                            const float *tile_bias
                                    = (fuse_now && bias_f != nullptr)
                                    ? (bias_f + jc + jr)
                                    : nullptr;
                            const fused_postop_t tile_op
                                    = fuse_now ? fused_op : fused_postop_t::none;

                            if (hot == nullptr) {
                                bf16_tail_kernel_128(a_panel, a_stride, b_tile,
                                        b_stride, c_tile, ldc_f, kb_act, mr_act,
                                        nr_act, beta_k, tile_bias, tile_op,
                                        nullptr, 0);
                            } else if (mr_act == MR && nr_act == NR) {
                                hot(a_panel, a_stride, b_tile, b_stride, c_tile,
                                        ldc_f, kb_act, beta_k, tile_bias,
                                        tile_op, nullptr, 0);
                            } else {
                                // Ragged edge: run the vector kernel over the
                                // whole MR x NR tile into scratch, then keep the
                                // live part. The padding contributes nothing --
                                // the packer zero-fills B past nr_act and the
                                // A rows past mr_act were just zeroed -- so this
                                // computes the same values the scalar tail did,
                                // roughly fifteen times faster.
                                //
                                // Worth the detour: with N=555 and NR=64 only
                                // 7.7% of columns are ragged, and sending them
                                // to the scalar tail cost 58% of throughput on
                                // an A10-8770E.
                                hot(a_panel, a_stride, b_tile, b_stride,
                                        s.c_tile.data(), NR, kb_act, 0.0f,
                                        nullptr, fused_postop_t::none, nullptr,
                                        0);
                                for (int m = 0; m < mr_act; ++m) {
                                    const float *srow = s.c_tile.data()
                                            + static_cast<size_t>(m) * NR;
                                    float *drow = c_tile
                                            + static_cast<size_t>(m) * ldc_f;
                                    if (beta_k == 0.0f) {
                                        std::memcpy(drow, srow,
                                                static_cast<size_t>(nr_act)
                                                        * sizeof(float));
                                    } else {
                                        for (int n = 0; n < nr_act; ++n)
                                            drow[n] = beta_k * drow[n]
                                                    + srow[n];
                                    }
                                }
                                // The epilogue cannot ride along here: the
                                // scratch is computed with beta 0, so bias and
                                // the activation have to see the merged value,
                                // not the bare product. Applied to the live
                                // sub-tile only.
                                if (fuse_now) {
                                    apply_bias_and_postop_tile(c_tile, ldc_f,
                                            mr_act, nr_act, tile_bias, tile_op);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- epilogue ---------------------------------------------------------
    // Skipped entirely when the kernels already did it. Running
    // apply_postops_tile() here as well would apply the activation twice.
    if (!can_fuse) {
        if (alpha != 1.0f) {
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) C[m * ldc_f + n] *= alpha;
        }
        apply_postops_tile(C, ldc_f, M, N, 0, 0, bias_f, params.postop_);
    }

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
