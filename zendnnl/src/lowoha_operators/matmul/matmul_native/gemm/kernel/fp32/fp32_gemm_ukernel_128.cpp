/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

//
// 128-bit FP32 GEMM microkernel for hosts without AVX-512, tuned for AMD
// family 15h (Bulldozer, Piledriver, Steamroller, Excavator).
//
// Why 128-bit rather than 256-bit, on hardware that has AVX:
//
// Family 15h implements the FPU as two 128-bit FMAC pipes. AMD's family 15h
// software optimization guide (publication 47414) states that only one 256-bit
// operation can issue per cycle, and that an extra cycle can be incurred for
// it, whereas the two 128-bit pipes each accept one operation per cycle. A
// 256-bit multiply-add therefore buys no throughput over two 128-bit ones and
// costs flexibility in scheduling, so this kernel works in __m128 throughout.
// The same guide's "rank-1 update" advice -- keep one operand broadcast from A
// and stream B -- is the shape of the inner loop below.
//
// The multiply-add itself prefers FMA4, which every family 15h part has,
// including Bulldozer, which has neither FMA3 nor F16C. FMA4's four-operand,
// non-destructive form also means the accumulator is never clobbered, so no
// register copies are needed to keep 12 accumulators live.
//
// XOP is deliberately not used here. It is available on all of these parts, but
// its instructions are integer and permute oriented (VPPERM, VPCMOV, integer
// multiply-accumulate, shifts and rotates) and offer nothing for an f32 GEMM
// inner loop. VPPERM would be useful for widening bf16 to f32 in one operation;
// that belongs in the bf16 path, not here. F16C likewise applies to f16
// conversion, not to f32 arithmetic.
//
// This file carries no target attribute. It compiles its kernel only when the
// translation unit's own target ISA already provides AVX, which for BullDNN
// means a -march=bdverN build; GCC does not expose the FMA4 intrinsics through
// a function target attribute, only through the command line. A plain x86-64
// build compiles select_ukernel_128() to a nullptr return and keeps the
// pre-existing scalar path.
//

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/fp32/fp32_gemm_ukernel.hpp"

#include <cstdlib>

#if defined(__AVX__)
#if defined(__FMA4__) || defined(__XOP__)
// The FMA4 and XOP intrinsics live in x86intrin.h, not immintrin.h: GCC keeps
// immintrin.h to Intel-defined intrinsics and puts the AMD-specific ones
// (fma4intrin.h, xopintrin.h) behind x86intrin.h, which includes immintrin.h in
// turn. Including only immintrin.h here fails with "_mm_macc_ps was not
// declared in this scope" even on a -march=bdver4 build where __FMA4__ is set.
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

#if defined(__AVX__)

namespace {

// One 128-bit multiply-add, using the best form this target offers.
//
// FMA4 (all of family 15h) is a single non-destructive VFMADDPS. FMA3 covers
// Piledriver onward and every Zen part, and is what a non-family-15h build
// without AVX-512 will pick up. The mul+add fallback keeps the kernel correct
// on a plain AVX target such as Sandy Bridge; it is two operations rather than
// one but still four lanes wide.
inline __m128 fmadd128(__m128 a, __m128 b, __m128 acc) {
#if defined(__FMA4__)
    return _mm_macc_ps(a, b, acc);
#elif defined(__FMA__)
    return _mm_fmadd_ps(a, b, acc);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), acc);
#endif
}

// MR=6 microkernel over kNR columns, instantiated for the NR values
// plan_fp32_gemm() produces: 64 for N >= 64, 32 for N >= 32, otherwise 16.
//
// A whole tile never fits in registers -- 6x64 floats would be 96 __m128
// accumulators against 16 XMM registers -- so the kernel sweeps K once per
// eight-column group, holding 6x2 = 12 accumulators plus two B vectors and one
// broadcast A value, 15 of 16 registers.
//
// Grouping by columns rather than rows is what keeps the memory traffic honest:
// each B element is still read exactly once overall, since a group reads only
// its own eight columns, and only A is re-read, once per group. For NR=64 that
// is 48k A loads against 96k multiply-adds, half a load per FMA, comfortably
// inside the two loads per cycle the load/store unit sustains (47414). The A
// panel is 6 x KB floats and comes back from L1, which is 16 KB on bdver1-3 and
// 32 KB per core on Excavator (50742).
//
// Twelve independent accumulators is also what FMA latency requires to keep both
// pipes busy: at roughly six cycles of latency and two issues per cycle,
// anything less leaves the pipes waiting on their own results.
template <int kNR>
void ukernel_6xnr_128(const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, float beta, const float *__restrict__ bias,
        fused_postop_t fused_op) {

    constexpr int kMR = 6;
    static_assert(kNR % 8 == 0, "kNR must be a multiple of the 8-column group");

    for (int n0 = 0; n0 < kNR; n0 += 8) {
        __m128 acc[kMR][2];
#pragma GCC unroll 6
        for (int m = 0; m < kMR; ++m) {
            acc[m][0] = _mm_setzero_ps();
            acc[m][1] = _mm_setzero_ps();
        }

        // Rank-1 update per k: broadcast one A element per row, stream eight B
        // columns, issue twelve independent multiply-adds.
        for (int kk = 0; kk < k; ++kk) {
            const float *b_row = pb + kk * b_stride + n0;
            const __m128 b0 = _mm_loadu_ps(b_row);
            const __m128 b1 = _mm_loadu_ps(b_row + 4);

#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                const __m128 a = _mm_broadcast_ss(pa + m * a_stride + kk);
                acc[m][0] = fmadd128(a, b0, acc[m][0]);
                acc[m][1] = fmadd128(a, b1, acc[m][1]);
            }
        }

        // beta is a flag rather than a scale here, matching scalar_microkernel():
        // zero means overwrite C, anything else means accumulate into it. Keep
        // that contract exactly -- the looper relies on it for K-blocking, where
        // the second and later K blocks must add to what the first wrote.
        if (beta == 0.0f) {
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                _mm_storeu_ps(C + m * ldc + n0, acc[m][0]);
                _mm_storeu_ps(C + m * ldc + n0 + 4, acc[m][1]);
            }
        } else {
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                float *c_row = C + m * ldc + n0;
                _mm_storeu_ps(
                        c_row, _mm_add_ps(_mm_loadu_ps(c_row), acc[m][0]));
                _mm_storeu_ps(c_row + 4,
                        _mm_add_ps(_mm_loadu_ps(c_row + 4), acc[m][1]));
            }
        }
    }

    // Bias and activation come from the shared epilogue so this kernel and the
    // scalar one cannot disagree numerically.
    apply_bias_and_postop_tile(C, ldc, kMR, kNR, bias, fused_op);
}

// One 256-bit multiply-add. 256-bit FMA is part of FMA4 itself, so this is
// available on every family 15h part including Bulldozer -- AVX2 is not
// involved, since what AVX2 adds over AVX is 256-bit integer, gather and
// variable shifts, none of which appear in an f32 GEMM inner loop.
inline __m256 fmadd256(__m256 a, __m256 b, __m256 acc) {
#if defined(__FMA4__)
    return _mm256_macc_ps(a, b, acc);
#elif defined(__FMA__)
    return _mm256_fmadd_ps(a, b, acc);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), acc);
#endif
}

// 256-bit counterpart of ukernel_6xnr_128, structurally identical but covering
// sixteen columns per pass instead of eight.
//
// 47414 says one 256-bit operation issues per cycle against two 128-bit ones, so
// the multiply-add ceiling is the same either way and width alone wins nothing.
// What it can win is everything around the multiply-adds: half the instructions,
// half the broadcasts per FLOP, and half the A re-reads, because sixteen columns
// per pass halves the number of passes over K. Against that, 256-bit operations
// double-dispatch as two macro-ops on this family and 256-bit stores are the
// slower direction.
//
// Measured on an A10-8770E (Excavator), f32 native_gemm, idle box: the two are
// indistinguishable, with 256-bit marginally behind at every size --
// 512x512x512 42.44 vs 42.47, 2048x512x512 44.30 vs 44.57, 128x1024x1024 47.99
// vs 48.06 GFLOPS. So the instruction-count saving does not convert into
// throughput here, which is why 128 is the default. Both produce identical
// results (40/40 of the pinned correctness tests pass either way).
//
// This is kept, rather than deleted, because the tradeoff is uarch-specific and
// only Excavator has been measured: bdver1/bdver2 share one decoder between the
// two cores of a module, where halving the instruction count is worth more than
// it is here. Re-measuring on those parts is a one-line env change.
template <int kNR>
void ukernel_6xnr_256(const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, float beta, const float *__restrict__ bias,
        fused_postop_t fused_op) {

    constexpr int kMR = 6;
    static_assert(
            kNR % 16 == 0, "kNR must be a multiple of the 16-column group");

    for (int n0 = 0; n0 < kNR; n0 += 16) {
        __m256 acc[kMR][2];
#pragma GCC unroll 6
        for (int m = 0; m < kMR; ++m) {
            acc[m][0] = _mm256_setzero_ps();
            acc[m][1] = _mm256_setzero_ps();
        }

        for (int kk = 0; kk < k; ++kk) {
            const float *b_row = pb + kk * b_stride + n0;
            const __m256 b0 = _mm256_loadu_ps(b_row);
            const __m256 b1 = _mm256_loadu_ps(b_row + 8);

#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                const __m256 a = _mm256_broadcast_ss(pa + m * a_stride + kk);
                acc[m][0] = fmadd256(a, b0, acc[m][0]);
                acc[m][1] = fmadd256(a, b1, acc[m][1]);
            }
        }

        // Same beta-as-a-flag contract as the 128-bit kernel.
        if (beta == 0.0f) {
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                _mm256_storeu_ps(C + m * ldc + n0, acc[m][0]);
                _mm256_storeu_ps(C + m * ldc + n0 + 8, acc[m][1]);
            }
        } else {
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                float *c_row = C + m * ldc + n0;
                _mm256_storeu_ps(c_row,
                        _mm256_add_ps(_mm256_loadu_ps(c_row), acc[m][0]));
                _mm256_storeu_ps(c_row + 8,
                        _mm256_add_ps(_mm256_loadu_ps(c_row + 8), acc[m][1]));
            }
        }
    }

    apply_bias_and_postop_tile(C, ldc, kMR, kNR, bias, fused_op);
}

// Which width to use, read once. 128 is the default because it is what has been
// measured good on Excavator; ZENDNNL_NATIVE_GEMM_SIMD_WIDTH=256 selects the
// wider kernel. The switch exists because the answer is uarch-dependent and only
// Excavator has been measured so far -- Bulldozer, Piledriver and Steamroller
// differ in decode width and 256-bit handling, so whichever wins here need not
// win there.
int native_gemm_simd_width() {
    static const int width = []() -> int {
        const char *env = std::getenv("ZENDNNL_NATIVE_GEMM_SIMD_WIDTH");
        if (env && std::atoi(env) == 256) { return 256; }
        return 128;
    }();
    return width;
}

} // namespace

ukernel_fn_t select_ukernel_128(int MR, int NR) {
    if (MR != 6) { return nullptr; }
    if (native_gemm_simd_width() == 256) {
        // NR=16 has no 256-bit instantiation: sixteen columns is one group, so a
        // 256-bit pass would hold only 2 accumulators per row and starve the FMA
        // pipes. Fall through to the 128-bit kernel for that shape.
        switch (NR) {
            case 64: return ukernel_6xnr_256<64>;
            case 32: return ukernel_6xnr_256<32>;
            default: break;
        }
    }
    switch (NR) {
        case 64: return ukernel_6xnr_128<64>;
        case 32: return ukernel_6xnr_128<32>;
        case 16: return ukernel_6xnr_128<16>;
        default: break;
    }
    // Any other tile shape keeps the scalar path: the looper only calls the hot
    // kernel for full MR x NR tiles, so an unsupported shape simply means the
    // planner chose blocking this kernel was not written for.
    return nullptr;
}

#else // !__AVX__

ukernel_fn_t select_ukernel_128(int, int) {
    return nullptr;
}

#endif // __AVX__

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
