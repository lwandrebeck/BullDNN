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
// 128-bit BF16 GEMM microkernel for hosts without avx512bf16.
//
// Shape: MR=6 rows by eight columns at a time, so twelve __m128 accumulators.
// With two widened B vectors and one A broadcast that is fifteen of sixteen XMM
// registers, one short of the limit -- which is deliberate. The FP32 kernel next
// door sits at fifteen too, and at that pressure GCC will rematerialize a load
// rather than spill, so the B vectors are pinned (see pin_reg there for the
// disassembly that prompted it).
//
// Per k-pair and eight-column group:
//
//   2 loads   B dwords, each 4 columns x 2 k
//   3 ops     widen the even k (one shift) and the odd k (two shifts)
//   6 bcast   A, one per row, shared by both column halves
//   12 FMA
//
// against the FP32 kernel's 4 loads + 12 broadcasts + 24 FMAs for the same two
// k. Same arithmetic, six extra shifts, half the B bytes. B is re-loaded once
// per k half rather than held, because holding both raw dwords across the two
// halves would need seventeen registers.
//
// Family 15h has no BF16 arithmetic of any kind, so this is FP32 maths on
// widened operands; the format buys traffic, not FLOPs. 47414 sec 2.3 gives the
// two 128-bit FMACs per compute unit that set the arithmetic ceiling, and the
// two 128-bit loads per cycle the LSU sustains that set the load ceiling. Ten
// load-unit operations per k-pair against twenty-four FMAs leaves the FMACs the
// bottleneck, which is where it should be.
//
// Built twice, once per multiply-accumulate flavour, because family 15h is split
// on this point: Bulldozer (bdver1) has FMA4 and no FMA3, everything from
// Piledriver on has both. A machine with one is not guaranteed to have the other
// -- no Zen or Intel part has FMA4 -- so the choice has to be made at runtime,
// not by an #if on the build's baseline. The arithmetic lives in
// bf16_gemm_ukernel_128_body.inc so there is one copy of it.
//
// Where both exist, FMA3 wins: measured 9% faster than FMA4 on Excavator, and
// GCC lowers either intrinsic to vfmadd231ps under -march=bdver4 anyway.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/bf16/bf16_gemm_ukernel_128.hpp"

#include "common/platform_info.hpp"
#include "common/zendnnl_global.hpp"

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#include <x86intrin.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Defined alongside the FP32 microkernels and shared with them.
void apply_bias_and_postop_tile(float *C, int ldc, int mr_count, int nr_count,
        const float *bias, fused_postop_t fused_op);

namespace {

// FMA4 is an AMD-only extension that platform_info does not report, since that
// comes from AOCL-utils and is Zen-oriented -- the same reason cpu_family there
// does not identify family 15h. CPUID Fn8000_0001_ECX bit 16 is the definitive
// answer, guarded on the leaf existing at all.
bool host_has_fma4() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx)) return false;
    if (eax < 0x80000001u) return false;
    if (!__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 16)) != 0;
#else
    return false;
#endif
}

// Shared by both flavours: widening is placement, not conversion, so it needs no
// rounding and does not depend on the multiply-accumulate form. A macro rather
// than plain functions so each flavour namespace gets its own copy under its own
// target pragma.
//
// In the VNNI layout a dword holds two consecutive k for one column, k even in
// the low half and k odd in the high half, which makes each widening a shift.
#define BF16_UK128_COMMON_HELPERS                                              \
    template <typename Vec>                                                    \
    inline void pin_reg(Vec &v0, Vec &v1) {                                    \
        asm("" : "+x"(v0), "+x"(v1));                                          \
    }                                                                          \
    inline __m128 widen_even_k(__m128i pairs) {                                \
        return _mm_castsi128_ps(_mm_slli_epi32(pairs, 16));                    \
    }                                                                          \
    inline __m128 widen_odd_k(__m128i pairs) {                                 \
        return _mm_castsi128_ps(                                               \
                _mm_slli_epi32(_mm_srli_epi32(pairs, 16), 16));                \
    }                                                                          \
    inline __m128i load_b_pairs(const uint16_t *p) {                           \
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));          \
    }

// ---- FMA3: Piledriver onward, and every Zen and Intel part -----------------
#pragma GCC push_options
#pragma GCC target("avx,fma")
namespace uk_fma3 {
using zendnnl::lowoha::matmul::native::apply_bias_and_postop_tile;
BF16_UK128_COMMON_HELPERS
inline __m128 fmadd128(__m128 a, __m128 b, __m128 acc) {
    return _mm_fmadd_ps(a, b, acc);
}
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/bf16/bf16_gemm_ukernel_128_body.inc"
} // namespace uk_fma3
#pragma GCC pop_options

// ---- FMA4: Bulldozer (bdver1), the one family 15h model without FMA3 --------
#pragma GCC push_options
#pragma GCC target("avx,fma4")
namespace uk_fma4 {
using zendnnl::lowoha::matmul::native::apply_bias_and_postop_tile;
BF16_UK128_COMMON_HELPERS
inline __m128 fmadd128(__m128 a, __m128 b, __m128 acc) {
    return _mm_macc_ps(a, b, acc);
}
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/bf16/bf16_gemm_ukernel_128_body.inc"
} // namespace uk_fma4
#pragma GCC pop_options

} // namespace

bf16_ukernel_128_fn_t select_bf16_ukernel_128(int MR, int NR) {
    // Both flavours are compiled whatever the build baseline, so the choice is
    // made here. FMA3 where available; FMA4 covers Bulldozer.
    enum class Flavour { none, fma3, fma4 };
    static const Flavour s_flavour = [] {
        auto &pinfo = zendnnl::common::zendnnl_platform_info();
        if (!pinfo.get_avx_status()) return Flavour::none;
        if (pinfo.get_fma_status()) return Flavour::fma3;
        if (host_has_fma4()) return Flavour::fma4;
        return Flavour::none;
    }();

    if (s_flavour == Flavour::none || MR != 6) return nullptr;

    if (s_flavour == Flavour::fma3) {
        switch (NR) {
            case 64: return &uk_fma3::bf16_ukernel_6xnr_128<64>;
            case 32: return &uk_fma3::bf16_ukernel_6xnr_128<32>;
            case 16: return &uk_fma3::bf16_ukernel_6xnr_128<16>;
            default: return nullptr;
        }
    }
    switch (NR) {
        case 64: return &uk_fma4::bf16_ukernel_6xnr_128<64>;
        case 32: return &uk_fma4::bf16_ukernel_6xnr_128<32>;
        case 16: return &uk_fma4::bf16_ukernel_6xnr_128<16>;
        default: return nullptr;
    }
}

void bf16_tail_kernel_128(const float *__restrict__ A_f32, int a_stride,
        const uint16_t *__restrict__ B_vnni, int b_stride,
        float *__restrict__ C, int ldc, int k, int mr_act, int nr_act,
        float beta, const float *__restrict__ bias, fused_postop_t fused_op,
        uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    const int k_pairs = k / 2;
    const bool k_odd = (k & 1) != 0;

    for (int m = 0; m < mr_act; ++m) {
        for (int n = 0; n < nr_act; ++n) {
            float sum = 0.0f;
            for (int kp = 0; kp < k_pairs; ++kp) {
                // Same VNNI addressing as the vector path: two uint16 per
                // column, even k first.
                const uint16_t *pair = B_vnni + kp * b_stride + n * 2;
                float b_even, b_odd;
                uint32_t be = static_cast<uint32_t>(pair[0]) << 16;
                uint32_t bo = static_cast<uint32_t>(pair[1]) << 16;
                std::memcpy(&b_even, &be, sizeof(float));
                std::memcpy(&b_odd, &bo, sizeof(float));
                sum += A_f32[m * a_stride + 2 * kp] * b_even;
                sum += A_f32[m * a_stride + 2 * kp + 1] * b_odd;
            }
            if (k_odd) {
                const uint16_t *pair = B_vnni + k_pairs * b_stride + n * 2;
                float b_even;
                uint32_t be = static_cast<uint32_t>(pair[0]) << 16;
                std::memcpy(&b_even, &be, sizeof(float));
                sum += A_f32[m * a_stride + 2 * k_pairs] * b_even;
            }
            float *c = &C[m * ldc + n];
            *c = (beta == 0.0f) ? sum : beta * (*c) + sum;
        }
    }

    apply_bias_and_postop_tile(C, ldc, mr_act, nr_act, bias, fused_op);

    if (C_bf16 != nullptr) {
        for (int m = 0; m < mr_act; ++m) {
            for (int n = 0; n < nr_act; ++n) {
                uint32_t bits;
                std::memcpy(&bits, &C[m * ldc + n], sizeof(bits));
                const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
                C_bf16[m * ldc_bf16 + n] = static_cast<uint16_t>(rounded >> 16);
            }
        }
    }
}

void widen_bf16_panel_to_fp32(const uint16_t *__restrict__ src, int src_stride,
        float *__restrict__ dst, int dst_stride, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t *s = src + r * src_stride;
        float *d = dst + r * dst_stride;
        // Left scalar deliberately. A shift, a load and a store per element is
        // memory bound, so the compiler's own vectorisation at whatever
        // baseline the build targets is as good as a hand-written version, and
        // it needs no target attribute here.
        for (int c = 0; c < cols; ++c) {
            uint32_t bits = static_cast<uint32_t>(s[c]) << 16;
            std::memcpy(&d[c], &bits, sizeof(float));
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
