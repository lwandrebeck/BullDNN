/******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SIMD tier selection policy for the SimdOps tags in simd_ops.hpp.
 *
 * Deliberately dependency-free: the policy is a pure function of CPU
 * feature bits, so it can be reasoned about and unit-tested without
 * pulling in platform_info (and therefore aocl-utils). Callers read the
 * feature bits from zendnnl_platform_info() and pass them in.
 ******************************************************************************/

#pragma once

namespace zendnnl {
namespace lowoha {
namespace simd {

// ---------------------------------------------------------------------------
// Which SimdOps<Tag> specialization a host should run.
//
//   scalar   -> scalar_tag       : 1 lane, portable
//   avx      -> avx_tag          : 8 lanes, AVX only        (Bulldozer)
//   avx_f16c -> avx_f16c_tag     : + FMA3, hardware F16C    (Piledriver,
//                                                            Steamroller)
//   avx2     -> avx2_tag         : + 256-bit integer bf16   (Excavator)
//   avx512   -> avx512_tag       : 16 lanes                 (Zen 4+, ...)
//
// Ordered weakest to strongest so comparisons express capability.
// ---------------------------------------------------------------------------
enum class simd_tier {
    scalar = 0,
    avx = 1,
    avx_f16c = 2,
    avx2 = 3,
    avx512 = 4,
};

// CPU feature bits the policy consumes, as reported by CPUID.
struct simd_isa {
    bool avx = false;
    bool f16c = false;
    bool fma = false;  // FMA3
    bool avx2 = false;
    bool avx512f = false;
    bool avx512bw_vl = false;  // AVX-512BW *and* AVX-512VL
};

/** @brief Pick the strongest SimdOps tier the host can execute.
 *
 *  AVX-512 requires BW+VL as well as F, matching the existing kernel gates
 *  (a host with F but not BW/VL would fault in the masked/byte-word paths).
 *  The AVX2 and F16C tiers additionally require FMA3, because their tags
 *  are compiled with target("...,fma"): claiming them on a host without
 *  FMA3 would emit instructions it cannot run. That is exactly the AMD
 *  family 15h split — Bulldozer has AVX but neither FMA3 nor F16C, so it
 *  lands on the plain avx tier.
 *
 *  @param isa CPU feature bits.
 *  @return the tier to dispatch to.
 */
constexpr simd_tier select_simd_tier(const simd_isa &isa) {
    if (isa.avx512f && isa.avx512bw_vl) {
        return simd_tier::avx512;
    }
    if (isa.avx && isa.fma && isa.f16c && isa.avx2) {
        return simd_tier::avx2;
    }
    if (isa.avx && isa.fma && isa.f16c) {
        return simd_tier::avx_f16c;
    }
    if (isa.avx) {
        return simd_tier::avx;
    }
    return simd_tier::scalar;
}

/** @brief Human-readable tier name, for apilog / profiling output. */
constexpr const char *simd_tier_name(simd_tier t) {
    switch (t) {
        case simd_tier::avx512: return "avx512";
        case simd_tier::avx2: return "avx2";
        case simd_tier::avx_f16c: return "avx_f16c";
        case simd_tier::avx: return "avx";
        case simd_tier::scalar: return "scalar";
    }
    return "scalar";
}

} // namespace simd
} // namespace lowoha
} // namespace zendnnl
