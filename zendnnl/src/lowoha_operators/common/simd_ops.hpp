/******************************************************************************
 * Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Cross-operator SIMD abstraction. No ATen / operator dependency.
 *
 * Two specializations selected at runtime via SimdOps<Tag>:
 *   - avx512_tag: 16 floats per vector, AVX-512 intrinsics via
 *                 __attribute__((target(...)))  (no -mavx512f flag needed)
 *   - scalar_tag: 1 float per vector, portable scalar fallback
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "common/float16.hpp"

namespace zendnnl {
namespace lowoha {
namespace simd {

// ---------------------------------------------------------------------------
// Tag types for compile-time SIMD dispatch.
// ---------------------------------------------------------------------------
struct avx512_tag {};
// AVX (256-bit, no AVX2 / AVX-512): AMD family 15h baseline, valid on
// Bulldozer through Excavator.
struct avx_tag {};
// AVX + F16C + FMA3: AMD family 15h from Piledriver onwards.
struct avx_f16c_tag {};
struct scalar_tag {};

template <typename Tag>
struct SimdOps;

// ===========================================================================
// Scalar specialization — one float lane, portable fallback.
// ===========================================================================
template <>
struct SimdOps<scalar_tag> {

    struct VecF32 {
        float v;
    };
    static constexpr int kFloatLanes = 1;

    static inline VecF32 vec_loadu(const float *p) { return VecF32 {*p}; }

    static inline void vec_storeu(float *p, VecF32 x) { *p = x.v; }

    static inline VecF32 vec_set1(float f) { return VecF32 {f}; }

    static inline VecF32 vec_add(VecF32 a, VecF32 b) {
        return VecF32 {a.v + b.v};
    }

    static inline VecF32 vec_sub(VecF32 a, VecF32 b) {
        return VecF32 {a.v - b.v};
    }

    static inline VecF32 vec_mul(VecF32 a, VecF32 b) {
        return VecF32 {a.v * b.v};
    }

    static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
        return VecF32 {a.v * b.v + c.v};
    }

    static inline VecF32 vec_max(VecF32 a, VecF32 b) {
        return VecF32 {std::fmax(a.v, b.v)};
    }

    static inline VecF32 vec_min(VecF32 a, VecF32 b) {
        return VecF32 {std::fmin(a.v, b.v)};
    }

    static inline float vec_hsum(VecF32 v) { return v.v; }

    static inline float vec_hmax(VecF32 v) { return v.v; }

    static inline VecF32 vec_exp_u20(VecF32 v) {
        return VecF32 {std::exp(v.v)};
    }

    static inline VecF32 vec_fexp_u20(VecF32 v) {
        return VecF32 {std::exp(v.v)};
    }

    static inline float vec_reduce_sum(VecF32 acc) { return acc.v; }

    static inline float vec_reduce_max(VecF32 acc) { return acc.v; }

    static inline VecF32 vec_mask_bf16_loadu(const uint16_t *p) {
        uint32_t u = static_cast<uint32_t>(p[0]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return VecF32 {f};
    }

    static inline void vec_bf16_storeu(uint16_t *dst, VecF32 v) {
        uint32_t u;
        std::memcpy(&u, &v.v, sizeof(u));
        uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
        dst[0] = static_cast<uint16_t>((u + rounding_bias) >> 16);
    }

    // ── FP16 (IEEE 754 half) ────────────────────────────────────────────
    static inline VecF32 vec_mask_f16_loadu(const uint16_t *p) {
        return VecF32 {zendnnl::common::float16_t::f16_to_f32_val(p[0])};
    }

    static inline void vec_f16_storeu(uint16_t *dst, VecF32 v) {
        dst[0] = zendnnl::common::float16_t::f32_to_f16_val(v.v);
    }
};

// ===========================================================================
// AVX specialization — 8 float lanes, enabled via target attribute.
//
// Baseline vector path for AMD family 15h (Bulldozer / Piledriver /
// Steamroller / Excavator) and any AVX-capable CPU without AVX-512.
//
// Deliberately restricted to the plain "avx" target so this header keeps
// compiling in every build configuration (including the default AVX-512
// build, which is not compiled with -mfma4/-mf16c/-mavx2):
//   - vec_fmadd uses mul+add (no FMA): FMA3 is unavailable on Bulldozer and
//     FMA4 intrinsics are not exposed by a target attribute alone. The
//     fused path lives in the avx_f16c_tag specialization (Piledriver+).
//   - 256-bit integer math (exp exponent build, bf16 pack/unpack) is done as
//     two 128-bit SSE halves because 256-bit integer ops are AVX2-only.
//   - FP16 <-> FP32 uses the scalar float16_t helpers because F16C is absent
//     on Bulldozer. avx_f16c_tag provides the hardware VCVTPH2PS path.
// ===========================================================================

#define LOWOHA_SIMD_AVX_ATTR __attribute__((target("avx")))

template <>
struct SimdOps<avx_tag> {

  using VecF32 = __m256;
  static constexpr int kFloatLanes = 8;

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_loadu(const float *p) {
    return _mm256_loadu_ps(p);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline void vec_storeu(float *p, VecF32 x) {
    _mm256_storeu_ps(p, x);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_set1(float f) {
    return _mm256_set1_ps(f);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_add(VecF32 a, VecF32 b) {
    return _mm256_add_ps(a, b);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_sub(VecF32 a, VecF32 b) {
    return _mm256_sub_ps(a, b);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_mul(VecF32 a, VecF32 b) {
    return _mm256_mul_ps(a, b);
  }

  // No FMA on the baseline AVX path (see class comment): mul+add.
  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_max(VecF32 a, VecF32 b) {
    return _mm256_max_ps(a, b);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_min(VecF32 a, VecF32 b) {
    return _mm256_min_ps(a, b);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline float vec_hsum(VecF32 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);          // 4 partial sums
    s = _mm_hadd_ps(s, s);                   // SSE3
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline float vec_hmax(VecF32 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);          // 4 partial maxima
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));  // max{0,2},{1,3}
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(m);
  }

  // ── Fast exp (~20 ULP, Malossi et al.) ──────────────────────────────
  // 256-bit port of the AVX-512 kernels. Mask-moves become blendv_ps and
  // the 2^n exponent build is split across two 128-bit halves (AVX2-free).

  LOWOHA_SIMD_AVX_ATTR
  static inline __m256 fexp_u20_ps_256(__m256 values) {
    const __m256 vec_c0 = _mm256_set1_ps(0.00010703434948458272f);
    const __m256 vec_c1 = _mm256_set1_ps(0.30354260500649682f);
    const __m256 vec_c2 = _mm256_set1_ps(-0.22433836478672356f);
    const __m256 vec_c3 = _mm256_set1_ps(-0.079204240219773236f);
    const __m256 vec_exp_log2ef =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));
    const __m256 vec_a =
      _mm256_set1_ps(static_cast<float>(std::pow(2.0, 23) / std::log2(2.0)));
    const __m256 vec_b =
      _mm256_set1_ps(static_cast<float>(std::pow(2.0, 23) * 127.0));
    const __m256 vec_ln_flt_min =
      _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));
    const __m256 vec_ln_flt_max =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));
    const __m256 vec_infinity =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));
    const __m256 vec_zero = _mm256_setzero_ps();

    const __m256 min_mask = _mm256_cmp_ps(values, vec_ln_flt_min, _CMP_LT_OS);
    const __m256 max_mask = _mm256_cmp_ps(values, vec_ln_flt_max, _CMP_GT_OS);

    __m256 vec_src = _mm256_mul_ps(values, vec_exp_log2ef);
    __m256 vec_fractional =
      _mm256_sub_ps(vec_src, _mm256_floor_ps(vec_src));

    __m256 vec_res = _mm256_add_ps(_mm256_mul_ps(vec_fractional, vec_c3), vec_c2);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_fractional, vec_res), vec_c1);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_fractional, vec_res), vec_c0);

    vec_src = _mm256_sub_ps(vec_src, vec_res);
    __m256 tmp = _mm256_add_ps(_mm256_mul_ps(vec_a, vec_src), vec_b);
    __m256 casted = _mm256_castsi256_ps(_mm256_cvttps_epi32(tmp));
    casted = _mm256_blendv_ps(casted, vec_zero, min_mask);
    casted = _mm256_blendv_ps(casted, vec_infinity, max_mask);
    return casted;
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline __m256 exp_u20_ps_256(__m256 values) {
    const __m256 vec_factorial_1 = _mm256_set1_ps(0.999999701f);
    const __m256 vec_factorial_2 = _mm256_set1_ps(0.499991506f);
    const __m256 vec_factorial_3 = _mm256_set1_ps(0.166676521f);
    const __m256 vec_factorial_4 = _mm256_set1_ps(0.0418978221f);
    const __m256 vec_factorial_5 = _mm256_set1_ps(0.00828929059f);
    const __m256 vec_exp_log2ef =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));
    const __m256 vec_half = _mm256_set1_ps(0.5f);
    const __m256 vec_one = _mm256_set1_ps(1.f);
    const __m256 vec_ln2f =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));
    const __m256 vec_ln_flt_min =
      _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));
    const __m256 vec_ln_flt_max =
      _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));
    const __m128i vec_126 = _mm_set1_epi32(126);
    constexpr int n_mantissa_bits = 23;

    const __m256 less_ln_flt_min_mask =
      _mm256_cmp_ps(values, vec_ln_flt_min, _CMP_LT_OS);
    __m256 vec_src = _mm256_min_ps(values, vec_ln_flt_max);
    vec_src = _mm256_max_ps(vec_src, vec_ln_flt_min);

    // round-to-negative-infinity via floor (AVX has no rounded cvt with mask)
    __m256 vec_fx = _mm256_add_ps(_mm256_mul_ps(vec_src, vec_exp_log2ef), vec_half);
    vec_fx = _mm256_floor_ps(vec_fx);
    __m256i vec_fx_i = _mm256_cvttps_epi32(vec_fx);

    // vec_exp_poly = vec_src - vec_fx * ln2f
    __m256 vec_exp_poly = _mm256_sub_ps(vec_src, _mm256_mul_ps(vec_fx, vec_ln2f));

    __m256 vec_res =
      _mm256_add_ps(_mm256_mul_ps(vec_exp_poly, vec_factorial_5), vec_factorial_4);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_exp_poly, vec_res), vec_factorial_3);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_exp_poly, vec_res), vec_factorial_2);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_exp_poly, vec_res), vec_factorial_1);
    vec_res = _mm256_add_ps(_mm256_mul_ps(vec_exp_poly, vec_res), vec_one);

    // Build 2^fx as (fx + 126) << 23 then *2 (see AVX-512 note). 256-bit
    // integer add/shift are AVX2-only, so split into two 128-bit halves.
    __m128i lo = _mm256_castsi256_si128(vec_fx_i);
    __m128i hi = _mm256_extractf128_si256(vec_fx_i, 1);
    lo = _mm_slli_epi32(_mm_add_epi32(lo, vec_126), n_mantissa_bits);
    hi = _mm_slli_epi32(_mm_add_epi32(hi, vec_126), n_mantissa_bits);
    __m256i two_pow_n_i =
      _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    __m256 vec_two_pow_n = _mm256_castsi256_ps(two_pow_n_i);
    vec_two_pow_n =
      _mm256_blendv_ps(vec_two_pow_n, _mm256_setzero_ps(), less_ln_flt_min_mask);

    vec_res = _mm256_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm256_mul_ps(vec_res, _mm256_set1_ps(2.f));
    return vec_res;
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_exp_u20(VecF32 v) {
    return exp_u20_ps_256(v);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_fexp_u20(VecF32 v) {
    return fexp_u20_ps_256(v);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline float vec_reduce_sum(VecF32 acc) {
    return vec_hsum(acc);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline float vec_reduce_max(VecF32 acc) {
    return vec_hmax(acc);
  }

  // ── BF16 ────────────────────────────────────────────────────────────
  // bf16 is the top 16 bits of fp32, so conversion is pure bit-twiddling.
  // 256-bit integer ops are AVX2-only; done here as two 128-bit halves.

  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_mask_bf16_loadu(const uint16_t *p) {
    __m128i u  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    __m128i lo = _mm_slli_epi32(_mm_cvtepu16_epi32(u), 16);
    __m128i hi = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(u, 8)), 16);
    __m256i wide = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    return _mm256_castsi256_ps(wide);
  }

  // FP32 → BF16 store with round-to-nearest-even. Inverse of the loader.
  LOWOHA_SIMD_AVX_ATTR
  static inline void vec_bf16_storeu(uint16_t *dst, VecF32 v) {
    __m256i u = _mm256_castps_si256(v);
    const __m128i one = _mm_set1_epi32(1);
    const __m128i k7fff = _mm_set1_epi32(0x7FFF);
    __m128i half[2] = { _mm256_castsi256_si128(u),
                        _mm256_extractf128_si256(u, 1) };
    for (int i = 0; i < 2; ++i) {
      __m128i rounding_bias = _mm_add_epi32(
        _mm_and_si128(_mm_srli_epi32(half[i], 16), one), k7fff);
      half[i] = _mm_srli_epi32(_mm_add_epi32(half[i], rounding_bias), 16);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst),
                     _mm_packus_epi32(half[0], half[1]));
  }

  // ── FP16 (IEEE 754 half) ────────────────────────────────────────────
  // Bulldozer lacks F16C, so convert in software via float16_t. The
  // hardware VCVTPH2PS path lives in the avx_f16c_tag specialization.
  LOWOHA_SIMD_AVX_ATTR
  static inline VecF32 vec_mask_f16_loadu(const uint16_t *p) {
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; ++i)
      tmp[i] = zendnnl::common::float16_t::f16_to_f32_val(p[i]);
    return _mm256_loadu_ps(tmp);
  }

  LOWOHA_SIMD_AVX_ATTR
  static inline void vec_f16_storeu(uint16_t *dst, VecF32 v) {
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, v);
    for (int i = 0; i < 8; ++i)
      dst[i] = zendnnl::common::float16_t::f32_to_f16_val(tmp[i]);
  }
};

#undef LOWOHA_SIMD_AVX_ATTR

// ===========================================================================
// AVX + F16C + FMA3 specialization — 8 float lanes, Piledriver and later.
//
// Same 256-bit layout as avx_tag, refined for the family-15h cores that
// added FMA3 and F16C (Piledriver, Steamroller, Excavator — both absent on
// Bulldozer):
//   - vec_fmadd becomes a true fused multiply-add (VFMADD*PS): one
//     instruction and a single rounding instead of mul + add.
//   - FP16 <-> FP32 use the hardware VCVTPH2PS / VCVTPS2PH instead of the
//     scalar float16_t helpers.
//
// Everything else (loads, stores, reductions, exp kernels, bf16 pack and
// unpack) is inherited from SimdOps<avx_tag>, which is deliberately
// AVX2-free and therefore valid on every family-15h core.
//
// The inherited exp kernels are still written as mul + add, since they are
// shared with the Bulldozer tier, but they are not stuck with it: inlined
// into a caller that carries this tier's target attribute, the compiler
// contracts those products into FMAs (verified on GCC: this instantiation
// emits vfmadd*ps throughout the exp polynomials).
// ===========================================================================

#define LOWOHA_SIMD_AVX_F16C_ATTR __attribute__((target("avx,f16c,fma")))

template <>
struct SimdOps<avx_f16c_tag> : SimdOps<avx_tag> {

    LOWOHA_SIMD_AVX_F16C_ATTR
    static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
        return _mm256_fmadd_ps(a, b, c);
    }

    // 8 x IEEE 754 binary16 -> 8 x FP32 via VCVTPH2PS.
    LOWOHA_SIMD_AVX_F16C_ATTR
    static inline VecF32 vec_mask_f16_loadu(const uint16_t *p) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
        return _mm256_cvtph_ps(h);
    }

    // 8 x FP32 -> 8 x IEEE 754 binary16 via VCVTPS2PH, round-to-nearest-even.
    LOWOHA_SIMD_AVX_F16C_ATTR
    static inline void vec_f16_storeu(uint16_t *dst, VecF32 v) {
        __m128i h = _mm256_cvtps_ph(
                v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), h);
    }
};

#undef LOWOHA_SIMD_AVX_F16C_ATTR

// ===========================================================================
// AVX-512 specialization — 16 float lanes, enabled via target attribute.
//
// Note on FP16 conversion: the 512-bit forms _mm512_cvtph_ps and
// _mm512_cvtps_ph live in <avx512fintrin.h> and only require the avx512f
// target attribute.
// ===========================================================================

#define LOWOHA_SIMD_AVX512_ATTR \
    __attribute__((target("avx512f,avx512bw,avx512vl,fma")))

template <>
struct SimdOps<avx512_tag> {

    using VecF32 = __m512;
    static constexpr int kFloatLanes = 16;

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_loadu(const float *p) {
        return _mm512_loadu_ps(p);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline void vec_storeu(float *p, VecF32 x) {
        _mm512_storeu_ps(p, x);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_set1(float f) { return _mm512_set1_ps(f); }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_add(VecF32 a, VecF32 b) {
        return _mm512_add_ps(a, b);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_sub(VecF32 a, VecF32 b) {
        return _mm512_sub_ps(a, b);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_mul(VecF32 a, VecF32 b) {
        return _mm512_mul_ps(a, b);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_fmadd(VecF32 a, VecF32 b, VecF32 c) {
        return _mm512_fmadd_ps(a, b, c);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_max(VecF32 a, VecF32 b) {
        return _mm512_max_ps(a, b);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_min(VecF32 a, VecF32 b) {
        return _mm512_min_ps(a, b);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline float vec_hsum(VecF32 v) { return _mm512_reduce_add_ps(v); }

    LOWOHA_SIMD_AVX512_ATTR
    static inline float vec_hmax(VecF32 v) { return _mm512_reduce_max_ps(v); }

    // ── Fast exp (~20 ULP, Malossi et al.) ──────────────────────────────

    LOWOHA_SIMD_AVX512_ATTR
    static inline __m512 fexp_u20_ps_512(__m512 values) {
        static const __m512 vec_c0 = _mm512_set1_ps(0.00010703434948458272f);
        static const __m512 vec_c1 = _mm512_set1_ps(0.30354260500649682f);
        static const __m512 vec_c2 = _mm512_set1_ps(-0.22433836478672356f);
        static const __m512 vec_c3 = _mm512_set1_ps(-0.079204240219773236f);
        static const __m512 vec_exp_log2ef
                = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
        static const __m512 vec_a = _mm512_set1_ps(
                static_cast<float>(std::pow(2.0, 23) / std::log2(2.0)));
        static const __m512 vec_b
                = _mm512_set1_ps(static_cast<float>(std::pow(2.0, 23) * 127.0));
        static const __m512 vec_ln_flt_min
                = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
        static const __m512 vec_ln_flt_max
                = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
        static const __m512i vec_infinity = _mm512_set1_epi32(0x7F800000);
        static const __m512i vec_zero = _mm512_setzero_epi32();

        const __mmask16 min_mask
                = _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
        const __mmask16 max_mask
                = _mm512_cmp_ps_mask(values, vec_ln_flt_max, _CMP_GT_OS);

        __m512 vec_src = _mm512_mul_ps(values, vec_exp_log2ef);
        __m512 vec_fractional
                = _mm512_sub_ps(vec_src, _mm512_floor_ps(vec_src));

        __m512 vec_res = _mm512_fmadd_ps(vec_fractional, vec_c3, vec_c2);
        vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c1);
        vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c0);

        vec_src = _mm512_sub_ps(vec_src, vec_res);
        __m512 tmp = _mm512_fmadd_ps(vec_a, vec_src, vec_b);
        __m512i casted_integer = _mm512_cvttps_epi32(tmp);
        casted_integer
                = _mm512_mask_mov_epi32(casted_integer, min_mask, vec_zero);
        casted_integer
                = _mm512_mask_mov_epi32(casted_integer, max_mask, vec_infinity);
        return _mm512_castsi512_ps(casted_integer);
    }

    LOWOHA_SIMD_AVX512_ATTR
    static inline __m512 exp_u20_ps_512(__m512 values) {
        static const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f);
        static const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f);
        static const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f);
        static const __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f);
        static const __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f);
        static const __m512 vec_exp_log2ef
                = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));
        static const __m512 vec_half = _mm512_set1_ps(0.5f);
        static const __m512 vec_one = _mm512_set1_ps(1.f);
        static const __m512 vec_zero = _mm512_set1_ps(0.f);
        static const __m512 vec_ln2f
                = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));
        static const __m512 vec_ln_flt_min
                = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
        static const __m512 vec_ln_flt_max
                = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
        static const __m512i vec_126 = _mm512_set1_epi32(126);
        constexpr int n_mantissa_bits = 23;

        const __mmask16 less_ln_flt_min_mask
                = _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
        __m512 vec_src = _mm512_min_ps(values, vec_ln_flt_max);
        vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

        __m512 vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
        __m512i vec_fx_i = _mm512_cvt_roundps_epi32(
                vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

        __m512 vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

        __m512 vec_res = _mm512_fmadd_ps(
                vec_exp_poly, vec_factorial_5, vec_factorial_4);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

        // Construct 2^fx_i as two multiplications: poly * 2^(fx_i-1) * 2.
        // Direct construction of 2^fx_i would overflow the IEEE exponent field
        // when fx_i == 128 (exponent 255 = infinity). Splitting via (fx_i-1)+127
        // keeps the intermediate in the normal range (max exponent 254 = 2^127),
        // and the final *2 reaches 2^128 through normal FP multiplication.
        // Merge the (fx_i - 1 + 127) into a single (fx_i + 126).
        __m512i vec_two_pow_n_i = _mm512_add_epi32(vec_fx_i, vec_126);
        vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
        __m512 vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
        vec_two_pow_n = _mm512_mask_blend_ps(
                less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

        vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
        vec_res = _mm512_mul_ps(vec_res, _mm512_set1_ps(2.f));
        return vec_res;
    }

    // ── Public exp wrappers ─────────────────────────────────────────────

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_exp_u20(VecF32 v) { return exp_u20_ps_512(v); }

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_fexp_u20(VecF32 v) { return fexp_u20_ps_512(v); }

    // ── Reductions ──────────────────────────────────────────────────────

    LOWOHA_SIMD_AVX512_ATTR
    static inline float vec_reduce_sum(VecF32 acc) { return vec_hsum(acc); }

    LOWOHA_SIMD_AVX512_ATTR
    static inline float vec_reduce_max(VecF32 acc) { return vec_hmax(acc); }

    // ── BF16 ────────────────────────────────────────────────────────────

    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_mask_bf16_loadu(const uint16_t *p) {
        __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        __m512i wide = _mm512_slli_epi32(_mm512_cvtepu16_epi32(u), 16);
        return _mm512_castsi512_ps(wide);
    }

    // FP32 → BF16 store with round-to-nearest-even (no avx512bf16 ISA needed).
    // Inverse of vec_mask_bf16_loadu: packs 16 × FP32 into 16 × BF16.
    LOWOHA_SIMD_AVX512_ATTR
    static inline void vec_bf16_storeu(uint16_t *dst, VecF32 v) {
        __m512i u = _mm512_castps_si512(v);
        __m512i rounding_bias
                = _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(u, 16),
                                           _mm512_set1_epi32(1)),
                        _mm512_set1_epi32(0x7FFF));
        __m512i rounded
                = _mm512_srli_epi32(_mm512_add_epi32(u, rounding_bias), 16);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst),
                _mm512_cvtepi32_epi16(rounded));
    }

    // ── FP16 (IEEE 754 half) ────────────────────────────────────────────
    // Loads 16 × uint16_t storing IEEE half-precision values and widens to
    // 16 × FP32 lanes via the 512-bit form of VCVTPH2PS, which is part of
    // AVX-512F instruction set.
    LOWOHA_SIMD_AVX512_ATTR
    static inline VecF32 vec_mask_f16_loadu(const uint16_t *p) {
        __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        return _mm512_cvtph_ps(u);
    }

    // FP32 → FP16 store with round-to-nearest-even via vcvtps2ph
    // (AVX-512F intrinsic). Packs 16 × FP32 into 16 × IEEE half.
    LOWOHA_SIMD_AVX512_ATTR
    static inline void vec_f16_storeu(uint16_t *dst, VecF32 v) {
        __m256i h = _mm512_cvtps_ph(
                v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), h);
    }
};

#undef LOWOHA_SIMD_AVX512_ATTR

} // namespace simd
} // namespace lowoha
} // namespace zendnnl
