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
// Portable RMS / Layer normalization for hosts without AVX-512.
//
// The AVX-512 kernels are the fast path where they can run; without AVX-512 the
// only fallback was reference_kernel.cpp, which reads every element through
// load_scalar() -- a switch on the runtime dtype, per element -- and makes two
// separate passes over the row for LayerNorm (one for the mean, another for the
// squared deviations). Measured on an A10-8770E (Excavator), RMSNorm 32x4096:
// f32 0.203 ms, and BF16 *slower* at 0.558 ms despite moving half the bytes,
// which is the per-element dtype dispatch showing through.
//
// This kernel fixes three things:
//
//   - The dtype dispatch is hoisted out of the inner loop. Rows are processed by
//     a template instantiated on the source and destination types, so the switch
//     happens once per call rather than once per element.
//   - Statistics come from a single pass. sum and sum-of-squares accumulate
//     together and the variance is recovered as sum_sq/n - mean^2, which is what
//     the AVX-512 kernel already does (see rmsnorm/layernorm avx512 kernels), so
//     this matches the existing numerics rather than introducing a third
//     convention. It halves LayerNorm's read traffic.
//   - Arithmetic is 128-bit vector. Family 15h implements the FPU as two
//     128-bit FMAC pipes, and AMD publication 47414 notes only one 256-bit
//     operation issues per cycle, so 128-bit is the throughput width here. Four
//     independent accumulators cover the FMA latency.
//
// BF16 is handled in vector form rather than element-wise: widening is a 16-bit
// left shift, which is an unpack against zero, and narrowing is round-to-nearest
// -even done with integer ops. That is where most of the BF16 gain comes from.
//
// Scope is deliberately narrow: RMSNorm and LayerNorm, with f32 or bf16 in and
// out. gamma/beta may be f32, bf16 or f16 -- they are norm_size elements reused
// by every row, so they are widened to f32 once per call rather than converted
// per element. BatchNorm, f16 source/destination, and FUSED_ADD_RMS_NORM (which
// updates `residual` in place before normalising) still go to the reference
// kernel, which remains correct.
//

#include "lowoha_operators/normalization/kernel/normalization_portable_kernel.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/float16.hpp"

#if defined(__AVX__)
#if defined(__FMA4__) || defined(__XOP__)
// FMA4/XOP intrinsics live in x86intrin.h, not immintrin.h.
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif
#endif

namespace zendnnl {
namespace lowoha {
namespace normalization {

#if defined(__AVX__)

namespace {

// Best 128-bit multiply-add this target offers. FMA4 covers all of family 15h,
// including Bulldozer, which has no FMA3.
inline __m128 fmadd128(__m128 a, __m128 b, __m128 acc) {
#if defined(__FMA4__)
    return _mm_macc_ps(a, b, acc);
#elif defined(__FMA__)
    return _mm_fmadd_ps(a, b, acc);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), acc);
#endif
}

// ---------------------------------------------------------------------------
// Element access, resolved at compile time by dtype tag
// ---------------------------------------------------------------------------

struct f32_tag {};
struct bf16_tag {};

inline __m128 load4(const float *p, f32_tag) {
    return _mm_loadu_ps(p);
}

// BF16 -> FP32 is a 16-bit left shift: unpacking against zero puts each 16-bit
// payload into the high half of a 32-bit lane, which is exactly that shift.
inline __m128 load4(const uint16_t *p, bf16_tag) {
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    return _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), raw));
}

inline void store4(float *p, __m128 v, f32_tag) {
    _mm_storeu_ps(p, v);
}

// FP32 -> BF16 with round-to-nearest-even, matching
// bfloat16_t::f32_to_bf16_val: add 0x7FFF plus the rounding bit, then take the
// high half.
inline void store4(uint16_t *p, __m128 v, bf16_tag) {
    __m128i bits = _mm_castps_si128(v);
    const __m128i lsb
            = _mm_and_si128(_mm_srli_epi32(bits, 16), _mm_set1_epi32(1));
    bits = _mm_add_epi32(
            bits, _mm_add_epi32(_mm_set1_epi32(0x7FFF), lsb));
    const __m128i hi = _mm_srli_epi32(bits, 16);
    // Pack the four 32-bit lanes down to four 16-bit values.
    const __m128i packed = _mm_shuffle_epi8(hi,
            _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1,
                    -1));
    _mm_storel_epi64(reinterpret_cast<__m128i *>(p), packed);
}

inline float load1(const float *p, f32_tag) {
    return *p;
}

inline float load1(const uint16_t *p, bf16_tag) {
    const uint32_t bits = static_cast<uint32_t>(*p) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline void store1(float *p, float v, f32_tag) {
    *p = v;
}

inline void store1(uint16_t *p, float v, bf16_tag) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t rounding_bias = (bits >> 16) & 1;
    bits += 0x7FFF + rounding_bias;
    *p = static_cast<uint16_t>(bits >> 16);
}

inline float horizontal_add(__m128 v) {
    // Two shuffles and two adds; no AVX-512 reduce available here.
    const __m128 hi64 = _mm_movehl_ps(v, v);
    const __m128 sum2 = _mm_add_ps(v, hi64);
    const __m128 hi32 = _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1, 1, 1, 1));
    return _mm_cvtss_f32(_mm_add_ss(sum2, hi32));
}

// One row: accumulate sum and sum-of-squares in a single pass, four accumulator
// pairs deep so neither chain waits on its own latency.
template <typename SrcT, typename SrcTag>
inline void row_stats(const SrcT *in, uint64_t n, float &out_sum,
        float &out_sum_sq, SrcTag tag) {
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
    __m128 q0 = _mm_setzero_ps(), q1 = _mm_setzero_ps();
    __m128 q2 = _mm_setzero_ps(), q3 = _mm_setzero_ps();

    uint64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m128 v0 = load4(in + i, tag);
        const __m128 v1 = load4(in + i + 4, tag);
        const __m128 v2 = load4(in + i + 8, tag);
        const __m128 v3 = load4(in + i + 12, tag);
        s0 = _mm_add_ps(s0, v0);
        s1 = _mm_add_ps(s1, v1);
        s2 = _mm_add_ps(s2, v2);
        s3 = _mm_add_ps(s3, v3);
        q0 = fmadd128(v0, v0, q0);
        q1 = fmadd128(v1, v1, q1);
        q2 = fmadd128(v2, v2, q2);
        q3 = fmadd128(v3, v3, q3);
    }
    for (; i + 4 <= n; i += 4) {
        const __m128 v = load4(in + i, tag);
        s0 = _mm_add_ps(s0, v);
        q0 = fmadd128(v, v, q0);
    }

    float sum = horizontal_add(_mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3)));
    float sum_sq
            = horizontal_add(_mm_add_ps(_mm_add_ps(q0, q1), _mm_add_ps(q2, q3)));
    for (; i < n; ++i) {
        const float x = load1(in + i, tag);
        sum += x;
        sum_sq += x * x;
    }
    out_sum = sum;
    out_sum_sq = sum_sq;
}

// Second pass: y = (x - shift) * scale * gamma [+ beta]. RMSNorm passes
// shift = 0, so the same body serves both norms.
template <typename SrcT, typename DstT, typename SrcTag, typename DstTag>
inline void row_apply(const SrcT *in, DstT *out, uint64_t n, float shift,
        float scale, const float *gamma, const float *beta, SrcTag stag,
        DstTag dtag) {
    const __m128 vshift = _mm_set1_ps(shift);
    const __m128 vscale = _mm_set1_ps(scale);

    uint64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_mul_ps(_mm_sub_ps(load4(in + i, stag), vshift), vscale);
        if (gamma != nullptr) { v = _mm_mul_ps(v, _mm_loadu_ps(gamma + i)); }
        if (beta != nullptr) { v = _mm_add_ps(v, _mm_loadu_ps(beta + i)); }
        store4(out + i, v, dtag);
    }
    for (; i < n; ++i) {
        float v = (load1(in + i, stag) - shift) * scale;
        if (gamma != nullptr) { v *= gamma[i]; }
        if (beta != nullptr) { v += beta[i]; }
        store1(out + i, v, dtag);
    }
}

// gamma and beta arrive in whatever dtype the caller supplied (f32, bf16 or
// f16). They are norm_size elements and reused by every row, so widen them once
// per call into scratch rather than converting per element inside the row loop.
inline void widen_param(const void *src, data_type_t dt, uint64_t n,
        std::vector<float> &scratch) {
    scratch.resize(n);
    switch (dt) {
        case data_type_t::f32:
            std::memcpy(scratch.data(), src, n * sizeof(float));
            break;
        case data_type_t::bf16:
            for (uint64_t i = 0; i < n; ++i) {
                scratch[i] = load1(static_cast<const uint16_t *>(src) + i,
                        bf16_tag {});
            }
            break;
        case data_type_t::f16:
            for (uint64_t i = 0; i < n; ++i) {
                scratch[i] = float16_t::f16_to_f32_val(
                        static_cast<const uint16_t *>(src)[i]);
            }
            break;
        default: scratch.assign(n, 0.0f); break;
    }
}

// One instantiation per (src, dst) dtype pair. `subtract_mean` selects LayerNorm
// (centre the row) from RMSNorm (scale only), which is the only structural
// difference between the two.
template <typename SrcT, typename DstT, typename SrcTag, typename DstTag>
void run_rows(const void *input, void *output, const float *gamma,
        const float *beta, const norm_params &params, int num_threads,
        bool subtract_mean, SrcTag stag, DstTag dtag) {
    const uint64_t batch = params.batch;
    const uint64_t n = params.norm_size;
    const float eps = params.epsilon;
    const float inv_n = 1.0f / static_cast<float>(n);

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t b = 0; b < batch; ++b) {
        const SrcT *in = static_cast<const SrcT *>(input) + b * n;
        DstT *out = static_cast<DstT *>(output) + b * n;

        float sum = 0.0f, sum_sq = 0.0f;
        row_stats<SrcT, SrcTag>(in, n, sum, sum_sq, stag);

        float shift = 0.0f;
        float scale;
        if (subtract_mean) {
            const float mean = sum * inv_n;
            // sum_sq/n - mean^2, the same recovery the AVX-512 kernels use.
            const float var = sum_sq * inv_n - mean * mean;
            shift = mean;
            scale = 1.0f / std::sqrt(var + eps);
        } else {
            scale = 1.0f / std::sqrt(sum_sq * inv_n + eps);
        }

        row_apply<SrcT, DstT, SrcTag, DstTag>(
                in, out, n, shift, scale, gamma, beta, stag, dtag);
    }
}

} // namespace

#endif // __AVX__

bool normalization_portable_supported(const norm_params &params) {
#if !defined(__AVX__)
    (void)params;
    return false;
#else
    // FUSED_ADD_RMS_NORM is excluded on purpose: it updates `residual` in place
    // (residual[i] += input[i]) before normalising, a different data flow that
    // the reference kernel still handles.
    const bool norm_ok = params.norm_type == norm_type_t::RMS_NORM
            || params.norm_type == norm_type_t::LAYER_NORM;
    const auto dt_ok = [](data_type_t dt) {
        return dt == data_type_t::f32 || dt == data_type_t::bf16;
    };
    return norm_ok && dt_ok(params.src_dt) && dt_ok(params.dst_dt);
#endif
}

status_t normalization_portable(const void *input, void *output,
        const void *gamma, const void *beta, norm_params &params,
        int num_threads) {
#if !defined(__AVX__)
    (void)input;
    (void)output;
    (void)gamma;
    (void)beta;
    (void)params;
    (void)num_threads;
    return status_t::unimplemented;
#else
    if (!normalization_portable_supported(params)) {
        return status_t::unimplemented;
    }

    const bool subtract_mean = (params.norm_type == norm_type_t::LAYER_NORM);
    // RMSNorm has no shift; LayerNorm honours use_shift.
    const bool want_beta = subtract_mean && params.use_shift && beta != nullptr;

    std::vector<float> gamma_f32;
    std::vector<float> beta_f32;
    const float *g = nullptr;
    const float *bt = nullptr;
    if (params.use_scale && gamma != nullptr) {
        widen_param(gamma, params.gamma_dt, params.norm_size, gamma_f32);
        g = gamma_f32.data();
    }
    if (want_beta) {
        widen_param(beta, params.beta_dt, params.norm_size, beta_f32);
        bt = beta_f32.data();
    }

    const bool src_bf16 = (params.src_dt == data_type_t::bf16);
    const bool dst_bf16 = (params.dst_dt == data_type_t::bf16);

    if (!src_bf16 && !dst_bf16) {
        run_rows<float, float>(input, output, g, bt, params, num_threads,
                subtract_mean, f32_tag {}, f32_tag {});
    } else if (src_bf16 && dst_bf16) {
        run_rows<uint16_t, uint16_t>(input, output, g, bt, params, num_threads,
                subtract_mean, bf16_tag {}, bf16_tag {});
    } else if (src_bf16) {
        run_rows<uint16_t, float>(input, output, g, bt, params, num_threads,
                subtract_mean, bf16_tag {}, f32_tag {});
    } else {
        run_rows<float, uint16_t>(input, output, g, bt, params, num_threads,
                subtract_mean, f32_tag {}, bf16_tag {});
    }

    return status_t::success;
#endif
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl
