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

#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "operators/common/post_op.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;

// AVX-512 math helpers (avx512_exp, avx512_tanh, avx512_sigmoid, avx512_erf)
// are provided by avx512_math.hpp — shared with the microkernel epilogue.

// ============================================================================
// Main vectorized post-op application
// ============================================================================

__attribute__((target("avx512f,avx512bw,fma"))) static void
apply_postops_tile_avx512(float *C, int ldc, int m_count, int n_count,
        int n_offset, int m_offset, const float *bias,
        const std::vector<matmul_post_op> &postops) {

    // Bias addition (vectorized)
    if (bias != nullptr) {
        const float *bias_row = bias + n_offset;
        for (int mr = 0; mr < m_count; ++mr) {
            float *row = C + mr * ldc;
            int nr = 0;
            for (; nr + 15 < n_count; nr += 16) {
                __m512 c = _mm512_loadu_ps(row + nr);
                __m512 b = _mm512_loadu_ps(bias_row + nr);
                _mm512_storeu_ps(row + nr, _mm512_add_ps(c, b));
            }
            for (; nr < n_count; ++nr)
                row[nr] += bias_row[nr];
        }
    }

    for (const auto &po : postops) {
        switch (po.po_type) {

            case post_op_type_t::relu: {
                __m512 zero = _mm512_setzero_ps();
                __m512 alpha_v = _mm512_set1_ps(po.alpha);
                bool is_pure_relu = (po.alpha == 0.0f);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        if (is_pure_relu) {
                            v = _mm512_max_ps(v, zero);
                        } else {
                            // Leaky: v < 0 ? alpha*v : v
                            __mmask16 neg
                                    = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
                            v = _mm512_mask_mul_ps(v, neg, v, alpha_v);
                        }
                        _mm512_storeu_ps(row + nr, v);
                    }
                    for (; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f)
                            val = is_pure_relu ? 0.0f : po.alpha * val;
                    }
                }
                break;
            }

            case post_op_type_t::leaky_relu: {
                __m512 zero = _mm512_setzero_ps();
                __m512 alpha_v = _mm512_set1_ps(po.alpha);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        __mmask16 neg = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
                        v = _mm512_mask_mul_ps(v, neg, v, alpha_v);
                        _mm512_storeu_ps(row + nr, v);
                    }
                    for (; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f) val *= po.alpha;
                    }
                }
                break;
            }

            case post_op_type_t::elu: {
                __m512 zero = _mm512_setzero_ps();
                __m512 one = _mm512_set1_ps(1.0f);
                __m512 alpha_v = _mm512_set1_ps(po.alpha);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        __mmask16 neg = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
                        // ELU: alpha * (exp(v) - 1) for v < 0
                        __m512 elu_val = _mm512_mul_ps(
                                alpha_v, _mm512_sub_ps(avx512_exp(v), one));
                        v = _mm512_mask_mov_ps(v, neg, elu_val);
                        _mm512_storeu_ps(row + nr, v);
                    }
                    for (; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f) val = po.alpha * (std::exp(val) - 1.0f);
                    }
                }
                break;
            }

            case post_op_type_t::gelu_tanh: {
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
                __m512 half = _mm512_set1_ps(0.5f);
                __m512 one = _mm512_set1_ps(1.0f);
                __m512 c = _mm512_set1_ps(0.7978845608028654f); // sqrt(2/pi)
                __m512 c2 = _mm512_set1_ps(0.044715f);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 x = _mm512_loadu_ps(row + nr);
                        __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
                        __m512 inner
                                = _mm512_mul_ps(c, _mm512_fmadd_ps(c2, x3, x));
                        __m512 result = _mm512_mul_ps(half,
                                _mm512_mul_ps(x,
                                        _mm512_add_ps(
                                                one, avx512_tanh(inner))));
                        _mm512_storeu_ps(row + nr, result);
                    }
                    for (; nr < n_count; ++nr) {
                        float &v = row[nr];
                        float x3 = v * v * v;
                        v = 0.5f * v
                                * (1.0f
                                        + std::tanh(0.7978845608028654f
                                                * (v + 0.044715f * x3)));
                    }
                }
                break;
            }

            case post_op_type_t::gelu_erf: {
                // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
                __m512 half = _mm512_set1_ps(0.5f);
                __m512 one = _mm512_set1_ps(1.0f);
                __m512 inv_sqrt2 = _mm512_set1_ps(0.7071067811865476f);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 x = _mm512_loadu_ps(row + nr);
                        __m512 erf_val
                                = avx512_erf(_mm512_mul_ps(x, inv_sqrt2));
                        __m512 result = _mm512_mul_ps(half,
                                _mm512_mul_ps(x, _mm512_add_ps(one, erf_val)));
                        _mm512_storeu_ps(row + nr, result);
                    }
                    for (; nr < n_count; ++nr) {
                        float &v = row[nr];
                        v = 0.5f * v
                                * (1.0f + std::erf(v * 0.7071067811865476f));
                    }
                }
                break;
            }

            case post_op_type_t::sigmoid: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 x = _mm512_loadu_ps(row + nr);
                        _mm512_storeu_ps(row + nr, avx512_sigmoid(x));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = 1.0f / (1.0f + std::exp(-row[nr]));
                }
                break;
            }

            case post_op_type_t::swish: {
                // swish(x) = x * sigmoid(alpha * x)
                __m512 alpha_v = _mm512_set1_ps(po.alpha);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 x = _mm512_loadu_ps(row + nr);
                        __m512 sig = avx512_sigmoid(_mm512_mul_ps(alpha_v, x));
                        _mm512_storeu_ps(row + nr, _mm512_mul_ps(x, sig));
                    }
                    for (; nr < n_count; ++nr) {
                        float &v = row[nr];
                        v = v / (1.0f + std::exp(-po.alpha * v));
                    }
                }
                break;
            }

            case post_op_type_t::tanh: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 x = _mm512_loadu_ps(row + nr);
                        _mm512_storeu_ps(row + nr, avx512_tanh(x));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = std::tanh(row[nr]);
                }
                break;
            }

            case post_op_type_t::square: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        _mm512_storeu_ps(row + nr, _mm512_mul_ps(v, v));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] *= row[nr];
                }
                break;
            }

            case post_op_type_t::abs: {
                // abs = clear sign bit (AND with 0x7FFFFFFF)
                __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512i v = _mm512_loadu_si512(
                                reinterpret_cast<const __m512i *>(row + nr));
                        _mm512_storeu_si512(
                                reinterpret_cast<__m512i *>(row + nr),
                                _mm512_and_si512(v, mask));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = std::abs(row[nr]);
                }
                break;
            }

            case post_op_type_t::sqrt: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        _mm512_storeu_ps(row + nr, _mm512_sqrt_ps(v));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = std::sqrt(row[nr]);
                }
                break;
            }

            case post_op_type_t::exp: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        _mm512_storeu_ps(row + nr, avx512_exp(v));
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = std::exp(row[nr]);
                }
                break;
            }

            case post_op_type_t::log: {
                // log via exp: not efficient, use scalar fallback
                // TODO: vectorized log approximation
                for (int mr = 0; mr < m_count; ++mr)
                    for (int nr = 0; nr < n_count; ++nr)
                        C[mr * ldc + nr] = std::log(C[mr * ldc + nr]);
                break;
            }

            case post_op_type_t::clip: {
                // clip(x, alpha, beta) = min(max(x, alpha), beta)
                __m512 lo = _mm512_set1_ps(po.alpha);
                __m512 hi = _mm512_set1_ps(po.beta);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;
                    for (; nr + 15 < n_count; nr += 16) {
                        __m512 v = _mm512_loadu_ps(row + nr);
                        v = _mm512_max_ps(v, lo);
                        v = _mm512_min_ps(v, hi);
                        _mm512_storeu_ps(row + nr, v);
                    }
                    for (; nr < n_count; ++nr)
                        row[nr] = std::min(
                                std::max(row[nr], po.alpha), po.beta);
                }
                break;
            }

            case post_op_type_t::binary_add:
            case post_op_type_t::binary_mul: {
                if (po.buff == nullptr) break;
                // For 2D binary buffers, leading_dim should be set by the caller
                // to the full output N. The n_count fallback is correct only when
                // the caller tiles with proper n_offset into the binary buffer.
                int ld = (po.leading_dim > 0) ? po.leading_dim : n_count;
                bool is_1d = (po.dims.size() <= 1)
                        || (po.dims.size() == 2 && po.dims[0] == 1);
                bool is_add = (po.po_type == post_op_type_t::binary_add);
                bool bin_is_bf16 = (po.dtype == data_type_t::bf16);

                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    int nr = 0;

                    if (bin_is_bf16) {
                        // BF16 binary buffer: load BF16, convert to FP32, apply op
                        const uint16_t *bin_bf16
                                = static_cast<const uint16_t *>(po.buff);
                        for (; nr + 15 < n_count; nr += 16) {
                            __m512 c = _mm512_loadu_ps(row + nr);
                            const uint16_t *bp;
                            if (is_1d) {
                                bp = bin_bf16 + n_offset + nr;
                            } else {
                                bp = bin_bf16 + (m_offset + mr) * ld
                                        + (n_offset + nr);
                            }
                            // BF16 → FP32: zero-extend to 32-bit, shift left 16
                            __m256i bf = _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i *>(bp));
                            __m512 b = _mm512_castsi512_ps(_mm512_slli_epi32(
                                    _mm512_cvtepu16_epi32(bf), 16));
                            if (is_add)
                                _mm512_storeu_ps(row + nr, _mm512_add_ps(c, b));
                            else
                                _mm512_storeu_ps(row + nr, _mm512_mul_ps(c, b));
                        }
                        for (; nr < n_count; ++nr) {
                            const uint16_t *bp;
                            if (is_1d)
                                bp = bin_bf16 + n_offset + nr;
                            else
                                bp = bin_bf16 + (m_offset + mr) * ld
                                        + (n_offset + nr);
                            uint32_t bits = static_cast<uint32_t>(*bp) << 16;
                            float bval;
                            std::memcpy(&bval, &bits, sizeof(bval));
                            if (is_add)
                                row[nr] += bval;
                            else
                                row[nr] *= bval;
                        }
                    } else {
                        // FP32 binary buffer (default)
                        const float *bin = static_cast<const float *>(po.buff);
                        for (; nr + 15 < n_count; nr += 16) {
                            __m512 c = _mm512_loadu_ps(row + nr);
                            __m512 b;
                            if (is_1d) {
                                b = _mm512_loadu_ps(bin + n_offset + nr);
                            } else {
                                b = _mm512_loadu_ps(bin + (m_offset + mr) * ld
                                        + (n_offset + nr));
                            }
                            if (is_add)
                                _mm512_storeu_ps(row + nr, _mm512_add_ps(c, b));
                            else
                                _mm512_storeu_ps(row + nr, _mm512_mul_ps(c, b));
                        }
                        for (; nr < n_count; ++nr) {
                            float bval = is_1d ? bin[n_offset + nr]
                                               : bin[(m_offset + mr) * ld
                                                         + (n_offset + nr)];
                            if (is_add)
                                row[nr] += bval;
                            else
                                row[nr] *= bval;
                        }
                    }
                }
                break;
            }

            default: break;
        }
    }
}

// ============================================================================
// Portable post-op application
// ============================================================================

// Same arithmetic as apply_postops_tile_avx512() with no vector instructions, so
// hosts without AVX-512 -- every AMD family 15h part, and Zen 1/2/3 -- can run
// the native matmul epilogue instead of having the whole problem declined.
//
// Every case below is the scalar tail loop of its AVX-512 counterpart above,
// which each already carried one for the sub-16-element remainder. Keeping them
// literally identical is what makes the two paths agree numerically: the
// AVX-512 version's own remainder elements go through this same expression.
//
// The one deliberate difference is the transcendentals. The AVX-512 path uses
// the polynomial approximations in avx512_math.hpp (avx512_exp, avx512_tanh,
// avx512_sigmoid, avx512_erf) for full vectors, whereas this uses libm. libm is
// the more accurate of the two, so results can differ in the last bits -- the
// same tolerance the AVX-512 path's own scalar tail already relies on.
static void apply_postops_tile_scalar(float *C, int ldc, int m_count,
        int n_count, int n_offset, int m_offset, const float *bias,
        const std::vector<matmul_post_op> &postops) {

    if (bias != nullptr) {
        const float *bias_row = bias + n_offset;
        for (int mr = 0; mr < m_count; ++mr) {
            float *row = C + mr * ldc;
            for (int nr = 0; nr < n_count; ++nr) {
                row[nr] += bias_row[nr];
            }
        }
    }

    for (const auto &po : postops) {
        switch (po.po_type) {

            case post_op_type_t::relu: {
                const bool is_pure_relu = (po.alpha == 0.0f);
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f) {
                            val = is_pure_relu ? 0.0f : po.alpha * val;
                        }
                    }
                }
                break;
            }

            case post_op_type_t::leaky_relu: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f) { val *= po.alpha; }
                    }
                }
                break;
            }

            case post_op_type_t::elu: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &val = row[nr];
                        if (val < 0.0f) {
                            val = po.alpha * (std::exp(val) - 1.0f);
                        }
                    }
                }
                break;
            }

            case post_op_type_t::gelu_tanh: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &v = row[nr];
                        const float x3 = v * v * v;
                        v = 0.5f * v
                                * (1.0f
                                        + std::tanh(0.7978845608028654f
                                                * (v + 0.044715f * x3)));
                    }
                }
                break;
            }

            case post_op_type_t::gelu_erf: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &v = row[nr];
                        v = 0.5f * v
                                * (1.0f + std::erf(v * 0.7071067811865476f));
                    }
                }
                break;
            }

            case post_op_type_t::sigmoid: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = 1.0f / (1.0f + std::exp(-row[nr]));
                    }
                }
                break;
            }

            case post_op_type_t::swish: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float &v = row[nr];
                        v = v / (1.0f + std::exp(-po.alpha * v));
                    }
                }
                break;
            }

            case post_op_type_t::tanh: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = std::tanh(row[nr]);
                    }
                }
                break;
            }

            case post_op_type_t::square: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] *= row[nr];
                    }
                }
                break;
            }

            case post_op_type_t::abs: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = std::abs(row[nr]);
                    }
                }
                break;
            }

            case post_op_type_t::sqrt: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = std::sqrt(row[nr]);
                    }
                }
                break;
            }

            case post_op_type_t::exp: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = std::exp(row[nr]);
                    }
                }
                break;
            }

            case post_op_type_t::log: {
                for (int mr = 0; mr < m_count; ++mr) {
                    for (int nr = 0; nr < n_count; ++nr) {
                        C[mr * ldc + nr] = std::log(C[mr * ldc + nr]);
                    }
                }
                break;
            }

            case post_op_type_t::clip: {
                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        row[nr] = std::min(
                                std::max(row[nr], po.alpha), po.beta);
                    }
                }
                break;
            }

            case post_op_type_t::binary_add:
            case post_op_type_t::binary_mul: {
                if (po.buff == nullptr) { break; }
                const int ld = (po.leading_dim > 0) ? po.leading_dim : n_count;
                const bool is_1d = (po.dims.size() <= 1)
                        || (po.dims.size() == 2 && po.dims[0] == 1);
                const bool is_add = (po.po_type == post_op_type_t::binary_add);
                const bool bin_is_bf16 = (po.dtype == data_type_t::bf16);

                for (int mr = 0; mr < m_count; ++mr) {
                    float *row = C + mr * ldc;
                    for (int nr = 0; nr < n_count; ++nr) {
                        float bval;
                        if (bin_is_bf16) {
                            const uint16_t *bin_bf16
                                    = static_cast<const uint16_t *>(po.buff);
                            const uint16_t *bp = is_1d
                                    ? bin_bf16 + n_offset + nr
                                    : bin_bf16 + (m_offset + mr) * ld
                                            + (n_offset + nr);
                            const uint32_t bits
                                    = static_cast<uint32_t>(*bp) << 16;
                            std::memcpy(&bval, &bits, sizeof(bval));
                        } else {
                            const float *bin
                                    = static_cast<const float *>(po.buff);
                            bval = is_1d ? bin[n_offset + nr]
                                         : bin[(m_offset + mr) * ld
                                                 + (n_offset + nr)];
                        }
                        if (is_add) {
                            row[nr] += bval;
                        } else {
                            row[nr] *= bval;
                        }
                    }
                }
                break;
            }

            default: break;
        }
    }
}

void apply_postops_tile(float *C, int ldc, int m_count, int n_count,
        int n_offset, int m_offset, const float *bias,
        const std::vector<matmul_post_op> &postops) {
    // Cached once: detect_uarch() itself caches, this just avoids the repeated
    // indirection in what is a per-tile call.
    static const bool has_avx512 = detect_uarch().avx512f;
    if (has_avx512) {
        apply_postops_tile_avx512(
                C, ldc, m_count, n_count, n_offset, m_offset, bias, postops);
    } else {
        apply_postops_tile_scalar(
                C, ldc, m_count, n_count, n_offset, m_offset, bias, postops);
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
