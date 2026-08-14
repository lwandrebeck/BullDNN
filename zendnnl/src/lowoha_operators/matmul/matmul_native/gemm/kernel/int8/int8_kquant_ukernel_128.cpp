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

// Asymmetric per-group INT8 microkernel for GGML k-quants. The header carries the
// derivation, the instruction census and the saturation bound; this file is the
// arithmetic.

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_kquant_ukernel_128.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

#if defined(__x86_64__) || defined(__i386__)

bool kq_host_has_ssse3_sse41() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 9)) != 0 && (ecx & (1u << 19)) != 0;
}

// CPUID Fn8000_0001_ECX bit 11; see the symmetric kernel for why platform_info
// cannot answer this.
bool kq_host_has_xop() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx)) return false;
    if (eax < 0x80000001u) return false;
    if (!__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 11)) != 0;
}

// Keep a pair of vectors in registers across the loop body; emits nothing. The
// symmetric kernel's header explains why this is aimed at the accumulators rather
// than at B. This kernel holds fewer live values (no |A| per row, no sign
// results), so it has room to spare -- pinning is kept for the same reason, since
// GCC still interleaves all four rows' transients.
template <typename Vec>
inline void kq_pin_reg(Vec &v0, Vec &v1) {
#if defined(__GNUC__)
    asm("" : "+x"(v0), "+x"(v1));
#endif
}

// The body is shared between the flavours and between row counts; only
// reduce_accum() differs between flavours.
#define KQ_UK128_BODY                                                          \
    template <int kMR>                                                         \
    void ukernel_mr(const int8_t *__restrict__ A, int a_stride,                \
            const uint8_t *__restrict__ B_vnni, int b_stride,                  \
            float *__restrict__ C, int ldc, int k, int group_size,             \
            const float *__restrict__ wei_scale,                               \
            const float *__restrict__ wei_min, int ws_stride,                  \
            const int32_t *__restrict__ row_sums, int rs_stride,               \
            const float *__restrict__ src_scale, int ss_row, int ss_grp) {     \
        const int quads_per_group = group_size / KQ_VNNI_GRP;                  \
        const int n_groups = k / group_size;                                   \
        const bool src_uniform = (ss_row == 0 && ss_grp == 0);                 \
        const __m128 v_src = _mm_set1_ps(src_scale[0]);                        \
                                                                               \
        for (int g = 0; g < n_groups; ++g) {                                   \
            __m128i acc[kMR][2];                                               \
            for (int m = 0; m < kMR; ++m) {                                    \
                acc[m][0] = _mm_setzero_si128();                               \
                acc[m][1] = _mm_setzero_si128();                               \
            }                                                                  \
                                                                               \
            for (int q = 0; q < quads_per_group; ++q) {                        \
                const int kq = g * quads_per_group + q;                        \
                const uint8_t *bq = B_vnni + kq * b_stride;                    \
                const __m128i b0 = _mm_loadu_si128(                            \
                        reinterpret_cast<const __m128i *>(bq));                \
                const __m128i b1 = _mm_loadu_si128(                            \
                        reinterpret_cast<const __m128i *>(bq + 16));           \
                                                                               \
                for (int m = 0; m < kMR; ++m) {                                \
                    int32_t a_quad;                                            \
                    std::memcpy(&a_quad,                                       \
                            A + m * a_stride + kq * KQ_VNNI_GRP, 4);           \
                    const __m128i av = _mm_set1_epi32(a_quad);                 \
                    /* Codes unsigned, activations signed: exactly the operand \
                     * pairing PMADDUBSW takes, with no sign trick needed. */  \
                    acc[m][0] = reduce_accum(                                  \
                            _mm_maddubs_epi16(b0, av), acc[m][0]);             \
                    acc[m][1] = reduce_accum(                                  \
                            _mm_maddubs_epi16(b1, av), acc[m][1]);             \
                }                                                              \
                for (int m = 0; m < kMR; ++m)                                  \
                    kq_pin_reg(acc[m][0], acc[m][1]);                          \
            }                                                                  \
                                                                               \
            /* Flush: C += ss * (D * dot - M * rowsum). D and M are per column, \
             * the row sum is one scalar per row, so the correction is two      \
             * multiplies per group rather than any per-element work. */        \
            const __m128 d0 = _mm_loadu_ps(wei_scale + g * ws_stride);         \
            const __m128 d1 = _mm_loadu_ps(wei_scale + g * ws_stride + 4);     \
            const __m128 m0 = _mm_loadu_ps(wei_min + g * ws_stride);           \
            const __m128 m1 = _mm_loadu_ps(wei_min + g * ws_stride + 4);       \
            for (int m = 0; m < kMR; ++m) {                                    \
                const __m128 sv = src_uniform                                  \
                        ? v_src                                                \
                        : _mm_set1_ps(src_scale[m * ss_row + g * ss_grp]);     \
                const __m128 rs = _mm_set1_ps(static_cast<float>(              \
                        row_sums[m * rs_stride + g]));                         \
                const __m128 t0 = _mm_sub_ps(                                  \
                        _mm_mul_ps(_mm_cvtepi32_ps(acc[m][0]), d0),            \
                        _mm_mul_ps(m0, rs));                                   \
                const __m128 t1 = _mm_sub_ps(                                  \
                        _mm_mul_ps(_mm_cvtepi32_ps(acc[m][1]), d1),            \
                        _mm_mul_ps(m1, rs));                                   \
                float *c = C + m * ldc;                                        \
                _mm_storeu_ps(c,                                               \
                        _mm_add_ps(_mm_loadu_ps(c), _mm_mul_ps(t0, sv)));      \
                _mm_storeu_ps(c + 4,                                           \
                        _mm_add_ps(_mm_loadu_ps(c + 4), _mm_mul_ps(t1, sv)));  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    void ukernel(const int8_t *__restrict__ A, int a_stride,                   \
            const uint8_t *__restrict__ B_vnni, int b_stride,                  \
            float *__restrict__ C, int ldc, int k, int group_size,             \
            const float *__restrict__ wei_scale,                               \
            const float *__restrict__ wei_min, int ws_stride,                  \
            const int32_t *__restrict__ row_sums, int rs_stride,               \
            const float *__restrict__ src_scale, int ss_row, int ss_grp) {     \
        ukernel_mr<KQ_MR>(A, a_stride, B_vnni, b_stride, C, ldc, k, group_size, \
                wei_scale, wei_min, ws_stride, row_sums, rs_stride, src_scale,  \
                ss_row, ss_grp);                                               \
    }                                                                          \
                                                                               \
    void ukernel_m1(const int8_t *__restrict__ A, int a_stride,                \
            const uint8_t *__restrict__ B_vnni, int b_stride,                  \
            float *__restrict__ C, int ldc, int k, int group_size,             \
            const float *__restrict__ wei_scale,                               \
            const float *__restrict__ wei_min, int ws_stride,                  \
            const int32_t *__restrict__ row_sums, int rs_stride,               \
            const float *__restrict__ src_scale, int ss_row, int ss_grp) {     \
        ukernel_mr<1>(A, a_stride, B_vnni, b_stride, C, ldc, k, group_size,     \
                wei_scale, wei_min, ws_stride, row_sums, rs_stride, src_scale,  \
                ss_row, ss_grp);                                               \
    }

// ---- SSSE3/SSE4.1 ----------------------------------------------------------
#pragma GCC push_options
#pragma GCC target("ssse3,sse4.1")
namespace kq_sse {
inline __m128i reduce_accum(__m128i words, __m128i acc) {
    return _mm_add_epi32(acc, _mm_madd_epi16(words, _mm_set1_epi16(1)));
}
KQ_UK128_BODY
} // namespace kq_sse
#pragma GCC pop_options

// ---- XOP: family 15h only --------------------------------------------------
#pragma GCC push_options
#pragma GCC target("xop,ssse3,sse4.1")
namespace kq_xop {
// VPMADCSWD fuses the PMADDWD and the accumulate. Worth 1.13-1.20x on Excavator
// and nothing measurable on Piledriver -- see docs/perf; kept because it is never
// slower and the choice is per host.
inline __m128i reduce_accum(__m128i words, __m128i acc) {
    return (__m128i)__builtin_ia32_vpmadcswd(
            (__v8hi)words, (__v8hi)_mm_set1_epi16(1), (__v4si)acc);
}
KQ_UK128_BODY
} // namespace kq_xop
#pragma GCC pop_options

#endif // x86

} // namespace

int8_kquant_ukernel_128_fn_t select_int8_kquant_ukernel_128() {
#if defined(__x86_64__) || defined(__i386__)
    static const int8_kquant_ukernel_128_fn_t s_fn = [] {
        const char *no_xop = std::getenv("ZENDNNL_NATIVE_SYMQ_NO_XOP");
        const bool avoid_xop = no_xop != nullptr && no_xop[0] != '\0'
                && std::strcmp(no_xop, "0") != 0;
        if (!avoid_xop && kq_host_has_xop()) return &kq_xop::ukernel;
        if (!kq_host_has_ssse3_sse41())
            return static_cast<int8_kquant_ukernel_128_fn_t>(nullptr);
        return &kq_sse::ukernel;
    }();
    return s_fn;
#else
    return nullptr;
#endif
}

int8_kquant_ukernel_128_fn_t select_int8_kquant_ukernel_128_m1() {
#if defined(__x86_64__) || defined(__i386__)
    static const int8_kquant_ukernel_128_fn_t s_fn = [] {
        const char *no_xop = std::getenv("ZENDNNL_NATIVE_SYMQ_NO_XOP");
        const bool avoid_xop = no_xop != nullptr && no_xop[0] != '\0'
                && std::strcmp(no_xop, "0") != 0;
        if (!avoid_xop && kq_host_has_xop()) return &kq_xop::ukernel_m1;
        if (!kq_host_has_ssse3_sse41())
            return static_cast<int8_kquant_ukernel_128_fn_t>(nullptr);
        return &kq_sse::ukernel_m1;
    }();
    return s_fn;
#else
    return nullptr;
#endif
}

void int8_kquant_row_sums(const int8_t *__restrict__ A, int a_stride, int M,
        int K, int group_size, int32_t *__restrict__ out) {
    const int n_groups = K / group_size;
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < n_groups; ++g) {
            int32_t sum = 0;
            const int8_t *row = A + static_cast<size_t>(m) * a_stride
                    + static_cast<size_t>(g) * group_size;
            for (int j = 0; j < group_size; ++j) sum += row[j];
            out[static_cast<size_t>(m) * n_groups + g] = sum;
        }
    }
}

void int8_kquant_tail_128(const int8_t *__restrict__ A, int a_stride,
        const uint8_t *__restrict__ B_vnni, int b_stride, float *__restrict__ C,
        int ldc, int k, int group_size, int mr_act, int nr_act,
        const float *__restrict__ wei_scale, const float *__restrict__ wei_min,
        int ws_stride, const int32_t *__restrict__ row_sums, int rs_stride,
        const float *__restrict__ src_scale, int ss_row, int ss_grp) {

    const int quads_per_group = group_size / KQ_VNNI_GRP;
    const int n_groups = k / group_size;

    for (int m = 0; m < mr_act; ++m) {
        for (int n = 0; n < nr_act; ++n) {
            float sum = 0.0f;
            for (int g = 0; g < n_groups; ++g) {
                int32_t acc = 0;
                for (int q = 0; q < quads_per_group; ++q) {
                    const int kq = g * quads_per_group + q;
                    for (int j = 0; j < KQ_VNNI_GRP; ++j) {
                        const int32_t b = B_vnni[kq * b_stride
                                + n * KQ_VNNI_GRP + j];
                        const int32_t a
                                = A[m * a_stride + kq * KQ_VNNI_GRP + j];
                        acc += a * b;
                    }
                }
                const float ss = src_scale[m * ss_row + g * ss_grp];
                sum += ss
                        * (static_cast<float>(acc)
                                        * wei_scale[g * ws_stride + n]
                                - wei_min[g * ws_stride + n]
                                        * static_cast<float>(
                                                row_sums[m * rs_stride + g]));
            }
            C[m * ldc + n] += sum;
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
