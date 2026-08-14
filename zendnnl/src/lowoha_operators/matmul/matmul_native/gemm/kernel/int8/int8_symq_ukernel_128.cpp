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
// Per-group symmetric INT8 microkernel at 128 bits.
//
// Per VNNI quad (four K) and eight columns:
//
//   2 loads    B, sixteen bytes each holding four columns x four K
//   4 bcast    A, one dword per row
//   4 abs      |A| per row, hoisted out of the column loop
//   8 sign     apply A's sign to B, per row and column half
//   8 maddubs  byte products summed in pairs to words
//   8 reduce   words to dwords, accumulated
//
// The reduction is where the two flavours differ. Without XOP it is PMADDWD
// against a vector of ones followed by PADDD. XOP's VPMADCSWD does both in one
// instruction, so the flavour exists purely to delete eight PADDDs per quad.
// Family 15h is the only silicon that has XOP -- no Zen or Intel part does --
// so this is one of the few places where the ISA the target actually has buys
// something its successors cannot.
//
// PRECONDITION: every A and B byte must lie in [-127, 127]. -128 is excluded,
// and not as a formality -- it breaks the kernel two ways. PSIGNB negating -128
// wraps to -128 rather than +128, so the sign trick yields the wrong sign; and
// a PMADDUBSW lane summing two products of 128 * 128 reaches 32768, one past
// the signed 16-bit limit, so it saturates. Both failures are silent: the
// answer simply comes back wrong, which is why the unit test drives the
// extremes rather than trusting the reasoning.
//
// Within the contract the arithmetic is exact: a lane sums two products of
// |a|, |b| <= 127, so 32258 at worst against a limit of 32767. Every GGML
// quantiser satisfies this -- Q8_0 clamps to +/-127, Q4_0 nibbles map to
// [-8, 7], Q6_K codes to [-32, 31] -- which is the same guarantee llama.cpp's
// own kernels rest on.
//
// This holds only because both operands are symmetric s8. The asymmetric
// u8 * s8 form the AVX-512 kernel next door uses can reach 64770 in a lane, so
// it leans on VNNI's dword accumulate instead of building the same thing here.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"

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

// SSSE3 (CPUID.1:ECX[9]) and SSE4.1 (ECX[19]) -- the floor for this kernel.
bool host_has_ssse3_sse41() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 9)) != 0 && (ecx & (1u << 19)) != 0;
}

// XOP, like FMA4, is an AMD extension that platform_info does not report --
// that comes from AOCL-utils and is Zen-oriented, and no Zen part has either.
// CPUID Fn8000_0001_ECX bit 11 is the definitive answer.
bool host_has_xop() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx)) return false;
    if (eax < 0x80000001u) return false;
    if (!__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 11)) != 0;
}

// The body is shared between the flavours; only reduce_accum() differs.
#define SYMQ_UK128_BODY                                                        \
    void ukernel(const int8_t *__restrict__ A, int a_stride,                   \
            const int8_t *__restrict__ B_vnni, int b_stride,                   \
            float *__restrict__ C, int ldc, int k, int group_size,             \
            const float *__restrict__ wei_scale, int ws_stride,                \
            float src_scale) {                                                 \
        const int quads_per_group = group_size / SYMQ_VNNI_GRP;                \
        const int n_groups = k / group_size;                                   \
        const __m128 v_src = _mm_set1_ps(src_scale);                           \
                                                                               \
        for (int g = 0; g < n_groups; ++g) {                                   \
            /* s32 accumulators: four rows x two halves of four columns. */    \
            __m128i acc[SYMQ_MR][2];                                           \
            for (int m = 0; m < SYMQ_MR; ++m) {                                \
                acc[m][0] = _mm_setzero_si128();                               \
                acc[m][1] = _mm_setzero_si128();                               \
            }                                                                  \
                                                                               \
            for (int q = 0; q < quads_per_group; ++q) {                        \
                const int kq = g * quads_per_group + q;                        \
                const int8_t *bq = B_vnni + kq * b_stride;                     \
                const __m128i b0 = _mm_loadu_si128(                            \
                        reinterpret_cast<const __m128i *>(bq));                \
                const __m128i b1 = _mm_loadu_si128(                            \
                        reinterpret_cast<const __m128i *>(bq + 16));           \
                                                                               \
                for (int m = 0; m < SYMQ_MR; ++m) {                            \
                    int32_t a_quad;                                            \
                    std::memcpy(&a_quad,                                       \
                            A + m * a_stride + kq * SYMQ_VNNI_GRP, 4);         \
                    const __m128i av = _mm_set1_epi32(a_quad);                 \
                    const __m128i a_abs = _mm_abs_epi8(av);                    \
                    acc[m][0] = reduce_accum(                                  \
                            _mm_maddubs_epi16(a_abs, _mm_sign_epi8(b0, av)),   \
                            acc[m][0]);                                        \
                    acc[m][1] = reduce_accum(                                  \
                            _mm_maddubs_epi16(a_abs, _mm_sign_epi8(b1, av)),   \
                            acc[m][1]);                                        \
                }                                                              \
            }                                                                  \
                                                                               \
            /* Flush: one weight scale per column for this group. */           \
            const __m128 ws0 = _mm_loadu_ps(wei_scale + g * ws_stride);        \
            const __m128 ws1 = _mm_loadu_ps(wei_scale + g * ws_stride + 4);    \
            const __m128 s0 = _mm_mul_ps(ws0, v_src);                          \
            const __m128 s1 = _mm_mul_ps(ws1, v_src);                          \
            for (int m = 0; m < SYMQ_MR; ++m) {                                \
                float *c = C + m * ldc;                                        \
                _mm_storeu_ps(c,                                               \
                        _mm_add_ps(_mm_loadu_ps(c),                            \
                                _mm_mul_ps(_mm_cvtepi32_ps(acc[m][0]), s0)));  \
                _mm_storeu_ps(c + 4,                                           \
                        _mm_add_ps(_mm_loadu_ps(c + 4),                        \
                                _mm_mul_ps(_mm_cvtepi32_ps(acc[m][1]), s1)));  \
            }                                                                  \
        }                                                                      \
    }

// ---- SSSE3/SSE4.1: every family 15h part, and everything since -------------
#pragma GCC push_options
#pragma GCC target("ssse3,sse4.1")
namespace uk_sse {
// PMADDWD against ones turns four words into two dwords pairwise, which is the
// second half of the reduction; PADDD then accumulates.
inline __m128i reduce_accum(__m128i words, __m128i acc) {
    return _mm_add_epi32(acc, _mm_madd_epi16(words, _mm_set1_epi16(1)));
}
SYMQ_UK128_BODY
} // namespace uk_sse
#pragma GCC pop_options

// ---- XOP: family 15h only --------------------------------------------------
#pragma GCC push_options
#pragma GCC target("xop,ssse3,sse4.1")
namespace uk_xop {
// VPMADCSWD is exactly the pair above fused: dst = acc + (w[2i] * o[2i] +
// w[2i+1] * o[2i+1]) per dword. With o = 1 that is the same reduction in one
// instruction instead of two.
inline __m128i reduce_accum(__m128i words, __m128i acc) {
    // _mm_maddd_epi16 is declared behind #ifdef __XOP__ in xopintrin.h, which
    // was decided when immintrin.h was included -- before this pragma, so the
    // intrinsic is not in scope here even though the instruction is available.
    // The builtin carries no such guard.
    return (__m128i)__builtin_ia32_vpmadcswd(
            (__v8hi)words, (__v8hi)_mm_set1_epi16(1), (__v4si)acc);
}
SYMQ_UK128_BODY
} // namespace uk_xop
#pragma GCC pop_options

#endif // x86

} // namespace

int8_symq_ukernel_128_fn_t select_int8_symq_ukernel_128() {
#if defined(__x86_64__) || defined(__i386__)
    static const int8_symq_ukernel_128_fn_t s_fn = [] {
        // ZENDNNL_NATIVE_SYMQ_NO_XOP forces the portable flavour, so the two
        // can be compared on one machine -- the XOP build is otherwise
        // unreachable anywhere except family 15h.
        const char *no_xop = std::getenv("ZENDNNL_NATIVE_SYMQ_NO_XOP");
        const bool avoid_xop = no_xop != nullptr && no_xop[0] != '\0'
                && std::strcmp(no_xop, "0") != 0;
        if (!avoid_xop && host_has_xop()) return &uk_xop::ukernel;
        if (!host_has_ssse3_sse41())
            return static_cast<int8_symq_ukernel_128_fn_t>(nullptr);
        return &uk_sse::ukernel;
    }();
    return s_fn;
#else
    return nullptr;
#endif
}

void int8_symq_tail_128(const int8_t *__restrict__ A, int a_stride,
        const int8_t *__restrict__ B_vnni, int b_stride, float *__restrict__ C,
        int ldc, int k, int group_size, int mr_act, int nr_act,
        const float *__restrict__ wei_scale, int ws_stride, float src_scale) {

    const int quads_per_group = group_size / SYMQ_VNNI_GRP;
    const int n_groups = k / group_size;

    for (int m = 0; m < mr_act; ++m) {
        for (int n = 0; n < nr_act; ++n) {
            float sum = 0.0f;
            for (int g = 0; g < n_groups; ++g) {
                int32_t acc = 0;
                for (int q = 0; q < quads_per_group; ++q) {
                    const int kq = g * quads_per_group + q;
                    for (int j = 0; j < SYMQ_VNNI_GRP; ++j) {
                        // VNNI addressing: column n's quad starts at
                        // n * SYMQ_VNNI_GRP within the k-group row.
                        const int8_t b
                                = B_vnni[kq * b_stride + n * SYMQ_VNNI_GRP + j];
                        const int8_t a
                                = A[m * a_stride + kq * SYMQ_VNNI_GRP + j];
                        acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
                    }
                }
                sum += static_cast<float>(acc) * wei_scale[g * ws_stride + n]
                        * src_scale;
            }
            C[m * ldc + n] += sum;
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
