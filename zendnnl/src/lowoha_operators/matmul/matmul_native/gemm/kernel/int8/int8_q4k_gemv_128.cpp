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

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q4k_gemv_128.hpp"

#include <cpuid.h>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

// GGML's block layouts, byte for byte. Duplicated from the unpack rather than
// shared because this kernel's whole point is that it reads these and nothing
// else; a static_assert on each keeps the duplication honest.
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must match GGML");

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must match GGML");

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++e;
            }
            m &= 0x3FFu;
            bits = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// GGML's 6-bit scale/min unpacking: the first four pairs live in the low six
// bits of scales[0..7], the last four are stitched from the high bits.
inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One reduction per ROW. The per-sub-block version of this was the kernel's
// largest single cost -- eight shuffle/add latency chains per 256 weights
// against about thirty instructions of actual work -- and removing it roughly
// doubled the rate.
inline float hsum_ps(__m128 v) {
    v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0x4E));
    v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0xB1));
    return _mm_cvtss_f32(v);
}

// XOP, like FMA4, is an AMD extension that platform_info does not report -- that
// comes from AOCL-utils and is Zen-oriented, and no Zen part has either. CPUID
// Fn8000_0001_ECX bit 11 is the definitive answer. Family 15h is the only
// silicon that has it.
bool host_has_xop() {
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx)) return false;
    if (eax < 0x80000001u) return false;
    if (!__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx)) return false;
    return (ecx & (1u << 11)) != 0;
}

// ===========================================================================
// The compute, instantiated once per flavour. Exactly two operations differ, so
// they are the only things parameterised:
//
//   Q4K_HI_NIBBLE  portable: shift the 16-bit lanes right, then mask. The mask
//                  is needed only because _mm_srli_epi16 drags neighbouring
//                  bits across the byte boundary.
//                  XOP: VPSHLB shifts each BYTE by its own signed count, so a
//                  count of -4 yields the high nibble with no mask -- two
//                  instructions become one.
//   Q4K_ACCUM      portable: PMADDWD against a vector of ones, then PADDD.
//                  XOP: VPMADCSWD is exactly that pair, fused.
//
// Both sit on the 16-byte code load, of which there are eight per super-block,
// so the pair removes about two instructions per sixteen weights.
//
// Comments inside the macro must be the /* */ form: a // comment would be
// continued by the trailing backslash and swallow the following line.
// ===========================================================================
#define Q4K_GEMV_BODY                                                          \
    template <bool kIsQ5>                                                      \
    inline void superblock_dots(const uint8_t *qs, const uint8_t *qh,          \
            const int8_t *a, __m128i &d0123, __m128i &d4567) {                 \
        const __m128i mask0f = _mm_set1_epi8(0x0F);                            \
        const __m128i ones = _mm_set1_epi16(1);                                \
        const __m128i sixteen = _mm_set1_epi8(16);                             \
        __m128i v[8];                                                          \
        for (int c = 0; c < 4; ++c) {                                          \
            const uint8_t *q = qs + 32 * c;                                    \
            /* Q5_K's fifth bit is in a 32-byte qh shared by the super-block,   \
               whose mask advances two positions per 64-weight chunk. */       \
            const __m128i u1 = _mm_set1_epi8(static_cast<char>(1 << (2 * c))); \
            const __m128i u2 = _mm_set1_epi8(static_cast<char>(2 << (2 * c))); \
            __m128i acc_lo = _mm_setzero_si128();                              \
            __m128i acc_hi = _mm_setzero_si128();                              \
            for (int h = 0; h < 2; ++h) {                                      \
                const __m128i qb = _mm_loadu_si128(                            \
                        reinterpret_cast<const __m128i *>(q + 16 * h));        \
                __m128i lo = _mm_and_si128(qb, mask0f);                        \
                __m128i hi = Q4K_HI_NIBBLE(qb, mask0f);                        \
                if (kIsQ5) {                                                   \
                    const __m128i hb = _mm_loadu_si128(                        \
                            reinterpret_cast<const __m128i *>(qh + 16 * h));   \
                    lo = _mm_add_epi8(lo,                                      \
                            _mm_and_si128(                                     \
                                    _mm_cmpeq_epi8(                            \
                                            _mm_and_si128(hb, u1), u1),        \
                                    sixteen));                                 \
                    hi = _mm_add_epi8(hi,                                      \
                            _mm_and_si128(                                     \
                                    _mm_cmpeq_epi8(                            \
                                            _mm_and_si128(hb, u2), u2),        \
                                    sixteen));                                 \
                }                                                              \
                const __m128i al                                               \
                        = _mm_loadu_si128(reinterpret_cast<const __m128i *>(   \
                                a + 64 * c + 16 * h));                         \
                const __m128i ah                                               \
                        = _mm_loadu_si128(reinterpret_cast<const __m128i *>(   \
                                a + 64 * c + 32 + 16 * h));                    \
                acc_lo = Q4K_ACCUM(acc_lo, _mm_maddubs_epi16(lo, al), ones);   \
                acc_hi = Q4K_ACCUM(acc_hi, _mm_maddubs_epi16(hi, ah), ones);   \
            }                                                                  \
            v[2 * c] = acc_lo;                                                 \
            v[2 * c + 1] = acc_hi;                                             \
        }                                                                      \
        /* Eight 4-lane accumulators to eight scalars in six instructions.     \
           _mm_hadd_epi32(a, b) gives [a0+a1, a2+a3, b0+b1, b2+b3], so a       \
           second pass over two such results finishes four sub-blocks at once  \
           and leaves them packed in the order the flush wants. */             \
        d0123 = _mm_hadd_epi32(                                                \
                _mm_hadd_epi32(v[0], v[1]), _mm_hadd_epi32(v[2], v[3]));       \
        d4567 = _mm_hadd_epi32(                                                \
                _mm_hadd_epi32(v[4], v[5]), _mm_hadd_epi32(v[6], v[7]));       \
    }                                                                          \
                                                                               \
    inline void run(int N, int nsb, bool is_q5, const int8_t *A,               \
            const void *blocks, float *C, const float *src_scale, int ss_grp,  \
            const int32_t *rowsum, int nt) {                                   \
        _Pragma("omp parallel for schedule(static) num_threads(nt)")           \
        for (int n = 0; n < N; ++n) {                                          \
            __m128 accv = _mm_setzero_ps();                                    \
            for (int sb = 0; sb < nsb; ++sb) {                                 \
                const uint8_t *scales;                                         \
                const uint8_t *qs;                                             \
                const uint8_t *qh = nullptr;                                   \
                float d, dmin;                                                 \
                if (is_q5) {                                                   \
                    const block_q5_K *b                                        \
                            = static_cast<const block_q5_K *>(blocks)          \
                            + static_cast<size_t>(n) * nsb + sb;               \
                    d = fp16_to_fp32(b->d);                                    \
                    dmin = fp16_to_fp32(b->dmin);                              \
                    scales = b->scales;                                        \
                    qs = b->qs;                                                \
                    qh = b->qh;                                                \
                } else {                                                       \
                    const block_q4_K *b                                        \
                            = static_cast<const block_q4_K *>(blocks)          \
                            + static_cast<size_t>(n) * nsb + sb;               \
                    d = fp16_to_fp32(b->d);                                    \
                    dmin = fp16_to_fp32(b->dmin);                              \
                    scales = b->scales;                                        \
                    qs = b->qs;                                                \
                }                                                              \
                const int8_t *a = A + static_cast<size_t>(sb) * Q4K_SUPER;     \
                __m128i d0123, d4567;                                          \
                if (is_q5) {                                                   \
                    superblock_dots<true>(qs, qh, a, d0123, d4567);            \
                } else {                                                       \
                    superblock_dots<false>(qs, nullptr, a, d0123, d4567);      \
                }                                                              \
                /* The six-bit pairs still come out one at a time -- the       \
                   stitching does not vectorise usefully -- but there are only \
                   eight per 256 weights, and gathering them into registers    \
                   lets the rest of the flush run four sub-blocks wide. */     \
                uint8_t sc[8], mn[8];                                          \
                for (int j = 0; j < 8; ++j)                                    \
                    get_scale_min_k4(j, scales, &sc[j], &mn[j]);               \
                const int g0 = sb * 8;                                         \
                const __m128 vd = _mm_set1_ps(d);                              \
                const __m128 vdmin = _mm_set1_ps(dmin);                        \
                for (int half = 0; half < 2; ++half) {                         \
                    const int j0 = half * 4;                                   \
                    const __m128 vdot                                          \
                            = _mm_cvtepi32_ps(half ? d4567 : d0123);           \
                    const __m128 vsc = _mm_setr_ps(                            \
                            sc[j0], sc[j0 + 1], sc[j0 + 2], sc[j0 + 3]);       \
                    const __m128 vmn = _mm_setr_ps(                            \
                            mn[j0], mn[j0 + 1], mn[j0 + 2], mn[j0 + 3]);       \
                    const __m128 vrs = _mm_cvtepi32_ps(                        \
                            _mm_loadu_si128(                                   \
                                    reinterpret_cast<const __m128i *>(         \
                                            rowsum + g0 + j0)));               \
                    const __m128 vss = ss_grp                                  \
                            ? _mm_loadu_ps(src_scale + g0 + j0)                \
                            : _mm_set1_ps(src_scale[0]);                       \
                    const __m128 term = _mm_sub_ps(                            \
                            _mm_mul_ps(_mm_mul_ps(vd, vsc), vdot),             \
                            _mm_mul_ps(_mm_mul_ps(vdmin, vmn), vrs));          \
                    accv = _mm_add_ps(accv, _mm_mul_ps(vss, term));            \
                }                                                              \
            }                                                                  \
            C[n] = hsum_ps(accv);                                              \
        }                                                                      \
    }

// ---- portable: SSSE3, every family 15h part and anything newer -------------
namespace gemv_sse {
#define Q4K_HI_NIBBLE(qb, mask) _mm_and_si128(_mm_srli_epi16((qb), 4), (mask))
#define Q4K_ACCUM(acc, words, ones) \
    _mm_add_epi32((acc), _mm_madd_epi16((words), (ones)))
Q4K_GEMV_BODY
#undef Q4K_HI_NIBBLE
#undef Q4K_ACCUM
} // namespace gemv_sse

// ---- XOP: family 15h only ---------------------------------------------------
#pragma GCC push_options
#pragma GCC target("xop,ssse3,sse4.1")
namespace gemv_xop {
// The intrinsics are declared behind #ifdef __XOP__ in xopintrin.h, which was
// decided when the headers above were included -- before this pragma -- so they
// are not in scope even though the instructions are available here. The builtins
// carry no such guard.
#define Q4K_HI_NIBBLE(qb, mask)                                                \
    ((__m128i) __builtin_ia32_vpshlb(                                          \
            (__v16qi)(qb), (__v16qi) _mm_set1_epi8(-4)))
#define Q4K_ACCUM(acc, words, ones)                                            \
    ((__m128i) __builtin_ia32_vpmadcswd(                                       \
            (__v8hi)(words), (__v8hi)(ones), (__v4si)(acc)))
Q4K_GEMV_BODY
#undef Q4K_HI_NIBBLE
#undef Q4K_ACCUM
} // namespace gemv_xop
#pragma GCC pop_options

} // namespace

bool int8_q4k_gemv_supported(int M, int N, int K, int ggml_type) {
    if (M != 1) return false;
    if (ggml_type != 12 && ggml_type != 13) return false;
    if (N <= 0 || K <= 0) return false;
    return K % Q4K_SUPER == 0;
}

bool int8_q4k_gemv_128(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads) {

    if (!int8_q4k_gemv_supported(1, N, K, ggml_type)) return false;
    if (A == nullptr || blocks == nullptr || C == nullptr
            || src_scale == nullptr)
        return false;

    const int nsb = K / Q4K_SUPER; // super-blocks per weight row
    const int n_groups = K / Q4K_SUB;
    const bool is_q5 = (ggml_type == 13);

    // The min term needs sum(a) per 32-wide group. It does not depend on the
    // weight column, so it is built once for the whole call rather than per row:
    // K additions against N*K multiply-adds.
    std::vector<int32_t> rowsum(static_cast<size_t>(n_groups), 0);
    for (int g = 0; g < n_groups; ++g) {
        const int8_t *a = A + static_cast<size_t>(g) * Q4K_SUB;
        int32_t s = 0;
        for (int j = 0; j < Q4K_SUB; ++j) s += a[j];
        rowsum[g] = s;
    }

    const int nt = nthreads > 0 ? nthreads : 1;

    // Chosen once per call, not per super-block: an indirect call for each of
    // the N*nsb blocks would cost more than the two instructions XOP saves.
    // ZENDNNL_NATIVE_Q4K_NO_XOP forces the portable flavour so that the two can
    // be compared on one machine, which is the only way to attribute a
    // difference to the instruction selection rather than to the day.
    static const bool s_use_xop = [] {
        const char *no_xop = std::getenv("ZENDNNL_NATIVE_Q4K_NO_XOP");
        const bool avoid = no_xop != nullptr && no_xop[0] != '\0'
                && std::strcmp(no_xop, "0") != 0;
        return !avoid && host_has_xop();
    }();

    if (s_use_xop) {
        gemv_xop::run(N, nsb, is_q5, A, blocks, C, src_scale, ss_grp,
                rowsum.data(), nt);
    } else {
        gemv_sse::run(N, nsb, is_q5, A, blocks, C, src_scale, ss_grp,
                rowsum.data(), nt);
    }

    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
