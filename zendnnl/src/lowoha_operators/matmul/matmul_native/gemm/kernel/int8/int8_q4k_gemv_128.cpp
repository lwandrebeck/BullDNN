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

// Q6_K is the odd one out. Six-bit codes split across two planes, sub-blocks of
// SIXTEEN rather than 32, an 8-bit SIGNED scale per sub-block, and no min term
// at all: it is symmetric about 32, so w = d * sc * (q - 32). That -32 folds
// into a row sum, which is the same shape of correction Q4_K's min needs, so
// the two share the idea if not the code.
struct block_q6_K {
    uint8_t ql[128]; // low 4 bits
    uint8_t qh[64];  // high 2 bits
    int8_t scales[16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must match GGML");

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
        /* Four accumulators live, not eight. Holding all eight until the end   \
           spilled: eight of them plus the three constants plus u1/u2 plus the  \
           four in-flight temps is past sixteen XMM registers, and the object   \
           carried eleven spill stores in this loop. Reducing each half as soon \
           as its two chunks are done costs nothing -- the hadds happen either  \
           way -- and halves what has to stay live. */                          \
        __m128i v0, v1, v2, v3;                                                \
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
            if ((c & 1) == 0) {                                                \
                v0 = acc_lo;                                                   \
                v1 = acc_hi;                                                   \
            } else {                                                           \
                v2 = acc_lo;                                                   \
                v3 = acc_hi;                                                   \
                /* _mm_hadd_epi32(a, b) gives [a0+a1, a2+a3, b0+b1, b2+b3], so \
                   a second pass over two such results finishes four sub-blocks \
                   at once, packed in the order the flush wants. */            \
                const __m128i d = _mm_hadd_epi32(                               \
                        _mm_hadd_epi32(v0, v1), _mm_hadd_epi32(v2, v3));       \
                if (c == 1) {                                                  \
                    d0123 = d;                                                 \
                } else {                                                       \
                    d4567 = d;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
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

// ===========================================================================
// Q6_K decode, portable SSSE3 only for now. Kept out of the two-flavour macro
// above deliberately: XOP's saving there is VPSHLB on the high nibble and a
// fused accumulate, and until this kernel is shown to be worth having at all,
// carrying it in two flavours is two things to keep correct rather than one.
//
// A 128-weight chunk interleaves FOUR streams, which is what makes the layout
// awkward:
//
//   weights   0..31   ql[l] low nibble   + qh[l] bits 0-1   scale sc[is+0]
//   weights  32..63   ql[l+32] low       + qh[l] bits 2-3   scale sc[is+2]
//   weights  64..95   ql[l] high nibble  + qh[l] bits 4-5   scale sc[is+4]
//   weights  96..127  ql[l+32] high      + qh[l] bits 6-7   scale sc[is+6]
//
// with is = l/16. So one 16-byte step is exactly one sub-block of each stream
// and needs no further splitting -- the sub-block IS the vector, which is why
// there are sixteen reductions per super-block here against Q4_K's eight.
//
// Codes reach 63 and stay unsigned for PMADDUBSW: 63*127*2 = 16002, inside
// int16. The -32 never enters the vector arithmetic; it comes out in the flush
// as -32 * rowsum, exactly like Q4_K's min term.
inline void superblock_dots_q6(
        const uint8_t *ql, const uint8_t *qh, const int8_t *a, __m128i *dots) {
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m128i mask03 = _mm_set1_epi8(0x03);
    const __m128i ones = _mm_set1_epi16(1);
    __m128i acc[16];

    for (int c = 0; c < 2; ++c) {
        const uint8_t *QL = ql + 64 * c;
        const uint8_t *QH = qh + 32 * c;
        const int8_t *AA = a + 128 * c;
        for (int l = 0; l < 32; l += 16) {
            const __m128i q_lo = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(QL + l));
            const __m128i q_hi = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(QL + l + 32));
            const __m128i h = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(QH + l));

            // The two high bits of each code, moved into position 4-5. The
            // shifts are 16-bit lane shifts, but masking to 0x03 first and to
            // the nibble after keeps every bit inside its own byte.
            const __m128i w0 = _mm_or_si128(_mm_and_si128(q_lo, mask0f),
                    _mm_slli_epi16(_mm_and_si128(h, mask03), 4));
            const __m128i w1 = _mm_or_si128(_mm_and_si128(q_hi, mask0f),
                    _mm_slli_epi16(
                            _mm_and_si128(_mm_srli_epi16(h, 2), mask03), 4));
            const __m128i w2 = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(q_lo, 4), mask0f),
                    _mm_slli_epi16(
                            _mm_and_si128(_mm_srli_epi16(h, 4), mask03), 4));
            const __m128i w3 = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(q_hi, 4), mask0f),
                    _mm_slli_epi16(
                            _mm_and_si128(_mm_srli_epi16(h, 6), mask03), 4));

            const int base = c * 8 + (l / 16);
            const __m128i a0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(AA + l));
            const __m128i a1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(AA + l + 32));
            const __m128i a2 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(AA + l + 64));
            const __m128i a3 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(AA + l + 96));

            acc[base + 0] = _mm_madd_epi16(_mm_maddubs_epi16(w0, a0), ones);
            acc[base + 2] = _mm_madd_epi16(_mm_maddubs_epi16(w1, a1), ones);
            acc[base + 4] = _mm_madd_epi16(_mm_maddubs_epi16(w2, a2), ones);
            acc[base + 6] = _mm_madd_epi16(_mm_maddubs_epi16(w3, a3), ones);
        }
    }

    // Sixteen 4-lane accumulators to sixteen scalars, already in scale order.
    for (int g = 0; g < 4; ++g) {
        dots[g] = _mm_hadd_epi32(
                _mm_hadd_epi32(acc[4 * g + 0], acc[4 * g + 1]),
                _mm_hadd_epi32(acc[4 * g + 2], acc[4 * g + 3]));
    }
}

void run_q6(int N, int nsb, const int8_t *A, const void *blocks, float *C,
        const float *src_scale, int ss_grp, const int32_t *rowsum16, int nt) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int n = 0; n < N; ++n) {
        __m128 accv = _mm_setzero_ps();
        for (int sb = 0; sb < nsb; ++sb) {
            const block_q6_K *b = static_cast<const block_q6_K *>(blocks)
                    + static_cast<size_t>(n) * nsb + sb;
            const float d = fp16_to_fp32(b->d);
            const int8_t *a = A + static_cast<size_t>(sb) * Q4K_SUPER;

            __m128i dots[4];
            superblock_dots_q6(b->ql, b->qh, a, dots);

            const int g0 = sb * 16; // sixteen sub-blocks per super-block
            const __m128 vd = _mm_set1_ps(d);
            const __m128 v32 = _mm_set1_ps(32.0f);
            for (int g = 0; g < 4; ++g) {
                const int j0 = g * 4;
                const __m128 vdot = _mm_cvtepi32_ps(dots[g]);
                const __m128 vsc = _mm_setr_ps(b->scales[j0], b->scales[j0 + 1],
                        b->scales[j0 + 2], b->scales[j0 + 3]);
                const __m128 vrs = _mm_cvtepi32_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(rowsum16 + g0 + j0)));
                // The ACTIVATION scales are per 32 elements -- the backend
                // sets src_scale.dims = {n, k/QK8_0} -- while a Q6_K sub-block
                // is SIXTEEN wide, so each pair of sub-blocks shares one. This
                // is where the first version was wrong: it indexed one scale
                // per sub-block, walked off the end of the buffer, and produced
                // NaN in llama-perplexity while the unit test passed, because
                // the test had been written to the kernel's convention rather
                // than the caller's.
                const int k0 = (g0 + j0) / 2;
                const __m128 vss = ss_grp
                        ? _mm_setr_ps(src_scale[k0], src_scale[k0],
                                src_scale[k0 + 1], src_scale[k0 + 1])
                        : _mm_set1_ps(src_scale[0]);
                // d * sc * (dot - 32 * rowsum)
                const __m128 term = _mm_mul_ps(_mm_mul_ps(vd, vsc),
                        _mm_sub_ps(vdot, _mm_mul_ps(v32, vrs)));
                accv = _mm_add_ps(accv, _mm_mul_ps(vss, term));
            }
        }
        C[n] = hsum_ps(accv);
    }
}

} // namespace

bool int8_q4k_gemv_supported(int M, int N, int K, int ggml_type) {
    if (M != 1) return false;
    // 12 Q4_K, 13 Q5_K, 14 Q6_K. Q6_K matters more than its share of tensors
    // suggests: a "Q4_K_M" model puts Q6_K on the output/lm_head and some
    // attention tensors, and that was measured at a THIRD of decode time.
    if (ggml_type != 12 && ggml_type != 13 && ggml_type != 14) return false;
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
    const bool is_q5 = (ggml_type == 13);
    const bool is_q6 = (ggml_type == 14);
    // Q6_K scales one sub-block of SIXTEEN; the others one of 32. The row sums
    // and the activation scales both follow that granularity.
    const int sub = is_q6 ? 16 : Q4K_SUB;
    const int n_groups = K / sub;

    // The min term needs sum(a) per 32-wide group. It does not depend on the
    // weight column, so it is built once for the whole call rather than per row:
    // K additions against N*K multiply-adds.
    std::vector<int32_t> rowsum(static_cast<size_t>(n_groups), 0);
    for (int g = 0; g < n_groups; ++g) {
        const int8_t *a = A + static_cast<size_t>(g) * sub;
        int32_t s = 0;
        for (int j = 0; j < sub; ++j) s += a[j];
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

    if (is_q6) {
        // Portable only for now; see the note above run_q6.
        run_q6(N, nsb, A, blocks, C, src_scale, ss_grp, rowsum.data(), nt);
    } else if (s_use_xop) {
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
