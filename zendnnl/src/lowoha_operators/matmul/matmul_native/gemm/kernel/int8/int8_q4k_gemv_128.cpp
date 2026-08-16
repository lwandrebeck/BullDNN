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
            // Subnormal: normalise into a float exponent.
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
inline void get_scale_min_k4(
        int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One reduction per ROW, not per sub-block. The per-sub-block version of this
// was the kernel's largest single cost: eight shuffle/add latency chains per 256
// weights, against about thirty instructions of actual work.
inline float hsum_ps(__m128 v) {
    v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0x4E));
    v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0xB1));
    return _mm_cvtss_f32(v);
}

// One super-block: eight 32-wide dot products of the unsigned codes against the
// activations. The nibble pair is what makes this cheap -- one 16-byte load of
// qs serves sixteen weights of sub-block 2c through its low nibbles and sixteen
// of sub-block 2c+1 through its high ones, so the whole super-block's 256
// weights cost eight loads rather than the 256 bytes an unpacked form would.
//
// PMADDUBSW wants (unsigned, signed) in that order and the codes are already
// unsigned, so unlike the symmetric kernel there is no sign trick and no
// saturation to reason about: 15*127*2 = 3810 for Q4_K and 31*127*2 = 7874 for
// Q5_K, both far inside int16.
template <bool kIsQ5>
inline void superblock_dots(const uint8_t *qs, const uint8_t *qh,
        const int8_t *a, __m128i &d0123, __m128i &d4567) {
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i sixteen = _mm_set1_epi8(16);
    __m128i v[8];

    for (int c = 0; c < 4; ++c) {
        const uint8_t *q = qs + 32 * c;
        // Q5_K's fifth bit comes from a 32-byte qh shared by the super-block,
        // whose mask advances two positions per 64-weight chunk.
        const __m128i u1 = _mm_set1_epi8(static_cast<char>(1 << (2 * c)));
        const __m128i u2 = _mm_set1_epi8(static_cast<char>(2 << (2 * c)));

        __m128i acc_lo = _mm_setzero_si128();
        __m128i acc_hi = _mm_setzero_si128();
        for (int h = 0; h < 2; ++h) {
            const __m128i qb = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(q + 16 * h));
            __m128i lo = _mm_and_si128(qb, mask0f);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), mask0f);

            if (kIsQ5) {
                const __m128i hb = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(qh + 16 * h));
                // bit set -> add 16. cmpeq against the mask turns the selected
                // bit into a full-byte predicate.
                lo = _mm_add_epi8(lo,
                        _mm_and_si128(
                                _mm_cmpeq_epi8(_mm_and_si128(hb, u1), u1),
                                sixteen));
                hi = _mm_add_epi8(hi,
                        _mm_and_si128(
                                _mm_cmpeq_epi8(_mm_and_si128(hb, u2), u2),
                                sixteen));
            }

            const __m128i al = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(a + 64 * c + 16 * h));
            const __m128i ah = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(a + 64 * c + 32 + 16 * h));

            acc_lo = _mm_add_epi32(acc_lo,
                    _mm_madd_epi16(_mm_maddubs_epi16(lo, al), ones));
            acc_hi = _mm_add_epi32(acc_hi,
                    _mm_madd_epi16(_mm_maddubs_epi16(hi, ah), ones));
        }
        v[2 * c] = acc_lo;
        v[2 * c + 1] = acc_hi;
    }

    // Eight 4-lane accumulators to eight scalars in six instructions rather than
    // eight horizontal reductions. _mm_hadd_epi32(a, b) yields
    // [a0+a1, a2+a3, b0+b1, b2+b3], so a second pass over two such results
    // finishes four sub-blocks at once and leaves them already packed in the
    // order the scale flush wants them.
    d0123 = _mm_hadd_epi32(
            _mm_hadd_epi32(v[0], v[1]), _mm_hadd_epi32(v[2], v[3]));
    d4567 = _mm_hadd_epi32(
            _mm_hadd_epi32(v[4], v[5]), _mm_hadd_epi32(v[6], v[7]));
}

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

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int n = 0; n < N; ++n) {
        __m128 accv = _mm_setzero_ps();
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t *scales;
            const uint8_t *qs;
            const uint8_t *qh = nullptr;
            float d, dmin;

            if (is_q5) {
                const block_q5_K *b
                        = static_cast<const block_q5_K *>(blocks)
                        + static_cast<size_t>(n) * nsb + sb;
                d = fp16_to_fp32(b->d);
                dmin = fp16_to_fp32(b->dmin);
                scales = b->scales;
                qs = b->qs;
                qh = b->qh;
            } else {
                const block_q4_K *b
                        = static_cast<const block_q4_K *>(blocks)
                        + static_cast<size_t>(n) * nsb + sb;
                d = fp16_to_fp32(b->d);
                dmin = fp16_to_fp32(b->dmin);
                scales = b->scales;
                qs = b->qs;
            }

            const int8_t *a = A + static_cast<size_t>(sb) * Q4K_SUPER;
            __m128i d0123, d4567;
            if (is_q5) {
                superblock_dots<true>(qs, qh, a, d0123, d4567);
            } else {
                superblock_dots<false>(qs, nullptr, a, d0123, d4567);
            }

            // The six-bit scale and min pairs still come out one at a time --
            // get_scale_min_k4's stitching does not vectorise usefully -- but
            // they are only eight per 256 weights, and gathering them into
            // registers lets the rest of the flush run four sub-blocks wide.
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j)
                get_scale_min_k4(j, scales, &sc[j], &mn[j]);

            const int g0 = sb * 8;
            const __m128 vd = _mm_set1_ps(d);
            const __m128 vdmin = _mm_set1_ps(dmin);

            for (int half = 0; half < 2; ++half) {
                const int j0 = half * 4;
                const __m128 vdot = _mm_cvtepi32_ps(half ? d4567 : d0123);
                const __m128 vsc = _mm_setr_ps(sc[j0], sc[j0 + 1], sc[j0 + 2],
                        sc[j0 + 3]);
                const __m128 vmn = _mm_setr_ps(mn[j0], mn[j0 + 1], mn[j0 + 2],
                        mn[j0 + 3]);
                const __m128 vrs = _mm_cvtepi32_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(
                                rowsum.data() + g0 + j0)));
                const __m128 vss = ss_grp
                        ? _mm_loadu_ps(src_scale + g0 + j0)
                        : _mm_set1_ps(src_scale[0]);

                const __m128 term = _mm_sub_ps(
                        _mm_mul_ps(_mm_mul_ps(vd, vsc), vdot),
                        _mm_mul_ps(_mm_mul_ps(vdmin, vmn), vrs));
                accv = _mm_add_ps(accv, _mm_mul_ps(vss, term));
            }
        }
        C[n] = hsum_ps(accv);
    }

    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
