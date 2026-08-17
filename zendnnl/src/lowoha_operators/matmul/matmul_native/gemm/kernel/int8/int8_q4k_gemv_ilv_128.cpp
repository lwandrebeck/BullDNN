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
// Q4_K decode GEMV over a four-row interleaved layout.
//
// THE LANE ALGEBRA, which is the whole design.
//
// PMADDUBSW takes u8 against s8 and sums ADJACENT PAIRS into s16: eight s16
// lanes out of sixteen bytes. PMADDWD then multiplies s16 by s16 and again sums
// adjacent pairs, giving four s32. So a 128-bit chain collapses sixteen input
// bytes to four s32 lanes, and the question is only which four things those
// lanes should be. Make them FOUR OUTPUT ROWS and no horizontal reduction is
// ever needed.
//
// That fixes the operand order. For the s32 lane i to be row i, the sixteen
// weight bytes must be
//
//   W = [r0k0 r0k1 r0k2 r0k3 | r1k0 .. r1k3 | r2k0 .. r2k3 | r3k0 .. r3k3]
//
// and the activation must be the same four k broadcast four times, which is one
// instruction: the four activations are four contiguous bytes, so a 32-bit
// broadcast does it.
//
//   A = [k0 k1 k2 k3] x4                        _mm_set1_epi32(load dword)
//
// Then PMADDUBSW gives [r0(k0k1) r0(k2k3) r1(k0k1) r1(k2k3) ...] and PMADDWD
// against a scale vector duplicated per row, [s0 s0 s1 s1 s2 s2 s3 s3], both
// finishes the row dot AND applies the six-bit sub-block scale -- the same
// double duty ggml gets, and the reason the fp32 epilogue shrinks to almost
// nothing.
//
// THE PACKED FORM follows from W. Store eight bytes per four-k group, byte j
// holding W[2j] low and W[2j+1] high. Then
//
//   lo = AND(B, 0x0F)          [W0 W2 W4 .. W14]
//   hi = AND(SRL(B, 4), 0x0F)  [W1 W3 W5 .. W15]
//   W  = UNPACKLO8(lo, hi)     [W0 W1 W2 .. W15]
//
// This layout costs MORE per multiply in the inner loop than the row-at-a-time
// kernel, which gets a whole sixteen-weight operand from one 16-byte load and a
// single AND. It has to earn that back by deleting six VPHADDD and an hsum_ps
// per super-block per row and shrinking the fp32 epilogue from about a hundred
// instructions to a dozen.
//
// It does not earn it back if the unpack is done per operand: the first cut
// loaded eight bytes at a time and measured slower than what it replaced. Both
// operands must come out of ONE 16-byte load, via unpacklo and unpackhi of the
// same lo/hi pair, so the load and the masking amortise over 32 weights. Op
// counts per weight-multiply: about 0.68 for the row-at-a-time kernel, 0.63 for
// the eight-byte form -- a wash, and measured as one -- and about 0.41 here.
//
// ACCUMULATION DEPTH. A Q4_K code is 0..15 and the activation contract excludes
// -128, so one PMADDUBSW lane is at most 15*127*2 = 3810 and eight accumulate to
// 30480, inside the s16 limit of 32767. Eight four-k groups is 32 k, which is
// exactly one sub-block, which is exactly the span one scale covers. The
// arithmetic and the format agree, so the widening lands on a sub-block boundary
// with no partial state.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q4k_gemv_ilv_128.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {
namespace {

constexpr int kSuper = 256; // weights per super-block
constexpr int kSub = 32;    // weights per sub-block, one scale each
constexpr int kSubs = kSuper / kSub;
constexpr int kRows = 4; // rows interleaved, one per s32 lane
constexpr int kQ4K = 12; // GGML_TYPE_Q4_K

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must match GGML");

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 0, m = man;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++e;
            }
            bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
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

void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// The repacked weight.
//
// One entry per (weight, N, K). Laid out row-group major so a row group's whole
// stream is contiguous: for four rows, every super-block, every four-k group.
// ---------------------------------------------------------------------------
struct Repacked {
    // kRows * K / 2 bytes: the interleaved nibbles.
    std::vector<uint8_t> qs;
    // Per row group, per super-block, per sub-block, per row: the six-bit
    // scale and min, as s16 ready for PMADDWD and for the float term.
    std::vector<int16_t> sc;
    std::vector<int16_t> mn;
    // Per row group, per super-block, per row.
    std::vector<float> d;
    std::vector<float> dmin;
    size_t bytes() const {
        return qs.size() + sc.size() * 2 + mn.size() * 2 + d.size() * 4
                + dmin.size() * 4;
    }
};

// Build the interleaved form from GGML's row-major blocks.
//
// Cost is O(N*K) and paid once per weight, against O(N*K) per TOKEN in the
// kernel, so it is amortised over the whole generation. It is not free the first
// time, which is why the first rep of every arm is the slow one.
std::unique_ptr<Repacked> repack(int N, int K, const void *blocks) {
    const int nsb = K / kSuper;
    const int ngrp = N / kRows;
    auto r = std::make_unique<Repacked>();
    r->qs.resize(static_cast<size_t>(N) * K / 2);
    r->sc.resize(static_cast<size_t>(N) * nsb * kSubs);
    r->mn.resize(static_cast<size_t>(N) * nsb * kSubs);
    r->d.resize(static_cast<size_t>(N) * nsb);
    r->dmin.resize(static_cast<size_t>(N) * nsb);

    const auto *src = static_cast<const block_q4_K *>(blocks);

    for (int g = 0; g < ngrp; ++g) {
        for (int sb = 0; sb < nsb; ++sb) {
            // Scalar six-bit unpack, once per weight rather than once per token.
            for (int rr = 0; rr < kRows; ++rr) {
                const block_q4_K &b
                        = src[static_cast<size_t>(g * kRows + rr) * nsb + sb];
                const size_t base
                        = ((static_cast<size_t>(g) * nsb + sb) * kSubs) * kRows
                        + rr;
                for (int j = 0; j < kSubs; ++j) {
                    uint8_t sc, mn;
                    get_scale_min_k4(j, b.scales, &sc, &mn);
                    r->sc[base + static_cast<size_t>(j) * kRows] = sc;
                    r->mn[base + static_cast<size_t>(j) * kRows] = mn;
                }
                const size_t dbase
                        = (static_cast<size_t>(g) * nsb + sb) * kRows + rr;
                r->d[dbase] = fp16_to_fp32(b.d);
                r->dmin[dbase] = fp16_to_fp32(b.dmin);
            }

            // Nibbles. GGML's Q4_K packing is per 64-WEIGHT CHUNK, not per
            // 128-weight half: inside chunk c the low nibbles of bytes
            // 32c..32c+31 are k = 64c..64c+31 and the high nibbles of the SAME
            // bytes are k = 64c+32..64c+63. Read the row-at-a-time kernel's
            // chunk loop against its activation offsets if this ever looks
            // arbitrary -- it derives from there, and getting it wrong is
            // silently wrong rather than a crash.
            uint8_t *dst = r->qs.data()
                    + (static_cast<size_t>(g) * nsb + sb) * (kRows * kSuper / 2);
            for (int kq = 0; kq < kSuper / 4; ++kq) {
                for (int rr = 0; rr < kRows; ++rr) {
                    const block_q4_K &b
                            = src[static_cast<size_t>(g * kRows + rr) * nsb
                                    + sb];
                    uint8_t w[4];
                    for (int t = 0; t < 4; ++t) {
                        const int k = kq * 4 + t;
                        const int c = k / 64;
                        const int rem = k % 64;
                        const int byte = 32 * c + (rem % 32);
                        w[t] = (rem < 32) ? (b.qs[byte] & 0x0F)
                                          : (b.qs[byte] >> 4);
                    }
                    // byte j holds W[2j] low, W[2j+1] high; W index = rr*4 + t.
                    uint8_t *o = dst + static_cast<size_t>(kq) * 8 + rr * 2;
                    o[0] = static_cast<uint8_t>(w[0] | (w[1] << 4));
                    o[1] = static_cast<uint8_t>(w[2] | (w[3] << 4));
                }
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Cache. Same shape as the symq weight-facts cache: a mutex, a map, a bound.
//
// Keyed on the caller's pointer plus the shape, which is what every weight cache
// here does and is only safe because a GGML weight is loaded once and never
// moves. A reallocated buffer landing on the same address would serve stale
// bytes; nothing in this path may key a SAFETY decision this way.
// ---------------------------------------------------------------------------
struct Key {
    const void *w;
    int N, K;
    bool operator==(const Key &o) const {
        return w == o.w && N == o.N && K == o.K;
    }
};
struct KeyHash {
    size_t operator()(const Key &k) const {
        size_t h = std::hash<const void *>()(k.w);
        h ^= static_cast<size_t>(k.N) * 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.K) * 0x85ebca6bu + (h << 6) + (h >> 2);
        return h;
    }
};

std::mutex &cache_mutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<Key, std::unique_ptr<Repacked>, KeyHash> &cache_map() {
    static std::unordered_map<Key, std::unique_ptr<Repacked>, KeyHash> m;
    return m;
}

size_t cache_budget() {
    static const size_t b = [] {
        const char *e = std::getenv("ZENDNNL_Q4K_ILV_CACHE_MB");
        const long mb = (e != nullptr) ? std::atol(e) : 0;
        return static_cast<size_t>((mb > 0 ? mb : 4096)) * 1024u * 1024u;
    }();
    return b;
}

// Returns nullptr when the budget is spent, and the caller then declines rather
// than repacking per call -- paying an O(N*K) repack every token would be far
// worse than the kernel this is trying to beat.
const Repacked *get_or_repack(const Key &key, const void *blocks) {
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto &m = cache_map();
    auto it = m.find(key);
    if (it != m.end()) return it->second.get();

    size_t resident = 0;
    for (const auto &kv : m) resident += kv.second->bytes();
    auto built = repack(key.N, key.K, blocks);
    if (resident + built->bytes() > cache_budget()) return nullptr;

    const Repacked *raw = built.get();
    m.emplace(key, std::move(built));
    return raw;
}

// ---------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------
void run_group(const Repacked &r, int g, int nsb, const int8_t *A,
        const int32_t *rowsum, const float *src_scale, int ss_grp, float *C) {

    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128 out = _mm_setzero_ps();

    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t *qs = r.qs.data()
                + (static_cast<size_t>(g) * nsb + sb) * (kRows * kSuper / 2);
        const size_t scbase
                = ((static_cast<size_t>(g) * nsb + sb) * kSubs) * kRows;
        const size_t dbase = (static_cast<size_t>(g) * nsb + sb) * kRows;
        const int8_t *a = A + static_cast<size_t>(sb) * kSuper;

        const __m128 vd = _mm_loadu_ps(r.d.data() + dbase);
        const __m128 vdmin = _mm_loadu_ps(r.dmin.data() + dbase);

        for (int j = 0; j < kSubs; ++j) {
            // Eight four-k groups fill one sub-block, and eight is exactly the
            // s16 headroom, so this accumulator never has to be flushed early.
            __m128i acc16 = _mm_setzero_si128();
            const uint8_t *qj = qs + static_cast<size_t>(j) * (kSub / 4) * 8;
            const int8_t *aj = a + static_cast<size_t>(j) * kSub;

            // TWO four-k groups per iteration. The first cut loaded eight
            // bytes and built one operand with an unpacklo, and measured SLOWER
            // than the row-at-a-time kernel it was meant to beat: that kernel
            // gets a whole sixteen-weight PMADDUBSW operand from one 16-byte
            // load and a single AND, so paying a half-width load plus an extra
            // unpack per operand handed back everything the reduction saved.
            //
            // A full 16-byte load covers two adjacent groups, and unpacklo and
            // unpackhi of the SAME lo/hi pair yield both operands -- so the
            // load, the AND and the shift are amortised over 32 weights instead
            // of 16. The activations for both groups are eight contiguous bytes,
            // so one load and two dword shuffles replace two broadcasts.
            for (int kq = 0; kq < kSub / 4; kq += 2) {
                const __m128i b = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(qj + kq * 8));
                const __m128i lo = _mm_and_si128(b, mask0f);
                const __m128i hi
                        = _mm_and_si128(_mm_srli_epi16(b, 4), mask0f);
                const __m128i w0 = _mm_unpacklo_epi8(lo, hi);
                const __m128i w1 = _mm_unpackhi_epi8(lo, hi);

                const __m128i a8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(aj + kq * 4));
                const __m128i av0 = _mm_shuffle_epi32(a8, 0x00);
                const __m128i av1 = _mm_shuffle_epi32(a8, 0x55);

                acc16 = _mm_add_epi16(acc16, _mm_maddubs_epi16(w0, av0));
                acc16 = _mm_add_epi16(acc16, _mm_maddubs_epi16(w1, av1));
            }

            // [s0 s0 s1 s1 s2 s2 s3 s3]: PMADDWD folds the pair belonging to
            // each row, so widening and scaling are one instruction.
            const __m128i sc4 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(
                            r.sc.data() + scbase + static_cast<size_t>(j) * kRows));
            const __m128i scv = _mm_unpacklo_epi16(sc4, sc4);
            const __m128 dot = _mm_cvtepi32_ps(_mm_madd_epi16(acc16, scv));

            const __m128i mn4 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(
                            r.mn.data() + scbase + static_cast<size_t>(j) * kRows));
            const __m128 mnf
                    = _mm_cvtepi32_ps(_mm_unpacklo_epi16(mn4, _mm_setzero_si128()));
            const __m128 rs = _mm_set1_ps(
                    static_cast<float>(rowsum[sb * kSubs + j]));
            const __m128 ss = _mm_set1_ps(
                    ss_grp ? src_scale[sb * kSubs + j] : src_scale[0]);

            const __m128 term = _mm_sub_ps(_mm_mul_ps(vd, dot),
                    _mm_mul_ps(_mm_mul_ps(vdmin, mnf), rs));
            out = _mm_add_ps(out, _mm_mul_ps(ss, term));
        }
    }
    _mm_storeu_ps(C + static_cast<size_t>(g) * kRows, out);
}

} // namespace

bool int8_q4k_gemv_ilv_supported(int M, int N, int K, int ggml_type) {
    if (M != 1) return false;
    if (ggml_type != kQ4K) return false;
    if (K <= 0 || K % kSuper != 0) return false;
    if (N <= 0 || N % kRows != 0) return false;
    return true;
}

void int8_q4k_gemv_ilv_clear_cache() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    cache_map().clear();
}

bool int8_q4k_gemv_ilv_128(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads) {
    if (!int8_q4k_gemv_ilv_supported(1, N, K, ggml_type)) return false;
    if (A == nullptr || blocks == nullptr || C == nullptr
            || src_scale == nullptr)
        return false;

    const Repacked *r = get_or_repack(Key {blocks, N, K}, blocks);
    if (r == nullptr) return false;

    const int nsb = K / kSuper;
    const int ngrp = N / kRows;

    // One sum per sub-block of the activation row, shared by every output. The
    // min term needs it and it is 32 adds per sub-block against N*K multiplies.
    std::vector<int32_t> rowsum(static_cast<size_t>(nsb) * kSubs);
    for (int i = 0; i < nsb * kSubs; ++i) {
        int32_t s = 0;
        const int8_t *p = A + static_cast<size_t>(i) * kSub;
        for (int t = 0; t < kSub; ++t) s += p[t];
        rowsum[i] = s;
    }

    const int nt = nthreads > 0 ? nthreads : 1;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int g = 0; g < ngrp; ++g) {
        run_group(*r, g, nsb, A, rowsum.data(), src_scale, ss_grp, C);
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
