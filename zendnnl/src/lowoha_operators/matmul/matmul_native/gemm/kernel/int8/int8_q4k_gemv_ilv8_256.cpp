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
// Q4_K decode GEMV, eight rows interleaved, 256-bit.
//
// THE LANE ALGEBRA. VPMADDUBSW on 32 bytes gives sixteen s16, each the sum of an
// adjacent byte pair; VPMADDWD then folds adjacent s16 pairs into eight s32. So a
// 256-bit chain turns 32 input bytes into eight s32 lanes, and making those lanes
// eight OUTPUT ROWS removes the horizontal reduction entirely. Each row therefore
// contributes four consecutive k per operand:
//
//   W = [r0k0..r0k3 | r1k0..r1k3 | ... | r7k0..r7k3]     32 bytes
//   A = [k0 k1 k2 k3] x8                                 VPBROADCASTD of a dword
//
// THE NIBBLE PLANE, which is the fix the four-row version needed. Pack byte j of a
// 32-byte block with group A's weight j in the LOW nibble and group B's weight j
// in the HIGH nibble. Then
//
//   lo = AND(B, 0x0F)           is group A's operand, complete, in order
//   hi = AND(SRL(B, 4), 0x0F)   is group B's operand, complete, in order
//
// Three instructions yield two full operands covering 64 weight-multiplies, with
// no interleaving step at all. The four-row kernel instead packed W[2j] and
// W[2j+1] into one byte and needed an unpacklo per operand; that cost is what ate
// the reduction saving and left it at 0.94x. It also would not have worked here:
// AVX2's unpacklo_epi8 acts within each 128-bit lane, so it cannot produce a
// 32-byte operand in this order.
//
// SCALES ARE PRE-DUPLICATED IN THE REPACK. VPMADDWD needs [s0 s0 s1 s1 .. s7 s7]
// so that the pair belonging to each row shares its scale, and building that from
// eight values at runtime would need lane-crossing shuffles. The repack writes the
// sixteen-wide form directly, so the kernel loads it and does nothing to it -- and
// the same instruction that widens s16 to s32 applies the six-bit scale, which is
// the double duty ggml gets and the reason the fp32 epilogue is small.
//
// ACCUMULATION DEPTH. A Q4_K code is 0..15 and the activation contract excludes
// -128, so one VPMADDUBSW lane is at most 15*127*2 = 3810 and eight accumulate to
// 30480, inside the s16 limit. Eight four-k groups is 32 k, exactly one sub-block,
// exactly the span of one scale: the widening lands on a sub-block boundary with
// no partial state.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q4k_gemv_ilv8_256.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {
namespace {

constexpr int kSuper = 256;
constexpr int kSub = 32;
constexpr int kSubs = kSuper / kSub;
constexpr int kRows = 8;
constexpr int kQ4K = 12;

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

// One Q4_K weight, by row and k. GGML packs per 64-WEIGHT CHUNK: inside chunk c
// the low nibbles of bytes 32c..32c+31 are k = 64c..64c+31 and the high nibbles of
// the SAME bytes are k = 64c+32..64c+63.
inline uint8_t code_at(const block_q4_K &b, int k) {
    const int c = k / 64;
    const int rem = k % 64;
    const int byte = 32 * c + (rem % 32);
    return (rem < 32) ? static_cast<uint8_t>(b.qs[byte] & 0x0F)
                      : static_cast<uint8_t>(b.qs[byte] >> 4);
}

struct Repacked {
    std::vector<uint8_t> qs;  // N*K/2, nibble-plane pairs of four-k groups
    std::vector<int16_t> scv; // per rowgroup/sb/sub: 16, pre-duplicated
    std::vector<int16_t> mn;  // per rowgroup/sb/sub: 8
    std::vector<float> d;     // per rowgroup/sb: 8
    std::vector<float> dmin;  // per rowgroup/sb: 8
    size_t bytes() const {
        return qs.size() + scv.size() * 2 + mn.size() * 2 + d.size() * 4
                + dmin.size() * 4;
    }
};

std::unique_ptr<Repacked> repack(int N, int K, const void *blocks) {
    const int nsb = K / kSuper;
    const int ngrp = N / kRows;
    auto r = std::make_unique<Repacked>();
    r->qs.resize(static_cast<size_t>(N) * K / 2);
    r->scv.resize(static_cast<size_t>(ngrp) * nsb * kSubs * 16);
    r->mn.resize(static_cast<size_t>(ngrp) * nsb * kSubs * kRows);
    r->d.resize(static_cast<size_t>(ngrp) * nsb * kRows);
    r->dmin.resize(static_cast<size_t>(ngrp) * nsb * kRows);

    const auto *src = static_cast<const block_q4_K *>(blocks);

    for (int g = 0; g < ngrp; ++g) {
        for (int sb = 0; sb < nsb; ++sb) {
            const size_t bi = static_cast<size_t>(g) * nsb + sb;

            for (int rr = 0; rr < kRows; ++rr) {
                const block_q4_K &b
                        = src[static_cast<size_t>(g * kRows + rr) * nsb + sb];
                r->d[bi * kRows + rr] = fp16_to_fp32(b.d);
                r->dmin[bi * kRows + rr] = fp16_to_fp32(b.dmin);
                for (int j = 0; j < kSubs; ++j) {
                    uint8_t sc, mn;
                    get_scale_min_k4(j, b.scales, &sc, &mn);
                    // [s0 s0 s1 s1 .. s7 s7]: the pair VPMADDWD folds for row rr.
                    int16_t *sv = r->scv.data() + (bi * kSubs + j) * 16;
                    sv[2 * rr + 0] = sc;
                    sv[2 * rr + 1] = sc;
                    r->mn[(bi * kSubs + j) * kRows + rr] = mn;
                }
            }

            // Nibble plane. Pair four-k groups 2p and 2p+1 into one 32-byte
            // block: byte j low is group 2p's weight j, high is group 2p+1's.
            uint8_t *dst
                    = r->qs.data() + bi * (static_cast<size_t>(kRows) * kSuper / 2);
            for (int p = 0; p < kSuper / 8; ++p) {
                uint8_t *B = dst + static_cast<size_t>(p) * 32;
                for (int j = 0; j < 32; ++j) {
                    const int rr = j / 4;
                    const int t = j % 4;
                    const block_q4_K &b
                            = src[static_cast<size_t>(g * kRows + rr) * nsb + sb];
                    const uint8_t a = code_at(b, (2 * p) * 4 + t);
                    const uint8_t c = code_at(b, (2 * p + 1) * 4 + t);
                    B[j] = static_cast<uint8_t>(a | (c << 4));
                }
            }
        }
    }
    return r;
}

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
        const char *e = std::getenv("ZENDNNL_Q4K_ILV8_CACHE_MB");
        const long mb = (e != nullptr) ? std::atol(e) : 0;
        return static_cast<size_t>((mb > 0 ? mb : 4096)) * 1024u * 1024u;
    }();
    return b;
}

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

bool host_has_avx2() {
    static const bool v = __builtin_cpu_supports("avx2") != 0;
    return v;
}

} // namespace

#pragma GCC push_options
#pragma GCC target("avx2")
namespace {

void run_group_avx2(const Repacked &r, int g, int nsb, const int8_t *A,
        const int32_t *rowsum, const float *src_scale, int ss_grp, float *C) {
    const __m256i mask0f = _mm256_set1_epi8(0x0F);
    __m256 out = _mm256_setzero_ps();

    for (int sb = 0; sb < nsb; ++sb) {
        const size_t bi = static_cast<size_t>(g) * nsb + sb;
        const uint8_t *qs
                = r.qs.data() + bi * (static_cast<size_t>(kRows) * kSuper / 2);
        const int8_t *a = A + static_cast<size_t>(sb) * kSuper;
        const __m256 vd = _mm256_loadu_ps(r.d.data() + bi * kRows);
        const __m256 vdmin = _mm256_loadu_ps(r.dmin.data() + bi * kRows);

        for (int j = 0; j < kSubs; ++j) {
            // Four pairs is eight four-k groups is 32 k: one sub-block, and
            // exactly the s16 headroom, so no early flush.
            __m256i acc16 = _mm256_setzero_si256();
            const uint8_t *qj = qs + static_cast<size_t>(j) * (kSub / 8) * 32;
            const int8_t *aj = a + static_cast<size_t>(j) * kSub;

            for (int p = 0; p < kSub / 8; ++p) {
                const __m256i b = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(qj + p * 32));
                const __m256i w0 = _mm256_and_si256(b, mask0f);
                const __m256i w1 = _mm256_and_si256(
                        _mm256_srli_epi16(b, 4), mask0f);

                int32_t d0, d1;
                std::memcpy(&d0, aj + p * 8, sizeof(d0));
                std::memcpy(&d1, aj + p * 8 + 4, sizeof(d1));
                const __m256i av0 = _mm256_set1_epi32(d0);
                const __m256i av1 = _mm256_set1_epi32(d1);

                acc16 = _mm256_add_epi16(
                        acc16, _mm256_maddubs_epi16(w0, av0));
                acc16 = _mm256_add_epi16(
                        acc16, _mm256_maddubs_epi16(w1, av1));
            }

            // Pre-duplicated, so this both widens and scales.
            const __m256i scv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(
                            r.scv.data() + (bi * kSubs + j) * 16));
            const __m256 dot
                    = _mm256_cvtepi32_ps(_mm256_madd_epi16(acc16, scv));

            const __m128i mn8 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(
                            r.mn.data() + (bi * kSubs + j) * kRows));
            const __m256 mnf
                    = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(mn8));
            const __m256 rs = _mm256_set1_ps(
                    static_cast<float>(rowsum[sb * kSubs + j]));
            const __m256 ss = _mm256_set1_ps(
                    ss_grp ? src_scale[sb * kSubs + j] : src_scale[0]);

            const __m256 term = _mm256_sub_ps(_mm256_mul_ps(vd, dot),
                    _mm256_mul_ps(_mm256_mul_ps(vdmin, mnf), rs));
            out = _mm256_add_ps(out, _mm256_mul_ps(ss, term));
        }
    }
    _mm256_storeu_ps(C + static_cast<size_t>(g) * kRows, out);
}

} // namespace
#pragma GCC pop_options

bool int8_q4k_gemv_ilv8_supported(int M, int N, int K, int ggml_type) {
    if (M != 1) return false;
    if (ggml_type != kQ4K) return false;
    if (K <= 0 || K % kSuper != 0) return false;
    if (N <= 0 || N % kRows != 0) return false;
    return host_has_avx2();
}

void int8_q4k_gemv_ilv8_clear_cache() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    cache_map().clear();
}

bool int8_q4k_gemv_ilv8_256(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads) {
    if (!int8_q4k_gemv_ilv8_supported(1, N, K, ggml_type)) return false;
    if (A == nullptr || blocks == nullptr || C == nullptr
            || src_scale == nullptr)
        return false;

    const Repacked *r = get_or_repack(Key {blocks, N, K}, blocks);
    if (r == nullptr) return false;

    const int nsb = K / kSuper;
    const int ngrp = N / kRows;

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
        run_group_avx2(*r, g, nsb, A, rowsum.data(), src_scale, ss_grp, C);
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
