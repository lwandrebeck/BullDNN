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
// Q6_K decode GEMV over pre-stitched codes.
//
// THE MAPPING, derived from the row-at-a-time kernel's own offsets rather than
// from GGML's documentation, because the kernel is what defines the convention
// that the scales and the row sums already agree with. In superblock_dots_q6,
// with QL = ql + 64c, QH = qh + 32c and AA = a + 128c, the four PMADDUBSW per
// (c, l) step pair weights with activations at AA+l, AA+l+32, AA+l+64, AA+l+96.
// Reading the operand construction back through those pairings gives, for
// m = 0..31 inside each half c:
//
//   k = 128c +  0 + m   ql[64c +  0 + m] & 0xF   qh[32c + m] bits 0-1
//   k = 128c + 32 + m   ql[64c + 32 + m] & 0xF   qh[32c + m] bits 2-3
//   k = 128c + 64 + m   ql[64c +  0 + m] >>  4   qh[32c + m] bits 4-5
//   k = 128c + 96 + m   ql[64c + 32 + m] >>  4   qh[32c + m] bits 6-7
//
// so one qh byte serves four codes, 64 apart in k. Stored in natural k order the
// codes line up with the activations, and the inner loop is four 16-byte loads
// against four 16-byte activation loads -- no shifts, no masks, no ors.
//
// WHAT IS DELIBERATELY UNCHANGED. The accumulator array stays at sixteen live.
// Cutting it to eight removed every stack spill in the object and measured
// SLOWER, 4.3 against 4.5: the spill stores go to L1 and the store queue absorbs
// them while sixteen independent PMADDUBSW chains keep both pipes fed, whereas
// reducing early inserts VPHADDD bursts into the middle of the loop. That result
// is recorded in q4k_decode_profile_a10.csv and is not to be re-litigated here.
//
// The epilogue is also copied rather than shared, including the detail that the
// ACTIVATION scale is per 32 while a Q6_K sub-block is SIXTEEN wide, so each
// pair of sub-blocks shares one scale. Getting that wrong once produced NaN in
// llama-perplexity while the unit test passed, because the test had been written
// to the kernel's convention instead of the caller's.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q6k_gemv_pre_128.hpp"

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
constexpr int kSubs = 16; // Q6_K sub-blocks per super-block, sixteen wide

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
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

struct Stitched {
    std::vector<uint8_t> codes; // N * K, natural k order
    std::vector<int8_t> scales; // N * nsb * 16
    std::vector<float> d;       // N * nsb, already widened
    size_t bytes() const {
        return codes.size() + scales.size() + d.size() * 4;
    }
};

std::unique_ptr<Stitched> stitch(int N, int K, const void *blocks) {
    const int nsb = K / kSuper;
    auto s = std::make_unique<Stitched>();
    s->codes.resize(static_cast<size_t>(N) * K);
    s->scales.resize(static_cast<size_t>(N) * nsb * kSubs);
    s->d.resize(static_cast<size_t>(N) * nsb);

    const auto *src = static_cast<const block_q6_K *>(blocks);
    for (int n = 0; n < N; ++n) {
        for (int sb = 0; sb < nsb; ++sb) {
            const block_q6_K &b = src[static_cast<size_t>(n) * nsb + sb];
            uint8_t *cd = s->codes.data()
                    + (static_cast<size_t>(n) * nsb + sb) * kSuper;
            for (int c = 0; c < 2; ++c) {
                const uint8_t *QL = b.ql + 64 * c;
                const uint8_t *QH = b.qh + 32 * c;
                uint8_t *K0 = cd + 128 * c;
                for (int m = 0; m < 32; ++m) {
                    const uint8_t h = QH[m];
                    K0[m] = static_cast<uint8_t>(
                            (QL[m] & 0x0F) | ((h & 0x03) << 4));
                    K0[m + 32] = static_cast<uint8_t>(
                            (QL[m + 32] & 0x0F) | (((h >> 2) & 0x03) << 4));
                    K0[m + 64] = static_cast<uint8_t>(
                            (QL[m] >> 4) | (((h >> 4) & 0x03) << 4));
                    K0[m + 96] = static_cast<uint8_t>(
                            (QL[m + 32] >> 4) | (((h >> 6) & 0x03) << 4));
                }
            }
            std::memcpy(s->scales.data()
                            + (static_cast<size_t>(n) * nsb + sb) * kSubs,
                    b.scales, kSubs);
            s->d[static_cast<size_t>(n) * nsb + sb] = fp16_to_fp32(b.d);
        }
    }
    return s;
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
std::unordered_map<Key, std::unique_ptr<Stitched>, KeyHash> &cache_map() {
    static std::unordered_map<Key, std::unique_ptr<Stitched>, KeyHash> m;
    return m;
}
size_t cache_budget() {
    static const size_t b = [] {
        const char *e = std::getenv("ZENDNNL_Q6K_PRE_CACHE_MB");
        const long mb = (e != nullptr) ? std::atol(e) : 0;
        return static_cast<size_t>((mb > 0 ? mb : 2048)) * 1024u * 1024u;
    }();
    return b;
}

// nullptr once the budget is spent: stitching per call would be O(N*K) every
// token, far worse than the bit work it replaces.
const Stitched *get_or_stitch(const Key &key, const void *blocks) {
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto &m = cache_map();
    auto it = m.find(key);
    if (it != m.end()) return it->second.get();
    size_t resident = 0;
    for (const auto &kv : m) resident += kv.second->bytes();
    auto built = stitch(key.N, key.K, blocks);
    if (resident + built->bytes() > cache_budget()) return nullptr;
    const Stitched *raw = built.get();
    m.emplace(key, std::move(built));
    return raw;
}

} // namespace

bool int8_q6k_gemv_pre_supported(int N, int K) {
    return N > 0 && K > 0 && K % kSuper == 0;
}

void int8_q6k_gemv_pre_clear_cache() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    cache_map().clear();
}

bool int8_q6k_gemv_pre_128(int N, int K, const int8_t *A, const void *blocks,
        float *C, const float *src_scale, int ss_grp, const int32_t *rowsum16,
        int nthreads) {
    if (!int8_q6k_gemv_pre_supported(N, K)) return false;
    if (A == nullptr || blocks == nullptr || C == nullptr
            || src_scale == nullptr || rowsum16 == nullptr)
        return false;

    const Stitched *st = get_or_stitch(Key {blocks, N, K}, blocks);
    if (st == nullptr) return false;

    const int nsb = K / kSuper;
    const int nt = nthreads > 0 ? nthreads : 1;
    const __m128i ones = _mm_set1_epi16(1);

#pragma omp parallel for schedule(static) num_threads(nt)
    for (int n = 0; n < N; ++n) {
        __m128 accv = _mm_setzero_ps();
        for (int sb = 0; sb < nsb; ++sb) {
            const size_t bi = static_cast<size_t>(n) * nsb + sb;
            const uint8_t *cd = st->codes.data() + bi * kSuper;
            const int8_t *sc8 = st->scales.data() + bi * kSubs;
            const int8_t *a = A + static_cast<size_t>(sb) * kSuper;

            // Sixteen live on purpose -- see the header.
            __m128i acc[16];
            for (int c = 0; c < 2; ++c) {
                const uint8_t *CD = cd + 128 * c;
                const int8_t *AA = a + 128 * c;
                for (int l = 0; l < 32; l += 16) {
                    const int base = c * 8 + (l / 16);
                    const __m128i w0 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(CD + l));
                    const __m128i w1 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(CD + l + 32));
                    const __m128i w2 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(CD + l + 64));
                    const __m128i w3 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(CD + l + 96));
                    const __m128i a0 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(AA + l));
                    const __m128i a1 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(AA + l + 32));
                    const __m128i a2 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(AA + l + 64));
                    const __m128i a3 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(AA + l + 96));

                    acc[base + 0]
                            = _mm_madd_epi16(_mm_maddubs_epi16(w0, a0), ones);
                    acc[base + 2]
                            = _mm_madd_epi16(_mm_maddubs_epi16(w1, a1), ones);
                    acc[base + 4]
                            = _mm_madd_epi16(_mm_maddubs_epi16(w2, a2), ones);
                    acc[base + 6]
                            = _mm_madd_epi16(_mm_maddubs_epi16(w3, a3), ones);
                }
            }

            __m128i dots[4];
            for (int g = 0; g < 4; ++g) {
                dots[g] = _mm_hadd_epi32(
                        _mm_hadd_epi32(acc[4 * g + 0], acc[4 * g + 1]),
                        _mm_hadd_epi32(acc[4 * g + 2], acc[4 * g + 3]));
            }

            const int g0 = sb * kSubs;
            const __m128 vd = _mm_set1_ps(st->d[bi]);
            const __m128 v32 = _mm_set1_ps(32.0f);
            for (int g = 0; g < 4; ++g) {
                const int j0 = g * 4;
                const __m128 vdot = _mm_cvtepi32_ps(dots[g]);
                const __m128 vsc = _mm_setr_ps(sc8[j0], sc8[j0 + 1],
                        sc8[j0 + 2], sc8[j0 + 3]);
                const __m128 vrs = _mm_cvtepi32_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(rowsum16 + g0 + j0)));
                // Activation scales are per 32 and a sub-block is sixteen, so a
                // PAIR of sub-blocks shares one scale.
                const int k0 = (g0 + j0) / 2;
                const __m128 vss = ss_grp
                        ? _mm_setr_ps(src_scale[k0], src_scale[k0],
                                src_scale[k0 + 1], src_scale[k0 + 1])
                        : _mm_set1_ps(src_scale[0]);
                const __m128 term = _mm_mul_ps(_mm_mul_ps(vd, vsc),
                        _mm_sub_ps(vdot, _mm_mul_ps(v32, vrs)));
                accv = _mm_add_ps(accv, _mm_mul_ps(vss, term));
            }
        }
        C[n] = _mm_cvtss_f32(_mm_add_ss(
                _mm_add_ss(accv, _mm_shuffle_ps(accv, accv, 0x55)),
                _mm_add_ss(_mm_movehl_ps(accv, accv),
                        _mm_shuffle_ps(accv, accv, 0xFF))));
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
