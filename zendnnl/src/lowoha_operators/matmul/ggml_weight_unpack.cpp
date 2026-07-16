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

/**
 * @file ggml_weight_unpack.cpp
 * @brief GGML weight unpacking for ZenDNN kernels
 */

#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>

#include "ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#include "lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

// GGML quant block formats (Q8_0, Q4_0, ...) are defined with a fixed
// 32-element group along K. The downstream AOCL sym-quant kernel must use
// the same group size as the weights, so it is a constant for this path.
constexpr int kGgmlGroupSize = 32;

struct block_q8_0  { uint16_t d; int8_t  qs[32];  };
struct block_q4_0  { uint16_t d; uint8_t qs[16];  };
struct block_q4_0x8 { uint16_t d[8]; uint8_t qs[128]; };

inline uint32_t fp32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    return bits;
}

inline float fp32_from_bits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(bits));
    return f;
}

// fp16 -> fp32, from llama.cpp
float fp16_to_fp32(uint16_t h) {
    const uint32_t w             = static_cast<uint32_t>(h) << 16;
    const uint32_t sign          = w & UINT32_C(0x80000000);
    const uint32_t two_w         = w + w;
    const uint32_t exp_offset    = UINT32_C(0xE0) << 23;
    const float    exp_scale     = fp32_from_bits(UINT32_C(0x7800000));
    const float    normalized    = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask    = UINT32_C(126) << 23;
    const float    denormalized  = fp32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
    const uint32_t result = sign |
        (two_w < (UINT32_C(1) << 27) ? fp32_to_bits(denormalized)
                                     : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

// fp32 -> bf16, from llama.cpp
uint16_t fp32_to_bf16(float s) {
    union { float f; uint32_t i; } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000)
        return (u.i >> 16) | 64; // NaN: force quiet
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

void write_scale(void *scale_buffer, bool use_bf16, int64_t idx, float val) {
    if (use_bf16)
        static_cast<uint16_t *>(scale_buffer)[idx] = fp32_to_bf16(val);
    else
        static_cast<float *>(scale_buffer)[idx] = val;
}

size_t align_up(size_t size, size_t alignment = 64) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ── Why eviction is DISABLED (UINT32_MAX capacity) ────────────────
// group_matmul_direct warms EVERY expert's weight per call — the active
// ones AND the currently-cold (M==0) ones — then hands the cached buffer
// pointers to the GEMM, which dereferences them AFTER the warm loop
// finishes.  A bounded capacity below the warmed pool size (up to
// 2 * total_experts entries for a fused gate/up + down layer) would let a
// later expert's add() evict() an EARLIER, still-in-use expert's buffer
// mid-call and std::free() it → use-after-free when the GEMM reads it.
// Pinning to an infinite capacity sidesteps that without per-pointer
// refcounts, mirroring pack_cache_singleton() in custom_kernel/pack.cpp;
// it explicitly overrides any global ZENDNNL_LRU_CACHE_CAPACITY setting.
// The footprint is model-bounded (one entry per distinct weight
// pointer/shape); weight-rotating workloads should call
// clear_ggml_weight_unpack_cache() during a quiescent window.
lru_cache_t<Key_matmul, void *> &get_ggml_reordered_weight_cache() {
    static lru_cache_t<Key_matmul, void *> cache(
        std::numeric_limits<uint32_t>::max());
    return cache;
}

std::mutex &get_ggml_reordered_weight_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

size_t ggml_scale_bytes(int64_t N, int64_t K, bool use_bf16_scales) {
    return static_cast<size_t>(N * (K / 32) *
                               (use_bf16_scales ? sizeof(uint16_t) : sizeof(float)));
}

status_t ggml_reorder_size(int N, int K, char trans, size_t &reorder_size) {
#if ZENDNNL_DEPENDS_AOCLDLP
    DLP_SYMM_STAT_QUANT symq_meta;
    symq_meta.group_size = kGgmlGroupSize;
    size_t raw_size = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
                          'r', trans, 'B', K, N, &symq_meta, nullptr);
    reorder_size = align_up(raw_size);
    return status_t::success;
#else
    log_error("GGML packed weights require AOCL DLP sym-quant reorder");
    return status_t::failure;
#endif
}

status_t ggml_reorder_unpacked_weights(int N, int K, int ldb, char trans,
                                       const int8_t *unpacked_weights,
                                       int8_t *reordered_weights) {
#if ZENDNNL_DEPENDS_AOCLDLP
    DLP_SYMM_STAT_QUANT symq_meta;
    symq_meta.group_size = kGgmlGroupSize;
    aocl_reorder_s8s8s32os32_sym_quant(
        'r', trans, 'B', unpacked_weights, reordered_weights, K, N, ldb,
        &symq_meta, nullptr);
    return status_t::success;
#else
    log_error("GGML packed weights require AOCL DLP sym-quant reorder");
    return status_t::failure;
#endif
}

} // anonymous namespace

bool ggml_is_sym_quant(const matmul_params &params) {
    size_t nelems = 1;
    for (auto d : params.quant_params.src_scale.dims) {
        nelems *= static_cast<size_t>(d);
    }
    return params.dtypes.wei == data_type_t::s8 &&
           params.dtypes.src == data_type_t::s8 &&
           !params.quant_params.src_zp.buff &&
           nelems > 1 &&
           (params.dtypes.dst == data_type_t::f32 ||
            params.dtypes.dst == data_type_t::bf16);
}

status_t validate_ggml_packed_inputs(const matmul_params &params,
                                     bool is_weights_const,
                                     int Batch_B, bool transB) {
    if (params.packing.pack_format_b != 1) {
        return status_t::success;
    }
    if (!is_weights_const) {
        log_error("GGML packed weights require constant weights for cached "
                  "out-of-place unpack/reorder");
        return status_t::failure;
    }
    if (Batch_B > 1) {
        log_error("GGML packed weights with Batch_B > 1 are not supported by "
                  "the single-buffer unpack/reorder cache");
        return status_t::failure;
    }
    // GGML packs weights as a fixed N x K row-major layout. The AOCL
    // sym-quant reorder must therefore be told the input is the transpose
    // of the logical B matrix (trans='t'), which corresponds to transB=true.
    // With transB=false, the reorder would misinterpret the unpacked buffer
    // as K x N and produce garbage.
    if (!transB) {
        log_error("GGML packed weights require transB=true; the unpacked "
                  "layout is N x K (transposed B)");
        return status_t::failure;
    }
    return status_t::success;
}

int64_t ggml_unpack_weight_buffer_size(int ggml_type, bool use_bf16_scales,
                                       int64_t N, int64_t K) {
    if (N <= 0 || K <= 0 || K % 32 != 0) return -1;
    int64_t ng          = K / 32;
    int64_t scale_bytes = N * ng * (use_bf16_scales ? 2 : 4);
    if (ggml_type == 8) return N * K       + scale_bytes;
    if (ggml_type == 2) return N * (K / 2) + scale_bytes;
    return -1;
}

int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
                              bool is_superblock, bool use_bf16_scales,
                              bool use_unsigned_q4, int64_t N, int64_t K,
                              int8_t **wei_ptr, void **scl_ptr,
                              void *unpack_buffer) {
    if (!weight_data || !wei_ptr || !scl_ptr) return -1;
    if (N <= 0 || K <= 0 || K % 32 != 0)      return -1;

    // If no destination is supplied, preserve the legacy inplace behavior.
    // A supplied destination lets callers keep packed GGML bytes immutable.
    void *buf = unpack_buffer ? unpack_buffer : const_cast<void *>(weight_data);

    const int64_t ng         = K / 32;
    const int64_t num_blocks = N * ng;

    const int64_t weight_bytes = (ggml_type == 8) ? N * K : N * (K / 2);
    *wei_ptr = static_cast<int8_t *>(buf);
    *scl_ptr = static_cast<void  *>(static_cast<int8_t *>(buf) + weight_bytes);

    // Q8_0: copy int8 weights directly, one scale per group.
    if (ggml_type == 8) {
        auto *blocks = static_cast<const block_q8_0 *>(weight_data);

        std::unique_ptr<int8_t[]>  tmp_weights(new int8_t[N * K]);
        const size_t scale_bytes =
            num_blocks * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));
        std::unique_ptr<uint8_t[]> tmp_scales(new uint8_t[scale_bytes]);

        const int64_t rows4 = (N / 4) * 4;

        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < rows4; row += 4) {
            for (int64_t g = 0; g < ng; g++) {
                const block_q8_0 &b0 = blocks[(row + 0) * ng + g];
                const block_q8_0 &b1 = blocks[(row + 1) * ng + g];
                const block_q8_0 &b2 = blocks[(row + 2) * ng + g];
                const block_q8_0 &b3 = blocks[(row + 3) * ng + g];

                write_scale(tmp_scales.get(), use_bf16_scales,
                            g * N + row + 0, fp16_to_fp32(b0.d));
                write_scale(tmp_scales.get(), use_bf16_scales,
                            g * N + row + 1, fp16_to_fp32(b1.d));
                write_scale(tmp_scales.get(), use_bf16_scales,
                            g * N + row + 2, fp16_to_fp32(b2.d));
                write_scale(tmp_scales.get(), use_bf16_scales,
                            g * N + row + 3, fp16_to_fp32(b3.d));

                std::memcpy(&tmp_weights[(row + 0) * K + g * 32], b0.qs, 32);
                std::memcpy(&tmp_weights[(row + 1) * K + g * 32], b1.qs, 32);
                std::memcpy(&tmp_weights[(row + 2) * K + g * 32], b2.qs, 32);
                std::memcpy(&tmp_weights[(row + 3) * K + g * 32], b3.qs, 32);
            }
        }

        for (int64_t row = rows4; row < N; row++) {
            for (int64_t g = 0; g < ng; g++) {
                const block_q8_0 &b = blocks[row * ng + g];
                write_scale(tmp_scales.get(), use_bf16_scales,
                            g * N + row, fp16_to_fp32(b.d));
                std::memcpy(&tmp_weights[row * K + g * 32], b.qs, 32);
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf,         tmp_weights.get(), N * K);
        std::memcpy(raw_buf + N * K, tmp_scales.get(),  scale_bytes);
        return 0;
    }

    // Q4_0 superblock: 8 rows interleaved, nibbles pre-signed via XOR 0x88.
    if (ggml_type == 2 && is_superblock) {
        if (N % 8 != 0) return -1;

        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t  scale_bytes     =
            num_blocks * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));

        std::unique_ptr<int8_t[]>  weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);

        auto *blocks = static_cast<const block_q4_0x8 *>(weight_data);

        #pragma omp parallel for schedule(static)
        for (int64_t r8 = 0; r8 < N / 8; r8++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0x8 local;
                std::memcpy(&local, &blocks[r8 * ng + g], sizeof(local));

                for (int ri = 0; ri < 8; ri++) {
                    const int64_t row = r8 * 8 + ri;

                    write_scale(scale_tmp.get(), use_bf16_scales,
                                g * N + row, fp16_to_fp32(local.d[ri]));

                    uint8_t src[16];
                    for (int i = 0; i < 8; i++) {
                        src[i]     = local.qs[ri * 8 + i];
                        src[i + 8] = local.qs[64 + ri * 8 + i];
                    }

                    int8_t *dst = &weight_tmp[(row * K + g * 32) / 2];
                    for (int i = 0; i < 8; i++) {
                        uint8_t lo0 = src[2*i]     & 0x0F;
                        uint8_t lo1 = src[2*i + 1] & 0x0F;
                        uint8_t hi0 = src[2*i]     >> 4;
                        uint8_t hi1 = src[2*i + 1] >> 4;

                        if (use_unsigned_q4) {
                            lo0 ^= 8; lo1 ^= 8; hi0 ^= 8; hi1 ^= 8;
                        }

                        dst[i]     = static_cast<int8_t>(lo0 | (lo1 << 4));
                        dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                    }
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf,                   weight_tmp.get(), weight_q4_bytes);
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(),  scale_bytes);
        return 0;
    }

    // Q4_0 regular: unsigned nibbles 0-15, subtract 8 to convert to signed S4.
    if (ggml_type == 2) {
        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t  scale_bytes     =
            num_blocks * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));

        std::unique_ptr<int8_t[]>  weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);

        auto *blocks = static_cast<const block_q4_0 *>(weight_data);

        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < N; row++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0 local;
                std::memcpy(&local, &blocks[row * ng + g], sizeof(local));

                write_scale(scale_tmp.get(), use_bf16_scales,
                            g * N + row, fp16_to_fp32(local.d));

                int8_t *dst = &weight_tmp[(row * K + g * 32) / 2];
                for (int i = 0; i < 8; i++) {
                    uint8_t lo0 = local.qs[2*i]     & 0x0F;
                    uint8_t lo1 = local.qs[2*i + 1] & 0x0F;
                    uint8_t hi0 = local.qs[2*i]     >> 4;
                    uint8_t hi1 = local.qs[2*i + 1] >> 4;

                    if (!use_unsigned_q4) {
                        lo0 = (lo0 - 8) & 0x0F;
                        lo1 = (lo1 - 8) & 0x0F;
                        hi0 = (hi0 - 8) & 0x0F;
                        hi1 = (hi1 - 8) & 0x0F;
                    }

                    dst[i]     = static_cast<int8_t>(lo0 | (lo1 << 4));
                    dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf,                   weight_tmp.get(), weight_q4_bytes);
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(),  scale_bytes);
        return 0;
    }

    return -1; // unsupported type
}

// Raw-s8 GGML unpack: cache the UNPACKED s8 weight (N x K) + {K/32, N} bf16
// scales and hand it back UN-reordered (mem_format 'n', pack_format_b cleared).
// The weight is then a plain per-group s8 weight, so the flat_n_tile (ALGO 3)
// per-group path reorders it INTERNALLY per N-tile (`do_tile`'s `{G, n_tile}`
// sym-quant repack) — exactly how a caller-provided per-group s8 weight
// already flows.  Used when the call may run on N-tile (ALGO 3 / AUTO) so the
// GEMM handles the reorder after N-tiling instead of pre-reordering here.
//
// Cached under a DISTINCT key marker (`native_gemm`) so a raw-s8 entry never
// aliases the AOCL-reordered entry (`aocl_dlp_blocked`) for the same
// (weight, K, N, ldb, trans) tuple, in case a process runs both modes.
static status_t unpack_ggml_raw_s8_and_cache(const void *&weight, int N, int K,
                                             int ldb, char trans,
                                             matmul_params &params) {
  const int64_t unpack_size = ggml_unpack_weight_buffer_size(8, true, N, K);
  if (unpack_size < 0) {
    log_error("GGML raw-s8 unpack failed: invalid dimensions");
    return status_t::failure;
  }
  const size_t weight_bytes = static_cast<size_t>(N) * static_cast<size_t>(K);

  Key_matmul cache_key(trans == 't', static_cast<unsigned int>(K),
                       static_cast<unsigned int>(N),
                       static_cast<unsigned int>(ldb), weight,
                       static_cast<uint32_t>(matmul_algo_t::native_gemm));

  lru_cache_t<Key_matmul, void *> &weight_cache =
      get_ggml_reordered_weight_cache();
  std::mutex &cache_mutex = get_ggml_reordered_weight_cache_mutex();

  void *cached_buffer = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (weight_cache.try_get(cache_key, cached_buffer)) {
      apilog_info("GGML raw-s8 unpack cache hit: N=", N, ", K=", K);
    }
  }

  if (!cached_buffer) {
    apilog_info("GGML raw-s8 unpack cache miss: N=", N, ", K=", K);
    void *buf =
        aligned_alloc(64, align_up(static_cast<size_t>(unpack_size)));
    if (!buf) {
      log_error("GGML raw-s8 unpack failed: allocation failed");
      return status_t::failure;
    }
    // out-params required by the ggml_unpack_weight_buffer API but unused on
    // this path: the unpacked weights are `buf` itself and the wei-scale
    // pointer is re-derived from `cached_buffer + weight_bytes` below.
    int8_t *unpacked_weights = nullptr;
    void   *unpacked_scales  = nullptr;
    if (ggml_unpack_weight_buffer(weight, 8, false, true, false, N, K,
                                  &unpacked_weights, &unpacked_scales,
                                  buf) != 0) {
      std::free(buf);
      log_error("GGML raw-s8 unpack failed");
      return status_t::failure;
    }
    (void)unpacked_weights;
    (void)unpacked_scales;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (weight_cache.try_get(cache_key, cached_buffer)) {
        std::free(buf);  // another thread filled it first
      } else {
        weight_cache.add(cache_key, buf);
        cached_buffer = buf;
      }
    }
  }

  // Hand back the raw s8 weight (N x K, transB) + {K/32, N} bf16 scales,
  // UN-reordered, so the per-group N-tile DLP path reorders it per-tile.
  weight = cached_buffer;
  params.mem_format_b = 'n';
  // The weight is now plain raw s8 (unpacked): clear the GGML packed flag so
  // the dispatch treats it exactly like a caller-provided per-group s8 weight
  // (check_m_tile_safe / check_n_tile_extra reject `pack_format_b != 0` on the
  // mem_format 'n' path, so leaving it set would veto ALGO 3).
  params.packing.pack_format_b = 0;
  params.quant_params.wei_scale.buff = static_cast<const void *>(
      static_cast<const uint8_t *>(cached_buffer) + weight_bytes);
  params.quant_params.wei_scale.dt = data_type_t::bf16;
  const int64_t ng = static_cast<int64_t>(K) / 32;
  params.quant_params.wei_scale.dims = {ng, static_cast<int64_t>(N)};
  // Symmetric-int8 compute discriminator.  The unpacked weight is now a
  // plain per-group s8 matrix (identical to a caller-provided per-group s8
  // weight), so flag the call as symmetric int8.  This is the field the
  // AOCL DLP sym-quant PREPACK gate keys on (`int8_aocl_warm_candidate`:
  // wei==s8 && compute in {s8,u8}) — without it the ahead-of-time warmer
  // skips the s8 family and the per-group N-tile reorder is packed lazily
  // on first routing instead of ahead of time.  Per-group is symmetric-
  // only, so s8 (never u8).  The runtime per-group call still runs on the
  // AOCL do_tile path regardless (`ck_per_group` forces it — the custom
  // kernel has no per-group support in this path), so this only flips the
  // prepack recognition, not the executor.
  params.dtypes.compute = data_type_t::s8;

  apilog_info("GGML raw-s8 unpack output: weight_bytes=", weight_bytes,
              ", scale_groups=", ng, ", scale_dt=bf16, mem_format=n");
  return status_t::success;
}

status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
                                      int ldb, char trans,
                                      matmul_params &params,
                                      bool skip_reorder) {
  apilog_info("GGML Q8_0 unpack: N=", N, ", K=", K,
              ", weight_address=", static_cast<const void *>(weight),
              ", skip_reorder=", (skip_reorder ? 1 : 0));

  // N-tile per-group DLP path: keep the weight raw so `do_tile` reorders it
  // per N-tile, instead of pre-reordering the full weight for AOCL here.
  if (skip_reorder) {
    return unpack_ggml_raw_s8_and_cache(weight, N, K, ldb, trans, params);
  }

  // Default: unpack + AOCL sym-quant reorder + cache (mem_format 'r').
  size_t reorder_size = 0;
  if (ggml_reorder_size(N, K, trans, reorder_size) != status_t::success) {
    return status_t::failure;
  }
  const size_t scale_bytes = ggml_scale_bytes(N, K, true);
  const size_t total_cache_bytes = align_up(reorder_size + scale_bytes);

  // Reordered GGML weight bytes depend only on the weight pointer, K, N,
  // ldb, trans, and the (constant) group size, so M is intentionally not
  // part of the cache key.
  Key_matmul cache_key(trans == 't', static_cast<unsigned int>(K),
                       static_cast<unsigned int>(N),
                       static_cast<unsigned int>(ldb), weight,
                       static_cast<uint32_t>(matmul_algo_t::aocl_dlp_blocked));

  lru_cache_t<Key_matmul, void *> &weight_cache =
      get_ggml_reordered_weight_cache();
  std::mutex &cache_mutex = get_ggml_reordered_weight_cache_mutex();

  void *cached_buffer = nullptr;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (weight_cache.try_get(cache_key, cached_buffer)) {
      apilog_info("GGML unpack/reorder cache hit: N=", N, ", K=", K);
    }
  }

  if (!cached_buffer) {
    apilog_info("GGML unpack/reorder cache miss: N=", N, ", K=", K);

    const int64_t unpack_size = ggml_unpack_weight_buffer_size(
        8, true, N, K);
    if (unpack_size < 0) {
      log_error("GGML weight unpacking failed: invalid dimensions");
      return status_t::failure;
    }

    void *unpack_buffer = aligned_alloc(64, align_up(static_cast<size_t>(unpack_size)));
    if (!unpack_buffer) {
      log_error("GGML weight unpacking failed: unpack allocation failed");
      return status_t::failure;
    }

    int8_t *unpacked_weights = nullptr;
    void   *unpacked_scales = nullptr;
    int ret = ggml_unpack_weight_buffer(
        weight, 8, false, true, false,
        N, K, &unpacked_weights, &unpacked_scales,
        unpack_buffer);
    if (ret != 0) {
      std::free(unpack_buffer);
      log_error("GGML weight unpacking failed");
      return status_t::failure;
    }

    void *new_cached_buffer = aligned_alloc(64, total_cache_bytes);
    if (!new_cached_buffer) {
      std::free(unpack_buffer);
      log_error("GGML weight reorder failed: cache allocation failed");
      return status_t::failure;
    }

    if (ggml_reorder_unpacked_weights(N, K, ldb, trans,
                                      unpacked_weights,
                                      static_cast<int8_t *>(new_cached_buffer))
        != status_t::success) {
      std::free(unpack_buffer);
      std::free(new_cached_buffer);
      return status_t::failure;
    }

    std::memcpy(static_cast<uint8_t *>(new_cached_buffer) + reorder_size,
                unpacked_scales, scale_bytes);
    std::free(unpack_buffer);

    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      if (weight_cache.try_get(cache_key, cached_buffer)) {
        apilog_info("GGML unpack/reorder cache filled by another thread: N=",
                    N, ", K=", K);
      } else {
        weight_cache.add(cache_key, new_cached_buffer);
        cached_buffer = new_cached_buffer;
        new_cached_buffer = nullptr;
        apilog_info("GGML unpack/reorder complete: cached ", total_cache_bytes,
                    " bytes (reordered weights + bf16 scales)");
      }
    }

    if (new_cached_buffer) {
      std::free(new_cached_buffer);
    }
  }

  weight = cached_buffer;
  params.mem_format_b = 'r';
  params.quant_params.wei_scale.buff = static_cast<const void *>(
      static_cast<const uint8_t *>(cached_buffer) + reorder_size);
  params.quant_params.wei_scale.dt = data_type_t::bf16;
  const int64_t ng = static_cast<int64_t>(K) / 32;
  params.quant_params.wei_scale.dims = {ng, static_cast<int64_t>(N)};

  apilog_info("GGML unpack/reorder output: reorder_bytes=", reorder_size,
              ", scale_groups=", ng, ", scale_dt=bf16");

  return status_t::success;
}

void clear_ggml_weight_unpack_cache() {
  std::lock_guard<std::mutex> lock(get_ggml_reordered_weight_cache_mutex());
  get_ggml_reordered_weight_cache().clear();
}

size_t ggml_weight_unpack_cache_size() {
  // get_size() takes the cache's own internal mutex; do not also hold the
  // module mutex here to avoid a lock-order dependency.
  return static_cast<size_t>(get_ggml_reordered_weight_cache().get_size());
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
