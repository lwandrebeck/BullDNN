/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/*
 * Portable reference kernels for grouped dynamic quantization (the MoE
 * path), mirroring the AVX-512 _native kernels in
 * dynamic_quant_impl/. They exist so hosts without AVX-512 — AMD family 15h
 * among them — can run grouped dynamic quantization instead of failing the
 * operator: the grouped path had no non-AVX-512 implementation at all, unlike
 * the plain per-token path, which already falls back to
 * dynamic_per_token_ref_kernel.cpp.
 *
 * The arithmetic deliberately mirrors dynamic_per_token_ref_kernel.cpp: skip
 * non-finite inputs when reducing, clamp through
 * compute_symmetric_scale_from_absmax(), quantize with std::nearbyint(), and
 * emit 0 for non-finite elements.
 */

#include "lowoha_operators/reorder/reorder_data_type/scalar_impl/scalar_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"

namespace zendnnl {
namespace lowoha {
namespace reorder {

namespace {

// Symmetric s8 scale from a row/group absolute maximum. Kept identical to the
// helper used by the AVX-512 and per-token reference kernels.
inline void group_ref_symmetric_scale(float absmax, float &scale) {
    if (absmax < 1e-10f) absmax = 1e-10f;
    scale = absmax / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
}

// Element readers, one per supported source dtype.
struct read_bf16 {
    static inline float load(const void *base, int64_t idx) {
        return common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(
                static_cast<const uint16_t *>(base)[idx]));
    }
};

struct read_f32 {
    static inline float load(const void *base, int64_t idx) {
        return static_cast<const float *>(base)[idx];
    }
};

struct read_f16 {
    static inline float load(const void *base, int64_t idx) {
        return common::float16_t::f16_to_f32_val(
                static_cast<const uint16_t *>(base)[idx]);
    }
};

// Flattened (expert, row) coordinate so one loop can be shared by OpenMP
// across sum(M_i), matching how the _native kernels schedule.
struct row_coord {
    size_t op;
    int64_t row;
};

inline std::vector<row_coord> build_row_map(const std::vector<int> &M) {
    std::vector<row_coord> rows;
    int64_t total = 0;
    for (int m : M) total += std::max(0, m);
    rows.reserve(static_cast<size_t>(total));
    for (size_t i = 0; i < M.size(); ++i) {
        for (int64_t m = 0; m < std::max(0, M[i]); ++m) {
            rows.push_back(row_coord {i, m});
        }
    }
    return rows;
}

// Quantize one contiguous span of `len` elements: absmax, scale, then round.
template <typename Reader>
inline void quantize_span(const void *src_base, int64_t src_off, int8_t *dst,
        int64_t len, float &scale_out) {
    float absmax = 0.0f;
    for (int64_t j = 0; j < len; ++j) {
        const float v = Reader::load(src_base, src_off + j);
        if (std::isfinite(v)) absmax = std::max(absmax, std::abs(v));
    }
    float scale;
    group_ref_symmetric_scale(absmax, scale);
    scale_out = scale;

    for (int64_t j = 0; j < len; ++j) {
        const float v = Reader::load(src_base, src_off + j);
        if (!std::isfinite(v)) {
            dst[j] = 0;
            continue;
        }
        dst[j] = static_cast<int8_t>(
                static_cast<int32_t>(std::nearbyint(v / scale)));
    }
}

// One scale per row: sources are [M_i, K_i] with row stride lda[i].
template <typename Reader>
void group_per_token_ref(const std::vector<const void *> &src,
        const std::vector<int> &M, const std::vector<int> &K,
        const std::vector<int> &lda, const std::vector<void *> &dst,
        const std::vector<int> &dst_lda, const std::vector<float *> &scales,
        int num_threads) {
    const std::vector<row_coord> rows = build_row_map(M);
    const int64_t n_rows = static_cast<int64_t>(rows.size());

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int64_t r = 0; r < n_rows; ++r) {
        const size_t i = rows[static_cast<size_t>(r)].op;
        const int64_t m = rows[static_cast<size_t>(r)].row;
        const int64_t k = K[i];
        int8_t *row_dst = static_cast<int8_t *>(dst[i]) + m * dst_lda[i];
        quantize_span<Reader>(
                src[i], m * lda[i], row_dst, k, scales[i][m]);
    }
}

// G scales per row: group_size = K_i / G, scale buffer indexed m * G + g.
template <typename Reader>
void group_per_group_ref(const std::vector<const void *> &src,
        const std::vector<int> &M, const std::vector<int> &K,
        const std::vector<int> &lda, const std::vector<void *> &dst,
        const std::vector<int> &dst_lda, const std::vector<float *> &scales,
        int64_t G, int num_threads) {
    const std::vector<row_coord> rows = build_row_map(M);
    const int64_t n_rows = static_cast<int64_t>(rows.size());

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int64_t r = 0; r < n_rows; ++r) {
        const size_t i = rows[static_cast<size_t>(r)].op;
        const int64_t m = rows[static_cast<size_t>(r)].row;
        const int64_t group_size = K[i] / G;
        int8_t *row_dst = static_cast<int8_t *>(dst[i]) + m * dst_lda[i];
        for (int64_t g = 0; g < G; ++g) {
            quantize_span<Reader>(src[i], m * lda[i] + g * group_size,
                    row_dst + g * group_size, group_size,
                    scales[i][m * G + g]);
        }
    }
}

} // namespace

void dynamic_per_token_group_quant_bf16_s8_ref(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int num_threads) {
    group_per_token_ref<read_bf16>(
            src, M, K, lda, dst, dst_lda, scales, num_threads);
}

void dynamic_per_token_group_quant_f32_s8_ref(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int num_threads) {
    group_per_token_ref<read_f32>(
            src, M, K, lda, dst, dst_lda, scales, num_threads);
}

void dynamic_per_token_group_quant_f16_s8_ref(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int num_threads) {
    group_per_token_ref<read_f16>(
            src, M, K, lda, dst, dst_lda, scales, num_threads);
}

void dynamic_per_group_group_quant_bf16_s8_ref(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int64_t G, int num_threads) {
    group_per_group_ref<read_bf16>(
            src, M, K, lda, dst, dst_lda, scales, G, num_threads);
}

void dynamic_per_group_group_quant_f32_s8_ref(
        const std::vector<const void *> &src, const std::vector<int> &M,
        const std::vector<int> &K, const std::vector<int> &lda,
        const std::vector<void *> &dst, const std::vector<int> &dst_lda,
        const std::vector<float *> &scales, int64_t G, int num_threads) {
    group_per_group_ref<read_f32>(
            src, M, K, lda, dst, dst_lda, scales, G, num_threads);
}

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
