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

// Moved here from the AOCL backend rather than copied. The stub that used to
// stand in for it when AOCL-DLP was compiled out said as much itself: the
// routine is backend-agnostic, and a second copy of weight-unpacking logic could
// drift from the original and silently change quantized weights. There is still
// exactly one definition; only its address changed.

#include "lowoha_operators/matmul/quantization/s4_upcast.hpp"

#include <cstddef>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

// Low nibble first, then high; s4 sign-extends through bit 3.
inline int8_t extract_s4_nibble(int8_t packed_byte, bool is_low_nibble) {
    const uint8_t ubyte = static_cast<uint8_t>(packed_byte);
    int8_t value = is_low_nibble ? (ubyte & 0x0F) : ((ubyte >> 4) & 0x0F);
    if (value & 0x08) { value |= 0xF0; }
    return value;
}

} // namespace

void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n, int ldb,
        bool is_transposed) {
#pragma omp parallel for collapse(2)
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            // Packed s4 index (ab: row*ldb+col; ba: col*ldb+row).
            std::size_t physical_idx = is_transposed
                    ? (static_cast<std::size_t>(col) * ldb + row)
                    : (static_cast<std::size_t>(row) * ldb + col);
            std::size_t packed_byte_idx = physical_idx / 2;
            bool is_low_nibble = (physical_idx % 2) == 0;

            int8_t s8_value = extract_s4_nibble(
                    weights[packed_byte_idx], is_low_nibble);
            std::size_t out_idx = static_cast<std::size_t>(row) * n + col;
            wei_s8[out_idx] = s8_value;
        }
    }
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
