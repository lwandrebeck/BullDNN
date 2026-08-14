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

#ifndef MATMUL_QUANTIZATION_S4_UPCAST_HPP
#define MATMUL_QUANTIZATION_S4_UPCAST_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {

/// Widen packed s4 nibbles to s8, sign-extending. No dequantization.
///
/// Shared by the W4A8 AOCL sym-quant reorder and the GGML Q4_0 unpack, which is
/// why it lives here rather than in the AOCL backend. It used to live there, and
/// a build without AOCL-DLP got a stub that threw -- so Q4_0, the most common
/// legacy GGUF quant, could not be unpacked at all in a build configured the
/// obvious way for family 15h, where AOCL-DLP declines every INT8 kernel anyway
/// and is reasonably compiled out. Nothing in the routine is backend-specific:
/// it shifts, masks and sign-extends.
///
/// @param weights       Packed s4 source, two nibbles per byte.
/// @param wei_s8        Destination, written k x n row-major (non-transposed).
/// @param k             Logical row count of the output.
/// @param n             Logical column count of the output.
/// @param ldb           Leading dimension of the packed source, in elements.
/// @param is_transposed True when the packed source is column-major (ba).
void cvt_s4_to_s8(const int8_t *weights, int8_t *wei_s8, int k, int n, int ldb,
        bool is_transposed);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_QUANTIZATION_S4_UPCAST_HPP
