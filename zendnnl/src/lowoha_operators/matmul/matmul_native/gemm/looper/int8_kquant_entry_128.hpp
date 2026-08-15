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

#ifndef MATMUL_NATIVE_INT8_KQUANT_ENTRY_128_HPP
#define MATMUL_NATIVE_INT8_KQUANT_ENTRY_128_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// True when this call is a decoded GGML k-quant: unsigned codes, a {2G, N}
/// weight scale region holding D and M, and a plain row-major weight.
///
/// The u8 weight dtype is the discriminator, and it is doing real work. A
/// {2G, N} scale also reads as a valid *symmetric* per-group scale of group size
/// K/(2G), so without something to tell the two apart a k-quant would be accepted
/// by the symmetric path and computed with its min term silently dropped.
/// is_int8_symq_candidate() requires s8 and so refuses these; this requires u8 and
/// so refuses those. The two predicates are deliberately disjoint.
bool is_int8_kquant_candidate(const matmul_params &params, int K, int N);

/// Run the call on the 128-bit asymmetric per-group INT8 GEMM, or decline.
///
/// Returns false, having written nothing to dst, whenever any part of the call is
/// outside what the kernel expresses. On family 15h a decline means the work is
/// not done at all -- AOCL-DLP refuses every INT8 kernel without AVX-512 VNNI --
/// so declining costs nothing that was working, while a wrong acceptance is the
/// only way this path can do harm.
bool int8_kquant_try_execute_128(const GemmDescriptor &desc, const void *src,
        const void *weight, void *dst, const void *bias,
        const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_KQUANT_ENTRY_128_HPP
