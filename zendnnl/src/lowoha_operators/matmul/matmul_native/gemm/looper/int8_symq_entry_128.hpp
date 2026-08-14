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

#ifndef MATMUL_NATIVE_INT8_SYMQ_ENTRY_128_HPP
#define MATMUL_NATIVE_INT8_SYMQ_ENTRY_128_HPP

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// True when this call *looks like* the symmetric per-group INT8 shape, judged
/// on dtypes and quantisation granularity alone. Cheap enough for a dispatch
/// gate; int8_symq_try_execute_128() re-checks everything and is the authority.
///
/// Kept separate so the dispatcher can tell "this is per-group INT8, which only
/// this path implements on a host without VNNI" from "this is INT8 the ordinary
/// native path would have taken".
bool is_int8_symq_candidate(const matmul_params &params, int K, int N);

/// Run the call on the 128-bit symmetric per-group INT8 GEMM, or decline.
///
/// Returns false, having written nothing to dst, whenever any part of the call
/// falls outside what the kernel expresses -- an unsupported dtype or
/// granularity, a non-zero beta, a bias or post-op it cannot apply, or operand
/// bytes outside the kernel's [-127, 127] contract. The caller then falls back
/// exactly as it would have without this path.
///
/// This is the only route to INT8 on family 15h: every INT8 algorithm in the
/// tree resolves to AOCL-DLP, which declines without AVX-512 VNNI and returns
/// without computing.
bool int8_symq_try_execute_128(const GemmDescriptor &desc, const void *src,
        const void *weight, void *dst, const void *bias,
        const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_SYMQ_ENTRY_128_HPP
