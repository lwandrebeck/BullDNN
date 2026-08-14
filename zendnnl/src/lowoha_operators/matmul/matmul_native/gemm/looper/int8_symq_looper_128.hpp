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

#ifndef MATMUL_NATIVE_INT8_SYMQ_LOOPER_128_HPP
#define MATMUL_NATIVE_INT8_SYMQ_LOOPER_128_HPP

#include <cstdint>

#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Symmetric per-group INT8 GEMM for hosts without avx512vnni.
///
/// C[M,N] = src_scale * sum_g wei_scale[g,n] * dot(A[m, g], B[g, n])
///
/// where g runs over groups of group_size consecutive K, which is the shape the
/// GGML weight path produces. On family 15h this is the only route to INT8 at
/// all: every INT8 algorithm in the tree resolves to AOCL-DLP, which declines
/// the work outright on a processor without AVX-512 VNNI, so there is no
/// incumbent path to improve on -- only a non-functional one to replace.
///
/// src_scale carries the activation scale at whichever granularity the caller
/// has: ss_row and ss_grp index it as src_scale[m * ss_row + g * ss_grp], so
/// {0,0} is per-tensor, {1,0} per-token, and {G,1} per-group. See the microkernel
/// header; the kernel flushes once per (row, group), so none of the three costs
/// more than the others.
///
/// A is s8 row-major M x K with leading dimension lda.
/// B is s8; transB selects N x K (the GGML layout, ldb spanning K) over K x N.
/// wei_scale is group-major: wei_scale[g * N + n], matching what the GGML
/// unpack writes, so no rearrangement is needed between them.
/// C is fp32 M x N, overwritten (this path has no beta).
///
/// Returns false without touching C when the shape is outside what the
/// microkernel can express -- K not a whole number of groups, or a group size
/// that is not a multiple of the VNNI quad -- so the caller can fall back.
/// All A and B bytes must lie in [-127, 127]; see the microkernel header for
/// why -128 is excluded and why every GGML quantiser satisfies this.
/// prepacked, when given, is B already in the microkernel's VNNI panel layout,
/// from the shared INT8 prepacked-weight cache. The bytes are identical to what
/// this looper's own packer produces, so the result is unchanged rather than
/// merely close, and the per-call pass over B disappears. Pass nullptr to pack
/// per call, which is always correct.
///
/// Deciding whether caching is permitted is the caller's job, not this one's:
/// it depends on the framework's is_weights_const promise, which this layer
/// cannot see.
bool int8_symq_execute_128(int M, int N, int K, int group_size,
        const int8_t *A, int lda, const int8_t *B, int ldb, bool transB,
        float *C, int ldc, const float *wei_scale, const float *src_scale,
        int ss_row, int ss_grp, int nthreads,
        const INT8PrepackedWeight *prepacked = nullptr);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_SYMQ_LOOPER_128_HPP
