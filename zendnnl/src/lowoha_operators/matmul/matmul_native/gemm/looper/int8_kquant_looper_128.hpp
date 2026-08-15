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

#ifndef MATMUL_NATIVE_INT8_KQUANT_LOOPER_128_HPP
#define MATMUL_NATIVE_INT8_KQUANT_LOOPER_128_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Asymmetric per-group INT8 GEMM for hosts without avx512vnni -- the shape
/// GGML's Q4_K and Q5_K produce.
///
/// C[M,N] = sum_g src_scale * ( wei_scale[g,n] * dot(A[m,g], q[g,n])
///                            - wei_min[g,n]   * rowsum(A[m,g]) )
///
/// A is s8 row-major M x K with leading dimension lda; every s8 value is
/// admissible here, unlike the symmetric path.
/// B is the UNSIGNED code array; transB selects N x K (the GGML layout) over
/// K x N. Codes must be k-quant codes, so at most 31 -- see the microkernel
/// header for the saturation bound that rests on.
/// wei_scale is D and wei_min is M, both group-major over the full N:
/// [g * N + n], which is what the GGML unpack writes.
///
/// The two may live in one allocation or two; this looper does not care, and a
/// test asserts the answers are bit-identical either way. The caller should care,
/// because it was measured: putting M immediately after D in one buffer places
/// the two streams a power of two apart for every realistic shape (2 MB at
/// N=4096, 128 KB at 1024^3), and on Piledriver -- 4-way L1d -- that costs 20%.
/// Displacing M by one cache line makes a single allocation the fastest of the
/// three arrangements tried, ahead of two separate ones. Excavator, with an
/// 8-way 32 KB L1d, shows none of this. See docs/perf.
/// src_scale is indexed src_scale[m * ss_row + g * ss_grp]: {0,0} per-tensor,
/// {1,0} per-token, {G,1} per-group.
/// C is fp32 M x N. beta scales what is already there:
/// C = beta * C + the product. beta = 0 overwrites, which is the common case
/// and the only one that avoids reading C at all.
///
/// Returns false without touching C when the shape is outside what the
/// microkernel expresses -- K not a whole number of groups, or a group size that
/// is not a multiple of the VNNI quad -- so the caller can fall back.
bool int8_kquant_execute_128(int M, int N, int K, int group_size,
        const int8_t *A, int lda, const uint8_t *B, int ldb, bool transB,
        float *C, int ldc, const float *wei_scale, const float *wei_min,
        const float *src_scale, int ss_row, int ss_grp, int nthreads,
        float beta = 0.0f);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_KQUANT_LOOPER_128_HPP
