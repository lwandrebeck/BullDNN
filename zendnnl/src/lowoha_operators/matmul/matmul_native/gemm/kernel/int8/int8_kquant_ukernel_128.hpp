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

#ifndef MATMUL_NATIVE_INT8_KQUANT_UKERNEL_128_HPP
#define MATMUL_NATIVE_INT8_KQUANT_UKERNEL_128_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// 128-bit microkernel for the asymmetric per-group weights GGML's k-quants
// produce -- Q4_K and Q5_K -- against a symmetrically quantised s8 activation.
//
// THE SHAPE. A k-quant weight dequantises as
//
//     w[k, n] = D[j, n] * q[k, n] - M[j, n]
//
// over sub-blocks j of consecutive K, with q UNSIGNED (0..15 for Q4_K, 0..31 for
// Q5_K) and D, M real-valued per (sub-block, column). Q4_0, Q8_0 and Q6_K have no
// M term and are handled by the symmetric kernel next door; Q4_K and Q5_K are why
// that kernel could not take them.
//
// Substituting into a GEMM and splitting the sum gives the two pieces this kernel
// computes:
//
//     C[m, n] = sum_j D[j, n] * (sum_{k in j} a[m, k] * q[k, n])
//             - sum_j M[j, n] * (sum_{k in j} a[m, k])
//
// The first is an ordinary per-group dot product. The second involves no weight
// at all: it is the activation's row sum over the sub-block, one scalar per (row,
// group), scaled by M. So the correction costs a vector multiply per group rather
// than any per-element work -- which is what makes this expressible here at all,
// given matmul_native has no weight zero-point.
//
// Row sums are the caller's to supply. They do not depend on n, so computing them
// in here would repeat the same additions once per column tile; the looper builds
// them once per call instead.
//
// WHY THIS IS CHEAPER THAN THE SYMMETRIC KERNEL. PMADDUBSW wants its first
// operand unsigned and its second signed, which is exactly what a k-quant offers:
// unsigned codes against a signed activation. The symmetric kernel has signed
// weights and has to manufacture an unsigned operand with abs() and re-apply the
// sign with PSIGNB -- four instructions per row and column half that vanish here.
// Per VNNI quad and eight columns:
//
//   2 loads    B, sixteen bytes each holding four columns x four K
//   4 bcast    A, one dword per row
//   8 maddubs  unsigned codes against signed activations
//   8 reduce   words to dwords, accumulated (VPMADCSWD under XOP, else PMADDWD)
//
// Twenty-two instructions against the symmetric kernel's thirty-four, for the
// same tile.
//
// SATURATION. A PMADDUBSW lane sums two products of an unsigned code and a signed
// activation: at most 2 * 31 * 128 = 7936 in magnitude, against the signed 16-bit
// limit of 32767. So unlike the symmetric kernel this one has no byte contract to
// enforce -- every s8 activation is admissible, -128 included -- provided the
// codes really are k-quant codes, which bounds them at 31. A caller handing over
// larger unsigned codes would be outside the contract: at q <= 255 a lane reaches
// 65280 and saturates silently.
constexpr int KQ_MR = 4;
constexpr int KQ_NR = 8;
constexpr int KQ_VNNI_GRP = 4;

/// One MR x NR tile.
///
///   A          s8, row-major, a_stride bytes between rows
///   B_vnni     u8 codes, VNNI-packed, b_stride bytes between k-groups
///   C          fp32, ldc floats between rows; accumulated into, not overwritten
///   k          number of K elements, a multiple of group_size
///   group_size K span of one (D, M) pair; a multiple of KQ_VNNI_GRP
///   wei_scale  D, indexed wei_scale[g * ws_stride + n]
///   wei_min    M, indexed the same way
///   row_sums   sum of A over each group: row_sums[m * rs_stride + g]
///   src_scale  activation scale, src_scale[m * ss_row + g * ss_grp]; see the
///              symmetric kernel's header for the three granularities
using int8_kquant_ukernel_128_fn_t = void (*)(const int8_t *__restrict__ A,
        int a_stride, const uint8_t *__restrict__ B_vnni, int b_stride,
        float *__restrict__ C, int ldc, int k, int group_size,
        const float *__restrict__ wei_scale, const float *__restrict__ wei_min,
        int ws_stride, const int32_t *__restrict__ row_sums, int rs_stride,
        const float *__restrict__ src_scale, int ss_row, int ss_grp);

/// Microkernel for this host, or nullptr where no 128-bit path is available.
/// Prefers XOP, which fuses the word-to-dword reduction; family 15h only.
int8_kquant_ukernel_128_fn_t select_int8_kquant_ukernel_128();

/// Single-row form, for the decode shape. See the symmetric kernel's equivalent:
/// the general tile reads MR rows unconditionally, so at M=1 it would compute
/// four rows to keep one.
int8_kquant_ukernel_128_fn_t select_int8_kquant_ukernel_128_m1();

/// Edge tiles of any mr_act x nr_act, and any k. Scalar and always correct.
void int8_kquant_tail_128(const int8_t *__restrict__ A, int a_stride,
        const uint8_t *__restrict__ B_vnni, int b_stride, float *__restrict__ C,
        int ldc, int k, int group_size, int mr_act, int nr_act,
        const float *__restrict__ wei_scale, const float *__restrict__ wei_min,
        int ws_stride, const int32_t *__restrict__ row_sums, int rs_stride,
        const float *__restrict__ src_scale, int ss_row, int ss_grp);

/// Row sums of A over each group: out[m * n_groups + g] = sum of A[m, k] for k in
/// group g. The correction term needs one per (row, group) and they do not depend
/// on the column, so they are built once per call rather than per column tile.
void int8_kquant_row_sums(const int8_t *__restrict__ A, int a_stride, int M,
        int K, int group_size, int32_t *__restrict__ out);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_KQUANT_UKERNEL_128_HPP
