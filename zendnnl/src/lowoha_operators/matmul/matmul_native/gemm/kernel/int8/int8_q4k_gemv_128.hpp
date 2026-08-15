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

#ifndef MATMUL_NATIVE_INT8_Q4K_GEMV_128_HPP
#define MATMUL_NATIVE_INT8_Q4K_GEMV_128_HPP

// ============================================================================
// Decode-shape GEMV straight out of the GGML k-quant blocks.
//
// WHY A SECOND KERNEL RATHER THAN A FASTER LOOPER. The GEMM path unpacks a
// k-quant weight to one byte per weight and packs that into panels. Both are
// right when there are many rows of activations to amortise them over, and both
// are pure loss at one row: measured against ggml on a 4096x14336 FFN, the GEMM
// path ran at a tenth of ggml's rate at one token before the prepacked cache and
// about half after it. What remains after the packing is gone is BANDWIDTH --
// the unpacked form is 1 byte per weight against GGML's 4.5 bits, so the GEMM
// path reads roughly twice what ggml does, and at one token the whole weight is
// read exactly once and never reused. No amount of caching fixes reading twice
// as much.
//
// So this kernel never materialises the unpacked weight at all. It reads the
// packed blocks, decodes nibbles into registers, and multiplies them there. The
// bytes it touches are exactly the bytes GGML stores, which is the floor.
//
// THE ARITHMETIC is the same rearrangement the k-quant looper uses -- w = D*q - M
// with the min term folded into a row sum -- so per sub-block of 32:
//
//   contribution = src_scale * ( d*sc * dot(a, q) - dmin*m * sum(a) )
//
// with q unsigned 0..15 and a signed. That is PMADDUBSW's operand order exactly,
// and the lane bound is comfortable: 15*127*2 = 3810 against 32767, so unlike
// the symmetric kernel there is no saturation hazard and no sign trick.
//
// WHAT IT DOES NOT DO. One row of activations only. Above that the GEMM path
// wins, because there the packing is amortised and the panels are reused; this
// kernel re-reads the weight for every row it is given. The caller picks.
// ============================================================================

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// GGML super-block span, and the sub-block a (D, M) pair covers.
constexpr int Q4K_SUPER = 256;
constexpr int Q4K_SUB = 32;

/// C[0, n] = sum over sub-blocks of ss * ( d*sc * dot(a, q) - dmin*m * sum(a) ),
/// read directly from @p blocks in GGML's packed layout.
///
/// @param N        weight rows, i.e. output columns
/// @param K        reduction extent; a multiple of Q4K_SUPER
/// @param ggml_type 12 for Q4_K, 13 for Q5_K
/// @param A        s8 activations, one row of K
/// @param blocks   N rows of K/256 packed super-blocks, row-major
/// @param C        f32 destination, N wide
/// @param src_scale activation scales; ss_grp selects per-tensor or per-group
/// @param ss_grp   0 when one scale covers everything, 1 when there is one per
///                 32-wide group (the layout GGML's q8_K activations produce)
/// @returns false when the shape or type is not one this kernel expresses, in
///          which case C is untouched and the caller must fall back.
bool int8_q4k_gemv_128(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads);

/// True when int8_q4k_gemv_128 would accept this shape. Cheap; no allocation.
bool int8_q4k_gemv_supported(int M, int N, int K, int ggml_type);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_Q4K_GEMV_128_HPP
