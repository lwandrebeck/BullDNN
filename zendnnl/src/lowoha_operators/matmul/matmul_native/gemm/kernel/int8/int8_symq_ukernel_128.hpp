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

#ifndef MATMUL_NATIVE_INT8_SYMQ_UKERNEL_128_HPP
#define MATMUL_NATIVE_INT8_SYMQ_UKERNEL_128_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// 128-bit symmetric per-group INT8 microkernel, for hosts without avx512vnni.
//
// This is the shape the GGML weight path produces and nothing here could
// consume: s8 weights with one scale per group of consecutive K, symmetric, no
// zero point. The existing INT8 microkernel is avx512vnni only and carries a
// per-tensor or per-channel weight scale, so neither its instruction set nor
// its scaling model fits.
//
// Symmetry is what makes this cheap on family 15h, which has no VNNI and no
// AVX2 before Excavator. A signed dot product is built from SSSE3's
// PMADDUBSW, which needs one unsigned operand:
//
//     dot(x_s8, y_s8) == maddubs(|x|, sign(y, x))
//
// PRECONDITION: all A and B bytes in [-127, 127]. -128 is excluded because
// PSIGNB cannot negate it and because a lane of two 128 * 128 products would
// saturate; both go wrong silently. Every GGML quantiser meets this (Q8_0
// clamps to +/-127, Q4_0 to [-8, 7], Q6_K to [-32, 31]). Within it a lane is
// bounded by 127 * 127 * 2 = 32258 against the signed 16-bit limit of 32767,
// so the arithmetic is exact. The asymmetric u8 * s8 case cannot make the same
// claim -- a lane there reaches 64770 -- which is what makes an exact
// asymmetric kernel expensive.
//
// Layout. B arrives packed as INT8 VNNI groups: four consecutive K for one
// column laid contiguously, NR columns per k-group. A is plain row-major s8, so
// four consecutive K of one row are one dword and broadcast in a single
// instruction.
//
// Scales. The s32 accumulator is flushed to fp32 every group_size K, because
// that is the span one weight scale covers. group_size must be a multiple of 4
// so a group boundary never falls inside a VNNI quad. The fp32 running sum
// lives in the caller's C tile rather than in registers: at MR*NR/4 registers
// for the s32 accumulators there is no room for a second set, and the flush is
// amortised over group_size K.
constexpr int SYMQ_MR = 4;
constexpr int SYMQ_NR = 8;
constexpr int SYMQ_VNNI_GRP = 4;

/// One MR x NR tile.
///
///   A          s8, row-major, a_stride bytes between rows
///   B_vnni     s8, VNNI-packed, b_stride bytes between k-groups
///   C          fp32, ldc floats between rows; accumulated into, not
///              overwritten, so the caller seeds it (beta handling is theirs)
///   k          number of K elements, must be a multiple of group_size
///   group_size K span of one weight scale; multiple of SYMQ_VNNI_GRP
///   wei_scale  one scale per (group, column): wei_scale[g * ws_stride + n]
///   src_scale  one scale for the whole tile (per-tensor activation scale)
using int8_symq_ukernel_128_fn_t = void (*)(const int8_t *__restrict__ A,
        int a_stride, const int8_t *__restrict__ B_vnni, int b_stride,
        float *__restrict__ C, int ldc, int k, int group_size,
        const float *__restrict__ wei_scale, int ws_stride, float src_scale);

/// Microkernel for this host, or nullptr where no 128-bit path is available.
/// Prefers the XOP flavour, which fuses the word-to-dword reduction and the
/// accumulate into one instruction; family 15h is the only silicon with XOP.
int8_symq_ukernel_128_fn_t select_int8_symq_ukernel_128();

/// Edge tiles of any mr_act x nr_act, and any k. Scalar and always correct.
void int8_symq_tail_128(const int8_t *__restrict__ A, int a_stride,
        const int8_t *__restrict__ B_vnni, int b_stride, float *__restrict__ C,
        int ldc, int k, int group_size, int mr_act, int nr_act,
        const float *__restrict__ wei_scale, int ws_stride, float src_scale);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_SYMQ_UKERNEL_128_HPP
