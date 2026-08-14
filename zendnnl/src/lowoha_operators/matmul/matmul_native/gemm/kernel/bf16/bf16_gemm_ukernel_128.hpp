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

#ifndef MATMUL_NATIVE_BF16_GEMM_UKERNEL_128_HPP
#define MATMUL_NATIVE_BF16_GEMM_UKERNEL_128_HPP

#include <cstdint>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// 128-bit BF16 GEMM microkernels for hosts without avx512bf16 -- AMD family
// 15h in particular, which has no BF16 dot product at all.
//
// There being no BF16 multiply-accumulate on such a host, the arithmetic is
// FP32: B is widened from BF16 in register and fed to ordinary FMAs. The gain
// over an FP32 GEMM is therefore not in the maths but in the traffic, since B
// is the operand streamed repeatedly across M tiles and stays half size in
// cache and memory.
//
// A arrives already widened to FP32, which the caller does once per packed
// panel. Widening it here instead would cost three extra operations per row
// per k to rebuild a value used by a single FMA -- the A broadcast has to be
// as cheap as it is in the FP32 kernel or the ratio of arithmetic to overhead
// collapses. B is widened in register precisely because it must not be
// expanded in memory: that would give back the traffic the format buys.
//
// B keeps the VNNI layout the avx512bf16 kernels use, so both paths consume
// the same packed weights. In that layout the dword for column n of k-pair kp
// holds k=2kp in its low half and k=2kp+1 in its high half, which makes each
// widening a single shift or mask.
using bf16_ukernel_128_fn_t = void (*)(const float *__restrict__ A_f32,
        int a_stride, const uint16_t *__restrict__ B_vnni, int b_stride,
        float *__restrict__ C, int ldc, int k, float beta,
        const float *__restrict__ bias, fused_postop_t fused_op,
        uint16_t *__restrict__ C_bf16, int ldc_bf16);

// Microkernel for the given MR and NR, or nullptr when this build has no
// 128-bit path or the shape is not one of the instantiated ones. Callers must
// treat nullptr as "use the tail kernel".
bf16_ukernel_128_fn_t select_bf16_ukernel_128(int MR, int NR);

// Edge tiles with arbitrary mr_act and nr_act. Scalar, and correct for any
// shape including odd k.
void bf16_tail_kernel_128(const float *__restrict__ A_f32, int a_stride,
        const uint16_t *__restrict__ B_vnni, int b_stride,
        float *__restrict__ C, int ldc, int k, int mr_act, int nr_act,
        float beta, const float *__restrict__ bias, fused_postop_t fused_op,
        uint16_t *__restrict__ C_bf16, int ldc_bf16);

// Widen a BF16 panel to FP32. Used on A before the microkernel sees it.
void widen_bf16_panel_to_fp32(const uint16_t *__restrict__ src, int src_stride,
        float *__restrict__ dst, int dst_stride, int rows, int cols);

// Bias plus one fused activation over an mr_count x nr_count tile of FP32 C.
// Defined alongside the FP32 microkernels; declared here so the BF16 looper can
// finish a ragged tile that the microkernel could not fuse for itself.
void apply_bias_and_postop_tile(float *C, int ldc, int mr_count, int nr_count,
        const float *bias, fused_postop_t fused_op);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BF16_GEMM_UKERNEL_128_HPP
