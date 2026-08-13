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

#ifndef MATMUL_NATIVE_FP32_GEMM_UKERNEL_HPP
#define MATMUL_NATIVE_FP32_GEMM_UKERNEL_HPP

#include <cstdint>
#include "lowoha_operators/matmul/matmul_native/common/avx512_math.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Function pointer type for FP32 GEMM microkernel dispatch.
using ukernel_fn_t = void (*)(const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, float beta, const float *__restrict__ bias,
        fused_postop_t fused_op);

// Hand-scheduled 6x64 asm microkernel (peak throughput).
__attribute__((target("avx512f,fma"))) void avx512_ukernel_6x64_asm(
        const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, float beta, const float *__restrict__ bias,
        fused_postop_t fused_op);

// Tail microkernel for edge tiles (dynamic MR/NR, masked).
__attribute__((target("avx512f,avx512bw,fma"))) void avx512_tail_kernel(
        const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, int mr_act, int nr_act, float beta,
        const float *__restrict__ bias, fused_postop_t fused_op);

// Scalar fallback for very small tiles.
void scalar_microkernel(const float *__restrict__ pa, int a_stride,
        const float *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, int mr_act, int nr_act, float beta,
        const float *__restrict__ bias, fused_postop_t fused_op);

// Bias and fused activation over an already-accumulated tile. Shared by
// scalar_microkernel() and the 128-bit microkernel so the epilogue arithmetic
// exists in exactly one place.
void apply_bias_and_postop_tile(float *C, int ldc, int mr_count, int nr_count,
        const float *bias, fused_postop_t fused_op);

// Select best microkernel for given MR and NR.
__attribute__((target("avx512f,fma"))) ukernel_fn_t select_ukernel(
        int MR, int NR);

// Select the 128-bit microkernel for hosts without AVX-512, or nullptr when
// this build has no 128-bit path compiled in.
//
// Family 15h has no AVX-512, so select_ukernel() above yields nothing usable and
// every full tile used to fall through to scalar_microkernel(). This is the
// replacement for those hosts, built on FMA4 (or FMA3 where the target has it)
// and 128-bit AVX. It is compiled only when the translation unit's target ISA
// provides AVX, which for BullDNN means a -march=bdverN build; a plain
// x86-64 build gets nullptr and the unchanged scalar path.
ukernel_fn_t select_ukernel_128(int MR, int NR);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_FP32_GEMM_UKERNEL_HPP
