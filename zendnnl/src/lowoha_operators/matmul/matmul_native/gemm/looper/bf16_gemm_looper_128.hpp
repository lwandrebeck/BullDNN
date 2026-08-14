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

#ifndef MATMUL_NATIVE_BF16_GEMM_LOOPER_128_HPP
#define MATMUL_NATIVE_BF16_GEMM_LOOPER_128_HPP

#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// BF16 GEMM for hosts without avx512bf16, family 15h in particular.
//
// Must be called instead of bf16_gemm_execute() on such a host, not after it:
// that function lives in a translation unit compiled wholesale for AVX-512, so
// entering it at all is unsafe where those features are missing.
//
// Returns false without touching dst when the planner picks a shape this path
// has no microkernel for, so the caller can fall back.
bool bf16_gemm_execute_128(const GemmDescriptor &desc, const UarchParams &uarch,
        const void *src, const void *weight, void *dst, const void *bias,
        matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BF16_GEMM_LOOPER_128_HPP
