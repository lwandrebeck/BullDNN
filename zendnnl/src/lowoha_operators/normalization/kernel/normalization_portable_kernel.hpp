/********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

#ifndef _NORMALIZATION_PORTABLE_KERNEL_HPP
#define _NORMALIZATION_PORTABLE_KERNEL_HPP

#include "lowoha_operators/normalization/lowoha_normalization_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

// ---------------------------------------------------------------------------
// Can this kernel serve the given problem?
//
// True for RMSNorm, LayerNorm and the fused residual-add RMSNorm with f32 or bf16
// source and destination, when the translation unit was compiled for a target
// providing AVX. gamma and beta may be any of f32/bf16/f16 -- they are widened
// once per call.
//
// False for BatchNorm, for f16 source/destination, and in builds whose target ISA
// has no AVX. Those keep using the reference kernel.
// ---------------------------------------------------------------------------
bool normalization_portable_supported(const norm_params &params);

// ---------------------------------------------------------------------------
// 128-bit vector RMSNorm / LayerNorm / fused-add RMSNorm for hosts without AVX-512.
//
// @param input       Source tensor, element type params.src_dt
// @param output      Destination tensor, element type params.dst_dt
// @param residual    FUSED_ADD_RMS_NORM only: updated in place with
//                    residual[i] += input[i] before normalising. Element type
//                    follows params.src_dt. May be nullptr for the other norms.
// @param gamma       Scale, or nullptr when !use_scale. Any of f32/bf16/f16.
// @param beta        Shift, or nullptr. LayerNorm only; ignored by RMSNorm.
// @param params      Normalization parameters
// @param num_threads Thread count for the batch loop
//
// @return status_t::success, or status_t::unimplemented if this kernel does not
//         serve the problem (callers should then fall through to the reference
//         kernel).
// ---------------------------------------------------------------------------
status_t normalization_portable(const void *input, void *output, void *residual,
        const void *gamma, const void *beta, norm_params &params,
        int num_threads);

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

#endif // _NORMALIZATION_PORTABLE_KERNEL_HPP
