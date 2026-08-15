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

#ifndef MATMUL_NATIVE_INT8_EPILOGUE_128_HPP
#define MATMUL_NATIVE_INT8_EPILOGUE_128_HPP

// ============================================================================
// The destination side of the two per-group INT8 paths.
//
// Both loopers compute the same thing -- an fp32 product written into a
// row-major C -- and both adapters previously declined the same four cases
// around it: a bf16 destination, beta != 0, a bias, and any post-op. None of
// those are properties of the arithmetic, so handling them twice in two files
// would be two chances to get them differently wrong. They live here once.
//
// beta is not here: it is the one of the four the looper has to do itself,
// because the microkernel accumulates into C and beta decides what C starts at.
// This file only widens an existing bf16 destination into the scratch the looper
// will then scale.
// ============================================================================

#include <vector>

#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Working state between prepare() and finish(). Holds the fp32 scratch a bf16
/// destination needs and the widened bias, so both outlive the looper call.
struct Int8Epilogue {
    std::vector<float> c_scratch;
    std::vector<float> bias_f32;
    const float *bias_ptr = nullptr;
    float *C = nullptr; // what to hand the looper
    int ldc = 0;
    bool dst_is_bf16 = false;
    bool needs_finish = false; // a bias or a post-op is present
};

/// Decide whether this call's destination, bias and post-op chain can be
/// honoured, and set up whatever they need. Returns false -- having logged why
/// under @p tag -- for anything that cannot be, which is the caller's signal to
/// decline the whole call rather than approximate it.
bool int8_epilogue_prepare(Int8Epilogue &ep, const GemmDescriptor &desc,
        void *dst, const void *bias, const matmul_params &params,
        const char *tag);

/// Apply the bias and post-ops, then narrow into a bf16 destination if that is
/// what the caller asked for. Call after the looper returns, and only if it
/// returned true.
void int8_epilogue_finish(Int8Epilogue &ep, const GemmDescriptor &desc,
        void *dst, const matmul_params &params);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_EPILOGUE_128_HPP
