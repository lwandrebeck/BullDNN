/*******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Error-out stubs for the AOCL-DLP backend, compiled only when ZenDNNL is
// built without AOCL-DLP (ZENDNNL_DEPENDS_AOCLDLP=0). They satisfy the link
// dependencies of the always-compiled LOWOHA matmul dispatch (run_dlp,
// matmul_batch_gemm_wrapper, weight-cache helpers and reorder caching) while
// making it explicit at runtime that the AOCL-DLP path is unavailable.
//
// The selecting layers (e.g. lowoha::matmul::matmul_direct) already reject
// AOCL-DLP kernels up front and return status_t::unimplemented. The compute
// entry points below (run_dlp / matmul_batch_gemm_wrapper) are the final
// backstop for any residual fall-through path (e.g. a deeper native/onednn
// decline that mutates the kernel to aocl_dlp). They are void, so returning
// normally would leave the caller's output buffer uninitialized (a silent
// wrong result); instead they throw so an unsupported call fails loudly.

#include "common/zendnnl_exceptions.hpp"
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::error_handling;

void run_dlp(char, char, char, int, int, int, float, float, int, int, int, char,
        char, const void *, const void *, void *, const matmul_data_types &,
        const matmul_params &, const void *, zendnnl::ops::matmul_algo_t,
        bool) {
    apilog_error(
            "AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    EXCEPTION_WITH_LOC(
            "AOCL-DLP matmul kernel (run_dlp) invoked but ZenDNNL was "
            "built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void matmul_batch_gemm_wrapper(char, char, char, int, int, int, float,
        const void *, int, const void *, int, float, void *, int,
        matmul_data_types &, int, int, int, char, char, size_t, size_t, size_t,
        const matmul_params &, const void *, int) {
    apilog_error(
            "AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
            "invoked but ZenDNNL was built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
    EXCEPTION_WITH_LOC(
            "AOCL-DLP batch matmul kernel (matmul_batch_gemm_wrapper) "
            "invoked but ZenDNNL was built without AOCL-DLP support "
            "(ZENDNNL_DEPENDS_AOCLDLP=0).");
}

void clear_aocl_matmul_weight_caches() {
    // No AOCL weight caches exist in this build; nothing to clear.
}

// The three W4A8 entry points below are referenced by always-compiled code
// (ggml_weight_unpack.cpp, group_matmul_dispatch.cpp,
// group_matmul_n_tile.cpp), so without stubs a ZENDNNL_DEPENDS_AOCLDLP=0
// build fails to link.
//
// cvt_s4_to_s8 is no longer stubbed here. It was always backend-agnostic -- it
// only sign-extends s4 nibbles -- and stubbing it meant a build without AOCL-DLP
// threw when asked to unpack a GGML Q4_0 weight. It now lives in
// matmul/quantization/s4_upcast.cpp, compiled unconditionally, so this build
// configuration can unpack Q4_0 like any other.

// Void, and its callers rely on the plain-s8 LRU actually being populated:
// returning quietly would leave them pointing at unconverted s4 data, so
// fail loudly instead.
void w4a8_populate_plain_s8_cache(const std::vector<const void *> &,
        const std::vector<int> &, const std::vector<int> &,
        const std::vector<int> &, const std::vector<bool> &,
        const std::vector<matmul_params> &, int, std::vector<void *> &,
        bool &) {
    apilog_error(
            "W4A8 plain-s8 cache population invoked but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    EXCEPTION_WITH_LOC(
            "W4A8 plain-s8 cache population invoked but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
}

// Returns a status and every caller checks it, so decline gracefully here
// rather than throwing.
status_t broadcast_w4a8_src_scale(
        matmul_params &, int, std::vector<uint8_t> &) {
    apilog_error(
            "W4A8 source-scale broadcast requested but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    return status_t::unimplemented;
}

template <typename T>
bool reorderAndCacheWeights(Key_matmul, const void *, void *&, const int,
        const int, const int, const char, const char, char,
        get_reorder_buff_size_func_ptr, reorder_func_ptr<T>, int) {
    apilog_error(
            "AOCL-DLP weight reorder requested but ZenDNNL was built "
            "without AOCL-DLP support (ZENDNNL_DEPENDS_AOCLDLP=0).");
    return false;
}

template bool reorderAndCacheWeights<int16_t>(Key_matmul, const void *, void *&,
        const int, const int, const int, const char, const char, char,
        get_reorder_buff_size_func_ptr, reorder_func_ptr<int16_t>, int);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
