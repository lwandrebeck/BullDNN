/********************************************************************************
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

#include "lowoha_sdpa.hpp"
#include <sstream>
#include "bmm_sdpa/lowoha_sdpa_bmm.hpp"
#include "common/logging.hpp"
#include "common/platform_info.hpp"
#include "lowoha_operators/sdpa/flash_sdpa/lowoha_flash_sdpa.hpp"
#include "lowoha_operators/sdpa/reference/lowoha_sdpa_ref_kernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace sdpa {

status_t sdpa_direct(const void *query, const void *key, const void *value,
        const void *attn_mask, void *output, sdpa_params &params) {
    // Create profiler instance for timing
    zendnnl::profile::profiler_t profiler;
    bool is_profile = is_profile_enabled();
    if (is_profile) { profiler.tbp_start(); }
    // Log string built lazily -- only after computation when profiling
    const bool needs_log = apilog_info_enabled() || is_profile;

    sdpa_kernel_t kernel = kernel_select(params);
    status_t st = status_t::failure;

    switch (kernel) {
        case sdpa_kernel_t::flash:
            st = flash_sdpa(query, key, value, attn_mask, output, params);
            break;
        case sdpa_kernel_t::bmm:
            // Enable this when we have complete support for bmm-based SDPA implementation.
            // st = bmm_based_sdpa(query, key, value, attn_mask, output, params);
            log_error(
                    "sdpa_direct: bmm-based SDPA implementation is not "
                    "supported yet");
            return status_t::failure;
        case sdpa_kernel_t::reference:
            st = reference_sdpa(query, key, value, attn_mask, output, params);
            break;
        default:
            log_error("sdpa_direct: unsupported kernel ",
                    kernel_to_string(kernel));
            return status_t::failure;
    }

    if (st != status_t::success) { return st; }

    if (is_profile) { profiler.tbp_stop(); }

    if (needs_log) {
        std::ostringstream ss;
        const int64_t eff_kv
                = (params.kv_seq_len > 0) ? params.kv_seq_len : params.seq_len;
        const int64_t eff_kv_heads = (params.kv_num_heads > 0)
                ? params.kv_num_heads
                : params.num_heads;
        ss << "LOWOHA sdpa_direct: batch=" << params.batch
           << ", num_heads=" << params.num_heads
           << ", kv_num_heads=" << params.kv_num_heads
           << ", eff_kv_num_heads=" << eff_kv_heads
           << ", seq_len=" << params.seq_len << ", kv_seq_len=" << eff_kv
           << ", head_dim=" << params.head_dim << ", scale=" << params.scale
           << ", is_causal=" << (params.is_causal ? "true" : "false")
           << ", has_mask="
           << (attn_mask != nullptr && params.mask_ndims > 0 ? "true" : "false")
           << ", qkv_dt=" << dtype_info(params.qkv_dt)
           << ", mask_dt=" << dtype_info(params.mask_dt)
           << ", kernel=" << kernel_to_string(kernel);
        apilog_info(ss.str());
        if (is_profile) {
            profilelog_verbose(ss.str(), ", time=", profiler.tbp_elapsedtime(),
                    profiler.get_res_str());
        }
    }

    return status_t::success;
}

} // namespace sdpa
} // namespace lowoha
} // namespace zendnnl
