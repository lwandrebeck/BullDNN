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

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_epilogue_128.hpp"

#include <cstdint>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"
#include "operators/common/post_op.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;

namespace {

// apply_postops_tile() ends both of its switches in `default: break;`. That is
// the right shape for a tile epilogue -- it is called per tile and cannot
// usefully fail -- but it means an unimplemented post-op is dropped without a
// word, and a dropped activation is a wrong answer that looks like a right one.
//
// So the accept list is explicit and enumerated rather than "anything but
// none", and it is exactly what the two appliers implement. softmax, pooling
// and mish are the three the enum has and the appliers do not; they are what
// this exists to catch. A post-op added to the enum later lands outside the list
// and is declined, which is the safe direction to be wrong in.
bool postop_is_applied(post_op_type_t t) {
    switch (t) {
        case post_op_type_t::none:
        case post_op_type_t::elu:
        case post_op_type_t::relu:
        case post_op_type_t::leaky_relu:
        case post_op_type_t::gelu_tanh:
        case post_op_type_t::gelu_erf:
        case post_op_type_t::sigmoid:
        case post_op_type_t::swish:
        case post_op_type_t::tanh:
        case post_op_type_t::square:
        case post_op_type_t::abs:
        case post_op_type_t::sqrt:
        case post_op_type_t::exp:
        case post_op_type_t::log:
        case post_op_type_t::clip:
        case post_op_type_t::binary_add:
        case post_op_type_t::binary_mul: return true;
        default: return false;
    }
}

float bf16_at(const void *p, size_t i) {
    const uint16_t *raw = static_cast<const uint16_t *>(p);
    return common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(raw[i]));
}

} // namespace

bool int8_epilogue_prepare(Int8Epilogue &ep, const GemmDescriptor &desc,
        void *dst, const void *bias, const matmul_params &params,
        const char *tag) {

    // ---- post-op chain ----------------------------------------------------
    bool any_postop = false;
    for (size_t i = 0; i < params.postop_.size(); ++i) {
        const auto t = params.postop_[i].po_type;
        if (t == post_op_type_t::none) continue;
        if (!postop_is_applied(t)) {
            log_info(tag,
                    ": post-op is not one the epilogue implements, declining "
                    "rather than dropping it");
            return false;
        }
        any_postop = true;
    }

    // ---- destination ------------------------------------------------------
    // The looper writes fp32. A bf16 destination gets a scratch, seeded from
    // what is already there when beta asks for it -- the looper scales that
    // seed, so the widening has to happen before the call, not after.
    ep.dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    if (desc.dst_dt == data_type_t::f32) {
        ep.C = static_cast<float *>(dst);
        ep.ldc = desc.ldc;
    } else if (ep.dst_is_bf16) {
        ep.c_scratch.assign(
                static_cast<size_t>(desc.M) * static_cast<size_t>(desc.N),
                0.0f);
        ep.C = ep.c_scratch.data();
        ep.ldc = desc.N;
        if (desc.beta != 0.0f) {
            for (int m = 0; m < desc.M; ++m)
                for (int n = 0; n < desc.N; ++n)
                    ep.C[static_cast<size_t>(m) * ep.ldc + n] = bf16_at(
                            dst, static_cast<size_t>(m) * desc.ldc + n);
        }
    } else {
        log_info(tag, ": only f32 and bf16 destinations are written, declining");
        return false;
    }

    // ---- bias -------------------------------------------------------------
    if (bias != nullptr) {
        if (desc.bias_dt == data_type_t::bf16) {
            ep.bias_f32.resize(desc.N);
            for (int n = 0; n < desc.N; ++n)
                ep.bias_f32[n] = bf16_at(bias, static_cast<size_t>(n));
            ep.bias_ptr = ep.bias_f32.data();
        } else if (desc.bias_dt == data_type_t::f32
                || desc.bias_dt == data_type_t::none) {
            // none is what a caller that never named a bias dtype leaves
            // behind; every in-tree producer of one pairs it with f32 data.
            ep.bias_ptr = static_cast<const float *>(bias);
        } else {
            log_info(tag, ": bias dtype is neither f32 nor bf16, declining");
            return false;
        }
    }

    ep.needs_finish = any_postop || (ep.bias_ptr != nullptr);
    return true;
}

void int8_epilogue_finish(Int8Epilogue &ep, const GemmDescriptor &desc,
        void *dst, const matmul_params &params) {

    if (ep.needs_finish) {
        // The whole of C in one call: these paths do not tile their epilogue,
        // so the tile is the matrix and both offsets are zero.
        apply_postops_tile(ep.C, ep.ldc, desc.M, desc.N, 0, 0, ep.bias_ptr,
                params.postop_);
    }

    if (ep.dst_is_bf16) {
        uint16_t *out = static_cast<uint16_t *>(dst);
        for (int m = 0; m < desc.M; ++m) {
            for (int n = 0; n < desc.N; ++n) {
                out[static_cast<size_t>(m) * desc.ldc + n]
                        = static_cast<uint16_t>(
                                common::bfloat16_t::f32_to_bf16_val(
                                        ep.C[static_cast<size_t>(m) * ep.ldc
                                                + n]));
            }
        }
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
