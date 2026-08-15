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

// ============================================================================
// Dispatch adapter for the asymmetric per-group (GGML k-quant) INT8 GEMM.
//
// The sibling adapter for the symmetric path carries the reasoning these two
// share -- why a decline is free on this hardware, why each refusal is a refusal
// rather than an approximation. What differs here:
//
//   * The weight is unsigned codes, so wei must be u8. That is also the
//     discriminator against the symmetric path, whose predicate requires s8.
//   * wei_scale is one region holding D and then, one cache line later, M. The
//     padding is why the offset is computed rather than assumed; see the unpack.
//   * There is no byte contract to check. The symmetric kernel excludes -128 and
//     scans both operands for it; here a lane sums two products of an unsigned
//     code and a signed activation, at most 2 * 31 * 128 = 7936 against the
//     16-bit limit, so every s8 activation is admissible. What must hold instead
//     is that the codes are k-quant codes -- at most 31 -- which the unpack
//     guarantees by construction and a caller supplying its own u8 weight would
//     have to guarantee itself.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_entry_128.hpp"

#include <cstdint>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_kquant_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_epilogue_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_looper_128.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

// Must match the unpack's layout: [ D | this many floats | M ].
constexpr int kKquantScalePadFloats = 16;

bool is_scalar_quant(const std::vector<int64_t> &dims) {
    for (int64_t d : dims)
        if (d > 1) return false;
    return true;
}

// The scale region is {2G, N}: G rows of D, then G rows of M. Returns G, or 0
// when the layout is not that.
int kquant_groups(const matmul_params &params, int K, int N) {
    const auto &dims = params.quant_params.wei_scale.dims;
    if (dims.size() != 2) return 0;
    if (static_cast<int>(dims[1]) != N) return 0;
    const int64_t two_g = dims[0];
    if (two_g <= 0 || (two_g % 2) != 0) return 0;
    const int g = static_cast<int>(two_g / 2);
    if (g <= 0 || K % g != 0) return 0;
    return g;
}

} // namespace

bool is_int8_kquant_candidate(const matmul_params &params, int K, int N) {
    if (params.dtypes.src != data_type_t::s8
            || params.dtypes.wei != data_type_t::u8)
        return false;
    // Codes must be a plain array; a blocked or still-packed buffer would be read
    // as row-major and produce confident nonsense.
    if (params.mem_format_b != 'n') return false;
    if (params.packing.pack_format_b != 0) return false;

    const int groups = kquant_groups(params, K, N);
    if (groups == 0) return false;
    const int group_size = K / groups;
    return group_size % KQ_VNNI_GRP == 0;
}

bool int8_kquant_try_execute_128(const GemmDescriptor &desc, const void *src,
        const void *weight, void *dst, const void *bias,
        const matmul_params &params) {

    if (desc.transA) return false;
    if (desc.M <= 0 || desc.N <= 0 || desc.K <= 0) return false;
    if (!is_int8_kquant_candidate(params, desc.K, desc.N)) return false;

    // Shared with the symmetric path: destination dtype, beta, bias, post-ops.
    Int8Epilogue epi;
    if (!int8_epilogue_prepare(epi, desc, dst, bias, params, "INT8 k-quant"))
        return false;

    const auto &qp = params.quant_params;
    if (qp.src_zp.buff != nullptr) {
        // A source zero point would add a second correction term, one this
        // kernel has no argument for.
        int32_t zp = 0;
        if (qp.src_zp.dt == data_type_t::s32)
            zp = *static_cast<const int32_t *>(qp.src_zp.buff);
        else if (qp.src_zp.dt == data_type_t::s8)
            zp = *static_cast<const int8_t *>(qp.src_zp.buff);
        else if (qp.src_zp.dt == data_type_t::u8)
            zp = *static_cast<const uint8_t *>(qp.src_zp.buff);
        else {
            log_info("INT8 k-quant: unrecognised zero-point dtype, declining");
            return false;
        }
        if (zp != 0) {
            log_info("INT8 k-quant: non-zero source zero point, declining");
            return false;
        }
    }

    if (qp.wei_scale.buff == nullptr) {
        log_info("INT8 k-quant: no weight scale, declining");
        return false;
    }
    if (qp.wei_scale.dt != data_type_t::f32) {
        // The unpack writes f32 for this path precisely so the flush needs no
        // widening pass; anything else is not a buffer this adapter wrote.
        log_info("INT8 k-quant: weight scales must be f32, declining");
        return false;
    }

    const int groups = kquant_groups(params, desc.K, desc.N);
    const int group_size = desc.K / groups;
    const float *D = static_cast<const float *>(qp.wei_scale.buff);
    // M follows D across the padding the unpack inserts to stop the two streams
    // aliasing; computed, never assumed to be at D + groups*N.
    const float *Mn = D + static_cast<size_t>(groups) * desc.N
            + kKquantScalePadFloats;

    // ---- activation scale, same three granularities as the symmetric path ----
    int ss_row = 0, ss_grp = 0;
    {
        const auto &d = qp.src_scale.dims;
        if (is_scalar_quant(d)) {
            ss_row = 0;
            ss_grp = 0;
        } else if ((d.size() == 2 && d[0] == desc.M && d[1] == 1)
                || (d.size() == 1 && d[0] == desc.M)) {
            // {M, 1} and the 1-D {M} that some callers write instead are the
            // same per-token layout; declining the second for being written
            // differently would be a decline over notation.
            ss_row = 1;
            ss_grp = 0;
        } else if (d.size() == 2 && d[0] == desc.M && d[1] == groups) {
            ss_row = groups;
            ss_grp = 1;
        } else {
            log_info("INT8 k-quant: source scale is not per-tensor, per-token or "
                     "per-group over this shape; declining");
            return false;
        }
    }
    const size_t n_src_scales = ss_row == 0
            ? size_t {1}
            : static_cast<size_t>(desc.M) * static_cast<size_t>(ss_row);

    // alpha folds into the activation scale, both multiplying the same product.
    const bool need_copy
            = (desc.alpha != 1.0f) || qp.src_scale.dt == data_type_t::bf16;
    std::vector<float> src_scale_buf;
    const float *src_scale = nullptr;
    float src_scale_one = desc.alpha;

    if (qp.src_scale.buff == nullptr) {
        src_scale = &src_scale_one;
        ss_row = 0;
        ss_grp = 0;
    } else if (qp.src_scale.dt == data_type_t::f32 && !need_copy) {
        src_scale = static_cast<const float *>(qp.src_scale.buff);
    } else if (qp.src_scale.dt == data_type_t::f32
            || qp.src_scale.dt == data_type_t::bf16) {
        src_scale_buf.resize(n_src_scales);
        if (qp.src_scale.dt == data_type_t::f32) {
            const float *raw = static_cast<const float *>(qp.src_scale.buff);
            for (size_t i = 0; i < n_src_scales; ++i)
                src_scale_buf[i] = raw[i] * desc.alpha;
        } else {
            const uint16_t *raw
                    = static_cast<const uint16_t *>(qp.src_scale.buff);
            for (size_t i = 0; i < n_src_scales; ++i) {
                src_scale_buf[i] = common::bfloat16_t::bf16_to_f32_val(
                                           static_cast<int16_t>(raw[i]))
                        * desc.alpha;
            }
        }
        src_scale = src_scale_buf.data();
    } else {
        log_info("INT8 k-quant: unrecognised source scale dtype, declining");
        return false;
    }

    const int nthreads = desc.num_threads > 0 ? desc.num_threads : 1;
    if (!int8_kquant_execute_128(desc.M, desc.N, desc.K, group_size,
                static_cast<const int8_t *>(src), desc.lda,
                static_cast<const uint8_t *>(weight), desc.ldb, desc.transB,
                epi.C, epi.ldc, D, Mn, src_scale, ss_row, ss_grp, nthreads,
                desc.beta))
        return false;
    int8_epilogue_finish(epi, desc, dst, params);
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
