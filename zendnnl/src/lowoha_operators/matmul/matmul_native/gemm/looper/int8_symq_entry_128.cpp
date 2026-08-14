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
// Dispatch adapter for the symmetric per-group INT8 GEMM.
//
// The looper next door is a numeric kernel with a narrow contract; this file is
// the part that decides whether a real matmul call satisfies it, and translates
// what does. Everything it cannot express is declined rather than approximated,
// because the fallback is not a worse answer -- it is the pre-existing
// behaviour, which on family 15h is AOCL-DLP returning without computing at
// all. A decline therefore costs nothing that was working, and a wrong
// acceptance would be the only way this path can do harm.
//
// The gates, and why each one is here rather than handled:
//
//   src s8, not u8      u8 * s8 is the asymmetric form. A PMADDUBSW lane can
//                       reach 64770 there, past the signed 16-bit limit, which
//                       is exactly why this kernel is cheap and why it may not
//                       be pointed at that case.
//   dst f32             the looper writes fp32. A bf16 dst would need a
//                       narrowing sweep over C, which belongs with post-op
//                       support rather than bolted on here.
//   beta == 0           the looper makes one pass over C and overwrites it.
//                       beta != 0 needs C read back and scaled first.
//   no bias, no post-op nothing here applies them, and silently dropping a
//                       post-op is the worst available outcome.
//   src zp zero         a source zero point makes the product asymmetric again.
//   src scale scalar    per-token source scales would need one scale per row in
//                       the flush, which the microkernel has no argument for.
//   wei scale {G, N}    the per-group layout the GGML unpack writes. G must
//                       divide K, and K/G must be a multiple of the VNNI quad.
//   bytes in [-127,127] the kernel's precondition. Every GGML quantiser and the
//                       in-tree symmetric quantiser (absmax/127) satisfy it, but
//                       an arbitrary caller-supplied s8 buffer need not, and
//                       -128 fails silently: PSIGNB cannot negate it and a lane
//                       of two 128*128 products saturates. So it is checked
//                       rather than assumed.
//
// The byte scan is O(M*K + N*K) against O(M*N*K) of arithmetic, so it is
// asymptotically free for a GEMM -- but not for the M=1 decode shape, where it
// lands on the same order as the work itself. That is the same problem the
// per-call weight packing has, and it wants the same answer: a cached verdict
// alongside the cached packed weight. Left for the change that adds the cache,
// so a correctness failure here cannot be confused with a caching bug.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_entry_128.hpp"

#include <cstdint>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

namespace {

// A quantisation tensor is per-tensor when it has no dims or every dim is 1.
bool is_scalar_quant(const std::vector<int64_t> &dims) {
    for (int64_t d : dims)
        if (d > 1) return false;
    return true;
}

// The per-group weight scale, as {groups, N}. Returns 0 when the layout is not
// the two-dimensional per-group form this path needs.
int wei_scale_groups(const matmul_params &params, int N) {
    const auto &dims = params.quant_params.wei_scale.dims;
    if (dims.size() != 2) return 0;
    if (static_cast<int>(dims[1]) != N) return 0;
    const int64_t g = dims[0];
    if (g <= 0 || g > INT32_MAX) return 0;
    return static_cast<int>(g);
}

bool has_any_postop(const matmul_params &params) {
    for (size_t i = 0; i < params.postop_.size(); ++i) {
        if (params.postop_[i].po_type != post_op_type_t::none) return true;
    }
    return false;
}

// -128 breaks the kernel silently, so the contract is verified rather than
// trusted. Both operands, because either one can carry it.
bool bytes_within_contract(const int8_t *p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] == static_cast<int8_t>(-128)) return false;
    return true;
}

bool rows_within_contract(const int8_t *p, int rows, int ld, int len) {
    for (int r = 0; r < rows; ++r) {
        if (!bytes_within_contract(p + static_cast<size_t>(r) * ld,
                    static_cast<size_t>(len)))
            return false;
    }
    return true;
}

} // namespace

bool is_int8_symq_candidate(const matmul_params &params, int K, int N) {
    if (params.dtypes.src != data_type_t::s8
            || params.dtypes.wei != data_type_t::s8)
        return false;
    const int groups = wei_scale_groups(params, N);
    if (groups == 0 || K % groups != 0) return false;
    const int group_size = K / groups;
    return group_size % SYMQ_VNNI_GRP == 0;
}

bool int8_symq_try_execute_128(const GemmDescriptor &desc, const void *src,
        const void *weight, void *dst, const void *bias,
        const matmul_params &params) {

    if (desc.transA) return false;
    if (desc.M <= 0 || desc.N <= 0 || desc.K <= 0) return false;

    if (!is_int8_symq_candidate(params, desc.K, desc.N)) return false;
    if (desc.dst_dt != data_type_t::f32) {
        log_info("INT8 symq: only an f32 destination is implemented, declining");
        return false;
    }
    if (desc.beta != 0.0f) {
        log_info("INT8 symq: beta != 0 needs C read back, declining");
        return false;
    }
    if (bias != nullptr || has_any_postop(params)) {
        log_info("INT8 symq: bias and post-ops are not applied here, declining");
        return false;
    }

    // ---- quantisation -----------------------------------------------------
    const auto &qp = params.quant_params;
    if (qp.src_zp.buff != nullptr && !is_scalar_quant(qp.src_zp.dims)) {
        log_info("INT8 symq: per-token source zero point, declining");
        return false;
    }
    if (qp.src_zp.buff != nullptr) {
        // A zero point of zero is symmetric and fine; anything else is not.
        int32_t zp = 0;
        if (qp.src_zp.dt == data_type_t::s32)
            zp = *static_cast<const int32_t *>(qp.src_zp.buff);
        else if (qp.src_zp.dt == data_type_t::s8)
            zp = *static_cast<const int8_t *>(qp.src_zp.buff);
        else if (qp.src_zp.dt == data_type_t::u8)
            zp = *static_cast<const uint8_t *>(qp.src_zp.buff);
        else {
            log_info("INT8 symq: unrecognised zero-point dtype, declining");
            return false;
        }
        if (zp != 0) {
            log_info("INT8 symq: non-zero source zero point, declining");
            return false;
        }
    }

    if (!is_scalar_quant(qp.src_scale.dims)) {
        log_info("INT8 symq: per-token source scale, declining");
        return false;
    }
    float src_scale = 1.0f;
    if (qp.src_scale.buff != nullptr) {
        if (qp.src_scale.dt == data_type_t::f32) {
            src_scale = *static_cast<const float *>(qp.src_scale.buff);
        } else if (qp.src_scale.dt == data_type_t::bf16) {
            src_scale = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(
                            *static_cast<const uint16_t *>(qp.src_scale.buff)));
        } else {
            log_info("INT8 symq: unrecognised source scale dtype, declining");
            return false;
        }
    }
    // alpha scales the product, and the source scale is applied to exactly the
    // same product once per group, so folding it in costs nothing. beta is
    // already required to be zero above.
    src_scale *= desc.alpha;

    const int groups = wei_scale_groups(params, desc.N);
    const int group_size = desc.K / groups;

    if (qp.wei_scale.buff == nullptr) {
        log_info("INT8 symq: no weight scale, declining");
        return false;
    }

    // The GGML unpack hands back bf16 scales; a caller supplying its own
    // per-group weight may hand back f32. Widen once per call for the former.
    // This is a pass over groups*N values, which is small against the GEMM but
    // not against an M=1 decode -- another reason the caching change wants to
    // cover the scales as well as the packed weight.
    const float *wei_scale = nullptr;
    std::vector<float> wei_scale_f32;
    if (qp.wei_scale.dt == data_type_t::f32) {
        wei_scale = static_cast<const float *>(qp.wei_scale.buff);
    } else if (qp.wei_scale.dt == data_type_t::bf16) {
        const uint16_t *raw = static_cast<const uint16_t *>(qp.wei_scale.buff);
        const size_t n = static_cast<size_t>(groups) * desc.N;
        wei_scale_f32.resize(n);
        for (size_t i = 0; i < n; ++i) {
            wei_scale_f32[i] = common::bfloat16_t::bf16_to_f32_val(
                    static_cast<int16_t>(raw[i]));
        }
        wei_scale = wei_scale_f32.data();
    } else {
        log_info("INT8 symq: unrecognised weight scale dtype, declining");
        return false;
    }

    // ---- the kernel's byte contract ---------------------------------------
    const int8_t *A = static_cast<const int8_t *>(src);
    const int8_t *B = static_cast<const int8_t *>(weight);
    if (!rows_within_contract(A, desc.M, desc.lda, desc.K)) {
        log_info("INT8 symq: source contains -128, outside the kernel's "
                 "contract; declining");
        return false;
    }
    const int b_rows = desc.transB ? desc.N : desc.K;
    const int b_len = desc.transB ? desc.K : desc.N;
    if (!rows_within_contract(B, b_rows, desc.ldb, b_len)) {
        log_info("INT8 symq: weight contains -128, outside the kernel's "
                 "contract; declining");
        return false;
    }

    const int nthreads = desc.num_threads > 0 ? desc.num_threads : 1;
    return int8_symq_execute_128(desc.M, desc.N, desc.K, group_size, A,
            desc.lda, B, desc.ldb, desc.transB, static_cast<float *>(dst),
            desc.ldc, wei_scale, src_scale, nthreads);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
