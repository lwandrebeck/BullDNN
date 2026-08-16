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
//   dst f32 or bf16     the looper writes fp32; a bf16 dst goes through a
//                       scratch and a narrowing sweep. Anything else declines.
//                       (int8_epilogue_128)
//   post-op implemented apply_postops_tile() ends in `default: break;`, so a
//                       post-op it does not implement would be dropped in
//                       silence. The three that are -- softmax, pooling, mish --
//                       decline instead. (int8_epilogue_128)
//   src zp zero         a source zero point makes the product asymmetric again.
//                       Correcting for it is possible -- subtract zp times the
//                       weight column sums, the same shape of correction the
//                       k-quant path already does for its min term -- but no
//                       caller here produces one, so it is declined rather than
//                       written blind.
//   src scale layout    per-tensor, per-token ({M,1} or {M}) and per-group
//                       ({M,G}) are taken; the kernel flushes once per (row,
//                       group) and so expresses all three. Anything else is a
//                       layout nothing in tree produces.
//   wei scale {G, N}    the per-group layout the GGML unpack writes. G must
//                       divide K, and K/G must be a multiple of the VNNI quad.
//   bytes in [-127,127] the kernel's precondition. Every GGML quantiser and the
//                       in-tree symmetric quantiser (absmax/127) satisfy it, but
//                       an arbitrary caller-supplied s8 buffer need not, and
//                       -128 fails silently: PSIGNB cannot negate it and a lane
//                       of two 128*128 products saturates. So it is checked
//                       rather than assumed.
//
// The weight-side work -- the byte scan and, when the GGML unpack hands over
// bf16, widening the per-group scales -- is O(N*K) and O(groups*N) against
// O(M*N*K) of arithmetic. That is asymptotically free for a GEMM and emphatically
// not free at M=1, where one row of A is read against the whole of B and those
// passes cost more than the multiply-adds. Both are facts about the weight rather
// than the call, so both are computed once and cached under the same
// is_weights_const promise the packed-weight cache relies on.
//
// The source is scanned every call, because it is the activation: it changes, and
// at M*K bytes it is small.
// ============================================================================

#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_entry_128.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_epilogue_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::matmul_config_t;

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

// ---------------------------------------------------------------------------
// What is true of a weight rather than of a call: whether it honours the byte
// contract, and its per-group scales as f32. Both are O(weight) work repeated
// per call otherwise, which at M=1 costs more than the arithmetic.
//
// Keyed on the PACKED buffer's address, not the caller's weight pointer, and
// that choice is load-bearing. The packed buffer belongs to the prepacked-weight
// cache: it is allocated there, lives as long as the entry, and holds exactly
// the bytes the microkernel will read. A verdict attached to it therefore cannot
// describe different bytes than the ones used.
//
// Keying on the caller's pointer instead is unsound in a way that is easy to
// miss and was caught here by the dispatch tests. A weight can be freed and a
// different weight of the same shape allocated at the same address; the key then
// matches and the verdict is served for the wrong bytes. For the packed data
// that yields a wrong answer, which is the risk the whole family of
// pointer-keyed weight caches in this tree already takes under
// is_weights_const. For the -128 check it would be worse than wrong: a safety
// gate silently skipped. Attaching it to the packed buffer removes that second
// failure entirely -- the verdict and the bytes share one lifetime.
//
// The scales ride along under the same key. In the path this exists for they are
// literally part of the same allocation: unpack_ggml_raw_s8_and_cache() writes
// [ s8 weight | bf16 scales ] into one buffer, so weight and scales share a
// lifetime by construction. The source pointer is recorded and compared anyway,
// so a weight arriving with a different scale buffer re-widens rather than
// reading a stale one.
struct SymqWeightFacts {
    bool in_contract = false;
    const void *scale_src = nullptr;
    std::vector<float> widened_scales; // empty when the caller gave f32
};

SymqWeightFacts *lookup_weight_facts(
        const void *packed, const void *scale_src, bool *out_is_new) {
    static std::unordered_map<const void *, std::unique_ptr<SymqWeightFacts>>
            s_cache;
    static std::mutex s_mutex;

    std::lock_guard<std::mutex> lock(s_mutex);
    auto it = s_cache.find(packed);
    if (it != s_cache.end() && it->second->scale_src == scale_src) {
        *out_is_new = false;
        return it->second.get();
    }
    auto facts = std::make_unique<SymqWeightFacts>();
    facts->scale_src = scale_src;
    SymqWeightFacts *raw = facts.get();
    s_cache[packed] = std::move(facts);
    *out_is_new = true;
    return raw;
}

void widen_bf16_scales(
        const void *src, size_t n, std::vector<float> &out) {
    const uint16_t *raw = static_cast<const uint16_t *>(src);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = common::bfloat16_t::bf16_to_f32_val(
                static_cast<int16_t>(raw[i]));
    }
}

} // namespace

bool is_int8_symq_candidate(const matmul_params &params, int K, int N) {
    if (params.dtypes.src != data_type_t::s8
            || params.dtypes.wei != data_type_t::s8)
        return false;
    // The weight must be a plain row-major (or transposed) s8 matrix. Two other
    // layouts reach this point wearing the same dtypes and the same {G, N}
    // weight scale, and reading either as row-major would be silent garbage
    // rather than a decline:
    //
    //   mem_format_b == 'r'    an AOCL sym-quant blocked buffer. The GGML path
    //                          produces exactly this today --
    //                          unpack_ggml_weights_and_cache() reorders for
    //                          AOCL unless its skip_reorder argument is set,
    //                          and the single-matmul caller does not set it.
    //   pack_format_b == 1     still GGML block-quantised, not yet unpacked.
    //
    // So a GGML call currently declines here instead of arriving. Making it
    // arrive means asking the unpack for the raw-s8 form when this path will
    // handle the shape, which is a change to the unpack's caller and belongs
    // with the test that proves it end to end.
    if (params.mem_format_b != 'n') return false;
    if (params.packing.pack_format_b != 0) return false;
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

    // Destination dtype, beta, bias and post-ops are all shared with the
    // k-quant path and all handled in int8_epilogue_128. It declines anything it
    // cannot honour exactly -- notably a post-op the applier would drop in
    // silence -- so a false here is still a decline, not a partial answer.
    Int8Epilogue epi;
    if (!int8_epilogue_prepare(epi, desc, dst, bias, params, "INT8 symq"))
        return false;

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

    const int groups = wei_scale_groups(params, desc.N);
    const int group_size = desc.K / groups;

    // Activation scale granularity. The API allows {1,1} per-tensor, {M,1}
    // per-token and {M,G} per-group, and the kernel takes all three because it
    // flushes once per (row, group) regardless -- the finest of them. This
    // matters more than it looks: the GGML path's own gate, ggml_is_sym_quant(),
    // requires a source scale with more than one element, so a per-tensor-only
    // kernel is one the GGML path can never reach.
    int ss_row = 0, ss_grp = 0;
    // How many weight groups share one activation scale. 1 for the usual case;
    // 2 for Q6_K, whose weights scale every SIXTEEN elements while GGML's
    // activations scale every 32. See the expansion below.
    int groups_per_src_scale = 1;
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
        } else if (d.size() == 2 && d[0] == desc.M && d[1] > 0
                && groups % d[1] == 0) {
            // COARSER than the weight groups, which is not a mistake by the
            // caller: a Q6_K weight carries a scale per sixteen elements, so
            // the unpack declares {K/16, N}, while GGML quantises activations
            // per 32 and llama.cpp passes {M, K/32}. Two weight groups then
            // share one activation scale.
            //
            // Declining that is what sent Q6_K to AOCL-DLP, which cannot run
            // per-group INT8 without VNNI and returns WITHOUT COMPUTING -- NaN
            // where AOCL-DLP is present, an error where it is not. It is also
            // why enabling Q6_K in the llama.cpp backend produced NaN under
            // llama-perplexity while every in-tree Q6_K test passed: those
            // tests all hand over pre-quantised s8 with matching scales.
            //
            // The microkernel indexes scales by a fixed stride and has no
            // divisor, so rather than teach it one the scales are expanded
            // below into the per-group form it already reads. That is
            // M * groups floats against M*N*K multiply-adds.
            groups_per_src_scale = groups / static_cast<int>(d[1]);
            ss_row = groups;
            ss_grp = 1;
        } else {
            log_info("INT8 symq: source scale is not per-tensor, per-token or "
                     "per-group over this shape; declining");
            return false;
        }
    }
    const size_t n_src_scales = ss_row == 0
            ? size_t(1)
            : static_cast<size_t>(desc.M) * static_cast<size_t>(ss_row);

    // alpha scales the product and the activation scale multiplies exactly the
    // same product, so it folds in. With more than one scale that means a scaled
    // copy; the buffer is O(M*G), it is activation-side and changes every call
    // anyway, and alpha != 1 is rare on this path. beta is already zero above.
    const bool need_scaled_copy
            = (desc.alpha != 1.0f) || qp.src_scale.dt == data_type_t::bf16;
    std::vector<float> src_scale_buf;
    const float *src_scale = nullptr;
    float src_scale_one = desc.alpha;

    if (qp.src_scale.buff != nullptr && groups_per_src_scale > 1) {
        // Replicate each activation scale across the weight groups it covers.
        const size_t n_coarse = static_cast<size_t>(desc.M)
                * static_cast<size_t>(groups / groups_per_src_scale);
        src_scale_buf.resize(static_cast<size_t>(desc.M) * groups);
        for (int m = 0; m < desc.M; ++m) {
            for (int g = 0; g < groups; ++g) {
                const size_t src_i = static_cast<size_t>(m)
                                * (groups / groups_per_src_scale)
                        + g / groups_per_src_scale;
                if (src_i >= n_coarse) return false; // shape lied; decline
                float v;
                if (qp.src_scale.dt == data_type_t::f32) {
                    v = static_cast<const float *>(qp.src_scale.buff)[src_i];
                } else if (qp.src_scale.dt == data_type_t::bf16) {
                    v = common::bfloat16_t::bf16_to_f32_val(static_cast<int16_t>(
                            static_cast<const uint16_t *>(
                                    qp.src_scale.buff)[src_i]));
                } else {
                    log_info("INT8 symq: unrecognised source scale dtype, "
                             "declining");
                    return false;
                }
                src_scale_buf[static_cast<size_t>(m) * groups + g]
                        = v * desc.alpha;
            }
        }
        src_scale = src_scale_buf.data();
    } else if (qp.src_scale.buff == nullptr) {
        src_scale = &src_scale_one; // no scale supplied: alpha alone
        ss_row = 0;
        ss_grp = 0;
    } else if (qp.src_scale.dt == data_type_t::f32 && !need_scaled_copy) {
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
        log_info("INT8 symq: unrecognised source scale dtype, declining");
        return false;
    }

    if (qp.wei_scale.buff == nullptr) {
        log_info("INT8 symq: no weight scale, declining");
        return false;
    }

    if (qp.wei_scale.dt != data_type_t::f32
            && qp.wei_scale.dt != data_type_t::bf16) {
        log_info("INT8 symq: unrecognised weight scale dtype, declining");
        return false;
    }

    // ---- weight-side facts, once per weight where that is allowed ----------
    const int8_t *A = static_cast<const int8_t *>(src);
    const int8_t *B = static_cast<const int8_t *>(weight);
    const int b_rows = desc.transB ? desc.N : desc.K;
    const int b_len = desc.transB ? desc.K : desc.N;
    const size_t n_scales = static_cast<size_t>(groups) * desc.N;

    static const int32_t s_weight_cache
            = matmul_config_t::instance().get_weight_cache();
    const bool can_cache = desc.is_weights_const && (s_weight_cache != 0);

    const float *wei_scale = nullptr;
    std::vector<float> scratch_scales;
    const INT8PrepackedWeight *prepacked = nullptr;

    if (can_cache) {
        const PrepackedWeightKey key {
                weight, desc.K, desc.N, desc.ldb, desc.transB};
        prepacked = INT8PrepackedWeightCache::instance().get_or_prepack(key, B);
    }

    if (prepacked != nullptr) {
        // The packed buffer is the authority: scan it rather than the caller's
        // B, so the verdict describes the bytes the microkernel will read. Its
        // zero padding is inside the contract, so it cannot cause a refusal.
        bool is_new = false;
        SymqWeightFacts *facts = lookup_weight_facts(
                prepacked->data, qp.wei_scale.buff, &is_new);
        if (is_new) {
            const size_t packed_len = static_cast<size_t>(prepacked->n_panels)
                    * (prepacked->K_padded / SYMQ_VNNI_GRP)
                    * INT8PrepackedWeight::stride();
            facts->in_contract
                    = bytes_within_contract(prepacked->data, packed_len);
            if (qp.wei_scale.dt == data_type_t::bf16) {
                widen_bf16_scales(qp.wei_scale.buff, n_scales,
                        facts->widened_scales);
            }
        }
        if (!facts->in_contract) {
            log_info("INT8 symq: weight contains -128, outside the kernel's "
                     "contract; declining");
            return false;
        }
        wei_scale = facts->widened_scales.empty()
                ? static_cast<const float *>(qp.wei_scale.buff)
                : facts->widened_scales.data();
    } else {
        if (!rows_within_contract(B, b_rows, desc.ldb, b_len)) {
            log_info("INT8 symq: weight contains -128, outside the kernel's "
                     "contract; declining");
            return false;
        }
        if (qp.wei_scale.dt == data_type_t::bf16) {
            widen_bf16_scales(qp.wei_scale.buff, n_scales, scratch_scales);
            wei_scale = scratch_scales.data();
        } else {
            wei_scale = static_cast<const float *>(qp.wei_scale.buff);
        }
    }

    // The source is the activation: it changes every call, so it is checked
    // every call. At M*K bytes that is negligible beside the arithmetic.
    if (!rows_within_contract(A, desc.M, desc.lda, desc.K)) {
        log_info("INT8 symq: source contains -128, outside the kernel's "
                 "contract; declining");
        return false;
    }

    const int nthreads = desc.num_threads > 0 ? desc.num_threads : 1;
    if (!int8_symq_execute_128(desc.M, desc.N, desc.K, group_size, A, desc.lda,
                B, desc.ldb, desc.transB, epi.C, epi.ldc, wei_scale, src_scale,
                ss_row, ss_grp, nthreads, prepacked, desc.beta))
        return false;
    int8_epilogue_finish(epi, desc, dst, params);
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
