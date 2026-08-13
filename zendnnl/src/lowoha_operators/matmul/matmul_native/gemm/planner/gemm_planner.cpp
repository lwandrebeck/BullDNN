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

#include "lowoha_operators/matmul/matmul_native/gemm/planner/gemm_planner.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "operators/matmul/matmul_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::ops::post_op_type_t;
using namespace zendnnl::error_handling;

static inline int round_up(int x, int m) {
    return ((x + m - 1) / m) * m;
}
static inline int round_down(int x, int m) {
    return (x / m) * m;
}

// Blocking override, for sweeping MB/NB/KB on a part before committing a
// cost-model change. Zero (the default, and the value for anything
// unparseable) means "leave the planner's choice alone".
static int env_block_override(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr) return 0;
    int x = std::atoi(v);
    return x > 0 ? x : 0;
}

// Pick MB no larger than mb_cap that tiles M evenly across nt threads.
//
// A cache budget alone says nothing about how M divides. MB=246 on M=1024
// leaves 40 rows as the last of five tiles, so one thread carries a sixth of
// the work its peers do and the wavefront waits on the stragglers; measured on
// Excavator that costs about 19% against MB=132, which yields eight tiles for
// four threads. Tile count matters more than the last tile being full: MB=114
// leaves a 98%-full last tile but nine tiles for four threads, and loses.
//
// So score candidates by tile count divisible by nt first, fullest last tile
// second. This is the M-dimension counterpart of choose_even_kb below and of
// the NB even-ization in plan_blocks, both of which K and N already got.
static int choose_even_mb(int M, int mb_cap, int MR, int nt, int kb,
        int elem_bytes, int cache_per_thread) {
    mb_cap = round_down(std::max(mb_cap, MR), MR);
    if (mb_cap >= M) return M;
    if (nt < 1) nt = 1;

    // Escape hatch for A/B measurement: reproduces the cache-budget-only MB
    // this function replaced. Requires a non-empty value other than "0", so
    // that an exported-but-empty variable does not silently disable the
    // planner -- which is exactly what it did during the first A/B attempt.
    static const bool s_off = [] {
        const char *v = std::getenv("ZENDNNL_NATIVE_GEMM_NO_EVEN_MB");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    if (s_off) return mb_cap;

    // Shrinking MB is not free: every extra i-tile re-streams the whole B
    // panel, so it only pays when it buys something. Two cases where it does.
    //
    //   1. The cap leaves fewer tiles than threads, so threads sit idle.
    //   2. The A block does not fit the cache each thread actually gets, so
    //      the big block streams from memory anyway.
    //
    // Where neither holds, keep the largest block the cache budget allows.
    // Measured on a Xeon 6767P at 8 threads, forcing the shrink anyway cost 9%
    // on 4096x1024x1024 and 5% on 8192x2048x2048 -- both had ample tiles and an
    // A block inside the 2 MB private L2. Excavator hits case 2 on essentially
    // every large shape (1 MB L2 shared by a module's two cores, so 512 KB a
    // thread), which is why the same shapes only ever gained there.
    const long long a_block
            = static_cast<long long>(mb_cap) * std::max(kb, 1) * elem_bytes;
    const int tiles_at_cap = (M + mb_cap - 1) / mb_cap;
    const bool starved_of_tiles = tiles_at_cap < nt;
    const bool a_block_spills
            = cache_per_thread > 0 && a_block > cache_per_thread;
    if (!starved_of_tiles && !a_block_spills) return mb_cap;

    const int t_min = (M + mb_cap - 1) / mb_cap;
    int best_mb = mb_cap;
    double best_score = -1.0;

    // Past a few times the thread count the tiles are so small that per-tile
    // overhead dominates, so cap the search rather than walking to M/MR.
    const int t_max = std::max(t_min + 8, 4 * nt);
    for (int t = t_min; t <= t_max; ++t) {
        int mb = round_up((M + t - 1) / t, MR);
        if (mb > mb_cap || mb < MR) continue;
        const int tiles = (M + mb - 1) / mb;
        const int last = M - (tiles - 1) * mb;
        if (last <= 0) continue;
        const double score = (tiles % nt == 0 ? 2.0 : 0.0)
                + static_cast<double>(last) / static_cast<double>(mb);
        if (score > best_score) {
            best_score = score;
            best_mb = mb;
        }
    }
    return best_mb;
}

static int choose_even_kb(int K, int kb_max) {
    if (kb_max >= K) return K;
    int n_blocks = (K + kb_max - 1) / kb_max;
    int even_kb = (K + n_blocks - 1) / n_blocks;
    return round_up(even_kb, 8);
}

// ============================================================================
// Base blocking plan (shared by BF16 and FP32)
// ============================================================================

BlockPlan plan_blocks(const GemmDescriptor &desc, const UarchParams &uarch) {
    BlockPlan plan;
    const int M = desc.M, N = desc.N, K = desc.K;
    const int elem = static_cast<int>(desc.wei_elem_size);
    const int nt = desc.num_threads;
    plan.num_threads = nt;

    plan.MR = 6;
    plan.NR = 16;
    const int MR = plan.MR, NR = plan.NR;

    // KB: A micro-panel (MR×KB×elem) in 80% L1, B tile (NR×KB×elem) in L2/2.
    int l1_budget = static_cast<int>(0.8 * uarch.l1d_bytes);
    int l2_budget = uarch.l2_bytes / 2;

    int kb_max_l1 = l1_budget / (MR * elem);
    int kb_max_l2 = l2_budget / (NR * elem);
    int kb_max = std::min(kb_max_l1, kb_max_l2);
    kb_max = std::max(kb_max, 64);
    plan.KB = choose_even_kb(K, kb_max);

    // NB: B panel (KB×NB×elem) in L2/2, L3-aware for multi-thread.
    int nb_from_l2 = l2_budget / (plan.KB * elem);
    int nb_from_l3 = nb_from_l2;
    if (nt > 1) {
        int cores_per_ccd = std::min(uarch.num_cores, 8);
        int threads_sharing_l3 = std::min(nt, cores_per_ccd);
        int l3_budget = static_cast<int>(0.5 * uarch.l3_bytes_per_ccd);
        nb_from_l3 = l3_budget / (threads_sharing_l3 * plan.KB * elem);
    }
    int nb_max = std::min(nb_from_l2, nb_from_l3);
    plan.NB = round_down(std::max(nb_max, NR), NR);
    if (plan.NB > N) plan.NB = N;

    if (plan.NB < N && plan.NB >= NR) {
        int n_jtiles = (N + plan.NB - 1) / plan.NB;
        int even_nb = round_up((N + n_jtiles - 1) / n_jtiles, NR);
        if (even_nb > 0 && even_nb <= plan.NB + NR)
            plan.NB = std::min(even_nb, N);
    }

    // MB: C tile (MB×NB×4) + A tile (MB×KB×elem) in L1+L2.
    int total_per_core = uarch.l1d_bytes + uarch.l2_bytes;
    int c_cost_per_row = plan.NB * 4;
    int a_cost_per_row = plan.KB * elem;
    plan.MB = round_down(
            std::max(total_per_core / (c_cost_per_row + a_cost_per_row), MR),
            MR);
    if (plan.MB > M) plan.MB = M;

    // Thread load balancing.
    if (nt > 1) {
        int min_tiles = nt;
        int ideal_tiles = 2 * nt;
        int target = min_tiles;
        {
            int ic_t = (M + plan.MB - 1) / plan.MB;
            int jc_t = (N + plan.NB - 1) / plan.NB;
            if (ic_t * jc_t >= ideal_tiles) target = ideal_tiles;
        }
        while (true) {
            int ic_t = (M + plan.MB - 1) / plan.MB;
            int jc_t = (N + plan.NB - 1) / plan.NB;
            if (ic_t * jc_t >= target) break;
            if (plan.NB > NR) {
                plan.NB -= NR;
                continue;
            }
            if (plan.MB > MR && M > MR) {
                plan.MB -= MR;
                continue;
            }
            break;
        }
    }

    return plan;
}

// ============================================================================
// FP32 GEMM plan
// ============================================================================

FP32GemmPlan plan_fp32_gemm(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;

    [[maybe_unused]] bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
        } else if (pt == post_op_type_t::gelu_tanh
                || pt == post_op_type_t::gelu_erf
                || pt == post_op_type_t::sigmoid || pt == post_op_type_t::tanh
                || pt == post_op_type_t::swish || pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    static thread_local struct {
        int M, N, K, threads;
        bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K
            && s_plan_cache.threads == desc.num_threads
            && s_plan_cache.transA == desc.transA
            && s_plan_cache.transB == desc.transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);

        if (desc.num_threads <= 1) {
            int b_bytes = K * NR_PACK * static_cast<int>(sizeof(float));
            if (b_bytes <= uarch.l2_bytes / 2) plan.KB = K;
        } else {
            int kb_b = (uarch.l2_bytes / 2)
                    / (NR_PACK * static_cast<int>(sizeof(float)));
            int kb_a = (uarch.l2_bytes / 2)
                    / (6 * static_cast<int>(sizeof(float)));
            int kb_max_mt = std::min(kb_a, kb_b);
            if (K <= kb_max_mt) {
                plan.KB = K;
            } else if (kb_max_mt > plan.KB) {
                int n_blocks = (K + kb_max_mt - 1) / kb_max_mt;
                plan.KB = ((K + n_blocks - 1) / n_blocks + 7) & ~7;
            }
        }

        s_plan_cache
                = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    plan.MR = 6;
    if (N >= 64 && !has_complex_activation) {
        plan.NR = 64;
    } else if (N >= 32) {
        plan.NR = 32;
    } else {
        plan.NR = 16;
    }

    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    {
        bool will_pack_a = desc.transA
                || (desc.lda * static_cast<int>(sizeof(float)) > 4096);
        int mb_cap = 0;
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int b_accessed_bytes = plan.KB * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            mb_cap = std::max(l2_for_a / (plan.KB * 4), plan.MR);
        } else {
            mb_cap = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                            / (plan.NB * 4 + plan.KB * 4),
                    plan.MR);
        }
        // choose_even_mb has the last word: it already returns a multiple of
        // MR bounded by M, and it needs the raw cache cap to tell whether a
        // single tile fits. Rounding down to MR afterwards would undo that --
        // on M=128 it turned a one-tile plan into 126 plus a two-row tail.
        const int l2_share = uarch.l2_bytes
                / std::max(std::min(plan.num_threads, uarch.cores_per_l2), 1);
        plan.MB = choose_even_mb(M, mb_cap, plan.MR, plan.num_threads, plan.KB,
                4, l2_share);
    }

    if (plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
        }
    }

    // Applied last so a sweep sees exactly the block sizes it asked for,
    // clamped only by what the loopers and microkernel require: MB a multiple
    // of MR, NB a multiple of NR, KB even (the packing routines step K in
    // pairs), and none of them past the problem dimension.
    static const int s_mb_env = env_block_override("ZENDNNL_NATIVE_GEMM_MB");
    static const int s_nb_env = env_block_override("ZENDNNL_NATIVE_GEMM_NB");
    static const int s_kb_env = env_block_override("ZENDNNL_NATIVE_GEMM_KB");
    if (s_mb_env > 0) {
        plan.MB = std::min(
                round_down(std::max(s_mb_env, plan.MR), plan.MR), M);
    }
    if (s_nb_env > 0) {
        plan.NB = std::min(
                round_down(std::max(s_nb_env, plan.NR), plan.NR), N);
    }
    if (s_kb_env > 0) {
        plan.KB = std::min(round_up(std::max(s_kb_env, 8), 8), K);
    }

    static bool s_log_fp32 = apilog_info_enabled();
    if (s_log_fp32) {
        apilog_info("Native FP32 GEMM plan: M=", M, " N=", N, " K=", K,
                " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                " MR=", plan.MR, " NR=", plan.NR,
                " threads=", plan.num_threads);
    }

    return FP32GemmPlan {plan};
}

// ============================================================================
// BF16 GEMM plan
// ============================================================================

BF16GemmPlan plan_bf16_gemm(const GemmDescriptor &desc,
        const UarchParams &uarch, const matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int K_padded = (K + 1) & ~1;

    bool has_activation = false;
    bool has_complex_activation = false;
    for (const auto &po : params.postop_) {
        auto pt = po.po_type;
        if (pt == post_op_type_t::relu || pt == post_op_type_t::leaky_relu) {
            has_activation = true;
        } else if (pt == post_op_type_t::gelu_tanh
                || pt == post_op_type_t::gelu_erf
                || pt == post_op_type_t::sigmoid || pt == post_op_type_t::tanh
                || pt == post_op_type_t::swish || pt == post_op_type_t::elu) {
            has_activation = true;
            has_complex_activation = true;
            break;
        }
    }

    static thread_local struct {
        int M, N, K, threads;
        bool transA, transB;
        BlockPlan plan;
    } s_plan_cache = {0, 0, 0, 0, false, false, {}};

    BlockPlan plan;
    if (s_plan_cache.M == M && s_plan_cache.N == N && s_plan_cache.K == K
            && s_plan_cache.threads == desc.num_threads
            && s_plan_cache.transA == desc.transA
            && s_plan_cache.transB == desc.transB) {
        plan = s_plan_cache.plan;
    } else {
        plan = plan_blocks(desc, uarch);
        plan.KB = (plan.KB + 1) & ~1;

        {
            int b_full_k_bytes
                    = K_padded * NR_PACK * static_cast<int>(sizeof(uint16_t));
            if (desc.num_threads <= 1) {
                int a_panel_bytes
                        = 6 * K_padded * static_cast<int>(sizeof(uint16_t));
                int l1_limit = static_cast<int>(0.8 * uarch.l1d_bytes);
                if (b_full_k_bytes <= uarch.l2_bytes / 2
                        && a_panel_bytes <= l1_limit) {
                    plan.KB = K_padded;
                }
            } else {
                int l2_for_a = uarch.l2_bytes / 2;
                int kb_a = l2_for_a / (6 * static_cast<int>(sizeof(uint16_t)));
                int kb_b = (uarch.l2_bytes / 2)
                        / (NR_PACK * static_cast<int>(sizeof(uint16_t)));
                int kb_max_mt = std::min(kb_a, kb_b);
                kb_max_mt = (kb_max_mt + 1) & ~1;
                if (K_padded <= kb_max_mt) {
                    plan.KB = K_padded;
                } else if (kb_max_mt > plan.KB) {
                    int n_blocks = (K_padded + kb_max_mt - 1) / kb_max_mt;
                    int even_kb
                            = ((K_padded + n_blocks - 1) / n_blocks + 7) & ~7;
                    plan.KB = (even_kb + 1) & ~1;
                }
            }
        }

        s_plan_cache
                = {M, N, K, desc.num_threads, desc.transA, desc.transB, plan};
    }

    const bool is_decode = (M <= 4);
    const char *path_name = "gemm";

    if (is_decode) {
        plan.MR = M;
        plan.MB = M;
        if (N >= 64)
            plan.NR = 64;
        else if (N >= 32)
            plan.NR = 32;
        else
            plan.NR = 16;
        path_name = "decode";
    } else if (N >= 64 && !has_complex_activation) {
        if (M % 6 == 0 || M >= 18) {
            plan.MR = 6;
        } else if (M % 4 == 0) {
            plan.MR = 4;
        } else if (M % 6 <= 3 && M > 12) {
            plan.MR = 4;
        } else {
            plan.MR = 6;
        }
        plan.NR = 64;
        path_name = "gemm-nr64";
    } else if (N >= 32) {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6 : (M % 4 == 0) ? 4 : 6;
        plan.NR = 32;
    } else {
        plan.MR = (M % 6 == 0 || M >= 18) ? 6 : (M % 4 == 0) ? 4 : 6;
        plan.NR = 16;
    }

    plan.NB = std::max(plan.NB / plan.NR * plan.NR, plan.NR);
    plan.NB = std::min(plan.NB, N);

    if (!is_decode) {
        bool will_pack_a
                = (desc.lda * static_cast<int>(sizeof(uint16_t)) > 4096);
        int mb_cap = 0;
        if (will_pack_a) {
            int b_lines_per_krow = ((plan.NR + 15) / 16) * 64;
            int k_pairs_kb = (plan.KB + 1) / 2;
            int b_accessed_bytes = k_pairs_kb * b_lines_per_krow;
            int l2_for_a = std::max(uarch.l2_bytes - b_accessed_bytes, 0);
            mb_cap = std::max(l2_for_a / (plan.KB * 2), plan.MR);
        } else {
            mb_cap = std::max((uarch.l1d_bytes + uarch.l2_bytes)
                            / (plan.NB * 4 + plan.KB * 2),
                    plan.MR);
        }
        // Same M-tiling problem the FP32 planner had: a cache budget alone can
        // leave a sliver as the last tile. Unlike the BRGEMM planners, which
        // rebalance when the tail falls below half a tile, nothing here caught
        // it. Validated on a Xeon 6767P, the only host here with avx512bf16:
        // 885 vs 591 GFLOPS at 1024x1024x1024 on eight threads (MB 132 vs 642).
        const int l2_share = uarch.l2_bytes
                / std::max(std::min(plan.num_threads, uarch.cores_per_l2), 1);
        plan.MB = choose_even_mb(M, mb_cap, plan.MR, plan.num_threads, plan.KB,
                2, l2_share);
    }

    if (!is_decode && plan.num_threads > 1) {
        int jc_tiles = (N + plan.NB - 1) / plan.NB;
        int ic_tiles = (M + plan.MB - 1) / plan.MB;
        int needed_ic = (plan.num_threads + jc_tiles - 1) / jc_tiles;
        if (needed_ic > ic_tiles && needed_ic > 1) {
            int m_panels = (M + plan.MR - 1) / plan.MR;
            int panels_per_block = std::max(m_panels / needed_ic, 1);
            plan.MB = panels_per_block * plan.MR;
            plan.MB = std::min(plan.MB, M);
            ic_tiles = (M + plan.MB - 1) / plan.MB;
        }
        if (ic_tiles * jc_tiles < plan.num_threads && plan.NB > plan.NR) {
            int needed_jc = (plan.num_threads + ic_tiles - 1) / ic_tiles;
            int nb_target = (N + needed_jc - 1) / needed_jc;
            nb_target = std::max((nb_target / plan.NR) * plan.NR, plan.NR);
            plan.NB = std::min(nb_target, plan.NB);
        }
    }

    static bool s_log_bf16 = apilog_info_enabled();
    if (s_log_bf16) {
        int jt = (N + plan.NB - 1) / plan.NB;
        int it = (M + plan.MB - 1) / plan.MB;
        apilog_info("Native BF16 GEMM plan: M=", M, " N=", N, " K=", K,
                " MB=", plan.MB, " NB=", plan.NB, " KB=", plan.KB,
                " MR=", plan.MR, " NR=", plan.NR, " path=", path_name,
                " ic=", it, " jc=", jt, " tiles=", it * jt,
                " threads=", plan.num_threads);
    }

    return BF16GemmPlan {
            plan, path_name, is_decode, has_activation, has_complex_activation};
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
