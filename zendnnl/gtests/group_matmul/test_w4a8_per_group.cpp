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

/// @file test_w4a8_per_group.cpp
/// @brief Grouped (MoE) W4A8 per-group matmul — the group-matmul analogue
///        of the single-matmul W4A8 tests, structured like
///        `test_ggml_per_group.cpp` for comprehensive MoE routing coverage.
///
/// Each expert owns:
///   * a bf16 source `[M, K]` with a per-token `{M, 1}` dynamic src_scale
///     (the runtime quantizes bf16 → s8 on the fly), and
///   * a per-group s4 weight `[K, N]` with per-group `{G, N}` wei_scale
///     (G = K / group_size).
///
/// The suite drives `group_matmul_direct` with many experts (15) and a sparse
/// active set (e.g. only 6 routed), modelling one MoE decode iteration.
/// Inactive experts carry `M == 0` (no routed tokens): the GEMM skips them.
/// Routed experts are validated against a per-expert reference matmul.
///
/// W4A8 on ALGO 3 (N-tile) currently falls back to ALGO 1 because
/// `check_n_tile_extra` rejects s4/u4 weights (the pre-OMP s4→s8 hoist
/// would mutate caller params that frameworks reuse across decode
/// iterations).  The prepack module still warms the W4A8 weight cache
/// (s4→s8 + AOCL sym-quant reorder) so the ALGO-1 fallback incurs no
/// first-call reorder spike.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest_utils.hpp"
#include "group_matmul_test_helpers.hpp"
#include "moe_test_utils.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_parallel_common.hpp"
#include "lowoha_operators/matmul/group_matmul/prepack/prepack.hpp"

namespace {

/// Build every expert with per-group s4 weights and a bf16 source with
/// dynamic per-token quantization, then drive `group_matmul_direct` with the
/// supplied per-expert active row counts (`rows[e] == 0` => inactive expert /
/// no routed tokens).  Routed experts are compared against a single-expert
/// reference matmul.
void run_w4a8_per_group_scenario(const std::string &label,
                                 const std::vector<int> &rows, uint64_t K,
                                 uint64_t N, uint64_t group_size) {
  ASSERT_EQ(K % group_size,
            0u) << label << ": K must be a multiple of group_size";
  const uint64_t G = K / group_size;
  ASSERT_GE(G, 2u) << label << ": need >= 2 groups for per-group scaling";

  const int E = static_cast<int>(rows.size());
  ASSERT_GT(E, 0) << label;

  reset_grp_matmul_caches();

  const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  const data_type_t out_dt = data_type_t::bf16;
  const data_type_t scale_dt = data_type_t::bf16;

  tensor_factory_t tf;

  std::vector<tensor_t> inp(E), wt(E), bias(E), out(E), out_ref(E);
  std::vector<int> active(E);

  const int64_t saved_seed = seed;

  for (int e = 0; e < E; ++e) {
    active[e] = rows[e];
    seed = saved_seed + 1 + static_cast<int64_t>(e);

    const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

    // ── Per-group s4 weight [K, N] with per-group {G, N} scale ──
    auto wei_scale = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
    wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::s4, 7.0, false,
                                   wei_scale);

    // ── bf16 source [Mbuf, K] with per-token {Mbuf, 1} dynamic scale ──
    auto src_scale = tf.zero_tensor({Mbuf, 1u}, scale_dt);
    inp[e] = tf.uniform_dist_tensor({Mbuf, K}, data_type_t::bf16, 2.0,
                                    false, src_scale, tensor_t());

    bias[e] = tf.uniform_dist_tensor({1u, N}, out_dt, 2.0);
    out[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_ref[e] = tf.zero_tensor({Mbuf, N}, out_dt);
  }
  seed = saved_seed;

  // ── Drive the grouped W4A8 path ──
  status_t st = group_matmul_kernel_test(inp, wt, bias, out, algo, 1.0f,
                                         0.0f, /*moe_postop=*/nullptr,
                                         /*gated_act=*/nullptr,
                                         /*pack_format_b=*/{}, active);
  ASSERT_EQ(st, status_t::success) << label << ": group_matmul_direct failed";

  // ── Compare every routed expert against single-expert reference ──
  const std::vector<post_op_type_t> ref_po;
  for (int e = 0; e < E; ++e) {
    if (rows[e] == 0) {
      continue;
    }
    std::vector<tensor_t> bin;
    status_t rst = matmul_forced_ref_kernel_test(inp[e], wt[e], bias[e],
                   out_ref[e], ref_po, bin,
                   /*use_LOWOHA=*/true, algo,
                   1.0f, 0.0f);
    ASSERT_EQ(rst, status_t::success)
        << label << ": reference failed (expert " << e << ")";
    bool expert_ok = true;
    // W4A8 tolerance: the s4→s8 conversion + bf16 dynamic quant introduces
    // more noise than pure INT8 per-group, so use a generous 128x epsilon.
    compare_tensor_2D_matrix(out[e], out_ref[e], static_cast<uint64_t>(rows[e]),
                             N, K, rtol_bf16, 128.0f * epsilon_bf16, expert_ok,
                             /*enable_f32_relaxation=*/false, 1.0f, true);
    EXPECT_TRUE(expert_ok) << label << ": output mismatch on expert " << e
                           << " (rows=" << rows[e] << ")";
  }
}

void run_w4a8_cross_algo_scenario(const std::string &label,
                                  const std::vector<int> &rows, uint64_t K,
                                  uint64_t N, uint64_t group_size) {
  ASSERT_EQ(K % group_size,
            0u) << label << ": K must be a multiple of group_size";
  const uint64_t G = K / group_size;
  const int E = static_cast<int>(rows.size());
  ASSERT_GT(E, 0) << label;

  reset_grp_matmul_caches();

  const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  const data_type_t scale_dt = data_type_t::bf16;
  const data_type_t out_dt = data_type_t::bf16;

  tensor_factory_t tf;
  std::vector<tensor_t> inp(E), wt(E), bias(E), out_a0(E), out_a1(E),
      out_a2(E), out_a3(E), out_a4(E), out_ref(E);
  std::vector<int> active(E);

  const int64_t saved_seed = seed;
  for (int e = 0; e < E; ++e) {
    active[e] = rows[e];
    seed = saved_seed + 1 + static_cast<int64_t>(e);
    const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

    auto wei_scale = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
    wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::s4, 7.0, false,
                                   wei_scale);
    auto src_scale = tf.zero_tensor({Mbuf, 1u}, scale_dt);
    inp[e] = tf.uniform_dist_tensor({Mbuf, K}, data_type_t::bf16, 2.0,
                                    false, src_scale, tensor_t());
    bias[e] = tf.zero_tensor({1u, N}, out_dt);
    out_a0[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_a1[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_a2[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_a3[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_a4[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_ref[e] = tf.zero_tensor({Mbuf, N}, out_dt);
  }
  seed = saved_seed;

  status_t st;
  {
    moe_test_utils::AlgoEnvGuard g(1);
    st = group_matmul_kernel_test(inp, wt, bias, out_a1, algo, 1.0f, 0.0f,
                                  nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(st, status_t::success) << label << ": ALGO 1 failed";

  {
    moe_test_utils::AlgoEnvGuard g(0);
    st = group_matmul_kernel_test(inp, wt, bias, out_a0, algo, 1.0f, 0.0f,
                                  nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(st, status_t::success) << label << ": ALGO 0 failed";

  {
    moe_test_utils::AlgoEnvGuard g(2);
    st = group_matmul_kernel_test(inp, wt, bias, out_a2, algo, 1.0f, 0.0f,
                                  nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(st, status_t::success) << label << ": ALGO 2 failed";

  {
    moe_test_utils::AlgoEnvGuard g(3);
    st = group_matmul_kernel_test(inp, wt, bias, out_a3, algo, 1.0f, 0.0f,
                                  nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(st, status_t::success) << label << ": ALGO 3 fallback failed";

  {
    moe_test_utils::AlgoEnvGuard g(4);
    st = group_matmul_kernel_test(inp, wt, bias, out_a4, algo, 1.0f, 0.0f,
                                  nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(st, status_t::success) << label << ": ALGO 4 failed";

  {
    moe_test_utils::AlgoEnvGuard g(1);
    const std::vector<post_op_type_t> ref_po;
    for (int e = 0; e < E; ++e) {
      if (rows[e] == 0) {
        continue;
      }
      std::vector<tensor_t> bin;
      st = matmul_forced_ref_kernel_test(inp[e], wt[e], bias[e], out_ref[e],
                                         ref_po, bin, true, algo, 1.0f, 0.0f);
      ASSERT_EQ(st, status_t::success) << label << ": ref failed e=" << e;
    }
  }

  const float abs_tol = 128.0f * epsilon_bf16;
  for (int e = 0; e < E; ++e) {
    if (rows[e] == 0) {
      continue;
    }
    const uint64_t M_e = static_cast<uint64_t>(rows[e]);
    bool ok = true;
    compare_tensor_2D_matrix(out_a1[e], out_ref[e], M_e, N, K, rtol_bf16,
                             abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << label << ": ALGO 1 vs ref mismatch (e=" << e << ")";

    ok = true;
    compare_tensor_2D_matrix(out_a0[e], out_a1[e], M_e, N, K, rtol_bf16,
                             abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << label << ": ALGO 0 vs 1 mismatch (e=" << e << ")";

    ok = true;
    compare_tensor_2D_matrix(out_a2[e], out_a1[e], M_e, N, K, rtol_bf16,
                             abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << label << ": ALGO 2 vs 1 mismatch (e=" << e << ")";

    ok = true;
    compare_tensor_2D_matrix(out_a3[e], out_a1[e], M_e, N, K, rtol_bf16,
                             abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << label << ": ALGO 3 vs 1 mismatch (e=" << e << ")";

    ok = true;
    compare_tensor_2D_matrix(out_a4[e], out_a1[e], M_e, N, K, rtol_bf16,
                             abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << label << ": ALGO 4 vs 1 mismatch (e=" << e << ")";
  }
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════
// MoE routing-pattern scenarios (15 experts, sparse active sets)
// ═══════════════════════════════════════════════════════════════════════

// Headline scenario: 15 experts, only 6 routed (interleaved) — one MoE decode
// iteration that fires 6 of 15 experts.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveInterleavedBF16) {
  std::vector<int> rows(15, 0);
  for (int e : {
         1, 3, 5, 8, 11, 14
       }) rows[e] = 32;
  run_w4a8_per_group_scenario("15/6 interleaved bf16", rows, /*K=*/128,
                              /*N=*/64, /*group_size=*/32);
}

// First 6 experts routed.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveContiguousFirstBF16) {
  std::vector<int> rows(15, 0);
  for (int e = 0; e < 6; ++e) {
    rows[e] = 24;
  }
  run_w4a8_per_group_scenario("15/6 first-6 bf16", rows, 128, 48, 32);
}

// Last 6 experts routed => expert 0 is inactive (M[0] == 0).  Stresses the
// prepack representative-expert selection (must skip M[i] <= 0).
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSixActiveContiguousLastBF16) {
  std::vector<int> rows(15, 0);
  for (int e = 9; e < 15; ++e) {
    rows[e] = 16;
  }
  run_w4a8_per_group_scenario("15/6 last-6 bf16", rows, 256, 32, 32);
}

// Non-uniform token counts across the routed experts (incl. M == 1) — stresses
// per-token dynamic quant over ragged M and the row-level grouped quant
// scheduler.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsVariedTokenCountsBF16) {
  std::vector<int> rows(15, 0);
  const int idx[6] = {0, 2, 4, 6, 9, 13};
  const int act[6] = {1, 2, 3, 5, 8, 13};
  for (int j = 0; j < 6; ++j) {
    rows[idx[j]] = act[j];
  }
  run_w4a8_per_group_scenario("15 varied tokens bf16", rows, 128, 64, 32);
}

// Single routed expert in the middle of the inactive set.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsSingleActiveBF16) {
  std::vector<int> rows(15, 0);
  rows[7] = 8;
  run_w4a8_per_group_scenario("15/1 single active bf16", rows, 96, 80, 32);
}

// Dense routing: every expert fires (no inactive experts).
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsAllActiveBF16) {
  std::vector<int> rows(15, 12);
  run_w4a8_per_group_scenario("15/15 all active bf16", rows, 128, 64, 32);
}

// Degenerate routing: no expert fires (all M == 0).  The call must succeed
// and produce no output.
TEST(GroupMatmulW4A8PerGroup, FifteenExpertsNoneActiveBF16) {
  std::vector<int> rows(15, 0);
  run_w4a8_per_group_scenario("15/0 none active bf16", rows, 128, 64, 32);
}

// ═══════════════════════════════════════════════════════════════════════
// Group-size variations (W4A8 supports flexible group sizes, not just 32)
// ═══════════════════════════════════════════════════════════════════════

// group_size = 128 (Qwen3-style, K=4096)
TEST(GroupMatmulW4A8PerGroup, GroupSize128Qwen3BF16) {
  std::vector<int> rows(8, 0);
  for (int e : {
         0, 2, 4, 7
       }) rows[e] = 7;
  run_w4a8_per_group_scenario("gs128 qwen3 bf16", rows, 4096, 4096, 128);
}

// group_size = 64
TEST(GroupMatmulW4A8PerGroup, GroupSize64BF16) {
  std::vector<int> rows(15, 0);
  for (int e : {
         1, 5, 10, 14
       }) rows[e] = 16;
  run_w4a8_per_group_scenario("gs64 bf16", rows, 256, 128, 64);
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-algo accuracy coverage
// ═══════════════════════════════════════════════════════════════════════

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSmallDecodeBF16) {
  run_w4a8_cross_algo_scenario("small", std::vector<int>(4, 32),
                               128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoMidRangeBF16) {
  run_w4a8_cross_algo_scenario("mid", std::vector<int>(10, 128),
                               128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSingleTokenBF16) {
  run_w4a8_cross_algo_scenario("M1", std::vector<int>(20, 1),
                               128, 64, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoSquareBF16) {
  run_w4a8_cross_algo_scenario("square", std::vector<int>(8, 16),
                               128, 128, 32);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoQwen3DecodeBF16) {
  run_w4a8_cross_algo_scenario("qwen3", std::vector<int>(8, 7),
                               4096, 4096, 128);
}

TEST(GroupMatmulW4A8PerGroup, CrossAlgoQwen3ManyOpsBF16) {
  run_w4a8_cross_algo_scenario("qwen16", std::vector<int>(16, 1),
                               4096, 4096, 128);
}

// ═══════════════════════════════════════════════════════════════════════
// ALGO 3 (N-tile) fallback validation
// ═══════════════════════════════════════════════════════════════════════

// W4A8 on ALGO 3 must fall back to ALGO 1 and still produce correct output.
// Pin ALGO 3 and validate the prepack invocation sees the call (the W4A8
// cache is warmed by prepack), even though execution routes to ALGO 1.
TEST(GroupMatmulW4A8PerGroup, Algo3FallbackToAlgo1BF16) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  moe_test_utils::AlgoEnvGuard              algo3(3);
  moe_test_utils::LastInvocationCaptureGuard prepack_capture;
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  std::vector<int> rows(15, 0);
  for (int e : {
         1, 3, 5, 8, 11, 14
       }) rows[e] = 32;
  run_w4a8_per_group_scenario("15/6 algo3 fallback bf16", rows, /*K=*/128,
                              /*N=*/64, /*group_size=*/32);

  auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid)
      << "prepack must run for the W4A8 per-group call even when ALGO 3 "
      "falls back to ALGO 1";
}

// Cross-algo comparison: ALGO 1 and ALGO 3 (fallback) must produce identical
// output for the same W4A8 input.
TEST(GroupMatmulW4A8PerGroup, Algo3MatchesAlgo1BF16) {
  const int E = 15;
  const uint64_t K = 128, N = 64, group_size = 32;
  const uint64_t G = K / group_size;
  const data_type_t scale_dt = data_type_t::bf16;
  const data_type_t out_dt = data_type_t::bf16;

  reset_grp_matmul_caches();

  tensor_factory_t tf;
  std::vector<tensor_t> inp(E), wt(E), bias(E), out_a1(E), out_a3(E);
  std::vector<int> active(E);

  std::vector<int> rows(E, 0);
  for (int e : {
         1, 3, 5, 8, 11, 14
       }) rows[e] = 32;

  const int64_t saved_seed = seed;
  for (int e = 0; e < E; ++e) {
    active[e] = rows[e];
    seed = saved_seed + 1 + static_cast<int64_t>(e);
    const uint64_t Mbuf = static_cast<uint64_t>(rows[e] > 0 ? rows[e] : 1);

    auto ws = tf.uniform_dist_tensor({G, N}, scale_dt, 2.0);
    wt[e] = tf.uniform_dist_tensor({K, N}, data_type_t::s4, 7.0, false, ws);
    auto ss = tf.zero_tensor({Mbuf, 1u}, scale_dt);
    inp[e] = tf.uniform_dist_tensor({Mbuf, K}, data_type_t::bf16, 2.0,
                                    false, ss, tensor_t());
    bias[e] = tf.zero_tensor({1u, N}, out_dt);
    out_a1[e] = tf.zero_tensor({Mbuf, N}, out_dt);
    out_a3[e] = tf.zero_tensor({Mbuf, N}, out_dt);
  }
  seed = saved_seed;

  const matmul_algo_t algo = matmul_algo_t::aocl_dlp_blocked;
  status_t s;

  {
    moe_test_utils::AlgoEnvGuard g(1);
    reset_grp_matmul_caches();
    s = group_matmul_kernel_test(inp, wt, bias, out_a1, algo, 1.0f, 0.0f,
                                 nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(s, status_t::success) << "ALGO 1 failed";

  {
    moe_test_utils::AlgoEnvGuard g(3);
    reset_grp_matmul_caches();
    s = group_matmul_kernel_test(inp, wt, bias, out_a3, algo, 1.0f, 0.0f,
                                 nullptr, nullptr, {}, active);
  }
  ASSERT_EQ(s, status_t::success) << "ALGO 3 (fallback) failed";

  const float abs_tol = 128.0f * epsilon_bf16;
  for (int e = 0; e < E; ++e) {
    if (rows[e] == 0) {
      continue;
    }
    bool ok = true;
    compare_tensor_2D_matrix(out_a3[e], out_a1[e],
                             static_cast<uint64_t>(rows[e]), N, K,
                             rtol_bf16, abs_tol, ok, false, 1.0f, true);
    EXPECT_TRUE(ok) << "ALGO 3 vs ALGO 1 mismatch on expert " << e
                    << " — fallback should produce identical output";
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Prepack W4A8 cache warming validation
// ═══════════════════════════════════════════════════════════════════════

// Validate that prepack warms the W4A8 cache for ALL experts (including
// inactive ones) so a later decode iteration that routes to a currently-cold
// expert pays no first-fire reorder spike.
TEST(GroupMatmulW4A8PerGroup, PrepackWarmsAllExpertsBF16) {
  namespace prepack = zendnnl::lowoha::matmul::group_matmul_prepack;
  moe_test_utils::LastInvocationCaptureGuard prepack_capture;
  prepack::clear_fingerprint_cache_for_test();
  prepack::test_api::clear_last_invocation_stats();

  std::vector<int> rows(15, 0);
  for (int e : {
         1, 3, 5, 8, 11, 14
       }) rows[e] = 32;  // 6 of 15 routed
  run_w4a8_per_group_scenario("15/6 prepack warm", rows, /*K=*/128,
                              /*N=*/64, /*group_size=*/32);

  auto stats = prepack::test_api::get_last_invocation_stats();
  ASSERT_TRUE(stats.valid)
      << "prepack must fire for the W4A8 per-group call";
  // The prepack warmer should attempt all 15 experts (total_attempted >= 15),
  // not just the 6 routed ones.
  EXPECT_GE(stats.aocl.total_attempted, 15)
      << "prepack must warm all 15 experts' W4A8 weight cache (not just the "
      "6 routed ones) to eliminate first-fire reorder spikes on "
      "rotating-experts MoE patterns";
}
