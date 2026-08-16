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
// Symmetric per-group INT8 GEMM: microkernel, scalar tail, and looper.
//
// Most of these call the native entry points directly, one layer below the
// matmul API, so a kernel or looper fault is not filtered through dispatch. The
// Int8SymqDispatch suite at the bottom then drives the public API, covering what
// direct calls cannot: that a per-group INT8 call actually arrives here rather
// than at AOCL-DLP, which declines without AVX-512 VNNI and returns without
// computing.
//
// FLAVOURS. select_int8_symq_ukernel_128() prefers XOP and caches its choice in
// a function-local static, so one process exercises exactly one flavour. Which
// one is reported by the ReportsWhichFlavourRan test below, and the nightly
// script runs this suite twice, the second time with
// ZENDNNL_NATIVE_SYMQ_NO_XOP=1, so both are covered on family 15h. Do not add a
// test that expects a particular flavour: only family 15h has XOP.
//
// TOLERANCE. Errors are scaled by max|reference| over the tensor, not by each
// element's own magnitude. A C element is a sum of K/group_size signed group
// terms, so elements routinely cancel to near zero while their absolute error
// stays at the rounding of the terms that built them; a per-element relative
// test reports 1e-3 on a kernel that is exact to fp32. Normalising by the
// tensor's own scale is the usual GEMM criterion and still catches a
// scale-indexing slip, which moves elements by their full magnitude. Cases with
// positive-only data are included as well: those cannot cancel, so every
// element carries its own weight there.
// ============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "gtest_utils.hpp"
#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_kquant_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q4k_gemv_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_entry_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_looper_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_entry_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {
namespace {

constexpr double kTolerance = 1e-6;

// Named so the scalar-scale call sites have something to take the address of.
constexpr float kOne = 1.0f;
constexpr float kPointOne = 0.01f;

// Reference. int64 accumulation, so unlike the kernel it cannot saturate even
// if the contract were violated, and double for the scale sum. The group and
// scale indexing is written out from the header's contract rather than borrowed
// from the looper: an off-by-one in the group-major scale layout is the failure
// most likely to look plausible, and sharing the indexing would hide it.
// ss is indexed the way the kernel indexes it: ss[m * ss_row + g * ss_grp], so
// one reference covers per-tensor, per-token and per-group activation scales.
void reference_gemm(int M, int N, int K, int gs, const int8_t *A, int lda,
        const int8_t *B, int ldb, bool transB, const float *ws, const float *ss,
        std::vector<float> &out, int ss_row = 0, int ss_grp = 0) {
    const int n_groups = K / gs;
    out.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int g = 0; g < n_groups; ++g) {
                int64_t acc = 0;
                for (int j = 0; j < gs; ++j) {
                    const int k = g * gs + j;
                    const int64_t a = A[static_cast<size_t>(m) * lda + k];
                    const int64_t b = transB
                            ? B[static_cast<size_t>(n) * ldb + k]
                            : B[static_cast<size_t>(k) * ldb + n];
                    acc += a * b;
                }
                sum += static_cast<double>(acc)
                        * static_cast<double>(
                                ws[static_cast<size_t>(g) * N + n])
                        * static_cast<double>(
                                ss[m * ss_row + g * ss_grp]);
            }
            out[static_cast<size_t>(m) * N + n] = static_cast<float>(sum);
        }
    }
}

// Worst error over the compared extent, scaled by that extent's own magnitude;
// see the tolerance note at the top. An unwritten element counts as an outright
// failure rather than a large error.
//
// ld_got and ld_ref are separate from the extent on purpose: the tail kernel
// claims only an mr x nr sub-block of a full-width tile, so the extent being
// compared is narrower than the row stride of either buffer. Folding the two
// together silently reads the wrong row for every m >= 1.
double worst_scaled_error(const std::vector<float> &got, int ld_got,
        const std::vector<float> &ref, int ld_ref, int M, int N,
        int *out_nan_count) {
    double max_abs = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            max_abs = std::max(max_abs,
                    static_cast<double>(std::fabs(
                            ref[static_cast<size_t>(m) * ld_ref + n])));
        }
    }
    const double scale = std::max(max_abs, 1e-30);

    double worst = 0.0;
    int nans = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const float g = got[static_cast<size_t>(m) * ld_got + n];
            if (std::isnan(g)) {
                ++nans;
                continue;
            }
            const double e = std::fabs(static_cast<double>(g)
                                     - static_cast<double>(ref[static_cast<size_t>(
                                                                       m)
                                                       * ld_ref
                                               + n]))
                    / scale;
            worst = std::max(worst, e);
        }
    }
    *out_nan_count = nans;
    return worst;
}

// All bytes in [-127, 127]: the kernel's precondition. -128 is excluded because
// PSIGNB cannot negate it and because a lane of two 128*128 products would
// saturate -- both silently. Every GGML quantiser meets this bound.
void fill_s8(std::vector<int8_t> &v, std::mt19937 &rng, bool positive_only) {
    std::uniform_int_distribution<int> d(positive_only ? 1 : -127, 127);
    for (auto &x : v) x = static_cast<int8_t>(d(rng));
}

void fill_scales(std::vector<float> &v, std::mt19937 &rng) {
    std::uniform_real_distribution<float> d(0.002f, 0.05f);
    for (auto &x : v) x = d(rng);
}

// Poison, so a path that fails to write an element cannot pass by leaving a
// plausible value behind.
std::vector<float> poisoned(size_t n) {
    return std::vector<float>(n, std::numeric_limits<float>::quiet_NaN());
}

// Pack B into the VNNI layout the microkernel reads: for each group of four
// consecutive K, `width` columns lie contiguously, four bytes each.
void pack_b_vnni(const int8_t *B, int ldb, bool transB, int K, int width,
        int col0, int N, std::vector<int8_t> &dst) {
    const int n_quads = K / SYMQ_VNNI_GRP;
    dst.assign(static_cast<size_t>(n_quads) * width * SYMQ_VNNI_GRP, 0);
    for (int kq = 0; kq < n_quads; ++kq) {
        for (int n = 0; n < width; ++n) {
            for (int j = 0; j < SYMQ_VNNI_GRP; ++j) {
                const int k = kq * SYMQ_VNNI_GRP + j;
                const int col = col0 + n;
                if (col >= N) continue;
                dst[static_cast<size_t>(kq) * width * SYMQ_VNNI_GRP
                        + n * SYMQ_VNNI_GRP + j]
                        = transB ? B[static_cast<size_t>(col) * ldb + k]
                                 : B[static_cast<size_t>(k) * ldb + col];
            }
        }
    }
}

// ---------------------------------------------------------------- microkernel

TEST(Int8SymqUkernel128, ReportsWhichFlavourRan) {
    const int8_symq_ukernel_128_fn_t fn = select_int8_symq_ukernel_128();
    if (fn == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";
    // Not an assertion: which flavour is correct depends on the host. Recorded
    // so a log makes clear which one the rest of this suite exercised.
    RecordProperty("ukernel_selected", "yes");
    SUCCEED() << "a 128-bit INT8 microkernel was selected; set "
                 "ZENDNNL_NATIVE_SYMQ_NO_XOP=1 to exercise the portable "
                 "flavour on a host that has XOP";
}

// One MR x NR tile straight into the microkernel, across the group sizes GGML
// produces (32 for Q4_0 and Q8_0, 16 for Q6_K), a group spanning the whole tile,
// and a K long enough to flush many times.
TEST(Int8SymqUkernel128, TileMatchesInt64Reference) {
    const int8_symq_ukernel_128_fn_t hot = select_int8_symq_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    struct { int K, gs; bool positive; } cases[] = {
            {32, 32, false}, {64, 16, false}, {128, 32, false},
            {128, 16, true}, {256, 64, false}, {4096, 32, false},
            {4096, 16, false}, {32, 32, true},
    };

    std::mt19937 rng(20260814);
    for (const auto &c : cases) {
        SCOPED_TRACE(testing::Message() << "K=" << c.K << " gs=" << c.gs
                                        << " positive=" << c.positive);
        const int M = SYMQ_MR, N = SYMQ_NR;
        const int n_groups = c.K / c.gs;
        std::vector<int8_t> A(static_cast<size_t>(M) * c.K);
        std::vector<int8_t> B(static_cast<size_t>(c.K) * N);
        std::vector<float> ws(static_cast<size_t>(n_groups) * N);
        fill_s8(A, rng, c.positive);
        fill_s8(B, rng, c.positive);
        fill_scales(ws, rng);
        const float ss = 0.0137f;

        std::vector<int8_t> packed;
        pack_b_vnni(B.data(), N, /*transB=*/false, c.K, N, 0, N, packed);

        // The microkernel accumulates into C rather than overwriting it, so the
        // caller seeds it; zero here is the seed the looper uses.
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), c.K, packed.data(), N * SYMQ_VNNI_GRP, C.data(), N, c.K,
                c.gs, ws.data(), N, &ss, 0, 0);

        std::vector<float> ref;
        reference_gemm(M, N, c.K, c.gs, A.data(), c.K, B.data(), N,
                /*transB=*/false, ws.data(), &ss, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
        EXPECT_EQ(nans, 0);
    }
}

// The contract's extremes, which is where this kernel was originally wrong: it
// first claimed -128 was admissible. At +/-127 a PMADDUBSW lane reaches 32258
// against the signed 16-bit limit of 32767, so the arithmetic must be exact;
// at -128 it would saturate and PSIGNB would produce the wrong sign, both
// silently. Driving every byte to the bound is what caught that.
TEST(Int8SymqUkernel128, IsExactAtTheContractExtremes) {
    const int8_symq_ukernel_128_fn_t hot = select_int8_symq_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = SYMQ_MR, N = SYMQ_NR, K = 256, gs = 32;
    for (int pattern = 0; pattern < 4; ++pattern) {
        SCOPED_TRACE(testing::Message() << "sign pattern " << pattern);
        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        std::vector<int8_t> B(static_cast<size_t>(K) * N);
        // Every byte at +/-127, in four sign arrangements: all positive, all
        // negative, and both mixed, so the sign trick is exercised in each
        // direction on both operands.
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = ((pattern & 1) && (i % 2)) ? -127 : 127;
        for (size_t i = 0; i < B.size(); ++i)
            B[i] = ((pattern & 2) && (i % 3)) ? -127 : 127;

        std::vector<float> ws(static_cast<size_t>(K / gs) * N, 1.0f);
        std::vector<int8_t> packed;
        pack_b_vnni(B.data(), N, false, K, N, 0, N, packed);
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), K, packed.data(), N * SYMQ_VNNI_GRP, C.data(), N, K, gs,
                ws.data(), N, &kOne, 0, 0);

        std::vector<float> ref;
        reference_gemm(M, N, K, gs, A.data(), K, B.data(), N, false, ws.data(),
                &kOne, ref);
        // Unit scales and integral sums: these values are representable
        // exactly in fp32, so anything but equality means saturation.
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                EXPECT_FLOAT_EQ(C[m * N + n], ref[m * N + n])
                        << "at (" << m << "," << n << ")";
    }
}

// The scalar tail finishes ragged tiles, so it must agree with the reference at
// every partial extent -- including nr_act below four, where the vector path
// has no expression at all.
TEST(Int8SymqUkernel128, TailMatchesReferenceForEveryRaggedExtent) {
    const int K = 64, gs = 16;
    std::mt19937 rng(99);
    for (int mr = 1; mr <= SYMQ_MR; ++mr) {
        for (int nr = 1; nr <= SYMQ_NR; ++nr) {
            SCOPED_TRACE(testing::Message() << "mr_act=" << mr << " nr_act=" << nr);
            const int N = SYMQ_NR;
            std::vector<int8_t> A(static_cast<size_t>(SYMQ_MR) * K);
            std::vector<int8_t> B(static_cast<size_t>(K) * N);
            std::vector<float> ws(static_cast<size_t>(K / gs) * N);
            fill_s8(A, rng, false);
            fill_s8(B, rng, false);
            fill_scales(ws, rng);
            const float ss = 0.0137f;

            std::vector<int8_t> packed;
            pack_b_vnni(B.data(), N, false, K, N, 0, N, packed);
            std::vector<float> C(static_cast<size_t>(SYMQ_MR) * N, 0.0f);
            int8_symq_tail_128(A.data(), K, packed.data(), N * SYMQ_VNNI_GRP,
                    C.data(), N, K, gs, mr, nr, ws.data(), N, &ss, 0, 0);

            std::vector<float> ref;
            reference_gemm(SYMQ_MR, N, K, gs, A.data(), K, B.data(), N, false,
                    ws.data(), &ss, ref);
            int nans = 0;
            // Only the mr x nr sub-block is claimed; the rest must be
            // untouched, which zero-seeded C makes checkable.
            // ld_ref is the full tile width even though only mr x nr is claimed.
            EXPECT_LT(worst_scaled_error(C, N, ref, N, mr, nr, &nans),
                    kTolerance);
            EXPECT_EQ(nans, 0);
            for (int m = 0; m < SYMQ_MR; ++m) {
                for (int n = 0; n < N; ++n) {
                    if (m >= mr || n >= nr) {
                        EXPECT_FLOAT_EQ(C[m * N + n], 0.0f)
                                << "tail wrote outside its extent at (" << m
                                << "," << n << ")";
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------- looper

struct LooperShape {
    int M, N, K, gs;
    bool transB;
    int nthreads;
    bool positive;
    const char *why;
};

class Int8SymqLooper128 : public ::testing::TestWithParam<LooperShape> {};

TEST_P(Int8SymqLooper128, MatchesInt64Reference) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const LooperShape s = GetParam();
    const int lda = s.K;
    const int ldb = s.transB ? s.K : s.N;
    const int ldc = s.N;
    const int n_groups = s.K / s.gs;

    std::mt19937 rng(4242);
    std::vector<int8_t> A(static_cast<size_t>(s.M) * lda);
    std::vector<int8_t> B(s.transB ? static_cast<size_t>(s.N) * ldb
                                   : static_cast<size_t>(s.K) * ldb);
    std::vector<float> ws(static_cast<size_t>(n_groups) * s.N);
    fill_s8(A, rng, s.positive);
    fill_s8(B, rng, s.positive);
    fill_scales(ws, rng);
    const float ss = 0.0137f;

    std::vector<float> C = poisoned(static_cast<size_t>(s.M) * ldc);
    ASSERT_TRUE(int8_symq_execute_128(s.M, s.N, s.K, s.gs, A.data(), lda,
            B.data(), ldb, s.transB, C.data(), ldc, ws.data(), &ss, 0, 0, s.nthreads))
            << "looper declined a shape it should express";

    std::vector<float> ref;
    reference_gemm(s.M, s.N, s.K, s.gs, A.data(), lda, B.data(), ldb, s.transB,
            ws.data(), &ss, ref);
    int nans = 0;
    const double err = worst_scaled_error(C, ldc, ref, s.N, s.M, s.N, &nans);
    EXPECT_EQ(nans, 0) << nans << " output elements were never written";
    EXPECT_LT(err, kTolerance);
}

INSTANTIATE_TEST_SUITE_P(Shapes, Int8SymqLooper128,
        ::testing::Values(
                LooperShape {4, 8, 32, 32, false, 1, false, "one tile, one group"},
                LooperShape {4, 8, 64, 16, false, 1, false, "Q6_K group size"},
                LooperShape {1, 512, 512, 32, true, 4, false, "decode, GGML layout"},
                LooperShape {7, 13, 96, 32, false, 1, false, "ragged M and N"},
                LooperShape {5, 70, 128, 16, false, 2, false, "crosses panel tail"},
                LooperShape {128, 256, 512, 32, true, 4, false, "prompt GEMM"},
                LooperShape {3, 64, 32, 32, true, 1, false, "M below MR"},
                LooperShape {4, 8, 4096, 32, false, 1, false, "long K"},
                LooperShape {64, 64, 64, 64, false, 2, false, "group spans K"},
                LooperShape {2, 9, 48, 16, true, 3, false, "odd everything"},
                LooperShape {4, 65, 32, 32, false, 4, false, "N one past a panel"},
                LooperShape {256, 32, 256, 32, false, 4, false, "tall M"},
                LooperShape {68, 130, 256, 32, true, 4, true, "positive only"},
                LooperShape {4, 8, 128, 16, false, 1, true, "positive, gs=16"}),
        [](const ::testing::TestParamInfo<LooperShape> &i) {
            // gtest requires alphanumerics and underscores only, and aborts the
            // whole binary (not just this suite) if a name violates that.
            std::string n(i.param.why);
            for (char &c : n)
                if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
            return n;
        });

// Shapes the microkernel cannot express are refused rather than approximated,
// so a caller can fall back. C must be left alone: a partially written output
// would be worse than a refusal.
TEST(Int8SymqLooper128, DeclinesShapesItCannotExpress) {
    struct { int M, N, K, gs; const char *why; } bad[] = {
            {1, 8, 30, 32, "K is not a whole number of groups"},
            {1, 8, 32, 6, "group size splits a VNNI quad"},
            {1, 8, 96, 0, "group size zero"},
            {0, 8, 32, 32, "empty M"},
            {1, 0, 32, 32, "empty N"},
            {1, 8, 0, 32, "empty K"},
    };
    for (const auto &b : bad) {
        SCOPED_TRACE(b.why);
        std::vector<int8_t> A(256, 1), B(256, 1);
        std::vector<float> ws(256, 1.0f);
        std::vector<float> C(64, -7.0f);
        EXPECT_FALSE(int8_symq_execute_128(b.M, b.N, b.K, b.gs, A.data(),
                std::max(b.K, 1), B.data(), std::max(b.N, 1), false, C.data(),
                8, ws.data(), &kOne, 0, 0, 1));
        for (float v : C) EXPECT_FLOAT_EQ(v, -7.0f) << "C was written";
    }
}

// The looper overwrites C rather than accumulating (this path has no beta), so
// a stale buffer must not survive into the result.
TEST(Int8SymqLooper128, OverwritesRatherThanAccumulates) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = 5, N = 70, K = 64, gs = 32;
    std::mt19937 rng(7);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(K) * N);
    std::vector<float> ws(static_cast<size_t>(K / gs) * N);
    fill_s8(A, rng, false);
    fill_s8(B, rng, false);
    fill_scales(ws, rng);

    std::vector<float> first(static_cast<size_t>(M) * N, 0.0f);
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), N,
            false, first.data(), N, ws.data(), &kPointOne, 0, 0, 2));

    // Same inputs into a buffer full of garbage must give the same answer.
    std::vector<float> second(static_cast<size_t>(M) * N, 12345.0f);
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), N,
            false, second.data(), N, ws.data(), &kPointOne, 0, 0, 2));
    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_FLOAT_EQ(first[i], second[i]) << "at " << i;
}

// C rows are addressed by ldc, which the looper must honour rather than
// assuming a packed N -- the GGML caller hands it a slice of a wider buffer.
TEST(Int8SymqLooper128, HonoursLdcAndLeavesPaddingAlone) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = 9, N = 20, K = 64, gs = 32, ldc = 37;
    std::mt19937 rng(11);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(K) * N);
    std::vector<float> ws(static_cast<size_t>(K / gs) * N);
    fill_s8(A, rng, false);
    fill_s8(B, rng, false);
    fill_scales(ws, rng);

    std::vector<float> C(static_cast<size_t>(M) * ldc, -99.0f);
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), N,
            false, C.data(), ldc, ws.data(), &kPointOne, 0, 0, 2));

    std::vector<float> ref;
    reference_gemm(M, N, K, gs, A.data(), K, B.data(), N, false, ws.data(),
            &kPointOne, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, ldc, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0);
    for (int m = 0; m < M; ++m)
        for (int n = N; n < ldc; ++n)
            EXPECT_FLOAT_EQ(C[static_cast<size_t>(m) * ldc + n], -99.0f)
                    << "padding written at (" << m << "," << n << ")";
}

// Thread count must not change the answer: threads own disjoint column panels,
// so any difference means the partitioning overlaps or leaves a gap.
TEST(Int8SymqLooper128, IsInvariantInThreadCount) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = 33, N = 200, K = 128, gs = 32;
    std::mt19937 rng(5);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(N) * K); // transB
    std::vector<float> ws(static_cast<size_t>(K / gs) * N);
    fill_s8(A, rng, false);
    fill_s8(B, rng, false);
    fill_scales(ws, rng);

    std::vector<float> single = poisoned(static_cast<size_t>(M) * N);
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), K,
            true, single.data(), N, ws.data(), &kPointOne, 0, 0, 1));

    for (int nt : {2, 3, 4, 8}) {
        SCOPED_TRACE(testing::Message() << "nthreads=" << nt);
        std::vector<float> many = poisoned(static_cast<size_t>(M) * N);
        ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), K,
                true, many.data(), N, ws.data(), &kPointOne, 0, 0, nt));
        for (size_t i = 0; i < single.size(); ++i)
            ASSERT_FLOAT_EQ(single[i], many[i]) << "at " << i;
    }
}

// ============================================================================
// Asymmetric per-group (GGML k-quant) microkernel.
//
// The reference below dequantises each weight the way GGML defines it,
// w = D * q - M, and then multiplies. The kernel instead computes
// sum(D * dot) - sum(M * rowsum), which is the same quantity rearranged so the
// min term costs one multiply per group instead of one per element. Building the
// reference from the factored form would make the test agree with the algebra
// rather than check it, which is the whole risk in this kernel.
// ============================================================================

namespace {

// out[k, n] semantics with a transposed (GGML) B: q is indexed [n][k].
void reference_kquant(int M, int N, int K, int gs, const int8_t *A, int lda,
        const uint8_t *q, int ldq, const float *D, const float *Min,
        const float *ss, int ss_row, int ss_grp, std::vector<float> &out) {
    const int n_groups = K / gs;
    out.assign(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int g = 0; g < n_groups; ++g) {
                double gsum = 0.0;
                for (int j = 0; j < gs; ++j) {
                    const int k = g * gs + j;
                    // Dequantise, then multiply -- the definition, not the
                    // rearrangement the kernel uses.
                    const double w
                            = static_cast<double>(D[static_cast<size_t>(g) * N + n])
                                    * static_cast<double>(
                                            q[static_cast<size_t>(n) * ldq + k])
                            - static_cast<double>(
                                    Min[static_cast<size_t>(g) * N + n]);
                    gsum += static_cast<double>(A[static_cast<size_t>(m) * lda + k])
                            * w;
                }
                sum += gsum * static_cast<double>(ss[m * ss_row + g * ss_grp]);
            }
            out[static_cast<size_t>(m) * N + n] = static_cast<float>(sum);
        }
    }
}

// Pack unsigned codes into the VNNI layout the kernel reads.
void pack_q_vnni(const uint8_t *q, int ldq, int K, int width, int N,
        std::vector<uint8_t> &dst) {
    const int n_quads = K / KQ_VNNI_GRP;
    dst.assign(static_cast<size_t>(n_quads) * width * KQ_VNNI_GRP, 0);
    for (int kq = 0; kq < n_quads; ++kq) {
        for (int n = 0; n < width; ++n) {
            if (n >= N) continue;
            for (int j = 0; j < KQ_VNNI_GRP; ++j) {
                const int k = kq * KQ_VNNI_GRP + j;
                dst[static_cast<size_t>(kq) * width * KQ_VNNI_GRP
                        + n * KQ_VNNI_GRP + j]
                        = q[static_cast<size_t>(n) * ldq + k];
            }
        }
    }
}

void fill_codes(std::vector<uint8_t> &v, std::mt19937 &rng, int max_code) {
    std::uniform_int_distribution<int> d(0, max_code);
    for (auto &x : v) x = static_cast<uint8_t>(d(rng));
}

} // namespace

TEST(Int8KquantUkernel128, TileMatchesTheDequantiseThenMultiplyReference) {
    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    // 15 is the Q4_K code ceiling, 31 the Q5_K one.
    struct { int K, gs, max_code; const char *why; } cases[] = {
            {32, 32, 15, "one group, Q4_K codes"},
            {32, 32, 31, "one group, Q5_K codes"},
            {256, 32, 15, "eight groups"},
            {256, 16, 31, "group 16"},
            {4096, 32, 15, "long K"},
            {64, 64, 15, "group spans the tile"},
    };

    std::mt19937 rng(515);
    for (const auto &c : cases) {
        SCOPED_TRACE(testing::Message() << c.why << " K=" << c.K
                                        << " gs=" << c.gs);
        const int M = KQ_MR, N = KQ_NR, groups = c.K / c.gs;
        std::vector<int8_t> A(static_cast<size_t>(M) * c.K);
        std::vector<uint8_t> q(static_cast<size_t>(N) * c.K);
        std::vector<float> D(static_cast<size_t>(groups) * N);
        std::vector<float> Min(static_cast<size_t>(groups) * N);
        fill_s8(A, rng, false);
        fill_codes(q, rng, c.max_code);
        fill_scales(D, rng);
        fill_scales(Min, rng);
        const float ss = 0.0137f;

        std::vector<uint8_t> packed;
        pack_q_vnni(q.data(), c.K, c.K, N, N, packed);
        std::vector<int32_t> rs(static_cast<size_t>(M) * groups);
        int8_kquant_row_sums(A.data(), c.K, M, c.K, c.gs, rs.data());

        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), c.K, packed.data(), N * KQ_VNNI_GRP, C.data(), N, c.K,
                c.gs, D.data(), Min.data(), N, rs.data(), groups, &ss, 0, 0);

        std::vector<float> ref;
        reference_kquant(M, N, c.K, c.gs, A.data(), c.K, q.data(), c.K, D.data(),
                Min.data(), &ss, 0, 0, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
        EXPECT_EQ(nans, 0);
    }
}

// The bound this kernel rests on: an unsigned code times a signed activation,
// summed in pairs, reaches 2 * 31 * 128 = 7936 against the 16-bit limit. So every
// s8 activation is admissible here including -128, which the symmetric kernel must
// exclude. Drive both extremes and require exactness, not a tolerance: with unit
// scales these sums are integers and representable.
TEST(Int8KquantUkernel128, IsExactAtTheExtremesIncludingMinus128) {
    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = KQ_MR, N = KQ_NR, K = 256, gs = 32, groups = K / gs;
    for (int pattern = 0; pattern < 4; ++pattern) {
        SCOPED_TRACE(testing::Message() << "pattern " << pattern);
        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        std::vector<uint8_t> q(static_cast<size_t>(N) * K);
        for (size_t i = 0; i < A.size(); ++i) {
            const bool neg = (pattern & 1) ? (i % 2) == 0 : (i % 3) == 0;
            A[i] = neg ? static_cast<int8_t>(-128) : static_cast<int8_t>(127);
        }
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = static_cast<uint8_t>((pattern & 2) ? 31 : 15);

        std::vector<float> D(static_cast<size_t>(groups) * N, 1.0f);
        std::vector<float> Min(static_cast<size_t>(groups) * N, 1.0f);
        const float ss = 1.0f;

        std::vector<uint8_t> packed;
        pack_q_vnni(q.data(), K, K, N, N, packed);
        std::vector<int32_t> rs(static_cast<size_t>(M) * groups);
        int8_kquant_row_sums(A.data(), K, M, K, gs, rs.data());

        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), K, packed.data(), N * KQ_VNNI_GRP, C.data(), N, K, gs,
                D.data(), Min.data(), N, rs.data(), groups, &ss, 0, 0);

        std::vector<float> ref;
        reference_kquant(M, N, K, gs, A.data(), K, q.data(), K, D.data(),
                Min.data(), &ss, 0, 0, ref);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n)
                EXPECT_FLOAT_EQ(C[m * N + n], ref[m * N + n])
                        << "at (" << m << "," << n << ")";
    }
}

// A zero min must reduce this kernel to a plain per-group dot product, and a zero
// code must leave only the min correction. Two degenerate cases that isolate the
// two terms from each other.
TEST(Int8KquantUkernel128, TermsIsolateWhenTheOtherIsZero) {
    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = KQ_MR, N = KQ_NR, K = 128, gs = 32, groups = K / gs;
    std::mt19937 rng(99);
    for (int which = 0; which < 2; ++which) {
        SCOPED_TRACE(which == 0 ? "min = 0, dot only" : "codes = 0, min only");
        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        std::vector<uint8_t> q(static_cast<size_t>(N) * K);
        fill_s8(A, rng, false);
        if (which == 0)
            fill_codes(q, rng, 15);
        else
            std::fill(q.begin(), q.end(), uint8_t {0});

        std::vector<float> D(static_cast<size_t>(groups) * N);
        std::vector<float> Min(static_cast<size_t>(groups) * N, 0.0f);
        fill_scales(D, rng);
        if (which == 1) fill_scales(Min, rng);
        const float ss = 0.0137f;

        std::vector<uint8_t> packed;
        pack_q_vnni(q.data(), K, K, N, N, packed);
        std::vector<int32_t> rs(static_cast<size_t>(M) * groups);
        int8_kquant_row_sums(A.data(), K, M, K, gs, rs.data());

        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), K, packed.data(), N * KQ_VNNI_GRP, C.data(), N, K, gs,
                D.data(), Min.data(), N, rs.data(), groups, &ss, 0, 0);

        std::vector<float> ref;
        reference_kquant(M, N, K, gs, A.data(), K, q.data(), K, D.data(),
                Min.data(), &ss, 0, 0, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
        EXPECT_EQ(nans, 0);
    }
}

TEST(Int8KquantUkernel128, RowSumsMatchAScalarSum) {
    const int M = 5, K = 96, gs = 32, groups = K / gs;
    std::mt19937 rng(7);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    std::vector<int32_t> rs(static_cast<size_t>(M) * groups);
    int8_kquant_row_sums(A.data(), K, M, K, gs, rs.data());
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < groups; ++g) {
            int32_t want = 0;
            for (int j = 0; j < gs; ++j) want += A[m * K + g * gs + j];
            EXPECT_EQ(rs[m * groups + g], want) << "m=" << m << " g=" << g;
        }
    }
}

// The single-row kernel must agree with the general one on the row they share, and
// the scalar tail with both at every ragged extent.
TEST(Int8KquantUkernel128, SingleRowAndTailAgreeWithTheTile) {
    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    const int8_kquant_ukernel_128_fn_t hot1
            = select_int8_kquant_ukernel_128_m1();
    if (hot == nullptr || hot1 == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int N = KQ_NR, K = 128, gs = 32, groups = K / gs;
    std::mt19937 rng(4242);
    std::vector<int8_t> A(static_cast<size_t>(KQ_MR) * K);
    std::vector<uint8_t> q(static_cast<size_t>(N) * K);
    std::vector<float> D(static_cast<size_t>(groups) * N);
    std::vector<float> Min(static_cast<size_t>(groups) * N);
    fill_s8(A, rng, false);
    fill_codes(q, rng, 15);
    fill_scales(D, rng);
    fill_scales(Min, rng);
    const float ss = 0.0137f;

    std::vector<uint8_t> packed;
    pack_q_vnni(q.data(), K, K, N, N, packed);
    std::vector<int32_t> rs(static_cast<size_t>(KQ_MR) * groups);
    int8_kquant_row_sums(A.data(), K, KQ_MR, K, gs, rs.data());

    std::vector<float> c_tile(static_cast<size_t>(KQ_MR) * N, 0.0f);
    hot(A.data(), K, packed.data(), N * KQ_VNNI_GRP, c_tile.data(), N, K, gs,
            D.data(), Min.data(), N, rs.data(), groups, &ss, 0, 0);

    std::vector<float> c_one(N, 0.0f);
    hot1(A.data(), K, packed.data(), N * KQ_VNNI_GRP, c_one.data(), N, K, gs,
            D.data(), Min.data(), N, rs.data(), groups, &ss, 0, 0);
    for (int n = 0; n < N; ++n)
        EXPECT_FLOAT_EQ(c_one[n], c_tile[n]) << "one-row kernel differs at " << n;

    for (int mr = 1; mr <= KQ_MR; ++mr) {
        for (int nr = 1; nr <= KQ_NR; ++nr) {
            SCOPED_TRACE(testing::Message() << "mr=" << mr << " nr=" << nr);
            std::vector<float> c_tail(static_cast<size_t>(KQ_MR) * N, 0.0f);
            int8_kquant_tail_128(A.data(), K, packed.data(), N * KQ_VNNI_GRP,
                    c_tail.data(), N, K, gs, mr, nr, D.data(), Min.data(), N,
                    rs.data(), groups, &ss, 0, 0);
            for (int m = 0; m < mr; ++m)
                for (int n = 0; n < nr; ++n)
                    EXPECT_NEAR(c_tail[m * N + n], c_tile[m * N + n],
                            1e-3 * std::fabs(c_tile[m * N + n]) + 1e-4);
        }
    }
}

// Per-token and per-group activation scales, the granularities the GGML path
// supplies; the k-quant flush applies them exactly where the symmetric one does.
TEST(Int8KquantUkernel128, PerTokenAndPerGroupActivationScales) {
    const int8_kquant_ukernel_128_fn_t hot = select_int8_kquant_ukernel_128();
    if (hot == nullptr) GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = KQ_MR, N = KQ_NR, K = 128, gs = 32, groups = K / gs;
    std::mt19937 rng(31337);
    for (int which = 0; which < 2; ++which) {
        const bool per_group = which == 1;
        SCOPED_TRACE(per_group ? "per-group {M,G}" : "per-token {M,1}");
        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        std::vector<uint8_t> q(static_cast<size_t>(N) * K);
        std::vector<float> D(static_cast<size_t>(groups) * N);
        std::vector<float> Min(static_cast<size_t>(groups) * N);
        std::vector<float> ss(static_cast<size_t>(M) * (per_group ? groups : 1));
        fill_s8(A, rng, false);
        fill_codes(q, rng, 15);
        fill_scales(D, rng);
        fill_scales(Min, rng);
        fill_scales(ss, rng);
        const int ss_row = per_group ? groups : 1;
        const int ss_grp = per_group ? 1 : 0;

        std::vector<uint8_t> packed;
        pack_q_vnni(q.data(), K, K, N, N, packed);
        std::vector<int32_t> rs(static_cast<size_t>(M) * groups);
        int8_kquant_row_sums(A.data(), K, M, K, gs, rs.data());

        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        hot(A.data(), K, packed.data(), N * KQ_VNNI_GRP, C.data(), N, K, gs,
                D.data(), Min.data(), N, rs.data(), groups, ss.data(), ss_row,
                ss_grp);

        std::vector<float> ref;
        reference_kquant(M, N, K, gs, A.data(), K, q.data(), K, D.data(),
                Min.data(), ss.data(), ss_row, ss_grp, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
        EXPECT_EQ(nans, 0);
    }
}

// ---- k-quant looper --------------------------------------------------------

namespace {

struct KqShape {
    int M, N, K, gs;
    bool transB;
    int nthreads;
    int max_code;
    int ss_kind; // 0 per-tensor, 1 per-token {M,1}, 2 per-group {M,G}
    const char *why;
};

} // namespace

class Int8KquantLooper128 : public ::testing::TestWithParam<KqShape> {};

TEST_P(Int8KquantLooper128, MatchesTheDequantiseThenMultiplyReference) {
    if (select_int8_kquant_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const KqShape s = GetParam();
    const int lda = s.K;
    const int ldq = s.transB ? s.K : s.N;
    const int ldc = s.N;
    const int groups = s.K / s.gs;

    std::mt19937 rng(20260815);
    std::vector<int8_t> A(static_cast<size_t>(s.M) * lda);
    std::vector<uint8_t> q(s.transB ? static_cast<size_t>(s.N) * ldq
                                    : static_cast<size_t>(s.K) * ldq);
    std::vector<float> D(static_cast<size_t>(groups) * s.N);
    std::vector<float> Min(static_cast<size_t>(groups) * s.N);
    fill_s8(A, rng, false);
    fill_codes(q, rng, s.max_code);
    fill_scales(D, rng);
    fill_scales(Min, rng);

    const int ss_row = s.ss_kind == 0 ? 0 : (s.ss_kind == 1 ? 1 : groups);
    const int ss_grp = s.ss_kind == 2 ? 1 : 0;
    std::vector<float> ss(s.ss_kind == 0
                    ? size_t {1}
                    : static_cast<size_t>(s.M) * (s.ss_kind == 1 ? 1 : groups));
    fill_scales(ss, rng);

    std::vector<float> C = poisoned(static_cast<size_t>(s.M) * ldc);
    ASSERT_TRUE(int8_kquant_execute_128(s.M, s.N, s.K, s.gs, A.data(), lda,
            q.data(), ldq, s.transB, C.data(), ldc, D.data(), Min.data(),
            ss.data(), ss_row, ss_grp, s.nthreads));

    // The reference indexes q as [n][k]; transpose when the caller gave K x N so
    // one reference serves both layouts.
    std::vector<uint8_t> q_nk(static_cast<size_t>(s.N) * s.K);
    for (int n = 0; n < s.N; ++n)
        for (int k = 0; k < s.K; ++k)
            q_nk[static_cast<size_t>(n) * s.K + k] = s.transB
                    ? q[static_cast<size_t>(n) * ldq + k]
                    : q[static_cast<size_t>(k) * ldq + n];

    std::vector<float> ref;
    reference_kquant(s.M, s.N, s.K, s.gs, A.data(), lda, q_nk.data(), s.K,
            D.data(), Min.data(), ss.data(), ss_row, ss_grp, ref);
    int nans = 0;
    const double err = worst_scaled_error(C, ldc, ref, s.N, s.M, s.N, &nans);
    EXPECT_EQ(nans, 0) << nans << " output elements were never written";
    EXPECT_LT(err, kTolerance);
}

// K=4096 and K=2048 exercise the K blocking, which splits the group loop; the min
// correction is applied per group inside it, so an off-by-one in a block's group
// offset shows up in those rows and nowhere else. Ragged M exercises the padded
// path, which must pad the row sums and the per-token scales alongside A.
INSTANTIATE_TEST_SUITE_P(Shapes, Int8KquantLooper128,
        ::testing::Values(
                KqShape {4, 8, 32, 32, false, 1, 15, 0, "one tile one group"},
                KqShape {4, 8, 256, 32, false, 1, 31, 0, "Q5_K codes"},
                KqShape {1, 512, 512, 32, true, 4, 15, 0, "decode GGML layout"},
                KqShape {7, 13, 96, 32, false, 1, 15, 0, "ragged M and N"},
                KqShape {5, 70, 128, 16, false, 2, 15, 0, "crosses panel tail"},
                KqShape {128, 256, 512, 32, true, 4, 15, 0, "prompt GEMM"},
                KqShape {3, 64, 32, 32, true, 1, 15, 0, "M below MR"},
                KqShape {4, 8, 4096, 32, false, 1, 15, 0, "long K blocked"},
                KqShape {64, 64, 64, 64, false, 2, 15, 0, "group spans K"},
                KqShape {2, 9, 48, 16, true, 3, 31, 0, "odd everything"},
                KqShape {4, 65, 32, 32, false, 4, 15, 0, "N past a panel"},
                KqShape {33, 72, 2048, 32, true, 4, 15, 1, "per token ragged M"},
                KqShape {12, 128, 512, 32, true, 4, 15, 2, "per group scales"},
                KqShape {1, 128, 2048, 32, true, 2, 31, 1, "decode per token"}),
        [](const ::testing::TestParamInfo<KqShape> &i) {
            std::string n(i.param.why);
            for (char &c : n)
                if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
            return n;
        });

TEST(Int8KquantLooper128Extra, DeclinesShapesItCannotExpress) {
    std::vector<int8_t> A(256, 1);
    std::vector<uint8_t> q(256, 1);
    std::vector<float> D(256, 1.0f), Min(256, 0.0f), C(64, -7.0f);
    const float ss = 1.0f;
    struct { int K, gs; const char *why; } bad[] = {
            {30, 32, "K is not a whole number of groups"},
            {32, 6, "group size splits a VNNI quad"},
            {32, 0, "group size zero"},
    };
    for (const auto &b : bad) {
        SCOPED_TRACE(b.why);
        EXPECT_FALSE(int8_kquant_execute_128(1, 8, b.K, b.gs, A.data(),
                std::max(b.K, 1), q.data(), 8, false, C.data(), 8, D.data(),
                Min.data(), &ss, 0, 0, 1));
        for (float v : C) EXPECT_FLOAT_EQ(v, -7.0f) << "C was written";
    }
}

TEST(Int8KquantLooper128Extra, IsInvariantInThreadCount) {
    if (select_int8_kquant_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = 33, N = 200, K = 256, gs = 32, groups = K / gs;
    std::mt19937 rng(5);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<uint8_t> q(static_cast<size_t>(N) * K);
    std::vector<float> D(static_cast<size_t>(groups) * N);
    std::vector<float> Min(static_cast<size_t>(groups) * N);
    fill_s8(A, rng, false);
    fill_codes(q, rng, 31);
    fill_scales(D, rng);
    fill_scales(Min, rng);
    const float ss = 0.01f;

    std::vector<float> single = poisoned(static_cast<size_t>(M) * N);
    ASSERT_TRUE(int8_kquant_execute_128(M, N, K, gs, A.data(), K, q.data(), K,
            true, single.data(), N, D.data(), Min.data(), &ss, 0, 0, 1));
    for (int nt : {2, 3, 4, 8}) {
        SCOPED_TRACE(testing::Message() << "nthreads=" << nt);
        std::vector<float> many = poisoned(static_cast<size_t>(M) * N);
        ASSERT_TRUE(int8_kquant_execute_128(M, N, K, gs, A.data(), K, q.data(),
                K, true, many.data(), N, D.data(), Min.data(), &ss, 0, 0, nt));
        for (size_t i = 0; i < single.size(); ++i)
            ASSERT_FLOAT_EQ(single[i], many[i]) << "at " << i;
    }
}

// The looper must not care whether D and M come from one allocation or two: the
// two candidate ways of carrying the min through matmul_params differ only in
// that, and if the answers ever differed the benchmark comparing them would be
// measuring a bug.
TEST(Int8KquantLooper128Extra, OneBufferAndTwoBuffersAgreeExactly) {
    if (select_int8_kquant_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    const int M = 16, N = 128, K = 512, gs = 32, groups = K / gs;
    std::mt19937 rng(808);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<uint8_t> q(static_cast<size_t>(N) * K);
    fill_s8(A, rng, false);
    fill_codes(q, rng, 15);

    // One allocation: D in the first groups*N floats, M in the next -- the
    // {2G, N} wei_scale layout.
    std::vector<float> joint(static_cast<size_t>(2) * groups * N);
    fill_scales(joint, rng);
    // Two allocations holding the same values -- the separate-wei_zp layout.
    std::vector<float> D(joint.begin(), joint.begin() + groups * N);
    std::vector<float> Min(joint.begin() + groups * N, joint.end());
    const float ss = 0.0137f;

    std::vector<float> c_joint = poisoned(static_cast<size_t>(M) * N);
    std::vector<float> c_split = poisoned(static_cast<size_t>(M) * N);
    ASSERT_TRUE(int8_kquant_execute_128(M, N, K, gs, A.data(), K, q.data(), K,
            true, c_joint.data(), N, joint.data(), joint.data() + groups * N,
            &ss, 0, 0, 4));
    ASSERT_TRUE(int8_kquant_execute_128(M, N, K, gs, A.data(), K, q.data(), K,
            true, c_split.data(), N, D.data(), Min.data(), &ss, 0, 0, 4));
    for (size_t i = 0; i < c_joint.size(); ++i)
        ASSERT_FLOAT_EQ(c_joint[i], c_split[i]) << "at " << i;
}

// ---- Q4_K / Q5_K unpack ----------------------------------------------------
//
// Builds real GGML super-blocks, runs them through the library's unpack via
// matmul_direct's entry point, and checks the decoded codes and scales against
// GGML's own dequantisation formula. The packers here are the inverse of
// get_scale_min_k4 and of the nibble/high-bit walk, written from the format
// rather than from the library's decoder, so a shared misreading cannot cancel
// out.

namespace {

constexpr int kKsuper = 256;
constexpr int kKsub = 32;

uint16_t f32_to_fp16_pow2(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const uint32_t sign = (bits >> 31) & 1;
    const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
    const uint32_t mant = bits & 0x7FFFFF;
    return static_cast<uint16_t>(
            (sign << 15) | (static_cast<uint32_t>(exp + 15) << 10) | (mant >> 13));
}

// Inverse of get_scale_min_k4: pack eight six-bit scales and eight six-bit mins
// into twelve bytes.
void pack_scales_mins_k4(const uint8_t *sc, const uint8_t *mn, uint8_t *out) {
    std::memset(out, 0, 12);
    for (int j = 0; j < 4; ++j) {
        out[j] = static_cast<uint8_t>(sc[j] & 63);
        out[j + 4] = static_cast<uint8_t>(mn[j] & 63);
    }
    for (int j = 4; j < 8; ++j) {
        // low four bits of sc[j] and mn[j] go in the last four bytes...
        out[j + 4] = static_cast<uint8_t>((sc[j] & 0xF) | ((mn[j] & 0xF) << 4));
        // ...and their top two bits ride in the top two bits of earlier bytes.
        out[j - 4] = static_cast<uint8_t>(out[j - 4] | ((sc[j] >> 4) << 6));
        out[j - 0] = static_cast<uint8_t>(out[j - 0] | ((mn[j] >> 4) << 6));
    }
}

struct KquantSource {
    std::vector<uint8_t> blocks;   // packed GGML blocks, N rows
    std::vector<uint8_t> codes;    // expected codes, N x K
    std::vector<float> D, Min;     // expected scales, group-major {G, N}
};

// Build N x K of Q4_K (type 12) or Q5_K (13). Scales are exact powers of two and
// the six-bit factors are small integers, so d * sc survives fp16 exactly and the
// comparison measures decoding rather than rounding.
KquantSource build_kquant(int type, int N, int K, std::mt19937 &rng) {
    const int nsb = K / kKsuper;
    const int groups = K / kKsub;
    const size_t block_bytes = (type == 12) ? 144u : 176u;
    const int max_code = (type == 12) ? 15 : 31;

    KquantSource src;
    src.blocks.assign(static_cast<size_t>(N) * nsb * block_bytes, 0);
    src.codes.assign(static_cast<size_t>(N) * K, 0);
    src.D.assign(static_cast<size_t>(groups) * N, 0.0f);
    src.Min.assign(static_cast<size_t>(groups) * N, 0.0f);

    std::uniform_int_distribution<int> dcode(0, max_code);
    std::uniform_int_distribution<int> dsix(1, 40);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};

    for (int n = 0; n < N; ++n) {
        for (int sb = 0; sb < nsb; ++sb) {
            const float d = exact[(n + sb) % 4];
            const float dmin = exact[(n + sb + 1) % 4];
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; ++j) {
                sc[j] = static_cast<uint8_t>(dsix(rng));
                mn[j] = static_cast<uint8_t>(dsix(rng));
                const int g = sb * 8 + j;
                src.D[static_cast<size_t>(g) * N + n] = d * sc[j];
                src.Min[static_cast<size_t>(g) * N + n] = dmin * mn[j];
            }

            uint8_t codes[kKsuper];
            for (int e = 0; e < kKsuper; ++e)
                codes[e] = static_cast<uint8_t>(dcode(rng));
            std::memcpy(&src.codes[static_cast<size_t>(n) * K + sb * kKsuper],
                    codes, kKsuper);

            uint8_t *blk = &src.blocks[(static_cast<size_t>(n) * nsb + sb)
                    * block_bytes];
            const uint16_t d16 = f32_to_fp16_pow2(d);
            const uint16_t dm16 = f32_to_fp16_pow2(dmin);
            std::memcpy(blk, &d16, 2);
            std::memcpy(blk + 2, &dm16, 2);
            pack_scales_mins_k4(sc, mn, blk + 4);

            uint8_t *ql = blk + (type == 12 ? 16 : 48);
            uint8_t *qh = (type == 13) ? blk + 16 : nullptr;
            if (qh) std::memset(qh, 0, 32);
            uint8_t u1 = 1, u2 = 2;
            for (int c = 0; c < kKsuper / 64; ++c) {
                for (int l = 0; l < 32; ++l) {
                    const uint8_t lo = codes[c * 64 + l];
                    const uint8_t hi = codes[c * 64 + 32 + l];
                    ql[c * 32 + l] = static_cast<uint8_t>(
                            (lo & 0x0F) | ((hi & 0x0F) << 4));
                    if (qh) {
                        if (lo >= 16) qh[l] = static_cast<uint8_t>(qh[l] | u1);
                        if (hi >= 16) qh[l] = static_cast<uint8_t>(qh[l] | u2);
                    }
                }
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
    }
    return src;
}

} // namespace

class GgmlKquantUnpack : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        clear_all_weight_caches();
        clear_ggml_weight_unpack_cache();
    }
    void TearDown() override {
        clear_all_weight_caches();
        clear_ggml_weight_unpack_cache();
    }
};

// The unpack is reached through matmul_direct, so this also proves a Q4_K/Q5_K
// call gets past ggml_is_sym_quant and the type validation. The matmul itself is
// expected to decline for now -- no dispatch exists yet -- which is why the
// status is not asserted; what is asserted is that the weight was decoded.
TEST_P(GgmlKquantUnpack, DecodesCodesAndScalesLikeGgml) {
    const int type = GetParam();
    const int N = 8, K = 512;
    const int groups = K / kKsub;
    std::mt19937 rng(type == 12 ? 1212 : 1313);
    KquantSource src = build_kquant(type, N, K, rng);

    // Drive the library's unpack directly: it is the entry point matmul_direct
    // uses, and calling it here keeps the test on the decode rather than on
    // whatever dispatch does afterwards.
    const void *weight = src.blocks.data();
    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = (type == 12) ? data_type_t::s4 : data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = type;
    std::vector<float> src_scales(static_cast<size_t>(4) * groups, 0.01f);
    p.quant_params.src_scale.buff = src_scales.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {4, groups};

    ASSERT_EQ(unpack_ggml_weights_and_cache(weight, N, K, K, 't', p),
            status_t::success);

    // Codes come back unsigned and row-major N x K.
    EXPECT_EQ(p.dtypes.wei, data_type_t::u8)
            << "codes must be advertised unsigned, which is also what keeps the "
               "symmetric path away";
    EXPECT_EQ(p.mem_format_b, 'n');
    EXPECT_EQ(p.packing.pack_format_b, 0);
    ASSERT_EQ(p.quant_params.wei_scale.dims.size(), 2u);
    EXPECT_EQ(p.quant_params.wei_scale.dims[0], 2 * groups);
    EXPECT_EQ(p.quant_params.wei_scale.dims[1], N);
    EXPECT_EQ(p.quant_params.wei_scale.dt, data_type_t::f32);

    const uint8_t *codes = static_cast<const uint8_t *>(weight);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            ASSERT_EQ(codes[static_cast<size_t>(n) * K + k],
                    src.codes[static_cast<size_t>(n) * K + k])
                    << "code at (" << n << "," << k << ")";

    // D at the start of the scale region, M one cache line past its end.
    const float *D = static_cast<const float *>(p.quant_params.wei_scale.buff);
    const float *Min = D + static_cast<size_t>(groups) * N + 16;
    for (int g = 0; g < groups; ++g) {
        for (int n = 0; n < N; ++n) {
            const size_t i = static_cast<size_t>(g) * N + n;
            EXPECT_FLOAT_EQ(D[i], src.D[i]) << "D at g=" << g << " n=" << n;
            EXPECT_FLOAT_EQ(Min[i], src.Min[i]) << "M at g=" << g << " n=" << n;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Types, GgmlKquantUnpack, ::testing::Values(12, 13),
        [](const ::testing::TestParamInfo<int> &i) {
            return i.param == 12 ? std::string("Q4_K") : std::string("Q5_K");
        });

// A decoded k-quant must be refused by the symmetric path rather than computed
// with its min dropped: {2G, N} reads as a valid symmetric per-group scale of
// group size K/(2G), so the u8 dtype is the only thing standing between a Q4_K
// weight and a confidently wrong answer.
TEST(GgmlKquantUnpackExtra, SymmetricPathRefusesADecodedKquant) {
    const int N = 8, K = 512, groups = K / kKsub;
    std::mt19937 rng(77);
    KquantSource src = build_kquant(12, N, K, rng);
    const void *weight = src.blocks.data();
    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s4;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = 12;
    std::vector<float> ss(static_cast<size_t>(4) * groups, 0.01f);
    p.quant_params.src_scale.buff = ss.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {4, groups};
    ASSERT_EQ(unpack_ggml_weights_and_cache(weight, N, K, K, 't', p),
            status_t::success);

    EXPECT_FALSE(native::is_int8_symq_candidate(p, K, N))
            << "the symmetric kernel accepted a k-quant; it would compute it "
               "with the min term silently dropped";
}

// ---- Q4_K / Q5_K end to end ------------------------------------------------
//
// A GGML k-quant weight through matmul_direct: unpack, dispatch, kernel. This is
// the case every other k-quant test only approximates, and the reason the whole
// asymmetric path exists -- most GGUFs in circulation are Q4_K.

class Int8KquantDispatch : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        if (select_int8_kquant_ukernel_128() == nullptr)
            GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";
        setenv("ZENDNNL_NATIVE_SYMQ_128", "1", 1);
        clear_all_weight_caches();
        clear_ggml_weight_unpack_cache();
    }
    void TearDown() override {
        clear_all_weight_caches();
        clear_ggml_weight_unpack_cache();
    }
};

TEST_P(Int8KquantDispatch, PackedWeightComputesEndToEnd) {
    const int type = GetParam();
    const int M = 6, N = 64, K = 512;
    const int groups = K / kKsub;

    std::mt19937 rng(type == 12 ? 4120 : 4130);
    KquantSource src = build_kquant(type, N, K, rng);

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    // Per-group activation scales: more than one element, which is what
    // ggml_is_sym_quant requires of a GGML call.
    std::vector<float> ss(static_cast<size_t>(M) * groups);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < ss.size(); ++i) ss[i] = exact[(i + 1) % 4];

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = (type == 12) ? data_type_t::s4 : data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = type;
    p.quant_params.src_scale.buff = ss.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f, A.data(),
                      K, src.blocks.data(), K, nullptr, 0.0f, C.data(), N,
                      /*is_weights_const=*/true, batch, p),
            status_t::success);

    // Reference from the codes and scales the packer started with, dequantising
    // the GGML way and then multiplying.
    std::vector<float> ref;
    reference_kquant(M, N, K, kKsub, A.data(), K, src.codes.data(), K,
            src.D.data(), src.Min.data(), ss.data(), groups, 1, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0) << "dst was never written -- the k-quant call did not "
                          "reach a kernel that computes";
}

INSTANTIATE_TEST_SUITE_P(Types, Int8KquantDispatch, ::testing::Values(12, 13),
        [](const ::testing::TestParamInfo<int> &i) {
            return i.param == 12 ? std::string("Q4_K") : std::string("Q5_K");
        });

// The decode shape, which is where a k-quant model spends most of its time, and
// the one that routes to the one-row microkernel.
TEST_P(Int8KquantDispatch, DecodeShapeReachesTheKernel) {
    const int type = GetParam();
    const int M = 1, N = 128, K = 512;
    const int groups = K / kKsub;
    std::mt19937 rng(type);
    KquantSource src = build_kquant(type, N, K, rng);

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    std::vector<float> ss(static_cast<size_t>(M) * groups, 0.0625f);
    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = (type == 12) ? data_type_t::s4 : data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = type;
    p.quant_params.src_scale.buff = ss.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 4;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, true, M, N, K, 1.0f, A.data(), K,
                      src.blocks.data(), K, nullptr, 0.0f, C.data(), N, true,
                      batch, p),
            status_t::success);

    std::vector<float> ref;
    reference_kquant(M, N, K, kKsub, A.data(), K, src.codes.data(), K,
            src.D.data(), src.Min.data(), ss.data(), groups, 1, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0);
}

// ------------------------------------------------------------------ dispatch

// Through the public matmul API rather than the kernel entry point, so these
// cover the part the tests above cannot: that a per-group INT8 call actually
// reaches this kernel instead of AOCL-DLP declining it, and that the adapter's
// gates translate the call correctly. ZENDNNL_MATMUL_ALGO is not used -- the
// dispatcher is asked for the native algo directly, the way a caller would.
//
// On a host WITH AVX-512 VNNI the existing INT8 path is preferred and these
// would exercise that instead, so they force the 128-bit path with
// ZENDNNL_NATIVE_SYMQ_128; that knob exists precisely because family 15h cannot
// run the reference, so the two paths can only be compared elsewhere.
// The packed-weight cache is keyed on the caller's weight pointer under the
// is_weights_const promise, which a test violates as soon as it frees one weight
// and allocates another: the new buffer can land at the same address with the
// same shape, and the cache then serves the previous weight's panels. That is a
// property of every pointer-keyed weight cache in this tree, not of this path,
// and clear_all_weight_caches() is the sanctioned answer -- its own comment names
// "between test cases or model swap" as the use. Tests that need two weights at
// once keep both alive instead.
class Int8SymqDispatch : public ::testing::Test {
protected:
    void SetUp() override {
        if (select_int8_symq_ukernel_128() == nullptr)
            GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";
        setenv("ZENDNNL_NATIVE_SYMQ_128", "1", 1);
        clear_all_weight_caches();
        // The GGML unpack keeps its own cache, separate from the native weight
        // caches and keyed the same way on the caller's pointer. Leaving it
        // populated let a GGML test in test_matmul.cpp and a GGML test here alias
        // each other's weights: this suite passed alone and failed when the two
        // ran together, which is the most annoying shape a cache bug can take.
        clear_ggml_weight_unpack_cache();
    }
    void TearDown() override {
        clear_all_weight_caches();
        clear_ggml_weight_unpack_cache();
    }
};

// Build the params a GGML per-group weight produces: s8 x s8 -> f32, a
// per-tensor source scale, and a {groups, N} weight scale.
struct SymqCall {
    int M, N, K, gs;
    bool transB;
    std::vector<int8_t> A, B;
    std::vector<float> ws;
    std::vector<float> C;
    float src_scale = 0.0137f;

    SymqCall(int m, int n, int k, int g, bool tb, std::mt19937 &rng)
        : M(m), N(n), K(k), gs(g), transB(tb) {
        A.resize(static_cast<size_t>(M) * K);
        B.resize(static_cast<size_t>(N) * K);
        ws.resize(static_cast<size_t>(K / gs) * N);
        C.assign(static_cast<size_t>(M) * N,
                std::numeric_limits<float>::quiet_NaN());
        fill_s8(A, rng, false);
        fill_s8(B, rng, false);
        fill_scales(ws, rng);
    }

    // Per-token {M,1} and per-group {M,G} activation scales, which is what the
    // GGML path actually supplies -- ggml_is_sym_quant() requires a source scale
    // with more than one element, so a per-tensor-only kernel is unreachable
    // from it. Filled always; make_params() selects which one it advertises.
    std::vector<float> src_scales_token, src_scales_group;

    void fill_vector_scales(std::mt19937 &rng) {
        src_scales_token.resize(static_cast<size_t>(M));
        src_scales_group.resize(static_cast<size_t>(M) * (K / gs));
        fill_scales(src_scales_token, rng);
        fill_scales(src_scales_group, rng);
    }

    matmul_params p_token() {
        matmul_params p = make_params();
        p.quant_params.src_scale.buff = src_scales_token.data();
        p.quant_params.src_scale.dims = {M, 1};
        return p;
    }

    matmul_params p_group() {
        matmul_params p = make_params();
        p.quant_params.src_scale.buff = src_scales_group.data();
        p.quant_params.src_scale.dims = {M, K / gs};
        return p;
    }

    matmul_params make_params() {
        matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.quant_params.src_scale.buff = &src_scale;
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {1};
        p.quant_params.wei_scale.buff = ws.data();
        p.quant_params.wei_scale.dt = data_type_t::f32;
        p.quant_params.wei_scale.dims
                = {static_cast<int64_t>(K / gs), static_cast<int64_t>(N)};
        return p;
    }

    status_t run(matmul_algo_t algo, int nthreads = 4) {
        matmul_params p = make_params();
        p.lowoha_algo = algo;
        p.num_threads = nthreads;
        matmul_batch_params_t batch;
        return matmul_direct('r', false, transB, M, N, K, 1.0f, A.data(), K,
                B.data(), transB ? K : N, nullptr, 0.0f, C.data(), N, true,
                batch, p);
    }
};

TEST_F(Int8SymqDispatch, GgmlShapedCallReachesTheKernelAndComputes) {
    std::mt19937 rng(31337);
    const matmul_algo_t algos[]
            = {matmul_algo_t::native_gemm, matmul_algo_t::native_brgemm};
    // Both calls constructed up front and kept alive: two distinct weights must
    // have two distinct addresses for the weight cache to tell them apart.
    // The transposed layout is the one the GGML unpack produces.
    std::vector<SymqCall> calls;
    for (size_t i = 0; i < 2; ++i)
        calls.emplace_back(128, 256, 512, 32, /*transB=*/true, rng);

    for (size_t i = 0; i < 2; ++i) {
        SCOPED_TRACE(testing::Message()
                << "algo=" << static_cast<int>(algos[i]));
        SymqCall &c = calls[i];
        ASSERT_EQ(c.run(algos[i]), status_t::success);

        std::vector<float> ref;
        reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K,
                true, c.ws.data(), &c.src_scale, ref);
        int nans = 0;
        // Poisoned C plus an exact-to-fp32 comparison: this fails both when the
        // call is declined and returns without computing (NaNs survive) and when
        // it computes the wrong thing.
        EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
                kTolerance);
        EXPECT_EQ(nans, 0) << "dst was never written -- the call did not reach "
                              "the kernel";
    }
}

// The second and later calls on a const weight read a packed copy from the
// cache instead of packing again. Those bytes are supposed to be identical to
// what the per-call packer produces, so every call must agree with the reference
// and with the first call exactly -- not merely closely.
TEST_F(Int8SymqDispatch, RepeatedCallsOnAConstWeightAgreeExactly) {
    std::mt19937 rng(808);
    SymqCall c(64, 200, 256, 32, true, rng);
    std::vector<float> ref;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);

    std::vector<float> first;
    for (int call = 0; call < 4; ++call) {
        SCOPED_TRACE(testing::Message() << "call " << call);
        std::fill(c.C.begin(), c.C.end(),
                std::numeric_limits<float>::quiet_NaN());
        ASSERT_EQ(c.run(matmul_algo_t::native_gemm), status_t::success);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
                kTolerance);
        EXPECT_EQ(nans, 0);
        if (call == 0)
            first = c.C;
        else
            for (size_t i = 0; i < first.size(); ++i)
                ASSERT_FLOAT_EQ(first[i], c.C[i])
                        << "cached packed weight disagrees with the first call "
                           "at "
                        << i;
    }
}

TEST_F(Int8SymqDispatch, DecodeShapeReachesTheKernel) {
    std::mt19937 rng(4);
    SymqCall c(1, 512, 512, 32, true, rng);
    ASSERT_EQ(c.run(matmul_algo_t::native_gemm, 4), status_t::success);
    std::vector<float> ref;
    reference_gemm(1, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, 1, c.N, &nans), kTolerance);
    EXPECT_EQ(nans, 0);
}

// -128 is outside the kernel's contract and fails silently inside it, so the
// adapter must refuse the call rather than compute a plausible wrong answer.
// The result is a fallback, not an error: on a host without VNNI that fallback
// declines too, so what this asserts is that dst is not filled with garbage.
TEST_F(Int8SymqDispatch, RefusesOperandsOutsideTheByteContract) {
    std::mt19937 rng(5);
    // Kept alive together, so the second weight cannot inherit the first's
    // address and with it the first's in-contract verdict.
    std::vector<SymqCall> calls;
    for (int i = 0; i < 2; ++i) calls.emplace_back(8, 64, 128, 32, true, rng);

    for (int which = 0; which < 2; ++which) {
        SCOPED_TRACE(which == 0 ? "-128 in the source" : "-128 in the weight");
        SymqCall &c = calls[which];
        if (which == 0)
            c.A[c.A.size() / 2] = static_cast<int8_t>(-128);
        else
            c.B[c.B.size() / 3] = static_cast<int8_t>(-128);

        std::vector<float> ref;
        reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K,
                true, c.ws.data(), &c.src_scale, ref);

        c.run(matmul_algo_t::native_gemm);
        // Either the call was declined outright (dst still poisoned) or some
        // other backend computed it correctly. What must not happen is a
        // confidently wrong answer from this kernel.
        int nans = 0;
        const double err
                = worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans);
        const bool declined = nans == c.M * c.N;
        EXPECT_TRUE(declined || err < kTolerance)
                << "computed a wrong answer for an out-of-contract operand: "
                << "err=" << err << " nans=" << nans;
    }
}

// The granularity the GGML path actually supplies: ggml_is_sym_quant() requires
// a source scale with more than one element, so a per-tensor-only kernel is
// unreachable from it however well the per-tensor case works.
//
// Driven at the looper rather than through matmul_direct, and not for
// convenience: validate_matmul_inputs() rejects a per-token source scale paired
// with a per-group weight scale whenever the caller supplies the weight scale
// itself (pack_format_b != 1), which is what a synthetic test does. A real GGML
// call sets pack_format_b == 1 and the unpack populates the weight scale
// afterwards, so the check is skipped there. Reproducing that from a gtest would
// mean building a GGML-packed buffer; until there is one, the kernel contract is
// what these cover.
TEST(Int8SymqLooper128Scales, PerTokenAndPerGroupActivationScales) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    std::mt19937 rng(9001);
    // Ragged M included on purpose: a row-indexed scale plus an M that is not a
    // multiple of MR is the combination that can read past the scale buffer,
    // since the vector kernel reads MR rows' scales and only M % MR exist.
    for (int M : {1, 2, 3, 5, 7, 12, 13, 64}) {
        for (int which = 0; which < 2; ++which) {
            const bool per_group = which == 1;
            SCOPED_TRACE(testing::Message()
                    << "M=" << M
                    << (per_group ? " per-group {M,G}" : " per-token {M,1}"));
            const int N = 72, K = 128, gs = 32;
            const int groups = K / gs;
            std::vector<int8_t> A(static_cast<size_t>(M) * K);
            std::vector<int8_t> B(static_cast<size_t>(N) * K);
            std::vector<float> ws(static_cast<size_t>(groups) * N);
            std::vector<float> ss(static_cast<size_t>(M)
                    * (per_group ? groups : 1));
            fill_s8(A, rng, false);
            fill_s8(B, rng, false);
            fill_scales(ws, rng);
            fill_scales(ss, rng);
            const int ss_row = per_group ? groups : 1;
            const int ss_grp = per_group ? 1 : 0;

            std::vector<float> C = poisoned(static_cast<size_t>(M) * N);
            ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K,
                    B.data(), K, /*transB=*/true, C.data(), N, ws.data(),
                    ss.data(), ss_row, ss_grp, /*nthreads=*/2));

            std::vector<float> ref;
            reference_gemm(M, N, K, gs, A.data(), K, B.data(), K, true,
                    ws.data(), ss.data(), ref, ss_row, ss_grp);
            int nans = 0;
            EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
            EXPECT_EQ(nans, 0);
        }
    }
}

// A per-tensor scale must keep behaving exactly as it did before the strides
// existed: same answer whether it arrives as {0,0} strides or as a per-token
// vector whose entries happen to be equal.
TEST(Int8SymqLooper128Scales, PerTensorAgreesWithAnEquivalentPerTokenVector) {
    if (select_int8_symq_ukernel_128() == nullptr)
        GTEST_SKIP() << "no 128-bit INT8 microkernel on this host";

    std::mt19937 rng(77);
    const int M = 9, N = 64, K = 128, gs = 32;
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    std::vector<int8_t> B(static_cast<size_t>(N) * K);
    std::vector<float> ws(static_cast<size_t>(K / gs) * N);
    fill_s8(A, rng, false);
    fill_s8(B, rng, false);
    fill_scales(ws, rng);
    const float one_scale = 0.0137f;
    std::vector<float> as_vector(static_cast<size_t>(M), one_scale);

    std::vector<float> c_scalar = poisoned(static_cast<size_t>(M) * N);
    std::vector<float> c_vector = poisoned(static_cast<size_t>(M) * N);
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), K,
            true, c_scalar.data(), N, ws.data(), &one_scale, 0, 0, 2));
    ASSERT_TRUE(int8_symq_execute_128(M, N, K, gs, A.data(), K, B.data(), K,
            true, c_vector.data(), N, ws.data(), as_vector.data(), 1, 0, 2));
    for (size_t i = 0; i < c_scalar.size(); ++i)
        ASSERT_FLOAT_EQ(c_scalar[i], c_vector[i]) << "at " << i;
}

// The real thing: a GGML Q8_0 packed weight through matmul_direct, which is what
// llama.cpp hands over. This is the case every other test in this file only
// approximates, and it needs the library's own unpack to run first --
// unpack_ggml_weights_and_cache() decodes the blocks, and on a host without
// AVX-512 VNNI now asks for the raw s8 form this kernel can read rather than an
// AOCL reorder it cannot.
//
// Scales are exact powers of two on purpose. Q8_0 stores its scale as fp16 and
// the unpack rewrites it as bf16, so an arbitrary f32 scale is rounded twice
// before the kernel sees it and the comparison would be measuring that instead
// of the kernel. Powers of two survive both formats exactly.
TEST_F(Int8SymqDispatch, GgmlQ8_0PackedWeightComputesEndToEnd) {
    constexpr int kBlk = 32; // Q8_0 group size
    constexpr size_t kBlkBytes = 34; // fp16 scale + 32 int8
    const int M = 6, N = 64, K = 128;
    const int groups = K / kBlk;

    std::mt19937 rng(20260814);
    // Weight rows are N: the GGML layout is N x K, i.e. transB.
    std::vector<int8_t> wt(static_cast<size_t>(N) * K);
    fill_s8(wt, rng, false);

    // repack_weights_q8_0 wants scales column-major [groups x N], which is the
    // group-major {G, N} layout the kernel wants too.
    std::vector<float> wei_scales(static_cast<size_t>(groups) * N);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < wei_scales.size(); ++i) wei_scales[i] = exact[i % 4];

    std::vector<uint8_t> packed(
            static_cast<size_t>(N) * groups * kBlkBytes);
    repack_weights_q8_0(wt.data(), wei_scales.data(), static_cast<int64_t>(N),
            static_cast<int64_t>(K), packed.data());

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    // Per-group activation scales: more than one element, which is what
    // ggml_is_sym_quant() requires.
    std::vector<float> src_scales(static_cast<size_t>(M) * groups);
    for (size_t i = 0; i < src_scales.size(); ++i)
        src_scales[i] = exact[(i + 1) % 4];

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1; // GGML packed
    p.packing.ggml_type_b = 8; // Q8_0
    p.quant_params.src_scale.buff = src_scales.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;

    const status_t st = matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
            A.data(), K, packed.data(), K, nullptr, 0.0f, C.data(), N,
            /*is_weights_const=*/true, batch, p);
    ASSERT_EQ(st, status_t::success);

    std::vector<float> ref;
    reference_gemm(M, N, K, kBlk, A.data(), K, wt.data(), K, /*transB=*/true,
            wei_scales.data(), src_scales.data(), ref, groups, 1);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0) << "dst was never written -- the GGML call did not reach "
                          "a kernel that computes";
}

// Q4_0 and Q6_K take the same route as Q8_0 above -- unpack to raw s8 with
// per-group scales, then this kernel -- and are the other two types the unpack
// decodes. Q4_0 is what most legacy GGUFs use; Q6_K is the one k-quant that fits,
// being symmetric, and it differs in carrying one scale per sixteen weights
// instead of per thirty-two.
//
// Pack Q6_K here rather than in gtest_utils because nothing else needs it yet.
// The six-bit code is split across a low-nibble array and a high-two-bit array,
// and this is the exact inverse of the assembly in ggml_weight_unpack.cpp: get it
// wrong and the weights are plausible but wrong numbers, which is why the
// reference below is built from the values fed in rather than from the blocks.
namespace {

constexpr int kQ6kSuper = 256;
constexpr int kQ6kGroup = 16;

struct BlockQ6K {
    uint8_t ql[kQ6kSuper / 2];
    uint8_t qh[kQ6kSuper / 4];
    int8_t scales[kQ6kSuper / 16];
    uint16_t d;
};

uint16_t f32_to_fp16_exact(float v) {
    // Only used for exact powers of two, so no rounding logic is needed.
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const uint32_t sign = (bits >> 31) & 1;
    const int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
    const uint32_t mant = bits & 0x7FFFFF;
    return static_cast<uint16_t>((sign << 15)
            | (static_cast<uint32_t>(exp + 15) << 10) | (mant >> 13));
}

// codes are the biased six-bit values in [0, 63], row-major [N x K].
void pack_q6_k(const int8_t *values, const int8_t *sub_scales, float d, int N,
        int K, std::vector<uint8_t> &out) {
    const int nsb = K / kQ6kSuper;
    out.assign(static_cast<size_t>(N) * nsb * sizeof(BlockQ6K), 0);
    auto *blocks = reinterpret_cast<BlockQ6K *>(out.data());
    for (int row = 0; row < N; ++row) {
        for (int sb = 0; sb < nsb; ++sb) {
            BlockQ6K &b = blocks[row * nsb + sb];
            b.d = f32_to_fp16_exact(d);
            for (int j = 0; j < kQ6kSuper / 16; ++j)
                b.scales[j] = sub_scales[j];
            const int8_t *v
                    = values + static_cast<size_t>(row) * K + sb * kQ6kSuper;
            auto code = [&](int e) {
                return static_cast<int>(v[e]) + 32; // [-32,31] -> [0,63]
            };
            for (int n = 0; n < kQ6kSuper; n += 128) {
                uint8_t *ql = b.ql + (n / 2);
                uint8_t *qh = b.qh + (n / 4);
                for (int l = 0; l < 32; ++l) {
                    const int c0 = code(n + l), c32 = code(n + l + 32);
                    const int c64 = code(n + l + 64), c96 = code(n + l + 96);
                    ql[l] = static_cast<uint8_t>((c0 & 0x0F) | ((c64 & 0x0F) << 4));
                    ql[l + 32] = static_cast<uint8_t>(
                            (c32 & 0x0F) | ((c96 & 0x0F) << 4));
                    qh[l] = static_cast<uint8_t>((c0 >> 4) | ((c32 >> 4) << 2)
                            | ((c64 >> 4) << 4) | ((c96 >> 4) << 6));
                }
            }
        }
    }
}

} // namespace

TEST_F(Int8SymqDispatch, GgmlQ4_0PackedWeightComputesEndToEnd) {
    constexpr int kBlk = 32;
    constexpr size_t kBlkBytes = 18; // fp16 scale + 16 nibble bytes
    const int M = 5, N = 64, K = 128;
    const int groups = K / kBlk;

    std::mt19937 rng(606);
    // Q4_0 nibbles must round-trip losslessly, so values live in [-8, 7].
    std::vector<int8_t> wt(static_cast<size_t>(N) * K);
    std::uniform_int_distribution<int> d4(-8, 7);
    for (auto &x : wt) x = static_cast<int8_t>(d4(rng));

    std::vector<float> wei_scales(static_cast<size_t>(groups) * N);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < wei_scales.size(); ++i) wei_scales[i] = exact[i % 4];

    std::vector<uint8_t> packed(static_cast<size_t>(N) * groups * kBlkBytes);
    repack_weights_q4_0(wt.data(), wei_scales.data(), static_cast<int64_t>(N),
            static_cast<int64_t>(K), packed.data());

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    std::vector<float> src_scales(static_cast<size_t>(M) * groups);
    for (size_t i = 0; i < src_scales.size(); ++i)
        src_scales[i] = exact[(i + 1) % 4];

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);
    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s4; // Q4_0 arrives as s4 and is widened
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = 2; // Q4_0
    p.quant_params.src_scale.buff = src_scales.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, true, M, N, K, 1.0f, A.data(), K,
                      packed.data(), K, nullptr, 0.0f, C.data(), N, true, batch,
                      p),
            status_t::success);

    std::vector<float> ref;
    reference_gemm(M, N, K, kBlk, A.data(), K, wt.data(), K, true,
            wei_scales.data(), src_scales.data(), ref, groups, 1);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0) << "dst was never written -- the Q4_0 call did not reach "
                          "a kernel that computes";
}

TEST_F(Int8SymqDispatch, GgmlQ6_KPackedWeightComputesEndToEnd) {
    const int M = 5, N = 64, K = kQ6kSuper; // K must be a whole super-block
    const int groups = K / kQ6kGroup; // 16 scales per super-block

    std::mt19937 rng(6006);
    // Q6_K codes are six bits biased to [-32, 31].
    std::vector<int8_t> wt(static_cast<size_t>(N) * K);
    std::uniform_int_distribution<int> d6(-32, 31);
    for (auto &x : wt) x = static_cast<int8_t>(d6(rng));

    // The effective scale is d * scales[j]. Both factors are chosen so the
    // product is exact in fp16 and bf16: d is a power of two and the sub-block
    // scales are small integers.
    const float d = 0.03125f; // 2^-5
    std::vector<int8_t> sub_scales(kQ6kSuper / 16);
    for (size_t j = 0; j < sub_scales.size(); ++j)
        sub_scales[j] = static_cast<int8_t>(1 + (j % 4));

    std::vector<uint8_t> packed;
    pack_q6_k(wt.data(), sub_scales.data(), d, N, K, packed);

    // Group-major {G, N} scales, the layout the unpack writes and the kernel
    // reads. Every row of this weight shares one set of sub-block scales.
    std::vector<float> wei_scales(static_cast<size_t>(groups) * N);
    for (int g = 0; g < groups; ++g)
        for (int n = 0; n < N; ++n)
            wei_scales[static_cast<size_t>(g) * N + n]
                    = d * static_cast<float>(sub_scales[g % sub_scales.size()]);

    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    std::vector<float> src_scales(static_cast<size_t>(M) * groups);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < src_scales.size(); ++i)
        src_scales[i] = exact[(i + 1) % 4];

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);
    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = data_type_t::s8; // six-bit codes go out as s8
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = 14; // Q6_K
    p.quant_params.src_scale.buff = src_scales.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, true, M, N, K, 1.0f, A.data(), K,
                      packed.data(), K, nullptr, 0.0f, C.data(), N, true, batch,
                      p),
            status_t::success);

    std::vector<float> ref;
    reference_gemm(M, N, K, kQ6kGroup, A.data(), K, wt.data(), K, true,
            wei_scales.data(), src_scales.data(), ref, groups, 1);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0) << "dst was never written -- the Q6_K call did not reach "
                          "a kernel that computes";
}

// Granularities and options the adapter does not implement must be declined, not
// approximated. Each of these would otherwise be silently wrong: a per-token
// source scale applied as a scalar, a dropped bias, an ignored beta.
TEST_F(Int8SymqDispatch, DeclinesWhatItCannotExpress) {
    std::mt19937 rng(6);

    struct Variant {
        const char *why;
        std::function<void(matmul_params &, std::vector<float> &)> mutate;
    };
    std::vector<float> per_token(8, 0.01f);
    const Variant variants[] = {
            {"an AOCL-reordered weight buffer",
                    [](matmul_params &p, std::vector<float> &) {
                        // What the GGML unpack hands back today: same dtypes,
                        // same {G, N} scale, blocked layout. Reading it as
                        // row-major would be silent garbage.
                        p.mem_format_b = 'r';
                    }},
            {"a still-packed GGML weight",
                    [](matmul_params &p, std::vector<float> &) {
                        p.packing.pack_format_b = 1;
                    }},
            {"source scale that is neither per-token nor per-group",
                    [](matmul_params &p, std::vector<float> &pt) {
                        // {M, 3}: not 1 column, not one per weight group.
                        p.quant_params.src_scale.buff = pt.data();
                        p.quant_params.src_scale.dims = {8, 3};
                    }},
            {"weight scale that is not {groups, N}",
                    [](matmul_params &p, std::vector<float> &) {
                        p.quant_params.wei_scale.dims = {64};
                    }},
            {"non-zero source zero point",
                    [](matmul_params &p, std::vector<float> &) {
                        static int32_t zp = 3;
                        p.quant_params.src_zp.buff = &zp;
                        p.quant_params.src_zp.dt = data_type_t::s32;
                        p.quant_params.src_zp.dims = {1};
                    }},
    };

    for (const Variant &v : variants) {
        SCOPED_TRACE(v.why);
        SymqCall c(8, 64, 128, 32, true, rng);
        matmul_params p = c.make_params();
        v.mutate(p, per_token);
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = 2;
        matmul_batch_params_t batch;
        matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(), c.K,
                c.B.data(), c.K, nullptr, 0.0f, c.C.data(), c.N, true, batch, p);
        // The 128-bit kernel must not have run: it would have overwritten every
        // element of the poisoned dst.
        bool any_nan = false;
        for (float x : c.C)
            if (std::isnan(x)) any_nan = true;
        EXPECT_TRUE(any_nan)
                << "the kernel computed a call it does not implement";
    }
}

// alpha scales the product and the source scale multiplies the same product, so
// the adapter folds one into the other. That is only correct if it lands on the
// result exactly once.
TEST_F(Int8SymqDispatch, FoldsAlphaIntoTheSourceScale) {
    std::mt19937 rng(7);
    SymqCall c(16, 64, 128, 32, true, rng);
    const float alpha = 2.5f;
    matmul_params p = c.make_params();
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;
    ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, alpha, c.A.data(),
                      c.K, c.B.data(), c.K, nullptr, 0.0f, c.C.data(), c.N, true,
                      batch, p),
            status_t::success);

    std::vector<float> ref;
    const float scaled_alpha = c.src_scale * alpha;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &scaled_alpha, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
            kTolerance);
    EXPECT_EQ(nans, 0);
}


// ===========================================================================
// The four cases both INT8 adapters used to decline outright: beta, bias,
// post-ops and a bf16 destination. They are shared code (int8_epilogue_128), so
// the symmetric path carries most of the coverage and the k-quant path checks
// that it is wired to the same thing.
// ===========================================================================

// beta scales what is already in C. The kernel accumulates into C, so this is
// the one of the four the looper has to do itself, and the one that can go wrong
// silently: a looper that keeps memsetting would give an answer that is correct
// except for the term that was asked for.
TEST_F(Int8SymqDispatch, ScalesTheExistingDestinationByBeta) {
    std::mt19937 rng(881);
    SymqCall c(8, 64, 128, 32, true, rng);
    const float beta = 0.5f;

    // Seed C with something that survives the multiply exactly.
    std::vector<float> seed(static_cast<size_t>(c.M) * c.N);
    for (size_t i = 0; i < seed.size(); ++i)
        seed[i] = static_cast<float>((i % 17)) - 8.0f;
    c.C = seed;

    matmul_params p = c.make_params();
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;
    ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(),
                      c.K, c.B.data(), c.K, nullptr, beta, c.C.data(), c.N, true,
                      batch, p),
            status_t::success);

    std::vector<float> ref;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] += beta * seed[i];

    int nans = 0;
    EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
            kTolerance);
    EXPECT_EQ(nans, 0);
}

// beta == 1 is its own path in the looper -- it skips the scaling multiply
// rather than doing it -- so it is checked rather than assumed to follow from
// beta == 0.5 working.
TEST_F(Int8SymqDispatch, BetaOneAccumulatesOntoTheDestination) {
    std::mt19937 rng(882);
    SymqCall c(4, 64, 128, 32, true, rng);
    std::vector<float> seed(static_cast<size_t>(c.M) * c.N);
    for (size_t i = 0; i < seed.size(); ++i) seed[i] = 0.25f * ((i % 9) - 4);
    c.C = seed;

    matmul_params p = c.make_params();
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;
    ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(),
                      c.K, c.B.data(), c.K, nullptr, 1.0f, c.C.data(), c.N, true,
                      batch, p),
            status_t::success);

    std::vector<float> ref;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] += seed[i];
    int nans = 0;
    EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
            kTolerance);
    EXPECT_EQ(nans, 0);
}

// Bias then ReLU, in that order. Ordering is the thing worth pinning: applying
// the activation before the bias gives a plausible-looking wrong answer, and
// with a bias large enough to flip signs the two differ everywhere.
TEST_F(Int8SymqDispatch, AppliesBiasThenReluInThatOrder) {
    std::mt19937 rng(883);
    SymqCall c(8, 64, 128, 32, true, rng);

    std::vector<float> ref;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);

    // A bias per column, sized so it changes the sign of a good share of the
    // result -- otherwise ReLU-before-bias and bias-before-ReLU agree and the
    // test proves nothing.
    float mag = 0.0f;
    for (float v : ref) mag = std::max(mag, std::abs(v));
    std::vector<float> bias(c.N);
    for (int n = 0; n < c.N; ++n)
        bias[n] = ((n % 2) ? 0.5f : -0.5f) * mag;

    matmul_params p = c.make_params();
    matmul_post_op relu;
    relu.po_type = ops::post_op_type_t::relu; // alpha 0 is the pure form
    p.postop_.push_back(relu);
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;
    ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(),
                      c.K, c.B.data(), c.K, bias.data(), 0.0f, c.C.data(), c.N,
                      true, batch, p),
            status_t::success);

    for (int m = 0; m < c.M; ++m)
        for (int n = 0; n < c.N; ++n) {
            float &v = ref[static_cast<size_t>(m) * c.N + n];
            v = std::max(v + bias[n], 0.0f);
        }
    int nans = 0;
    EXPECT_LT(worst_scaled_error(c.C, c.N, ref, c.N, c.M, c.N, &nans),
            kTolerance);
    EXPECT_EQ(nans, 0);
}

// A bf16 destination goes through an fp32 scratch and a narrowing sweep. The
// tolerance is bf16's, not the kernel's: 8 mantissa bits is about 2^-8.
TEST_F(Int8SymqDispatch, WritesABf16Destination) {
    std::mt19937 rng(884);
    SymqCall c(8, 64, 128, 32, true, rng);
    std::vector<uint16_t> dst(static_cast<size_t>(c.M) * c.N, 0x7FC0); // NaN

    matmul_params p = c.make_params();
    p.dtypes.dst = data_type_t::bf16;
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;
    ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(),
                      c.K, c.B.data(), c.K, nullptr, 0.0f, dst.data(), c.N, true,
                      batch, p),
            status_t::success);

    std::vector<float> got(dst.size());
    for (size_t i = 0; i < dst.size(); ++i)
        got[i] = common::bfloat16_t::bf16_to_f32_val(
                static_cast<int16_t>(dst[i]));

    std::vector<float> ref;
    reference_gemm(c.M, c.N, c.K, c.gs, c.A.data(), c.K, c.B.data(), c.K, true,
            c.ws.data(), &c.src_scale, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(got, c.N, ref, c.N, c.M, c.N, &nans), 1.0f / 128.0f);
    EXPECT_EQ(nans, 0) << "dst was never written";
}

// apply_postops_tile() ends both its switches in `default: break;`. softmax,
// pooling and mish are in the enum and in neither switch, so accepting one would
// drop it in silence and return the un-activated product -- a wrong answer that
// looks entirely reasonable. The adapter must decline instead.
TEST_F(Int8SymqDispatch, DeclinesAPostOpTheEpilogueWouldSilentlyDrop) {
    std::mt19937 rng(885);
    const ops::post_op_type_t dropped[]
            = {ops::post_op_type_t::mish, ops::post_op_type_t::softmax};
    for (ops::post_op_type_t t : dropped) {
        SymqCall c(8, 64, 128, 32, true, rng);
        matmul_params p = c.make_params();
        matmul_post_op po;
        po.po_type = t;
        p.postop_.push_back(po);
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = 2;
        matmul_batch_params_t batch;
        matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f, c.A.data(), c.K,
                c.B.data(), c.K, nullptr, 0.0f, c.C.data(), c.N, true, batch, p);
        bool any_nan = false;
        for (float x : c.C)
            if (std::isnan(x)) any_nan = true;
        EXPECT_TRUE(any_nan) << "ran a call whose post-op it would have dropped";
    }
}

// {M} and {M, 1} are the same per-token scale written two ways. The second used
// to be accepted and the first declined, which is a decline over notation.
//
// One group, not the usual 32-wide grouping, because the API validator upstream
// requires the source and weight group counts to match and per-token is one
// group: a per-token source scale is only legal against a per-channel {1, N}
// weight scale. That is a deliberate rule rather than an oversight, so the test
// is written inside it instead of against it.
TEST_F(Int8SymqDispatch, AcceptsAOneDimensionalPerTokenScale) {
    std::mt19937 rng(886);
    SymqCall c(8, 64, 128, /*gs=*/128, true, rng);
    c.fill_vector_scales(rng);

    std::vector<float> got[2];
    for (int form = 0; form < 2; ++form) {
        c.C.assign(static_cast<size_t>(c.M) * c.N,
                std::numeric_limits<float>::quiet_NaN());
        matmul_params p = c.make_params();
        p.quant_params.src_scale.buff = c.src_scales_token.data();
        p.quant_params.src_scale.dims = form == 0
                ? std::vector<int64_t>{c.M, 1}
                : std::vector<int64_t>{c.M};
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = 2;
        matmul_batch_params_t batch;
        ASSERT_EQ(matmul_direct('r', false, true, c.M, c.N, c.K, 1.0f,
                          c.A.data(), c.K, c.B.data(), c.K, nullptr, 0.0f,
                          c.C.data(), c.N, true, batch, p),
                status_t::success);
        got[form] = c.C;
    }
    for (size_t i = 0; i < got[0].size(); ++i) {
        ASSERT_FALSE(std::isnan(got[1][i])) << "the 1-D form did not compute";
        EXPECT_EQ(got[0][i], got[1][i]) << "at " << i;
    }
}

// llama.cpp's test-backend-ops finds this shape wrong through the GGML path --
// MUL_MAT(type_a=q4_K, m=16, k=256), relative error 1.1 to 2.5 against a 5e-4
// tolerance, and erratically: n=1 and n=7 pass, n=2..6 and n=8,9 fail. It is not
// the weight cache (ZENDNNL_MATMUL_WEIGHT_CACHE=0 gives the same failures) and
// it does not show at model shapes, which is why perplexity never caught it.
//
// K=256 is exactly ONE k-quant super-block, the smallest legal weight, and every
// other test here uses two or more. M sweeps the same range the failure does, so
// if the fault is in the kernel rather than in the GGML unpack above it, it
// reproduces here against the dequantise-then-multiply reference.
TEST_P(Int8KquantDispatch, SingleSuperBlockAcrossTheFailingRowCounts) {
    const int type = GetParam();
    const int N = 16, K = 256;    // one super-block, eight 32-wide scale groups
    const int groups = K / kKsub;

    for (int M = 1; M <= 9; ++M) {
        SCOPED_TRACE("M=" + std::to_string(M));
        // Each iteration frees the previous weight and allocates a new one,
        // which malloc will happily place at the same address -- and the GGML
        // unpack cache is keyed on exactly that address. Without this clear the
        // test measures the cache, not the kernel. Note that
        // ZENDNNL_MATMUL_WEIGHT_CACHE=0 does NOT cover this cache; it is a
        // separate one with its own clear.
        clear_ggml_weight_unpack_cache();
        clear_all_weight_caches();
        std::mt19937 rng(type * 1000 + M);
        KquantSource src = build_kquant(type, N, K, rng);

        std::vector<int8_t> A(static_cast<size_t>(M) * K);
        fill_s8(A, rng, false);
        std::vector<float> ss(static_cast<size_t>(M) * groups);
        const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
        for (size_t i = 0; i < ss.size(); ++i) ss[i] = exact[(i + 1) % 4];

        std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

        matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = (type == 12) ? data_type_t::s4 : data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.packing.pack_format_b = 1;
        p.packing.ggml_type_b = type;
        p.quant_params.src_scale.buff = ss.data();
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {M, groups};
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = 2;
        matmul_batch_params_t batch;

        ASSERT_EQ(matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                          A.data(), K, src.blocks.data(), K, nullptr, 0.0f,
                          C.data(), N, /*is_weights_const=*/true, batch, p),
                status_t::success);

        std::vector<float> ref;
        reference_kquant(M, N, K, kKsub, A.data(), K, src.codes.data(), K,
                src.D.data(), src.Min.data(), ss.data(), groups, 1, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
        EXPECT_EQ(nans, 0) << "dst never written";
    }
}

// The GGML Q8_0 path AS LLAMA.CPP ACTUALLY CALLS IT: f32 activations handed
// straight over, with the library doing the quantisation. Every other Q8_0 test
// here pre-quantises to s8 and supplies its own src_scale, which is a different
// path -- and the one that was covered while the real one was not.
//
// test-backend-ops finds this returning an UNTOUCHED destination on a host with
// AOCL-DLP compiled in: relative error 58 to 102, and a "GFLOPS" figure of 669
// that is really just an empty result timed. AOCL-DLP cannot run per-group INT8
// without AVX-512 VNNI, so whichever gate should have routed this to the native
// kernel did not fire.
TEST_F(Int8SymqDispatch, GgmlQ8_0WithF32ActivationsComputes) {
    const int M = 6, N = 64, K = 256;
    const int groups = K / 32;
    std::mt19937 rng(9001);

    // GGML-packed Q8_0 weights, plus the s8 codes and scales they decode to, so
    // the answer can be checked rather than merely be non-NaN.
    std::vector<int8_t> q(static_cast<size_t>(N) * K);
    fill_s8(q, rng, false);
    std::vector<float> ws(static_cast<size_t>(groups) * N);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < ws.size(); ++i) ws[i] = exact[i % 4];
    std::vector<uint8_t> blocks(static_cast<size_t>(N) * groups * 34);
    repack_weights_q8_0(q.data(), ws.data(), static_cast<int64_t>(N),
            static_cast<int64_t>(K), blocks.data());

    // f32 activations -- the thing under test.
    std::vector<float> A_f32(static_cast<size_t>(M) * K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : A_f32) v = dist(rng);

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

    matmul_params p;
    p.dtypes.src = data_type_t::f32; // NOT s8: the library must quantise
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = 8; // Q8_0
    // Exactly what ggml-zendnn.cpp sets for this case: dynamic quantisation to
    // s8, no scale buffer (the library fills it), and a bf16 scale dtype.
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    p.quant_params.src_scale.buff = nullptr;
    p.quant_params.src_scale.dt = data_type_t::bf16;
    p.quant_params.src_scale.dims = {M, groups};
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      A_f32.data(), K, blocks.data(), K, nullptr, 0.0f, C.data(),
                      N, /*is_weights_const=*/true, batch, p),
            status_t::success);

    int nans = 0;
    for (float x : C)
        if (std::isnan(x)) ++nans;
    ASSERT_EQ(nans, 0) << "destination never written -- the call reached a "
                          "backend that returns without computing";

    // Reference straight from the decoded codes and scales, quantising A the
    // same way the library must: absmax/127 per row-group.
    std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int g = 0; g < groups; ++g) {
                for (int j = 0; j < 32; ++j) {
                    const int k = g * 32 + j;
                    acc += static_cast<double>(A_f32[m * K + k])
                            * static_cast<double>(q[n * K + k])
                            * static_cast<double>(ws[g * N + n]);
                }
            }
            ref[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
        }
    }
    // Loose: the library quantises A to 8 bits, the reference does not.
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), 0.05f);
}

// The decode GEMV reads GGML's packed blocks directly instead of going through
// the unpack. Same arithmetic as the k-quant looper, so it is held to the same
// reference -- dequantise, then multiply -- and to the looper itself, which is
// the comparison that matters: two independent routes to one answer.
TEST_P(Int8KquantDispatch, DecodeGemvMatchesTheReferenceAndTheLooper) {
    const int type = GetParam();
    const int groups_per_sb = 8;

    struct Shape {
        int N, K;
        const char *why;
    };
    const Shape shapes[] = {
            {64, 256, "one super-block"},
            {8, 512, "narrow, two super-blocks"},
            {128, 1024, "wider than a panel"},
            {33, 768, "ragged N"},
    };

    for (const Shape &sh : shapes) {
        SCOPED_TRACE(std::string(sh.why));
        clear_ggml_weight_unpack_cache();
        clear_all_weight_caches();

        std::mt19937 rng(type * 77 + sh.N + sh.K);
        KquantSource src = build_kquant(type, sh.N, sh.K, rng);

        const int n_groups = sh.K / kKsub;
        std::vector<int8_t> A(static_cast<size_t>(sh.K));
        fill_s8(A, rng, false);
        std::vector<float> ss(static_cast<size_t>(n_groups));
        const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
        for (size_t i = 0; i < ss.size(); ++i) ss[i] = exact[(i + 1) % 4];
        ASSERT_EQ(n_groups % groups_per_sb, 0);

        std::vector<float> C = poisoned(static_cast<size_t>(sh.N));
        ASSERT_TRUE(int8_q4k_gemv_128(sh.N, sh.K, type, A.data(),
                src.blocks.data(), C.data(), ss.data(), /*ss_grp=*/1,
                /*nthreads=*/2));

        std::vector<float> ref;
        reference_kquant(1, sh.N, sh.K, kKsub, A.data(), sh.K,
                src.codes.data(), sh.K, src.D.data(), src.Min.data(), ss.data(),
                /*ss_row=*/0, /*ss_grp=*/1, ref);
        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, sh.N, ref, sh.N, 1, sh.N, &nans),
                kTolerance);
        EXPECT_EQ(nans, 0) << "dst never written";
    }
}

// A per-tensor activation scale takes the other branch of the flush, and a
// kernel that ignored ss_grp would still pass the per-group test above by
// reading index 0 every time only if the scales happened to be equal. They are
// not, so this pins it.
TEST_P(Int8KquantDispatch, DecodeGemvHonoursAPerTensorScale) {
    const int type = GetParam();
    const int N = 64, K = 512;
    clear_ggml_weight_unpack_cache();
    clear_all_weight_caches();

    std::mt19937 rng(type * 31 + 5);
    KquantSource src = build_kquant(type, N, K, rng);
    std::vector<int8_t> A(static_cast<size_t>(K));
    fill_s8(A, rng, false);
    const float one_scale = 0.0625f;

    std::vector<float> C = poisoned(static_cast<size_t>(N));
    ASSERT_TRUE(int8_q4k_gemv_128(N, K, type, A.data(), src.blocks.data(),
            C.data(), &one_scale, /*ss_grp=*/0, /*nthreads=*/1));

    std::vector<float> flat(static_cast<size_t>(K / kKsub), one_scale);
    std::vector<float> ref;
    reference_kquant(1, N, K, kKsub, A.data(), K, src.codes.data(), K,
            src.D.data(), src.Min.data(), flat.data(), 0, 1, ref);
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, 1, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0);
}

// Q6_K decode. It matters more than its share of tensors suggests: a "Q4_K_M"
// model puts Q6_K on the output/lm_head and some attention tensors, and a
// profile of generation on the A10 put a THIRD of the time in ggml's
// vec_dot_q6_K_q8_K -- work this kernel did not touch before.
//
// The layout is the awkward part and is what this checks: four interleaved
// streams per 128-weight chunk, sub-blocks of sixteen rather than 32, and a
// symmetric -32 offset instead of a min term.
// Q6_K AS LLAMA.CPP CALLS IT: f32 activations, dynamic quantisation, and an
// activation scale per 32 elements. The existing Q6_K test hands over
// pre-quantised s8 with its own scales, which is a different path -- exactly
// the gap that let the Q8_0 silent-no-compute bug through.
//
// The mismatch this is built to expose: a Q6_K weight scales every SIXTEEN
// elements, so the unpack declares wei_scale {K/16, N}, while llama.cpp's
// activation scale is {M, K/32}. The symmetric adapter requires the source
// scale to be per-tensor, {M,1} or {M, groups} with groups from the WEIGHT --
// and K/32 is none of those. If that is why enabling Q6_K produced NaN under
// llama-perplexity, it reproduces here.
TEST_F(Int8SymqDispatch, GgmlQ6_KWithF32ActivationsComputes) {
    // M = 512 because that is llama-perplexity's chunk size, which is where the
    // NaN appears; M = 5 passed and told us nothing. The GEMM tiles M at four,
    // so a row count in the hundreds exercises the padded-tail and K-blocking
    // paths that five never reaches.
    const int M = 512, N = 64, K = kQ6kSuper * 2;

    std::mt19937 rng(6140);
    std::vector<int8_t> vals(static_cast<size_t>(N) * K);
    std::uniform_int_distribution<int> vd(-32, 31);
    for (auto &v : vals) v = static_cast<int8_t>(vd(rng));

    std::vector<int8_t> sub_scales(kQ6kSuper / 16);
    for (size_t i = 0; i < sub_scales.size(); ++i)
        sub_scales[i] = static_cast<int8_t>(1 + (i % 5));
    const float d = 0.03125f;

    std::vector<uint8_t> blocks;
    pack_q6_k(vals.data(), sub_scales.data(), d, N, K, blocks);

    std::vector<float> A_f32(static_cast<size_t>(M) * K);
    std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
    for (auto &v : A_f32) v = ad(rng);

    std::vector<float> C = poisoned(static_cast<size_t>(M) * N);

    matmul_params p;
    p.dtypes.src = data_type_t::f32; // library must quantise
    p.dtypes.wei = data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = 14; // Q6_K
    // Exactly what ggml-zendnn sets: dynamic to s8, no buffer, bf16 scales,
    // and dims keyed on QK8_0 = 32 regardless of the weight's grouping.
    p.dtypes.compute = data_type_t::s8;
    p.dynamic_quant = true;
    p.quant_params.src_scale.buff = nullptr;
    p.quant_params.src_scale.dt = data_type_t::bf16;
    p.quant_params.src_scale.dims = {M, K / 32};
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, /*transB=*/true, M, N, K, 1.0f,
                      A_f32.data(), K, blocks.data(), K, nullptr, 0.0f,
                      C.data(), N, /*is_weights_const=*/true, batch, p),
            status_t::success);

    int nans = 0;
    for (float x : C)
        if (std::isnan(x)) ++nans;
    ASSERT_EQ(nans, 0) << "destination holds NaN -- this is the llama-perplexity "
                          "failure reproduced in tree";

    // Reference: w = d * sc[j] * value over sixteen-wide sub-blocks.
    std::vector<float> ref(static_cast<size_t>(M) * N, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int j = 0; j < K / 16; ++j) {
                const int sc = sub_scales[j % (kQ6kSuper / 16)];
                double gsum = 0.0;
                for (int e = 0; e < 16; ++e) {
                    const int k = j * 16 + e;
                    gsum += static_cast<double>(A_f32[m * K + k])
                            * static_cast<double>(
                                    vals[static_cast<size_t>(n) * K + k]);
                }
                acc += static_cast<double>(d) * static_cast<double>(sc) * gsum;
            }
            ref[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
        }
    }
    // Loose: the library quantises the activations to 8 bits, the reference
    // does not.
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), 0.05f);
}

TEST(Int8Q6KGemv, MatchesTheDequantiseThenMultiplyReference) {
    struct Shape {
        int N, K;
        const char *why;
    };
    const Shape shapes[] = {
            {64, 256, "one super-block"},
            {16, 512, "two super-blocks"},
            {33, 768, "ragged N"},
    };

    for (const Shape &sh : shapes) {
        SCOPED_TRACE(std::string(sh.why));
        std::mt19937 rng(9060 + sh.N + sh.K);

        // Signed codes in [-32, 31]; pack_q6_k biases them to [0, 63].
        std::vector<int8_t> vals(static_cast<size_t>(sh.N) * sh.K);
        std::uniform_int_distribution<int> vd(-32, 31);
        for (auto &v : vals) v = static_cast<int8_t>(vd(rng));

        // Per-sub-block scales, exact so the comparison measures arithmetic
        // rather than fp16 rounding.
        std::vector<int8_t> sub_scales(kQ6kSuper / 16);
        for (size_t i = 0; i < sub_scales.size(); ++i)
            sub_scales[i] = static_cast<int8_t>(1 + (i % 5));
        const float d = 0.03125f;

        std::vector<uint8_t> blocks;
        pack_q6_k(vals.data(), sub_scales.data(), d, sh.N, sh.K, blocks);

        std::vector<int8_t> A(static_cast<size_t>(sh.K));
        fill_s8(A, rng, false);

        // ACTIVATION scales are per 32 elements -- that is what the GGML
        // backend supplies (src_scale.dims = {n, k/QK8_0}) -- even though a
        // Q6_K WEIGHT sub-block is sixteen wide, so each pair of sub-blocks
        // shares one. Writing this test in the kernel's convention instead of
        // the caller's is exactly how the first version passed while
        // llama-perplexity returned NaN, so the granularities are kept
        // deliberately distinct here.
        const int n_sub = sh.K / 16;   // weight sub-blocks
        const int n_groups = sh.K / 32; // activation scale groups
        std::vector<float> ss(static_cast<size_t>(n_groups));
        const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
        for (size_t i = 0; i < ss.size(); ++i) ss[i] = exact[(i + 1) % 4];

        std::vector<float> C = poisoned(static_cast<size_t>(sh.N));
        ASSERT_TRUE(int8_q4k_gemv_128(sh.N, sh.K, /*ggml_type=*/14, A.data(),
                blocks.data(), C.data(), ss.data(), /*ss_grp=*/1,
                /*nthreads=*/2));

        // Reference: w = d * sc[j] * value, straight from what the packer was
        // given, summed per sixteen-wide group with that group's scale.
        std::vector<float> ref(static_cast<size_t>(sh.N), 0.0f);
        for (int n = 0; n < sh.N; ++n) {
            double acc = 0.0;
            for (int j = 0; j < n_sub; ++j) {
                const int sc = sub_scales[j % (kQ6kSuper / 16)];
                double gsum = 0.0;
                for (int e = 0; e < 16; ++e) {
                    const int k = j * 16 + e;
                    gsum += static_cast<double>(A[k])
                            * static_cast<double>(vals[static_cast<size_t>(n)
                                            * sh.K
                                    + k]);
                }
                // sub-block j sits inside activation group j/2
                acc += static_cast<double>(ss[j / 2]) * static_cast<double>(d)
                        * static_cast<double>(sc) * gsum;
            }
            ref[n] = static_cast<float>(acc);
        }

        int nans = 0;
        EXPECT_LT(worst_scaled_error(C, sh.N, ref, sh.N, 1, sh.N, &nans),
                kTolerance);
        EXPECT_EQ(nans, 0) << "dst never written";
    }
}

TEST(Int8Q4KGemvGate, DeclinesWhatItDoesNotExpress) {
    EXPECT_FALSE(int8_q4k_gemv_supported(2, 64, 256, 12)) << "M>1 is the GEMM's";
    EXPECT_FALSE(int8_q4k_gemv_supported(1, 64, 128, 12)) << "K not whole blocks";
    EXPECT_FALSE(int8_q4k_gemv_supported(1, 64, 256, 8)) << "Q8_0 is not a k-quant";
    EXPECT_TRUE(int8_q4k_gemv_supported(1, 64, 256, 14)) << "Q6_K";
    EXPECT_TRUE(int8_q4k_gemv_supported(1, 64, 256, 12));
    EXPECT_TRUE(int8_q4k_gemv_supported(1, 64, 512, 13));
}

// The k-quant adapter shares the epilogue rather than having its own, so one
// case that exercises beta and bias together is enough to prove it is wired to
// it -- the arithmetic is covered above.
TEST_P(Int8KquantDispatch, HonoursBetaAndBias) {
    const int type = GetParam();
    const int M = 6, N = 64, K = 512;
    const int groups = K / kKsub;

    std::mt19937 rng(type == 12 ? 5120 : 5130);
    KquantSource src = build_kquant(type, N, K, rng);
    std::vector<int8_t> A(static_cast<size_t>(M) * K);
    fill_s8(A, rng, false);
    std::vector<float> ss(static_cast<size_t>(M) * groups);
    const float exact[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    for (size_t i = 0; i < ss.size(); ++i) ss[i] = exact[(i + 1) % 4];

    std::vector<float> ref;
    reference_kquant(M, N, K, kKsub, A.data(), K, src.codes.data(), K,
            src.D.data(), src.Min.data(), ss.data(), groups, 1, ref);

    const float beta = 0.25f;
    std::vector<float> seed(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < seed.size(); ++i) seed[i] = 0.5f * ((i % 7) - 3);
    std::vector<float> bias(N);
    for (int n = 0; n < N; ++n) bias[n] = 0.125f * ((n % 5) - 2);
    std::vector<float> C = seed;

    matmul_params p;
    p.dtypes.src = data_type_t::s8;
    p.dtypes.wei = (type == 12) ? data_type_t::s4 : data_type_t::s8;
    p.dtypes.dst = data_type_t::f32;
    p.packing.pack_format_b = 1;
    p.packing.ggml_type_b = type;
    p.quant_params.src_scale.buff = ss.data();
    p.quant_params.src_scale.dt = data_type_t::f32;
    p.quant_params.src_scale.dims = {M, groups};
    p.lowoha_algo = matmul_algo_t::native_gemm;
    p.num_threads = 2;
    matmul_batch_params_t batch;

    ASSERT_EQ(matmul_direct('r', false, true, M, N, K, 1.0f, A.data(), K,
                      src.blocks.data(), K, bias.data(), beta, C.data(), N, true,
                      batch, p),
            status_t::success);

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            const size_t i = static_cast<size_t>(m) * N + n;
            ref[i] += beta * seed[i] + bias[n];
        }
    int nans = 0;
    EXPECT_LT(worst_scaled_error(C, N, ref, N, M, N, &nans), kTolerance);
    EXPECT_EQ(nans, 0);
}

} // namespace
} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
