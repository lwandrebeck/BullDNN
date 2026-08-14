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
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"
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
    }
    void TearDown() override { clear_all_weight_caches(); }
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

} // namespace
} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
