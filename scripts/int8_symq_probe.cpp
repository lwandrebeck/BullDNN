// Standalone probe for the symmetric per-group INT8 path (ukernel + looper).
//
// Three modes, because they answer different questions:
//
//   validate    the looper against an int64/double reference, scale indexing
//               written longhand rather than borrowed from the looper
//   looper      end-to-end GOPS as a caller sees it today, B packed per call
//   ukernel     the hot loop alone over a pre-packed panel -- the instrument
//               for a change confined to the microkernel, where the looper's
//               per-call packing would otherwise dilute the effect
//
// Not a gtest: nothing dispatches to this path yet, so there is no library
// entry point to drive. It compiles the two sources directly with the same
// flags the library uses.

#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.hpp"

using namespace zendnnl::lowoha::matmul::native;

// Set by the build to name the tile/spill variant under test, so a log line
// cannot be attributed to the wrong binary.
#ifndef SYMQ_PROBE_VARIANT
#define SYMQ_PROBE_VARIANT "baseline"
#endif

namespace {

using clk = std::chrono::steady_clock;

// The kernels take the activation scale as a pointer plus strides now;
// this probe only ever uses the per-tensor form, i.e. strides {0, 0}.
constexpr float kOne = 1.0f;
constexpr float kPointOne = 0.01f;

double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// Anything past this cannot have happened on family 15h: 2 modules x 2 128-bit
// pipes at 2.8 GHz is ~11.2 G vector ops/s, and the kernel needs ~34 of them
// per 256 integer ops, so ~84 GOPS is the instruction-issue ceiling. 400 is far
// enough above it to only ever catch a didn't-run, never a real measurement.
constexpr double kGopsCeiling = 400.0;

std::string gops(double ops, double s) {
    if (s <= 0.0) return "IMPOSSIBLE(inf)";
    const double g = ops / s / 1e9;
    char buf[64];
    if (g > kGopsCeiling) {
        std::snprintf(buf, sizeof(buf), "IMPOSSIBLE(%.1f)", g);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f", g);
    }
    return buf;
}

// Every byte in [-127, 127]: the kernel's precondition, and what every GGML
// quantiser already guarantees.
void fill_s8(std::vector<int8_t> &v, std::mt19937 &rng, bool positive = false) {
    std::uniform_int_distribution<int> d(positive ? 1 : -127, 127);
    for (auto &x : v) x = static_cast<int8_t>(d(rng));
}

void fill_scales(std::vector<float> &v, std::mt19937 &rng) {
    std::uniform_real_distribution<float> d(0.002f, 0.05f);
    for (auto &x : v) x = d(rng);
}

// ---------------------------------------------------------------- validate --

// Reference: int64 accumulation so it cannot itself saturate, double for the
// scale sum, and the group/scale indexing spelled out from the header's
// contract rather than reused from the looper -- an off-by-one in the
// group-major scale layout is the failure most likely to look plausible.
void reference(int M, int N, int K, int gs, const int8_t *A, int lda,
        const int8_t *B, int ldb, bool transB, const float *ws, float ss,
        std::vector<float> &out) {
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
                        * static_cast<double>(ws[static_cast<size_t>(g) * N + n])
                        * static_cast<double>(ss);
            }
            out[static_cast<size_t>(m) * N + n] = static_cast<float>(sum);
        }
    }
}

struct Case {
    int M, N, K, gs;
    bool transB;
    int nt;
    // Positive-only data removes the cancellation that makes a signed C
    // element's own magnitude meaningless, so every element carries its full
    // weight and a scale-index slip cannot hide behind a near-zero reference.
    bool positive;
    const char *why;
};

int run_validate() {
    const Case cases[] = {
            {4, 8, 32, 32, false, 1, false, "one tile, one group"},
            {4, 8, 64, 16, false, 1, false, "Q6_K group size"},
            {1, 4096, 4096, 32, true, 4, false, "llama.cpp decode, GGML layout"},
            {7, 13, 96, 32, false, 1, false, "ragged M and N"},
            {5, 70, 128, 16, false, 2, false, "ragged, crosses panel tail"},
            {128, 256, 512, 32, true, 4, false, "prompt GEMM, GGML layout"},
            {3, 64, 32, 32, true, 1, false, "M below MR"},
            {4, 8, 4096, 32, false, 1, false, "long K, many flushes"},
            {64, 64, 64, 64, false, 2, false, "group spans whole K"},
            {2, 9, 48, 16, true, 3, false, "odd everything"},
            {4, 65, 32, 32, false, 4, false, "N one past a panel"},
            {256, 32, 256, 32, false, 4, false, "tall M"},
            {4, 8, 128, 32, false, 1, true, "positive only, 4 groups"},
            {4, 8, 128, 16, false, 1, true, "positive only, gs=16"},
            {68, 130, 256, 32, true, 4, true, "positive only, GGML layout"},
    };

    std::mt19937 rng(20260814);
    int failures = 0;

    for (const Case &c : cases) {
        const int lda = c.K;
        const int ldb = c.transB ? c.K : c.N;
        const int ldc = c.N;
        const int n_groups = c.K / c.gs;

        std::vector<int8_t> A(static_cast<size_t>(c.M) * lda);
        std::vector<int8_t> B(c.transB
                        ? static_cast<size_t>(c.N) * ldb
                        : static_cast<size_t>(c.K) * ldb);
        std::vector<float> ws(static_cast<size_t>(n_groups) * c.N);
        fill_s8(A, rng, c.positive);
        fill_s8(B, rng, c.positive);
        fill_scales(ws, rng);
        const float ss = 0.0137f;

        // Poison C so a path that fails to write cannot pass.
        std::vector<float> C(static_cast<size_t>(c.M) * ldc,
                std::numeric_limits<float>::quiet_NaN());

        const bool ok = int8_symq_execute_128(c.M, c.N, c.K, c.gs, A.data(), lda,
                B.data(), ldb, c.transB, C.data(), ldc, ws.data(), &ss, 0, 0, c.nt);
        if (!ok) {
            std::printf("FAIL  %-34s declined M=%d N=%d K=%d gs=%d\n", c.why,
                    c.M, c.N, c.K, c.gs);
            ++failures;
            continue;
        }

        std::vector<float> R;
        reference(c.M, c.N, c.K, c.gs, A.data(), lda, B.data(), ldb, c.transB,
                ws.data(), ss, R);

        // Scale the error by the tensor's own magnitude, not by each element's.
        // A C element here is a sum of K/gs signed group terms, so individual
        // elements cancel to near zero while the absolute error stays at the
        // rounding of the terms that built them -- a per-element relative test
        // reports 1e-3 on a kernel that is exact to fp32. Normalising by
        // max|ref| is the usual GEMM criterion and still catches a scale-index
        // slip, which moves elements by their own magnitude.
        double max_abs = 0.0;
        for (float v : R) max_abs = std::max(max_abs, (double)std::fabs(v));
        const double scale = std::max(max_abs, 1e-30);

        double worst = 0.0, worst_elt_rel = 0.0, worst_exp = 0.0;
        int wm = -1, wn = -1, nans = 0;
        for (int m = 0; m < c.M; ++m) {
            for (int n = 0; n < c.N; ++n) {
                const float got = C[static_cast<size_t>(m) * ldc + n];
                const float exp = R[static_cast<size_t>(m) * c.N + n];
                if (std::isnan(got)) {
                    ++nans;
                    continue;
                }
                const double err = std::fabs((double)got - (double)exp) / scale;
                if (err > worst) {
                    worst = err;
                    wm = m;
                    wn = n;
                    worst_exp = exp;
                    worst_elt_rel = std::fabs((double)got - (double)exp)
                            / std::max(1e-30, (double)std::fabs(exp));
                }
            }
        }
        const bool pass = nans == 0 && worst < 1e-6;
        std::printf("%-5s %-34s M=%-4d N=%-6d K=%-5d gs=%-3d t=%d  err/max|ref| "
                    "%.2e  (elt rel %.2e at ref %.3g, max|ref| %.3g%s)\n",
                pass ? "ok" : "FAIL", c.why, c.M, c.N, c.K, c.gs, c.nt, worst,
                worst_elt_rel, worst_exp, max_abs,
                nans ? ", UNWRITTEN NaNs" : "");
        if (!pass) ++failures;
    }

    // Shapes outside what the microkernel can express must be refused, not
    // approximated, and must leave C untouched.
    {
        std::vector<int8_t> A(64), B(64);
        std::vector<float> ws(64, 1.0f), C(64, -7.0f);
        struct {
            int K, gs;
            const char *why;
        } bad[] = {{30, 32, "K not a whole number of groups"},
                {32, 6, "group size splits a VNNI quad"}};
        for (auto &b : bad) {
            const bool ok = int8_symq_execute_128(1, 8, b.K, b.gs, A.data(),
                    b.K, B.data(), 8, false, C.data(), 8, ws.data(), &kOne, 0, 0, 1);
            const bool untouched = C[0] == -7.0f;
            const bool pass = !ok && untouched;
            std::printf("%-5s declined: %-24s (returned %d, C %s)\n",
                    pass ? "ok" : "FAIL", b.why, ok,
                    untouched ? "untouched" : "WRITTEN");
            if (!pass) ++failures;
        }
    }

    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "PASSED",
            failures);
    return failures ? 1 : 0;
}

// ------------------------------------------------------------------- bench --

struct Shape {
    int M, N, K, gs;
    bool transB;
    const char *tag;
};

void bench_looper(int reps, int nt) {
    const Shape shapes[] = {
            {1, 4096, 4096, 32, true, "decode qkv"},
            {1, 11008, 4096, 32, true, "decode ffn"},
            {1, 4096, 11008, 32, true, "decode ffn-down"},
            {128, 4096, 4096, 32, true, "prompt-128 qkv"},
            {512, 4096, 4096, 32, true, "prompt-512 qkv"},
            {1024, 1024, 1024, 32, false, "square 1024"},
            {128, 4096, 4096, 16, true, "prompt-128 Q6_K gs=16"},
    };

    std::mt19937 rng(7);
    std::printf("%-24s %-6s %-6s %-6s %-3s %-2s %10s %10s %10s\n", "shape", "M",
            "N", "K", "gs", "t", "best GOPS", "min GOPS", "spread");
    for (const Shape &s : shapes) {
        const int lda = s.K;
        const int ldb = s.transB ? s.K : s.N;
        const int ldc = s.N;
        const int n_groups = s.K / s.gs;

        std::vector<int8_t> A(static_cast<size_t>(s.M) * lda);
        std::vector<int8_t> B(s.transB
                        ? static_cast<size_t>(s.N) * ldb
                        : static_cast<size_t>(s.K) * ldb);
        std::vector<float> ws(static_cast<size_t>(n_groups) * s.N);
        std::vector<float> C(static_cast<size_t>(s.M) * ldc, 0.0f);
        fill_s8(A, rng);
        fill_s8(B, rng);
        fill_scales(ws, rng);

        const double ops = 2.0 * s.M * s.N * s.K;
        double best = 0.0, minv = 1e30;

        if (!int8_symq_execute_128(s.M, s.N, s.K, s.gs, A.data(), lda, B.data(),
                    ldb, s.transB, C.data(), ldc, ws.data(), &kPointOne, 0, 0, nt)) {
            std::printf("%-24s DECLINED\n", s.tag);
            continue;
        }
        for (int r = 0; r < reps; ++r) {
            const auto t0 = clk::now();
            int8_symq_execute_128(s.M, s.N, s.K, s.gs, A.data(), lda, B.data(),
                    ldb, s.transB, C.data(), ldc, ws.data(), &kPointOne, 0, 0, nt);
            const double dt = secs_since(t0);
            const double g = ops / dt / 1e9;
            if (g > best) best = g;
            if (g < minv) minv = g;
        }
        std::printf("%-24s %-6d %-6d %-6d %-3d %-2d %10s %10s %9.1f%%\n", s.tag,
                s.M, s.N, s.K, s.gs, nt, gops(ops, ops / best / 1e9).c_str(),
                gops(ops, ops / minv / 1e9).c_str(),
                100.0 * (best / minv - 1.0));
        std::fflush(stdout);
    }
}

// The hot loop alone: B pre-packed once, one 64-column panel walked as the
// looper walks it, single thread, no OMP and no packing in the timed region.
// This is what a microkernel-confined change moves.
void bench_ukernel(int reps) {
    const int8_symq_ukernel_128_fn_t hot = select_int8_symq_ukernel_128();
    if (hot == nullptr) {
        std::printf("no 128-bit ukernel on this host\n");
        return;
    }
    constexpr int kPanelW = 64;
    const int b_stride = kPanelW * SYMQ_VNNI_GRP;
    // Fixed C footprint, so a variant with a smaller tile pays for the extra
    // passes it makes over A and over the packed panel instead of being handed
    // a smaller problem. Every variant computes the same 8 x 64 block.
    constexpr int kMTile = 8;

    struct { int K, gs; const char *tag; } cfgs[] = {
            {1024, 32, "K=1024 gs=32"},
            {4096, 32, "K=4096 gs=32"},
            {1024, 16, "K=1024 gs=16"},
    };

    std::mt19937 rng(11);
    std::printf("%-16s %-4s %10s %10s %10s\n", "hot loop", "MR", "best GOPS",
            "min GOPS", "spread");
    for (auto &c : cfgs) {
        const int n_quads = c.K / SYMQ_VNNI_GRP;
        const int n_groups = c.K / c.gs;
        std::vector<int8_t> A(static_cast<size_t>(kMTile) * c.K);
        std::vector<int8_t> panel(static_cast<size_t>(n_quads) * b_stride);
        std::vector<float> ws(static_cast<size_t>(n_groups) * kPanelW);
        std::vector<float> C(static_cast<size_t>(kMTile) * kPanelW, 0.0f);
        fill_s8(A, rng);
        fill_s8(panel, rng);
        fill_scales(ws, rng);

        // One sweep = the whole 8 x 64 block, walked as the looper walks it.
        const double ops = 2.0 * kMTile * kPanelW * c.K;
        double best = 0.0, minv = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = clk::now();
            for (int ic = 0; ic < kMTile; ic += SYMQ_MR) {
                for (int jr = 0; jr < kPanelW; jr += SYMQ_NR) {
                    hot(A.data() + static_cast<size_t>(ic) * c.K, c.K,
                            panel.data() + jr * SYMQ_VNNI_GRP, b_stride,
                            C.data() + static_cast<size_t>(ic) * kPanelW + jr,
                            kPanelW, c.K, c.gs, ws.data() + jr, kPanelW, &kPointOne, 0, 0);
                }
            }
            const double dt = secs_since(t0);
            const double g = ops / dt / 1e9;
            if (g > best) best = g;
            if (g < minv) minv = g;
        }
        std::printf("%-16s %dx%-2d %10s %10s %9.1f%%\n", c.tag, SYMQ_MR,
                SYMQ_NR, gops(ops, ops / best / 1e9).c_str(),
                gops(ops, ops / minv / 1e9).c_str(),
                100.0 * (best / minv - 1.0));
        std::fflush(stdout);
    }
}

} // namespace

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "validate";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const int nt = argc > 3 ? std::atoi(argv[3]) : 4;

    // Proof of which flavour ran: the two arms must resolve to different code.
    // A pointer is not a name, but a differing pointer between arms plus
    // vpmadcswd in the disassembly settles it.
    std::printf("# ukernel=%p  ZENDNNL_NATIVE_SYMQ_NO_XOP=%s  variant=%s\n",
            reinterpret_cast<void *>(select_int8_symq_ukernel_128()),
            std::getenv("ZENDNNL_NATIVE_SYMQ_NO_XOP")
                    ? std::getenv("ZENDNNL_NATIVE_SYMQ_NO_XOP")
                    : "(unset)",
            SYMQ_PROBE_VARIANT);

    if (mode == "validate") return run_validate();
    if (mode == "looper") {
        bench_looper(reps, nt);
        return 0;
    }
    if (mode == "ukernel") {
        bench_ukernel(reps);
        return 0;
    }
    std::fprintf(stderr, "usage: %s validate|looper|ukernel [reps] [threads]\n",
            argv[0]);
    return 2;
}
