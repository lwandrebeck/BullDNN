// FP32 native GEMM throughput, for A/B-ing the microkernel's multiply-add form
// and its register pinning on a given microarchitecture.
//
// Driven through matmul_direct with the native algo named explicitly rather than
// through auto_tuner: the question here is what the native kernel does, not what
// the dispatcher would pick, and auto_tuner could answer with libxsmm.
//
// The flavour under test is selected at compile time in the kernel TU
// (ZENDNNL_FORCE_FMA / -mno-fma), so the variant name is passed in for the log
// line and a run whose label disagrees with its object is the one mistake this
// cannot detect for you.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"

using namespace zendnnl;
using namespace zendnnl::lowoha::matmul;
using clk = std::chrono::steady_clock;

namespace {

struct Shape {
    int M, K, N;
};

// Peak for a family 15h part is 2 FMACs x 4 lanes x 2 FLOP x clock x modules;
// 200 GFLOPS is far above any of them, so it only ever catches a didn't-run.
constexpr double kCeiling = 200.0;

std::string gflops(double flops, double secs) {
    char buf[64];
    if (secs <= 0.0) return "IMPOSSIBLE";
    const double g = flops / secs / 1e9;
    if (g > kCeiling)
        std::snprintf(buf, sizeof(buf), "IMPOSSIBLE(%.1f)", g);
    else
        std::snprintf(buf, sizeof(buf), "%.2f", g);
    return buf;
}

} // namespace

int main(int argc, char **argv) {
    const std::string variant = argc > 1 ? argv[1] : "default";
    const int reps = argc > 2 ? std::atoi(argv[2]) : 7;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 4;

    const Shape shapes[] = {
            {512, 512, 512},
            {2048, 512, 512},
            {1024, 1024, 1024},
            {512, 2048, 2048},
            {1, 4096, 4096},
    };

    std::mt19937 rng(20260814);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::printf("# variant=%s threads=%d reps=%d\n", variant.c_str(), threads,
            reps);
    std::printf("%-6s %-6s %-6s %12s %12s\n", "M", "K", "N", "best GFLOPS",
            "min GFLOPS");
    for (const Shape &s : shapes) {
        std::vector<float> A(static_cast<size_t>(s.M) * s.K);
        std::vector<float> B(static_cast<size_t>(s.K) * s.N);
        std::vector<float> C(static_cast<size_t>(s.M) * s.N, 0.0f);
        for (auto &x : A) x = dist(rng);
        for (auto &x : B) x = dist(rng);

        matmul_params p;
        p.dtypes.src = data_type_t::f32;
        p.dtypes.wei = data_type_t::f32;
        p.dtypes.dst = data_type_t::f32;
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = threads;
        matmul_batch_params_t batch;

        auto call = [&]() {
            return matmul_direct('r', false, false, s.M, s.N, s.K, 1.0f,
                    A.data(), s.K, B.data(), s.N, nullptr, 0.0f, C.data(), s.N,
                    true, batch, p);
        };
        if (call() != status_t::success) {
            std::printf("%-6d %-6d %-6d %12s\n", s.M, s.K, s.N, "DECLINED");
            continue;
        }

        const double flops = 2.0 * s.M * s.N * s.K;
        double best = 0.0, worst = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = clk::now();
            call();
            const double dt
                    = std::chrono::duration<double>(clk::now() - t0).count();
            const double g = flops / dt / 1e9;
            if (g > best) best = g;
            if (g < worst) worst = g;
        }
        std::printf("%-6d %-6d %-6d %12s %12s\n", s.M, s.K, s.N,
                gflops(flops, flops / best / 1e9).c_str(),
                gflops(flops, flops / worst / 1e9).c_str());
        std::fflush(stdout);
    }
    return 0;
}
