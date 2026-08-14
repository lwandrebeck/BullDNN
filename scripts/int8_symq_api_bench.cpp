// End-to-end throughput of the per-group INT8 path as a caller sees it, through
// matmul_direct rather than the kernel entry point, so the adapter's per-call
// work (operand scan, scale widening) and the looper's packing are all inside
// the timed region. That is the number llama.cpp would observe.
//
// is_weights_const is true and the same weight pointer is reused across
// repetitions, which is what a model does: the weight is packed once in
// principle, so any per-call packing shows up here as a gap between the first
// call and the rest -- or, before the cache exists, as no gap at all.

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
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

constexpr double kGopsCeiling = 400.0;

std::string gops(double ops, double secs) {
    if (secs <= 0.0) return "IMPOSSIBLE(inf)";
    const double g = ops / secs / 1e9;
    char buf[64];
    if (g > kGopsCeiling)
        std::snprintf(buf, sizeof(buf), "IMPOSSIBLE(%.1f)", g);
    else
        std::snprintf(buf, sizeof(buf), "%.2f", g);
    return buf;
}

struct Shape {
    int M, N, K, gs;
    const char *tag;
};

} // namespace

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 7;
    const int nthreads = argc > 2 ? std::atoi(argv[2]) : 4;
    const bool bf16_scales = argc > 3 && std::atoi(argv[3]) != 0;

    // The GGML layout: B is N x K (transB), scales are group-major {K/gs, N}.
    const Shape shapes[] = {
            {1, 4096, 4096, 32, "decode qkv"},
            {1, 11008, 4096, 32, "decode ffn"},
            {1, 4096, 11008, 32, "decode ffn-down"},
            {128, 4096, 4096, 32, "prompt-128 qkv"},
            {512, 4096, 4096, 32, "prompt-512 qkv"},
            {1024, 1024, 1024, 32, "square 1024"},
            {128, 4096, 4096, 16, "prompt-128 gs=16"},
    };

    std::mt19937 rng(20260814);
    std::uniform_int_distribution<int> d8(-127, 127);
    std::uniform_real_distribution<float> dsc(0.002f, 0.05f);

    std::printf("%-20s %-6s %-6s %-6s %-3s %-2s %10s %10s %10s\n", "shape", "M",
            "N", "K", "gs", "t", "first ms", "best GOPS", "min GOPS");
    for (const Shape &s : shapes) {
        std::vector<int8_t> A(static_cast<size_t>(s.M) * s.K);
        std::vector<int8_t> B(static_cast<size_t>(s.N) * s.K);
        std::vector<float> ws(static_cast<size_t>(s.K / s.gs) * s.N);
        std::vector<uint16_t> ws_bf16;
        std::vector<float> C(static_cast<size_t>(s.M) * s.N, 0.0f);
        for (auto &x : A) x = static_cast<int8_t>(d8(rng));
        for (auto &x : B) x = static_cast<int8_t>(d8(rng));
        for (auto &x : ws) x = dsc(rng);
        if (bf16_scales) {
            // Truncate to bf16 the way the GGML unpack stores them.
            ws_bf16.resize(ws.size());
            for (size_t i = 0; i < ws.size(); ++i) {
                uint32_t bits;
                std::memcpy(&bits, &ws[i], 4);
                ws_bf16[i] = static_cast<uint16_t>(bits >> 16);
            }
        }
        float src_scale = 0.0137f;

        matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.quant_params.src_scale.buff = &src_scale;
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {1};
        p.quant_params.wei_scale.buff = bf16_scales
                ? static_cast<const void *>(ws_bf16.data())
                : static_cast<const void *>(ws.data());
        p.quant_params.wei_scale.dt
                = bf16_scales ? data_type_t::bf16 : data_type_t::f32;
        p.quant_params.wei_scale.dims = {s.K / s.gs, s.N};
        p.lowoha_algo = matmul_algo_t::native_gemm;
        p.num_threads = nthreads;
        matmul_batch_params_t batch;

        const double ops = 2.0 * s.M * s.N * s.K;
        auto call = [&]() {
            return matmul_direct('r', false, /*transB=*/true, s.M, s.N, s.K,
                    1.0f, A.data(), s.K, B.data(), s.K, nullptr, 0.0f, C.data(),
                    s.N, /*is_weights_const=*/true, batch, p);
        };

        // First call separately: with a weight cache this is the one that packs.
        const auto t0 = clk::now();
        const status_t st = call();
        const double first_ms
                = std::chrono::duration<double>(clk::now() - t0).count() * 1e3;
        if (st != status_t::success) {
            std::printf("%-20s DECLINED (status=%d)\n", s.tag,
                    static_cast<int>(st));
            continue;
        }

        double best = 0.0, minv = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t1 = clk::now();
            call();
            const double dt
                    = std::chrono::duration<double>(clk::now() - t1).count();
            const double g = ops / dt / 1e9;
            if (g > best) best = g;
            if (g < minv) minv = g;
        }
        std::printf("%-20s %-6d %-6d %-6d %-3d %-2d %10.2f %10s %10s\n", s.tag,
                s.M, s.N, s.K, s.gs, nthreads, first_ms,
                gops(ops, ops / best / 1e9).c_str(),
                gops(ops, ops / minv / 1e9).c_str());
        std::fflush(stdout);
    }
    return 0;
}
