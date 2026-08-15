// Which way of carrying the k-quant min term is faster.
//
// Both candidates reach the same microkernel through the same two pointers, so
// the only thing that can differ is where D and M sit in memory:
//
//   joint  D and M in ONE allocation, M at offset groups*N -- the wei_scale
//          {2G, N} encoding
//   split  D and M in TWO allocations -- the separate wei_zp encoding
//
// A third arm is included because if layout matters at all it is the one that
// should win, and it costs nothing to ask:
//
//   interleaved  D and M adjacent per (group, column) -- one cache line serves
//                both, but it is not expressible in either candidate encoding
//                without an unpack that writes pairs
//
// Arms are interleaved within a pass and every pass is repeated, because these
// boxes drift; best-of is reported alongside the worst rep so a wandering row
// cannot pass as a result.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_kquant_ukernel_128.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/int8_kquant_looper_128.hpp"

using namespace zendnnl::lowoha::matmul::native;
using clk = std::chrono::steady_clock;

namespace {

constexpr double kCeiling = 400.0;

std::string gops(double ops, double secs) {
    if (secs <= 0.0) return "IMPOSSIBLE";
    const double g = ops / secs / 1e9;
    char buf[64];
    if (g > kCeiling)
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
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 4;
    const int passes = argc > 3 ? std::atoi(argv[3]) : 3;

    const Shape shapes[] = {
            {1, 4096, 4096, 32, "decode qkv"},
            {1, 11008, 4096, 32, "decode ffn"},
            {128, 4096, 4096, 32, "prompt-128"},
            {512, 4096, 4096, 32, "prompt-512"},
            {1024, 1024, 1024, 32, "square 1024"},
    };

    std::mt19937 rng(20260815);
    std::uniform_int_distribution<int> dcode(0, 15);
    std::uniform_real_distribution<float> dsc(0.002f, 0.05f);

    std::printf("# threads=%d reps=%d passes=%d\n", threads, reps, passes);
    std::printf("%-14s %-12s %10s %10s\n", "shape", "layout", "best GOPS",
            "min GOPS");

    for (const Shape &s : shapes) {
        const int groups = s.K / s.gs;
        const size_t nsc = static_cast<size_t>(groups) * s.N;

        std::vector<int8_t> A(static_cast<size_t>(s.M) * s.K);
        std::vector<uint8_t> q(static_cast<size_t>(s.N) * s.K);
        std::vector<float> C(static_cast<size_t>(s.M) * s.N, 0.0f);
        for (auto &x : A) x = static_cast<int8_t>(dcode(rng) - 8);
        for (auto &x : q) x = static_cast<uint8_t>(dcode(rng));

        // joint: one allocation, D then M
        std::vector<float> joint(2 * nsc);
        for (auto &x : joint) x = dsc(rng);
        // split: two allocations with the same values
        std::vector<float> D(joint.begin(), joint.begin() + nsc);
        std::vector<float> Mn(joint.begin() + nsc, joint.end());
        // interleaved: D and M adjacent per element, handed to the kernel as two
        // pointers one float apart with a doubled stride -- which the kernel's
        // ws_stride cannot express, so this arm runs the looper on a doubled-N
        // layout only to time the access pattern, not to produce a usable result.
        std::vector<float> inter(2 * nsc);
        for (size_t i = 0; i < nsc; ++i) {
            inter[2 * i] = D[i];
            inter[2 * i + 1] = Mn[i];
        }

        const float ss = 0.0137f;
        const double ops = 2.0 * s.M * s.N * s.K;

        // joint_pad: same single allocation, but M displaced by a cache line so
        // the two streams stop landing on the same sets. groups*N*4 is a power of
        // two for every shape here -- 2 MB at N=4096, 128 KB at 1024^3 -- which
        // is the classic way to make two streams alias each other.
        std::vector<float> joint_pad(2 * nsc + 16);
        std::memcpy(joint_pad.data(), D.data(), nsc * sizeof(float));
        std::memcpy(joint_pad.data() + nsc + 16, Mn.data(), nsc * sizeof(float));

        struct Arm {
            const char *name;
            const float *d;
            const float *m;
        };
        const Arm arms[] = {
                {"joint", joint.data(), joint.data() + nsc},
                {"split", D.data(), Mn.data()},
                {"joint_pad", joint_pad.data(), joint_pad.data() + nsc + 16},
        };
        constexpr int kArms = 3;

        double best[kArms] = {0.0, 0.0, 0.0};
        double worst[kArms] = {1e30, 1e30, 1e30};
        for (int p = 0; p < passes; ++p) {
            for (int a = 0; a < kArms; ++a) {
                // warm
                int8_kquant_execute_128(s.M, s.N, s.K, s.gs, A.data(), s.K,
                        q.data(), s.K, true, C.data(), s.N, arms[a].d,
                        arms[a].m, &ss, 0, 0, threads);
                for (int r = 0; r < reps; ++r) {
                    const auto t0 = clk::now();
                    int8_kquant_execute_128(s.M, s.N, s.K, s.gs, A.data(), s.K,
                            q.data(), s.K, true, C.data(), s.N, arms[a].d,
                            arms[a].m, &ss, 0, 0, threads);
                    const double dt
                            = std::chrono::duration<double>(clk::now() - t0)
                                      .count();
                    const double g = ops / dt / 1e9;
                    if (g > best[a]) best[a] = g;
                    if (g < worst[a]) worst[a] = g;
                }
            }
        }
        for (int a = 0; a < kArms; ++a) {
            std::printf("%-14s %-12s %10s %10s\n", s.tag, arms[a].name,
                    gops(ops, ops / best[a] / 1e9).c_str(),
                    gops(ops, ops / worst[a] / 1e9).c_str());
        }
        std::printf("%-14s %-12s joint %+.1f%%   joint_pad %+.1f%%  (vs split)\n",
                s.tag, "", 100.0 * (best[0] / best[1] - 1.0),
                100.0 * (best[2] / best[1] - 1.0));
        std::fflush(stdout);
    }
    return 0;
}
