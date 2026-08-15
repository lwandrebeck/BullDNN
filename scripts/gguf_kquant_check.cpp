// Validate the Q4_K / Q5_K decoder against files produced by llama.cpp itself.
//
// Every other k-quant test in this tree builds its own blocks, and its packer was
// written from the same format description as the decoder -- so the two agree by
// construction and a shared misreading would survive both. This reads real GGUF
// files and has no packer at all.
//
// Ground truth is the Q8_0 model the k-quants were requantised FROM. The same
// tensor in both files holds the same weights at different precision, so
// dequantising each and comparing tells the difference between "my decode is
// right, and Q4_K is lossy" (a few percent, correlated) and "my decode is wrong"
// (uncorrelated noise). A nibble-order or scale-packing error cannot produce
// values that track the original.
//
//   gguf_kquant_check <kquant.gguf> <q8_0.gguf>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "lowoha_operators/matmul/ggml_weight_unpack.hpp"
#include "lowoha_operators/matmul/lowoha_common.hpp"

using namespace zendnnl;
using namespace zendnnl::lowoha::matmul;

namespace {

// ---- minimal GGUF reader ---------------------------------------------------
// Enough of the format to find a tensor's type, shape and bytes: header, the
// key/value block (skipped, but it must be walked to reach the tensor table),
// the tensor table, then the aligned data region.

struct Reader {
    std::vector<uint8_t> buf;
    size_t pos = 0;

    bool load(const char *path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        const std::streamsize n = f.tellg();
        f.seekg(0);
        buf.resize(static_cast<size_t>(n));
        return static_cast<bool>(f.read(reinterpret_cast<char *>(buf.data()), n));
    }
    template <typename T> T get() {
        T v {};
        std::memcpy(&v, buf.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t len = get<uint64_t>();
        std::string s(reinterpret_cast<const char *>(buf.data() + pos), len);
        pos += len;
        return s;
    }
    void skip_value(uint32_t type);
};

// GGUF value types, in the order the format defines them.
void Reader::skip_value(uint32_t type) {
    switch (type) {
        case 0: case 1: pos += 1; break;              // int8, uint8
        case 2: case 3: pos += 2; break;              // int16, uint16
        case 4: case 5: case 6: pos += 4; break;      // int32, uint32, float32
        case 7: pos += 1; break;                      // bool
        case 8: str(); break;                         // string
        case 9: {                                     // array
            const uint32_t sub = get<uint32_t>();
            const uint64_t n = get<uint64_t>();
            for (uint64_t i = 0; i < n; ++i) skip_value(sub);
            break;
        }
        case 10: case 11: case 12: pos += 8; break;   // uint64, int64, float64
        default: std::fprintf(stderr, "unknown kv type %u\n", type); pos = buf.size();
    }
}

struct TensorInfo {
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct Gguf {
    Reader r;
    std::map<std::string, TensorInfo> tensors;
    size_t data_start = 0;

    bool open(const char *path) {
        if (!r.load(path)) return false;
        if (std::memcmp(r.buf.data(), "GGUF", 4) != 0) return false;
        r.pos = 4;
        r.get<uint32_t>(); // version
        const uint64_t n_tensors = r.get<uint64_t>();
        const uint64_t n_kv = r.get<uint64_t>();

        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = r.str();
            const uint32_t type = r.get<uint32_t>();
            if (key == "general.alignment" && type == 5) {
                alignment = r.get<uint32_t>();
            } else {
                r.skip_value(type);
            }
        }
        for (uint64_t i = 0; i < n_tensors; ++i) {
            const std::string name = r.str();
            TensorInfo ti;
            const uint32_t nd = r.get<uint32_t>();
            for (uint32_t d = 0; d < nd; ++d) ti.dims.push_back(r.get<uint64_t>());
            ti.type = r.get<uint32_t>();
            ti.offset = r.get<uint64_t>();
            tensors[name] = ti;
        }
        data_start = (r.pos + alignment - 1) / alignment * alignment;
        return true;
    }
    const uint8_t *tensor_data(const TensorInfo &t) const {
        return r.buf.data() + data_start + t.offset;
    }
};

// ---- Q8_0 dequantisation, the reference side -------------------------------
struct block_q8_0_ref {
    uint16_t d;
    int8_t qs[32];
};

float fp16(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;
        else {
            int e = -1; uint32_t m = man;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            bits = (sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000 | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void dequant_q8_0(const uint8_t *data, size_t n_elems, std::vector<float> &out) {
    out.resize(n_elems);
    const size_t nb = n_elems / 32;
    for (size_t b = 0; b < nb; ++b) {
        block_q8_0_ref blk;
        std::memcpy(&blk, data + b * sizeof(block_q8_0_ref), sizeof(blk));
        const float d = fp16(blk.d);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * blk.qs[j];
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <kquant.gguf> <q8_0.gguf>\n", argv[0]);
        return 2;
    }
    Gguf kq, q8;
    if (!kq.open(argv[1]) || !q8.open(argv[2])) {
        std::fprintf(stderr, "failed to open inputs\n");
        return 2;
    }

    // GGML type ids: 12 = Q4_K, 13 = Q5_K, 8 = Q8_0.
    int checked = 0, bad = 0;
    for (const auto &kv : kq.tensors) {
        const std::string &name = kv.first;
        const TensorInfo &t = kv.second;
        if (t.type != 12 && t.type != 13) continue;
        auto it = q8.tensors.find(name);
        if (it == q8.tensors.end() || it->second.type != 8) continue;

        // GGUF dims are [ne0, ne1, ...] with ne0 the fastest axis, which for a
        // weight is K; rows are ne1. That is exactly the N x K (transB) layout
        // the unpack expects.
        const int K = static_cast<int>(t.dims[0]);
        const int N = t.dims.size() > 1 ? static_cast<int>(t.dims[1]) : 1;
        if (K % 256 != 0) continue;

        const void *weight = kq.tensor_data(t);
        matmul_params p;
        p.dtypes.src = data_type_t::s8;
        p.dtypes.wei = (t.type == 12) ? data_type_t::s4 : data_type_t::s8;
        p.dtypes.dst = data_type_t::f32;
        p.packing.pack_format_b = 1;
        p.packing.ggml_type_b = static_cast<int>(t.type);
        const int groups = K / 32;
        std::vector<float> dummy_ss(static_cast<size_t>(4) * groups, 0.01f);
        p.quant_params.src_scale.buff = dummy_ss.data();
        p.quant_params.src_scale.dt = data_type_t::f32;
        p.quant_params.src_scale.dims = {4, groups};

        if (unpack_ggml_weights_and_cache(weight, N, K, K, 't', p)
                != status_t::success) {
            std::printf("%-40s UNPACK FAILED\n", name.c_str());
            ++bad;
            continue;
        }

        const uint8_t *codes = static_cast<const uint8_t *>(weight);
        const float *D = static_cast<const float *>(p.quant_params.wei_scale.buff);
        const float *M = D + static_cast<size_t>(groups) * N + 16;

        std::vector<float> ref;
        dequant_q8_0(q8.tensor_data(it->second),
                static_cast<size_t>(N) * K, ref);

        // Compare dequantised values. Q4_K is a lossy requantisation of these
        // very numbers, so they must track closely; a decode error cannot.
        double num = 0.0, den = 0.0, worst = 0.0;
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                const int g = k / 32;
                const double w
                        = static_cast<double>(D[static_cast<size_t>(g) * N + n])
                                * codes[static_cast<size_t>(n) * K + k]
                        - static_cast<double>(M[static_cast<size_t>(g) * N + n]);
                const double r = ref[static_cast<size_t>(n) * K + k];
                num += (w - r) * (w - r);
                den += r * r;
                worst = std::max(worst, std::fabs(w - r));
            }
        }
        const double rel = std::sqrt(num / (den > 0 ? den : 1));
        const char *verdict = rel < 0.15 ? "ok" : "MISMATCH";
        if (rel >= 0.15) ++bad;
        ++checked;
        std::printf("%-40s type=%-3u N=%-5d K=%-5d  rel_rms=%.4f  worst=%.4g  %s\n",
                name.c_str(), t.type, N, K, rel, worst, verdict);
    }

    std::printf("\n%d tensor(s) checked, %d bad\n", checked, bad);
    if (checked == 0) {
        std::printf("no Q4_K/Q5_K tensor was comparable -- nothing was proven\n");
        return 2;
    }
    return bad ? 1 : 0;
}
