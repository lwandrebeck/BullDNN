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

#ifndef MATMUL_NATIVE_INT8_Q6K_GEMV_PRE_128_HPP
#define MATMUL_NATIVE_INT8_Q6K_GEMV_PRE_128_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Q6_K decode GEMV over PRE-STITCHED codes.
//
// This one is not a guess. perf put 27.75% of a decode profile in run_q6 -- for
// 14.7% of the dispatched ops, so about 4.5x its share -- and perf annotate put
// that time in the bit work that rebuilds each six-bit code from the ql and qh
// planes: vpsrlw, vpand, vpsllw, vpor, vpshlb. Four hypotheses about the Q4_K
// kernel died before this, every one of them derived by reading rather than
// measuring, so the ordering matters: this is the first candidate a profiler
// pointed AT.
//
// A code is a pure function of the weight, so the stitching can happen once per
// weight instead of once per token. The cached form stores codes in natural k
// order, which is exactly the order the activations are read in, so the inner
// loop becomes four loads and four PMADDUBSW with no bit work at all.
//
// The cost is bytes: 256 code bytes plus 16 scales plus a 4-byte d against the
// block's 210, about 31% more of the Q6_K weight stream. Decode here runs at
// ~6 GB/s against a controller good for far more, and it is issue-bound rather
// than bandwidth-bound, which is the whole reason this trade is worth making.
//
// Returns false when it will not serve the call; the caller must then fall back
// rather than treat the destination as written.
bool int8_q6k_gemv_pre_128(int N, int K, const int8_t *A, const void *blocks,
        float *C, const float *src_scale, int ss_grp, const int32_t *rowsum16,
        int nthreads);

bool int8_q6k_gemv_pre_supported(int N, int K);

// Drops every cached stitched weight. For tests that reuse an address.
void int8_q6k_gemv_pre_clear_cache();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_Q6K_GEMV_PRE_128_HPP
