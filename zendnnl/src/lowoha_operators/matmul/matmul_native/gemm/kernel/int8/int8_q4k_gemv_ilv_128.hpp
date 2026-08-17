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

#ifndef MATMUL_NATIVE_INT8_Q4K_GEMV_ILV_128_HPP
#define MATMUL_NATIVE_INT8_Q4K_GEMV_ILV_128_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Q4_K decode GEMV over a FOUR-ROW INTERLEAVED weight layout.
//
// The row-at-a-time kernel next door (int8_q4k_gemv_128) reaches the same
// answer and measured 0.75x of ggml's repacked kernels on Excavator decode.
// Reading both, the arithmetic is a wash -- per output row per super-block each
// issues the equivalent of sixteen 128-bit PMADDUBSW, since ggml's 256-bit ops
// crack into two 128-bit uops on this family. What ggml does not pay for is the
// reduction: it interleaves eight rows so the lanes of its accumulator ARE eight
// outputs, and never reduces horizontally at all. The row-at-a-time kernel
// collapses a vector to a scalar for every output, six VPHADDD per super-block
// plus an hsum_ps at the end, and VPHADDD is multi-uop and slow here.
//
// Two other candidates were measured and ruled out first, so this is the one
// that is left rather than the one that looked best: deferring the s16 widening
// buys nothing because Q4K_ACCUM is already the XOP fused VPMADCSWD, and
// ablating the scalar six-bit scale unpacking outright did not move decode.
// See docs/perf/q4k_gemv_vs_ggml_structure.md.
//
// FOUR rows, not ggml's eight, because PMADDUBSW at 128 bits yields eight s16
// lanes and PMADDWD folds adjacent pairs, so four s32 lanes come out -- four
// outputs. Eight would need 256-bit, which cracks anyway.
//
// Returns false when it will not serve the call, in which case the caller must
// fall back rather than treat the destination as written.
bool int8_q4k_gemv_ilv_128(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads);

// Q4_K only, N a multiple of 4, K whole super-blocks. Q5_K and Q6_K stay with
// the row-at-a-time kernel: the fifth-bit plane and Q6_K's 16-wide sub-blocks
// both change the packing, and neither is worth writing before this layout is
// shown to pay on the type that was measured.
bool int8_q4k_gemv_ilv_supported(int M, int N, int K, int ggml_type);

// Drops every cached repacked weight. For tests that reuse an address.
void int8_q4k_gemv_ilv_clear_cache();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_Q4K_GEMV_ILV_128_HPP
