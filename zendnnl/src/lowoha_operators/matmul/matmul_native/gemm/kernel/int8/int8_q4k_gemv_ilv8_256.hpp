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

#ifndef MATMUL_NATIVE_INT8_Q4K_GEMV_ILV8_256_HPP
#define MATMUL_NATIVE_INT8_Q4K_GEMV_ILV8_256_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// Q4_K decode GEMV, EIGHT rows interleaved, 256-bit. The one experiment the
// phase-separated profile justifies.
//
// That profile (q4k_gemv_phase_profile.csv) put 15.5% of the GEMV in VPHADDD and
// VPHADDWD -- the largest single category, and about 7.4% of all decode -- and
// showed that only about a fifth of the kernel is the multiply. It also explained
// four dead hypotheses as one result: the reduction is a SERIAL dependency chain,
// so removing work that is not on it changed nothing, and the one change that did
// remove it (the four-row 128-bit interleave) paid for it with unpacking that
// landed on the new critical path -- 0.94x.
//
// So this attacks the same 15.5% with two things the 128-bit attempt lacked:
//
//   TWICE THE AMORTISATION. Eight rows per accumulator instead of four, so one
//   unpacked operand feeds twice the outputs, and a 32-byte load is still one
//   instruction -- 256-bit cracking into two 128-bit micro-ops affects the
//   ARITHMETIC, not the load or the shuffle.
//
//   A BETTER NIBBLE PLANE. The 128-bit version packed W[2j] and W[2j+1] into one
//   byte, which needed an unpacklo per operand to restore lane order. Here byte j
//   holds group A's weight j in its low nibble and group B's in its high nibble,
//   so AND and SRLI+AND yield two COMPLETE operands with no interleaving at all.
//   That also sidesteps AVX2's unpacklo being per-128-bit-lane, which would have
//   broken the ordering outright.
//
// The counter-argument, recorded because it may well win: 0.94x is still a loss,
// this has to find another ~6% before breaking even, and the reduction it deletes
// is worth only 7.4% of decode. If it lands near parity the answer is that ggml's
// kernel fits this problem and declining Q4_K on AVX2 hosts stays right.
//
// Requires AVX2, so Excavator and later within family 15h. Returns false when it
// will not serve the call; the caller must fall back rather than treat the
// destination as written.
bool int8_q4k_gemv_ilv8_256(int N, int K, int ggml_type, const int8_t *A,
        const void *blocks, float *C, const float *src_scale, int ss_grp,
        int nthreads);

// Q4_K only, N a multiple of 8, K whole super-blocks, and an AVX2 host.
bool int8_q4k_gemv_ilv8_supported(int M, int N, int K, int ggml_type);

void int8_q4k_gemv_ilv8_clear_cache();

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_INT8_Q4K_GEMV_ILV8_256_HPP
