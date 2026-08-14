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

// Body of the 128-bit BF16 microkernel, included once per FMA flavour.
//
// Not a header: no include guard, and it is meant to be included inside a
// namespace that has already defined fmadd128() and is under the right
// "#pragma GCC target". Family 15h needs two builds of this code -- FMA3 for
// Piledriver onward, FMA4 for Bulldozer, which has no FMA3 -- and a machine
// with one is not guaranteed to have the other, so the choice cannot be a
// compile-time #if on the build's baseline. Including twice keeps one copy of
// the arithmetic.
//
// Expects from the enclosing scope:
//   fmadd128(a, b, acc)   the flavour's multiply-accumulate
//   pin_reg(v0, v1)       register pinning barrier
//   widen_even_k/odd_k    BF16 -> FP32 placement
//   load_b_pairs          unaligned 128-bit load

// MR=6 by kNR columns. A is FP32 and row-major with a_stride; B is BF16 in VNNI
// layout with b_stride counted in uint16 per k-pair row.
template <int kNR>
void bf16_ukernel_6xnr_128(const float *__restrict__ pa, int a_stride,
        const uint16_t *__restrict__ pb, int b_stride, float *__restrict__ C,
        int ldc, int k, float beta, const float *__restrict__ bias,
        fused_postop_t fused_op, uint16_t *__restrict__ C_bf16, int ldc_bf16) {

    constexpr int kMR = 6;
    static_assert(kNR % 8 == 0, "kNR must be a multiple of the 8-column group");

    const int k_pairs = k / 2;
    const bool k_odd = (k & 1) != 0;

    for (int n0 = 0; n0 < kNR; n0 += 8) {
        __m128 acc[kMR][2];
#pragma GCC unroll 6
        for (int m = 0; m < kMR; ++m) {
            acc[m][0] = _mm_setzero_ps();
            acc[m][1] = _mm_setzero_ps();
        }

        for (int kp = 0; kp < k_pairs; ++kp) {
            // Two uint16 per column in this layout, hence the doubled offset.
            const uint16_t *b_kp = pb + kp * b_stride + n0 * 2;

            {
                __m128 b0 = widen_even_k(load_b_pairs(b_kp));
                __m128 b1 = widen_even_k(load_b_pairs(b_kp + 8));
                pin_reg(b0, b1);
                const float *a_col = pa + 2 * kp;
#pragma GCC unroll 6
                for (int m = 0; m < kMR; ++m) {
                    const __m128 a = _mm_broadcast_ss(a_col + m * a_stride);
                    acc[m][0] = fmadd128(a, b0, acc[m][0]);
                    acc[m][1] = fmadd128(a, b1, acc[m][1]);
                }
            }
            {
                __m128 b0 = widen_odd_k(load_b_pairs(b_kp));
                __m128 b1 = widen_odd_k(load_b_pairs(b_kp + 8));
                pin_reg(b0, b1);
                const float *a_col = pa + 2 * kp + 1;
#pragma GCC unroll 6
                for (int m = 0; m < kMR; ++m) {
                    const __m128 a = _mm_broadcast_ss(a_col + m * a_stride);
                    acc[m][0] = fmadd128(a, b0, acc[m][0]);
                    acc[m][1] = fmadd128(a, b1, acc[m][1]);
                }
            }
        }

        // Odd k: the final k-pair row carries a valid even half only.
        if (k_odd) {
            const uint16_t *b_kp = pb + k_pairs * b_stride + n0 * 2;
            __m128 b0 = widen_even_k(load_b_pairs(b_kp));
            __m128 b1 = widen_even_k(load_b_pairs(b_kp + 8));
            pin_reg(b0, b1);
            const float *a_col = pa + 2 * k_pairs;
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                const __m128 a = _mm_broadcast_ss(a_col + m * a_stride);
                acc[m][0] = fmadd128(a, b0, acc[m][0]);
                acc[m][1] = fmadd128(a, b1, acc[m][1]);
            }
        }

        // beta scales the incoming C rather than acting as an accumulate flag:
        // the looper passes beta/alpha for arbitrary alpha, and K-blocking
        // passes 1 for every block after the first.
        if (beta == 0.0f) {
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                _mm_storeu_ps(C + m * ldc + n0, acc[m][0]);
                _mm_storeu_ps(C + m * ldc + n0 + 4, acc[m][1]);
            }
        } else {
            const __m128 bv = _mm_set1_ps(beta);
#pragma GCC unroll 6
            for (int m = 0; m < kMR; ++m) {
                float *c_row = C + m * ldc + n0;
                _mm_storeu_ps(
                        c_row, fmadd128(bv, _mm_loadu_ps(c_row), acc[m][0]));
                _mm_storeu_ps(c_row + 4,
                        fmadd128(bv, _mm_loadu_ps(c_row + 4), acc[m][1]));
            }
        }
    }

    apply_bias_and_postop_tile(C, ldc, kMR, kNR, bias, fused_op);

    // Round to nearest even on the way back to BF16, matching
    // _mm512_cvtneps_pbh so the two paths do not drift on the last mantissa
    // bit. Truncation would.
    if (C_bf16 != nullptr) {
        for (int m = 0; m < kMR; ++m) {
            const float *src = C + m * ldc;
            uint16_t *dst = C_bf16 + m * ldc_bf16;
            for (int n = 0; n < kNR; ++n) {
                uint32_t bits;
                std::memcpy(&bits, &src[n], sizeof(bits));
                const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
                dst[n] = static_cast<uint16_t>(rounded >> 16);
            }
        }
    }
}
