# Getting Q4_K/Q5_K to this fork's k-quant kernels

Done, in `0002-ggml-zendnn-k-quant-support-and-dispatch-stats.patch`. Four things
had to line up; the first attempt got one of them and crashed, which is why they
are all written down.

## 1. Activations must stay FP32

`ggml_zendnn_compute_forward_mul_mat` converts activations to `vec_dot_type`
unless the weight is Q8_0, in which case it hands over raw FP32 and lets the
library quantise. For a k-quant `vec_dot_type` is **Q8_K**, which this backend's
type dispatch cannot consume -- so converting produces `GGML_ABORT`, not a
fallback. Three places test `src0->type == GGML_TYPE_Q8_0` (the conversion guard,
the `wdata` choice, and the B-type argument) and all three must include the
k-quants. Patching only the type dispatch, as I did first, aborts.

## 2. `packing.ggml_type_b` must be set

Q8_0 gets away without it because the library infers the type from the s8 weight
dtype. A k-quant is asymmetric and has to be named: 12 for Q4_K, 13 for Q5_K.

## 3. `quant_params.src_scale.dims` must be set

The dynamic-quant reorder refuses without it -- *"Reorder quantization requires
quant_params.src_scale dims and dt to be set"*. It is set per call site rather
than in the shared params helper, because it depends on the batch size, so the
`if constexpr` chains at both call sites (single and grouped) need the k-quant
types too. The group is 32, the same as `QK8_0`, so the shape is identical.

## 4. Dispatch had to stop choosing AOCL-DLP (fixed in this fork, not the patch)

With the above, the call reached `matmul_direct` and still failed: *"Selected
kernel aocl_dlp_blocked requires AOCL-DLP"*. The backend never names an
algorithm, so `kernel_select` used the default, which is `aocl_dlp_blocked` --
a backend that cannot run per-group INT8 without AVX-512 VNNI and returns without
computing. `lowoha_matmul_utils.cpp` now routes such shapes to `native_gemm` on
a host without VNNI, since that is the only kernel that can run them. Any caller
that does not name an algorithm was affected, not just llama.cpp.

## Verification

`llama-perplexity` on the same text: **1.0041 without, 1.0040 with** -- equal to
the precision the measurement has. The patch also adds `ZENDNN_DISPATCH_STATS=1`,
which prints accepted/rejected/executed counts per weight type at exit, so
"this fork made no difference" can be told apart from "this fork never ran":

```
weight type      accepted     rejected     executed
q4_K                  840            0          168
```

That counter exists because a Q4_K patch that *aborted* still produced plausible
`llama-bench` numbers -- llama-bench never validates output, and a backend that
is never called looks identical to one that is merely not faster.

## Result

qwen2.5-coder-1.5B Q4_K_M, FX-8370E, 8 threads, 3 reps:

| | pp128 | tg32 |
|---|---|---|
| ggml CPU | 146.92 | 15.19 |
| + this fork | 147.90 | **16.01** |

+5.4% on token generation, +0.7% on prompt. Modest next to the Q8_0 figures, and
the reason is that ggml's Q4_K kernels are good even without AVX2 -- it is the
most-used quantisation and is tuned accordingly -- whereas its Q8_0 path is not.
