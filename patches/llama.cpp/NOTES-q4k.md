# Why Q4_K does not yet reach this fork's k-quant kernels

Attempted 2026-08-15 and **abandoned for now**; recorded so the next attempt
starts from the failure rather than repeating it.

The obvious patch is small: add `GGML_TYPE_Q4_K` to `supports_op`, a
`block_q4_K` entry to `ggml_to_zendnn_type`, a dispatch case, and
`packing.ggml_type_b = 12` in the params. That compiles, and `llama-bench` even
reports plausible numbers (147.46 / 15.35 against 145.72 / 15.14 without) --
which is exactly why it is worth saying that **it crashes**:

```
llama-perplexity ... -> GGML_ABORT in ggml_zendnn_compute_forward_mul_mat
    #6 ggml_zendnn_compute_forward_mul_mat
    #7 ggml_backend_zendnn_graph_compute
```

Perplexity is 1.0041 without the patch and a core dump with it. llama-bench
never validates output, so it happily benchmarked a path that aborts on a
different shape.

## The actual obstacle

`ggml_zendnn_compute_forward_mul_mat` special-cases Q8_0 weights so the
activations stay FP32:

```c
if (src1->type != vec_dot_type && src0->type != GGML_TYPE_Q8_0) { /* convert */ }
const void * wdata = (src1->type == vec_dot_type || src0->type == GGML_TYPE_Q8_0)
                   ? src1->data : work_data;
... src0->type == GGML_TYPE_Q8_0 ? GGML_TYPE_F32 : vec_dot_type ...
```

For a k-quant, `vec_dot_type` is `GGML_TYPE_Q8_K`, so the activations are
converted to q8_K blocks and handed to a type dispatch that only knows F32, BF16
and Q8_0. Hence the abort.

## What a real implementation needs

Consume q8_K directly rather than forcing FP32 activations. That is more
attractive than it sounds, because `block_q8_K` is:

```c
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
```

- `qs` is already the s8 activation this fork's k-quant kernel wants;
- `d` is the per-256 activation scale;
- **`bsums` is already the per-16 activation row sum**, and two adjacent entries
  make the per-32 sum the min-term correction needs -- the very quantity
  `int8_kquant_row_sums()` computes by hand.

So the integration is: unpack q8_K into (s8 activations, per-group scales, row
sums), pass those alongside the Q4_K weight, and skip both the library's dynamic
quantisation and its own row-sum pass. Roughly a day's work, mostly in the
backend rather than in this fork.

Until then Q4_K stays on ggml's CPU backend, which on these parts is
substantially faster than its own Q8_0 path anyway (145.72 / 15.14 against
108.79 / 1.57 on an FX-8370E).
