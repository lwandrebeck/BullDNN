# Upstream issue drafts (amd/ZenDNN)

Fourteen defects found while porting ZenDNN to AMD family 15h (Bulldozer through
Excavator). Most are not family-15h-specific: they affect any host without
AVX-512 (Zen 1/2/3 included) or any build configured `--no-aocldlp`, and four
affect every host regardless of ISA -- issue 6 was verified failing identically
on an Intel Xeon 6767P with full AVX-512.

Each has a branch off `main` carrying a minimal fix. **All fourteen branches were
compiled and linked against `main` at `70f4dbf`** on an AMD PRO A10-8770E
(Excavator, 4 cores / 2 modules, no AVX-512) configured `--no-aocldlp`, which is
also where every reproduction below comes from.

**File issue 1 first — it gates the other thirteen.** `main` cannot link at all with
`--no-aocldlp`, so nobody can reproduce or test issues 2-11 in that
configuration until the W4A8 stubs exist. Issues 2-11 were reproduced on family 15h; issues 6 and 12 were additionally reproduced on an Intel Xeon 6767P with full AVX-512, so they are not ISA-specific at all. Each branch below was therefore built
on top of issue 1's fix.

| # | Issue | Severity | Affects | Branch |
| --- | --- | --- | --- | --- |
| 1 | `--no-aocldlp` fails to link | build blocker | any `--no-aocldlp` build | `upstream/no-aocldlp-link-w4a8-stubs` |
| 2 | Native matmul reinterprets BF16/INT8 as FP32 | heap corruption | INT8 case: **all hosts** | `upstream/native-dtype-reinterpretation` |
| 3 | AVX-512-only epilogue helpers called unguarded | SIGILL | no-AVX-512 hosts | `upstream/native-epilogue-avx512-unguarded` |
| 4 | Uncaught `dnnl::error` aborts the process | SIGABRT | no-AVX-512 hosts, BF16 | `upstream/onednn-uncaught-error-abort` |
| 5 | `scalar_microkernel` treats beta as a flag | wrong results | no-AVX-512 hosts | `upstream/native-beta-as-flag` |
| 6 | Flash SDPA returns all zeros | wrong results | **any `--no-aocldlp` build, any CPU** | `upstream/sdpa-zero-output-no-avx512` |
| 7 | LOWOHA embedding-bag dispatches AVX-512/AVX2 unconditionally | SIGILL | no-AVX-512 / no-AVX2 | `upstream/embag-unconditional-avx512` |
| 8 | Dynamic-quant reorder dispatches AVX-512 unconditionally | SIGILL | no-AVX-512 hosts | `upstream/dynamic-quant-avx512-unguarded` |
| 9 | auto_tuner offers AOCL-DLP candidates it cannot run | default path broken | any `--no-aocldlp` build | `upstream/autotuner-aocl-candidates` |
| 10 | benchdnn matmul ignores `--kernel_name` | harness defect | **all hosts** | `upstream/benchdnn-kernel-name-ignored` |
| 11 | Cost model fabricates an L3 that does not exist | mis-tuning | CPUs without L3 | `upstream/cost-model-phantom-l3` |
| 12 | Post-op chain silently reordered | wrong results | **all hosts, default config** | `upstream/postop-chain-reorder` |
| 13 | libxsmm BF16→FP32 gated off despite being implemented | no backend | `--no-aocldlp` builds | `upstream/libxsmm-bf16-f32-gate` |
| 14 | libxsmm strided operands sent to DLP | no backend | `--no-aocldlp` builds | `upstream/libxsmm-strided-partitioner` |

---

## 1. Build with `--no-aocldlp` fails to link: missing W4A8 stubs

`aocl_kernel_stub.cpp` provides no-op stubs for the AOCL-DLP entry points so the
library links when built without AOCL-DLP, but three W4A8 entry points were
added without corresponding stubs, so the link fails.

Missing: `cvt_s4_to_s8`, `w4a8_populate_plain_s8_cache`,
`broadcast_w4a8_src_scale`.

**Reproduce:** configure with `--no-aocldlp`, build. Verified on `main` at
`70f4dbf`: the link fails with exactly these three undefined references, and
succeeds with the branch applied.

**Fix:** add the three stubs alongside the existing ones — two log and throw via
`EXCEPTION_WITH_LOC`, the third returns `status_t::unimplemented`.

---

## 2. Native matmul hands BF16 and INT8 problems to the FP32 kernel

`native_matmul_execute()` accepts problems it has no kernel for and passes them
to the FP32 looper, which casts `src`/`weight` straight to `const float *`
without inspecting any dtype. Two paths:

- **BF16 without AVX512-BF16.** `bf16_gemm_execute` / `bf16_brgemm_execute` are
  gated on `uarch.avx512bf16`; the `else` branches fall through to the FP32
  GEMM/BRGEMM. 2-byte elements are read and written as 4-byte floats.
- **INT8 under algo 10 (`native_gemm`).** The `avx512vnni` guard exists only
  inside the `native_brgemm` branch, while the LOWOHA dispatcher accepts INT8
  for both native algos. Native GEMM has no INT8 kernel at all, so 1-byte values
  are read as 4-byte floats. **This one is not ISA-dependent** — it behaves the
  same on Zen 4/5.

Both write past the end of `dst` and return `true`, so the caller reports success
on a corrupted heap.

**Reproduce (BF16):** `benchdnn --op=matmul --lowoha=true --m=512 --k=512
--n=512 --sdt=bf16 --wdt=bf16 --ddt=bf16 --kernel_name=native_gemm` on a host
without AVX512-BF16. Prints a plausible timing, then SIGSEGV in `free()` during
teardown. **INT8:** same with `--sdt=s8 --wdt=s8 --ddt=f32` exits 0 with wrong
results.

**Fix:** decline both, as the INT8 BRGEMM path already does when AVX512-VNNI is
absent, so another backend runs the problem.

---

## 3. AVX-512-only epilogue helpers are called without an ISA guard (SIGILL)

`scale_tile()` is declared `target("avx512f")` and `apply_postops_tile()`
`target("avx512f,avx512bw,fma")`. Both are called unconditionally from the native
FP32 looper — eight call sites. On a host without AVX-512 the process dies.

Reached by exactly two things: a non-unit alpha, and any post-op the microkernel
epilogue cannot fuse. The looper's own `beta/alpha` rescaling makes the first
routine.

**Reproduce:** force the native path with a non-unit alpha, e.g.
`ZENDNNL_MATMUL_ALGO=10 ./gtests --gtest_filter=*TestMatmul.F32_F32*`:

```
Thread 2 "gtests" received signal SIGILL, Illegal instruction.
#0  native::scale_tile(float*, int, int, int, float)
=> vbroadcastss %xmm2,%zmm1
```

benchdnn does not catch it because it defaults to alpha=1 and fuses a lone relu
into the microkernel, so neither helper is reached.

**Fix:** give both a portable implementation and dispatch on `detect_uarch()`.
For the post-ops this is nearly free — every one of the 16 cases already carries
a scalar tail loop for the sub-16-element remainder, so the portable path is
those tails and agrees with the vector path by construction.

---

## 4. Uncaught `dnnl::error` takes the process down

Every oneDNN entry point can throw `dnnl::error`, and primitive-descriptor
creation does exactly that when the host ISA cannot serve the requested dtype —
BF16 without AVX-512 or AVX2-VNNI-2, for instance. Nothing catches it, so the
exception reaches `std::terminate`.

**Reproduce:** `benchdnn --op=matmul --lowoha=true --m=512 --k=512 --n=512
--sdt=bf16 --wdt=bf16 --ddt=bf16 --kernel_name=onednn` on a host without
AVX-512: SIGABRT, exit 134,
`terminate called after throwing an instance of 'dnnl::error'`.

**Fix:** catch inside `matmul_onednn_wrapper()` rather than at its call sites —
three of the five callers (`bmm_kernel`, `bmm_looper`, `matmul_partitioner`)
invoke it from inside an OpenMP parallel region, where an escaping exception may
not cross the region boundary. On failure, mark the AOCL-DLP fallback, the
convention the dispatch already uses for "did not compute", and let the call
sites fall through instead of returning success on an untouched `C`.

---

## 5. `scalar_microkernel` treats beta as a flag rather than a scale

The AVX-512 microkernels compute `acc += beta * C_old` with fmadd.
`scalar_microkernel()` zeroes C when beta is 0 and otherwise adds products
straight into C — `C_old + A*B`, weight 1. Correct only for beta ∈ {0, 1}.

The FP32 looper reaches other values by design: when alpha != 1 it defers scaling
and passes `beta/alpha` to the microkernel, an arbitrary ratio. So any host
without AVX-512 gets wrong results whenever alpha != 1 and beta != 0.

**Reproduce:** on a host without AVX-512, force the native FP32 path with random
alpha/beta (the gtests already do) and compare against the reference kernel.
Requires issue 3 to be fixed first, which otherwise faults before reaching this.

**Fix:** scale C by beta, matching the AVX-512 kernels.

---

## 6. Flash SDPA silently returns zeros on any build without AOCL-DLP

`zendnn_gemm()` in `lowoha_sdpa_flash_cpu.cpp` hardcodes the backend for the two
GEMMs the flash kernel delegates to:

```cpp
params.lowoha_algo = zendnnl::ops::matmul_algo_t::aocl_dlp;   // line 147
...
matmul_direct('r', TransA, TransB, m, n, k, alpha, a, lda, b, ldb,
              nullptr, beta, c, ldc, false, batch_params, params);  // line 153
```

`zendnn_gemm` is declared `void`, so `matmul_direct`'s `status_t` is discarded.
In a build without AOCL-DLP that call is rejected before dispatch and computes
nothing; the kernel carries on and emits its zero-initialised accumulator as the
attention output. Both GEMMs are affected -- QK^T (line 713) and PV (line 783).

Call path, from gdb:

```
#0 matmul::matmul_direct(...)
#1 sdpa::zendnn_gemm<float>(...)
#2 sdpa::cpu_flash_attention_sa<simd::avx2_tag, float, ...>(...)
#3 sdpa::sdpa_flash_run_avx2_internal(...)
```

**This is not ISA-dependent.** Reproduced on two opposite microarchitectures with
pristine `main` (plus issue 1's link stubs), both `--no-aocldlp`:

| Host | ISA | `requires AOCL-DLP` errors | SDPA suite |
| --- | --- | --- | --- |
| AMD PRO A10-8770E | Excavator, no AVX-512 | 62,772 | fails |
| Intel Xeon 6767P | avx512f + avx512_bf16 + avx512_vnni | 62,402 | fails |

A CPU with the fullest AVX-512 feature set available fails identically to one
with none, which rules out SIMD support as the cause.

**Reproduce:** build `--no-aocldlp`, run `./gtests --gtest_filter=*Sdpa*`. The log
fills with "Selected kernel aocl_dlp requires AOCL-DLP, but ZenDNNL was built
without AOCL-DLP support", emitted from the flash path, and the suite fails.

**Fix:** stop pinning the backend, and stop discarding the status. The minimal
correct change is to propagate the failure so the caller can fall back to the
reference kernel instead of returning zeros -- returning wrong numerics silently
is the actual defect. The branch currently carries only a stopgap that routes
flash to the reference kernel when AVX-512 is absent; that stopgap is now known
to be aimed at the wrong condition and should be replaced by the above.

**Fix verified.** With a backend chain in place of the hardcode -- AOCL-DLP
first, so a build that has it is unchanged, then oneDNN, libxsmm, native_gemm,
and an error log if none compute -- the SDPA suite on the Xeon 6767P goes from
6 passing to **751 passing**, run exit 0, with AOCL-DLP rejections falling from
62,402 to 120 (absorbed by the chain) and zero "no backend" errors. The
hardcoded backend was the entire defect; there is no second kernel-side bug.

## 7. LOWOHA embedding-bag dispatches AVX-512 and AVX2 kernels unconditionally

Two separate unguarded dispatches:

- `dispatch_kernel.hpp`'s `dispatch_avx512_kernel` runs AVX-512 kernels after the
  FBGEMM attempt with no `avx512f` check.
- `embag_operator_impl.cpp` selects the `_avx2` kernels without checking for
  AVX2 or FMA.

Both fault on hosts lacking the ISA.

**Fix:** check `get_avx512f_status()` / `get_avx2_status() && get_fma_status()`
and fall back to the existing reference kernel. Verified with 16,770 embag gtests
passing on an A10-8770E after the change.

---

## 8. Dynamic-quant reorder dispatches AVX-512 kernels unconditionally

The dynamic-quant reorder path calls AVX-512 kernels with no ISA guard (five
sites in `dynamic_dispatch.cpp`, plus the fast-path block in
`lowoha_reorder_utils.cpp`).

**Reproduce:** `./gtests --gtest_filter=*DynamicQuant*` on a host without
AVX-512 dies immediately (exit 132):

```
Thread 4 "gtests" received signal SIGILL, Illegal instruction.
#0  reorder::dynamic_per_token_quant_bf16_s8_native(unsigned short const*, signed char*, float*, long, long)
=> vpternlogd $0xff,%zmm4,%zmm4,%zmm4
```

Note the *static* quant/dequant path in the same module is correctly guarded --
`select_reorder_algo()` refuses `reorder_algo_t::native` unless avx512f and
avx512bw/vl are both present. Only the dynamic path is missing that check.

**Fix:** gate on `get_avx512f_status()` and select the reference kernels
otherwise. Note the guard must sit on the fast-path condition, not the whole
entry point — gating the entry disables a working portable scalar path and makes
`reorder_direct` return `isa_unsupported` for every dynamic-quant reorder.

---

## 9. auto_tuner's default candidates cannot run in a `--no-aocldlp` build

`get_algo_candidates()` defaults to `{aocl_dlp_blocked, onednn_blocked}`
regardless of build configuration. Since auto_tuner is the default matmul path,
the default path computes nothing in a `--no-aocldlp` build: the tuner picks
`aocl_dlp_blocked` and every call fails with "requires AOCL-DLP".

Worse than useless — the evaluate phase keeps whichever candidate timed fastest,
and "returned without computing" times as near-zero, so an unavailable backend
wins the comparison and is cached as best for that shape.

**Fix:** select the default from what the build actually contains. Beyond that,
the tuner should not time or cache a call that did not compute, and should retry
the next candidate rather than leave the output untouched — otherwise a candidate
list spanning backends with different dtype coverage cannot work.

---

## 10. benchdnn matmul silently ignores `--kernel_name`

Two defects combine so that every `--kernel_name` runs AOCL-DLP:

- `VALID_KERNEL_NAMES` omits `native_gemm`, `native_brgemm` and the
  `auto_tuner` spelling of `auto`, all real `matmul_algo_t` values. An unlisted
  name is not an error — `matmul_utils.cpp` substitutes
  `aocl_dlp` / `aocl_dlp_blocked`.
- `set_lowoha_matmul_params()` populates `params.dtypes` but never
  `params.lowoha_algo`, the field `kernel_select()` reads. It stays
  `matmul_algo_t::none`, which `kernel_select()` turns into `aocl_dlp_blocked`.

On a `--no-aocldlp` build every `--kernel_name` therefore dies after ~2 ms with
"Selected kernel aocl_dlp_blocked requires AOCL-DLP", making matmul
unbenchmarkable on any host that cannot use AOCL-DLP. The library itself is fine;
the ops-API path already honours the name via `set_forced_kernel()`.

**Separate observation:** the matmul driver has no output comparison at all, so
it cannot detect a wrong result — only an execution failure. A kernel that
computes garbage quickly passes.

---

## 11. Cost model fabricates an L3 that does not exist

`cost_model.cpp` hardcodes `l3_bytes_per_ccd = 33554432` (32 MB) when the
detected L3 size is 0. Family 15h parts have no L3 at all, so every blocking
decision is made against 32 MB of cache that isn't there.

**Fix:** fall back to the L2 size instead of a fixed constant.

**Related, not fixed:** the threading policy is written around Zen topology. On
family 15h the FPU is shared between the two cores of a module (AMD publication
47414), so FP throughput tracks module count, not core count — measured 1.96x
going from 1 to 4 threads on a 4-core/2-module part. `threads_sharing_l3` cannot
express that.

---

## 12. Post-op chains are silently reordered when the fused op is not first

The native loopers scan the post-op chain for the first *fusable* op --
relu / gelu_tanh / gelu_erf / sigmoid / tanh / swish -- hand it to the
microkernel epilogue, and apply everything else afterwards through
`apply_postops_tile()`. The scan accepts a match at any index, but the epilogue
runs *before* the remaining ops, so fusing an op that is not at the head of the
chain reorders it.

`binary_add:sigmoid` computes `binary_add(sigmoid(x))` instead of
`sigmoid(binary_add(x))`. A binary/residual op followed by an activation is an
ordinary pattern, and the result is wrong numerics with nothing reported.

**Not ISA-specific and not a `--no-aocldlp` issue.** Reproduced on an Intel Xeon
6767P with full AVX-512, where the AVX-512 microkernels and epilogue are the ones
running -- i.e. upstream's primary platform in its default configuration.

**Reproduce:** pin the chains through a gtest input file with alpha=1 and beta=0
(isolating this from alpha/beta handling) and force the native algo:

```
256,256,256,binary_add:sigmoid,native_gemm,false,false,1,0,,,tensor
256,256,256,sigmoid:binary_add,native_gemm,false,false,1,0,,,tensor
...
./gtests --op matmul --input_file F --ndims 2 --test 2 --seed 424242 \
         --gtest_filter="*TestMatmul.F32_F32/*"
```

| chain | fusable op | before fix | after fix |
| --- | --- | --- | --- |
| `binary_add:sigmoid` | second | FAIL | pass |
| `binary_mul:relu` | second | FAIL | pass |
| `clip:tanh` | second | FAIL | pass |
| `sigmoid:binary_add` | first | pass | pass |
| `relu:binary_mul` | first | pass | pass |
| `tanh:clip` | first | pass | pass |

6 passed / 6 failed before, 12 passed / 0 failed after.

**Fix:** fuse only when the fusable op sits at index 0; otherwise leave it to the
ordered post-op pass. Applied to all four loopers that share the pattern
(fp32/bf16 GEMM, fp32/bf16 BRGEMM).

---

## 13. LIBXSMM's implemented BF16 -> FP32 path is gated off

`matmul_partitioner.cpp` rejected every dtype combination whose destination was
not BF16 and returned `aocl_dlp`. But `run_libxsmm_std()` implements BF16 inputs
with an FP32 output: the bf16->f32 branch instantiates
`libxsmm_gemm<libxsmm_bfloat16, libxsmm_bfloat16, float>` with
`LIBXSMM_DATATYPE_F32` as the output type. The gate was narrower than the code
behind it, so a working kernel was unreachable.

The combination is not obscure -- flash SDPA accumulates attention in FP32 from
BF16 Q/K/V -- and in a build without AOCL-DLP nothing else can serve it: oneDNN
needs AVX-512 for BF16, the native kernels need AVX512-BF16.

**Reproduce** on a host without AVX-512, `--no-aocldlp`, m=32 k=96 n=91:

| dtypes | before | after |
| --- | --- | --- |
| bf16 -> bf16 | 18.05 GFLOPS | 18.05 GFLOPS |
| bf16 -> f32 | every backend fails | 18.31 GFLOPS |

**Fix:** widen the gate to the destination types `run_libxsmm_std()` handles.

---

## 14. LIBXSMM strided operands are sent to DLP instead of the direct kernel

`select_partition_kernel()` rejects any strided layout for LIBXSMM -- a leading
dimension wider than the packed minimum, or `ldc != N` -- and returns
`aocl_dlp`. The restriction belongs to the partitioner, not to LIBXSMM: the
unpartitioned path passes `lda/ldb/ldc` straight to `run_libxsmm_std()` and on to
`libxsmm_gemm`, like any BLAS-style GEMM.

So a strided GEMM that LIBXSMM can compute goes to AOCL-DLP, and without
AOCL-DLP it has no backend at all. Flash SDPA hits this on every call: it tiles a
`[batch, heads, seq, head_dim]` tensor, so its operands are strided slices. One
BF16 SDPA test logged "LibXSMM partitioned kernel does not support strided
layouts, falling back to DLP" 384 times, for shapes LIBXSMM computes fine when
called directly.

**Fix:** skip the partitioner for strided operands and let the direct kernel take
them. Contiguous cases are unchanged and still partitioned.

**Combined effect of 6, 13 and 14** on an A10-8770E (Excavator, `--no-aocldlp`),
SDPA suite:

| state | passing | "no backend" errors |
| --- | --- | --- |
| unfixed | 0 (all-zero output) | -- |
| + issue 6 | 146 | 11,250 |
| + issue 13 | 220 | 7,530 |
| + issue 14 | **251, exit 0** | **0** |

These three share one shape: a dispatch gate written more narrowly than the
kernel behind it, with AOCL-DLP as the catch-all that hid the narrowness from
anyone who had it. 13 and 14 are only reachable once 6 is fixed, so they are
best filed together.

---

## Checked and found correct (not defects)

Recorded so these are not re-reported:

- **Static quant/dequant reorder dispatch.** `reorder_dtype_dispatch.cpp` picks
  the `_avx512` kernels on `algo == reorder_algo_t::native`, which looks
  unguarded, but `select_reorder_algo()` in `lowoha_reorder_utils.cpp` will not
  return `native` unless avx512f **and** avx512bw/vl are present, and falls back
  to the reference kernel with a log message when an explicit `native` request
  cannot be served. Its comment even cites the Xeon Phi case of avx512f without
  bw/vl. This is more careful than the native matmul path (issue 3).
- **aocl-utils CPUID detection.** Uses raw CPUID rather than a
  microarchitecture table, so it reports family 15h features correctly with no
  changes needed.