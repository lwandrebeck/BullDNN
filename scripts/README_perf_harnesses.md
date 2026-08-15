# Native-kernel performance harnesses

Three small drivers used to produce the figures in `docs/perf/`. They exist
because benchdnn goes through the dispatcher and cannot isolate a single native
kernel, and because two of the questions they answer -- which multiply-add form,
and whether a microkernel spills -- are settled below the level a sweep can see.

They are standalone `.cpp` files rather than CMake targets on purpose: each is
compiled against an existing build tree in one command, so a variant can be built,
measured and thrown away without touching the library's build files.

| File | Answers |
| --- | --- |
| `int8_symq_probe.cpp` | Is the symmetric per-group INT8 kernel correct, and what does its hot loop reach in isolation? Modes: `validate`, `looper`, `ukernel`. |
| `int8_symq_api_bench.cpp` | What does a caller see through `matmul_direct`, with packing, operand checks and scale widening included? |
| `fp32_flavour_bench.cpp` + `fp32_flavour_ab.sh` | Which FP32 multiply-add form wins on this microarchitecture, and is the register pinning earning its place? |

## Building one

Substitute the target arch for the host: `bdver4` on the A10-8770E (Excavator),
`bdver2` on the FX-8370E (Piledriver). Never build bdver4 code for a bdver2 part;
it has no AVX2 and will take an illegal instruction.

```
# kernel-level probe: compiles the two kernel sources directly, no library needed
g++ -O3 -march=bdver2 -std=c++17 -fopenmp -DNDEBUG -I zendnnl/src \
    scripts/int8_symq_probe.cpp \
    zendnnl/src/lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_symq_ukernel_128.cpp \
    zendnnl/src/lowoha_operators/matmul/matmul_native/gemm/looper/int8_symq_looper_128.cpp \
    -o probe
./probe validate            # correctness, both flavours via ZENDNNL_NATIVE_SYMQ_NO_XOP=1
./probe ukernel 9           # hot loop alone
./probe looper 5 4          # end-to-end, 4 threads
```

The API-level benches link the built archive; `fp32_flavour_ab.sh` shows the full
link line and resolves `lib/` vs `lib64/` per dependency, so copy it from there.

## Reading the results on these boxes

* **Pin the governor before measuring on the FX-8370E.** On stock `schedutil` it
  idles at 1.4–1.8 GHz and identical code measured a 3x spread between passes.
  `sudo cpupower frequency-set -g performance`, and stop `packagekit` and
  `gnome-software`.
* **Idle the A10-8770E before every pass, and alternate the arms.** It is a 35W
  part that cannot hold a 4-thread SIMD load: the same binary measures 40.11 t/s
  hot and 84.36 t/s after five minutes idle. A back-to-back A/B therefore hands
  the first arm the cold run and manufactures a 2x difference — which is exactly
  what happened, see the retraction in `docs/perf/llamacpp_b10437.csv`. Use
  `a10_cooled_ab.sh`. Lowering the thread count does not help; 2, 3 and 4 all
  collapse. (An earlier version of this file claimed the A10's clock is flat and
  needs nothing. It is not and it does.)
* **A 2x effect that changes sign when you reorder the arms is an ordering
  artefact.** Check that before believing any large result: one bracketed pass
  read baseline 42.87 / fork 81.28 / baseline 39.94, which is the same magnitude
  pointing the other way.
* Both boxes resolve effects above roughly 10% and nothing below, so these
  harnesses report **best-of-N and the worst rep**, never a mean. If a row's own
  best exceeds its own min by more than ~10%, it cannot adjudicate a few-percent
  difference whatever its median says.
* When A/B-ing, include an arm whose code generation is provably identical to the
  baseline. Its apparent "gain" is that configuration's noise floor, measured
  inside the same pass rather than remembered from another day.
* `p.num_threads` overrides `OMP_NUM_THREADS`, so set it in the params; the
  environment variable will not change what the native path does.
