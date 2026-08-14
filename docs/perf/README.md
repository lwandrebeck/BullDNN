# Family 15h performance baselines

Reference numbers for the AMD family 15h parts BullDNN targets (Bulldozer,
Piledriver, Steamroller, Excavator). These exist so later changes can be
diffed against a known state, and so the four microarchitectures can be
compared against each other on identical shapes.

## Files

| File | Host | Microarchitecture |
| --- | --- | --- |
| `baseline_a10-8770e_excavator.csv` | AMD PRO A10-8770E, 4 cores / 2 modules, 2.8 GHz max | Excavator (bdver4) |
| `int8_symq_a10-8770e_excavator.csv` | same host, symmetric per-group INT8 | Excavator (bdver4) |
| `int8_symq_fx-8370e_piledriver.csv` | AMD FX-8370E, 8 cores / 4 modules | Piledriver (bdver2) |

The Piledriver file is the first data from a second family 15h microarchitecture,
and it disagrees with the Excavator one in a way worth knowing: XOP's fused
reduction is worth 1.13-1.20x on bdver4 and nothing at all on bdver2. Treat
per-flavour results as microarchitecture-specific until measured on both. That
box also needs its governor pinned to `performance` before it can be measured at
all — see the header of the file.

Note on reading the INT8 file: this box resolves effects above roughly 10% and
nothing below, and its means drift between batches, so the columns are best-of-N
rather than averages and the small differences in it are not results. The
decode rows (M=1) moved by 4-7x across the three stages recorded there, which is
well clear of that floor; the prompt and square rows did not move at all.

Columns are `dtype,threads,M,KN,GFLOPS`, where `KN` is the value used for both
K and N, and `GFLOPS` is what benchdnn reports. `FAIL` in the GFLOPS column
means no backend could run that configuration.

## How the numbers were taken

Built with `--no-aocldlp` (the family 15h configuration, since AOCL-DLP has no
support for these parts), so the backends in play are oneDNN, libxsmm and the
native kernels. Driven through the default `auto_tuner` path rather than a named
algorithm, so the figures reflect what an application actually gets:

```
OMP_NUM_THREADS=$threads benchdnn --op=matmul --lowoha=true \
    --m=$M --k=$KN --n=$KN --sdt=$dt --wdt=$dt --ddt=$dt \
    --kernel_name=auto_tuner --iters=$iters --warmup_iters=3
```

`iters` is 30, dropping to 10 for KN=2048 to keep the sweep's total runtime
reasonable on a 4-core part.

**Measure on an idle machine.** A sweep taken while a compile was running on all
four cores read about 2.6x low across every algorithm. Verify with `uptime` or
by checking for stray `cc1plus`/`gtests` processes before starting, and sanity
check one shape against a figure already in the table.

## What the Excavator numbers show

Peak observed f32 is 66.8 GFLOPS (4 threads, M=2048, K=N=512). Theoretical peak
for this part is 2 modules x 2 128-bit FMACs x 8 FLOP/cycle x 2.8 GHz =
89.6 GFLOPS, so the GEMM path reaches roughly 75% of peak.

Thread scaling stops at about 2x from 1 to 4 threads (1.96x at K=N=512, 2.20x
at 2048). That is not a software limit: on family 15h the FPU is **shared
between the two cores of a module** (see AMD 47414, the family 15h software
optimization guide), and this part has 4 cores in 2 modules. Throughput
therefore tracks module count, not core count. For FP-bound work, threads
beyond the module count buy nothing, which is worth remembering when reading
the cost model's threading policy -- it is written around Zen's topology, where
cores do not share FPUs.

bf16 reaches about 59% of f32 throughput. These parts have no bf16 hardware, so
libxsmm converts up to f32; libxsmm is also the only backend here that serves
bf16 at all, because oneDNN cannot create a bf16 primitive without AVX-512 or
AVX2-VNNI-2 and the native bf16 kernels require AVX512-BF16.

Throughput drops about 22% from K=N=512 to K=N=2048 (66.8 -> 52.3 at 4
threads). L2 is 1-2 MB shared per module and there is **no L3** on these parts
(see AMD 50742, the BKDG for models 60h-6Fh), while a 2048x2048 f32 matrix is
16 MB, so the large shapes are memory-bound.

M=1 (GEMV) is much lower, 5.8-9.6 GFLOPS for f32. That shape streams the whole
weight matrix for very little arithmetic, so it is bandwidth-bound rather than
FPU-bound: K=N=2048 f32 works out to roughly 11 GB/s of weight traffic.
