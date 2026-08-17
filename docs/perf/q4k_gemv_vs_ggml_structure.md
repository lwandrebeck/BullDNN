# What ggml's Q4_K decode does that BullDNN's does not

Written 2026-08-17, after `q4k_vs_repacked_ggml_a10.csv` measured the fork at
1.00x on prompt and 0.86-0.92x on generation against ggml's repacked kernels.
The premise of the question is right: these are hand-written kernels against a
hand-written kernel, and there is no reason to be behind. Reading both, the fork
is behind for structural reasons, not because ggml has an instruction it lacks.

Both sides read the Q4_K blocks **directly**, packed, with no unpack to int8, so
weight memory traffic is identical and this is not a bandwidth story. At ~6 t/s
on a 934 MB model that is ~6 GB/s against a dual-channel DDR3 controller, well
under peak, so decode here is issue/latency bound rather than memory bound --
which is why the differences below matter at all.

The two kernels compared:

- ggml `ggml_gemv_q4_K_8x8_q8_K`, `ggml/src/ggml-cpu/arch/x86/repack.cpp:1464`
- fork `superblock_dots` + `run`, `int8_q4k_gemv_128.cpp:163-302`

---

## 1. Eight outputs in eight lanes, versus one output and a reduction tree

ggml's repacked layout interleaves **8 weight rows**, so the 8 lanes of a
256-bit accumulator *are* 8 different output columns. The inner loop therefore
contains **no horizontal reduction at all**; at the end of the whole row it does
one `permutevar8x32` and one store for 8 results.

The fork computes **one output row per pass** and must collapse a vector to a
scalar. `int8_q4k_gemv_128.cpp:222` does

    d = _mm_hadd_epi32(_mm_hadd_epi32(v0, v1), _mm_hadd_epi32(v2, v3));

three `hadd`s per four sub-blocks, so **six per super-block per output row**,
plus a final `hsum_ps` per output at line 300. `VPHADDD` is a multi-uop,
high-latency instruction on family 15h. Against 16 `maddubs` of real work per
super-block, six hadds is a large fraction of the loop, and ggml pays none of it.

This is the big one, and it is the root of items 4 and 5 as well.

## 2. The sub-block scale is applied in the integer domain, for free

ggml widens s16 to s32 with `_mm256_madd_epi16(iacc, scales)` -- the same
instruction that widens also **multiplies by the 6-bit sub-block scale**. The
scale costs nothing.

The fork widens with `_mm_madd_epi16(words, ones)` (`Q4K_ACCUM`, line 448): a
multiply by one, doing no arithmetic. It then applies the scales later in fp32,
per four sub-blocks (lines 279-298): `cvtepi32_ps`, three `mul_ps`, a `sub_ps`
and an `add_ps`, plus a second `cvtepi32_ps` for the row-sum term. All of that
is work ggml folds into an instruction it had to issue anyway.

## 3. Widening every multiply instead of every eight

The fork widens after **every** `maddubs`. It does not have to. Q4_K codes are
unsigned 0..15 and the activation contract excludes -128, so one `maddubs` lane
is at most `15*127*2 = 3810`, and **eight** accumulate to 30480, inside the
signed 16-bit limit of 32767. ggml uses exactly that headroom: eight `maddubs`
with cheap `add_epi16`, then one `madd_epi16`.

Safe accumulation depth by type, same reasoning:

| type  | max code | per-lane max | safe accumulations |
|-------|---------:|-------------:|-------------------:|
| Q4_K  |       15 |         3810 |                  8 |
| Q5_K  |       31 |         7874 |                  4 |
| Q6_K  |       63 |        16002 |                  2 |

Caveat: on a host with XOP the fork's `Q4K_ACCUM` becomes a single fused
`VPMADCSWD` (line 466), so this costs one instruction rather than two, and the
saving here is smaller than it looks on the A10. It is still a saving, and on a
non-XOP host it is two instructions per multiply.

## 4. Six-bit scale unpacking is scalar, and repeated per output row

`get_scale_min_k4` is called **eight times per super-block per output row**
(line 274), scalar, and the results are then poked back into vectors with
`_mm_setr_ps`. ggml unpacks the packed 6-bit scales vectorially with
`shuffle_epi8` against `kmask1/2/3`, once for **eight** output rows, so the cost
is amortised eight ways and never leaves the vector domain.

## 5. The activation stream is re-read once per output row

In ggml's interleaved layout a single 32-byte weight load serves 8 output rows
at the same K, so one broadcast activation is reused eight times. In the fork
the activation loads `al`/`ah` (lines 204-209) sit inside the per-row loop, so
for N outputs the activation vector is streamed N times. It is L1-resident, so
this is issue bandwidth rather than memory, but it is N times the loads.

## 6. Where the fork is already equal, and should not be "fixed"

- **Direct packed reads.** Both consume the GGML blocks in place. No unpack.
- **The min correction.** ggml uses `bsums` precomputed in `block_q8_K`; the
  fork uses a precomputed `rowsum` array (line 287). Same trick, same cost.
- **The multiply itself.** Per output row per super-block both issue the
  equivalent of 16 128-bit `maddubs`. ggml's 256-bit ops crack into two
  128-bit uops on this family, so the arithmetic is a wash -- which is the
  point: the gap is overhead around the multiply, not the multiply.

---

## What to do, in order of expected value

1. **Interleave 8 rows in the prepacked weight cache.** The fork already has
   `INT8PrepackedWeightCache`, so there is somewhere to put a repacked layout
   without touching the caller. This removes item 1 entirely (no hadd, no
   `hsum_ps`), and items 4 and 5 fall out with it. It is the change that makes
   the other three worth doing.
2. **Fold the sub-block scale into the widening `madd_epi16`.** Requires the
   scales as s16 in the right lane order, which item 1's repack can prepare.
   Removes most of the fp32 epilogue.
3. **Defer widening to every 8 multiplies** (4 for Q5_K, 2 for Q6_K).
   Independent of the others and testable on its own.

None of this needs an instruction family 15h does not have. It needs ggml's
data layout.

**Measure first, as ever.** The ranking above is static reading, not profiling;
item 1 is the only one whose cost is clearly large. Item 3 is the cheapest to
try and the easiest to attribute, so it is the sensible first experiment even
though it is ranked third by expected size. Use `--no-repack` on both arms of
any comparison -- see `a10_dispatch_mystery.csv`.
