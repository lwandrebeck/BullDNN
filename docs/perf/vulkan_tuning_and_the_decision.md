# Vulkan is already tuned, the FX is retired, and BullDNN has no target left

2026-08-17. The close of this line of work, with the measurements that decided it.

## The A10's `-ngl 99` cannot be improved by configuration

Sixteen configurations on qwen2.5-1.5B Q4_K_M, cooled and alternated, llama.cpp
b10470. Nothing beat the default and several were worse.

Runtime flags:

| arm | prompt | decode |
|---|---|---|
| CPU (`-dev none -ngl 0`) | 16.6 | 6.7 |
| **`-ngl 99`, defaults** | **111.2** | **16.0** |
| `-fa on` explicit | 111.8 | 16.1 |
| `-fa off` | 105.8 | 14.9 |
| `-b 2048 -ub 256` | 77.3 | 16.0 |
| `-b 2048 -ub 1024` | 111.5 | 16.0 |
| `-nkvo` | 111.9 | **10.6** |
| `-ctk q8_0 -ctv q8_0` | 107.8 | 15.7 |
| `-t 2` | 112.1 | 16.0 |

Vulkan environment knobs:

| arm | prompt | decode |
|---|---|---|
| baseline | 112.1 | 16.1 |
| `GGML_VK_FORCE_MMVQ=1` | 111.8 | 16.0 |
| `GGML_VK_DISABLE_MMVQ=1` | 111.9 | 16.1 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=16` | 109.7 | 15.8 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=256` | 111.8 | 16.1 |
| `GGML_VK_DISABLE_FUSION=1` | 110.5 | 15.9 |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 111.3 | 16.0 |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 111.7 | 16.1 |

Two are worth knowing as *negative* levers: `-fa off` costs 7% of decode, so the
`auto` default is already choosing flash attention; and `-nkvo` costs **34%** of
decode, so the KV cache must stay on the device. `-ub` below 512 wrecks prompt.

The reason nothing helps is structural: the A10 is an APU whose iGPU shares one
memory controller with the CPU, so `-ngl 99` already offloads everything, there is
no VRAM pressure to relieve, and decode is capped by that shared controller.

## Register spills: measured, and there are none

The CPU-side technique -- read the asm, look for spills and pin registers -- has a
real GPU analogue, and RADV exposes it: `MESA_SHADER_CACHE_DISABLE=true
RADV_DEBUG=shaderstats` prints SGPR/VGPR counts, spilled counts and scratch size
per pipeline. Across the 18 shaders compiled for a decode run:

| VGPRs | shaders | waves/SIMD (GCN, 256 budget) |
|-------|--------:|------------------------------|
| <= 40 | 11 | 6-10 |
| 64 | 2 | 4 |
| 84 | 2 | 3 |
| 128 | 3 | **2 of 10** |

**Zero spilled VGPRs and zero scratch, everywhere.** So the spill hypothesis is
dead. What the numbers do show is three shaders at 128 VGPRs, which caps occupancy
at two waves per SIMD and is the genuine analogue of register pressure here --
latency hiding rather than spill traffic.

That is real headroom but it is not low-hanging: VGPR count is a consequence of
tile sizes and unrolling in the GLSL, upstream tunes those across many devices,
and the stats only expose a pipeline hash so attributing a count to a shader needs
`RADV_DEBUG=asm` correlation first. Anyone picking this up should start there and
should not expect a flag to do it.

## The FX: partial offload works, and `-ngl 99` is not the best setting

Qwen2.5-Coder-7B Q4_K_M, 4.36 GB against 4 GB of VRAM, so it cannot fully fit --
the partial-offload case.

| `-ngl` | prompt | decode |
|-------:|-------:|-------:|
| CPU | 4.7 | 2.1 |
| 8 | 12.9 | 2.7 |
| 16 | 14.8 | 3.6 |
| 20 | 16.4 | 4.4 |
| **24** | 17.9 | **5.3** |
| 99 | **18.2** | 4.6 |

Monotonic up to 24, and then `-ngl 99` gives back 15% of decode while gaining 0.3
of prompt: pushing every layer at a GPU that cannot hold them leaves too little
room for the KV cache and compute buffers. **On a model that overflows VRAM, sweep
`-ngl` rather than assuming 99.** That generalises to any 4 GB card.

## Why the work stops here

The FX has the fastest CPU and the fastest GPU of the two boxes for models that
fit, and it is still the wrong machine:

- **Capacity.** 15 GB RAM and 4 GB VRAM. The homelab's 30B MoE (~18 GB) does not
  fit at any speed.
- **Perf per watt goes the other way.** Same model and flags: FX+RX560 47.9 t/s
  decode at ~170 W (95 W CPU + 75 W GPU) against the A10 iGPU's 16.0 t/s at ~35 W.
  2.6x the throughput and **1.6x worse per watt**; prompt likewise, 2.35 against
  3.18 t/s/W.

And BullDNN's only remaining target *was* the FX. On bdver4 ggml is built
`-march=native`, which unlocks its `q4_K_8x8` repack path, and the fork ties it on
prompt and loses 0.75x on decode -- so the AVX2 gate already declines everything
there. Both builds were symmetric throughout: ggml at `-march=native` resolving to
bdver4/bdver2, BullDNN at `-march=bdver4`/`bdver2`. The comparison was never
stacked.

Seven optimisation attempts on that decode path produced one 4% win which did not
reproduce on the other microarchitecture. The conclusion is that ggml's kernels
fit this problem and the GPU beats both.

**What survives.** The correctness work, which stands regardless: bf16 small-MR
microkernels, per-tensor and per-channel INT8 scales, f32 kernel selection, the
libxsmm AVX2-on-bdver2 SIGILL, the MoE crash guards, and the measurement
discipline in these files. And for the homelab, one sentence: run `-ngl 99` with
defaults on the Excavator nodes, and do not spend time on CPU matmul.
