# Patches against llama.cpp

Applied to a `ggml-org/llama.cpp` checkout to build it against this fork. Each is
a defect in llama.cpp rather than a local preference, and each should go upstream;
they live here so the benchmark configuration is reproducible in the meantime.

Tested against **b10437** (`16d222f`).

## 0001 — ggml-zendnn: custom `ZENDNN_ROOT` build fixes

`docs/backend/ZenDNN.md` documents "Option 2: Use Custom ZenDNN Installation",
pointing `ZENDNN_ROOT` at a ZenDNN built from source. That path does not work as
written; three separate things stop it, and all three were hit in order on a
family 15h host.

1. **`libaocl-dlp.a` is hardcoded in the archive link list.** ZenDNN can be built
   `--no-aocldlp`, and on any host without AVX-512 that is the sensible
   configuration -- AOCL-DLP declines every INT8 kernel there and, measured on an
   A10-8770E, runs BF16 at roughly half the speed of ggml's own CPU backend. A
   build without it has no such archive and the link fails. Now linked only if
   present.

2. **`deps/fbgemm/include` is missing from the include list.** ZenDNN's installed
   public headers include `fbgemm/FbgemmEmbedding.h`
   (`lowoha_operators/embedding_bag/fbgemm_kernel.hpp`), so *any* custom
   `ZENDNN_ROOT` fails to compile the backend, regardless of host.

3. **The link mode is chosen from llama.cpp's own `BUILD_SHARED_LIBS`.** That
   says nothing about how the custom ZenDNN was built. ZenDNN's documented build
   produces an archive by default, so with llama.cpp's default shared build the
   backend links `-lzendnnl`, which does not exist, and fails. Now decided by
   which library is actually present under `${ZENDNN_ROOT}/zendnnl/lib`, falling
   back to the old behaviour when CMake is going to download and build ZenDNN
   itself.

Apply with:

```sh
cd llama.cpp
git apply /path/to/BullDNN/patches/llama.cpp/0001-*.patch
cmake -B build -DGGML_VULKAN=ON -DGGML_ZENDNN=ON \
      -DZENDNN_ROOT=/path/to/BullDNN/build/install -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## Note on AOCL-DLP for family 15h

Build this fork `--no-aocldlp` on Bulldozer through Excavator. AOCL-DLP cannot
run INT8 there at all, and where it does run it loses: forcing
`ZENDNNL_MATMUL_ALGO=1` (aocl_dlp_blocked) for BF16 prompt processing on an
A10-8770E gave 37-52 t/s against 68-69 for ggml's CPU backend, with a spread wide
enough that the default dispatcher's intermittent choice of it is visible as
run-to-run variance. See `docs/perf/llamacpp_b10437.csv`.
