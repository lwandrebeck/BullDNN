#!/bin/bash
# Cooled kernel-vs-ggml comparison via llama.cpp's test-backend-ops.
#
# Two things this box forces on any measurement:
#
#  1. It thermally throttles ~2x under sustained SIMD load, so every measurement
#     needs a cooldown first. Idling on the `performance` governor cools slowly;
#     dropping to `powersave` parks the cores at ~800 MHz and cools far faster,
#     so the governor is switched down for the wait and back up for the run.
#  2. test-backend-ops only reaches this fork's kernels with the ggml-zendnn
#     adaptive fallback disabled -- its gate rejects any matmul with a token
#     count <= 128, which is every decode shape and, off by one, pp128 as well.
#
# The two backends are run adjacently at each shape so the ratio is taken under
# matched conditions; absolute GFLOPS still move with temperature.
set -u
BIN=${BIN:-$HOME/llama.cpp/build-vkz/bin/test-backend-ops}
OUT=${OUT:-/tmp/a10_backend_ops.csv}
COOL=${COOL:-90}
MDIM=${MDIM:-4096}   # weight rows, i.e. output columns
KDIM=${KDIM:-14336}  # reduction extent
TYPES=${TYPES:-q4_K q5_K}
NS=${NS:-1 8 512}    # token counts
# hwmon indices are not stable across reboots -- k10temp was hwmon1 one boot and
# hwmon2 the next -- so find it by name rather than by number.
TEMP=$(for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name" 2>/dev/null)" = "k10temp" ] && echo "$h/temp1_input" && break
done)
TEMP=${TEMP:-/dev/null}

gov() { sudo -n cpupower frequency-set -g "$1" >/dev/null 2>&1; }

echo "type,n,backend,gflops,temp_mC" > "$OUT"
for t in $TYPES; do
    for n in $NS; do
        gov powersave
        sleep "$COOL"
        gov performance
        for b in CPU ZenDNN; do
            tb=$(cat $TEMP 2>/dev/null || echo NA)
            g=$(GGML_ZENDNN_ADAPTIVE_FALLBACK=0 timeout 400 "$BIN" perf \
                    -o MUL_MAT -b $b \
                    -p "type_a=$t,type_b=f32,m=$MDIM,n=$n,k=$KDIM" 2>&1 \
                | grep -oE '[0-9.]+ GFLOPS' | head -1 | cut -d' ' -f1)
            echo "$t,$n,$b,$g,$tb" >> "$OUT"
        done
    done
done
gov performance
echo "results in $OUT"
