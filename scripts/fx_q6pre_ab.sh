#!/bin/sh
# Q6_K pre-stitched codes on the FX -- the box where this fork actually
# dispatches, and so the box whose answer decides the default.
#
# ONE binary; arms differ only by ZENDNNL_NATIVE_Q6K_PRESTITCH.
#
# -t 4 not -t 8: family 15h has one FPU per MODULE, so four threads on four
# modules is the throughput configuration and eight would contend for the same
# FPUs. Governor pinned to performance -- this box idles at 1.4-1.8 GHz on
# schedutil and has measured a 3x spread on identical code without it.
#
# IGNORE_GPU=1 because the FX has an RX 560, and without it the fork declines all
# prompt-sized work and only decode would differ between arms.
set -u
REPS=${REPS:-3}
COOL=${COOL:-150}
OUT=/tmp/fx_q6pre_ab.csv

cd "$HOME/BullDNN/build" || exit 1
make -j8 zendnnl > /tmp/fxq6_b.log 2>&1 || { echo BUILD_FAIL; exit 1; }
make install > /dev/null 2>&1 || { echo INSTALL_FAIL; exit 1; }
cd "$HOME/llama.cpp/build-vkz" || exit 1
cmake --build . --target llama-cli -j8 > /tmp/fxq6_l.log 2>&1 \
    || { echo LINK_FAIL; tail -20 /tmp/fxq6_l.log; exit 1; }

echo "arm,rep,pp_tps,tg_tps,q4k_exec,q6k_exec" > "$OUT"
for rep in $(seq 1 "$REPS"); do
    for arm in stitch pre; do
        v=0; [ "$arm" = pre ] && v=1

        sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
        sleep "$COOL"
        sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
        sleep 5

        log=/tmp/fxq6_${arm}_$rep.log
        GGML_ZENDNN_IGNORE_GPU=1 ZENDNNL_NATIVE_Q6K_PRESTITCH=$v \
        ZENDNN_DISPATCH_STATS=1 \
          timeout -s INT 500 ./bin/llama-cli -m "$HOME/models/qwen-Q4_K_M.gguf" \
            -f /tmp/q4k_prompt.txt -n 64 -t 4 -dev none -ngl 0 -no-cnv -st \
            --seed 1234 --ignore-eos --no-repack < /dev/null 2>&1 \
          | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K|^q6_K" > "$log"

        line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
        pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
        tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
        q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
        q6=$(grep -a "^q6_K" "$log" | awk '{print $4}'); q6=${q6:-0}
        echo "$arm,$rep,${pp:-NA},${tg:-NA},$q4,$q6" >> "$OUT"
    done
done
touch /tmp/fx_q6pre_ab.done
