#!/bin/sh
# Rebuild BullDNN, relink llama-cli against it, and measure decode.
# $1 = label for the results file.
#
# llama.cpp links libzendnnl_archive.a statically, so a kernel change needs
# BullDNN built AND installed AND llama-cli relinked, in that order, or the
# measurement silently describes the previous kernel.
set -u
LABEL=${1:-run}
REPS=${REPS:-3}
COOL=${COOL:-150}
OUT=/tmp/gemv_$LABEL.csv

cd "$HOME/BullDNN/build" || exit 1
make -j4 zendnnl > /tmp/gemv_build.log 2>&1 || { echo "BUILD FAILED"; tail -20 /tmp/gemv_build.log; exit 1; }
make install > /tmp/gemv_install.log 2>&1 || { echo "INSTALL FAILED"; exit 1; }

cd "$HOME/llama.cpp/build-vkz" || exit 1
cmake --build . --target llama-cli -j4 > /tmp/gemv_link.log 2>&1 \
    || { echo "LINK FAILED"; tail -20 /tmp/gemv_link.log; exit 1; }

echo "label,rep,pp_tps,tg_tps,q4k_exec" > "$OUT"
for rep in $(seq 1 "$REPS"); do
    sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
    sleep "$COOL"
    sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
    sleep 5

    log=/tmp/gemv_${LABEL}_$rep.log
    GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1 ZENDNN_DISPATCH_STATS=1 \
      timeout -s INT 400 ./bin/llama-cli -m "$HOME/models/qwen-Q4_K_M.gguf" \
        -f /tmp/q4k_prompt.txt -n 64 -t 4 -dev none -ngl 0 -no-cnv -st \
        --seed 1234 --no-repack < /dev/null 2>&1 \
      | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K" > "$log"

    line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
    pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
    tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
    q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
    echo "$LABEL,$rep,${pp:-NA},${tg:-NA},$q4" >> "$OUT"
done
touch /tmp/gemv_$LABEL.done
