#!/bin/sh
# Row-at-a-time against the four-row 128-bit and eight-row 256-bit interleaves.
# ONE binary; the arms differ only by ZENDNNL_NATIVE_Q4K_GEMV_ILV (0/1/2).
set -u
REPS=${REPS:-3}
COOL=${COOL:-150}
OUT=/tmp/ilv8_ab.csv

cd "$HOME/BullDNN/build" || exit 1
make -j4 zendnnl > /tmp/i8b.log 2>&1 || { echo BUILD_FAIL; exit 1; }
make install > /dev/null 2>&1 || { echo INSTALL_FAIL; exit 1; }
cd "$HOME/llama.cpp/build-vkz" || exit 1
cmake --build . --target llama-cli -j4 > /tmp/i8l.log 2>&1 \
    || { echo LINK_FAIL; tail -20 /tmp/i8l.log; exit 1; }

echo "arm,rep,pp_tps,tg_tps,q4k_exec" > "$OUT"
for rep in $(seq 1 "$REPS"); do
    for arm in row ilv4 ilv8; do
        case $arm in
            row)  v=0 ;;
            ilv4) v=1 ;;
            ilv8) v=2 ;;
        esac

        sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
        sleep "$COOL"
        sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
        sleep 5

        log=/tmp/i8_${arm}_$rep.log
        GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1 \
        ZENDNNL_NATIVE_Q4K_GEMV_ILV=$v ZENDNN_DISPATCH_STATS=1 \
          timeout -s INT 400 ./bin/llama-cli -m "$HOME/models/qwen-Q4_K_M.gguf" \
            -f /tmp/q4k_prompt.txt -n 64 -t 4 -dev none -ngl 0 -no-cnv -st \
            --seed 1234 --ignore-eos --no-repack < /dev/null 2>&1 \
          | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K" > "$log"

        line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
        pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
        tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
        q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
        echo "$arm,$rep,${pp:-NA},${tg:-NA},$q4" >> "$OUT"
    done
done
touch /tmp/ilv8_ab.done
