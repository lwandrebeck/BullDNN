#!/bin/sh
# Q6_K decode: sixteen live accumulators against eight.
#
# Not runtime-switchable, so this builds TWO binaries and alternates them with a
# cooldown before each run -- which is the only safe way to compare two builds on
# a box that throttles. --ignore-eos pins both to 64 tokens.
#
# Expects /tmp/q6_fix.cpp (the eight-accumulator version) and /tmp/q6_old.cpp
# (whatever it replaced) to exist.
set -u
REPS=${REPS:-3}
COOL=${COOL:-150}
SRC=$HOME/BullDNN/zendnnl/src/lowoha_operators/matmul/matmul_native/gemm/kernel/int8/int8_q4k_gemv_128.cpp
OUT=/tmp/q6_ab.csv

build_as() {
    cp "$2" "$SRC" || exit 1
    cd "$HOME/BullDNN/build" || exit 1
    make -j4 zendnnl > /tmp/q6_b_$1.log 2>&1 || { echo "BUILD_FAIL $1"; exit 1; }
    make install > /dev/null 2>&1 || { echo "INSTALL_FAIL $1"; exit 1; }
    cd "$HOME/llama.cpp/build-vkz" || exit 1
    cmake --build . --target llama-cli -j4 > /tmp/q6_l_$1.log 2>&1 \
        || { echo "LINK_FAIL $1"; exit 1; }
    cp bin/llama-cli "bin/llama-cli-$1" || exit 1
}

build_as fix /tmp/q6_fix.cpp
build_as old /tmp/q6_old.cpp

cd "$HOME/llama.cpp/build-vkz" || exit 1
echo "arm,rep,pp_tps,tg_tps,q4k_exec,q6k_exec" > "$OUT"
for rep in $(seq 1 "$REPS"); do
    for arm in old fix; do
        sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
        sleep "$COOL"
        sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
        sleep 5

        log=/tmp/q6_${arm}_$rep.log
        GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1 ZENDNN_DISPATCH_STATS=1 \
          timeout -s INT 400 "./bin/llama-cli-$arm" \
            -m "$HOME/models/qwen-Q4_K_M.gguf" -f /tmp/q4k_prompt.txt -n 64 \
            -t 4 -dev none -ngl 0 -no-cnv -st --seed 1234 --ignore-eos \
            --no-repack < /dev/null 2>&1 \
          | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K|^q6_K" > "$log"

        line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
        pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
        tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
        q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
        q6=$(grep -a "^q6_K" "$log" | awk '{print $4}'); q6=${q6:-0}
        echo "$arm,$rep,${pp:-NA},${tg:-NA},$q4,$q6" >> "$OUT"
    done
done
# leave the tree holding the fixed version
cp /tmp/q6_fix.cpp "$SRC"
touch /tmp/q6_ab.done
