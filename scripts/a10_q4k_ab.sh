#!/bin/sh
# Does BullDNN's Q4_K beat ggml's REPACKED Q4_K on Excavator?
#
# One binary (build-vkz) for every arm, so nothing differs but flags:
#
#   A  stock+repack   default env. The A10 has AVX2 so the fork's own gate
#                     declines every op (verified: empty stats table). This is
#                     what a user gets today, and it is the arm to beat.
#   B  stock-repack   --no-repack, fork still declining. Isolates what repack
#                     itself is worth, so C can be read against the right thing.
#   C  bulldnn        --no-repack + IGNORE_AVX2/IGNORE_GPU, so the fork takes
#                     q4_K.
#
# llama-bench cannot do this: it has no --no-repack and ignores LLAMA_ARG_REPACK
# (measured). llama-cli can, and prints "[ Prompt: X t/s | Generation: Y t/s ]".
# -st is required or it drops into the chat UI and spins on EOF stdin, which
# wrote a 395 MB log into a RAM-backed /tmp the first time.
#
# Arms alternate and every one is preceded by a cooldown: this part throttles
# ~2x under sustained SIMD and would otherwise measure the heatsink.
set -u
BIN=$HOME/llama.cpp/build-vkz/bin/llama-cli
MODEL=$HOME/models/qwen-Q4_K_M.gguf
PROMPT=/tmp/q4k_prompt.txt
OUT=/tmp/q4k_ab
REPS=${REPS:-3}
COOL=${COOL:-150}
NGEN=${NGEN:-64}

mkdir -p $OUT
rm -f $OUT/*.log $OUT/results.csv $OUT/DONE
echo "arm,rep,pp_tps,tg_tps,q4k_exec,q6k_exec" > $OUT/results.csv

cool() {
    sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
    sleep "$1"
    sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
    sleep 5
}

run_arm() {
    arm=$1; rep=$2
    log=$OUT/${arm}_${rep}.log
    case $arm in
        A) pfx="";  extra="" ;;
        B) pfx="";  extra="--no-repack" ;;
        C) pfx="GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1"; extra="--no-repack" ;;
        D) pfx="GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1 GGML_ZENDNN_ONLY_TYPES=q4_K"; extra="--no-repack" ;;
    esac
    # grep, not a plain redirect: the chat UI can flood stdout and /tmp is tmpfs.
    # shellcheck disable=SC2086
    env $pfx ZENDNN_DISPATCH_STATS=1 timeout -s INT 400 \
        "$BIN" -m "$MODEL" -f "$PROMPT" -n $NGEN -t 4 -dev none -ngl 0 \
               -no-cnv -st --seed 1234 $extra < /dev/null 2>&1 \
      | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K|^q6_K" > "$log"

    # French locale prints the decimal as a comma.
    line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
    pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
    tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
    q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
    q6=$(grep -a "^q6_K" "$log" | awk '{print $4}'); q6=${q6:-0}
    echo "$arm,$rep,${pp:-NA},${tg:-NA},$q4,$q6" >> $OUT/results.csv
}

for rep in $(seq 1 "$REPS"); do
    for arm in A C D; do
        cool "$COOL"
        run_arm "$arm" "$rep"
    done
done
touch $OUT/DONE
