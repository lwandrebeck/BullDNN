#!/bin/sh
# Does dispatching the MoE experts to this fork help? qwen3-coder-30b-a3b.
#
# Three arms, ONE binary:
#   stock       no IGNORE flags, so the AVX2 rule declines everything -- what a
#               user gets today on Excavator.
#   nommid      fork active but MUL_MAT_ID declined: attention and the dense
#               matmuls only. This is what the fork did before today.
#   mmid        MUL_MAT_ID allowed too, so the experts dispatch as well.
#
# --no-repack and --ignore-eos on every arm: the first because ggml's repack
# otherwise claims Q4_K before the fork sees it, the second because without it the
# arms generate different token counts and the KV cache grows differently.
set -u
REPS=${REPS:-3}
COOL=${COOL:-150}
NGEN=${NGEN:-32}
MODEL=$HOME/models/qwen3-coder-30b-a3b-Q4_K_M.gguf
OUT=/tmp/moe_ab.csv

cd "$HOME/llama.cpp/build-vkz" || exit 1
echo "arm,rep,pp_tps,tg_tps,q4k_exec,q6k_exec,peak_rss_gb" > "$OUT"

for rep in $(seq 1 "$REPS"); do
    for arm in stock nommid mmid; do
        case $arm in
            stock)  env_set="" ;;
            nommid) env_set="GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1" ;;
            mmid)   env_set="GGML_ZENDNN_IGNORE_AVX2=1 GGML_ZENDNN_IGNORE_GPU=1 GGML_ZENDNN_ALLOW_MUL_MAT_ID=1" ;;
        esac

        sudo -n cpupower frequency-set -g powersave >/dev/null 2>&1
        sleep "$COOL"
        sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
        sleep 5

        log=/tmp/moe_${arm}_$rep.log
        # shellcheck disable=SC2086
        /usr/bin/time -v env $env_set ZENDNN_DISPATCH_STATS=1 \
          timeout -s INT 900 ./bin/llama-cli -m "$MODEL" -p hi -n "$NGEN" \
            -t 4 -dev none -ngl 0 -no-cnv -st --seed 1234 --ignore-eos \
            --no-repack < /dev/null 2>&1 \
          | grep -a --line-buffered -E "Prompt:.*Generation:|^q4_K|^q6_K|Maximum resident" \
          > "$log"

        line=$(grep -a "Prompt:" "$log" | head -1 | tr ',' '.')
        pp=$(echo "$line" | grep -oP "Prompt: *\K[0-9.]+")
        tg=$(echo "$line" | grep -oP "Generation: *\K[0-9.]+")
        q4=$(grep -a "^q4_K" "$log" | awk '{print $4}'); q4=${q4:-0}
        q6=$(grep -a "^q6_K" "$log" | awk '{print $4}'); q6=${q6:-0}
        kb=$(grep -a "Maximum resident" "$log" | grep -oE "[0-9]+$"); kb=${kb:-0}
        gb=$(awk -v k="$kb" 'BEGIN{printf "%.1f", k/1048576}')
        echo "$arm,$rep,${pp:-NA},${tg:-NA},$q4,$q6,$gb" >> "$OUT"
    done
done
touch /tmp/moe_ab.done
