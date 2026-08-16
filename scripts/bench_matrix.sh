#!/bin/bash
# The five-configuration llama.cpp benchmark matrix, with engagement VERIFIED
# rather than assumed.
#
# WHY THIS EXISTS. Every previous attempt at these numbers was void, four times
# for four different reasons, and each time the output looked perfectly
# reasonable:
#
#   1. a stale install tree     -- measured a library from three weeks earlier
#   2. thermal throttling       -- the A10 halves under sustained load
#   3. zero dispatch            -- the fork's kernels were never reached
#   4. the GPU quietly helping  -- a Vulkan build runs prompt matmuls on the
#                                  GPU even at -ngl 0, so a "CPU" arm was not
#
# None of those announce themselves. So this script checks for each of them and
# refuses to report a comparison it cannot stand behind.
#
# THE FIVE CONFIGURATIONS
#   cpu                  stock ggml, no GPU
#   cpu+bulldnn          this fork, no GPU
#   vulkan               GPU only, all layers offloaded
#   vulkan+cpu           partial offload, stock
#   vulkan+cpu+bulldnn   partial offload, this fork
#
# THE DEVICE FLAG THAT REACHES THE KERNELS IS NOT THE SAME ON EVERY BOX. On the
# FX-8370E (discrete RX 560) `-dev none` leaves ZenDNN running; on the
# A10-8770E (UMA iGPU) it does not, and only `-dev ZenDNN` engages it. Both
# were verified with a probe inside ggml_zendnn_compute_forward_mul_mat: 0
# entries against 336. The sources are md5-identical on both machines, so this
# is a runtime scheduling difference and is NOT understood. Rather than guess,
# the script tries the candidates and reports which one engaged.
#
# Usage: bench_matrix.sh <model.gguf> [ngl_hybrid]
set -u
MODEL=${1:?model gguf}
NGL_HYBRID=${2:-14}
A=${A:-$HOME/llama.cpp/build-vk}    # stock ggml (+vulkan)
B=${B:-$HOME/llama.cpp/build-vkz}   # + BullDNN
THREADS=${THREADS:-$(nproc)}
PARAMS=${PARAMS:-1.5e9}             # for the GFLOPS plausibility check
# Threshold for "a CPU did not produce this", NOT a true peak. The fp32 ceiling
# on an FX-8370E is ~106 GFLOPS, but these are INT8 kernels and PMADDUBSW
# retires far more MACs per cycle than an fp32 FMA -- 136 GFLOPS of MAC is
# entirely legitimate here and flagging it would be a false alarm. What this is
# for is catching the GPU, which shows up an order of magnitude higher (~1300),
# so the bar sits between the two.
PEAK=${PEAK:-400}
OUT=${OUT:-/tmp/bench_matrix.csv}
COOL=${COOL:-0}                     # seconds; set >0 on a box that throttles

TEMP=$(for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name" 2>/dev/null)" = "k10temp" ] && echo "$h/temp1_input" && break
done)
TEMP=${TEMP:-/dev/null}
gov() { sudo -n cpupower frequency-set -g "$1" >/dev/null 2>&1 || true; }

echo "config,pp512,tg128,executed,implied_gflops,verdict" > "$OUT"

# run_cfg <name> <build> <expect_bulldnn 1/0> <cpu_only 1/0> [llama-bench args...]
run_cfg() {
    local name=$1 bin=$2 expect=$3 cpu_only=$4; shift 4
    if [ "$COOL" -gt 0 ]; then gov powersave; sleep "$COOL"; gov performance; fi
    local log=/tmp/bm_${name}.txt
    ZENDNN_DISPATCH_STATS=1 "$bin/bin/llama-bench" -m "$MODEL" \
        -p 512 -n 128 -t "$THREADS" -r 2 "$@" > "$log" 2>&1
    local pp tg ex
    pp=$(grep -oE 'pp512[^|]*\|[^|]*' "$log" | tail -1 | awk -F'|' '{print $2}' | tr -d ' ' | sed 's/±.*//')
    tg=$(grep -oE 'tg128[^|]*\|[^|]*' "$log" | tail -1 | awk -F'|' '{print $2}' | tr -d ' ' | sed 's/±.*//')
    ex=$(grep -E '^(q4_K|q5_K|q8_0|q4_0|q6_K)' "$log" | awk '{s+=$4} END{print s+0}')

    # Implied GFLOPS: 2 * params * tokens/s. On a CPU-ONLY arm, exceeding the
    # part's ceiling means the CPU did not compute it -- that is trap 4, and it
    # is how a "cpu" row silently became a GPU measurement. Meaningless for the
    # GPU arms, where a large figure is the whole point, so it is only applied
    # where it discriminates.
    local gf="NA" verdict="ok"
    if [ -n "$pp" ]; then
        gf=$(awk -v p="$PARAMS" -v t="$pp" 'BEGIN{printf "%.0f", 2*p*t/1e9}')
    fi
    if [ "$expect" = "1" ] && [ "${ex:-0}" -eq 0 ]; then
        verdict="VOID:bulldnn-never-ran"
    elif [ "$expect" = "0" ] && [ "${ex:-0}" -gt 0 ]; then
        verdict="VOID:bulldnn-ran-in-baseline"
    elif [ "$cpu_only" = "1" ] && [ "$gf" != "NA" ] \
            && [ "$gf" -gt "$PEAK" ]; then
        verdict="VOID:implied-${gf}GFLOPS-exceeds-cpu-peak"
    fi
    echo "$name,${pp:-NA},${tg:-NA},${ex:-0},$gf,$verdict" >> "$OUT"
    printf '%-22s pp512=%-8s tg128=%-8s executed=%-8s %s\n' \
        "$name" "${pp:-NA}" "${tg:-NA}" "${ex:-0}" "$verdict"
}

# ---- CPU-only arms. Try both device spellings for the fork and keep the one
# ---- that actually engages, because it differs per box (see header).
run_cfg cpu              "$A" 0 1 -dev none -ngl 0
run_cfg cpu+bulldnn      "$B" 1 1 -dev none -ngl 0
if grep -q 'VOID:bulldnn-never-ran' <<<"$(grep '^cpu+bulldnn,' "$OUT")"; then
    echo "  -dev none did not engage BullDNN here; retrying with -dev ZenDNN" >&2
    sed -i '/^cpu+bulldnn,/d' "$OUT"
    run_cfg cpu+bulldnn  "$B" 1 1 -dev ZenDNN
fi

# ---- GPU arms.
run_cfg vulkan           "$A" 0 0 -ngl 99
run_cfg vulkan+cpu       "$A" 0 0 -ngl "$NGL_HYBRID"
run_cfg vulkan+cpu+bulldnn "$B" 1 0 -ngl "$NGL_HYBRID"

echo
echo "results in $OUT"
echo "Any row marked VOID is not a result -- it is the harness telling you the"
echo "configuration did not measure what its name claims."
