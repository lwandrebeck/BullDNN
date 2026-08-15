#!/bin/bash
# Cooled, alternating A/B for llama.cpp on the A10-8770E.
#
# WHY THIS EXISTS, and why a plain back-to-back A/B on this box is worthless:
# the A10-8770E is a 35W part that cannot hold a 4-thread SIMD load. The same
# binary measures 40.11 t/s hot and 84.36 t/s after five minutes idle. So an
# "A then B" run hands A the cold pass and B the hot one and manufactures a 2x
# difference out of nothing. That is not hypothetical -- it produced a fictitious
# 2x Q4_K "regression" in docs/perf/llamacpp_b10437.csv that stood for a day.
#
# The effect is 2x while the clock drops only ~9%, because ggml's barriers
# spin-wait: throttling is not symmetric across the two modules, so threads stop
# arriving at the barrier together and the early ones burn PAUSE loops. Retired
# instructions nearly double for bit-identical page-faults and cache-misses.
# Lowering the thread count does not help -- 2, 3 and 4 all collapse.
#
# So: idle before EVERY pass, and alternate the arms. Read the two arms' means,
# and if two runs of the same arm disagree by more than a couple of percent,
# throw the batch away rather than reporting it.
#
# Usage: a10_cooled_ab.sh <model.gguf> <baseline-build-dir> <bulldnn-build-dir>
#        (build dirs are llama.cpp build trees containing bin/llama-bench)
set -u
MODEL=${1:?model gguf}
BIN_A=${2:?baseline build dir}
BIN_B=${3:?bulldnn build dir}
PAIRS=${PAIRS:-3}
COOL=${COOL:-150}
THREADS=${THREADS:-4}
OUT=${OUT:-/tmp/a10_cooled_ab.csv}
TEMP=/sys/class/hwmon/hwmon1/temp1_input   # k10temp

echo "run,arm,pp128,tg32,instructions,temp_before_mC" > "$OUT"

run_one() {
    local run=$1 arm=$2 bin=$3
    sleep "$COOL"
    local tb; tb=$(cat $TEMP 2>/dev/null || echo NA)
    perf stat -x, -e instructions -o "/tmp/ab_perf_${run}.txt" \
        "$bin/bin/llama-bench" -m "$MODEL" -p 128 -n 32 -t "$THREADS" -ngl 0 -r 3 \
        > "/tmp/ab_log_${run}.txt" 2>/dev/null
    local pp tg ins
    pp=$(grep -oE 'pp128[^|]*\|[^|]*' "/tmp/ab_log_${run}.txt" | tail -1 | awk -F'|' '{print $2}' | tr -d ' ' | sed 's/±.*//')
    tg=$(grep -oE 'tg32[^|]*\|[^|]*'  "/tmp/ab_log_${run}.txt" | tail -1 | awk -F'|' '{print $2}' | tr -d ' ' | sed 's/±.*//')
    ins=$(grep ',instructions,' "/tmp/ab_perf_${run}.txt" | head -1 | cut -d, -f1)
    echo "$run,$arm,$pp,$tg,$ins,$tb" >> "$OUT"
}

for i in $(seq 1 "$PAIRS"); do
    run_one "${i}a" baseline "$BIN_A"
    run_one "${i}b" bulldnn  "$BIN_B"
done
echo "results in $OUT"
