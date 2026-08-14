#!/bin/bash
# Unattended validation and benchmark run for the family 15h work.
#
# Launch with:  nohup setsid ~/validate.sh >/dev/null 2>&1 </dev/null &
# Survives loss of the controlling workstation. Results land in ~/logs/<stamp>/.
#
# Replaces overnight_a10.sh, whose first phase spent 2h40 running
# BatchMatmul.F32_3D four times under a 40-minute cap. Every run timed out, for
# both the new planner and the old, which did answer the question it was asked --
# choose_even_mb is not responsible for that test being slow -- but at a cost far
# out of proportion to one bit of information. Dropping it takes this run from
# six to eight hours down to two or three.
#
# Four phases:
#   1. FP32 correctness shards, the planner and packing changes being committed
#      but only spot-checked.
#   2. BF16 native against libxsmm across enough shapes to find the crossover.
#      Five shapes measured so far give ratios from 0.86 to 1.32, so where one
#      wins over the other is currently unknown.
#   3. BF16 thread scaling. Never measured; the FP32 equivalent found +12% to
#      +113% from the same planner change.
#   4. A-packing policy. Still open: not packing edged the default by about 3%
#      where K forms a single block.
#
# Everything that compares two things runs them adjacent inside a pass. Absolute
# GFLOPS on this box drifts between passes -- libxsmm was measured at 40.19,
# 27.25 and 38.54 on one shape with no code change -- so only within-pass ratios
# are meaningful.
set -u
cd ~/BullDNN/build || exit 1

# Set VALIDATE_RESUME_DIR to an existing logdir to continue a run that a reboot
# or a lost box cut short: the driver appends rather than truncates, and phase 1
# skips shards already recorded there. Written after a crash lost fourteen of
# sixteen shards with no way to pick up from the two that had finished.
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="${VALIDATE_RESUME_DIR:-$HOME/logs/$STAMP}"
mkdir -p "$LOGDIR"
exec >> "$LOGDIR/driver.log" 2>&1

# This tree is synced file by file rather than cloned, so rev-parse finds no
# repository and every log said "head unknown" -- useless for the one question an
# unattended result has to answer later, which is what code produced it. The sync
# writes the commit to .synced_head; git stays as the fallback for a checkout.
HEAD_ID=$(cat ~/BullDNN/.synced_head 2>/dev/null \
        || git -C ~/BullDNN rev-parse --short HEAD 2>/dev/null \
        || echo unknown)

echo "start $(date -Is) host=$(hostname)"
echo "logdir $LOGDIR"
echo "head $HEAD_ID"

gflops() { awk -v m="$1" '$1==m{print $NF}' | tail -1; }

# dtype algo M K N iters warmup postop outfile
mkin() {
    local po_field="" po_dt=""
    if [ "$8" != none ]; then po_field="$8"; po_dt="f32"; fi
    printf '%s, %s, %s, %s, %s, false, , %s, %s, %s, true, false, false, 1.0, 0.0, none, 0, f32, %s\n' \
        "$3" "$4" "$5" "$6" "$1" "$po_field" "$po_dt" "$2" "$7" > "$9"
}

# ---------------------------------------------------------------------------
# 1. FP32 correctness. BatchMatmul excluded: it is the suite that ate every
#    earlier budget. Failures so far are all s8/u8 sources forced onto a kernel
#    that declines int8, so f32fail is the number that matters.
# ---------------------------------------------------------------------------
echo "=== [1] FP32 correctness shards ==="
for SH in 3 17 29 41 53 67 79 91 103 115 127 139 151 163 175 187; do
    LOG="$LOGDIR/shard_${SH}.log"
    if grep -q "^shard=$SH rc=" "$LOGDIR/driver.log" 2>/dev/null; then
        echo "skip shard=$SH, already recorded"
        continue
    fi
    T0=$(date +%s)
    GTEST_TOTAL_SHARDS=400 GTEST_SHARD_INDEX=$SH ZENDNNL_MATMUL_ALGO=10 \
        timeout 900 ./zendnnl/gtests/gtests \
        --gtest_filter="Matmul/TestMatmul.*" --seed 424242 > "$LOG" 2>&1
    RC=$?
    echo "shard=$SH rc=$RC elapsed=$(( $(date +%s) - T0 ))s ok=$(grep -cE '^\[       OK \]' "$LOG") fail=$(grep -cE '^\[  FAILED  \].*GetParam' "$LOG") f32fail=$(grep -E '^\[  FAILED  \].*GetParam' "$LOG" | grep -c 'src_dtype=f32')"
done

# ---------------------------------------------------------------------------
# 2. BF16: native against libxsmm, and both FMA flavours of native. The FMA4
#    build is otherwise unreachable here -- Piledriver onward prefers FMA3 and
#    no Intel part has FMA4 -- so this is the only place it runs.
# ---------------------------------------------------------------------------
echo "=== [2] BF16 native vs libxsmm ==="
BF="$LOGDIR/bf16_shapes.txt"
echo "shape native_fma3 native_fma4 libxsmm ratio_fma3" > "$BF"
for S in 128:128:128:400 256:256:256:200 512:512:512:100 768:768:768:60 \
         1024:1024:1024:40 2048:1024:1024:20 1024:2048:1024:20 \
         1024:1024:2048:20 333:777:555:100 100:1000:1000:100 \
         2048:512:512:40 512:2048:512:40 4096:1024:512:15 \
         64:4096:4096:20 32:1024:1024:200; do
    IFS=: read -r M K N IT <<< "$S"
    mkin bf16:bf16:bf16 native_gemm "$M" "$K" "$N" "$IT" 5 none /tmp/v_n.txt
    mkin bf16:bf16:bf16 libxsmm "$M" "$K" "$N" "$IT" 5 none /tmp/v_l.txt
    A3=$(env -u ZENDNNL_NATIVE_BF16_FMA4 OMP_NUM_THREADS=4 ./benchdnn/benchdnn \
        --op=matmul --lowoha=true --input_file=/tmp/v_n.txt 2>/dev/null | gflops "$M")
    A4=$(ZENDNNL_NATIVE_BF16_FMA4=1 OMP_NUM_THREADS=4 ./benchdnn/benchdnn \
        --op=matmul --lowoha=true --input_file=/tmp/v_n.txt 2>/dev/null | gflops "$M")
    LX=$(OMP_NUM_THREADS=4 ./benchdnn/benchdnn \
        --op=matmul --lowoha=true --input_file=/tmp/v_l.txt 2>/dev/null | gflops "$M")
    R=$(echo "${A3:-0} ${LX:-0}" | awk '{if ($2>0) printf "%.2f", $1/$2; else print "-"}')
    echo "${M}x${K}x${N} ${A3:--} ${A4:--} ${LX:--} $R" >> "$BF"
    echo "swept bf16 ${M}x${K}x${N}"
done

# ---------------------------------------------------------------------------
# 3. BF16 thread scaling. The FPU is shared per compute unit on this family, so
#    scaling tracks modules rather than cores: a second thread on the same
#    module measured 1.10x on FP32 against 1.87x for one thread on each of two.
#    Worth knowing whether BF16 behaves the same.
# ---------------------------------------------------------------------------
echo "=== [3] BF16 thread scaling ==="
TS="$LOGDIR/bf16_threads.txt"
echo "nt shape native libxsmm" > "$TS"
for NT in 1 2 4; do
    for S in 1024:1024:1024:30 333:777:555:60 2048:1024:1024:15; do
        IFS=: read -r M K N IT <<< "$S"
        mkin bf16:bf16:bf16 native_gemm "$M" "$K" "$N" "$IT" 5 none /tmp/v_n.txt
        mkin bf16:bf16:bf16 libxsmm "$M" "$K" "$N" "$IT" 5 none /tmp/v_l.txt
        A=$(OMP_NUM_THREADS=$NT ./benchdnn/benchdnn --op=matmul --lowoha=true \
            --input_file=/tmp/v_n.txt 2>/dev/null | gflops "$M")
        L=$(OMP_NUM_THREADS=$NT ./benchdnn/benchdnn --op=matmul --lowoha=true \
            --input_file=/tmp/v_l.txt 2>/dev/null | gflops "$M")
        echo "$NT ${M}x${K}x${N} ${A:--} ${L:--}" >> "$TS"
        echo "scaled nt=$NT ${M}x${K}x${N}"
    done
done

# ---------------------------------------------------------------------------
# 4. A-packing policy. The default packs once lda passes 4096 bytes, a threshold
#    its own comment attributes to the Zen4/5 L1 stride prefetcher. Forced on
#    against forced off, so the threshold can be set from data.
# ---------------------------------------------------------------------------
echo "=== [4] FP32 A-packing policy ==="
PK="$LOGDIR/packing.txt"
echo "nt M K N packOFF packON default" > "$PK"
for NT in 4 2; do
    for S in 1024:1025:1024:120 1024:2048:1024:60 1024:4096:1024:30 \
             1024:8192:1024:15 512:4096:512:60 2048:4096:512:20; do
        IFS=: read -r M K N IT <<< "$S"
        mkin f32:f32:f32 native_gemm "$M" "$K" "$N" "$IT" 10 none /tmp/v_p.txt
        OFF=$(ZENDNNL_NATIVE_GEMM_PACK_A=0 OMP_NUM_THREADS=$NT ./benchdnn/benchdnn \
            --op=matmul --lowoha=true --input_file=/tmp/v_p.txt 2>/dev/null | gflops "$M")
        ON=$(ZENDNNL_NATIVE_GEMM_PACK_A=1 OMP_NUM_THREADS=$NT ./benchdnn/benchdnn \
            --op=matmul --lowoha=true --input_file=/tmp/v_p.txt 2>/dev/null | gflops "$M")
        DEF=$(env -u ZENDNNL_NATIVE_GEMM_PACK_A OMP_NUM_THREADS=$NT ./benchdnn/benchdnn \
            --op=matmul --lowoha=true --input_file=/tmp/v_p.txt 2>/dev/null | gflops "$M")
        echo "$NT $M $K $N ${OFF:--} ${ON:--} ${DEF:--}" >> "$PK"
        echo "packed nt=$NT ${M}x${K}x${N}"
    done
done

echo "done $(date -Is)"
echo "RESULTS IN $LOGDIR"
