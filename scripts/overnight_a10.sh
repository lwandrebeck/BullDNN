#!/bin/bash
# Detached overnight validation run for the family 15h native GEMM work.
#
# Launch with:   nohup setsid ~/overnight.sh >/dev/null 2>&1 </dev/null &
# It keeps running if the controlling workstation goes away. Everything lands
# in ~/logs/<stamp>/ on the machine that runs it.
#
# Three phases, in order of what is most worth knowing:
#   1. BatchMatmul.F32_3D new vs old planner -- the open regression question.
#   2. f32 correctness shards with BatchMatmul excluded, since that suite is
#      what exhausted every earlier budget.
#   3. A wide shape sweep at 1, 2 and 4 threads, new vs old.
set -u
cd ~/BullDNN/build || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$HOME/logs/$STAMP"
mkdir -p "$LOGDIR"
exec > "$LOGDIR/driver.log" 2>&1

echo "start $(date -Is) host=$(hostname)"
echo "logdir $LOGDIR"

gflops() { awk -v m="$1" '$1==m{print $NF}' | tail -1; }

mkin() {
    # M K N iters warmup outfile
    printf '%s, %s, %s, %s, f32:f32:f32, false, , , , native_gemm, true, false, false, 1.0, 0.0, none, 0, f32, %s\n' \
        "$1" "$2" "$3" "$4" "$5" > "$6"
}

# --------------------------------------------------------------------------
# 1. BatchMatmul.F32_3D: is the >420s runtime caused by choose_even_mb?
#    Earlier attempts never got a clean answer -- the new side timed out and
#    the old side was killed by hand. 40 minutes per side settles it.
# --------------------------------------------------------------------------
echo "=== [1] BatchMatmul.F32_3D new vs old ==="
for MODE in new old; do
    for IDX in 11 23; do
        LOG="$LOGDIR/b3d_${MODE}_${IDX}.log"
        T0=$(date +%s)
        if [ "$MODE" = new ]; then
            env -u ZENDNNL_NATIVE_GEMM_NO_EVEN_MB ZENDNNL_MATMUL_ALGO=10 \
                timeout 2400 ./zendnnl/gtests/gtests \
                --gtest_filter="BatchMatmul/TestBatchMatmul.F32_3D/$IDX" \
                --seed 424242 > "$LOG" 2>&1
        else
            ZENDNNL_NATIVE_GEMM_NO_EVEN_MB=1 ZENDNNL_MATMUL_ALGO=10 \
                timeout 2400 ./zendnnl/gtests/gtests \
                --gtest_filter="BatchMatmul/TestBatchMatmul.F32_3D/$IDX" \
                --seed 424242 > "$LOG" 2>&1
        fi
        RC=$?
        T1=$(date +%s)
        OK=$(grep -cE '^\[       OK \]' "$LOG")
        echo "b3d mode=$MODE idx=$IDX elapsed=$((T1 - T0))s rc=$RC ok=$OK"
    done
done

# --------------------------------------------------------------------------
# 2. f32 correctness. Matmul/TestMatmul only: BatchMatmul contains the
#    pathologically slow cases and would eat the whole night.
#    Every failure so far has been an s8/u8 source forced onto a kernel that
#    declines int8, so f32fail is the number that matters.
# --------------------------------------------------------------------------
echo "=== [2] correctness shards (Matmul/TestMatmul, algo 10 forced) ==="
for SH in 3 17 29 41 53 67 79 91 103 115 127 139 151 163 175 187; do
    LOG="$LOGDIR/shard_${SH}.log"
    T0=$(date +%s)
    GTEST_TOTAL_SHARDS=400 GTEST_SHARD_INDEX=$SH ZENDNNL_MATMUL_ALGO=10 \
        timeout 1800 ./zendnnl/gtests/gtests \
        --gtest_filter="Matmul/TestMatmul.*" --seed 424242 > "$LOG" 2>&1
    RC=$?
    T1=$(date +%s)
    OK=$(grep -cE '^\[       OK \]' "$LOG")
    FAIL=$(grep -cE '^\[  FAILED  \].*GetParam' "$LOG")
    F32=$(grep -E '^\[  FAILED  \].*GetParam' "$LOG" | grep -c 'src_dtype=f32')
    echo "shard=$SH rc=$RC elapsed=$((T1 - T0))s ok=$OK fail=$FAIL f32fail=$F32"
done

# --------------------------------------------------------------------------
# 3. Wide shape sweep, new vs old, three thread counts. New and old run
#    adjacent inside each pass because absolute GFLOPS drifts between passes
#    on this box while within-pass ratios stay stable.
# --------------------------------------------------------------------------
echo "=== [3] wide shape sweep ==="
SWEEP="$LOGDIR/sweep.txt"
echo "nt shape | new x3 | old x3" > "$SWEEP"
for NT in 4 2 1; do
    for S in 1024:1024:1024:150 512:512:512:300 333:777:555:300 \
             2048:512:512:200 1024:1024:64:400 100:1000:1000:300 \
             128:1024:1024:300 4096:1024:1024:60 768:768:3072:100 \
             3072:768:768:100 32:4096:4096:150 2048:2048:2048:40; do
        IFS=: read -r M K N IT <<< "$S"
        F="/tmp/ov_${M}_${K}_${N}.txt"
        mkin "$M" "$K" "$N" "$IT" 30 "$F"
        NEW=""
        OLD=""
        for P in 1 2 3; do
            A=$(env -u ZENDNNL_NATIVE_GEMM_NO_EVEN_MB OMP_NUM_THREADS=$NT \
                ./benchdnn/benchdnn --op=matmul --lowoha=true \
                --input_file="$F" 2>/dev/null | gflops "$M")
            B=$(ZENDNNL_NATIVE_GEMM_NO_EVEN_MB=1 OMP_NUM_THREADS=$NT \
                ./benchdnn/benchdnn --op=matmul --lowoha=true \
                --input_file="$F" 2>/dev/null | gflops "$M")
            NEW="$NEW $A"
            OLD="$OLD $B"
        done
        echo "$NT ${M}x${K}x${N} |$NEW |$OLD" >> "$SWEEP"
        echo "swept nt=$NT ${M}x${K}x${N}"
    done
done

# --------------------------------------------------------------------------
# 4. A-packing policy. The repack fix made packing free at small lda but it
#    still loses badly for K >= 4096, where KB=K makes the packed block far
#    larger than any cache. The default packs whenever lda > 4096 bytes, i.e.
#    every K > 1024 in f32. This sweeps forced-on against forced-off over a
#    range of lda and shapes so the threshold can be set from data.
# --------------------------------------------------------------------------
echo "=== [4] A-packing policy sweep ==="
PACK="$LOGDIR/packing.txt"
echo "nt M K N | packOFF x3 | packON x3" > "$PACK"
for NT in 4 2; do
    for S in 1024:1025:1024:120 1024:2048:1024:60 1024:4096:1024:30 \
             1024:8192:1024:15 1024:16384:1024:8 512:4096:512:60 \
             2048:4096:512:20 256:8192:2048:15; do
        IFS=: read -r M K N IT <<< "$S"
        F="/tmp/pk_${M}_${K}_${N}.txt"
        mkin "$M" "$K" "$N" "$IT" 10 "$F"
        OFF=""
        ON=""
        for P in 1 2 3; do
            A=$(ZENDNNL_NATIVE_GEMM_PACK_A=0 OMP_NUM_THREADS=$NT \
                ./benchdnn/benchdnn --op=matmul --lowoha=true \
                --input_file="$F" 2>/dev/null | gflops "$M")
            B=$(ZENDNNL_NATIVE_GEMM_PACK_A=1 OMP_NUM_THREADS=$NT \
                ./benchdnn/benchdnn --op=matmul --lowoha=true \
                --input_file="$F" 2>/dev/null | gflops "$M")
            OFF="$OFF $A"
            ON="$ON $B"
        done
        echo "$NT $M $K $N |$OFF |$ON" >> "$PACK"
        echo "packed nt=$NT ${M}x${K}x${N}"
    done
done

echo "done $(date -Is)"
echo "RESULTS IN $LOGDIR"
