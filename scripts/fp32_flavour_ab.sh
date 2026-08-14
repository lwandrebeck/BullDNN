#!/bin/bash
# A/B the FP32 microkernel's multiply-add form and register pinning on this host.
#
# ZENDNNL_FORCE_FMA is compile-time and only one translation unit reads it, so
# each variant recompiles that one object with extra flags, swaps it into the
# archive, and relinks the bench. Rebuilding the whole library three times would
# take twenty minutes and change nothing else that matters.
#
# The FMA4 arm needs -mno-fma as well: on a part with both forms GCC lowers
# _mm_macc_ps to vfmadd231ps, so without it the "FMA4" arm silently measures FMA3
# and the comparison reads as a tie.
set -eu
cd "$(dirname "$0")/.."  # repo root
ARCH=${ARCH:-bdver2}
REPS=${REPS:-7}
THREADS=${THREADS:-8}
BUILD=build/zendnnl
UK=zendnnl/src/lowoha_operators/matmul/matmul_native/gemm/kernel/fp32/fp32_gemm_ukernel_128.cpp
OBJ=$BUILD/src/CMakeFiles/zendnnl_archive.dir/lowoha_operators/matmul/matmul_native/gemm/kernel/fp32/fp32_gemm_ukernel_128.cpp.o
ARCHIVE=$BUILD/src/libzendnnl_archive.a
D=build/install/deps

cp "$OBJ" /tmp/fp32_uk_pristine.o
cp "$ARCHIVE" /tmp/libzendnnl_pristine.a

INC="-I zendnnl/src -I $BUILD -isystem $D/aoclutils/include -isystem $D/json/include \
 -isystem $D/onednn/include -isystem $D/libxsmm/include -isystem $D/fbgemm/include"
[ -d "$D/aocldlp/include" ] && INC="$INC -isystem $D/aocldlp/include"

# Debian puts these in lib/, Fedora in lib64/, so resolve rather than assume.
lib_of() { # dep, filename
  local hit
  hit=$(find "$D/$1" -name "$2" 2>/dev/null | head -1)
  [ -n "$hit" ] || { echo "missing $1/$2" >&2; exit 1; }
  echo "$hit"
}
LIBS="$ARCHIVE -Wl,--push-state,--whole-archive $(lib_of aoclutils libaoclutils.a)"
if [ -d "$D/aocldlp" ]; then
  LIBS="$LIBS $(lib_of aocldlp libaocl-dlp.a)"
fi
LIBS="$LIBS $(lib_of onednn libdnnl.a) -Wl,--pop-state -ldl \
 -Wl,--push-state,--whole-archive $(lib_of libxsmm libxsmm.a) $(lib_of fbgemm libfbgemm.a) \
 -Wl,--pop-state $(lib_of fbgemm libcpuinfo.a) $(lib_of fbgemm libasmjit.a) -lrt"

BASE_FLAGS="-O3 -march=$ARCH -std=c++17 -fopenmp -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 \
 -DZENDNNL_DEPENDS_AOCLUTILS=1 -DZENDNNL_DEPENDS_JSON=1 -DZENDNNL_DEPENDS_ONEDNN=1 \
 -DZENDNNL_DEPENDS_LIBXSMM=1 -DZENDNNL_DEPENDS_FBGEMM=1 -DZENDNNL_DEPENDS_PARLOOPER=0"
if grep -q "ZENDNNL_DEPENDS_AOCLDLP:BOOL=ON" build/CMakeCache.txt 2>/dev/null; then
  BASE_FLAGS="$BASE_FLAGS -DZENDNNL_DEPENDS_AOCLDLP=1"
else
  BASE_FLAGS="$BASE_FLAGS -DZENDNNL_DEPENDS_AOCLDLP=0"
fi

run_variant() { # name, extra compile flags, source override
  local name="$1" extra="$2" src="${3:-$UK}"
  g++ $BASE_FLAGS $extra $INC -c "$src" -o /tmp/uk_$name.o
  cp /tmp/libzendnnl_pristine.a "$ARCHIVE"
  ar r "$ARCHIVE" /tmp/uk_$name.o 2>/dev/null
  # ar matches on basename, so rename to the object the archive already holds.
  cp /tmp/uk_$name.o /tmp/fp32_gemm_ukernel_128.cpp.o
  ar r "$ARCHIVE" /tmp/fp32_gemm_ukernel_128.cpp.o
  g++ -O3 -march=$ARCH -std=c++17 -fopenmp -DNDEBUG $INC \
      scratch/fp32_flavour_bench.cpp $LIBS -o scratch/fp32_bench_$name
  echo "== $name =="
  ./scratch/fp32_bench_$name "$name" "$REPS" "$THREADS"
}

# The pinning arm needs a source edit, not a define: strip the asm barrier.
sed 's|    asm("" : "+x"(v0), "+x"(v1));|    (void)v0; (void)v1; /* pinning disabled for A/B */|' \
    "$UK" > /tmp/fp32_uk_nopin.cpp

run_variant fma3   "-DZENDNNL_FORCE_FMA=3"
run_variant fma4   "-DZENDNNL_FORCE_FMA=4 -mno-fma"
run_variant muladd "-DZENDNNL_FORCE_FMA=0 -ffp-contract=off"
run_variant nopin  "" /tmp/fp32_uk_nopin.cpp

# Leave the tree as it was found.
cp /tmp/libzendnnl_pristine.a "$ARCHIVE"
echo "archive restored"
