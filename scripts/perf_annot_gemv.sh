#!/bin/sh
# gemv_xop::run is only reached at M=1, so every sample in it is decode -- no
# time window needed. Group the hot instructions by mnemonic to see what class of
# work dominates rather than which single address.
D=/tmp/perf_dec.data
S='zendnnl::lowoha::matmul::native::(anonymous namespace)::gemv_xop::run(int, int, bool, signed char const*, void const*, float*, float const*, int, int const*, int) [clone ._omp_fn.0]'

sudo -n perf annotate -i "$D" --stdio -s "$S" 2>/dev/null \
  | grep -E '^ +[0-9]+\.[0-9]+ :' > /tmp/annot.txt
echo "annotated lines: $(wc -l < /tmp/annot.txt)"
echo
echo "=== share by instruction mnemonic ==="
awk '{pct=$1; for(i=1;i<=NF;i++) if($i ~ /^[0-9a-f]+:$/){print pct, $(i+1); break}}' /tmp/annot.txt \
  | awk '{s[$2]+=$1} END{for(k in s) printf "%7.2f  %s\n", s[k], k}' | sort -rn | head -18
echo
echo "=== hottest single instructions ==="
sort -rn /tmp/annot.txt | head -12 | sed 's/^ *//'
