#!/bin/sh
# Phase-separated report. This perf wants absolute timestamps, not percentages,
# so read the first and last sample time out of perf script and slice from there.
# The prompt and the model load are all at the front, so a tail window is decode.
D=/tmp/perf_dec.data

sudo -n perf script -i "$D" -F time 2>/dev/null | tr -d ' :' | grep -E '^[0-9]' > /tmp/pt.txt
FIRST=$(head -1 /tmp/pt.txt)
LAST=$(tail -1 /tmp/pt.txt)
echo "span: $FIRST .. $LAST"

CUT=$(awk -v a="$FIRST" -v b="$LAST" 'BEGIN{printf "%.6f", a + 0.25*(b-a)}')
echo "decode window starts at $CUT"
echo

for w in "$FIRST,$LAST" "$CUT,$LAST"; do
    case "$w" in
        "$FIRST,$LAST") echo "=== WHOLE RUN ===" ;;
        *) echo "=== DECODE ONLY (last 75%) ===" ;;
    esac
    sudo -n perf report -i "$D" --stdio --no-children --sort=symbol \
        --time "$w" --percent-limit 0.4 2>/dev/null \
      | grep -E '^ +[0-9]+\.[0-9]+%' \
      | sed 's/\[\.\] //; s/(anonymous namespace):://g; s/zendnnl::lowoha::matmul::native:://g; s/zendnnl::lowoha:://g' \
      | cut -c1-96 | head -12
    echo
done
