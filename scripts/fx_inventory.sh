#!/bin/sh
# Full gtest inventory on the FX at a known commit, so there is finally a
# complete baseline rather than the targeted subsets used so far.
set -u
cd "$HOME/BullDNN/build" || exit 1

sudo -n cpupower frequency-set -g performance >/dev/null 2>&1

make -j8 zendnnl > /tmp/inv_build.log 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED" > /tmp/inventory.status; touch /tmp/inventory.done; exit 1; fi

cd zendnnl/gtests || exit 1
./gtests > /tmp/inventory.log 2>&1
echo "gtests exit=$?" > /tmp/inventory.status

{
    echo "commit: $(cat "$HOME/BullDNN/.synced_head")"
    echo "ran:    $(grep -c '^\[ RUN' /tmp/inventory.log)"
    echo "passed: $(grep -c '^\[       OK \]' /tmp/inventory.log)"
    echo "failed: $(grep -cE '^\[  FAILED  \] [A-Za-z].*ms\)$' /tmp/inventory.log)"
    echo
    echo "--- failures by suite ---"
    grep -E '^\[  FAILED  \] [A-Za-z].*ms\)$' /tmp/inventory.log \
        | sed 's/^\[  FAILED  \] //; s/[.\/].*//' | sort | uniq -c | sort -rn
} >> /tmp/inventory.status 2>&1

touch /tmp/inventory.done
