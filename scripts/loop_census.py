#!/usr/bin/env python3
"""Count what a microkernel's innermost loop actually issues.

Source review cannot see a spill. Two of this project's real findings -- the INT8
accumulators living on the stack, and load folding in the FP32 kernel -- were
invisible until someone counted instructions between the branch targets, so this
is the instrument for that.

  loop_census.py <object-file> <symbol-substring> [--loops N]

Prints, for the innermost backward branch (and optionally the next few), the
instruction count, how many are vector ops, and how much stack traffic there is.
Stack traffic in a compute loop is a spill; zero is what a tuned kernel looks
like.
"""
import re
import subprocess
import sys

STACK = re.compile(r"\(%rsp\)")
SPILL_ST = re.compile(r"v?mov\w*\s+%[xy]mm[0-9]+,-?0x[0-9a-f]+\(%rsp\)")
SPILL_LD = re.compile(r"v?mov\w*\s+-?0x[0-9a-f]+\(%rsp\),%[xy]mm[0-9]+")
REGMOV = re.compile(r"v?mov(?:dqa|aps|dqu|ups)\s+%[xy]mm[0-9]+,%[xy]mm[0-9]+")


def loops_of(func_text):
    rows = []
    for line in func_text.split("\n"):
        m = re.match(r"\s*([0-9a-f]+):\t(.*)", line)
        if m:
            rows.append((int(m.group(1), 16), m.group(2).strip()))
    back = set()
    for addr, text in rows:
        m = re.match(r"j\w+\s+([0-9a-f]+)", text)
        if m:
            tgt = int(m.group(1), 16)
            if tgt < addr:
                back.add((tgt, addr))
    return rows, sorted(back, key=lambda x: x[1] - x[0])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    obj, want = sys.argv[1], sys.argv[2]
    n_loops = 1
    if "--loops" in sys.argv:
        n_loops = int(sys.argv[sys.argv.index("--loops") + 1])

    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", obj],
                         capture_output=True, text=True).stdout
    for func in re.split(r"\n(?=[0-9a-f]+ <)", dis):
        head = func.split("\n")[0]
        if want not in head or ">:" not in head:
            continue
        rows, loops = loops_of(func)
        if not loops:
            print("%-28s no backward branch" % want)
            return 0
        for lo, hi in loops[:n_loops]:
            body = [t for a, t in rows if lo <= a <= hi]
            vec = [t for t in body if re.match(r"v?p|v", t)]
            print("%-28s loop 0x%x-0x%x: %3d insns, %3d vector, "
                  "spill_st=%d spill_ld=%d reg_mov=%d stack_any=%d"
                  % (want, lo, hi, len(body), len(vec),
                     sum(1 for t in body if SPILL_ST.search(t)),
                     sum(1 for t in body if SPILL_LD.search(t)),
                     sum(1 for t in body if REGMOV.search(t)),
                     sum(1 for t in body if STACK.search(t))))
        return 0
    print("%-28s symbol not found in %s" % (want, obj))
    return 1


if __name__ == "__main__":
    sys.exit(main())
