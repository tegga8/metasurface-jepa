#!/bin/bash
# Are the jobs being suspended? Elapsed time is barely advancing between checks.
echo "=== scontrol ==="
for J in 642476 642594 642602; do
  echo "--- $J"
  scontrol show job "$J" 2>/dev/null | tr ' ' '\n' | grep -E "^(JobState|RunTime|SuspendTime|TimeLimit|NodeList|Reason)=" | head -6
done
echo "=== log mtimes (progress) ==="
ls -l --time-style=+%H:%M:%S "$HOME"/logs/t3res_642476.out "$HOME"/logs/t3res64_642594.out "$HOME"/logs/gate8scaleCPU_642602.out 2>/dev/null | awk '{print $6, $7}'
echo "=== newest artefacts ==="
ls -lt --time-style=+%H:%M:%S "$HOME"/runs/t3res/*.json "$HOME"/runs/gate8/*.json 2>/dev/null | head -4 | awk '{print $6, $7}'
echo SUSCHK_DONE
