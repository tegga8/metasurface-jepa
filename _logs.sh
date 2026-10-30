#!/bin/bash
# Full logs for the two silent jobs.
echo "=== gate8scaleCPU_642602.out ==="
cat "$HOME/logs/gate8scaleCPU_642602.out" 2>/dev/null | tail -14
echo "=== t3res64_642594.out ==="
cat "$HOME/logs/t3res64_642594.out" 2>/dev/null | tail -8
echo "=== t3res64 run log ==="
tail -4 "$HOME/runs/t3res/logs/res64_k3.log" 2>/dev/null
echo "=== t3res final design log ==="
tail -3 "$HOME/runs/t3res/logs/res48_k0.log" 2>/dev/null
echo LOGS_DONE
