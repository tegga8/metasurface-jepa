#!/bin/bash
# S0 gate-8 (642697) result + queue.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== S0 gate8 log ==="
grep -E "gates 8|^  seed|win-rate|pooled win|GATE 8|PASS|FAIL|exit=" \
    "$HOME/logs/gate8s0_642697.out" 2>/dev/null | tail -18
echo "=== json ==="
ls -l "$HOME/runs/gate8/scale_s0.json" 2>/dev/null
echo S0CHK_DONE
