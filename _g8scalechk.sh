#!/bin/bash
# Scaling gate-8 job (642579) + resolution test (642476) status.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== gate8 scale log ==="
grep -E "===|win rate|pooled|GATE 8|PASS|FAIL|Error|Traceback|exit=" \
    "$HOME/logs/gate8scale_642579.out" 2>/dev/null | tail -20
echo "=== gate8 json ==="
ls -l "$HOME/runs/gate8/"*scale*.json 2>/dev/null
echo "=== t3res ==="
ls "$HOME/runs/t3res"/res48*.json 2>/dev/null
tail -3 "$HOME/logs/t3res_642476.out" 2>/dev/null
echo CHK_DONE
