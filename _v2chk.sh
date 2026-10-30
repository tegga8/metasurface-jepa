#!/bin/bash
# v2 identity + gate 8 (642741).
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== log ==="
grep -E "^=== |primary|MAE|AAE|gate|exit=|pooled win|GATE 8|PASS|FAIL" \
    "$HOME/logs/v2id_642741.out" 2>/dev/null | tail -20
echo "=== json ==="
ls -l "$HOME/runs/v2id/"*.json "$HOME/runs/gate8/v2_schedfix.json" 2>/dev/null
echo V2CHK_DONE
