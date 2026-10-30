#!/bin/bash
# Empty-cell control job (642465) status.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== log ==="
tail -14 "$HOME/logs/t3ctrl_642465.out" 2>/dev/null || echo "no log yet"
echo T3CTRLCHK_DONE
