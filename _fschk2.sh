#!/bin/bash
# The two full-split jobs (642373/642374): queue state + their logs.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== fs_imp1 (642373) ==="
tail -6 "$HOME/logs/fs_642373.out" 2>/dev/null || echo "no log"
echo "=== fs_imp2 (642374) ==="
tail -6 "$HOME/logs/fs_642374.out" 2>/dev/null || echo "no log"
echo "FSCHK2_DONE"
