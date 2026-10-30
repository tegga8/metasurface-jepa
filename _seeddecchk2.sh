#!/bin/bash
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== full decode log ==="
cat "$HOME/logs/seeddec_657838.out" 2>/dev/null
echo SEEDDECCHK2_DONE
