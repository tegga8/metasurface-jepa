#!/bin/bash
# Decode job (645603): status, produced sets, any error.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== log ==="
tail -24 "$HOME/logs/p23decode_645603.out" 2>/dev/null || echo "no log"
echo "=== sets ==="
ls -l "$HOME/runs/phase3f/out" 2>/dev/null | head -12
echo P23CHK_DONE
