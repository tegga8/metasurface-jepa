#!/bin/bash
# Gate-8 job status: queue, headline lines from the log, saved JSONs.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== gate8 log (headline lines) ==="
grep -E "===|win rate|GATE 8|PASS|FAIL" "$HOME/logs/gate8_642386.out" 2>/dev/null | tail -20
echo "=== saved JSONs ==="
ls -l "$HOME/runs/gate8/" 2>/dev/null
echo "G8CHK_DONE"
