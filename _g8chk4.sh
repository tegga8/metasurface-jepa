#!/bin/bash
# Gate-8c (642394) status: headline lines + saved JSONs.
LOG="$HOME/logs/gate8_642394.out"
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== headline lines ==="
if [ -f "$LOG" ]; then
  grep -E "start |win-rate|pooled win|GATE 8|PASS|FAIL|Error|exit=" "$LOG" | tail -24
else
  echo "no log at $LOG"
fi
echo "=== saved JSONs ==="
ls -l "$HOME/runs/gate8/" 2>/dev/null
echo "G8CHK4_DONE"
