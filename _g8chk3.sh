#!/bin/bash
# Correct log for the gate8b run + the exact error lines.
echo "=== log file ==="
ls -l "$HOME/logs/" | grep -i gate8 | tail -5
echo "=== headline lines (642391) ==="
grep -E "===|win rate|GATE 8|PASS|FAIL" "$HOME/logs/gate8_642391.out" 2>/dev/null | tail -20
echo "=== error lines (642391) ==="
grep -E "Error|error|exit=" "$HOME/logs/gate8_642391.out" 2>/dev/null | tail -8
echo "G8CHK3_DONE"
