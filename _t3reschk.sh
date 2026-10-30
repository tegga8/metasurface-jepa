#!/bin/bash
# Task-3 resolution test (642476): progress + headline numbers.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== files ==="
ls "$HOME/runs/t3res"/res*_k*.json 2>/dev/null | wc -l
echo "=== log tail ==="
tail -8 "$HOME/logs/t3res_642476.out" 2>/dev/null || echo "no log"
echo T3RESCHK_DONE
