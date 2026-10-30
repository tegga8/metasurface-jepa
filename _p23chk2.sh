#!/bin/bash
# Latest p23decode log + produced sets.
L=$(ls -t "$HOME"/logs/p23decode_*.out 2>/dev/null | head -1)
echo "log: $L"
tail -22 "$L" 2>/dev/null
echo "=== sets ==="
ls -l "$HOME/runs/phase3f/out" 2>/dev/null | head -12
echo P23CHK2_DONE
