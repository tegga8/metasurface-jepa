#!/bin/bash
L=$(ls -t "$HOME"/logs/surrtransfer_*.out 2>/dev/null | head -1)
echo "log: $L"
cat "$L" 2>/dev/null | tail -30
echo SURRTCHK_DONE
