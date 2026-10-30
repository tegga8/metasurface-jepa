#!/bin/bash
L=$(ls -t "$HOME"/logs/imp2score2_*.out 2>/dev/null | head -1)
echo "log: $L"
cat "$L" 2>/dev/null | tail -45
echo IMP2SCORE2CHK_DONE
