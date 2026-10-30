#!/bin/bash
L=$(ls -t "$HOME"/logs/seeddec_*.out 2>/dev/null | head -1)
echo "log: $L"
tail -30 "$L" 2>/dev/null
echo === dec dir ===
ls "$HOME/runs/seedtest/dec" 2>/dev/null | head -13
echo SEEDDECCHK_DONE
