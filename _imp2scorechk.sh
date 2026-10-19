#!/bin/bash
# imp2 scoring job (645722): full log.
L=$(ls -t "$HOME"/logs/imp2score_*.out 2>/dev/null | head -1)
echo "log: $L"
cat "$L" 2>/dev/null | tail -60
echo IMP2SCORECHK_DONE
