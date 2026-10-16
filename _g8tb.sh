#!/bin/bash
# Full traceback from the scaling gate-8 attempt.
L=$(ls -t "$HOME"/logs/gate8scale_*.out 2>/dev/null | head -1)
echo "file: $L"
sed -n '1,60p' "$L" 2>/dev/null
echo TB_DONE
