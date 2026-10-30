#!/bin/bash
# Inputs for the imp2 phase-2/3 run.
echo "=== ~/runs/phase3 contents ==="
ls -l "$HOME/runs/phase3" 2>/dev/null | head -12
echo "=== existing sets (out/) ==="
ls "$HOME/runs/phase3/out" 2>/dev/null | head -12
echo "=== feasible spectra anywhere ==="
ls -l "$HOME"/runs/p2/*.npz 2>/dev/null | head -6
find "$HOME/runs" -maxdepth 2 -name "*feasib*" 2>/dev/null | head -5
echo "=== driver + ckpt ==="
ls -l "$HOME/imp2/_phase3_models.py" "$HOME/imp2/runs_out/final.pt" 2>/dev/null
echo "=== b40 targets ==="
ls -l "$HOME/runs/phase3/b40_merged.json" "$HOME/runs/b40_merged.json" 2>/dev/null
echo P23PRE_DONE
