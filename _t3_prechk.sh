#!/bin/bash
# Precondition check for the task-3 window re-run: the four npz sets exist and
# the fixed-window harness is present in the checkout.
cd "$HOME/metasurface-jepa" || exit 1
echo "=== harness ==="
grep -c "ring_fixed" _meep_novel.py
echo "=== npz sets ==="
for TAG in jepa_v1_spec_m20 jepa_v1_spec_p20 jepa_v2_spec_m20 jepa_v2_spec_p20; do
  f="$HOME/runs/phase3/out/$TAG.npz"
  if [ -f "$f" ]; then echo "OK   $TAG  $(du -h "$f" | cut -f1)"; else echo "MISS $f"; fi
done
echo "=== disk ==="
df -h "$HOME" | tail -1
echo "T3PRECHK DONE"
