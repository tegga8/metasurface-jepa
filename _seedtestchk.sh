#!/bin/bash
# Seed-test progress: queue, log lines, per-set file counts.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== log ==="
L=$(ls -t "$HOME"/logs/seedtest_*.out 2>/dev/null | head -1)
echo "log: $L"
grep -E "^=== SET|srun exit|already complete|SEEDTEST DONE" "$L" 2>/dev/null | tail -12
echo "=== files per set ==="
for d in "$HOME"/runs/seedtest/meep/*/; do
  [ -d "$d" ] || continue
  echo "  $(basename "$d"): $(ls "$d"/e_0_*.json 2>/dev/null | wc -l)/18"
done
echo SEEDTESTCHK_DONE
