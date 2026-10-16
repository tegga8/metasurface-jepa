#!/bin/bash
# Verify the staged scaling checkpoints (sizes must match 59 / 123 / 278 MB) and
# clean the stray overwritten copy.
rm -f "$HOME/seed0.pt"
echo "=== staged ==="
ls -l "$HOME/scale_ckpts"/*.pt 2>/dev/null
echo "=== md5 (first 8 hex) ==="
for f in "$HOME"/scale_ckpts/*.pt; do
  printf "%s " "$(basename "$f")"
  md5sum "$f" | cut -c1-8
done
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo STAGE2_DONE
