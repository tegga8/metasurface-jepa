#!/bin/bash
# Light precondition check for the scaling gate-8 run.
echo "=== scaling configs ==="
ls "$HOME/imp2/configs/scaling/" 2>/dev/null
echo "=== weights present ==="
for f in "$HOME/imp2/weights/spec_encoder.pth" "$HOME/imp2/weights/metadit_surrogate.pt"; do
  [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
done
ls "$HOME/imp2/weights/" 2>/dev/null | head -5
echo "=== data ==="
ls "$HOME/imp2/data/metadit/split_data/val_set.mat" 2>/dev/null && echo "val OK"
echo "=== existing scaling ckpts on cluster ==="
ls -l "$HOME"/*.pt 2>/dev/null | head -5
echo PRE2_DONE
