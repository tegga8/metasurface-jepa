#!/bin/bash
# Quota + whether the scaling configs are in the checkout we run from.
echo "=== quota ==="
quota -s 2>/dev/null || df -h "$HOME" | tail -1
echo "=== home size ==="
du -sh "$HOME" 2>/dev/null | tail -1
echo "=== scaling configs (imp2 checkout) ==="
ls "$HOME/imp2/configs/scaling/" 2>/dev/null
echo "=== spectrum weights ==="
ls -l "$HOME/imp2"/weights/*.pth 2>/dev/null | head -3
echo PRE_DONE
