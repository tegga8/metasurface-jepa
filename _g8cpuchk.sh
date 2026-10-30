#!/bin/bash
# CPU scaling gate-8 (642602) + resolution test progress.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== gate8 CPU log ==="
tail -12 "$HOME/logs/gate8scaleCPU_642602.out" 2>/dev/null || echo "no log"
echo "=== gate8 jsons ==="
ls -l "$HOME/runs/gate8/"*scale*.json 2>/dev/null
echo "=== t3res64 ==="
tail -2 "$HOME/logs/t3res64_642594.out" 2>/dev/null
ls "$HOME/runs/t3res"/res*.json 2>/dev/null | wc -l
echo CHK3_DONE
