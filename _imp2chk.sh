#!/bin/bash
# imp2 phase-2/3 job (645620): status + per-set progress.
echo "=== queue ==="
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo "=== log tail ==="
tail -20 "$HOME/logs/imp2p23_645620.out" 2>/dev/null || echo "no log"
echo "=== files per set ==="
for t in jepa_imp2_A jepa_imp2_B jepa_imp2_B_true jepa_imp2_spec_m20 jepa_imp2_spec_p20; do
  echo "  $t: $(ls "$HOME/runs/p23imp2/meep/$t"/e_0_*.json 2>/dev/null | wc -l)"
done
echo IMP2P23CHK_DONE
