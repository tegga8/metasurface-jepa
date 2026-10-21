#!/bin/bash
# What did the phase-2 decodes actually produce? Occupancy is the tell: a near-empty
# decode cannot encode any spectrum, so every target would measure the same.
echo "=== decode summary (imp2, A'/B') ==="
cat "$HOME/runs/p23imp2/dec_imp2/phase2_summary.json" 2>/dev/null
echo "=== corpus occupancy for reference ==="
python - <<'PY'
import os
import numpy as np
from scipy import io
p = os.path.expanduser("~/imp2/data/metadit/split_data/val_set.mat")
d = io.loadmat(p)
pat = d["pattern"]
occ = pat.reshape(pat.shape[0], -1).mean(axis=1)
print(f"  corpus occupancy: median {np.median(occ):.3f} p10 {np.percentile(occ,10):.3f} "
      f"p90 {np.percentile(occ,90):.3f}")
PY
echo P2DECCHK_DONE
