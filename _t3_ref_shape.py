"""Print the stored reference curve's shape (normalized) at coarse spacing.

Two scalar combos' references differ by a constant 1.4185 at every frequency --
they are the same curve up to scale, which cannot be a cavity resonance (the
cells differ in size). Seeing the shape settles whether the denominator is the
source spectrum, an air-path resonance, or something else.
"""

import glob
import json
import os

import numpy as np

base = r"D:\metamaterials_scratch\t3"
sets = ["jepa_v1_spec_m20", "jepa_v1_spec_p20"]
FREQS = np.linspace(0.1, 0.2, 301)

for tag in sets:
    d = os.path.join(base, tag)
    f = sorted(glob.glob(os.path.join(d, "e_0_*.json")))[0]
    j = json.load(open(f, encoding="utf-8"))
    for k, r in j.get("refs", {}).items():
        fe = np.abs(np.asarray(r["flux"], dtype=float))
        med = float(np.median(fe))
        print(f"\n{tag} ref {k}: peak/median = {fe.max() / med:.2f}")
        line = []
        for i in range(0, 301, 15):
            line.append(f"{FREQS[i]:.3f}:{fe[i] / med:9.2e}")
        for row in range(0, len(line), 5):
            print("   " + "  ".join(line[row:row + 5]))
        # width of the main lobe relative to the band
        above = fe / med > 0.5
        print(f"   fraction of band above 0.5*median: {100 * above.mean():.0f}%")
        print(f"   argmax at omega={FREQS[int(np.nanargmax(fe))]:.4f}")
