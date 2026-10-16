"""Build the empty-cell control inputs for the task-3 passivity gate.

The gate compares a sample against an empty-cell reference and calls a peak above
1.02 a solver failure. Feeding it an EMPTY pattern as the sample makes the sample
and the reference physically identical, so the measured ratio is the
instrument's own error: whatever this returns above/below 1 is the noise floor
and the systematic offset of the statistic, not physics.

Two files, one per scalar combo actually used by the affected sets, each with a
single all-zero 64x64 pattern.
"""

import os

import numpy as np

OUT = r"D:\metamaterials_scratch"
COMBOS = {"m20": (2.2, 0.6629, 3.5867), "p20": (3.3, 0.9943, 5.3801)}

for tag, (l, h, r) in COMBOS.items():
    p = os.path.join(OUT, f"_empty_{tag}.npz")
    np.savez(p,
             patterns=np.zeros((1, 64, 64), dtype=np.uint8),
             scalars=np.array([[l, h, r]], dtype=np.float64),
             names=np.array(["EMPTY"]))
    print("wrote", p, "scalars", (l, h, r))
