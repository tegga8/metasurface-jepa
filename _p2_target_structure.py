"""Structural comparison of the phase-2 targets against the corpus.

The targets are convex mixtures of real spectra. Linear mixing preserves passivity
and causality (real weights, both parts combined with the same weights), so they are
physically admissible -- but it also AVERAGES AWAY the sharp resonances that make a
transmission spectrum a fingerprint of a particular geometry. That is a distribution
shift, not a physics violation, and this measures it:

  roughness   mean|d|T|/domega| / mean|T|      how jagged the curve is
  peakiness   max|T| / median|T|               how strong the resonances are
  n_extrema   sign changes of d|T|             how many features the curve has

Reported per set, against the corpus itself, because the corpus is what the model
was trained on and what "realizable" actually means here.
"""

import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
NPZ = sys.argv[1] if len(sys.argv) > 1 else r"D:\metamaterials_scratch\random_spectra_feasible.npz"
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")


def stats(re, im):
    m = np.sqrt(np.asarray(re, float) ** 2 + np.asarray(im, float) ** 2)
    rough = np.mean(np.abs(np.diff(m, axis=1)) / np.maximum(m.mean(axis=1, keepdims=True), 1e-9),
                    axis=1)
    peaky = m.max(axis=1) / np.maximum(np.median(m, axis=1), 1e-9)
    d = np.diff(m, axis=1)
    ext = np.sum(np.abs(np.diff(np.sign(d), axis=1)) > 0, axis=1)
    return rough, peaky, ext


cases = {}
d = np.load(NPZ, allow_pickle=True)
c = io.loadmat(SPLIT)
cases["corpus"] = stats(c["real"], c["imag"])
cases["A' feasible"] = stats(d["A_real"], d["A_imag"])
cases["B' infeasible"] = stats(d["B_real"], d["B_imag"])

print(f"{'set':16s} {'roughness':>10s} {'peak/median':>12s} {'extrema':>9s}")
for k, (r, p, e) in cases.items():
    print(f"{k:16s} {np.median(r):10.4f} {np.median(p):12.3f} {np.median(e):9.0f}")

rc = np.median(cases["corpus"][0])
print("\nrelative to the corpus:")
for k in ("A' feasible", "B' infeasible"):
    r, p, e = cases[k]
    print(f"  {k:14s} roughness {np.median(r) / rc:5.2f}x | "
          f"peakiness {np.median(p) / np.median(cases['corpus'][1]):5.2f}x | "
          f"features {np.median(e) / np.median(cases['corpus'][2]):5.2f}x")
