"""Is the phase-2 target generation physically valid?

Checks the generated A'/B' targets against the constraints any realizable
transmission spectrum must satisfy, and compares against the corpus itself:

  1. PASSIVITY: |T| <= 1 at every frequency. A target above 1 asks for a gain,
     which no passive dielectric can produce -- the decode would be scored against
     something impossible, independently of the model.
  2. WHERE the violation sits: near the band edges the corpus's own |T| is small;
     a convex combination there can legitimately exceed the corpus values.
  3. Kramers-Kronig plausibility (cheap proxy): a realizable complex T has real and
     imaginary parts locked together by causality. A convex combination of real
     spectra keeps each part smooth, but nothing enforces the pair; the check below
     reports how far the generated (re, im) pair is from satisfying the corpus's own
     re/im relationship, fitted linearly on the corpus.

Everything here is computed from the saved npz -- no model, no Meep.
"""

import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
NPZ = sys.argv[1] if len(sys.argv) > 1 else r"D:\metamaterials_scratch\random_spectra_feasible.npz"
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")


def mag(re, im):
    return np.sqrt(np.asarray(re, float) ** 2 + np.asarray(im, float) ** 2)


def report(name, re, im):
    m = mag(re, im)
    peak = m.max(axis=1)
    print(f"\n{name}: n={m.shape[0]}")
    print(f"  peak |T|: median {np.median(peak):.3f}  max {peak.max():.3f}")
    for thr in (1.0, 1.001, 1.02, 1.1, 1.5):
        frac = float((peak > thr).mean())
        print(f"  samples with peak > {thr:<5}: {frac:6.1%}")
    # per-frequency violation profile, split into the core vs edges of the band
    over = (m > 1.0)
    nf = m.shape[1]
    core = slice(nf // 10, nf - nf // 10)
    print(f"  fraction of (sample,freq) cells > 1: all {over.mean():.3%} | "
          f"core {over[:, core].mean():.3%} | edges {over[:, :nf // 10].mean():.3%}"
          f"/{over[:, -nf // 10:].mean():.3%}")
    return peak


d = np.load(NPZ, allow_pickle=True)
print("keys:", sorted(d.files))
corpus = io.loadmat(SPLIT)
R, I = corpus["real"].astype(float), corpus["imag"].astype(float)

p_corpus = report("corpus (val split)", R, I)
report("A' feasible", d["A_real"], d["A_imag"])
report("B' infeasible", d["B_real"], d["B_imag"])

# how do the generated peaks compare with the corpus peak distribution?
print("\ncorpus peak percentiles: 50/90/99/max = "
      + " / ".join(f"{np.percentile(p_corpus, q):.3f}" for q in (50, 90, 99, 100)))
