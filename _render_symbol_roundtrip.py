"""Render the symbol round-trip: original glyph vs decoded geometry per model.

Rows: original | S2 slim | S1 small | L1 wide (decoded hard maps, prob > 0.5).
Usage: python _render_symbol_roundtrip.py [scratchpad_dir]
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SP = sys.argv[1] if len(sys.argv) > 1 else REPO

z = np.load(os.path.join(REPO, "symbol_designs.npz"), allow_pickle=True)
names = z["names"].tolist()
occ_true = z["patterns"].astype(float)

models = [("S2 slim", "symbol_dec_S2.npz"),
          ("S1 small", "symbol_dec_S1.npz"),
          ("L1 wide", "symbol_dec_L1.npz"),
          ("base (v2, seed0)", "symbol_dec_base(v2,s0).npz")]

rows = [("original", occ_true)]
for label, fname in models:
    p = os.path.join(SP, fname)
    if not os.path.exists(p):
        print("missing", p)
        continue
    d = np.load(p)
    pats = np.squeeze((d["prob"] > 0.5).astype(float))
    rows.append((label, pats))

n = len(names)
fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 2.1 * len(rows)))
for r, (label, pats) in enumerate(rows):
    for c in range(n):
        ax = axes[r, c]
        ax.imshow(pats[c], cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(names[c], fontsize=9)
        if c == 0:
            ax.set_ylabel(label, fontsize=10)
fig.suptitle("Symbol round-trip — Meep spectrum -> model decode "
             "(scenario A: full mask, all scalars unknown)", fontsize=12)
fig.tight_layout()
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO, "symbol_roundtrip_preview.png")
fig.savefig(out, dpi=115)
print("saved", out)
