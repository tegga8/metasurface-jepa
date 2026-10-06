"""Meep-vs-Meep spectrum comparison: symbol glyph vs the geometry each model decoded.

Targets : meep_symbols_out.json (9 designs, symbol order).
Decoded : symbol_decodes_meep.json (36 designs: S2 x9, S1 x9, L1 x9, base x9).
|a_s/a_e| ("oz" tag, same parity the round-trip used) is phase-convention
invariant, so no convention fit is needed. Reports normalized L1 error
(mean |d| / std(target)) and Pearson correlation of the |T| curves.

Usage: python _compare_decodes_spectra.py <scratchpad_dir>
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SP = sys.argv[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = np.load(os.path.join(REPO, "symbol_designs.npz"),
                allow_pickle=True)["names"].tolist()
tgt = json.load(open(os.path.join(SP, "meep_symbols_out.json")))["designs"]
dec = json.load(open(os.path.join(SP, "symbol_decodes_meep.json")))["designs"]
tag = "oz"

targets = np.array([np.array(d["modes"][tag]["mag"]) for d in tgt])  # [9,301]
models = ["S2", "S1", "L1", "base"]
freqs = np.linspace(0.1, 0.2, 301)

curves = {}
for mi, model in enumerate(models):
    mags, errs, corrs = [], [], []
    for i in range(len(names)):
        m = np.array(dec[mi * len(names) + i]["modes"][tag]["mag"])
        mags.append(m)
        std = max(float(targets[i].std()), 1e-9)
        errs.append(float(np.abs(m - targets[i]).mean() / std))
        corrs.append(float(np.corrcoef(targets[i], m)[0, 1])
                     if m.std() > 0 else 0.0)
    curves[model] = (np.stack(mags), np.array(errs), np.array(corrs))
    print(f"[{model}] mean err={np.mean(errs):.3f}  "
          f"mean |T| corr={np.mean(corrs):.3f}", flush=True)
    for i, nm in enumerate(names):
        print(f"    {nm:14s} err={errs[i]:.3f} corr={corrs[i]:.3f}", flush=True)

colors = {"S2": "tab:blue", "S1": "tab:orange", "L1": "tab:green",
          "base": "tab:red"}
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, (nm, ax) in enumerate(zip(names, axes.ravel())):
    ax.plot(freqs, targets[i], "k-", lw=2.4, label="symbol (Meep)")
    for model, (mags, errs, corrs) in curves.items():
        ax.plot(freqs, mags[i], color=colors[model], lw=1.2, alpha=0.85,
                label=f"{model} decoded (err {errs[i]:.2f})")
    ax.set_title(nm, fontsize=11)
    ax.set_xlabel("freq", fontsize=8)
    ax.set_ylabel("|T|", fontsize=8)
    if targets[i].max() < 10:
        ax.set_ylim(0, max(1.2, targets[i].max() * 1.15,
                           max(m.max() for m, _, _ in curves.values()) * 1.1))
axes.ravel()[0].legend(fontsize=6)
fig.suptitle("Meep |T|: symbol geometry vs the geometry our models decoded "
             "from its spectrum", fontsize=13)
fig.tight_layout()
out = os.path.join(REPO, "symbol_spectrum_match_meep.png")
fig.savefig(out, dpi=115)
print("saved", out)
