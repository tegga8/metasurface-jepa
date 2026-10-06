"""Symbol Meep spectrum vs the spectrum of the geometry OUR models decoded.

For every symbol: target = the symbol's Meep spectrum mapped through the fitted
CST convention (same as the round-trip); decoded = each model's decoded
occupancy -> assembled [B,3,64,64] geometry -> frozen MetaDiT surrogate ->
[B,2,301] predicted spectrum. Reports the normalized L1 error
(eval_scenarios definition) and |T| magnitude correlation per symbol, and
renders a 9-panel figure.

Usage: python _symbol_spectrum_check.py <scratchpad_dir>
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_scenarios as es
import _novel_modeltest as nmt
from data.factorize import assemble_metadit_geometry

SP = sys.argv[1]
conv = json.load(open(os.path.join(SP, "_meep_convention.json")))
meep = json.load(open(os.path.join(SP, "meep_symbols_out.json")))
z = np.load(os.path.join(REPO, "symbol_designs.npz"), allow_pickle=True)
names = z["names"].tolist()
l, h, r = (float(v) for v in z["scalars"])

targets = torch.stack([nmt.meep_spec_from_entry(dd, conv)
                       for dd in meep["designs"]])  # [9, 2, 301]

models = [
    ("S2 slim", "configs/scaling/unified_s2_slim.yaml",
     os.path.join(SP, "s2_out", "s2", "seed0.pt"), "symbol_dec_S2.npz"),
    ("S1 small", "configs/scaling/unified_s1_small.yaml",
     os.path.join(SP, "s1_out", "s1", "seed0.pt"), "symbol_dec_S1.npz"),
    ("L1 wide", "configs/scaling/unified_l1_wide.yaml",
     os.path.join(SP, "l1a_ckpt", "l1", "seed0.pt"), "symbol_dec_L1.npz"),
    ("base (v2,s0)", "configs/unified.yaml",
     os.path.join(SP, "schedfix_ckpt", "schedfix", "seed0.pt"),
     "symbol_dec_base(v2,s0).npz"),
]
colors = {"S2 slim": "tab:blue", "S1 small": "tab:orange",
          "L1 wide": "tab:green", "base (v2,s0)": "tab:red"}

freqs = np.linspace(0.1, 0.2, 301)
mag_t = targets.pow(2).sum(1).sqrt().numpy()  # [9, 301]

curves = {}
for name, cfg, ckpt, decf in models:
    _, surrogate = nmt.load_model(os.path.join(REPO, cfg), ckpt)
    prob = np.load(os.path.join(SP, decf))["prob"].astype(np.float32)
    occ = torch.from_numpy((prob > 0.5).astype(np.float32))
    b = occ.shape[0]
    geom = assemble_metadit_geometry(occ, torch.full((b,), l),
                                     torch.full((b,), h), torch.full((b,), r))
    with torch.no_grad():
        pred = surrogate(geom).prediction
    err = es._spectrum_error_per_sample(pred, targets).numpy()
    mag_p = pred.pow(2).sum(1).sqrt().numpy()
    corr = np.array([np.corrcoef(mag_t[i], mag_p[i])[0, 1]
                     for i in range(len(names))])
    curves[name] = (mag_p, err, corr)
    print(f"[{name}] mean err={err.mean():.3f}  "
          f"mean |T| corr={corr.mean():.3f}", flush=True)
    for i, nm in enumerate(names):
        print(f"    {nm:14s} err={err[i]:.3f} corr={corr[i]:.3f}", flush=True)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, (nm, ax) in enumerate(zip(names, axes.ravel())):
    ax.plot(freqs, mag_t[i], "k-", lw=2.4, label="symbol (Meep)")
    for name, (mag_p, err, corr) in curves.items():
        ax.plot(freqs, mag_p[i], color=colors[name], lw=1.2, alpha=0.85,
                label=f"{name} (err {err[i]:.2f})")
    ax.set_title(nm, fontsize=11)
    ax.set_xlabel("freq", fontsize=8)
    ax.set_ylabel("|T|", fontsize=8)
axes.ravel()[0].legend(fontsize=6)
fig.suptitle("Symbol Meep spectrum vs decoded-geometry spectrum "
             "(through the frozen surrogate)", fontsize=13)
fig.tight_layout()
out = os.path.join(REPO, "symbol_spectrum_match.png")
fig.savefig(out, dpi=115)
print("saved", out)
