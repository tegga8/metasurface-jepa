"""Seed test, step 1: decode candidates for Meep-based selection (base v1, frozen).

Design (operator spec, 2026-10-26):
  - 18 realizable target spectra (the realizability-screened A' set; measured l1n
    0.348 vs the corpus's own self-density 0.331, i.e. genuinely near-realizable)
  - 5 candidate geometries per target, varied by GUIDANCE STRENGTH
    w in {0.5, 0.75, 1.0, 1.25, 1.5} -- classifier-free guidance, two forward
    passes (real goal, null goal) combined as z = z_null + w*(z_real - z_null)
  - the same procedure repeated on SHUFFLED targets (a derangement of the same 18),
    so the control gets identical treatment
  - Meep then selects, per target, the candidate with the lowest measured spectrum
    error; the paired real-vs-shuffled comparison is the seed test

Decoding only -- the full-wave pass is a separate job. Nothing is trained: base v1
is frozen.

Output: ~/runs/seedtest/dec/decoded_{real,shuf}_w<w>.npz (18 designs each) plus
index.json recording, for every design, which target it was conditioned on.
"""

import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import _novel_modeltest as nmt  # noqa: E402
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402
from runtime.physics_controls import derangement_permutation  # noqa: E402

W_LIST = (0.5, 0.75, 1.0, 1.25, 1.5)
N_TARGETS = 18
OUT = os.path.expanduser("~/runs/seedtest/dec")
CKPT = os.path.expanduser("~/metasurface-jepa/checkpoints/unified/final.pt")
TARGETS = os.path.expanduser("~/runs/phase3f/random_spectra.npz")


def tag(w):
    return f"w{int(round(w * 100)):03d}"


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "unified.yaml"),
                              encoding="utf-8"))
    model, surrogate = nmt.load_model(os.path.join(REPO, "configs", "unified.yaml"),
                                      CKPT)
    model.eval()

    z = np.load(TARGETS, allow_pickle=True)
    re = torch.from_numpy(z["A_real"][:N_TARGETS].astype(np.float32))
    im = torch.from_numpy(z["A_imag"][:N_TARGETS].astype(np.float32))
    spec_real = torch.stack([re, im], dim=1)                       # [n,2,301]

    b = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
    carriers = torch.from_numpy(b["patterns"][:N_TARGETS].astype(np.float32)).unsqueeze(1)
    l, h, r = (float(v) for v in b["scalars"])
    sv = torch.tensor([[l, h, r]] * N_TARGETS, dtype=torch.float32)
    sk = torch.zeros(N_TARGETS, 3, dtype=torch.bool)               # all unknown
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(carriers, ratio=1.0)                         # fully masked

    # shuffled control: a derangement of the same 18 targets
    perm_idx = derangement_permutation(N_TARGETS, spec_real.device, seed=0)
    spec_shuf = spec_real[perm_idx]

    index = {}
    for arm, spec in (("real", spec_real), ("shuf", spec_shuf)):
        with torch.no_grad():
            out_r = model(carriers, sv, sk, spec, M, goal_mode="real",
                          with_target=False)
            out_n = model(carriers, sv, sk, spec, M, goal_mode="null",
                          with_target=False)
        for w in W_LIST:
            z_hat = out_n["z_hat"] + w * (out_r["z_hat"] - out_n["z_hat"])
            s_pred = (out_n["scalar_pred"]
                      + w * (out_r["scalar_pred"] - out_n["scalar_pred"]))
            prob = model.decode_occupancy_prob(z_hat, s_pred, scalar_known=sk,
                                               scalar_values=sv)
            hard = (prob > 0.5).float()
            occ = (prob > 0.5).to(torch.uint8).squeeze(1).numpy()
            path = os.path.join(OUT, f"decoded_{arm}_{tag(w)}.npz")
            np.savez(path, patterns=occ,
                     scalars=np.tile(np.array([l, h, r]), (N_TARGETS, 1)),
                     names=np.array([f"{arm}_{tag(w)}/tgt{k:02d}"
                                     for k in range(N_TARGETS)]),
                     source_idx=np.arange(N_TARGETS))
            # what did the surrogate think? (diagnostic only, not the selection)
            with torch.no_grad():
                pred = surrogate(assemble_metadit_geometry(
                    hard, torch.full((N_TARGETS,), l), torch.full((N_TARGETS,), h),
                    torch.full((N_TARGETS,), r))).prediction
            err = ((pred - spec).abs().mean(dim=(1, 2))
                   / spec.std(dim=(1, 2)).clamp_min(1e-9))
            print(f"{arm} {tag(w)}: mean occupancy {occ.mean():.3f} | "
                  f"surrogate err {err.mean():.3f}", flush=True)
            for k in range(N_TARGETS):
                index[f"{arm}_{tag(w)}/tgt{k:02d}"] = {
                    "arm": arm, "w": w, "slot": int(k),
                    "conditioned_on": int(k) if arm == "real" else int(perm_idx[k]),
                }
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": TARGETS, "ckpt": CKPT, "w_list": list(W_LIST),
                   "designs": index}, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
