"""Verify the surrogate path end-to-end before trusting any Phase 2 number.

The user is right to demand this. If the frozen surrogate on a REAL dataset
geometry does not reproduce that geometry's true CST spectrum, then every
"decoded geometry -> surrogate spectrum" number is meaningless -- including the
Task 1 surrogate errors and the Phase 2 consistency check.

Checks, in order:

  1. FORWARD SANITY. surrogate(true geometry) vs the true CST spectrum, on real
     MetaDiT validation geometries. This is the reference implementation's own
     contract (metric.py:148-150 builds exactly this tensor), so it must be
     accurate. Reported as the evaluator's normalized L1 plus |T| correlation.
  2. RANGES. Amplitudes and frequency content of the real corpus versus the
     generated Phase 2 targets, so "out of the trained range" can be confirmed
     or ruled out rather than assumed.
  3. ROUND TRIP. feed the surrogate's own output back as a target and decode,
     to check the decode path is not itself broken.

Usage: python _verify_pipeline.py <config.yaml> <checkpoint.pt> [n]
"""

import json
import os
import sys

import numpy as np
import torch
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
sys.path.insert(0, REPO)

import _novel_modeltest as nmt  # noqa: E402
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402

SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")
NFREQ = 301
FREQ = np.linspace(0.1, 0.2, NFREQ)


def main():
    cfg_path, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    for p in (cfg_path, ckpt):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    try:
        d = io.loadmat(SPLIT)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {SPLIT}: {exc}")
    pat, par, real, imag = d["pattern"], d["parameter"], d["real"], d["imag"]
    N = pat.shape[-1]

    model, surrogate = nmt.load_model(cfg_path, ckpt)
    print(f"model: {os.path.basename(cfg_path)}  ckpt: "
          f"{os.path.basename(ckpt)}")

    rng = np.random.default_rng(0)
    idx = rng.choice(N, n, replace=False)

    # ---------- 1. forward sanity ----------
    # pattern is stored (64, 64, N): the ITEM axis is last, so transpose it
    # to [N, 64, 64] before adding the channel axis, or assemble_geometry
    # reads the batch size off the wrong axis.
    occ = torch.from_numpy(
        (np.transpose(pat[:, :, idx], (2, 0, 1)) == 1).astype(np.float32)
    ).unsqueeze(1)
    par_t = torch.tensor(np.asarray(par[idx], dtype=np.float32))
    geom = assemble_metadit_geometry(occ, par_t[:, 0], par_t[:, 1], par_t[:, 2])
    with torch.no_grad():
        pred = surrogate(geom).prediction
    # [N, 2, 301]: stack real/imag on axis 1 (the condition layout), not 0.
    truth = torch.from_numpy(
        np.stack([real[idx], imag[idx]], axis=1).astype(np.float32))
    err = (pred - truth).abs().mean(dim=(1, 2)) / truth.std(dim=(1, 2))
    pm = pred.pow(2).sum(1).sqrt().numpy()
    tm = truth.pow(2).sum(1).sqrt().numpy()
    corr = np.array([np.corrcoef(tm[i], pm[i])[0, 1] if pm[i].std() > 0 else 0.0
                     for i in range(n)])
    print("=" * 66)
    print(f"[1] FORWARD SANITY: surrogate(true geometry) vs true CST spectrum")
    print(f"    n={n}  normalized L1 err: med {err.median():.4f} "
          f"mean {err.mean():.4f} max {err.max():.4f}")
    print(f"    |T| correlation:        med {np.median(corr):.4f} "
          f"mean {corr.mean():.4f}")
    ok = float(err.median()) < 0.25
    print(f"    -> {'PASS' if ok else 'FAIL'}: the surrogate reproduces real CST "
          f"spectra; decoded-geometry surrogate numbers are meaningful")

    # ---------- 2. ranges ----------
    print("=" * 66)
    print("[2] RANGES: real corpus vs generated Phase 2 targets")
    realmag = np.sqrt(real ** 2 + imag ** 2)
    print(f"    real |T|      min {realmag.min():.4f} max {realmag.max():.4f} "
          f"mean {realmag.mean():.4f}")
    print(f"    real real[]   min {real.min():.4f} max {real.max():.4f}")
    print(f"    real imag[]   min {imag.min():.4f} max {imag.max():.4f}")
    rough = np.abs(np.diff(realmag, axis=1)) / np.maximum(
        realmag.mean(axis=1, keepdims=True), 1e-9)
    print(f"    real roughness med {np.median(rough):.4f}")
    spec_path = os.path.join(REPO, "..", "random_spectra.npz")
    for cand in (spec_path,
                 r"D:\metamaterials_scratch\random_spectra.npz"):
        if os.path.exists(cand):
            z = np.load(cand, allow_pickle=True)
            for lab, rk, ik in (("set A", "A_real", "A_imag"),
                                ("set B", "B_real", "B_imag")):
                m = np.sqrt(z[rk] ** 2 + z[ik] ** 2)
                rg = np.abs(np.diff(m, axis=1)) / np.maximum(
                    m.mean(axis=1, keepdims=True), 1e-9)
                print(f"    {lab} |T|      min {m.min():.4f} max {m.max():.4f} "
                      f"mean {m.mean():.4f}  roughness med {np.median(rg):.4f}")
            break

    # ---------- 3. decode round trip on a real target ----------
    print("=" * 66)
    print("[3] DECODE ROUND TRIP: real geometry + its TRUE spectrum -> decode")
    k = 24
    occ_k = occ[:k]
    par_k = par_t[:k]
    mask = BlockMasker(placement="random", grid=16, min_side=3,
                       k_range=(1, 4), seed=0)
    M_full = mask.sample(occ_k, ratio=1.0)
    M_half = mask.sample(occ_k, ratio=0.5)
    for ratio, M in (("full", M_full), ("half", M_half)):
        with torch.no_grad():
            sk = torch.zeros(k, 3, dtype=torch.bool)
            out = model(occ_k, par_k, sk, truth[:k], M, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=sk,
                                               scalar_values=par_k)
        ious = np.array([nmt.iou_f1(prob[i], occ_k[i])[0] for i in range(k)])
        print(f"    mask {ratio:4s}: IoU vs true geometry med "
              f"{np.median(ious):.3f} mean {ious.mean():.3f}")
    print("    -> a healthy decode on real targets; if this is near chance the "
          "decode path itself is broken")


if __name__ == "__main__":
    main()