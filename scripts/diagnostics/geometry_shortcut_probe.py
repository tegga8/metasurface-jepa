#!/usr/bin/env python3
"""Read-only geometry shortcut/probe sanity diagnostic.

Compares identical fixed validation geometries and split across:
  - trivial features directly extracted from the input geometry
  - trained EMA representation (mean-pooled)
  - released MetaDiT geometry encoder (mean-pooled)
  - random-init geometry encoder (mean-pooled)

The existing probe produced catastrophic negative R2 values. The critical
sanity check here is whether the trivial input features can recover the
dataset's explicit physical parameters. If that baseline fails, the probe
pipeline must not be interpreted.
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from assembly import build_model, load_into_model
from data.dataset import MetaDiTDataset, collate_batch
from encoders.geometry_encoder import GeometryEncoder


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs/milestone_b.yaml"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-geoms", type=int, default=512)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--out", default=str(REPO_ROOT / "checkpoints/milestone_b/geometry_shortcut_probe.json"))
    return p.parse_args()


def split_indices(n, val_fraction, seed):
    if n < 8 or not 0 < val_fraction < 1:
        raise ValueError("Need n>=8 and 0<val_fraction<1")
    perm = np.random.RandomState(seed).permutation(n)
    nv = max(1, int(round(val_fraction * n)))
    if nv >= n:
        nv = n - 1
    return Split(train_idx=perm[nv:], val_idx=perm[:nv])


def trivial_features(G):
    if G.ndim != 4 or tuple(G.shape[1:]) != (3, 64, 64):
        raise ValueError(f"Expected [N,3,64,64], got {tuple(G.shape)}")
    occ = ((G[:, 0] != 0) | (G[:, 1] != 0)).float()
    nocc = occ.flatten(1).sum(1).clamp_min(1.0)
    r = (G[:, 0] * occ).flatten(1).sum(1) / nocc
    h = (G[:, 1] * occ).flatten(1).sum(1) / nocc
    l = G[:, 2].flatten(1).mean(1)
    return torch.stack([
        occ.flatten(1).mean(1),
        r, h, l,
        G[:, 0].flatten(1).amax(1),
        G[:, 1].flatten(1).amax(1),
    ], dim=1)


def fit_ridge(Xtr, ytr, Xva, lam):
    Xtr = np.asarray(Xtr, np.float64)
    ytr = np.asarray(ytr, np.float64)
    Xva = np.asarray(Xva, np.float64)
    mu = Xtr.mean(0, keepdims=True)
    sd = np.maximum(Xtr.std(0, keepdims=True), 1e-12)
    A = (Xtr - mu).T @ (Xtr - mu)
    A += lam * Xtr.shape[0] * np.eye(Xtr.shape[1])
    yc = ytr - ytr.mean(0, keepdims=True)
    W = np.linalg.solve(A, (Xtr - mu).T @ yc)
    pred = (Xva - mu) @ W + ytr.mean(0, keepdims=True)
    return pred


def r2(y, pred):
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    ssr = ((y - pred) ** 2).sum(0)
    sst = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    out = np.full(y.shape[1], np.nan)
    ok = sst > 1e-14
    out[ok] = 1.0 - ssr[ok] / sst[ok]
    return out


def probe(X, params, split, lam):
    X = torch.as_tensor(X).cpu()
    if X.ndim == 3:
        X = X.mean(1)
    X = X.numpy()
    y = np.asarray(params, np.float64)
    pred = fit_ridge(
        X[split.train_idx], y[split.train_idx],
        X[split.val_idx], lam
    )
    vals = r2(y[split.val_idx], pred)
    names = ["l_lattice", "h_atom", "r_atom"]
    d = {f"{n}_r2": float(v) for n, v in zip(names, vals)}
    d["mean_r2"] = float(np.nanmean(vals))
    return d


def collect(enc, batches, device):
    if next(enc.parameters()).device != device:
        raise RuntimeError(
            f"Encoder is on {next(enc.parameters()).device}, requested {device}"
        )
    was_training = enc.training
    enc.eval()
    try:
        with torch.no_grad():
            return torch.cat([enc(g.to(device)).cpu() for g in batches], 0)
    finally:
        enc.train(was_training)


def checksum(model):
    return sum(p.detach().double().sum().item() for p in model.parameters())


def build_random(hidden, heads, depth, seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GeometryEncoder(hidden=hidden, num_heads=heads, depth=depth).to(device)


def main():
    a = args()
    device = torch.device(a.device)
    with open(a.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(a.checkpoint)
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ccfg = ck.get("cfg", cfg)

    val_path = Path(a.data_root) / "split_data/val_set.mat" if a.data_root else REPO_ROOT / ccfg["data"]["val_split"]
    ds = MetaDiTDataset(str(val_path))
    n = min(a.max_geoms, len(ds))

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        drop_last=False, collate_fn=collate_batch)
    gs, left = [], n
    for G, _ in loader:
        take = min(left, G.shape[0])
        gs.append(G[:take].clone())
        left -= take
        if left == 0:
            break
    G = torch.cat(gs, 0)
    params = np.asarray(ds.parameter[:n], np.float64)

    if G.shape[0] != n or params.shape != (n, 3):
        raise RuntimeError(f"Geometry/parameter alignment failed: {G.shape}, {params.shape}")

    split = split_indices(n, a.val_fraction, a.probe_seed)
    triv = trivial_features(G)
    triv_probe = probe(triv, params, split, a.ridge_lambda)

    print("\nTRIVIAL INPUT BASELINE")
    print(triv_probe)

    if not np.isfinite(triv_probe["mean_r2"]) or triv_probe["mean_r2"] < 0.90:
        raise RuntimeError(
            "TRIVIAL BASELINE FAILED. Do not interpret representation probes. "
            f"Got {triv_probe}"
        )

    spec_path = REPO_ROOT / ccfg["weights"]["spectrum"]
    metadit_path = REPO_ROOT / ccfg["weights"]["metadit"]

    model = build_model(ccfg["model"], str(spec_path), device=device,
                        init_from_metadit=False, metadit_weights=str(metadit_path))
    load_into_model(model, ck["model"], device)
    from train.engine import restore_ema_state
    restore_ema_state(model, ck.get("ema_state"))
    before = checksum(model)

    X_trained = collect(model.ema, gs, device)

    hidden = int(ccfg["model"].get("hidden", 384))
    heads = int(ccfg["model"].get("num_heads", 6))
    depth = int(ccfg["model"].get("geo_depth", 6))

    released = GeometryEncoder(hidden=hidden, num_heads=heads, depth=depth).to(device)
    payload = torch.load(metadit_path, map_location="cpu", weights_only=False)
    released.init_from_metadit(payload, blocks_to_take=depth)
    X_released = collect(released, gs, device)

    random_enc = build_random(hidden, heads, depth, a.probe_seed, device)
    X_random = collect(random_enc, gs, device)

    results = {
        "trivial_input": triv_probe,
        "trained_ema": probe(X_trained, params, split, a.ridge_lambda),
        "released_vit": probe(X_released, params, split, a.ridge_lambda),
        "random_init": probe(X_random, params, split, a.ridge_lambda),
    }

    if checksum(model) != before:
        raise RuntimeError("Checkpoint model changed during read-only diagnostic")

    print("\nIDENTICAL-SPLIT PROBE RESULTS")
    print("----------------------------------------")
    print(f"{'representation':<18}{'l_lattice':>14}{'h_atom':>14}{'r_atom':>14}{'mean R2':>14}")
    for name, d in results.items():
        print(f"{name:<18}{d['l_lattice_r2']:>14.6f}{d['h_atom_r2']:>14.6f}"
              f"{d['r_atom_r2']:>14.6f}{d['mean_r2']:>14.6f}")

    trained = results["trained_ema"]["mean_r2"]
    random_r2 = results["random_init"]["mean_r2"]
    released_r2 = results["released_vit"]["mean_r2"]
    base = triv_probe["mean_r2"]

    print("\nINTERPRETATION")
    if trained > base + 0.05:
        print("trained representation beats the trivial baseline by >0.05 mean R2.")
    else:
        print("trained representation does NOT materially beat the trivial baseline.")
    print(f"trained - trivial : {trained - base:+.6f}")
    print(f"trained - random  : {trained - random_r2:+.6f}")
    print(f"trained - released: {trained - released_r2:+.6f}")
    print("These physical-parameter probes do NOT prove spatial-geometry understanding.")

    out = Path(a.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": str(ckpt),
            "step": ck.get("step"),
            "epoch": ck.get("epoch"),
            "n_geoms": n,
            "probe_seed": a.probe_seed,
            "val_fraction": a.val_fraction,
            "ridge_lambda": a.ridge_lambda,
            "results": results,
            "read_only_verified": True,
        }, f, indent=2)

    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
