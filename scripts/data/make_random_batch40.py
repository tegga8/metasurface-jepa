"""Generate a random 40-design batch (20 symmetric + 20 asymmetric).

Dataset-style morphology (superellipse brick + large circular bites + optional
hole, morphology clean-up — same family as the free-form novel set) with two
constraints:
  * symmetric  : the mask is made exactly mirror-symmetric across a randomly
                 chosen axis set (x / y / both), features mirror-completed;
  * asymmetric : body and features may sit off-centre; no symmetry constraint.
Occupancy targets follow each set's own linspace ladder over the dataset's
p5..p95 occupancy percentiles; every design is verified D4-novel against the
released splits (byte-exact canonical hash). Shared scalars = the MEAN of the
dataset's scalar parameters (computed over the released splits).

Usage: python make_random_batch40.py [--out DIR] [--data DIR]
Writes batch40_designs.npz (+ batch40_preview.png) to --out.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
from scipy import io, ndimage

SIZE = 64
RNG = np.random.RandomState(20261007)

_Y, _X = np.ogrid[0:SIZE, 0:SIZE]


def superellipse(a, b, n, cx=31.5, cy=31.5):
    return ((np.abs(_X - cx) / a) ** n + (np.abs(_Y - cy) / b) ** n) <= 1.0


def disk(cx, cy, r):
    return ((_X - cx) ** 2 + (_Y - cy) ** 2) <= r * r


def plus_arms(w, ln, cx=31.5, cy=31.5):
    return ((np.abs(_X - cx) <= w / 2) & (np.abs(_Y - cy) <= ln / 2)) | \
           ((np.abs(_Y - cy) <= w / 2) & (np.abs(_X - cx) <= ln / 2))


def clean(mask):
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    bg = ~mask
    lab, nc = ndimage.label(bg)
    sizes = np.bincount(lab.ravel())
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    for c in range(1, nc + 1):
        if sizes[c] < 12 and c not in border:
            mask |= (lab == c)
    lab, nc = ndimage.label(mask)
    if nc > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = (lab == np.argmax(sizes))
    return mask


def gen(target_occ, symmetric):
    for _ in range(800):
        kx = ky = False
        if symmetric:
            mode = RNG.randint(0, 3)
            kx = mode in (0, 2)
            ky = mode in (1, 2)
        ox = 0.0 if symmetric else RNG.uniform(-7, 7)
        oy = 0.0 if symmetric else RNG.uniform(-7, 7)
        a, b = RNG.uniform(13, 30, 2)
        n = RNG.uniform(3.0, 9.0)
        mask = superellipse(a, b, n, 31.5 + ox * 0.4, 31.5 + oy * 0.4)

        for _k in range(RNG.randint(0, 4)):
            rb = RNG.uniform(7, 21)
            side = RNG.randint(0, 5)
            if side == 4:
                cx = 31.5 + RNG.uniform(-16, 16)
                cy = 31.5 + RNG.uniform(-16, 16)
            else:
                off = RNG.uniform(0.5, 1.15) * max(a, b)
                cx = 31.5 + (off if side == 0 else -off if side == 1
                             else RNG.uniform(-4, 4))
                cy = 31.5 + (off if side == 2 else -off if side == 3
                             else RNG.uniform(-4, 4))
            bite = disk(cx, cy, rb)
            if symmetric:
                if kx:
                    bite |= np.fliplr(bite)
                if ky:
                    bite |= np.flipud(bite)
            mask &= ~bite

        if RNG.random() < 0.5:
            hx, hy = RNG.uniform(3, 13, 2)
            kind = RNG.randint(0, 3)
            jx = 0.0 if symmetric else RNG.uniform(-6, 6)
            jy = 0.0 if symmetric else RNG.uniform(-6, 6)
            if kind == 0:
                hole = superellipse(hx, hy, 2.0, 31.5 + jx, 31.5 + jy)
            elif kind == 1:
                hole = superellipse(hx, hy, 10.0, 31.5 + jx, 31.5 + jy)
            else:
                hole = plus_arms(RNG.uniform(4, 9), RNG.uniform(9, 26),
                                 31.5 + jx, 31.5 + jy)
            if symmetric:
                if kx:
                    hole |= np.fliplr(hole)
                if ky:
                    hole |= np.flipud(hole)
            mask &= ~hole

        if mask.sum() == 0:
            continue
        if symmetric:
            if kx:
                mask = mask & np.fliplr(mask)
            if ky:
                mask = mask & np.flipud(mask)
        mask = clean(mask)
        if not symmetric:
            if (mask == np.fliplr(mask)).all() or \
               (mask == np.flipud(mask)).all():
                continue
        occ = int(mask.sum())
        if abs(occ - target_occ) <= 150:
            return mask.astype(np.uint8), (kx, ky)
    raise RuntimeError(f"generation failed (occ={target_occ}, sym={symmetric})")


def d4_canon_bytes(pat):
    b = pat.tobytes()
    for k in range(1, 4):
        b = min(b, np.rot90(pat, k).tobytes())
    f = np.fliplr(pat)
    b = min(b, f.tobytes())
    for k in range(1, 4):
        b = min(b, np.rot90(f, k).tobytes())
    return b


def edge_transitions(p):
    a = p.astype(bool)
    return (np.abs(np.diff(a, axis=1)).sum() +
            np.abs(np.diff(a, axis=0)).sum()) / 2.0


def n_holes(p):
    lab, nc = ndimage.label(p == 0)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    return nc - len(border)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data", "metadit"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    root = None
    for cand in (os.path.join(args.data, "split_data"), args.data):
        if all(os.path.exists(os.path.join(cand, f"{s}_set.mat"))
               for s in ("train", "val", "test")):
            root = cand
            break
    assert root, "dataset splits not staged"

    d = io.loadmat(os.path.join(root, "train_set.mat"))
    occ_all = d["pattern"].reshape(-1, d["pattern"].shape[-1]).sum(axis=0)

    params = np.concatenate(
        [io.loadmat(os.path.join(root, f"{s}_set.mat"))["parameter"]
         for s in ("train", "val", "test")], axis=0)
    scalars = params.mean(axis=0)
    print(f"dataset mean scalars (l, h, r) over {len(params)} items: "
          f"{scalars.tolist()}", flush=True)

    targets = np.percentile(occ_all, np.linspace(8, 92, 20)).astype(int)
    designs, tags, sym_flags = [], [], []
    for sym, label in ((True, "sym"), (False, "asym")):
        for k, t in enumerate(targets):
            pat, (kx, ky) = gen(int(t), sym)
            designs.append(pat)
            tags.append(f"{label}{k:02d}")
            sym_flags.append((kx, ky))
            assert int(pat.sum()) > 300, tags[-1]

    ds_canon = set()
    for split in ("train", "val", "test"):
        dd = io.loadmat(os.path.join(root, f"{split}_set.mat"))
        pp = dd["pattern"]
        for i in range(pp.shape[-1]):
            ds_canon.add(hashlib.md5(d4_canon_bytes(
                np.ascontiguousarray(pp[:, :, i]))).digest())
    novel = [hashlib.md5(d4_canon_bytes(p)).digest() not in ds_canon
             for p in designs]
    print(f"novel under D4 vs {len(ds_canon)} dataset items: "
          f"{sum(novel)}/{len(novel)}", flush=True)
    assert all(novel), "collision with the dataset"

    for k, (pat, (kx, ky)) in enumerate(zip(designs, sym_flags)):
        sx = bool((pat == np.fliplr(pat)).all())
        sy = bool((pat == np.flipud(pat)).all())
        occ = int(pat.sum())
        print(f"  {tags[k]:6s} occ={occ:4d} edge/sqrt={edge_transitions(pat) / np.sqrt(occ):.2f} "
              f"holes={n_holes(pat)} sym_x={sx} sym_y={sy}", flush=True)
        if k < 20:
            assert sx or sy, f"{tags[k]} not symmetric"
            assert (kx == sx if kx else True) and (ky == sy if ky else True), tags[k]
        else:
            assert not sx and not sy, f"{tags[k]} accidentally symmetric"

    np.savez(os.path.join(args.out, "batch40_designs.npz"),
             patterns=np.stack(designs), scalars=scalars,
             names=np.array(tags))
    print("saved", os.path.join(args.out, "batch40_designs.npz"), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 10, figsize=(18, 8))
    for r in range(4):
        for c in range(10):
            i = r * 10 + c
            ax = axes[r, c]
            ax.imshow(designs[i], cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(f"{tags[i]} ({int(designs[i].sum())})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("batch40 — 20 symmetric (top two rows) + 20 asymmetric "
                 "(bottom two rows), shared scalars", fontsize=12)
    fig.tight_layout()
    png = os.path.join(args.out, "batch40_preview.png")
    fig.savefig(png, dpi=110)
    print("saved", png, flush=True)


if __name__ == "__main__":
    main()
