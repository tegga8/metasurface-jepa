"""Generate symbolic meta-atom geometries as occupancy patterns (64 x 64).

Requested set (operator, 2026-10-05): swastika, cross, crescent moon, aum,
khanda (Sikh), dharmachakra (Buddhist), yin-yang, torii (Shinto), faravahar
(Zoroastrian).

Contract and conventions mirror the free-form novel set
(`_novel_designs.py`) so the Meep runner (`_meep_novel.py`) and the model
round-trip tester consume these identically:
  * output npz holds `patterns` (N x 64 x 64 uint8) + shared `scalars` (3,),
  * the same big-solid / smooth-boundary morphology treatment (closing,
    opening, small-hole fill, speck removal) is applied to every glyph,
  * the same D4-canonical byte-exact novelty check runs against the released
    dataset (loud skip if the splits are not staged).

Unlike the free-form set these are DELIBERATE glyphs: each is drawn with the
widest solid strokes that keep the symbol legible at 64 px, then smoothed.
Occupancy therefore lands below the dataset's blob percentile band and is
reported per design rather than forced.

Usage: python make_symbol_designs.py [--out DIR] [--data DIR]
Writes symbol_designs.npz and symbol_designs_preview.png to --out.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
from scipy import io, ndimage

SIZE = 64
CC = (SIZE - 1) / 2.0  # 31.5
SCALARS = (2.75, 0.75, 4.25)

_Y, _X = np.ogrid[0:SIZE, 0:SIZE]


def disk(cx, cy, r):
    return ((_X - cx) ** 2 + (_Y - cy) ** 2) <= r * r


def rect(x0, x1, y0, y1):
    return (_X >= x0) & (_X <= x1) & (_Y >= y0) & (_Y <= y1)


def ring(cx, cy, r_out, r_in):
    return disk(cx, cy, r_out) & ~disk(cx, cy, r_in)


def _seg_dist(ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 == 0:
        return np.hypot(_X - ax, _Y - ay)
    t = np.clip(((_X - ax) * vx + (_Y - ay) * vy) / l2, 0.0, 1.0)
    return np.hypot(_X - (ax + t * vx), _Y - (ay + t * vy))


def stroke(pts, width):
    """Thick polyline (width may be scalar or per-segment list)."""
    pts = [(float(x), float(y)) for x, y in pts]
    ws = [width] * (len(pts) - 1) if np.isscalar(width) else list(width)
    out = np.zeros((SIZE, SIZE), dtype=bool)
    for (a, b), w in zip(zip(pts[:-1], pts[1:]), ws):
        out |= _seg_dist(a[0], a[1], b[0], b[1]) <= w / 2.0
    return out


def arc_pts(cx, cy, r, deg0, deg1, n=48):
    t = np.radians(np.linspace(deg0, deg1, n))
    return [(cx + r * np.cos(a), cy - r * np.sin(a)) for a in t]


def poly(pts):
    from matplotlib.path import Path
    p = Path([(float(x), float(y)) for x, y in pts])
    xs, ys = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    return p.contains_points(grid).reshape(SIZE, SIZE)


def sym_swastika():
    m = rect(26, 37, 4, 60) | rect(4, 60, 26, 37)
    m |= rect(37, 53, 4, 15)      # top arm foot -> right
    m |= rect(49, 60, 37, 53)     # right arm foot -> down
    m |= rect(11, 27, 49, 60)     # bottom arm foot -> left
    m |= rect(4, 15, 11, 27)      # left arm foot -> up
    return m


def sym_cross():
    return rect(26, 37, 4, 60) | rect(4, 60, 18, 29)


def sym_crescent():
    return disk(CC, CC, 28) & ~disk(CC + 10.5, CC - 4.5, 25.5)


def sym_aum():
    m = stroke(arc_pts(28, 25, 13.5, 60, 310), 5.5)
    m |= stroke(arc_pts(33.5, 25, 5.5, 75, 285), 4.5)
    m |= stroke(arc_pts(31, 44, 9.5, 60, 300), 5.5)
    m |= stroke([(36, 51.5), (44, 45), (46, 34)], 4.5)
    m |= disk(37.5, 5.5, 2.8)
    m |= stroke(arc_pts(37.5, 10.5, 5.5, 200, 340), 2.6)
    return m


def sym_khanda():
    m = ring(31.5, 32, 19.0, 15.8)
    m |= poly([(31.5, 2.5), (29.5, 10), (28.2, 30), (34.8, 30), (33.5, 10)])
    m |= rect(29.5, 33.5, 30, 44)
    m |= disk(31.5, 47, 3.4)
    m |= stroke([(5, 6), (4.5, 20), (9, 36), (16, 48), (22, 55)], 4.2)
    m |= stroke([(58, 6), (58.5, 20), (54, 36), (47, 48), (41, 55)], 4.2)
    return m


def sym_chakra():
    m = ring(CC, CC, 27.5, 21.0) | disk(CC, CC, 6.8)
    for k in range(12):
        a = np.radians(k * 30.0)
        u = (np.cos(a), -np.sin(a))
        m |= stroke([(CC + 6.0 * u[0], CC + 6.0 * u[1]),
                     (CC + 22.0 * u[0], CC + 22.0 * u[1])], 2.5)
    return m


def sym_yinyang():
    outer = disk(CC, CC, 28)
    small_top = disk(CC, CC - 14, 14)
    small_bottom = disk(CC, CC + 14, 14)
    dark = (outer & (_Y >= CC)) | small_top
    dark &= ~small_bottom
    dark &= ~disk(CC, CC - 14, 4.0)
    dark |= disk(CC, CC + 14, 4.0)
    return dark


def sym_torii():
    m = poly([(2, 15), (61, 15), (58, 6.5), (5, 6.5)])
    m |= rect(9, 54.5, 19.5, 25)
    m |= rect(13, 20.5, 15, 61)
    m |= rect(43, 50.5, 15, 61)
    m |= rect(28.5, 34.5, 15, 19.5)
    return m


def sym_faravahar():
    m = disk(CC, 13.5, 5.2)
    m |= stroke([(CC, 17), (CC, 30)], 8.5)
    m |= stroke([(30, 22), (17, 16), (6, 13), (2.5, 9)], [8.5, 7.0, 4.0])
    m |= stroke([(33, 22), (46, 16), (57, 13), (60.5, 9)], [8.5, 7.0, 4.0])
    m |= poly([(CC, 30), (26.5, 42), (28.5, 54), (31.5, 46),
               (34.5, 54), (36.5, 42)])
    return m


SYMBOLS = [
    ("swastika", sym_swastika),
    ("cross", sym_cross),
    ("crescent", sym_crescent),
    ("aum", sym_aum),
    ("khanda", sym_khanda),
    ("dharmachakra", sym_chakra),
    ("yinyang", sym_yinyang),
    ("torii", sym_torii),
    ("faravahar", sym_faravahar),
]


def clean(mask, keep_all_components=False):
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
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
        if not keep_all_components:
            mask = (lab == np.argmax(sizes))
        else:
            for c in range(1, nc + 1):
                if sizes[c] < 15:
                    mask &= (lab != c)
    return mask


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
    return (np.abs(np.diff(a, axis=1)).sum() + np.abs(np.diff(a, axis=0)).sum()) / 2.0


def n_holes(p):
    lab, nc = ndimage.label(p == 0)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    return nc - len(border)


def load_novelty_hashes(data_dir):
    cands = [
        os.path.join(data_dir, "split_data"),
        data_dir,
    ]
    ds_canon = set()
    for root in cands:
        if all(os.path.exists(os.path.join(root, f"{s}_set.mat"))
               for s in ("train", "val", "test")):
            for split in ("train", "val", "test"):
                dd = io.loadmat(os.path.join(root, f"{split}_set.mat"))
                pp = dd["pattern"]
                for i in range(pp.shape[-1]):
                    ds_canon.add(hashlib.md5(d4_canon_bytes(
                        np.ascontiguousarray(pp[:, :, i]))).digest())
            return ds_canon, root
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "metadit"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    patterns = []
    multi_component = {"khanda", "yinyang"}
    for name, fn in SYMBOLS:
        mask = clean(fn(), keep_all_components=(name in multi_component))
        occ = int(mask.sum())
        assert occ > 250, f"{name}: glyph collapsed (occ={occ})"
        patterns.append(mask.astype(np.uint8))
        print(f"  {name:14s} occ={occ:4d} "
              f"({100.0 * occ / (SIZE * SIZE):.1f}%) "
              f"edge/sqrt(area)={edge_transitions(mask) / np.sqrt(occ):.2f} "
              f"holes={n_holes(mask)}", flush=True)

    ds_canon, root = load_novelty_hashes(args.data)
    if ds_canon is None:
        print("WARNING: dataset splits not staged; novelty check SKIPPED", flush=True)
    else:
        novel = [hashlib.md5(d4_canon_bytes(d)).digest() not in ds_canon
                 for d in patterns]
        print(f"novel under D4 (byte-exact) vs {len(ds_canon)} dataset items "
              f"from {root}:", novel, flush=True)
        assert all(novel), "some symbols collide with the dataset"

    npz_path = os.path.join(args.out, "symbol_designs.npz")
    np.savez(npz_path, patterns=np.stack(patterns),
             scalars=np.array(SCALARS))
    print("saved", npz_path, flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(9, 9.4))
    for ax, (name, _), pat in zip(axes.ravel(), SYMBOLS, patterns):
        ax.imshow(pat, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{name}  occ={int(pat.sum())}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    l, h, r = SCALARS
    fig.suptitle(f"Symbol designs — shared scalars l={l} h={h} r={r}", fontsize=13)
    fig.tight_layout()
    png_path = os.path.join(args.out, "symbol_designs_preview.png")
    fig.savefig(png_path, dpi=110)
    print("saved", png_path, flush=True)


if __name__ == "__main__":
    main()
