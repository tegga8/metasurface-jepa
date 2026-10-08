"""Side-by-side: real MetaDiT geometries vs our novel designs vs the decodes.

Row 1 is real geometry taken from MetaDiT's own validation split (the same 20
occupancy-stratified items used to calibrate the Meep harness, so the columns
are comparable). Rows 2-4 are the novel designs and the two decode settings.

Also reports STYLE STATISTICS, because "looks like MetaDiT" is a claim about a
distribution, not about one picture. Compared per class:

  occupancy      fraction of the 64x64 cell filled
  holes          connected background components fully enclosed by material
  edge_density   boundary pixel count / 64 -- how finely the outline is cut
  mirror_score   mean IoU(P, g(P)) over the mirror group {I, fliplr, flipud,
                 rot180}; dense blobs overlap by ~0.4 by chance, so ~0.4 means
                 no symmetry
  rot_score      same over the 4 rotations
  components     connected material blobs

Usage: python _render_vs_metadit.py <decoded_dir>
"""

import json
import os
import sys

import numpy as np
from scipy import io
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")


def _load_npz(path):
    try:
        return np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


def iou(a, b):
    a = a > 0.5
    b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def mirror_score(p):
    ts = [p, np.fliplr(p), np.flipud(p), np.rot90(p, 2)]
    return float(np.mean([iou(p, t) for t in ts]))


def rot_score(p):
    return float(np.mean([iou(p, np.rot90(p, k)) for k in range(4)]))


def edge_density(p):
    """Fraction of boundary pixels: outline pixels per cell area."""
    m = p > 0.5
    b = np.zeros_like(m)
    b[1:, :] |= m[1:, :] != m[:-1, :]
    b[:-1, :] |= m[1:, :] != m[:-1, :]
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    b[:, :-1] |= m[:, 1:] != m[:, :-1]
    return float(b.sum()) / m.size


def n_holes(p):
    """Background components not touching the border."""
    m = p > 0.5
    bg = ~m
    lab, n = ndimage.label(bg)
    if n == 0:
        return 0
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    return int(n - len(border))


def n_components(p):
    lab, n = ndimage.label(p > 0.5)
    return int(n)


def stats(pats):
    return dict(
        occ=float(np.mean([p.mean() for p in pats])),
        holes=float(np.mean([n_holes(p) for p in pats])),
        edge=float(np.mean([edge_density(p) for p in pats])),
        mirror=float(np.mean([mirror_score(p) for p in pats])),
        rot=float(np.mean([rot_score(p) for p in pats])),
        comps=float(np.mean([n_components(p) for p in pats])),
    )


def main():
    dec_dir = sys.argv[1]
    nshow = 12

    try:
        d = io.loadmat(SPLIT)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read MetaDiT split {SPLIT}: {exc}")
    pat_v, par_v = d["pattern"], d["parameter"]
    n_items = pat_v.shape[-1]
    occ_v = pat_v.reshape(-1, n_items).sum(axis=0)
    cand = np.where((occ_v >= 1040) & (occ_v <= 2240))[0]
    order = cand[np.argsort(occ_v[cand])]
    picks = [int(order[int((i + 0.5) / nshow * (len(order) - 1))])
             for i in range(nshow)]
    real = [((pat_v[:, :, i] == 1).astype(np.uint8)) for i in picks]

    z = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    df = _load_npz(os.path.join(dec_dir, "decoded_full.npz"))
    dh = _load_npz(os.path.join(dec_dir, "decoded_half.npz"))
    names = z["names"].tolist()
    src = df["source_idx"].tolist()
    rows = {names[int(i)]: k for k, i in enumerate(src)}

    # Pick novel designs spanning both classes for the picture, restricted to
    # those the model actually produced a decode for (asym02 and asym15 failed
    # the passivity gate and were dropped before decoding).
    want = ([n for n in names if n.startswith("sym")][:nshow // 2]
            + [n for n in names if n.startswith("asym")][:nshow // 2])
    novel_names = [n for n in want if n in rows][:nshow]
    novel = [z["patterns"][names.index(n)] for n in novel_names]
    decf = [df["patterns"][rows[n]] for n in novel_names]
    dech = [dh["patterns"][rows[n]] for n in novel_names]

    rowsimg = [("real MetaDiT\n(val split)", real[:len(novel_names)]),
               ("our novel\n(original)", novel),
               ("decoded\nfull mask", decf),
               ("decoded\n50% mask", dech)]
    ncol = len(novel_names)
    fig, axes = plt.subplots(4, ncol, figsize=(1.25 * ncol, 5.4))
    axes = np.atleast_2d(axes)
    for r, (lab, imgs) in enumerate(rowsimg):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"occ={int(imgs[c].sum())}", fontsize=7)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
    fig.suptitle("Real MetaDiT geometries vs our novel designs vs the model's "
                 "decodes", fontsize=12)
    fig.tight_layout()
    out = os.path.join(REPO, "vs_metadit.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)

    # Style statistics over the FULL sets, not just the pictured ones.
    allreal = [((pat_v[:, :, i] == 1).astype(np.uint8))
               for i in cand[::max(1, len(cand) // 400)]]
    sets = {
        "real MetaDiT (n=%d)" % len(allreal): allreal,
        "our novel all 40": list(z["patterns"]),
        "decoded full mask": list(df["patterns"]),
        "decoded 50% mask": list(dh["patterns"]),
    }
    print(f"\n{'set':26s} {'occ':>6s} {'holes':>6s} {'edge':>6s} "
          f"{'mirror':>7s} {'rot':>6s} {'comps':>6s}")
    for k, v in sets.items():
        s = stats(v)
        print(f"{k:26s} {s['occ']:6.3f} {s['holes']:6.2f} {s['edge']:6.3f} "
              f"{s['mirror']:7.3f} {s['rot']:6.3f} {s['comps']:6.2f}")


if __name__ == "__main__":
    main()