"""Side-by-side: the same novel geometries decoded by three models.

Rows, all conditioned on the IDENTICAL Meep-derived [2,301] targets:
  1  original novel geometry (not in the MetaDiT dataset)
  2  v2 decode  (sched-fix checkpoint; the arm whose gate FAILED, 0.0531)
  3  v1 decode  (retrained base, v1 staging, the 0.0490 protocol)
  4  MetaDiT-S  (released diffusion model, seed 0, cfg 4.0)

Reports IoU against the original and the style statistics, so the visual
comparison is backed by numbers rather than left to the eye.

Usage: python _render_v1_v2_metadit.py <dec_v2_dir> <dec_v1_dir> <metadit_npz>
"""

import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _render_vs_metadit import (edge_density, mirror_score,  # noqa: E402
                                n_components, n_holes)

REPO = os.path.dirname(os.path.abspath(__file__))
N_SHOW = 10


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


def style(pats):
    return dict(
        occ=float(np.mean([p.mean() for p in pats])),
        holes=float(np.mean([n_holes(p) for p in pats])),
        edge=float(np.mean([edge_density(p) for p in pats])),
        mirror=float(np.mean([mirror_score(p) for p in pats])),
        comps=float(np.mean([n_components(p) for p in pats])),
    )


def main():
    v2_dir, v1_dir, md_path = sys.argv[1:4]
    b = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    names = b["names"].tolist()
    v2 = _load_npz(os.path.join(v2_dir, "decoded_full.npz"))
    v1 = _load_npz(os.path.join(v1_dir, "decoded_full.npz"))
    md = _load_npz(md_path)

    def rowmap(d):
        return {names[int(i)]: k for k, i in enumerate(d["source_idx"].tolist())}

    r2, r1 = rowmap(v2), rowmap(v1)
    md_names = {str(x).split("/")[-1]: k for k, x in enumerate(md["names"])}
    common = [n for n in names if n in r2 and n in r1 and n in md_names]
    print(f"comparable across all three: {len(common)} designs")
    cols = common[:N_SHOW]

    fig, axes = plt.subplots(4, len(cols), figsize=(1.35 * len(cols), 5.9))
    axes = np.atleast_2d(axes)
    layers = [
        ("original\n(novel)", lambda n: b["patterns"][names.index(n)]),
        ("v2 decode\n(sched-fix)", lambda n: v2["patterns"][r2[n]]),
        ("v1 decode\n(retrained)", lambda n: v1["patterns"][r1[n]]),
        ("MetaDiT-S\ndiffusion", lambda n: md["patterns"][md_names[n]]),
    ]
    for r, (lab, get) in enumerate(layers):
        for c, n in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(get(n), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(n, fontsize=7)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
    fig.suptitle("One input, three decoders: v2 (failed gate) / v1 (retrained "
                 "base) / MetaDiT-S", fontsize=12)
    fig.tight_layout()
    out = os.path.join(REPO, "v1_v2_metadit_compare.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)

    print(f"\n=== IoU vs the original novel geometry (n={len(common)}) ===")
    for lab, get in (("v2 (sched-fix)", layers[1][1]),
                     ("v1 (retrained)", layers[2][1]),
                     ("MetaDiT-S", layers[3][1])):
        v = np.array([iou(get(n), layers[0][1](n)) for n in common])
        print(f"  {lab:16s} mean {v.mean():.3f}  median {np.median(v):.3f}  "
              f"min {v.min():.3f}  max {v.max():.3f}")

    print(f"\n=== style statistics ===")
    print(f"{'set':16s} {'occ':>6s} {'holes':>6s} {'edge':>6s} {'mirror':>7s} "
          f"{'comps':>6s}")
    for lab, get in (("original novel", layers[0][1]),
                     ("v2 (sched-fix)", layers[1][1]),
                     ("v1 (retrained)", layers[2][1]),
                     ("MetaDiT-S", layers[3][1])):
        s = style([get(n) for n in common])
        print(f"{lab:16s} {s['occ']:6.3f} {s['holes']:6.2f} {s['edge']:6.3f} "
              f"{s['mirror']:7.3f} {s['comps']:6.2f}")
    print("  (real MetaDiT corpus: occ 0.428 holes 0.55 edge 0.125 mirror "
          "1.000 comps 1.12)")


if __name__ == "__main__":
    main()