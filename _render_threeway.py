"""Three-way render: novel originals, our JEPA decode, MetaDiT-S decode.

Same input for both models -- the identical [2,301] Meep-derived condition
tensors. Row order:

  1  original novel geometry (not in the MetaDiT dataset)
  2  our unified 192-D JEPA, full mask
  3  our unified 192-D JEPA, 50% mask
  4  MetaDiT-S diffusion decode (seed 0, cfg 4.0)

Also reports IoU against the original and the style statistics, so the visual
comparison is backed by numbers rather than left to the eye. Note MetaDiT's
restore_structure mirrors a 32x32 quadrant, so its outputs are exactly
mirror-symmetric by construction; ours are not.

Usage: python _render_threeway.py <b40_meep.json> <decoded_dir> <metadit_npz>
"""

import json
import os
import sys

import numpy as np
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _render_vs_metadit import (edge_density, mirror_score, n_components,  # noqa: E402
                                n_holes, rot_score)

REPO = os.path.dirname(os.path.abspath(__file__))
N_SHOW = 10


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


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
    meep_path, dec_dir, md_path = sys.argv[1:4]
    # Key on NAMES throughout: the Meep entries carry `idx` in three
    # different index spaces (originals 0..39, decodes 0..37 over the
    # passivity-clean subset), but names are unique across all three runs.
    orig = {(e.get("name") or str(e["idx"])).split("/")[-1]
            for e in _load_json(meep_path)["designs"] if e.get("passivity_ok")}
    print(f"original passivity-clean geometries: {len(orig)}")

    z = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    df = _load_npz(os.path.join(dec_dir, "decoded_full.npz"))
    dh = _load_npz(os.path.join(dec_dir, "decoded_half.npz"))
    md = _load_npz(md_path)
    names = z["names"].tolist()

    rows = {names[int(i)]: k for k, i in enumerate(df["source_idx"].tolist())}
    md_names = {str(n).split("/")[-1]: k for k, n in enumerate(md["names"])}

    common = [n for n in names if n in rows and n in md_names and n in orig]
    if not common:
        sys.exit("no designs matched across the three runs")
    print(f"comparable: {len(common)} designs "
          f"(JEPA full+half and MetaDiT-S all present)")
    cols = common[:N_SHOW]

    ious = {"jepa_full": [], "jepa_half": [], "metadit": []}
    for n in common:
        o = z["patterns"][names.index(n)]
        ious["jepa_full"].append(iou(df["patterns"][rows[n]], o))
        ious["jepa_half"].append(iou(dh["patterns"][rows[n]], o))
        ious["metadit"].append(iou(md["patterns"][md_names[n]], o))

    fig, axes = plt.subplots(4, len(cols), figsize=(1.35 * len(cols), 5.8))
    axes = np.atleast_2d(axes)
    layers = [("original\n(novel)", lambda n: z["patterns"][names.index(n)]),
              ("JEPA\nfull mask", lambda n: df["patterns"][rows[n]]),
              ("JEPA\n50% mask", lambda n: dh["patterns"][rows[n]]),
              ("MetaDiT-S\ndiffusion", lambda n: md["patterns"][md_names[n]])]
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
    fig.suptitle("Same input, three geometries: original novel / our JEPA "
                 "(full + 50% mask) / MetaDiT-S", fontsize=12)
    fig.tight_layout()
    out = os.path.join(REPO, "threeway_vs_metadit.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)

    print(f"\n=== IoU vs the original novel geometry (n={len(common)}) ===")
    for k, lab in (("jepa_full", "JEPA full mask"), ("jepa_half", "JEPA 50% mask"),
                   ("metadit", "MetaDiT-S diffusion")):
        v = np.array(ious[k])
        print(f"  {lab:20s} mean {v.mean():.3f}  median {np.median(v):.3f}  "
              f"min {v.min():.3f}  max {v.max():.3f}")

    print(f"\n=== style statistics ===")
    print(f"{'set':22s} {'occ':>6s} {'holes':>6s} {'edge':>6s} "
          f"{'mirror':>7s} {'comps':>6s}")
    sets = {
        "original novel": [z["patterns"][names.index(n)] for n in common],
        "JEPA full mask": [df["patterns"][rows[n]] for n in common],
        "JEPA 50% mask": [dh["patterns"][rows[n]] for n in common],
        "MetaDiT-S": [md["patterns"][md_names[n]] for n in common],
    }
    for k, v in sets.items():
        s = style(v)
        print(f"{k:22s} {s['occ']:6.3f} {s['holes']:6.2f} {s['edge']:6.3f} "
              f"{s['mirror']:7.3f} {s['comps']:6.2f}")
    print("  (real MetaDiT corpus reference: occ 0.428 holes 0.55 edge 0.125 "
          "mirror 1.000 comps 1.12)")


if __name__ == "__main__":
    main()