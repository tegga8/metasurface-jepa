"""Symmetrise the decodes, then judge whether fragmentation is a symmetry fault.

MetaDiT's corpus is exactly mirror-symmetric because restore_structure mirrors a
32x32 quadrant, and its decodes score mirror 1.000 with 1.47 components and 0.21
holes. Ours score 0.90 / 3.50 components / 2.05 holes on identical input. That
raises a concrete question: is the fragmentation merely broken symmetry, or a
deeper decoder fault?

For each decode P the group-invariant version is the intersection over the mirror
group {I, fliplr, flipud, rot180}:

    sym(P) = P & fliplr(P) & flipud(P) & rot180(P)

which is by construction exactly mirror-symmetric. This reports IoU and the style
statistics before and after. Reading:

  * if IoU barely drops and fragmentation largely clears, the fault is broken
    symmetry and the decoder's geometry is otherwise sound;
  * if fragmentation survives symmetrisation, the fault is structural -- the
    decoder is emitting genuinely disconnected, hole-riddled bodies regardless.

Symmetrising is a no-op on the geometry IoU measure's intent -- it can only
shrink the pattern -- so an IoU drop is expected; the question is how much.

Usage: python _symmetrise_decodes.py <decoded_dir> <out_dir>
"""

import json
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _render_vs_metadit import (edge_density, mirror_score, n_components,  # noqa: E402
                                n_holes)


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


def symmetrise(p):
    """Largest subset invariant under the mirror group."""
    b = p > 0
    return (b & np.fliplr(b) & np.flipud(b) & np.rot90(b, 2)).astype(np.uint8)


def style(pats):
    return dict(
        occ=float(np.mean([p.mean() for p in pats])),
        holes=float(np.mean([n_holes(p) for p in pats])),
        edge=float(np.mean([edge_density(p) for p in pats])),
        mirror=float(np.mean([mirror_score(p) for p in pats])),
        comps=float(np.mean([n_components(p) for p in pats])),
    )


def main():
    dec_dir, out_dir = sys.argv[1:3]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.exit(f"could not create {out_dir}: {exc}")
    z = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    names = z["names"].tolist()

    for tag in ("full", "half"):
        d = _load_npz(os.path.join(dec_dir, f"decoded_{tag}.npz"))
        src = d["source_idx"].tolist()
        pats = d["patterns"]
        sym = np.stack([symmetrise(p) for p in pats])
        np.savez(os.path.join(out_dir, f"decoded_{tag}.npz"), patterns=sym,
                 scalars=d["scalars"], names=d["names"],
                 source_idx=d["source_idx"])
        rows = {names[int(i)]: k for k, i in enumerate(src)}
        orig = [z["patterns"][names.index(n)] for n, k in
                sorted(rows.items(), key=lambda kv: kv[1])]
        before, after = [], []
        for k, o in enumerate(orig):
            before.append(iou(pats[k], o))
            after.append(iou(sym[k], o))
        b, a = np.array(before), np.array(after)
        print(f"\n=== {tag} mask (n={len(sym)}) ===")
        print(f"  IoU vs original   before {b.mean():.3f} -> after {a.mean():.3f}"
              f"   (delta {a.mean() - b.mean():+.3f})")
        print(f"  emptied cells     before {(pats.mean() - sym.mean()):.4f}"
              f"  ({100 * (pats.mean() - sym.mean()) / max(pats.mean(), 1e-9):.1f}%)")
        print(f"  {'set':22s} {'occ':>6s} {'holes':>6s} {'edge':>6s} "
              f"{'mirror':>7s} {'comps':>6s}")
        for lab, ps in (("before", pats), ("after symmetrised", sym)):
            s = style(list(ps))
            print(f"  {lab:22s} {s['occ']:6.3f} {s['holes']:6.2f} {s['edge']:6.3f} "
                  f"{s['mirror']:7.3f} {s['comps']:6.2f}")
    print("\n  (real MetaDiT corpus: occ 0.428 holes 0.55 edge 0.125 mirror 1.000 "
          "comps 1.12 | MetaDiT-S decode: 0.398 0.21 0.121 1.000 1.47)")
    print(f"\nsaved symmetrised decodes to {out_dir}")


if __name__ == "__main__":
    main()