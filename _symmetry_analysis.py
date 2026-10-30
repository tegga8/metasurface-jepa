"""Break the novel round-trip down by symmetry class.

The 40 designs are 20 "symmetric" and 20 asymmetric. Pooling them hides the
interesting question, so this splits every reported quantity by class.

ON THE SYMMETRY METRIC: the generator (scripts/data/make_random_batch40.py)
enforces MIRROR symmetry -- `mask & np.fliplr(mask)`, optionally with a second
flip -- not 4-fold rotational symmetry. Measuring D4 (rotations x reflections)
on these therefore scores a property they were never built to have, and gives a
misleading ~0.75 for the originals. Both groups are reported here:

  mirror_score(P) = mean IoU(P, g(P)) over {I, fliplr, flipud, rot180}
                    -- the symmetry the generator actually enforces
  rot_score(P)    = mean IoU(P, g(P)) over the 4 rotations
                    -- reported for contrast; ~chance for these designs

For dense blobs at ~44% fill, two unrelated patterns still overlap by ~0.4, so
a score near 0.4 means no symmetry survives.
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
NFREQ = 301


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


def load_by_name(path):
    out = {}
    for e in _load_json(path).get("designs", []):
        if e.get("passivity_ok"):
            out[(e.get("name") or str(e["idx"])).split("/")[-1]] = e
    return out


def curve(e):
    return np.abs(np.array(e["modes"]["def"]["re"])
                  + 1j * np.array(e["modes"]["def"]["im"]))


def iou(a, b):
    a = a > 0.5
    b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def mirror_transforms(p):
    """The mirror group the generator actually enforces: I, fliplr, flipud, rot180."""
    return [p, np.fliplr(p), np.flipud(p), np.rot90(p, 2)]


def rot_transforms(p):
    return [np.rot90(p, k) for k in range(4)]


def group_score(p, transforms):
    return float(np.mean([iou(p, t) for t in transforms]))


def sym_score(p):
    """Mirror symmetry: the property these designs were generated with."""
    return group_score(p, mirror_transforms(p))


def rot_score(p):
    return group_score(p, rot_transforms(p))


def metrics(a, b):
    std = max(float(b.std()), 1e-12)
    err = float(np.abs(a - b).mean() / std)
    corr = float(np.corrcoef(np.abs(a), np.abs(b))[0, 1]) if a.std() > 0 else 0.0
    return err, corr


def main():
    orig_p, full_p, half_p, dec_dir = sys.argv[1:5]
    orig, full, half = (load_by_name(p) for p in (orig_p, full_p, half_p))
    try:
        z = np.load(os.path.join(REPO, "batch40_designs.npz"),
                    allow_pickle=True)
        df = np.load(os.path.join(dec_dir, "decoded_full.npz"),
                     allow_pickle=True)
        dh = np.load(os.path.join(dec_dir, "decoded_half.npz"),
                     allow_pickle=True)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read designs/decodes: {exc}")

    names = z["names"].tolist()
    rows = {names[int(i)]: k for k, i in enumerate(df["source_idx"].tolist())}
    common = [n for n in names if n in rows and n in orig and n in full
              and n in half]
    cls = {n: ("sym" if n.startswith("sym") else "asym") for n in common}

    print(f"{'design':10s} {'class':5s} {'IoU full':>9s} {'IoU half':>9s} "
          f"{'mir orig':>9s} {'mir full':>9s} {'mir half':>9s} "
          f"{'rot orig':>9s}")
    rec = []
    for n in common:
        k = rows[n]
        o = z["patterns"][names.index(n)]
        of, oh = df["patterns"][k], dh["patterns"][k]
        rec.append(dict(name=n, cls=cls[n],
                        iou_full=iou(of, o), iou_half=iou(oh, o),
                        sym_o=sym_score(o), sym_f=sym_score(of),
                        sym_h=sym_score(oh), rot_o=rot_score(o)))
        r = rec[-1]
        print(f"{n:10s} {r['cls']:5s} {r['iou_full']:9.3f} {r['iou_half']:9.3f} "
              f"{r['sym_o']:9.3f} {r['sym_f']:9.3f} {r['sym_h']:9.3f} "
              f"{r['rot_o']:9.3f}")

    print("\n=== by class (mean) ===")
    print(f"{'class':6s} {'n':>3s} {'IoU full':>9s} {'IoU half':>9s} "
          f"{'mir orig':>9s} {'mir full':>9s} {'mir half':>9s} "
          f"{'rot orig':>9s}")
    for c in ("sym", "asym"):
        s = [r for r in rec if r["cls"] == c]
        if not s:
            continue

        def m(k):
            return float(np.mean([r[k] for r in s]))
        print(f"{c:6s} {len(s):3d} {m('iou_full'):9.3f} {m('iou_half'):9.3f} "
              f"{m('sym_o'):9.3f} {m('sym_f'):9.3f} {m('sym_h'):9.3f} "
              f"{m('rot_o'):9.3f}")

    # Meep-vs-Meep, split by class
    print("\n=== Meep-vs-Meep, split by class ===")
    for tag, dec in (("full mask", full), ("half mask", half)):
        print(f"\n  -- {tag} --")
        for c in ("sym", "asym"):
            sub = [n for n in common if cls[n] == c]
            if len(sub) < 4:
                continue
            perm = RNG.permutation(len(sub))
            re_r, sh_r, re_c, sh_c = [], [], [], []
            for i, n in enumerate(sub):
                er, cr = metrics(curve(dec[n]), curve(orig[n]))
                es, cs = metrics(curve(dec[n]), curve(orig[sub[perm[i]]]))
                re_r.append(er); re_c.append(cr)
                sh_r.append(es); sh_c.append(cs)
            print(f"   {c:5s} n={len(sub):2d}  REAL err={np.mean(re_r):.4f} "
                  f"corr={np.mean(re_c):+.4f} | SHUFFLED err={np.mean(sh_r):.4f} "
                  f"corr={np.mean(sh_c):+.4f} | median err margin="
                  f"{np.median(np.array(re_r) - np.array(sh_r)):+.4f} "
                  f"corr margin={np.median(np.array(re_c) - np.array(sh_c)):+.4f}")


if __name__ == "__main__":
    main()