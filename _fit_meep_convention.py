"""Fit and validate the Meep->CST convention on the fixed harness.

The model is conditioned on a complex spectrum in MetaDiT's [2,301] (real, imag)
layout, but a Meep run produces a complex transmission coefficient in Meep's own
convention. Something has to map one to the other. The previous
_meep_convention.json did this with a 2-parameter global phase rotation fitted on
4 items, with residuals up to 3898 and one item discarded as unusable -- and it
was fitted on output from the defective harness, so it is not a baseline worth
resurrecting.

This fits several candidate maps on real validation geometries whose true CST
spectra we know, then reports HELD-OUT error, because a convention that only
works on the items it was fitted to is worthless:

  phase    global rotation only:            t -> conj(t)^c * exp(i(a + b*u))
  scalar   per-frequency complex gain:      t -> conj(t)^c * g(f)
  full     per-frequency complex 2x2 map:   [Re,Im] -> M(f) [Re,Im] + b(f)

`c` is the conjugation flag (Meep and CST may differ in the sign convention for
the imaginary part; this is fitted, not assumed). Error is the same normalized
L1 the evaluator uses, mean|pred-target|/std(target), plus |T| correlation.

Designs whose ringdown hit the cap are excluded: their spectra are not
converged, so they cannot calibrate anything.

Usage: python _fit_meep_convention.py <merged_calib.json> [out.json]
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DESIGNS = os.path.join(REPO, "meep_calib_designs.npz")
TARGETS = os.path.join(REPO, "meep_calib_targets.npz")
NFREQ = 301


def load(merged_path):
    try:
        d = json.load(open(merged_path))
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {merged_path}: {exc}")
    designs = {x["idx"]: x for x in d.get("designs", [])}
    z = np.load(DESIGNS, allow_pickle=True)
    t = np.load(TARGETS, allow_pickle=True)
    keep = []
    for i in range(len(z["patterns"])):
        e = designs.get(i)
        if e is None:
            print(f"  skip idx {i}: no Meep result")
            continue
        if not e.get("passivity_ok", False):
            print(f"  skip idx {i}: failed passivity")
            continue
        if e.get("hit_cap"):
            print(f"  skip idx {i}/{e.get('name')}: ringdown hit the cap "
                  f"(not converged)")
            continue
        m = e["modes"].get("def")
        if not m or "re" not in m:
            print(f"  skip idx {i}: no complex mode data (pre-phase-fix run)")
            continue
        keep.append((i, e, t["real"][i], t["imag"][i]))
    return keep


def ugrid(nf=NFREQ):
    return np.arange(nf) / (nf - 1)


def _cplx(re, im):
    """Build an explicit complex128 array from real/imag parts.

    Written this way rather than `a + 1j * b` so the dtype is evident at the
    point of construction instead of having to be inferred.
    """
    out = np.empty(len(re), dtype=np.complex128)
    out.real = np.asarray(re, dtype=float)
    out.imag = np.asarray(im, dtype=float)
    return out


def _meep_t(e, conj):
    """The design's Meep complex transmission, optionally conjugated."""
    m = e["modes"]["def"]
    t = _cplx(m["re"], m["im"])
    return np.conj(t) if conj else t


def fit_phase(train, conj):
    """Global rotation exp(i(a + b*u)): 2 parameters."""
    def obj(p):
        a, b = p
        rot = np.exp(1j * (a + b * ugrid()))
        r = 0.0
        for _, e, gr, gi in train:
            t = _meep_t(e, conj)
            r += float(np.abs(t * rot - _cplx(gr, gi)).sum())
        return r
    from scipy.optimize import minimize
    best = None
    for c0 in (True, False):
        r = minimize(obj, x0=[0.0, 0.0], method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    return {"kind": "phase", "conj": conj, "a": float(a), "b": float(b)}


def apply_phase(e, conv):
    t = _meep_t(e, conv["conj"])
    return t * np.exp(1j * (conv["a"] + conv["b"] * ugrid()))


def fit_scalar(train, conj):
    """Per-frequency complex gain g(f) = cst / meep, averaged over designs."""
    nf = NFREQ
    num = np.zeros(nf, dtype=np.complex128)
    den = np.zeros(nf, dtype=np.complex128)
    for _, e, gr, gi in train:
        t = _meep_t(e, conj)
        y = _cplx(gr, gi)
        num += np.conj(t) * y
        den += np.abs(t) ** 2
    g = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    # Smooth in frequency: a raw per-point ratio is noisy where |t| is small.
    w = 9
    kern = np.full(w, 1.0 / w)
    gg = np.asarray(np.convolve(g, kern, mode="same"), dtype=np.complex128)
    return {"kind": "scalar", "conj": conj, "g_real": gg.real.tolist(),
            "g_imag": gg.imag.tolist()}


def apply_scalar(e, conv):
    t = _meep_t(e, conv["conj"])
    g = _cplx(conv["g_real"], conv["g_imag"])
    return t * g


def fit_full(train, conj):
    """Per-frequency complex 2x2 map + bias: [Re,Im] -> M [Re,Im] + b."""
    nf = NFREQ
    A = np.zeros((nf, 4, 5))
    for i, (_, e, gr, gi) in enumerate(train):
        t = _meep_t(e, conj)
        x = np.stack([t.real, t.imag, np.ones(nf)], axis=1)  # [nf,3]
        y = np.stack([np.asarray(gr, float), np.asarray(gi, float)],
                     axis=1)                                  # [nf,2]
        for j in range(2):
            A[:, j, :4] += x * y[:, j:j + 1]
            A[:, j, 4] += y[:, j]
    lam = 1e-8
    coef = np.zeros((nf, 2, 4))
    for f in range(nf):
        for j in range(2):
            M = A[f, j, :4].reshape(4, 4) + lam * np.eye(4)
            v = A[f, j, 4]
            coef[f, j] = np.linalg.solve(M, v)
    return {"kind": "full", "conj": conj,
            "coef": coef.reshape(nf, 8).tolist()}


def apply_full(e, conv):
    t = _meep_t(e, conv["conj"])
    x = np.stack([t.real, t.imag], axis=1)
    c = np.array(conv["coef"]).reshape(-1, 2, 4)
    out = np.einsum("fj,fjc->fc", x, c[:, :, :2]) + c[:, :, 2]
    return out[:, 0] + 1j * out[:, 1]


APPLY = {"phase": apply_phase, "scalar": apply_scalar, "full": apply_full}
FIT = {"phase": fit_phase, "scalar": fit_scalar, "full": fit_full}


def score(pred, gr, gi):
    target = _cplx(gr, gi)
    std = max(float(target.std()), 1e-12)
    err = float(np.abs(pred - target).mean() / std)
    mt, mp = np.abs(target), np.abs(pred)
    corr = float(np.corrcoef(mt, mp)[0, 1]) if mp.std() > 0 else 0.0
    return err, corr


def main():
    merged = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    keep = load(merged)
    if len(keep) < 4:
        sys.exit(f"only {len(keep)} usable designs; need >= 4")
    print(f"usable designs: {len(keep)}")

    # Leave-one-out over every candidate x conjugation flag.
    results = []
    for kind in ("phase", "scalar", "full"):
        for conj in (False, True):
            errs, corrs = [], []
            for h in range(len(keep)):
                train = [d for i, d in enumerate(keep) if i != h]
                conv = FIT[kind](train, conj)
                e, c = score(APPLY[kind](keep[h][1], conv), keep[h][2], keep[h][3])
                errs.append(e)
                corrs.append(c)
            results.append((float(np.mean(errs)), float(np.mean(corrs)),
                            kind, conj, errs, corrs))
            print(f"[{kind:6s} conj={str(conj):5s}] LOO err={np.mean(errs):.4f} "
                  f"corr={np.mean(corrs):.4f}  (per-design err max "
                  f"{max(errs):.4f})", flush=True)

    results.sort(key=lambda r: r[0])
    best = results[0]
    print(f"\nBEST: {best[2]} conj={best[3]}  held-out err={best[0]:.4f} "
          f"corr={best[1]:.4f}")

    # Refit on everything for the deployed convention.
    full_train = [d for _, d, _, _ in keep]
    conv = FIT[best[2]](full_train, best[3])
    errs = [score(APPLY[best[2]](e, conv), gr, gi)[0] for _, e, gr, gi in keep]
    print(f"in-sample (all {len(keep)}): err={np.mean(errs):.4f}")

    print(f"\n{'design':22s} {'occ':>5} {'heldout err':>12} {'corr':>7}")
    for (i, e, gr, gi), eh, ch in zip(keep, best[4], best[5]):
        print(f"{e.get('name', i):22s} {e['occ']:5d} {eh:12.4f} {ch:7.3f}")

    if out_path:
        conv["loo_err"] = best[0]
        conv["loo_corr"] = best[1]
        conv["n_designs"] = len(keep)
        conv["note"] = ("fitted on Meep output from the fixed harness "
                        "(loss tangent 1e-3); held-out error is "
                        "leave-one-out over the validation geometries")
        try:
            with open(out_path, "w") as f:
                json.dump(conv, f, indent=1)
        except OSError as exc:
            sys.exit(f"could not write {out_path}: {exc}")
        print("saved", out_path)


if __name__ == "__main__":
    main()