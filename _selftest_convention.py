"""Self-test for _fit_meep_convention.py.

Two bugs slipped into the per-frequency 2x2 fit before it was ever run against
real data (a 4-regressor/3-regressor mismatch, then X^T y accumulated as sum y).
This checks the fitters against synthetic data where the answer is known, so a
shape or algebra error shows up as a large residual instead of silently
producing a plausible-looking convention.

Run: python _selftest_convention.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fit_meep_convention as F  # noqa: E402

NF = F.NFREQ
RNG = np.random.default_rng(0)


def make_fake(n_designs=19, seed=0):
    """Synthetic Meep spectra + CST targets related by a KNOWN 2x2 map.

    Both the real and imaginary parts vary independently per design. That
    matters: if im were identical across designs (as in a first attempt at
    this fixture), the regression design matrix [Re, Im, 1] would be rank
    deficient -- column Im would be collinear with the constant column -- and
    the ridge term would return an arbitrary solution that fits nothing.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, NF)
    entries, targets = [], []
    for _ in range(n_designs):
        re = (np.sin(6 * u + rng.uniform(0, 6.28))
              * (0.4 + 0.6 * rng.random()) + rng.uniform(-0.3, 0.3))
        im = (np.cos(5 * u + rng.uniform(0, 6.28))
              * (0.3 + 0.6 * rng.random()) + rng.uniform(-0.3, 0.3))
        entries.append({"modes": {"def": {"re": re.tolist(),
                                          "im": im.tolist()}}})
        # ground truth per-frequency complex 2x2 map + bias
        a = 1.1 + 0.05 * np.sin(3 * u)
        b = -0.4 + 0.03 * np.cos(2 * u)
        c = 0.7 + 0.02 * np.sin(5 * u)
        d = 1.3 - 0.04 * np.cos(4 * u)
        gr = a * re - b * im + 0.05
        gi = c * re + d * im - 0.03
        targets.append((gr, gi))
    return entries, targets


def main():
    entries, targets = make_fake()
    train = [(i, e, gr, gi)
             for i, (e, (gr, gi)) in enumerate(zip(entries, targets))]

    print("=== full 2x2 map must recover a known transform ===")
    errs = []
    for h in range(5):                      # small LOO sample for speed
        tr = [d for i, d in enumerate(train) if i != h]
        conv = F.fit_full(tr, conj=False)
        pred = F.apply_full(train[h][1], conv)
        e, c = F.score(pred, train[h][2], train[h][3])
        errs.append(e)
    print(f"  held-out err mean={np.mean(errs):.2e} max={max(errs):.2e}")
    assert np.mean(errs) < 1e-3, "fit_full failed to recover a known 2x2 map"

    print("=== scalar map on the same data (should be worse, not exact) ===")
    errs_s = []
    for h in range(5):
        tr = [d for i, d in enumerate(train) if i != h]
        conv = F.fit_scalar(tr, conj=False)
        e, _ = F.score(F.apply_scalar(train[h][1], conv), train[h][2],
                       train[h][3])
        errs_s.append(e)
    print(f"  held-out err mean={np.mean(errs_s):.2e}")
    assert np.mean(errs_s) > np.mean(errs), (
        "scalar should not beat the exact 2x2 map on 2x2-generated data")

    print("=== shape contracts ===")
    conv = F.fit_full(train, conj=True)
    c = np.array(conv["coef"])
    assert c.shape == (NF, 6), c.shape
    p = F.apply_full(train[0][1], conv)
    assert p.shape == (NF,), p.shape
    assert np.iscomplexobj(p), "apply_full must return complex"
    print(f"  full coef {c.shape} -> complex {p.shape} OK")

    conv = F.fit_scalar(train, conj=False)
    assert len(conv["g_real"]) == NF and len(conv["g_imag"]) == NF
    p = F.apply_scalar(train[0][1], conv)
    assert p.shape == (NF,) and np.iscomplexobj(p)
    print("  scalar gains OK")

    conv = F.fit_phase(train, conj=True)
    p = F.apply_phase(train[0][1], conv)
    assert p.shape == (NF,) and np.iscomplexobj(p)
    print("  phase OK")

    print("\nSELFTEST PASS")


if __name__ == "__main__":
    main()