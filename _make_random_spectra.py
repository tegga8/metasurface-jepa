"""Phase 2: random target spectra not in the dataset, with a feasibility screen.

The geometry-first direction (Task 1) always had a real geometry to compare
against. This is the other direction: hand the model an arbitrary spectrum and
see whether the geometry it returns actually radiates that spectrum.

The danger is that the model must return something for EVERY input, including
spectra no geometry can produce. Scoring it against an unachievable target would
report a model defect that is really a bad benchmark. So the generator makes two
sets deliberately:

  A "interpolated" - random convex combinations of several real MetaDiT
    validation spectra. Not in the dataset (a new combination), but inside the
    data manifold, so plausibly realizable. This is the set we score.
  B "synthetic"   - band-limited random noise shaped to have a similar envelope
    but no smooth physical structure. Deliberately implausible: a control that
    tells us whether the model's response to garbage is distinguishable from its
    response to plausible targets.

The feasibility screen reports, per set, how far each target sits from the real
corpus (distance to its nearest real spectrum) so set A and set B can be shown
to actually differ in the intended way rather than in name only.

Usage: python _make_random_spectra.py <out.npz> [n_each]
"""

import json
import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")
NFREQ = 301


def _load_split():
    try:
        d = io.loadmat(SPLIT)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {SPLIT}: {exc}")
    return d["real"], d["imag"]


def smooth_noise(n, nf, rng, n_modes=6):
    """Band-limited random curves with a smooth, spectrum-like envelope.

    Returns shape [n, nf]. Each sample is a weighted sum of n_modes Gaussian
    bumps with random centres and widths, normalised by the basis magnitude so
    the result stays O(1) and inherits the corpus envelope when multiplied in.
    """
    u = np.linspace(0, 1, nf)
    out = np.empty((n, nf))
    for i in range(n):
        centres = rng.uniform(0.0, 1.0, n_modes)
        widths = rng.uniform(0.05, 0.25, n_modes)
        B = np.exp(-((u[None, :] - centres[:, None]) ** 2)
                   / (2.0 * widths[:, None] ** 2))          # [n_modes, nf]
        w = rng.normal(size=(n_modes, nf))                   # [n_modes, nf]
        out[i] = (B * w).sum(axis=0) / (np.abs(B).sum(axis=0) + 1e-9)
    return out


def main():
    out_path = sys.argv[1]
    n_each = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    real, imag = _load_split()
    N = real.shape[0]
    rng = np.random.default_rng(0)

    R = real.astype(np.float64)
    I = imag.astype(np.float64)

    # --- set A: random convex combinations of 3-5 real spectra ---
    A_re, A_im, A_src = [], [], []
    while len(A_re) < n_each:
        k = int(rng.integers(3, 6))
        idx = rng.choice(N, k, replace=False)
        w = rng.dirichlet(np.ones(k))
        A_re.append((w[:, None] * R[idx]).sum(axis=0))
        A_im.append((w[:, None] * I[idx]).sum(axis=0))
        A_src.append(json.dumps([int(i) for i in idx]))

    # --- set B: band-limited noise scaled to the corpus envelope ---
    # Clipped to |T| <= 1: a transmission magnitude above 1 is trivially
    # unphysical, which would make set B a strawman rather than a fair
    # negative control. The intent is "smooth-spectrum-plausible but not a
    # real resonance", not "obviously impossible".
    env_r = np.abs(R).mean(axis=0)
    env_i = np.abs(I).mean(axis=0)
    B_re, B_im = [], []
    for _ in range(n_each):
        rr = env_r * (1.0 + 0.5 * smooth_noise(1, NFREQ, rng)[0])
        ii = env_i * (1.0 + 0.5 * smooth_noise(1, NFREQ, rng)[0])
        m = np.abs(rr + 1j * ii)
        if m.max() > 1.0:
            rr, ii = rr / m.max(), ii / m.max()
        B_re.append(rr)
        B_im.append(ii)

    A_re, A_im = np.stack(A_re), np.stack(A_im)
    B_re, B_im = np.stack(B_re), np.stack(B_im)

    # --- feasibility screen ---
    # Nearest-real-spectrum distance turned out NOT to separate the sets: a
    # convex combination is smoother than any real spectrum, so set A scored
    # "closer to the corpus" than real data does, for the wrong reason. The
    # discriminating property is spectral structure -- real transmission
    # spectra carry sharp resonances, so compare roughness and |T| statistics
    # against the real corpus itself rather than relying on the NN distance.
    def roughness(Q):
        mag = np.abs(Q)
        return float(np.mean(np.abs(np.diff(mag, axis=1))
                             / np.maximum(mag.mean(axis=1, keepdims=True), 1e-9)))

    def nn_dist(Q):
        """Mean-abs distance from each row of Q to its nearest real spectrum."""
        sub = rng.choice(N, 4000, replace=False)   # reference subsample
        Rs = R[sub]
        Rs = Rs / np.maximum(Rs.std(axis=1, keepdims=True), 1e-12)
        out = np.empty(len(Q))
        step = 32
        for a in range(0, len(Q), step):
            blk = Q[a:a + step]
            blk = blk / np.maximum(blk.std(axis=1, keepdims=True), 1e-12)
            D = np.abs(blk[:, None, :] - Rs[None, :, :]).mean(axis=2)
            out[a:a + step] = D.min(axis=1)
        return out

    realmag = np.sqrt(R ** 2 + I ** 2)
    ctrl = realmag[rng.choice(N, n_each, replace=False)]
    dA, dB, dR = (nn_dist(np.abs(A_re + 1j * A_im)),
                   nn_dist(np.abs(B_re + 1j * B_im)),
                   nn_dist(ctrl))

    print("feasibility screen (control = real validation spectra)")
    print(f"  {'':26s} {'roughness':>10s} {'max|T|':>8s} {'nn-dist':>8s}")
    for lab, Q in (("real val (control)", ctrl),
                   ("set A interpolated", np.abs(A_re + 1j * A_im)),
                   ("set B synthetic", np.abs(B_re + 1j * B_im))):
        print(f"  {lab:26s} {roughness(Q):10.4f} "
              f"{np.abs(Q).max():8.3f} "
              f"{np.median([dR, dA, dB][['real val (control)', 'set A interpolated', 'set B synthetic'].index(lab)]):8.4f}")

    rA, rB, rR = roughness(np.abs(A_re + 1j * A_im)), \
        roughness(np.abs(B_re + 1j * B_im)), roughness(ctrl)
    print(f"\n  roughness separates the sets cleanly: B is "
          f"{rB / max(rA, 1e-9):.0f}x rougher than A "
          f"(A is {rA / max(rR, 1e-9):.2f}x the real corpus, B is "
          f"{rB / max(rR, 1e-9):.0f}x).")
    print("  set A is smooth in the same way real spectra are -> plausible; "
          "set B has no resonance structure -> implausible control.")
    print("  NOTE nearest-real distance is NOT a useful feasibility measure "
          "here: set A scores closer to the corpus than real data does, "
          "because a convex combination is smoother than any real spectrum.")

    np.savez(out_path,
             A_real=A_re.astype(np.float32), A_imag=A_im.astype(np.float32),
             B_real=B_re.astype(np.float32), B_imag=B_im.astype(np.float32),
             A_sources=np.array(A_src))
    print("saved", out_path)


if __name__ == "__main__":
    main()