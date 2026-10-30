"""Render the novel geometries alongside what the model decoded from them.

Three aligned rows per design: the original novel geometry, the decode under a
full mask (no geometry visible), and the decode under a 50% mask. Aligning them
as triplets makes the thing that actually matters visible by eye: whether the
decode keeps the original's silhouette or collapses onto a generic blob.

A second figure overlays the Meep |T| curves for a handful of designs, real
pairing against the shuffled null, so the spectrum side can be seen too.

Usage: python _render_novel_roundtrip.py <b40_meep.json> <dec_full.json> \
           <dec_half.json> <decoded_dir>

<decoded_dir> holds decoded_full.npz / decoded_half.npz as written by
_decode_novel.py, and is where the decoded occupancy maps come from.
"""

import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
NFREQ = 301
PER_FIG = 12


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
            nm = e.get("name") or str(e["idx"])
            out[nm.split("/")[-1]] = e
    return out


def curve(e):
    return np.abs(np.array(e["modes"]["def"]["re"])
                  + 1j * np.array(e["modes"]["def"]["im"]))


def _load_npz(path):
    try:
        return np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


def main():
    orig_path, full_path, half_path, dec_dir = sys.argv[1:5]
    orig = load_by_name(orig_path)
    full = load_by_name(full_path)
    half = load_by_name(half_path)
    z = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    names_all = z["names"].tolist()

    df = _load_npz(os.path.join(dec_dir, "decoded_full.npz"))
    dh = _load_npz(os.path.join(dec_dir, "decoded_half.npz"))
    pats_full, pats_half = df["patterns"], dh["patterns"]
    # Decoded npz rows are keyed back to their originals only through
    # source_idx, which indexes batch40_designs.npz.
    rows = {names_all[int(i)]: k
            for k, i in enumerate(df["source_idx"].tolist())}

    common = [n for n in names_all if n in rows and n in orig
              and n in full and n in half]
    print(f"comparable designs: {len(common)}")
    orig_pats = {n: z["patterns"][names_all.index(n)] for n in common}

    out_pngs = []
    n_fig = (len(common) + PER_FIG - 1) // PER_FIG
    for fi in range(n_fig):
        chunk = common[fi * PER_FIG:(fi + 1) * PER_FIG]
        n = len(chunk)
        fig, axes = plt.subplots(3, n, figsize=(1.15 * n, 3.9))
        axes = np.atleast_2d(axes)
        for c, nm in enumerate(chunk):
            k = rows.get(nm)
            axes[0, c].imshow(orig_pats[nm], cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            if pats_full is not None and k is not None:
                axes[1, c].imshow(pats_full[k], cmap="gray", vmin=0, vmax=1,
                                  interpolation="nearest")
            if pats_half is not None and k is not None:
                axes[2, c].imshow(pats_half[k], cmap="gray", vmin=0, vmax=1,
                                  interpolation="nearest")
            axes[0, c].set_title(nm, fontsize=7)
            for r in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        for r, lab in enumerate(("original\n(novel)", "decoded\nfull mask",
                                 "decoded\n50% mask")):
            axes[r, 0].set_ylabel(lab, fontsize=7)
        fig.suptitle("Novel geometries vs what the model decoded from their "
                     "Meep spectra", fontsize=11)
        fig.tight_layout()
        out = os.path.join(REPO, f"novel_roundtrip_{fi + 1}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        out_pngs.append(out)
        print("saved", out)

    # spectrum overlay for a few designs
    picks = [n for n in common if n.startswith("sym")][:4] + \
            [n for n in common if n.startswith("asym")][:4]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.5 * len(picks), 5))
    axes = np.atleast_2d(axes)
    freqs = np.linspace(0.1, 0.2, NFREQ)
    for c, nm in enumerate(picks):
        o = curve(orig[nm])
        axes[0, c].plot(freqs, o, "k-", lw=2.2, label="original (Meep)")
        axes[0, c].plot(freqs, curve(full[nm]), color="tab:blue", lw=1.1,
                        alpha=.85, label="decoded full")
        axes[0, c].plot(freqs, curve(half[nm]), color="tab:green", lw=1.1,
                        alpha=.85, ls="--", label="decoded 50%")
        axes[0, c].set_title(nm, fontsize=9)
        axes[1, c].plot(freqs, o, "k-", lw=2.2, label="original (Meep)")
        other = picks[(c + 3) % len(picks)]
        axes[1, c].plot(freqs, curve(orig[other]), color="tab:red", lw=1.1,
                        ls=":", label=f"shuffled ({other})")
        axes[1, c].set_title(f"{nm} vs shuffled", fontsize=9)
        for r in range(2):
            axes[r, c].set_xlabel("freq", fontsize=7)
            axes[r, c].set_ylabel("|T|", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    fig.suptitle("Meep |T|: decoded vs original (top), and the shuffled null "
                 "(bottom)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(REPO, "novel_roundtrip_spectra.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()