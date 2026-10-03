"""Phase 4 — target-separation probe (Task 1 of the target-information check).

Measures how much the frozen SpectrumFilm's conditioning actually separates
`z_y_occ_spec` across DIFFERENT spectra for the SAME occupancy — i.e. how much
spectrum-specific information the L_cond target carries at all.

Local, no training: the film is FROZEN, so its state is (near enough) its init;
`--film-std` re-initializes it in place so the effect of a larger init can be
checked before spending a training run.

Run:
    python scripts/diagnostics/spectrum_film_separation.py \
        --config configs/unified.yaml --samples 8
    python scripts/diagnostics/spectrum_film_separation.py \
        --config configs/unified.yaml --samples 8 --film-std 0.1
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
EVAL = os.path.join(REPO_ROOT, "scripts", "eval")
for _p in (REPO_ROOT, SRC, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from assembly import build_unified_model
from runtime.physics_controls import make_shuffled_spectrum


def token_cos_dist(a, b):
    """Mean over tokens of (1 - cos) for two (T, D) embeddings (mirrors L_cond)."""
    return float((1.0 - F.cosine_similarity(a, b, dim=-1)).mean().item())


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {"mean": float(xs.mean()), "median": float(np.median(xs)),
            "min": float(xs.min()), "max": float(xs.max()), "n": int(xs.size)}


def main():
    ap = argparse.ArgumentParser(description="Phase 4 target-separation probe")
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--film-std", type=float, default=None,
                    help="if set, re-initialize the frozen SpectrumFilm weights "
                         "with this std before probing")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    spec_w = os.path.join(REPO_ROOT, cfg["weights"]["spectrum"])
    model = build_unified_model(cfg, spec_w, device=args.device)

    if args.film_std is not None:
        with torch.no_grad():
            for head in model.spectrum_film.heads:
                torch.nn.init.normal_(head.weight, std=args.film_std)
                torch.nn.init.zeros_(head.bias)
                head.bias[:model.hidden].fill_(1.0)
        print(f"[probe] re-initialized SpectrumFilm weights with std={args.film_std}",
              flush=True)
    model.eval()

    import eval_scenarios as es
    occ, sv, spec = es._load_val_batch(cfg, torch.device(args.device), smoke=False,
                                       n_samples=args.samples)
    b = occ.shape[0]
    sk = torch.ones(b, 3, dtype=torch.bool, device=occ.device)
    M = torch.ones(b, 16, 16, device=occ.device)   # target ignores the mask

    # Different REAL spectra for the same occupancy: the batch's spectra deranged
    # (the same control the real/null/shuffled gates use).
    variants = [spec]
    for s in (1, 2, 3):
        variants.append(make_shuffled_spectrum(spec, seed=s))

    with torch.no_grad():
        zs, zr = [], []
        for S in variants:
            out = model(occ, sv, sk, S, M, with_target=True)
            zs.append(out["z_y_occ_spec"])   # (B, 256, 192)
            zr.append(out["z_y_raw"])
        # determinism / noise floor: same spectrum twice
        out_rep = model(occ, sv, sk, spec, M, with_target=True)

    cross, vs_raw, noise = [], [], []
    for i in range(b):
        for a in range(len(variants)):
            for c in range(a + 1, len(variants)):
                cross.append(token_cos_dist(zs[a][i], zs[c][i]))
        vs_raw.append(token_cos_dist(zs[0][i], zr[0][i]))
        noise.append(token_cos_dist(zs[0][i], out_rep["z_y_occ_spec"][i]))

    raw_norm = float(zr[0].norm(dim=-1).mean().item())
    report = {
        "film_std": args.film_std if args.film_std is not None else "config-default (0.02)",
        "n_samples": int(b),
        "n_variants": len(variants),
        "cross_spectrum_cos_dist": stats(cross),
        "spec_vs_raw_cos_dist": stats(vs_raw),
        "within_spectrum_noise": stats(noise),
        "mean_token_norm_z_y_raw": raw_norm,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
