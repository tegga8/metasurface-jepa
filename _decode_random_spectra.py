"""Phase 2, step 2: push random target spectra through the model and decode.

The geometry-first direction (Task 1) always had a real geometry to compare
against. Here there is none: the only input is the spectrum. For each target the
model returns a geometry, and the question is whether that geometry actually
radiates the spectrum it was asked for.

The occupancy input is fully masked (ratio 1.0), so the carrier geometry carries
no information -- it exists only because the model's signature expects one. Real
novel geometries are used as carriers so the conditions are comparable with the
Task 1 full-mask setting.

Both target sets are decoded:
  A  interpolated spectra -- smooth like the real corpus (roughness 1.21x),
     plausibly realizable. This is the set we score.
  B  synthetic spectra -- 32x rougher than the corpus, no resonance structure.
     A negative control: if the model produces something equally "correct" for
     these, the earlier agreement means nothing.

Reports the surrogate-side spectrum error against the target for each set. The
surrogate is not evidence about physics -- it is what generated the targets in
the first place -- so this is a consistency check, not a result. The full-wave
Meep-vs-Meep check on these decodes is what actually answers the question, and
it runs separately.

Usage: python _decode_random_spectra.py <random_spectra.npz> <ckpt> <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
sys.path.insert(0, REPO)

import eval_scenarios as es  # type: ignore[import-not-found]  # noqa: E402
import _novel_modeltest as nmt  # noqa: E402
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402


def _load_npz(path):
    try:
        return np.load(path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


def main():
    spec_path, ckpt, out_dir = sys.argv[1:4]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.exit(f"could not create {out_dir}: {exc}")

    z = _load_npz(spec_path)
    b = _load_npz(os.path.join(REPO, "batch40_designs.npz"))
    names_b = b["names"].tolist()
    l, h, r = (float(v) for v in b["scalars"])

    model, surrogate = nmt.load_model(os.path.join(REPO, "configs/unified.yaml"),
                                     ckpt)

    summary = {}
    for tag, rk, ik in (("A_interpolated", "A_real", "A_imag"),
                        ("B_synthetic", "B_real", "B_imag")):
        target = torch.from_numpy(
            np.stack([np.stack([z[rk][i], z[ik][i]])
                      for i in range(len(z[rk]))]).astype(np.float32))
        n = target.shape[0]
        # Carrier geometries, fully masked: they carry no information.
        carriers = torch.from_numpy(
            b["patterns"][:n].astype(np.float32)).unsqueeze(1)
        sv = torch.tensor([[l, h, r]] * n, dtype=torch.float32)
        masker = BlockMasker(placement="random", grid=16, min_side=3,
                             k_range=(1, 4), seed=42)
        M = masker.sample(carriers, ratio=1.0)      # fully masked

        with torch.no_grad():
            sk = torch.zeros(n, 3, dtype=torch.bool)
            out = model(carriers, sv, sk, target, M, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=sk, scalar_values=sv)
            hard = (prob > 0.5).float()
            geom = assemble_metadit_geometry(
                hard, torch.full((n,), l), torch.full((n,), h),
                torch.full((n,), r))
            pred = surrogate(geom).prediction

        err = es._spectrum_error_per_sample(pred, target).numpy()
        occ = (prob > 0.5).to(torch.uint8).squeeze(1).numpy()
        np.savez(os.path.join(out_dir, f"decoded_{tag}.npz"), patterns=occ,
                 scalars=np.tile(np.array([l, h, r]), (n, 1)),
                 names=np.array([f"{tag}/{names_b[i]}" for i in range(n)]),
                 source_idx=np.arange(n))
        summary[tag] = {"surrogate_spec_err": float(err.mean()),
                        "mean_occupancy": float(occ.mean())}
        print(f"[{tag}] n={n}  surrogate spec err={err.mean():.3f}  "
              f"mean occupancy={occ.mean():.3f}", flush=True)
        print("    per-target err: " + " ".join(
            f"{v:.2f}" for v in err), flush=True)

    try:
        with open(os.path.join(out_dir, "phase2_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    except OSError as exc:
        sys.exit(f"could not write summary: {exc}")
    print("saved", out_dir)


if __name__ == "__main__":
    main()