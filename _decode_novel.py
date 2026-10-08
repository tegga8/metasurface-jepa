"""Task 1, step 2: push the novel geometries' Meep spectra through the model.

Takes the fixed-harness Meep spectra for the 40 novel geometries, maps them into
MetaDiT's [2,301] complex target layout with the validated convention, decodes
occupancy under a full mask and a 50% mask, and writes the decoded geometries
out as an npz for the Meep-on-decodes confirmation.

Designs whose Meep spectrum failed the passivity gate are dropped here rather
than fed to the model: an unphysical spectrum is not evidence about the decoder,
and including it would silently contaminate the round-trip statistic.

Reports, per mask setting:
  * IoU / F1 of the decoded occupancy against the true novel geometry
  * normalized L1 error of the decoded geometry's surrogate spectrum against
    the Meep target (surrogate side only -- the full-wave check is Meep-vs-Meep
    and runs later, on the cluster)

Usage: python _decode_novel.py <batch40_merged.json> <convention.json> <ckpt> <out_dir>
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

# eval_scenarios lives in scripts/eval/ and is put on sys.path above; the same
# import appears in tests/test_eval_seed.py and the other probe scripts. It is
# resolved at runtime, not statically.
import eval_scenarios as es  # type: ignore[import-not-found]  # noqa: E402
import _novel_modeltest as nmt  # noqa: E402
from _fit_meep_convention import APPLY  # noqa: E402
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402

APPLY_BY_KIND = APPLY


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {what} {path}: {exc}")


def entry_to_target(entry, conv):
    """Meep entry -> MetaDiT's [2,301] (real, imag) target via the convention."""
    t = APPLY_BY_KIND[conv["kind"]](entry, conv)
    return torch.from_numpy(np.stack([t.real, t.imag]).astype(np.float32))


def main():
    meep_path, conv_path, ckpt, out_dir = sys.argv[1:5]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.exit(f"could not create {out_dir}: {exc}")

    conv = _load_json(conv_path, "convention")
    entries = _load_json(meep_path, "Meep output")["designs"]
    z = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
    names = z["names"].tolist()
    pats_all = z["patterns"]
    l, h, r = (float(v) for v in z["scalars"])

    keep = [e for e in entries if e.get("passivity_ok")]
    dropped = [e for e in entries if not e.get("passivity_ok")]
    print(f"Meep entries: {len(entries)}  usable: {len(keep)}  "
          f"dropped on passivity: {[e.get('name') or e['idx'] for e in dropped]}")

    idx = [e["idx"] for e in keep]
    target = torch.stack([entry_to_target(e, conv) for e in keep])  # (N,2,301)
    pats = torch.from_numpy(pats_all[idx].astype(np.float32)).unsqueeze(1)
    n = len(idx)
    sv = torch.tensor([[l, h, r]] * n, dtype=torch.float32)
    print(f"target tensor {tuple(target.shape)}  occupancy {tuple(pats.shape)}")

    model, surrogate = nmt.load_model(os.path.join(REPO, "configs/unified.yaml"),
                                     ckpt)

    summary = {}
    for ratio, tag in ((1.0, "full"), (0.5, "half")):
        masker = BlockMasker(placement="random", grid=16, min_side=3,
                             k_range=(1, 4), seed=42)
        M = masker.sample(pats, ratio=ratio)
        with torch.no_grad():
            sk = torch.zeros(n, 3, dtype=torch.bool)
            out = model(pats, sv, sk, target, M, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=sk, scalar_values=sv)
            hard = (prob > 0.5).float()
            geom = assemble_metadit_geometry(
                hard, torch.full((n,), l), torch.full((n,), h),
                torch.full((n,), r))
            pred = surrogate(geom).prediction

        err = es._spectrum_error_per_sample(pred, target).numpy()
        tm = target.pow(2).sum(1).sqrt().numpy()
        pm = pred.pow(2).sum(1).sqrt().numpy()
        corr = np.array([np.corrcoef(tm[i], pm[i])[0, 1] if pm[i].std() > 0
                         else 0.0 for i in range(n)])
        ious = np.array([nmt.iou_f1(prob[i], pats[i])[0] for i in range(n)])

        occ = (prob > 0.5).to(torch.uint8).squeeze(1).numpy()
        np.savez(os.path.join(out_dir, f"decoded_{tag}.npz"),
                 patterns=occ,
                 scalars=np.tile(np.array([l, h, r]), (n, 1)),
                 names=np.array([f"{tag}/{names[i]}" for i in idx]),
                 source_idx=np.array(idx))
        summary[tag] = {"iou_mean": float(ious.mean()),
                        "spec_err_mean": float(err.mean()),
                        "corr_mean": float(corr.mean())}
        print(f"[{tag:4s}] IoU={ious.mean():.3f}  surrogate spec err="
              f"{err.mean():.3f}  |T| corr={corr.mean():.3f}", flush=True)
        print("    per-design IoU: " + " ".join(
            f"{nm.split('/')[-1]}={v:.2f}" for nm, v in
            zip([names[i] for i in idx], ious)), flush=True)

    try:
        with open(os.path.join(out_dir, "decode_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    except OSError as exc:
        sys.exit(f"could not write decode summary: {exc}")
    print("saved decoded geometries to", out_dir)


if __name__ == "__main__":
    main()