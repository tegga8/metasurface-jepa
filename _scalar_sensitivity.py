"""Task 3: scalar sensitivity -- how much do the scalars move geometry/spectrum?

The operator asked to "make sure scalars are appropriately given or test it on
multiple scalars for a same type of geometry and samples and check how much
change in geometry or spectrum scalars produce".

Motivation beyond the request: every decode run so far used
``scalar_known = zeros``, and ``_effective_scalars`` (src/assembly.py:449)
evaluates ``torch.where(False, scalar_values, scalar_pred)`` -- so the true
scalar values were silently ignored and ALL THREE scalars were PREDICTED. The
known-scalar path uses the 6-dim ``[value, known-flag]`` conditioning of
architecture_v5.md section 3.2, which has never been exercised. This measures it.

Protocol, on a fixed set of novel geometries so only the scalars vary:

  regime A  all three scalars UNKNOWN   (what every earlier result used)
  regime B  all three scalars GIVEN     (known flags set, true values used)
  regime C  one scalar at a time GIVEN  (isolate each scalar's contribution)

and, within each regime, a sweep where the SUPPLIED scalars are perturbed away
from the geometry's true values, measuring:
  * geometry IoU against the true novel geometry
  * spectrum error: the decoded geometry through the frozen surrogate vs the
    geometry's own Meep-derived target
  * the decoded geometry's occupancy/holes/components (does it stay well formed)

Runs for BOTH models: the unified 192-D JEPA (v1 checkpoint) and MetaDiT-S
conditioned on identical [2,301] targets, so the scalar response can be
compared directly.

Usage: python _scalar_sensitivity.py <ckpt> <metadit_npz> <out.json> [n_designs]
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "external", "metadit"))

import _novel_modeltest as nmt  # noqa: E402
from _compare_decoded_meep import curve, load_dir_or_file  # noqa: E402
from _render_vs_metadit import (mirror_score, n_components,  # noqa: E402
                                n_holes)
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402


def iou(a, b):
    a = a > 0.5
    b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {what} {path}: {exc}")


def style(pats):
    return dict(
        holes=float(np.mean([n_holes(p) for p in pats])),
        mirror=float(np.mean([mirror_score(p) for p in pats])),
        comps=float(np.mean([n_components(p) for p in pats])),
        occ=float(np.mean([p.mean() for p in pats])),
    )


def main():
    ckpt, md_npz, out_json = sys.argv[1:4]
    n_des = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"), "convention")
    from _fit_meep_convention import APPLY

    b = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
    names = b["names"].tolist()
    l0, h0, r0 = (float(v) for v in b["scalars"])

    # Meep targets for the first n_des novel designs (passivity-clean only).
    orig_meep = load_dir_or_file(
        os.path.join(os.path.dirname(out_json), "b40_merged.json"))[0]

    def to_target(name):
        e = orig_meep[name]
        t = APPLY[conv["kind"]](e, conv)
        return np.stack([t.real, t.imag]).astype(np.float32)

    picks = [nm for nm in names if nm in orig_meep][:n_des]
    print(f"scalar sensitivity on {len(picks)} novel geometries "
          f"(true scalars l={l0:.3f} h={h0:.3f} r={r0:.3f})")
    idx = [names.index(nm) for nm in picks]
    occ = torch.from_numpy(b["patterns"][idx].astype(np.float32)).unsqueeze(1)
    target = torch.from_numpy(np.stack([to_target(nm) for nm in picks]))

    # Perturbation grid: multiplicative offsets on the SUPPLIED scalars.
    offsets = [("true", 0.0), ("-20%", -0.20), ("+20%", +0.20)]
    base = np.array([l0, h0, r0])

    model, surrogate = nmt.load_model(os.path.join(REPO, "configs/unified.yaml"),
                                     ckpt)

    md = np.load(md_npz, allow_pickle=True) if md_npz != "-" else None
    md_names = {str(x).split("/")[-1]: i for i, x in enumerate(md["names"])} \
        if md is not None else {}

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    Mfull = masker.sample(occ, ratio=1.0)

    results = {"true_scalars": base.tolist(), "n": len(picks), "runs": []}

    def decode(sv, known, tag, model_name):
        sv_t = torch.tensor(np.tile(sv, (len(picks), 1)), dtype=torch.float32)
        with torch.no_grad():
            out = model(occ, sv_t, known, target, Mfull, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=known,
                                               scalar_values=sv_t)
        hard = (prob > 0.5).float()
        geom = assemble_metadit_geometry(hard, sv_t[:, 0], sv_t[:, 1], sv_t[:, 2])
        with torch.no_grad():
            pred = surrogate(geom).prediction
        err = float((pred - target).abs().mean(dim=(1, 2)).div(
            target.std(dim=(1, 2))).mean())
        pats = (prob > 0.5).to(torch.uint8).squeeze(1).numpy()
        ious = [iou(pats[k], occ[k, 0].numpy()) for k in range(len(picks))]
        st = style(list(pats))
        row = {"model": model_name, "regime": tag, "supplied": sv.tolist(),
               "iou": float(np.mean(ious)), "spec_err": err, **st}
        results["runs"].append(row)
        print(f"  {model_name:9s} {tag:26s} IoU={row['iou']:.3f} "
              f"spec_err={err:.3f} occ={st['occ']:.3f} "
              f"holes={st['holes']:.2f} comps={st['comps']:.2f}", flush=True)
        return pats

    print("\n--- JEPA: regime A, all three scalars UNKNOWN (earlier runs) ---")
    kA = torch.zeros(len(picks), 3, dtype=torch.bool)
    decode(base, kA, "A unknown x3", "jepa")
    for lab, off in offsets[1:]:
        decode(base * (1 + off), kA, f"A unknown x3 ({lab} sup)", "jepa")

    print("\n--- JEPA: regime B, all three scalars GIVEN ---")
    kB = torch.ones(len(picks), 3, dtype=torch.bool)
    for lab, off in offsets:
        decode(base * (1 + off), kB, f"B given x3 ({lab} sup)", "jepa")

    print("\n--- JEPA: regime C, one scalar GIVEN at a time ---")
    for si, sname in enumerate(("l", "h", "r")):
        kC = torch.zeros(len(picks), 3, dtype=torch.bool)
        kC[:, si] = True
        for lab, off in offsets:
            sv = base * (1 + off)
            decode(sv, kC, f"C given {sname} only ({lab} sup)", "jepa")

    if md is not None:
        # MetaDiT-S has NO scalar input: it conditions on the spectrum alone
        # (generate.py -> model_kwargs=dict(y=condition, ...)). So a literal
        # scalar sweep is undefined for it. The honest analogue: perturb the
        # scalars, build the spectrum that perturbed geometry WOULD radiate
        # (frozen surrogate, cheap), condition MetaDiT on THAT, and see whether
        # its decode tracks the scalars through the spectrum.
        print("\n--- MetaDiT-S: no scalar input; tested via the spectrum the "
              "perturbed scalars imply ---")
        from model.dit import DIT_MODEL
        from diffusion import create_diffusion
        from metric import restore_structure

        diffusion = create_diffusion("500", learn_sigma=False)
        md_model = DIT_MODEL["metadit_s"](diffusion=diffusion,
                                          condition_channel=301)
        md_model.load_state_dict(
            torch.load(os.path.join(REPO, "data", "metadit", "weights",
                                    "metadit-small.bin"),
                       map_location="cpu"), strict=True)
        md_model.eval()

        class _CfgShim(torch.nn.Module):
            """p_sample_loop calls next(model.parameters()).device, so the
            model handed to it must be a Module. forward_with_cfg is a bound
            method, not one, hence this shim."""

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, t, **kw):
                return self.m.forward_with_cfg(x, t, **kw)

        shim = _CfgShim(md_model).eval()

        def surrogate_of(occv, sv):
            sv_t = torch.tensor(np.tile(sv, (len(picks), 1)),
                                dtype=torch.float32)
            g = assemble_metadit_geometry(occv, sv_t[:, 0], sv_t[:, 1],
                                          sv_t[:, 2])
            with torch.no_grad():
                return surrogate(g).prediction

        for lab, off in offsets:
            sv = base * (1 + off)
            # Spectrum implied by the SAME geometry at perturbed scalars.
            cond = surrogate_of(occ, sv)
            torch.manual_seed(0)
            z = torch.randn(len(picks), 3, 32, 32)
            z = torch.cat([z, z], 0)
            c2 = torch.cat([cond, torch.ones_like(cond) * 0.5], 0)
            with torch.no_grad():
                sm = diffusion.p_sample_loop(
                    shim, z.shape, z, clip_denoised=False,
                    model_kwargs=dict(y=c2, cfg_scale=4.0), progress=False)
            sm, _ = sm.chunk(2, dim=0)
            pats = np.stack([
                (lambda f: (f[0] > f[0].mean()).to(torch.uint8).numpy())(
                    restore_structure(sm[j].detach().cpu().clone()))
                for j in range(len(picks))])
            ious = [iou(pats[k], occ[k, 0].numpy()) for k in range(len(picks))]
            st = style(list(pats))
            row = {"model": "metadit_s",
                   "regime": f"spectrum-implied ({lab} sup)",
                   "supplied": sv.tolist(), "iou": float(np.mean(ious)), **st}
            results["runs"].append(row)
            print(f"  metadit_s spectrum-implied ({lab:4s} scalars) "
                  f"IoU={row['iou']:.3f} occ={st['occ']:.3f} "
                  f"holes={st['holes']:.2f} comps={st['comps']:.2f}", flush=True)

    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    except OSError as exc:
        sys.exit(f"could not write {out_json}: {exc}")
    print("\nsaved", out_json)
    print("NOTE MetaDiT-S conditions on the SPECTRUM only; it has no scalar "
          "input, so its 'supplied scalar' sweep varies the TARGET spectrum's "
          "associated scalars only through the dataset pairing -- i.e. it is "
          "structurally insensitive to the scalars by construction.")


if __name__ == "__main__":
    main()