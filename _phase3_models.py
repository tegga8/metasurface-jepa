"""Task 3 (scalar sensitivity) + MetaDiT task 2: model-side runs, patterns saved.

Supersedes the pattern-less *_scalar_sensitivity.py* for the v1/v2/MetaDiT
comparison: every row here writes a design npz (patterns + scalars + names)
that *_meep_novel.py* can run, because the geometry-side tables alone cannot say
whether the scalars changed the *physics*.

Rows produced
-------------
JEPA (per checkpoint, tag "v1" / "v2"):
  B_true    regime B (all three scalars GIVEN, true values) on the same
            Meep->CST target task 1 used -- directly comparable to the
            task1_full rows (regime A, identical geometry set and target).
  spec_m20  spectrum-only decode (scalars unknown), conditioned on the
            surrogate spectrum of the SAME original geometry at -20% scalars.
  spec_p20  same at +20%.

MetaDiT-S (no scalar input; gets exactly the same three conditions):
  spec_true, spec_m20, spec_p20
  A, B      task 2: the random-spectrum sets, from random_spectra.npz.

The matched rows exist because of the confound recorded in VALIDATION_PLAN.md
4.4-5: the previous MetaDiT scalar table conditioned MetaDiT on a surrogate
spectrum while the JEPA rows used a Meep->CST spectrum, so the two were not
comparable. Here both models get bit-identical conditions.

Usage:
  python _phase3_models.py <out_dir> [--n 10] [--jepa v1:<ckpt> v2:<ckpt>]
                           [--metadit-weights PATH] [--skip-metadit]
                           [--skip-task2]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "external", "metadit"))

import _novel_modeltest as nmt  # noqa: E402
from _compare_decoded_meep import load_dir_or_file  # noqa: E402
from _render_vs_metadit import mirror_score, n_components, n_holes  # noqa: E402
from data.factorize import assemble_metadit_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402

NFREQ = 301
OFFSETS = (("spec_true", 0.0), ("spec_m20", -0.20), ("spec_p20", +0.20))


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {what} {path}: {exc}")


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def style(pats):
    return dict(
        holes=float(np.mean([n_holes(p) for p in pats])),
        mirror=float(np.mean([mirror_score(p) for p in pats])),
        comps=float(np.mean([n_components(p) for p in pats])),
        occ=float(np.mean([p.mean() for p in pats])),
    )


def save_row(out_dir, tag, row, pats, scalars, names):
    path = os.path.join(out_dir, f"{tag}_{row}.npz")
    np.savez(path, patterns=np.stack([(p > 0.5).astype(np.uint8) for p in pats]),
             scalars=np.tile(np.asarray(scalars, dtype=np.float64),
                             (len(pats), 1)),
             names=np.array([f"{row}/{n}" for n in names]),
             source_idx=np.arange(len(pats)))
    print(f"    saved {os.path.basename(path)}  {len(pats)} designs", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--jepa", nargs="*", default=[],
                    help="tag:ckpt_path pairs, e.g. v1:/path/final.pt")
    ap.add_argument("--metadit-weights", default=os.path.join(
        REPO, "data", "metadit", "weights", "metadit-small.bin"))
    ap.add_argument("--skip-metadit", action="store_true")
    ap.add_argument("--skip-task2", action="store_true")
    ap.add_argument("--task2-only", action="store_true",
                    help="MetaDiT on the random-spectrum sets only; skips the "
                         "scalar-matched rows (and needs no b40 targets)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    conv = _load_json(os.path.join(REPO, "_meep_convention.json"), "convention")
    from _fit_meep_convention import APPLY

    b = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
    names_all = b["names"].tolist()
    l0, h0, r0 = (float(v) for v in b["scalars"])
    base = np.array([l0, h0, r0])

    orig_meep = {} if args.task2_only else load_dir_or_file(os.path.join(
        os.path.dirname(args.out_dir), "b40_merged.json"))[0]

    picks = [nm for nm in names_all if nm in orig_meep][:args.n]
    idx = [names_all.index(nm) for nm in picks]
    occ_np = b["patterns"][idx].astype(np.float32)
    occ = torch.from_numpy(occ_np).unsqueeze(1)
    print(f"task3 rows on {len(picks)} novel geometries "
          f"(true scalars l={l0:.3f} h={h0:.3f} r={r0:.3f})", flush=True)

    def to_target(name):
        t = APPLY[conv["kind"]](orig_meep[name], conv)
        return np.stack([t.real, t.imag]).astype(np.float32)

    target = (torch.from_numpy(np.stack([to_target(nm) for nm in picks]))
              if picks else torch.zeros(0, 2, NFREQ))

    results = {"true_scalars": base.tolist(), "n": len(picks), "rows": []}

    # ---------------- JEPA rows ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.jepa:
        tag, ckpt = spec.split(":", 1)
        print(f"\n=== JEPA {tag}  ({ckpt}) ===", flush=True)
        model, surrogate = nmt.load_model(
            os.path.join(REPO, "configs/unified.yaml"), ckpt)
        model = model.to(device)

        def decode(sv, known, tgt, label):
            sv_t = torch.tensor(np.tile(sv, (len(picks), 1)),
                                dtype=torch.float32, device=device)
            masker = BlockMasker(placement="random", grid=16, min_side=3,
                                 k_range=(1, 4), seed=42)
            mfull = masker.sample(occ.to(device), ratio=1.0)
            with torch.no_grad():
                out = model(occ.to(device), sv_t, known, tgt.to(device), mfull,
                            goal_mode="real", with_target=False)
                prob = model.decode_occupancy_prob(
                    out["z_hat"], out["scalar_pred"], scalar_known=known,
                    scalar_values=sv_t)
                hard = (prob > 0.5).float()
                geom = assemble_metadit_geometry(hard, sv_t[:, 0], sv_t[:, 1],
                                                sv_t[:, 2])
                pred = surrogate(geom).prediction
            err = float((pred - tgt.to(device)).abs().mean(dim=(1, 2)).div(
                tgt.to(device).std(dim=(1, 2))).mean())
            pats = (prob > 0.5).to(torch.uint8).cpu().squeeze(1).numpy()
            ious = [iou(pats[k], occ_np[k]) for k in range(len(picks))]
            st = style(list(pats))
            print(f"  {tag} {label:12s} IoU={np.mean(ious):.3f} "
                  f"spec_err={err:.3f} occ={st['occ']:.3f} "
                  f"holes={st['holes']:.2f} comps={st['comps']:.2f}", flush=True)
            results["rows"].append({"model": f"jepa_{tag}", "row": label,
                                    "iou": float(np.mean(ious)),
                                    "spec_err": err, **st})
            return pats, sv_t.cpu().numpy()[0]

        # regime B, true scalars, on the task-1 target
        kB = torch.ones(len(picks), 3, dtype=torch.bool, device=device)
        pats, sv = decode(base, kB, target, "B_true")
        save_row(args.out_dir, f"jepa_{tag}", "B_true", pats, sv, picks)

        # spectrum-only, matched conditions with MetaDiT:
        # the surrogate spectrum of the SAME geometry at perturbed scalars
        for row, off in OFFSETS[1:]:
            sv = base * (1 + off)
            sv_t = torch.tensor(np.tile(sv, (len(picks), 1)),
                                dtype=torch.float32, device=device)
            g = assemble_metadit_geometry(occ.to(device), sv_t[:, 0],
                                          sv_t[:, 1], sv_t[:, 2])
            with torch.no_grad():
                cond = surrogate(g).prediction
            kA = torch.zeros(len(picks), 3, dtype=torch.bool, device=device)
            pats, sv = decode(sv, kA, cond.cpu(), row)
            save_row(args.out_dir, f"jepa_{tag}", row, pats, sv, picks)
            np.savez(os.path.join(args.out_dir, f"jepa_{tag}_{row}_cond.npz"),
                     real=cond[:, 0].cpu().numpy(),
                     imag=cond[:, 1].cpu().numpy(), names=np.array(picks))

    # ---------------- MetaDiT rows ----------------
    if not args.skip_metadit:
        from model.dit import DIT_MODEL  # type: ignore[import-not-found]
        from diffusion import create_diffusion  # type: ignore[import-not-found]
        from metric import restore_structure  # type: ignore[import-not-found]

        diffusion = create_diffusion("500", learn_sigma=False)
        md_model = DIT_MODEL["metadit_s"](diffusion=diffusion,
                                          condition_channel=NFREQ)
        md_model.load_state_dict(torch.load(args.metadit_weights,
                                            map_location="cpu"), strict=True)
        md_model.to(device).eval()
        print(f"\n=== MetaDiT-S ({device}) ===", flush=True)

        class _Shim(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, t, **kw):
                return self.m.forward_with_cfg(x, t, **kw)

        shim = _Shim(md_model).to(device).eval()

        def generate(cond, label):
            """cond: [B, 2, 301] CST-domain. Returns [B, 64, 64] uint8."""
            B = cond.shape[0]
            torch.manual_seed(0)
            z = torch.randn(B, 3, 32, 32, device=device)
            z = torch.cat([z, z], 0)
            c = torch.cat([cond.to(device),
                           torch.ones_like(cond).to(device) * 0.5], 0)
            t0 = time.time()
            with torch.no_grad():
                sm = diffusion.p_sample_loop(
                    shim, z.shape, z, clip_denoised=False,
                    model_kwargs=dict(y=c, cfg_scale=4.0), progress=False,
                    device=device)
            sm, _ = sm.chunk(2, dim=0)
            pats = []
            for j in range(B):
                full = restore_structure(sm[j].detach().cpu().clone())
                pats.append((full[0] > full[0].mean()).to(torch.uint8).numpy())
            print(f"  metadit {label:12s} {time.time() - t0:.1f}s "
                  f"occ={np.mean([p.mean() for p in pats]):.3f}", flush=True)
            return np.stack(pats)

        # matched scalar rows: identical conditions to the JEPA spectrum-only
        # rows (build them here rather than reusing a JEPA model's surrogate --
        # same math, but this keeps MetaDiT's rows independent of the ckpts)
        if not args.task2_only:
            if args.jepa:
                ref_ckpt = args.jepa[0].split(":", 1)[1]
                _, surrogate = nmt.load_model(
                    os.path.join(REPO, "configs/unified.yaml"), ref_ckpt)
            else:
                from physics.physics_loop import load_surrogate
                import yaml
                with open(os.path.join(REPO, "configs", "unified.yaml"),
                          encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                surrogate = load_surrogate(config.get("physics", {}).get(
                    "surrogate_path", "data/metadit/weights/surrogate_model.bin"))

            md_specs = {}
            for row, off in OFFSETS:
                sv = base * (1 + off)
                sv_t = torch.tensor(np.tile(sv, (len(picks), 1)),
                                    dtype=torch.float32)
                g = assemble_metadit_geometry(occ, sv_t[:, 0], sv_t[:, 1],
                                              sv_t[:, 2])
                with torch.no_grad():
                    cond = surrogate(g).prediction
                md_specs[row] = cond
                pats = generate(cond, row)
                ious = [iou(pats[k], occ_np[k]) for k in range(len(picks))]
                st = style(list(pats))
                print(f"    IoU={np.mean(ious):.3f} holes={st['holes']:.2f} "
                      f"comps={st['comps']:.2f} mirror={st['mirror']:.3f}",
                      flush=True)
                results["rows"].append({"model": "metadit_s", "row": row,
                                        "iou": float(np.mean(ious)),
                                        **st})
                save_row(args.out_dir, "metadit", row, pats, sv, picks)

        # ---- task 2: random-spectrum sets ----
        if not args.skip_task2:
            z = np.load(os.path.join(os.path.dirname(args.out_dir),
                                     "random_spectra.npz"), allow_pickle=True)
            for tag, rk, ik in (("A_interpolated", "A_real", "A_imag"),
                                ("B_synthetic", "B_real", "B_imag")):
                cond = torch.from_numpy(np.stack([z[rk], z[ik]],
                                                 axis=1).astype(np.float32))
                pats = generate(cond, tag)
                occm = float(np.mean([p.mean() for p in pats]))
                print(f"    mean occupancy {occm:.3f}", flush=True)
                np.savez(os.path.join(args.out_dir, f"metadit_{tag}.npz"),
                         patterns=pats,
                         scalars=np.tile(base, (len(pats), 1)),
                         names=np.array([f"{tag}/{i:02d}"
                                         for i in range(len(pats))]),
                         source_idx=np.arange(len(pats)))
                results["rows"].append({"model": "metadit_s", "row": tag,
                                        "occ": occm, "n": len(pats)})

    with open(os.path.join(args.out_dir, "phase3_rows.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\nsaved", os.path.join(args.out_dir, "phase3_rows.json"))
    print("PHASE3 ROWS DONE")


if __name__ == "__main__":
    main()
