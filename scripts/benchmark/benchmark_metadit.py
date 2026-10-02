"""MetaDiT-comparable benchmark driver for the unified JEPA.

Runs the model on a held-out split under Scenario A (pure inverse design: full
occupancy mask + all scalars unknown -- the scenario MetaDiT's paper metric maps
onto) and reports MAE / AAE / AAE&K in the paper's units, alongside the existing
secondary suite and reference baselines.

See docs/benchmarking/README.md (comparability contract) and METRICS.md.

Run:
    python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
        --samples 512 --candidates 4 --device cpu \
        --out checkpoints/benchmark/scenarioA.json

    # pipeline smoke on synthetic data with an untrained model (no checkpoint):
    python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
        --smoke --samples 4 --candidates 2 --device cpu
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
EVAL_DIR = os.path.join(REPO_ROOT, "scripts", "eval")
BENCH_DIR = os.path.join(REPO_ROOT, "scripts", "benchmark")
for _p in (REPO_ROOT, SRC_DIR, EVAL_DIR, BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import yaml

from assembly import build_unified_model, load_into_model
from data.factorize import factorize_geometry, assemble_metadit_geometry
from data.mask import BlockMasker
from physics.physics_loop import load_surrogate
import eval_scenarios as es
import metadit_metrics as mm
import candidate_sampling as cs
import baselines as bl

# Reference numbers. MetaDiT-S row reproduced locally (checkpoints/phase0); the
# paper rows are from checkpoints/phase0/REPORT.md §4-§7.
PAPER = {
    "metadit_s": {"MAE": 0.0801, "AAE": 48.2495},
    "vanilla_dit": {"MAE": 0.1677, "AAE": 100.9437},
    "avg1": {"MAE": 0.5860, "AAE": 352.7424},
    "aaek": {"AAE&2": 58.80, "AAE&4": 68.73},
    "surrogate_floor_mae": 0.0084,
}


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _split_path(cfg, split):
    key = f"{split}_split"
    if key in cfg.get("data", {}):
        return _resolve(cfg["data"][key])
    return _resolve(os.path.join("data", "metadit", "split_data",
                                 f"{split}_set.mat"))


def _load_model_and_surrogate(cfg, ckpt_path, device, smoke):
    spec_weights = _resolve(cfg["weights"]["spectrum"])
    if not os.path.exists(spec_weights):
        raise RuntimeError(f"released spectrum encoder missing: {spec_weights}")
    model = build_unified_model(cfg, spec_weights, device=device)
    if not smoke:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"checkpoint {ckpt_path!r} not found. The trained checkpoint is a "
                "cloud artifact (CLOUD_TRAINING.md); use --smoke for a synthetic "
                "pipeline check.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_into_model(model, ckpt["model"], device=device, strict=True)
        from train.engine import restore_ema_state
        restore_ema_state(model, ckpt.get("ema_state", {}))
    model.eval()
    surr_path = _resolve(cfg["weights"]["surrogate"])
    if not os.path.exists(surr_path):
        raise FileNotFoundError(f"surrogate weights missing: {surr_path}")
    surrogate = load_surrogate(surr_path, device=device)
    return model, surrogate


def _iter_batches(cfg, split, n_total, batch_size, device, smoke):
    if smoke:
        b = max(1, min(batch_size, n_total or batch_size))
        occ, sv, spec = es._make_synthetic_batch(b, device)
        yield occ, sv, spec
        return
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader

    path = _split_path(cfg, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split} split missing: {path}")
    ds = MetaDiTDataset(path, max_samples=(n_total or 0), seed=0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    for G, S in loader:
        occ, sv = factorize_geometry(G)
        yield occ.to(device), sv.to(device), S.to(device)


def _scenario_inputs(scenario, occ, device, masker, seed):
    b = occ.shape[0]
    if scenario == "A":
        sk = torch.zeros(b, 3, dtype=torch.bool, device=device)
        ratio = 1.0
    elif scenario == "B":
        sk = es._scenario_b_known_flags(b, device)
        ratio = 0.5
    else:  # C
        sk = torch.ones(b, 3, dtype=torch.bool, device=device)
        ratio = 0.25
    M = masker.sample(occ, ratio=ratio).to(device)
    return M, sk


def _mean(vals):
    return float(np.mean(vals)) if vals else None


def run(cfg, args):
    device = torch.device(args.device)
    model, surrogate = _load_model_and_surrogate(cfg, args.checkpoint, device,
                                                 args.smoke)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=999)
    n_total = args.samples if args.samples and args.samples > 0 else None

    mae_items, aae_items = [], []
    aae_cols = None
    sec = {"normalized_l1": [], "occupancy_iou_masked": [],
           "occupancy_f1_masked": [], "pred_occupancy_fraction": [],
           "scalar_mae_unknown": [], "scalar_mae_known": []}
    first = None
    all_spec = []
    n_done = 0

    with torch.no_grad():
        for occ, sv, spec in _iter_batches(cfg, args.split, n_total,
                                            args.batch_size, device, args.smoke):
            M, sk = _scenario_inputs(args.scenario, occ, device, masker, args.seed)
            out = model(occ, sv, sk, spec, M, goal_mode="real", with_target=False)

            geom_det, _ = model.decode_geometry(
                out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M,
                scalar_known=sk, scalar_values=sv, hard_forward=True)
            spec_det = surrogate(geom_det).prediction

            mae_items.append(mm.per_item_mae(spec, spec_det))
            aae_items.append(mm.per_item_aae(spec, spec_det))
            mat = cs.candidate_aae_matrix(
                model, surrogate, out, occ, M, sk, sv, spec,
                args.noise_std, args.candidates, seed=args.seed + 7)
            aae_cols = mat if aae_cols is None else torch.cat([aae_cols, mat], 0)

            # Secondary suite (per-batch means).
            sec["normalized_l1"].append(
                float(es._spectrum_error_per_sample(spec_det, spec).mean().item()))
            raw_prob = model.decode_occupancy_prob(
                out["z_hat"], out["scalar_pred"], scalar_known=sk, scalar_values=sv)
            om = es._occupancy_metrics(raw_prob, occ, mask=M)
            sec["occupancy_iou_masked"].append(om["masked_region"]["iou"])
            sec["occupancy_f1_masked"].append(om["masked_region"]["f1"])
            sec["pred_occupancy_fraction"].append(om["pred_occupancy_fraction"])
            unknown, known = ~sk, sk
            if unknown.any():
                sec["scalar_mae_unknown"].append(
                    float((out["scalar_pred"] - sv)[unknown].abs().mean().item()))
            if known.any():
                sec["scalar_mae_known"].append(
                    float((out["scalar_pred"] - sv)[known].abs().mean().item()))

            if first is None:
                first = (occ, sv, spec, M, sk)
            all_spec.append(spec.cpu())
            n_done += occ.shape[0]

    mae_all = torch.cat(mae_items)
    aae_all = torch.cat(aae_items)
    primary = {"MAE": float(mae_all.mean().item()),
               "AAE": float(aae_all.mean().item()),
               "n": int(n_done)}
    for k in (2, 4):
        if aae_cols is not None and aae_cols.shape[1] >= k:
            primary[f"AAE&{k}"] = mm.aae_and_k(aae_cols, k)

    # Baselines.
    all_spec = torch.cat(all_spec, 0).to(device)
    baseline_out = {
        "avg1": bl.avg1_metrics(all_spec, bl.mean_spectrum(all_spec)),
    }
    obsv, svf, specf, Mf, skf = first
    baseline_out["surrogate_floor"] = bl.surrogate_floor_metrics(
        obsv, svf, specf, surrogate)
    if args.nn_samples > 0:
        tr_specs, tr_occ, tr_sv = _load_train_representations(
            cfg, device, args.smoke, args.nn_samples)
        baseline_out["nn"] = bl.nn_metrics(
            all_spec[:args.nn_samples], tr_specs, tr_occ, tr_sv, surrogate)

    result = {
        "meta": {
            "architecture_id": getattr(model, "architecture_id", None),
            "split": args.split,
            "scenario": args.scenario,
            "samples": int(n_done),
            "candidates": int(args.candidates),
            "noise_std": float(args.noise_std),
            "candidate_generator": "latent-jitter",
            "deterministic_model": True,
            "is_diffusion_seed_diversity": False,
            "data_mode": "SMOKE (synthetic)" if args.smoke else "REAL",
            "device": str(device),
        },
        "primary": primary,
        "secondary": {k: _mean(v) for k, v in sec.items()},
        "baselines": baseline_out,
        "references": PAPER,
        "metadit_reproduced": _metadit_reproduced(),
    }
    return result


def _load_train_representations(cfg, device, smoke, n_train):
    if smoke:
        occ, sv, spec = es._make_synthetic_batch(max(2, n_train), device)
        return spec, occ, sv
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    path = _split_path(cfg, "train")
    ds = MetaDiTDataset(path, max_samples=n_train, seed=0)
    loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 2),
                        shuffle=False, num_workers=0, collate_fn=collate_batch)
    occs, svs, specs = [], [], []
    for G, S in loader:
        o, sv = factorize_geometry(G)
        occs.append(o)
        svs.append(sv)
        specs.append(S)
    return (torch.cat(specs).to(device), torch.cat(occs).to(device),
            torch.cat(svs).to(device))


def _metadit_reproduced():
    path = os.path.join(REPO_ROOT, "checkpoints", "phase0", "seed0_metric.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _print_table(result):
    p = result["primary"]
    print("\n=== Unified JEPA vs MetaDiT (MAE / AAE, paper units) ===")
    print(f"split={result['meta']['split']} scenario={result['meta']['scenario']} "
          f"n={p['n']} mode={result['meta']['data_mode']}")
    rows = [("Unified JEPA (ours)", p["MAE"], p["AAE"], "Scenario " + result["meta"]["scenario"])]
    b = result["baselines"]
    if "nn" in b:
        rows.append(("NN retrieval", b["nn"]["MAE"], b["nn"]["AAE"], f"n={b['nn']['n']}"))
    rows.append(("AVG1 (mean spectrum)", b["avg1"]["MAE"], b["avg1"]["AAE"], "eval-split mean"))
    rows.append(("surrogate floor", b["surrogate_floor"]["MAE"],
                 b["surrogate_floor"]["AAE"], "G_true -> surrogate"))
    r = result.get("metadit_reproduced")
    if r:
        rows.append(("MetaDiT-S (reproduced)", r["MAE"], r["AAE"], "seed0.json"))
    rows.append(("MetaDiT-S (paper)", PAPER["metadit_s"]["MAE"],
                 PAPER["metadit_s"]["AAE"], "reference"))
    rows.append(("vanilla DiT (paper)", PAPER["vanilla_dit"]["MAE"],
                 PAPER["vanilla_dit"]["AAE"], "reference"))
    print(f"{'arm':<26}{'MAE':>10}{'AAE':>12}  notes")
    for name, m, a, note in rows:
        print(f"{name:<26}{m:>10.4f}{a:>12.3f}  {note}")
    if "AAE&2" in p or "AAE&4" in p:
        aaek = {k: p[k] for k in p if k.startswith("AAE&")}
        ks = "  ".join(f"{k}={v:.2f}" for k, v in sorted(aaek.items()))
        print(f"ours AAE&K: {ks}   (paper: {PAPER['aaek']})  "
              f"[latent-jitter analogue, not seed diversity]")
    print("secondary:", {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in result["secondary"].items()})


def main():
    parser = argparse.ArgumentParser(description="MetaDiT-comparable benchmark")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--scenario", type=str, default="A",
                        choices=["A", "B", "C"])
    parser.add_argument("--samples", type=int, default=512,
                        help="number of held-out items (0 = whole split)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=4,
                        help="K for the AAE&K analogue")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="latent-jitter sigma for the candidate generator")
    parser.add_argument("--nn-samples", type=int, default=0,
                        help="0 disables the (slower) NN baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data + untrained model (pipeline check only)")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    result = run(cfg, args)
    _print_table(result)
    if args.out:
        out = _resolve(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
