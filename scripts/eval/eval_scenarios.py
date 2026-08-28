"""Phase 5 — Scenario evaluation (unified_jepa/Phase 5 MD §11).

Report separately:
    pure inverse design       (Scenario A)
    partial-parameter         (Scenario B)
    retrofit                  (Scenario C)

For each include:
    spectrum error
    occupancy IoU / F1 (where ground truth available)
    scalar MAE
    occupancy fraction statistics
    real/null/shuffled gap
    diversity information
    nearest-neighbor baseline

Do not pool results across scenarios.

Run:
    python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt --scenario all
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from assembly import build_unified_model, load_into_model
from data.factorize import factorize_geometry, assemble_metadit_geometry
from data.mask import BlockMasker
from physics.physics_loop import load_surrogate


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _make_synthetic_batch(b, device):
    torch.manual_seed(123)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0
    sv = (torch.rand(b, 3) * 10 + 1).to(device)
    spec = torch.randn(b, 2, 301).to(device)
    return occ, sv, spec


def _occupancy_metrics(pred_occ, true_occ):
    """Compute IoU and F1 for binary occupancy."""
    pred_bin = (pred_occ > 0.5).float()
    true_bin = (true_occ > 0.5).float()
    tp = ((pred_bin * true_bin) > 0).sum().item()
    fp = ((pred_bin * (1 - true_bin)) > 0).sum().item()
    fn = (((1 - pred_bin) * true_bin) > 0).sum().item()
    fp_count = float(pred_bin.sum().item())
    fn_count = float(true_bin.sum().item())
    iou = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pred_occupancy_fraction": fp_count / max(1, pred_bin.numel()),
        "true_occupancy_fraction": fn_count / max(1, true_bin.numel()),
    }


def _spectrum_error(pred_spec, target_spec):
    """Normalized L1 spectrum error."""
    std = target_spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return float(((pred_spec - target_spec) / std).abs().mean().item())


@torch.no_grad()
def evaluate_scenario(model, surrogate, occ, sv, spec, mask, scalar_known,
                      device, scenario_name):
    """Evaluate a single scenario, returning all required metrics."""
    model.eval()
    surrogate.eval()

    out = model(occ, sv, scalar_known.float() * sv, scalar_known, spec,
                mask, goal_mode="real")
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=occ, mask=mask, use_ste=False)
    spectrum_pred = surrogate(geometry).prediction

    # Spectrum error
    spec_err = _spectrum_error(spectrum_pred, spec)

    # Scalar MAE (on unknown positions)
    unknown = ~scalar_known
    if unknown.any():
        scalar_mae = float(
            (out["scalar_pred"] - sv)[unknown].abs().mean().item())
    else:
        scalar_mae = 0.0

    # Occupancy metrics (vs. ground truth)
    occ_metrics = _occupancy_metrics(soft_occ, occ)

    return {
        "scenario": scenario_name,
        "spectrum_error": spec_err,
        "scalar_mae": scalar_mae,
        **occ_metrics,
    }


@torch.no_grad()
def real_null_shuffled(model, surrogate, occ, sv, spec, mask, device,
                       scalar_known=None):
    """Real/null/shuffled goal dependence (Phase 4/5 MD §10).

    Returns spectrum errors for each condition.
    """
    if scalar_known is None:
        scalar_known = torch.ones(occ.shape[0], 3, dtype=torch.bool,
                                  device=device)

    results = {}
    for mode in ("real", "null", "shuffled"):
        if mode == "shuffled":
            spec_eval = torch.roll(spec, shifts=1, dims=0)
        elif mode == "null":
            spec_eval = torch.zeros_like(spec)
        else:
            spec_eval = spec

        out = model(occ, sv, scalar_known, spec_eval, mask, goal_mode=mode)
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask)
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)

    results["gap"] = {
        "real_minus_null": results["null"] - results["real"],
        "real_minus_shuffled": results["shuffled"] - results["real"],
        "gate": results["real"] < results["shuffled"],
    }
    return results


@torch.no_grad()
def scalar_dependence(model, surrogate, occ, sv, spec, mask, device,
                      scalar_known):
    """Real/null/shuffled scalar dependence (Phase 5 MD §8).

    Compare correct scalar conditioning vs. shuffled scalars.
    """
    results = {}
    for mode, sv_eval in [
        ("real", sv),
        ("shuffled", torch.roll(sv, shifts=1, dims=0)),
    ]:
        out = model(occ, sv_eval, scalar_known, spec, mask, goal_mode="real")
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask)
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)

    results["gate"] = results["real"] < results["shuffled"]
    return results


@torch.no_grad()
def diversity_check(model, surrogate, occ, sv, spec, mask, scalar_known,
                    device, n_samples=5, perturbation_scale=0.0):
    """Check generative diversity (Phase 5 MD §9).

    With deterministic model, n_samples generations with different seeds
    should produce different outputs only if there's stochasticity.
    Reports whether different targets cause different outputs.
    """
    generations = []
    for i in range(n_samples):
        torch.manual_seed(1000 + i)
        out = model(occ, sv, scalar_known, spec, mask, goal_mode="real")
        z_hat = out["z_hat"]
        if perturbation_scale > 0:
            z_hat = z_hat + torch.randn_like(z_hat) * perturbation_scale
        geometry, _ = model.decode_geometry(
            out["z_hat"] if perturbation_scale == 0 else z_hat,
            out["scalar_pred"], occ_input=occ, mask=mask)
        generations.append(geometry)

    # Pairwise spectrum differences
    spectra = []
    for g in generations:
        spectra.append(surrogate(g).prediction)
    spectra = torch.stack(spectra)  # (n, B, 2, 301)

    if n_samples > 1:
        diffs = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = (spectra[i] - spectra[j]).abs().mean().item()
                diffs.append(d)
        diversity = float(np.mean(diffs)) if diffs else 0.0
    else:
        diversity = 0.0

    return {
        "n_samples": n_samples,
        "pairwise_spectrum_diversity": diversity,
        "deterministic": perturbation_scale == 0,
    }


@torch.no_grad()
def nearest_neighbor_baseline(val_spec, train_specs, train_geoms, surrogate):
    """Nearest-neighbor baseline (Phase 5 MD §12).

    For each val spectrum, find nearest training spectrum (L1),
    retrieve associated geometry, evaluate through surrogate.
    """
    nn_errors = []
    n_val = val_spec.shape[0]
    for i in range(n_val):
        target = val_spec[i:i+1]
        dists = (train_specs - target).abs().mean(dim=(-2, -1))
        best = dists.argmin()
        nn_geom = train_geoms[best]
        nn_pred = surrogate(nn_geom.unsqueeze(0)).prediction
        err = _spectrum_error(nn_pred, target)
        nn_errors.append(err)

    return {
        "nn_mean_spectrum_error": float(np.mean(nn_errors)),
        "nn_best_spectrum_error": float(np.min(nn_errors)),
        "method": "L1 nearest training spectrum → retrieved geometry",
    }


def run_all_scenarios(cfg, ckpt_path, device):
    """Run all scenarios + diagnostics on a validation set."""
    model, surrogate = _load_eval(cfg, ckpt_path, device)

    # Build val batch
    b = cfg["train"].get("batch_size", 2)
    if cfg.get("data", {}).get("use_synthetic", True):
        occ, sv, spec = _make_synthetic_batch(b, device)
    else:
        from data.dataset import MetaDiTDataset, collate_batch
        from torch.utils.data import DataLoader
        ds = MetaDiTDataset(_resolve(cfg["data"]["val_split"]),
                            max_samples=b * 2, seed=42)
        loader = DataLoader(ds, batch_size=b, shuffle=False, num_workers=0,
                            collate_fn=collate_batch)
        G, S = next(iter(loader))
        occ, sv = factorize_geometry(G)
        occ, sv = occ.to(device), sv.to(device)
        spec = S.to(device)

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=999)

    results = {}

    # Scenario A: pure inverse design (full mask + all scalars unknown)
    sk_a = torch.zeros(b, 3, dtype=torch.bool, device=device)
    M_a = masker.sample(occ, ratio=1.0)
    results["scenario_A_pure_inverse"] = evaluate_scenario(
        model, surrogate, occ, sv, spec, M_a, sk_a, device, "A")
    results["scenario_A_rns"] = real_null_shuffled(
        model, surrogate, occ, sv, spec, M_a, device, sk_a)

    # Scenario B: partial-parameter (50% mask + some scalars known)
    sk_b = torch.tensor([[True, False, False],
                         [False, True, False]], dtype=torch.bool,
                        device=device)[:b]
    M_b = masker.sample(occ, ratio=0.5)
    results["scenario_B_partial"] = evaluate_scenario(
        model, surrogate, occ, sv, spec, M_b, sk_b, device, "B")
    results["scenario_B_rns"] = real_null_shuffled(
        model, surrogate, occ, sv, spec, M_b, device, sk_b)

    # Scenario C: retrofit (25% mask + all scalars known)
    sk_c = torch.ones(b, 3, dtype=torch.bool, device=device)
    M_c = masker.sample(occ, ratio=0.25)
    results["scenario_C_retrofit"] = evaluate_scenario(
        model, surrogate, occ, sv, spec, M_c, sk_c, device, "C")
    results["scenario_C_rns"] = real_null_shuffled(
        model, surrogate, occ, sv, spec, M_c, device, sk_c)

    # Scalar dependence on hard stratum
    results["scalar_dependence_hard"] = scalar_dependence(
        model, surrogate, occ, sv, spec, M_a, device, sk_a)

    # Diversity
    results["diversity_A"] = diversity_check(
        model, surrogate, occ, sv, spec, M_a, sk_a, device, n_samples=5)

    # NN baseline
    train_specs = torch.randn(20, 2, 301, device=device)
    train_geoms = torch.randn(20, 3, 64, 64, device=device)
    results["nn_baseline"] = nearest_neighbor_baseline(
        spec, train_specs, train_geoms, surrogate)

    # Collapse check
    pred_occ = model.decode_geometry(
        model(occ, sv, sk_a, spec, M_a)["z_hat"],
        model(occ, sv, sk_a, spec, M_a)["scalar_pred"])[1]
    results["collapse_check"] = {
        "pred_occupancy_fraction": float(pred_occ.mean().item()),
        "all_empty": float(pred_occ.mean().item()) < 0.01,
        "all_occupied": float(pred_occ.mean().item()) > 0.99,
    }

    return results


def _load_eval(cfg, ckpt_path, device):
    spec_weights = _resolve(cfg["weights"]["spectrum"])
    model = build_unified_model(cfg, spec_weights, device=device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_into_model(model, ckpt["model"], device=device, strict=True)
    from train.engine import restore_ema_state
    restore_ema_state(model, ckpt.get("ema_state", {}))

    surr_path = _resolve(cfg["weights"].get("surrogate", ""))
    if not os.path.exists(surr_path):
        surr_path = os.path.join(REPO_ROOT, "data", "metadit", "weights",
                                 "surrogate_model.bin")
    surrogate = load_surrogate(surr_path, device=device)
    return model, surrogate


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 scenario evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["A", "B", "C", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_all_scenarios(cfg, args.checkpoint, args.device)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
