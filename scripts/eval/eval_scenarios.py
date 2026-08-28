"""Phase 5 — Scenario evaluation (unified_jepa/Phase 5 MD §11, fix pass).

Authoritative scientific evaluator for the unified JEPA model.

Report separately:
    pure inverse design       (Scenario A)
    partial-parameter         (Scenario B)
    retrofit                  (Scenario C)

Never pool results across scenarios.

Run:
    python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt --scenario all

Normal invocation uses the REAL validation split. Synthetic data is allowed
only under the explicit --smoke flag (never an implicit fallback).
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
from runtime.physics_controls import make_shuffled_spectrum


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _make_synthetic_batch(b, device):
    """Explicit smoke-only synthetic batch (Fix 7: never the default)."""
    torch.manual_seed(123)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0
    sv = (torch.rand(b, 3) * 10 + 1).to(device)
    spec = torch.randn(b, 2, 301).to(device)
    return occ, sv, spec


def _occupancy_metrics(pred_occ, true_occ, mask=None):
    """IoU/F1 for binary occupancy; when mask provided, computes masked-region
    (completion) and visible-region metrics separately (Fix 14)."""
    pred_bin = (pred_occ > 0.5).float()
    true_bin = (true_occ > 0.5).float()

    def _region_metrics(p, t):
        tp = ((p * t) > 0).sum().item()
        fp = ((p * (1 - t)) > 0).sum().item()
        fn = (((1 - p) * t) > 0).sum().item()
        iou = tp / max(1, tp + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}

    out = {"pred_occupancy_fraction": float(pred_bin.mean().item()),
           "true_occupancy_fraction": float(true_bin.mean().item())}

    if mask is not None:
        # mask: (B,16,16), 1=visible, 0=masked. Upsample to pixel space.
        up = mask.view(mask.shape[0], 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
        vis = up > 0.5
        masked = ~vis
        out["masked_region"] = _region_metrics(pred_bin[masked], true_bin[masked])
        out["visible_region"] = _region_metrics(pred_bin[vis], true_bin[vis])
    else:
        out.update(_region_metrics(pred_bin, true_bin))
    return out


def _spectrum_error(pred_spec, target_spec):
    """Normalized L1 spectrum error."""
    std = target_spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return float(((pred_spec - target_spec) / std).abs().mean().item())


@torch.no_grad()
def evaluate_scenario(model, surrogate, occ, sv, spec, mask, scalar_known,
                      device, scenario_name):
    """Evaluate a single scenario, returning all required metrics.

    Fix 7: uses the exact model signature
        model(occupancy, scalar_values, scalar_known, spectrum, mask, ...)
    Fix 14: scalar MAE reported separately for known (0 by construction) and
    unknown positions; occupancy metrics split by masked/visible region.
    """
    model.eval()
    surrogate.eval()

    out = model(occ, sv, scalar_known, spec, mask, goal_mode="real")
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=occ, mask=mask, use_ste=False,
        scalar_known=scalar_known, scalar_values=sv)
    spectrum_pred = surrogate(geometry).prediction

    spec_err = _spectrum_error(spectrum_pred, spec)

    # Scalar MAE: known vs unknown reported separately (Fix 14).
    unknown = ~scalar_known
    known = scalar_known
    scalar_mae_unknown = float(
        (out["scalar_pred"] - sv)[unknown].abs().mean().item()) if unknown.any() else 0.0
    scalar_mae_known = float(
        (out["scalar_pred"] - sv)[known].abs().mean().item()) if known.any() else 0.0

    occ_metrics = _occupancy_metrics(soft_occ, occ, mask=mask)

    return {
        "scenario": scenario_name,
        "spectrum_error": spec_err,
        "scalar_mae_unknown": scalar_mae_unknown,
        "scalar_mae_known": scalar_mae_known,
        **occ_metrics,
    }


@torch.no_grad()
def real_null_shuffled(model, surrogate, occ, sv, spec, mask, device,
                       scalar_known=None, generator=None):
    """Real/null/shuffled goal dependence (Phase 4/5 MD §10).

    Fix 8: uses make_shuffled_spectrum (canonical derangement). Requires
    B >= 2 for a meaningful shuffled control; otherwise the shuffled gate is
    marked infeasible rather than claiming a comparison.
    """
    if scalar_known is None:
        scalar_known = torch.ones(occ.shape[0], 3, dtype=torch.bool,
                                  device=device)
    b = occ.shape[0]

    results = {}
    for mode in ("real", "null", "shuffled"):
        if mode == "shuffled":
            if b < 2:
                results["shuffled"] = None
                continue
            spec_eval = make_shuffled_spectrum(spec, generator=generator)
        elif mode == "null":
            spec_eval = torch.zeros_like(spec)
        else:
            spec_eval = spec

        out = model(occ, sv, scalar_known, spec_eval, mask, goal_mode=mode)
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv)
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)

    results["gap"] = {}
    if results.get("shuffled") is not None:
        results["gap"] = {
            "real_minus_null": results["null"] - results["real"],
            "real_minus_shuffled": results["shuffled"] - results["real"],
            "gate": results["real"] < results["shuffled"],
        }
    else:
        results["gap"] = {
            "real_minus_null": results["null"] - results["real"],
            "shuffled_infeasible": "batch size < 2 (no valid derangement)",
        }
    return results


@torch.no_grad()
def scalar_dependence(model, surrogate, occ, sv, spec, mask, device,
                      scalar_known):
    """Scalar conditioning dependence (Phase 5 MD §8, Fix 9).

    Evaluated with a NON-EMPTY known-scalar subset — the all-unknown regime
    zeroes all scalar inputs, so real-vs-shuffled would be identical inputs
    and cannot prove scalar usage. Keeps the true original scalars for the
    decode/assembly path; only the conditioning input is perturbed in the
    shuffled branch (Fix 9)."""
    assert scalar_known.any(), (
        "scalar_dependence requires at least one known scalar; the all-unknown "
        "regime cannot demonstrate scalar usage")
    b = occ.shape[0]
    if b < 2:
        return {"gate": None, "shuffled_infeasible": "batch size < 2"}

    results = {}
    for mode, sv_cond in [
        ("real", sv),
        ("shuffled", make_shuffled_spectrum(sv, seed=0)),
    ]:
        out = model(occ, sv_cond, scalar_known, spec, mask, goal_mode="real")
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv)  # true values preserved
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)

    results["gate"] = results["real"] < results["shuffled"]
    return results


@torch.no_grad()
def diversity_check(model, surrogate, occ, sv, spec, mask, scalar_known,
                    device, n_samples=5, perturbation_scale=0.0):
    """Check generative diversity (Phase 5 MD §9)."""
    generations = []
    for i in range(n_samples):
        torch.manual_seed(1000 + i)
        out = model(occ, sv, scalar_known, spec, mask, goal_mode="real")
        z_hat = out["z_hat"]
        if perturbation_scale > 0:
            z_hat = z_hat + torch.randn_like(z_hat) * perturbation_scale
        geometry, _ = model.decode_geometry(
            out["z_hat"] if perturbation_scale == 0 else z_hat,
            out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv)
        generations.append(geometry)

    spectra = []
    for g in generations:
        spectra.append(surrogate(g).prediction)
    spectra = torch.stack(spectra)

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
def nearest_neighbor_baseline(val_spec, train_specs, train_occupancy,
                              train_scalars, surrogate):
    """Real training-split nearest-neighbor baseline (Fix 15).

    val target spectrum → nearest spectrum in the REAL training split →
    associated REAL training geometry (assembled from stored occupancy +
    scalars only for the selected neighbors) → frozen surrogate → error.
    """
    nn_errors = []
    nn_dists = []
    n_val = val_spec.shape[0]
    for i in range(n_val):
        target = val_spec[i:i+1]
        dists = (train_specs - target).abs().mean(dim=(-2, -1))
        best = int(dists.argmin())
        nn_dists.append(float(dists[best].item()))
        # Assemble only the selected neighbor's geometry (memory-safe).
        occ_b = train_occupancy[best:best+1]
        sc_b = train_scalars[best:best+1]
        nn_geom = assemble_metadit_geometry(
            occ_b, sc_b[:, 0], sc_b[:, 1], sc_b[:, 2])
        nn_pred = surrogate(nn_geom).prediction
        nn_errors.append(_spectrum_error(nn_pred, target))

    return {
        "nn_mean_spectrum_error": float(np.mean(nn_errors)),
        "nn_best_spectrum_error": float(np.min(nn_errors)),
        "nn_mean_retrieval_distance": float(np.mean(nn_dists)),
        "method": "L1 nearest REAL training spectrum → retrieved REAL geometry",
    }


def _load_val_batch(cfg, device, smoke):
    """Authoritative real-data validation batch (Fix 7: real by default)."""
    b = cfg["train"].get("batch_size", 2)
    if smoke:
        return _make_synthetic_batch(b, device)
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    val_path = _resolve(cfg["data"]["val_split"])
    if not os.path.exists(val_path):
        raise RuntimeError(
            f"real validation split missing: {val_path}. Evaluation in real "
            "mode requires the real dataset; use --smoke for synthetic only.")
    ds = MetaDiTDataset(val_path, max_samples=b * 2, seed=42)
    loader = DataLoader(ds, batch_size=b, shuffle=False, num_workers=0,
                        collate_fn=collate_batch)
    G, S = next(iter(loader))
    occ, sv = factorize_geometry(G)
    return occ.to(device), sv.to(device), S.to(device)


def _load_train_representations(cfg, device, smoke, n_train=200):
    """Load real training split in factorized (memory-safe) form for the NN
    baseline (Fix 15). Returns (train_spectra, train_occupancy, train_scalars)."""
    if smoke:
        b = cfg["train"].get("batch_size", 2)
        torch.manual_seed(7)
        occs, svs, specs = [], [], []
        for _ in range(n_train // b):
            o, s, sp = _make_synthetic_batch(b, device)
            occs.append(o)
            svs.append(s)
            specs.append(sp)
        return (torch.cat(specs), torch.cat(occs), torch.cat(svs))
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    train_path = _resolve(cfg["data"]["train_split"])
    if not os.path.exists(train_path):
        raise RuntimeError(f"real training split missing for NN baseline: {train_path}")
    ds = MetaDiTDataset(train_path, max_samples=n_train, seed=0)
    loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 2),
                        shuffle=False, num_workers=0, collate_fn=collate_batch)
    occs, svs, specs = [], [], []
    for G, S in loader:
        o, sv = factorize_geometry(G)
        occs.append(o)
        svs.append(sv)
        specs.append(S)
    return torch.cat(specs).to(device), torch.cat(occs).to(device), \
        torch.cat(svs).to(device)


def run_all_scenarios(cfg, ckpt_path, device, smoke=False):
    """Run all scenarios + diagnostics on the real validation split."""
    model, surrogate = _load_eval(cfg, ckpt_path, device)
    b = cfg["train"].get("batch_size", 2)

    occ, sv, spec = _load_val_batch(cfg, device, smoke)
    if occ.shape[0] < 2:
        # Shuffled controls need B >= 2; pad by reloading a larger batch.
        raise RuntimeError("validation batch must have >= 2 samples for "
                           "shuffled-spectrum controls")

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

    # Scalar dependence on a NON-EMPTY known-scalar stratum (Fix 9):
    # fully-masked occupancy + exactly one known scalar.
    sk_one = torch.zeros(b, 3, dtype=torch.bool, device=device)
    sk_one[:, 0] = True
    results["scalar_dependence_one_known"] = scalar_dependence(
        model, surrogate, occ, sv, spec, M_a, device, sk_one)
    sk_two = torch.zeros(b, 3, dtype=torch.bool, device=device)
    sk_two[:, :2] = True
    results["scalar_dependence_two_known"] = scalar_dependence(
        model, surrogate, occ, sv, spec, M_a, device, sk_two)

    # Diversity
    results["diversity_A"] = diversity_check(
        model, surrogate, occ, sv, spec, M_a, sk_a, device, n_samples=5)

    # NN baseline on the REAL training split (Fix 15).
    train_specs, train_occ, train_sv = _load_train_representations(
        cfg, device, smoke, n_train=200)
    results["nn_baseline"] = nearest_neighbor_baseline(
        spec, train_specs, train_occ, train_sv, surrogate)

    # Collapse check
    out_c = model(occ, sv, sk_a, spec, M_a)
    pred_occ = model.decode_geometry(
        out_c["z_hat"], out_c["scalar_pred"],
        scalar_known=sk_a, scalar_values=sv)[1]
    results["collapse_check"] = {
        "pred_occupancy_fraction": float(pred_occ.mean().item()),
        "all_empty": float(pred_occ.mean().item()) < 0.01,
        "all_occupied": float(pred_occ.mean().item()) > 0.99,
    }

    results["_data_mode"] = "SMOKE (synthetic)" if smoke else "REAL"
    return results


def _load_eval(cfg, ckpt_path, device):
    spec_weights = _resolve(cfg["weights"]["spectrum"])
    if not os.path.exists(spec_weights):
        raise RuntimeError(
            f"released spectrum encoder missing: {spec_weights}")
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
        description="Phase 5 scenario evaluation (authoritative)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["A", "B", "C", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="Explicit smoke mode: synthetic data allowed. "
                             "Never used for scientific evaluation.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_all_scenarios(cfg, args.checkpoint, args.device,
                                smoke=args.smoke)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
