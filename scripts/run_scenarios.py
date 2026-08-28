"""Phase 4 — Scenario evaluation for unified JEPA (Phase 4 MD §7-§12).

Implements:
- Scenario A: pure inverse design (full mask + all scalars unknown + target spectrum)
- Scenario B: partial-parameter conditioning (masked + some scalars known)
- Scenario C: retrofit/constrained completion (mostly known + region masked)
- Real/null/shuffled dependence (§10)
- Nearest-neighbor baseline (§12)
- Generative diversity (§11)

Run:
    python scripts/run_scenarios.py --checkpoint checkpoints/unified/latest.pt \
        --scenario all --config configs/unified.yaml
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from assembly import build_unified_model, load_into_model, saveable_state_dict
from data.factorize import factorize_geometry, assemble_metadit_geometry, assemble_geometry
from data.mask import BlockMasker
from physics.physics_loop import load_surrogate, physics_loss


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_model_and_surrogate(cfg, ckpt_path, device):
    """Load unified model from checkpoint + frozen surrogate."""
    spec_weights = _resolve_path(cfg["weights"]["spectrum"])
    model = build_unified_model(cfg, spec_weights, device=device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_into_model(model, ckpt["model"], device=device, strict=True)
    from train.engine import restore_ema_state
    restore_ema_state(model, ckpt.get("ema_state", {}))

    surr_path = _resolve_path(cfg["weights"].get("metadit", ""))
    if not os.path.exists(surr_path):
        surr_path = os.path.join(REPO_ROOT, "data", "metadit", "weights",
                                 "surrogate_model.bin")
    surrogate = load_surrogate(surr_path, device=device)
    return model, surrogate


# ---------------------------------------------------------------------------
# Scenario inputs
# ---------------------------------------------------------------------------

class ScenarioInputs:
    """Construct scenario-specific inputs from a dataset batch."""

    def __init__(self, occ, sv, spec, mask_ratio, scalar_known, b, device):
        self.occ = occ
        self.sv = sv
        self.spec = spec
        self.mask = BlockMasker(placement="random", grid=16, min_side=3,
                                k_range=(1, 4), seed=100).sample(occ, mask_ratio)
        self.scalar_known = scalar_known
        self.b = b
        self.device = device

    @classmethod
    def scenario_a(cls, occ, sv, spec, b, device, mask_ratio=1.0):
        """Pure inverse design: full mask + all scalars unknown."""
        sk = torch.zeros(b, 3, dtype=torch.bool, device=device)
        return cls(occ, sv, spec, mask_ratio, sk, b, device)

    @classmethod
    def scenario_b(cls, occ, sv, spec, b, device, mask_ratio=0.5):
        """Partial-parameter conditioning: masked occupancy + some scalars known."""
        sk = torch.tensor([[True, False, False],
                           [False, True, False]], dtype=torch.bool,
                          device=device)[:b]
        return cls(occ, sv, spec, mask_ratio, sk, b, device)

    @classmethod
    def scenario_c(cls, occ, sv, spec, b, device, mask_ratio=0.25):
        """Retrofit/constrained completion: mostly known + small masked region."""
        sk = torch.ones(b, 3, dtype=torch.bool, device=device)
        return cls(occ, sv, spec, mask_ratio, sk, b, device)


# ---------------------------------------------------------------------------
# Inference / generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_geometry(model, inputs, surrogate, device, n_samples=1,
                      goal_mode="real"):
    """Generate geometry for a given scenario.

    Deterministic: returns 1 sample per call.
    For diversity analysis, call n_samples times with different scalar seeds.
    """
    model.eval()
    surrogate.eval()

    out = model(inputs.occ, inputs.sv, inputs.scalar_known, inputs.spec,
                inputs.mask, goal_mode=goal_mode)
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=inputs.occ, mask=inputs.mask, use_ste=False)
    spectrum_pred = surrogate(geometry).prediction

    return {
        "geometry": geometry,
        "soft_occ": soft_occ,
        "spectrum_pred": spectrum_pred,
        "target_spectrum": inputs.spec,
        "scalar_pred": out["scalar_pred"],
        "z_hat": out["z_hat"],
        "z_y_raw": out.get("z_y_raw", None),
    }


@torch.no_grad()
def spectrum_error(spectrum_pred, spectrum_target):
    """Normalized L1 spectrum error (Phase 4 MD §5). Returns scalar float."""
    std = spectrum_target.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return float(((spectrum_pred - spectrum_target) / std).abs().mean().item())


# ---------------------------------------------------------------------------
# Scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(model, surrogate, occ, sv, spec, b, device,
                      scenario_name, mask_ratio=0.5):
    """Evaluate one scenario (A, B, or C) and return metrics."""
    if scenario_name == "A":
        inputs = ScenarioInputs.scenario_a(occ, sv, spec, b, device, mask_ratio=1.0)
    elif scenario_name == "B":
        inputs = ScenarioInputs.scenario_b(occ, sv, spec, b, device, mask_ratio=mask_ratio)
    elif scenario_name == "C":
        inputs = ScenarioInputs.scenario_c(occ, sv, spec, b, device, mask_ratio=mask_ratio)
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    gen = generate_geometry(model, inputs, surrogate, device)
    spec_err = spectrum_error(gen["spectrum_pred"], gen["target_spectrum"])
    scalar_err = (gen["scalar_pred"] - sv).abs().mean().item()

    metrics = {
        "scenario": scenario_name,
        "spectrum_error": spec_err,
        "scalar_error": scalar_err,
        "mask_ratio": float((inputs.mask == 0).float().mean()),
        "scalars_unknown": float((~inputs.scalar_known).float().mean()),
    }
    return metrics, gen


# ---------------------------------------------------------------------------
# Real/null/shuffled dependence (Phase 4 MD §10)
# ---------------------------------------------------------------------------

def real_null_shuffled_evaluation(model, surrogate, occ, sv, spec, mask,
                                  device, scalar_known=None,
                                  goal_modes=("real", "null", "shuffled")):
    """Evaluate spectrum error for real, null, and shuffled goal conditions.

    Primary gate (Phase 4 MD §10): real must outperform shuffled,
    not merely real > null. The gate is evaluated on the HARD stratum
    (full occupancy mask + all scalars unknown) — callers must pass the
    appropriate scalar_known / mask for the stratum being evaluated.
    """
    b = occ.shape[0]
    if scalar_known is None:
        scalar_known = torch.ones(b, 3, dtype=torch.bool, device=device)
    results = {}

    for gm in goal_modes:
        if gm == "shuffled":
            spec_eval = torch.roll(spec, shifts=1, dims=0)
        elif gm == "null":
            spec_eval = torch.zeros_like(spec)
        else:
            spec_eval = spec

        out = model(occ, sv, scalar_known, spec_eval, mask, goal_mode=gm)
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask)
        spectrum_pred = surrogate(geometry).prediction

        # For shuffled/null, the "target" is still the original spec
        err = spectrum_error(spectrum_pred, spec)
        results[gm] = {"spectrum_error": err}

    # Primary gate (hard stratum)
    results["gate"] = {
        "real_outperforms_shuffled":
            results["real"]["spectrum_error"] < results["shuffled"]["spectrum_error"],
        "real_better_than_null":
            results["real"]["spectrum_error"] < results["null"]["spectrum_error"],
    }
    return results


# ---------------------------------------------------------------------------
# Nearest-neighbor baseline (Phase 4 MD §12)
# ---------------------------------------------------------------------------

def nearest_neighbor_baseline(val_batch, train_data, surrogate, device,
                              k=5):
    """Nearest-neighbor retrieval baseline.

    For each val spectrum, find the nearest training spectrum (by L1),
    retrieve the associated geometry, and evaluate through the surrogate.

    Args:
        val_batch: (occ [B,1,64,64], sv [B,3], spec [B,2,301])
        train_data: list of (occ, sv, spec) batches
        surrogate:  frozen MetaDiT surrogate

    Returns:
        dict with nn_spectrum_error, nn_scalar_error, nn_topk_match_rate
    """
    val_occ, val_sv, val_spec = val_batch
    b = val_occ.shape[0]
    nn_geometries = []
    nn_spectra = []

    with torch.no_grad():
        for i in range(b):
            target_spec = val_spec[i:i+1]
            best_dist = float("inf")
            best_geom = None
            best_spec = None
            for t_occ, t_sv, t_spec in train_data:
                # Per-sample comparison within the training batch
                for j in range(t_occ.shape[0]):
                    t_occ_s = t_occ[j:j+1]
                    t_sv_s = t_sv[j:j+1]
                    t_spec_s = t_spec[j:j+1]
                    dist = (t_spec_s - target_spec).abs().mean().item()
                    if dist < best_dist:
                        best_dist = dist
                        from data.factorize import assemble_metadit_geometry
                        best_geom = assemble_metadit_geometry(
                            t_occ_s, t_sv_s[:, 0], t_sv_s[:, 1], t_sv_s[:, 2])
                        best_spec = t_spec_s
                        if best_dist < 1e-10:
                            break
            nn_geometries.append(best_geom)
            nn_spectra.append(best_spec)

        nn_geom = torch.cat(nn_geometries, dim=0)
        nn_spec_pred = surrogate(nn_geom).prediction

        nn_spec_err = spectrum_error(nn_spec_pred, val_spec)
        return {
            "nn_spectrum_error": nn_spec_err,
            "nn_mean_spectrum_dist": float(best_dist),
            "nn_method": "L1 nearest training spectrum → retrieved geometry",
        }


# ---------------------------------------------------------------------------
# Generative diversity (Phase 4 MD §11)
# ---------------------------------------------------------------------------

@torch.no_grad()
def diversity_metrics(generations):
    """Compute diversity and uniqueness of generated geometries.

    Args:
        generations: list of [B, 3, 64, 64] geometry tensors (one per sample)

    Returns:
        dict with pairwise distances, uniqueness, etc.
    """
    if len(generations) <= 1:
        return {"diversity": float("nan"), "n_generations": len(generations),
                "deterministic": True}

    # Flatten geometries
    flats = [g.flatten(1) for g in generations]  # each (B, 3*64*64)
    flats = torch.cat(flats, dim=0)  # (n*B, 3*64*64)

    # Pairwise cosine distances
    norms = flats / (flats.norm(dim=-1, keepdim=True) + 1e-8)
    sim = norms @ norms.T
    dist = 1 - sim
    n = flats.shape[0]
    tri = dist.triu(diagonal=1)
    nnz = tri[tri > 0]
    if len(nnz) == 0:
        return {"diversity": 0.0, "n_pairs": n * (n - 1) // 2,
                "deterministic": True}

    return {
        "diversity_mean": float(nnz.mean().item()),
        "diversity_std": float(nnz.std().item()),
        "diversity_min": float(nnz.min().item()),
        "n_pairs": n * (n - 1) // 2,
        "deterministic": False,
    }


# ---------------------------------------------------------------------------
# Main scenario runner
# ---------------------------------------------------------------------------

def run_scenarios(cfg, ckpt_path, device, scenario="all", n_val=2):
    """Run all scenarios and return a report dict."""
    model, surrogate = load_model_and_surrogate(cfg, ckpt_path, device)

    # Build a synthetic validation batch
    torch.manual_seed(42)
    b = cfg["train"].get("batch_size", 2)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0
    sv = (torch.rand(b, 3) * 10 + 1).to(device)
    spec = torch.randn(b, 2, 301).to(device)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=999)
    M = masker.sample(occ, ratio=0.5)

    results = {}

    if scenario in ("A", "all"):
        metrics, gen = evaluate_scenario(
            model, surrogate, occ, sv, spec, b, device, "A")
        results["scenario_A"] = metrics

    if scenario in ("B", "all"):
        metrics, gen = evaluate_scenario(
            model, surrogate, occ, sv, spec, b, device, "B", mask_ratio=0.5)
        results["scenario_B"] = metrics

    if scenario in ("C", "all"):
        metrics, gen = evaluate_scenario(
            model, surrogate, occ, sv, spec, b, device, "C", mask_ratio=0.1)
        results["scenario_C"] = metrics

    if scenario in ("dependence", "all"):
        # MD §10: the primary gate is the HARD stratum — full occupancy mask +
        # all scalars unknown (the pure-inverse-design regime). Report easy
        # separately; never pool.
        M_hard = masker.sample(occ, ratio=1.0)
        sk_hard = torch.zeros(b, 3, dtype=torch.bool, device=device)
        results["real_null_shuffled_hard"] = real_null_shuffled_evaluation(
            model, surrogate, occ, sv, spec, M_hard, device,
            scalar_known=sk_hard)
        # Easy stratum (low mask + scalars known), reported separately.
        M_easy = masker.sample(occ, ratio=0.1)
        sk_easy = torch.ones(b, 3, dtype=torch.bool, device=device)
        results["real_null_shuffled_easy"] = real_null_shuffled_evaluation(
            model, surrogate, occ, sv, spec, M_easy, device,
            scalar_known=sk_easy)

    if scenario in ("baseline", "all"):
        # Synthetic training data for NN baseline
        torch.manual_seed(99)
        train_data = []
        for _ in range(10):
            t_occ = (torch.rand(b, 1, 64, 64) > 0.5).float().to(device)
            t_occ[:, :, :32, :32] = 1.0
            t_sv = (torch.rand(b, 3) * 10 + 1).to(device)
            t_spec = torch.randn(b, 2, 301).to(device)
            train_data.append((t_occ, t_sv, t_spec))
        results["nearest_neighbor"] = nearest_neighbor_baseline(
            (occ, sv, spec), train_data, surrogate, device)

    if scenario in ("diversity", "all"):
        generations = []
        for _ in range(5):
            gen = generate_geometry(
                model, ScenarioInputs.scenario_a(occ, sv, spec, b, device),
                surrogate, device)
            generations.append(gen["geometry"])
        results["diversity"] = diversity_metrics(generations)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 scenario evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["A", "B", "C", "dependence", "baseline",
                                 "diversity", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_scenarios(cfg, args.checkpoint, args.device, args.scenario)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
