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
from runtime.physics_controls import (
    make_shuffled_spectrum, derange_batch_tensor,
)


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


def _scenario_b_known_flags(b, device):
    """Scenario-B scalar-known flags for ARBITRARY batch size (Fix 5, spec §7).

    Deterministic rows, each with exactly one known scalar, rotating over
    l -> h -> r so all three single-known variants appear (audit B11):
        row 0: [True,  False, False]
        row 1: [False, True,  False]
        row 2: [False, False, True]
        row 3: [True,  False, False]
        ...
    """
    rows = []
    for i in range(b):
        row = [False, False, False]
        row[i % 3] = True  # rotate l-known / h-known / r-known
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)


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


def _spectrum_error_per_sample(pred_spec, target_spec):
    """Per-sample normalized L1 spectrum error, shape (B,) — audit B27.

    The real-vs-shuffled gate is a PAIRED comparison over samples, so the batch
    mean alone hides how many samples actually support it (and how variable the
    difference is). With the historical evaluation batch of 2 the mean was the
    only thing reported, and the gate was decided by a single swap.
    """
    std = target_spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return ((pred_spec - target_spec) / std).abs().mean(dim=(-2, -1))


def _spectrum_error(pred_spec, target_spec):
    """Normalized L1 spectrum error (batch mean)."""
    return float(_spectrum_error_per_sample(pred_spec, target_spec).mean().item())


@torch.no_grad()
def evaluate_scenario(model, surrogate, occ, sv, spec, mask, scalar_known,
                      device, scenario_name):
    """Evaluate a single scenario, returning all required metrics.

    Fix 7: uses the exact model signature
        model(occupancy, scalar_values, scalar_known, spectrum, mask, ...)
    Fix 14: scalar MAE reported separately for known and unknown positions
    (the known-position value is the head's raw prediction error — known values
    are substituted only later at assembly, not in scalar_pred); occupancy
    metrics split by masked/visible region.
    Fix 4 (scientific deployment): the SPECTRUM metric measures the geometry
    that would actually be deployed — occupancy logits hard-thresholded to
    binary via hard_forward=True with visible pixels retained, then MetaDiT
    assembly → surrogate.
    Audit B7: the occupancy IoU/F1/fraction diagnostics are computed on the
    model's RAW sigmoid occupancy (decode_occupancy_prob) — the retained/hard
    occupancy would make visible-region IoU identically 1.0 and hide the
    model's true occupancy quality.
    """
    model.eval()
    surrogate.eval()

    out = model(occ, sv, scalar_known, spec, mask, goal_mode="real")
    # Deployed (binary) geometry for the scientific spectrum metric.
    geometry, _ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=occ, mask=mask, use_ste=False,
        scalar_known=scalar_known, scalar_values=sv,
        hard_forward=True)
    spectrum_pred = surrogate(geometry).prediction

    spec_err = _spectrum_error(spectrum_pred, spec)

    # Scalar MAE: known vs unknown reported separately (Fix 14).
    unknown = ~scalar_known
    known = scalar_known
    scalar_mae_unknown = float(
        (out["scalar_pred"] - sv)[unknown].abs().mean().item()) if unknown.any() else 0.0
    scalar_mae_known = float(
        (out["scalar_pred"] - sv)[known].abs().mean().item()) if known.any() else 0.0

    # Occupancy quality on the RAW sigmoid probability (audit B7).
    raw_prob = model.decode_occupancy_prob(
        out["z_hat"], out["scalar_pred"],
        scalar_known=scalar_known, scalar_values=sv)
    occ_metrics = _occupancy_metrics(raw_prob, occ, mask=mask)

    return {
        "scenario": scenario_name,
        "spectrum_error": spec_err,
        "scalar_mae_unknown": scalar_mae_unknown,
        "scalar_mae_known": scalar_mae_known,
        **occ_metrics,
    }


@torch.no_grad()
def real_null_shuffled(model, surrogate, occ, sv, spec, mask, device,
                       scalar_known=None, generator=None, seed=None,
                       gate_threshold=0.5):
    """Real/null/shuffled goal dependence (Phase 4/5 MD §10).

    Fix 8: uses make_shuffled_spectrum (canonical derangement). Requires
    B >= 2 for a meaningful shuffled control; otherwise the shuffled gate is
    marked infeasible rather than claiming a comparison.
    Audit B12: pass `seed` (or an explicit generator) so the derangement is
    reproducible — an unseeded control changes run to run for B > 2.
    """
    if scalar_known is None:
        scalar_known = torch.ones(occ.shape[0], 3, dtype=torch.bool,
                                  device=device)
    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    b = occ.shape[0]

    results = {}
    per_sample = {}
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
        # Fix 4 (scientific deployment): binary deployed occupancy for the
        # spectrum comparison (same hard_forward rule as evaluate_scenario).
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv, hard_forward=True)
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)
        per_sample[mode] = _spectrum_error_per_sample(
            spectrum_pred, spec).detach().cpu()

    results["gap"] = {}
    if results.get("shuffled") is not None:
        # Audit B27: report the PAIRED per-sample statistics, not just the batch
        # means. The gate is `real < shuffled` per scenario; with a small
        # evaluation batch the means alone cannot distinguish a real effect from
        # one swap, so the fraction of samples supporting the comparison and the
        # spread of the paired difference are part of the result.
        paired = per_sample["shuffled"] - per_sample["real"]
        results["gap"] = {
            "real_minus_null": results["null"] - results["real"],
            "real_minus_shuffled": results["shuffled"] - results["real"],
            **_gate_beats_fraction(per_sample["real"], per_sample["shuffled"],
                                   gate_threshold),
            "n_samples": int(b),
            "paired_diff_mean": float(paired.mean().item()),
            "paired_diff_std": float(paired.std(unbiased=True).item()) if b > 1 else None,
            "per_sample_real": [float(v) for v in per_sample["real"]],
            "per_sample_shuffled": [float(v) for v in per_sample["shuffled"]],
        }
    else:
        results["gap"] = {
            "real_minus_null": results["null"] - results["real"],
            "shuffled_infeasible": "batch size < 2 (no valid derangement)",
        }
    return results


def _gate_threshold(cfg):
    """Primary-gate win-rate threshold (config `eval.gate_beats_fraction_min`)."""
    return float(cfg.get("eval", {}).get("gate_beats_fraction_min", 0.5))


def _gate_beats_fraction(per_sample_real, per_sample_shuffled, threshold):
    """Primary gate: the PAIRED per-sample win rate (operator decision 2026-09-13).

    The gate used to be `mean(real) < mean(shuffled)` on a batch. Measured on the
    full validation split (17,488 samples) the error distribution has a heavy
    tail — 27 samples (0.15 %) score 1–30 while the median is 0.075 — so a single
    catastrophic sample can flip a mean-based verdict on a small batch, and did:
    the 32-sample gate read real 0.8779 against a full-split mean of 0.1196.

    The paired win rate counts samples rather than magnitudes, so one outlier
    cannot decide it: it answers "on how many samples does the true conditioning
    beat the wrong one?" rather than "by how much on average". Measured here it is
    0.9939 over the full split.

    Reported alongside, never replacing, the mean difference: a model could in
    principle win by a hair on most samples and lose badly on a few, which the win
    rate alone would not show.
    """
    wins = float((per_sample_real < per_sample_shuffled).float().mean().item())
    return {
        "gate": bool(wins > float(threshold)),
        "gate_statistic": "real_beats_shuffled_fraction",
        "gate_threshold": float(threshold),
        "real_beats_shuffled_fraction": wins,
        "gate_mean_criterion": bool(
            float(per_sample_real.mean()) < float(per_sample_shuffled.mean())),
        "gate_mean_criterion_note": (
            "secondary: mean(real) < mean(shuffled), reported for continuity but "
            "no longer the primary statistic (fragile to the error tail)"),
    }


@torch.no_grad()
def scalar_dependence(model, surrogate, occ, sv, spec, mask, device,
                      scalar_known, gate_threshold=0.5):
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
    per_sample = {}
    for mode, sv_cond in [
        ("real", sv),
        # Scalar control: derange the scalar conditioning VALUES via the
        # generic batch-tensor derangement (cleanup item 4). The decode/assembly
        # path below still receives the TRUE sv for known positions.
        ("shuffled", derange_batch_tensor(sv, seed=0)),
    ]:
        out = model(occ, sv_cond, scalar_known, spec, mask, goal_mode="real")
        # Fix 4 (scientific deployment): binary deployed occupancy for the
        # spectrum comparison.
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv,  # true values preserved
            hard_forward=True)
        spectrum_pred = surrogate(geometry).prediction
        results[mode] = _spectrum_error(spectrum_pred, spec)
        per_sample[mode] = _spectrum_error_per_sample(
            spectrum_pred, spec).detach().cpu()

    results.update(_gate_beats_fraction(
        per_sample["real"], per_sample["shuffled"], gate_threshold))
    return results


@torch.no_grad()
def spectrum_sensitivity_probe(model, surrogate, occ, sv, spec, mask, scalar_known,
                               device, scales=(0.0, 0.01, 0.05, 0.10), seed=1234):
    """Does the decoded design track the target spectrum? (architecture_v5.md §8.3)

    The spec requires: "perturb the target spectrum slightly with everything else
    fixed and confirming the decoded design changes proportionally, not just that
    raw latent variance looks healthy". That probe did not exist — the evaluator's
    `diversity_check` runs at `perturbation_scale=0`, which is a DETERMINISM check
    (its own docstring says the result "must not be presented as genuine generative
    diversity"), and with `perturbation_scale > 0` it perturbs the LATENT, not the
    target spectrum. So the one family of check that tests output-diversity collapse
    across varying conditions was never measured.

    This is the literal probe: for a set of relative perturbation scales, add
    `scale * per-sample-spectrum-std * fixed-noise` to the TARGET SPECTRUM and
    measure how far the decoded design moves. Everything else — occupancy, scalars,
    mask, model weights — is fixed, and the noise comes from a fixed generator so
    the curve is reproducible.

    A flat curve means the design does not respond to the target at all: exactly the
    Failure Mode 2 that a healthy latent-space metric would not catch (§8.3).

    Returns the curve plus `design_moves` (did ANY non-zero scale move the design)
    and whether the response is monotone in the scale.
    """
    gen = torch.Generator().manual_seed(int(seed))
    unit_noise = torch.randn(spec.shape, generator=gen).to(spec.device)
    per_sample_std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    def decode(spec_in):
        out = model(occ, sv, scalar_known, spec_in, mask, goal_mode="real",
                    with_target=False)
        geometry, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv, hard_forward=True)
        prob = model.decode_occupancy_prob(
            out["z_hat"], out["scalar_pred"], scalar_known=scalar_known,
            scalar_values=sv)
        return geometry, (prob > 0.5).float(), prob.flatten(1).mean(dim=1)

    base_geom, base_bin, base_frac = decode(spec)
    curve = {}
    any_move = False
    for s in scales:
        s = float(s)
        spec_p = spec if s == 0.0 else spec + s * per_sample_std * unit_noise
        geom, binary, frac = decode(spec_p)
        geom_move = float((geom - base_geom).norm() / (base_geom.norm() + 1e-12))
        flipped = float((binary != base_bin).float().mean())
        frac_shift = float((frac - base_frac).abs().mean())
        curve[str(s)] = {"geometry_relative_change": geom_move,
                         "occupancy_pixels_flipped": flipped,
                         "predicted_occupancy_fraction_shift": frac_shift}
        if s > 0.0 and (flipped > 0.0 or geom_move > 1e-6):
            any_move = True
    monotone = all(
        curve[str(scales[i])]["occupancy_pixels_flipped"]
        <= curve[str(scales[i + 1])]["occupancy_pixels_flipped"] + 1e-12
        for i in range(len(scales) - 1))
    return {
        "scales": [float(s) for s in scales],
        "curve": curve,
        "design_moves": bool(any_move),
        "pixels_flipped_monotone_in_scale": bool(monotone),
        "note": ("architecture_v5.md §8.3: a flat curve means the design does not "
                 "track the target spectrum (Failure Mode 2); the largest scale's "
                 "occupancy_pixels_flipped is the headline number."),
    }


@torch.no_grad()
def diversity_check(model, surrogate, occ, sv, spec, mask, scalar_known,
                    device, n_samples=5, perturbation_scale=0.0):
    """Repeated-generation diagnostic for the DETERMINISTIC inverse mapping.

    Cleanup item 5: the unified inverse-design path has NO stochastic
    latent/noise input — the mapping (context, goal) -> geometry is
    deterministic. There is deliberately no manufactured generative
    stochasticity. This diagnostic therefore:
      - with perturbation_scale == 0 (default): verifies DETERMINISM —
        repeated generations from the same input must be identical, so the
        reported pairwise diversity is 0 and "deterministic" is True;
      - with perturbation_scale > 0: applies an EXPLICIT latent-space
        perturbation as a diagnostic-only sensitivity probe. This is NOT an
        architectural stochastic mechanism and must not be presented as
        genuine generative diversity.

    Returns:
        dict with n_samples, pairwise_spectrum_diversity, and the
        "deterministic" flag (True iff perturbation_scale == 0).
    """
    generations = []
    for i in range(n_samples):
        torch.manual_seed(1000 + i)
        out = model(occ, sv, scalar_known, spec, mask, goal_mode="real")
        z_hat = out["z_hat"]
        if perturbation_scale > 0:
            z_hat = z_hat + torch.randn_like(z_hat) * perturbation_scale
        # Fix 4 (scientific deployment): binary deployed occupancy for the
        # generated geometry (same hard_forward rule as evaluate_scenario).
        geometry, _ = model.decode_geometry(
            out["z_hat"] if perturbation_scale == 0 else z_hat,
            out["scalar_pred"], occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv, hard_forward=True)
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
def cfg_guidance_sweep(model, surrogate, occ, sv, spec, mask, scalar_known,
                       device, weights=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)):
    """Classifier-free guidance at inference, swept over the guidance weight
    (architecture_v5.md §3.5.1).

    Audit B24: training prepares the unconditional branch (goal dropout +
    null-goal steps) but **nothing in the pipeline ever called `cfg_forward`**,
    so `w` was never exercised and the CFG machinery was dead code. This wires
    it into the authoritative evaluator.

    Run on the hard stratum, where the design's gates apply. The endpoints are
    meaningful: `w = 0` is the pure-null (unconditional) prediction and `w = 1`
    is exactly the plain real-goal forward, so the sweep brackets conditioned
    and unconditioned. A curve that is flat in `w` means the goal conditioning
    has no effect on the deployed design — the same Failure Mode 2 the
    real-vs-shuffled gate tests, measured through the guided path instead.

    Returns:
        dict mapping str(w) -> normalized spectrum error of the guided design.
    """
    from predictor.guidance import cfg_forward

    out = {}
    for w in weights:
        z_guided, scalar_guided, _ = cfg_forward(
            model, occ, sv, scalar_known, spec, mask, w)
        geometry, _ = model.decode_geometry(
            z_guided, scalar_guided, occ_input=occ, mask=mask,
            scalar_known=scalar_known, scalar_values=sv, hard_forward=True)
        spectrum_pred = surrogate(geometry).prediction
        out[str(float(w))] = float(_spectrum_error(spectrum_pred, spec))
    return out


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


def _load_val_batch(cfg, device, smoke, n_samples=None):
    """Authoritative real-data validation batch (Fix 7: real by default).

    Audit B27: the batch size is the GATE's sample size. It used to be
    `cfg["train"]["batch_size"]` (2), so every real/null/shuffled comparison, the
    shuffled control (a 2-item derangement = one swap), the scalar-dependence
    checks and the whole guidance sweep were decided by TWO samples. The
    training batch size is an optimisation choice and has no business setting the
    evaluation's statistical power, so evaluation now has its own knob:
    `eval.n_samples` in the config, overridable with `--samples`.
    """
    if n_samples is None:
        n_samples = int(cfg.get("eval", {}).get("n_samples", 32))
    b = max(2, int(n_samples))          # the shuffled control needs B >= 2
    if smoke:
        return _make_synthetic_batch(b, device)
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    val_path = _resolve(cfg["data"]["val_split"])
    if not os.path.exists(val_path):
        raise RuntimeError(
            f"real validation split missing: {val_path}. Evaluation in real "
            "mode requires the real dataset; use --smoke for synthetic only.")
    ds = MetaDiTDataset(val_path, max_samples=b, seed=42)
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


def _collapse_metrics(model, occ, sv, spec, mask, scalar_known, device):
    """Occupancy-collapse diagnostics (architecture_v5.md §8.3 check 11).

    Audit B13: the predicted occupancy fraction uses the SAME definition as the
    rest of the evaluator — the raw sigmoid probability thresholded at 0.5 —
    and reports PER-SAMPLE variability, not only a global mean (a decoder that
    predicts the majority class everywhere shows up as a near-constant
    fraction across samples).
    """
    with torch.no_grad():
        out = model(occ, sv, scalar_known, spec, mask)
        prob = model.decode_occupancy_prob(
            out["z_hat"], out["scalar_pred"],
            scalar_known=scalar_known, scalar_values=sv)
        bin_occ = (prob > 0.5).float()
        per_sample = bin_occ.flatten(1).mean(dim=1)      # (B,)
        frac = float(bin_occ.mean().item())
    return {
        "pred_occupancy_fraction": float(per_sample.mean().item()),
        "pred_occupancy_fraction_std": float(
            per_sample.std(unbiased=False).item()),
        "pred_occupancy_fraction_min": float(per_sample.min().item()),
        "pred_occupancy_fraction_max": float(per_sample.max().item()),
        "all_empty": frac < 0.01,
        "all_occupied": frac > 0.99,
    }


def _eval_seeds(cfg, eval_seed):
    """Evaluation-seed-derived stochastic seeds (A1 measurement control).

    The evaluator is NOT parameterized by a shell seed: the 512 validation items are
    fixed (`MetaDiTDataset(..., seed=42)`) and the masker is seeded once. This helper
    derives the seeds that actually vary a Scenario draw, so `--eval-seed E` gives an
    independent evaluation draw while `E=0` reproduces the historical numbers exactly
    (masker 999; derangements `train.seed + {1,2,3}`).

    Eval seeds vary MASK PLACEMENT and the SHUFFLED control only — never the items,
    the ratio, the placement mode, or the gate.
    """
    base = int(cfg.get("train", {}).get("seed", 42))
    off = int(eval_seed)
    return {"masker": 999 + off, "A": base + 1 + off, "B": base + 2 + off,
            "C": base + 3 + off}


def run_all_scenarios(cfg, ckpt_path, device, smoke=False, n_samples=None,
                      scenarios=("A", "B", "C"), eval_seed=0):
    """Run all scenarios + diagnostics on the real validation split.

    Audit B27: `n_samples` sets the evaluation batch for every comparison below —
    the scenario metrics, the real/null/shuffled gates, the shuffled control, the
    scalar-dependence checks and the guidance sweep. It defaults to
    `eval.n_samples` (32); it is deliberately independent of
    `train.batch_size`.

    `scenarios` selects which of A/B/C are evaluated (review C2: the CLI
    `--scenario` flag was parsed but ignored). The scenario-independent
    diagnostics (scalar dependence, diversity, sensitivity, CFG sweep, NN,
    collapse) always run.
    """
    model, surrogate = _load_eval(cfg, ckpt_path, device)

    occ, sv, spec = _load_val_batch(cfg, device, smoke, n_samples=n_samples)
    b = occ.shape[0]
    if b < 2:
        raise RuntimeError("validation batch must have >= 2 samples for "
                           "shuffled-spectrum controls")


    seeds = _eval_seeds(cfg, eval_seed)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seeds["masker"])
    results = {}

    # Scenario A: pure inverse design (full mask + all scalars unknown)
    sk_a = torch.zeros(b, 3, dtype=torch.bool, device=device)
    # Fix (CUDA mask bug): masker.sample returns CPU tensors — move to the
    # active device before any model forward.
    M_a = masker.sample(occ, ratio=1.0).to(device)
    assert M_a.device == occ.device, "scenario A mask must be on the model device"
    if "A" in scenarios:
        results["scenario_A_pure_inverse"] = evaluate_scenario(
            model, surrogate, occ, sv, spec, M_a, sk_a, device, "A")
        results["scenario_A_rns"] = real_null_shuffled(
            model, surrogate, occ, sv, spec, M_a, device, sk_a,
            seed=seeds["A"],
            gate_threshold=_gate_threshold(cfg))

    # Scenario B: partial-parameter (50% mask + some scalars known)
    # Fix 5 (spec §7): construct the known-flags pattern programmatically for
    # ARBITRARY batch size (the old fixed 2-row tensor sliced [:b] was only
    # safe for b <= 2). Deterministic alternating rows preserve the intended
    # representative partial-known semantics: every row has exactly one known
    # scalar, alternating which one.
    if "B" in scenarios:
        sk_b = _scenario_b_known_flags(b, device)
        M_b = masker.sample(occ, ratio=0.5).to(device)
        assert M_b.device == occ.device, "scenario B mask must be on the model device"
        results["scenario_B_partial"] = evaluate_scenario(
            model, surrogate, occ, sv, spec, M_b, sk_b, device, "B")
        results["scenario_B_rns"] = real_null_shuffled(
            model, surrogate, occ, sv, spec, M_b, device, sk_b,
            seed=seeds["B"],
            gate_threshold=_gate_threshold(cfg))

    # Scenario C: retrofit (25% mask + all scalars known)
    if "C" in scenarios:
        sk_c = torch.ones(b, 3, dtype=torch.bool, device=device)
        M_c = masker.sample(occ, ratio=0.25).to(device)
        assert M_c.device == occ.device, "scenario C mask must be on the model device"
        results["scenario_C_retrofit"] = evaluate_scenario(
            model, surrogate, occ, sv, spec, M_c, sk_c, device, "C")
        results["scenario_C_rns"] = real_null_shuffled(
            model, surrogate, occ, sv, spec, M_c, device, sk_c,
            seed=seeds["C"],
            gate_threshold=_gate_threshold(cfg))

    # Scalar dependence on a NON-EMPTY known-scalar stratum (Fix 9):
    # fully-masked occupancy + exactly one known scalar.
    sk_one = torch.zeros(b, 3, dtype=torch.bool, device=device)
    sk_one[:, 0] = True
    results["scalar_dependence_one_known"] = scalar_dependence(
        model, surrogate, occ, sv, spec, M_a, device, sk_one,
        gate_threshold=_gate_threshold(cfg))
    sk_two = torch.zeros(b, 3, dtype=torch.bool, device=device)
    sk_two[:, :2] = True
    results["scalar_dependence_two_known"] = scalar_dependence(
        model, surrogate, occ, sv, spec, M_a, device, sk_two,
        gate_threshold=_gate_threshold(cfg))

    # Diversity: determinism (same input -> identical output). Its docstring is
    # explicit that this is NOT a spectrum-sensitivity measurement.
    results["diversity_A"] = diversity_check(
        model, surrogate, occ, sv, spec, M_a, sk_a, device, n_samples=5)

    # architecture_v5.md §8.3's actual probe: perturb the TARGET SPECTRUM and
    # confirm the decoded design tracks it. This was the missing gate.
    results["spectrum_sensitivity_A"] = spectrum_sensitivity_probe(
        model, surrogate, occ, sv, spec, M_a, sk_a, device)

    # Classifier-free guidance sweep on the hard stratum (audit B24: cfg_forward
    # previously had no caller anywhere in the pipeline, so the guidance weight
    # was never exercised).
    results["cfg_guidance_sweep_A"] = cfg_guidance_sweep(
        model, surrogate, occ, sv, spec, M_a, sk_a, device)

    # NN baseline on the REAL training split (Fix 15).
    train_specs, train_occ, train_sv = _load_train_representations(
        cfg, device, smoke, n_train=200)
    results["nn_baseline"] = nearest_neighbor_baseline(
        spec, train_specs, train_occ, train_sv, surrogate)

    # Collapse check (audit B13): same occupancy definition as the evaluator
    # (raw-sigmoid threshold) with per-sample fraction variability.
    results["collapse_check"] = _collapse_metrics(
        model, occ, sv, spec, M_a, sk_a, device)

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
        raise RuntimeError(
            f"surrogate weights not found at the configured path "
            f"{surr_path!r} — refusing to silently fall back to a hardcoded "
            "location (audit B18); stage the released surrogate or fix "
            "weights.surrogate")
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
    parser.add_argument("--samples", type=int, default=None,
                        help="Evaluation batch size = the gate's sample size and "
                             "the sensitivity of EVERY comparison it reports "
                             "(audit B27). Defaults to eval.n_samples in the "
                             "config (32); must be >= 2 for the shuffled "
                             "control.")

    parser.add_argument("--smoke", action="store_true",
                        help="Explicit smoke mode: synthetic data allowed. "
                             "Never used for scientific evaluation.")
    parser.add_argument("--out", type=str, default="",
                        help="write the results JSON to this path (review C1)")
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="evaluation seed: reseeds the mask draw and the "
                             "shuffled control (A1). 0 reproduces the historical "
                             "numbers exactly; items/split/ratio/gate unchanged.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scenarios = (("A", "B", "C") if args.scenario == "all"
                 else (args.scenario,))
    results = run_all_scenarios(cfg, args.checkpoint, args.device,
                                smoke=args.smoke, n_samples=args.samples,
                                scenarios=scenarios, eval_seed=args.eval_seed)
    text = json.dumps(results, indent=2, default=float)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
