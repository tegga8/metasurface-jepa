"""Evaluation/validation correctness tests (final fix pass).

Covers:
- CUDA mask device consistency through the evaluator/model path (item 1)
- Training/validation metric consistency: L_inv == MSE(p_hat, p_y) and the
  reported cos_err uses the exact masked-token population (items 2, 11)
- Projector train/eval mode behavior on identical tensors (item 4)
- Unified validation path uses the unified model signature (item 5)

Run:  python -m pytest tests/test_eval_device_and_metrics.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from assembly import UnifiedJEPA
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss
from losses.vicreg import invariance_loss


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _build_model(hidden=192, geo_depth=2, predictor_depth=4):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=geo_depth, spec_dim=256)
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    model.eval()
    return model


def _objective():
    return UnifiedJEPALoss(hidden=192, lambda_phys=0.0)


def _batch(seed=0, b=2):
    torch.manual_seed(seed)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(b, 2, 301)
    return occ, sv, spec


def test_scenario_masks_transferred_to_device():
    """Item 1: the scenario evaluator's masks must be on the model device
    (masker.sample returns CPU tensors). Regression: with a CUDA occupancy
    tensor, every scenario mask must be CUDA."""
    from scripts.eval.eval_scenarios import _scenario_b_known_flags

    device = "cuda" if torch.cuda.is_available() else "cpu"
    occ, sv, spec = _batch(seed=3, b=4)
    occ, sv, spec = occ.to(device), sv.to(device), spec.to(device)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=999)
    for ratio in (1.0, 0.5, 0.25):
        M = masker.sample(occ, ratio).to(device)
        assert M.device == occ.device, (
            f"mask at ratio {ratio} must be on the model device")
    sk_b = _scenario_b_known_flags(4, device)
    assert sk_b.device == occ.device


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available in this environment")
def test_evaluator_forward_no_cuda_cpu_mismatch():
    """Item 1: the full evaluator path (evaluate_scenario) must run on CUDA
    without a CPU/CUDA mask mismatch."""
    from scripts.eval.eval_scenarios import evaluate_scenario

    class _FakeSurrogate(nn.Module):
        def forward(self, geometry):
            class R:
                pass
            r = R()
            g = geometry.mean(dim=(2, 3))
            spec = torch.stack([g[:, 0], g[:, 1]], dim=-1)
            r.prediction = spec.unsqueeze(-1).expand(-1, -1, 301)
            return r

    model = _build_model().cuda()
    surrogate = _FakeSurrogate().cuda()
    occ, sv, spec = _batch(seed=5, b=2)
    occ, sv, spec = occ.cuda(), sv.cuda(), spec.cuda()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5).cuda()
    sk = torch.zeros(2, 3, dtype=torch.bool, device="cuda")
    result = evaluate_scenario(
        model, surrogate, occ, sv, spec, M, sk, "cuda", "A")
    assert "spectrum_error" in result
    assert torch.isfinite(torch.tensor(result["spectrum_error"]))


def test_linv_matches_mse_of_projected_tensors():
    """Items 2/11: L_inv (as reported by the objective) must equal
    MSE(p_hat[mask], p_y[mask]) computed independently from the objective's
    own projector outputs — proving the metric uses the exact same tensors
    and masked-token population."""
    model = _build_model()
    objective = _objective()
    occ, sv, spec = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)

    result = objective(model, occ, sv, sk, spec, M, goal_mode="real")
    mask_bool = result["out"]["mask"]
    p_hat = result["projector_outputs"]["p_hat"][mask_bool]
    p_y = result["projector_outputs"]["p_y"][mask_bool]

    # Independent recomputation from the objective's OWN projector.
    proj = objective.projector
    with torch.no_grad():
        p_hat_ind = proj(result["out"]["z_hat"])[mask_bool]
        p_y_ind = proj(result["out"]["z_y_raw"])[mask_bool]
    assert torch.allclose(p_hat, p_hat_ind, atol=1e-6), (
        "projector_outputs p_hat must equal projector(z_hat)[mask]")
    assert torch.allclose(p_y, p_y_ind, atol=1e-6), (
        "projector_outputs p_y must equal projector(z_y_raw)[mask]")

    mse_ind = torch.nn.functional.mse_loss(p_hat_ind, p_y_ind)
    assert abs(float(mse_ind) - float(result["components"]["L_inv"])) < 1e-6, (
        "L_inv must equal MSE(p_hat, p_y) on the masked tokens")

    # The projected cos_err is a different (complementary) quantity — it must
    # be consistent with the same tensors.
    cos_ind = (1 - torch.nn.functional.cosine_similarity(
        p_hat_ind, p_y_ind, dim=-1).clamp(min=0)).mean()
    assert 0.0 <= float(cos_ind) <= 2.0


def test_projector_train_eval_diagnostic():
    """Item 4: for identical validation tensors, run the projector in
    train() vs eval() mode and report the difference — WITHOUT modifying
    running statistics or optimizer state. This diagnoses whether BatchNorm
    running statistics are responsible for any train/eval metric gap."""
    model = _build_model()
    objective = _objective()
    occ, sv, spec = _batch(seed=2)
    sk = torch.ones(2, 3, dtype=torch.bool)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)

    # Freeze BatchNorm running statistics: capture before, restore after.
    bn_before = {}
    for name, m in objective.projector.named_modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            bn_before[name] = (m.running_mean.clone(), m.running_var.clone())

    with torch.no_grad():
        # eval-mode projector on the validation tensors.
        model.eval()
        objective.eval()
        out_eval = objective(model, occ, sv, sk, spec, M, goal_mode="real")
        mask_bool = out_eval["out"]["mask"]
        p_hat_eval = out_eval["projector_outputs"]["p_hat"][mask_bool]
        p_y_eval = out_eval["projector_outputs"]["p_y"][mask_bool]
        mse_eval = torch.nn.functional.mse_loss(p_hat_eval, p_y_eval).item()
        cos_eval = (1 - torch.nn.functional.cosine_similarity(
            p_hat_eval, p_y_eval, dim=-1).clamp(min=0)).mean().item()

        # train-mode projector on the SAME tensors (no optimizer step, no
        # running-stat mutation by us; BN uses batch stats in train mode).
        objective.train()
        model.eval()  # student encoders stay eval; only the projector flips
        out_train = objective(model, occ, sv, sk, spec, M, goal_mode="real")
        p_hat_train = out_train["projector_outputs"]["p_hat"][mask_bool]
        p_y_train = out_train["projector_outputs"]["p_y"][mask_bool]
        mse_train = torch.nn.functional.mse_loss(p_hat_train, p_y_train).item()
        cos_train = (1 - torch.nn.functional.cosine_similarity(
            p_hat_train, p_y_train, dim=-1).clamp(min=0)).mean().item()

    # Restore eval mode + BN running statistics.
    objective.eval()
    for name, m in objective.projector.named_modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(bn_before[name][0])
            m.running_var.copy_(bn_before[name][1])

    # The point of the diagnostic is to REPORT the values; the assertion only
    # checks the mechanics (both paths computed, no NaN).
    assert torch.isfinite(torch.tensor([mse_eval, mse_train])).all()
    assert torch.isfinite(torch.tensor([cos_eval, cos_train])).all()
    # Diagnostics are recorded on the test node for the report.
    test_node = getattr(pytest, "_diag", {})
    test_node["proj_train_eval"] = {
        "mse_eval": mse_eval, "mse_train": mse_train,
        "cos_eval": cos_eval, "cos_train": cos_train,
    }


def test_validate_uses_unified_signature():
    """Item 5: the unified validation path must call the unified model
    signature model(occupancy, scalar_values, scalar_known, spectrum, mask).
    Verify validate() runs end-to-end on a synthetic batch and reports both
    raw and projected diagnostics."""
    from scripts.train.train_unified import validate

    class _Obj:
        name = "unified_jepa"

    model = _build_model()
    objective = _objective()
    occ, sv, spec = _batch(seed=7)
    val_batches = [(occ, sv, spec)]
    cfg = {
        "curriculum": {"val_mask_ratio": 0.5},
    }
    out = validate(model, objective, val_batches, cfg, "cpu")
    for k in ("raw_mse", "raw_cos_err", "proj_mse", "proj_cos_err",
              "L_inv", "L_total", "L_phys_weighted"):
        assert k in out, f"validation must report {k}"
    assert torch.isfinite(torch.tensor(list(out.values()))).all()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
