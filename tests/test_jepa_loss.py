"""Unit tests for the JEPA loss target stop-gradient contract (audit item 1).

The shared projection head is applied to both ẑ and z_y before the cosine (§4).
The target-side projection must run under torch.no_grad(): gradients flow ONLY
through ẑ (soft-JEPA target contract). Previously proj(target) had no no_grad
guard — a grad-carrying target (any caller not already detached, e.g. a future
module) leaked its gradient into the shared head.

Run:  python tests/test_jepa_loss.py        (also collectable by pytest)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from losses.jepa_loss import ProjectionMLP, jepa_loss  # noqa: E402


def _setup(hidden=8, seed=0):
    torch.manual_seed(seed)
    proj = ProjectionMLP(hidden=hidden)
    pred = torch.randn(2, 4, hidden, requires_grad=True)
    target = torch.randn(2, 4, hidden, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [False, True, True, True]])
    return proj, pred, target, mask


def _pred_only_grad(proj, pred, target, mask):
    """Gradient of the shared head when the target path is fully detached (the
    reference behavior the stop-gradient contract must match)."""
    loss, _ = jepa_loss(pred, target.detach(), mask, proj=proj)
    loss.backward()
    g = proj.net[0].weight.grad.clone()
    proj.zero_grad()
    return g


def test_target_projection_receives_no_gradient_by_default():
    """With the default stop_grad_target=True, a grad-carrying target must not
    contribute to the projection head's gradient (== pred-only reference)."""
    proj, pred, target, mask = _setup()
    loss, _ = jepa_loss(pred, target, mask, proj=proj)   # target NOT detached by caller
    loss.backward()
    g_with_target = proj.net[0].weight.grad.clone()
    proj.zero_grad()

    g_pred_only = _pred_only_grad(proj, pred, target, mask)
    assert torch.allclose(g_with_target, g_pred_only, atol=1e-6), (
        "gradient flowed into the shared projection via the target path: "
        f"max diff {((g_with_target - g_pred_only).abs().max().item()):.2e}")


def test_ema_per_sample_loss_matches_stopgrad_baseline():
    """Per-sample masked losses identical whether the caller detaches or relies on
    the internal no_grad — the fix must be numerically invisible in the forward."""
    proj, pred, target, mask = _setup(seed=1)
    l_detached, per_detached = jepa_loss(pred, target.detach(), mask, proj=proj)
    l_guarded, per_guarded = jepa_loss(pred, target, mask, proj=proj)
    assert abs(l_detached.item() - l_guarded.item()) < 1e-12
    assert torch.allclose(per_detached, per_guarded, atol=1e-12)


def test_lejepa_opt_out_flag_still_allows_target_gradient():
    """LeJEPA (student-as-target, SIGReg-stabilized, no stop-grad by design) passes
    stop_grad_target=False: with the flag off, the target path MUST contribute to
    the head's gradient (proving the guard exists but is opt-out)."""
    proj, pred, target, mask = _setup(seed=2)
    loss, _ = jepa_loss(pred, target, mask, proj=proj, stop_grad_target=False)
    loss.backward()
    g_lejepa = proj.net[0].weight.grad.clone()
    proj.zero_grad()
    g_pred_only = _pred_only_grad(proj, pred, target, mask)
    assert not torch.allclose(g_lejepa, g_pred_only, atol=1e-6), (
        "stop_grad_target=False should leave the target path active")


def test_no_projection_when_proj_is_none():
    proj, pred, target, mask = _setup()
    l, per = jepa_loss(pred, target, mask, proj=None)
    assert l.ndim == 0 or l.numel() == 1
    assert per.shape == (2,)


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