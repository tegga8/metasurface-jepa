"""Checkpoint save/load integrity tests (architecture-repair spec §30).

The training loop's resumability contract lives in the engine's
save_checkpoint / load_checkpoint: mandatory metadata (objective_name,
objective_state, optimizer param-shape ownership, scheduler state, EMA momentum
counters, RNG state), strict objective-name match, and loud failure on a
freshly-initialized projector when the checkpoint's objective_state is missing
(spec §12/§30: never silently resume a projector that should have been loaded).

Run:  python tests/test_checkpoint_resume.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import pytest
import torch
import torch.nn as nn

from assembly import load_into_model, saveable_state_dict  # noqa: E402
from losses.objectives import VICRegObjective  # noqa: E402
from train.engine import (  # noqa: E402
    collect_ema_state, load_checkpoint, restore_ema_state, save_checkpoint,
)


class _Ema:
    def __init__(self):
        self.momentum_start = 0.99
        self.momentum_end = 1.0
        self.total_steps = 0

    def set_total_steps(self, n):
        self.total_steps = n

    def update(self, encoder, step):
        self.momentum_start = 0.99
        self.momentum_end = 1.0


class _SmallModel(nn.Module):
    def __init__(self, in_feat=4, out_feat=4):
        super().__init__()
        self.ema = _Ema()
        self.linear = nn.Linear(in_feat, out_feat)

    def forward(self, G, S, M):
        raise NotImplementedError("state round-trip only")


def _objective():
    return VICRegObjective(
        lambda_inv=1.0, lambda_var=1.0, lambda_cov=1.0,
        projector_input_dim=4, projector_hidden_dim=8, projector_output_dim=4)


def _model_and_opt(seed=0, in_feat=4, out_feat=4):
    torch.manual_seed(seed)
    model = _SmallModel(in_feat=in_feat, out_feat=out_feat)
    obj = _objective()
    params = [p for p in list(model.parameters()) + list(obj.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: 0.5 ** (s // 10))
    return model, obj, opt, sched


def _mutate(model, obj):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        for p in obj.parameters():
            p.add_(torch.randn_like(p))


def _advance(opt, sched, steps=3):
    for _ in range(steps):
        opt.step()
        sched.step()


def test_full_roundtrip_restores_everything(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, opt1, sched1 = _model_and_opt(seed=0)
    _advance(opt1, sched1, steps=5)
    w1 = {k: v.clone() for k, v in m1.state_dict().items()}
    os1 = {k: v.clone() for k, v in obj1.state_dict().items()}
    od1 = opt1.state_dict()
    sd1 = sched1.state_dict()
    ema1 = collect_ema_state(m1)
    ema1["total_steps"] = 42

    # Seed a fresh stream, save (captures the post-seed state), then the FIRST
    # draw of that stream is what a restore must replay.
    torch.manual_seed(123)
    save_checkpoint(path, m1, obj1, opt1, sched1, {"lr": 1e-3}, global_step=7,
                    epoch=2, metrics={"cos_err_r0.5": 0.1},
                    health={"status": "HEALTHY"}, ema_state=ema1,
                    best_state={"primary": 0.1, "step": 7})
    expected_next = torch.randn(3).clone()   # first draw of the seeded stream

    torch.manual_seed(999)
    m2, obj2, opt2, sched2 = _model_and_opt(seed=1)
    _mutate(m2, obj2)                        # everything is wrong before load
    loaded = load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")
    restore_ema_state(m2, loaded.get("ema_state"))   # caller-owned step, like the driver

    for k in w1:
        assert torch.equal(m2.state_dict()[k], w1[k]), f"model {k} not restored"
    for k in os1:
        assert torch.equal(obj2.state_dict()[k], os1[k]), f"objective {k} not restored"
    od2, sd2 = opt2.state_dict(), sched2.state_dict()
    assert sd2 == sd1, "scheduler state not restored"
    assert set(od2["state"]) == set(od1["state"])
    assert od2["param_groups"] == od1["param_groups"]
    assert m2.ema.total_steps == 42, "EMA momentum counters not restored"

    # RNG replay: the draw after load equals the draw that would have followed
    # the saved state.
    assert torch.equal(torch.randn(3), expected_next), \
        "RNG state must replay the exact next draw after load"


def _other_objective():
    """A parametric-objective stand-in with a DIFFERENT name (for the strict
    objective-name mismatch gate)."""

    class _Other(nn.Module):
        name = "some_other_objective"

        def forward(self, *a):
            raise NotImplementedError

    return _Other()


def test_strict_objective_name_mismatch_raises(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, opt1, sched1 = _model_and_opt(seed=0)
    save_checkpoint(path, m1, obj1, opt1, sched1, {}, global_step=1,
                    ema_state=collect_ema_state(m1))

    m2 = _SmallModel()
    obj2 = _other_objective()
    opt2 = torch.optim.AdamW(
        [p for p in m2.parameters() if p.requires_grad], lr=1e-3)
    with pytest.raises(RuntimeError, match="objective"):
        load_checkpoint(path, m2, obj2, opt2, None, "cpu")

    # Explicit opt-in with a compatible objective + optimizer loads (documented
    # escape hatch, not the default).
    m3 = _SmallModel()
    obj3 = _objective()
    opt3 = torch.optim.AdamW(
        [p for p in list(m3.parameters()) + list(obj3.parameters())
         if p.requires_grad], lr=1e-3)
    load_checkpoint(path, m3, obj3, opt3, None, "cpu", strict_objective=False)


def test_optimizer_ownership_mismatch_raises(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, opt1, sched1 = _model_and_opt(seed=0, in_feat=4, out_feat=4)
    save_checkpoint(path, m1, obj1, opt1, sched1, {}, global_step=1,
                    ema_state=collect_ema_state(m1))

    # Different-shaped model -> optimizer param fingerprint diverges -> refuse.
    m2, obj2, opt2, sched2 = _model_and_opt(seed=3, in_feat=8, out_feat=8)
    with pytest.raises(RuntimeError, match="fingerprint|shape|owner"):
        load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")


def test_missing_objective_state_fails_loudly(tmp_path):
    """A parametric objective (owns a trainable projector) must NOT silently
    resume with a freshly-initialized projector when objective_state is missing
    (spec §12/§30)."""
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, _, _ = _model_and_opt(seed=0)
    torch.save({"objective_name": "jepa_vicreg",
                "model": saveable_state_dict(m1)}, path)

    m2, obj2, opt2, sched2 = _model_and_opt(seed=4)
    with pytest.raises(RuntimeError, match="objective_state"):
        load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")


def test_collect_restore_ema_state_roundtrip():
    m = _SmallModel()
    s = collect_ema_state(m)
    assert s == {"momentum_start": 0.99, "momentum_end": 1.0, "total_steps": 0}
    s["total_steps"] = 100
    m2 = _SmallModel()
    restore_ema_state(m2, s)
    assert m2.ema.total_steps == 100
    restore_ema_state(m2, None)   # no-op on missing state
    restore_ema_state(m2, {})
    assert m2.ema.total_steps == 100


def test_model_only_best_checkpoint_loadable(tmp_path):
    """train_milestone_b saves `{exp}_{objective}_best_model.pt` as a plain
    model state dict; it must round-trip through the standard helpers."""
    path = str(tmp_path / "best_model.pt")
    m1, _, _, _ = _model_and_opt(seed=0)
    sd = saveable_state_dict(m1)
    torch.save(sd, path)
    m2, _, _, _ = _model_and_opt(seed=1)
    load_into_model(m2, torch.load(path, map_location="cpu", weights_only=False), "cpu")
    for k in sd:
        assert torch.equal(m2.state_dict()[k], sd[k]), f"{k} not restored"


if __name__ == "__main__":
    import pathlib
    import tempfile
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as td:
                        fn(pathlib.Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)