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
                    epoch=2, micro_step=0, is_epoch_end=True,
                    metrics={"cos_err_r0.5": 0.1},
                    health={"status": "HEALTHY"}, ema_state=ema1,
                    best_prediction={"primary": 0.1, "metrics": {"cos_err_r0.5": 0.1}, "step": 7, "health": {"status": "HEALTHY"}},
                    best_healthy_prediction={},
                    masker_rng_state=None,
                    device="cpu", artifact_type="full",
    )
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
                    epoch=0, micro_step=0, is_epoch_end=True,
                    ema_state=collect_ema_state(m1),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None,
                    device="cpu", artifact_type="full")

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
                    epoch=0, micro_step=0, is_epoch_end=True,
                    ema_state=collect_ema_state(m1),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None,
                    device="cpu", artifact_type="full")

    # Different-shaped model -> optimizer param fingerprint diverges -> refuse.
    m2, obj2, opt2, sched2 = _model_and_opt(seed=3, in_feat=8, out_feat=8)
    with pytest.raises(RuntimeError, match="fingerprint|shape|owner"):
        load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")


def test_optimizer_ownership_checked_before_state_restoration(tmp_path):
    """The optimizer ownership check must run BEFORE any checkpoint state is
    restored into the model/objective/optimizer. A mismatched optimizer must
    raise while the live model/objective/optimizer still hold their PRE-LOAD
    (mutated) state — proving the check precedes state mutation."""
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, opt1, sched1 = _model_and_opt(seed=0, in_feat=4, out_feat=4)
    save_checkpoint(path, m1, obj1, opt1, sched1, {}, global_step=1,
                    epoch=0, micro_step=0, is_epoch_end=True,
                    ema_state=collect_ema_state(m1),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None,
                    device="cpu", artifact_type="full")

    # Live objects are deliberately MUTATED (simulating a run that already
    # moved on); a mismatched optimizer must raise BEFORE load_into_model /
    # objective.load_state_dict overwrite them.
    m2, obj2, opt2, sched2 = _model_and_opt(seed=3, in_feat=8, out_feat=8)
    _mutate(m2, obj2)
    w_before = {k: v.clone() for k, v in m2.state_dict().items()}
    os_before = {k: v.clone() for k, v in obj2.state_dict().items()}
    with pytest.raises(RuntimeError, match="fingerprint|shape|owner"):
        load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")
    # Nothing was restored: model + objective keep their pre-load state.
    for k in w_before:
        assert torch.equal(m2.state_dict()[k], w_before[k]), (
            f"model {k} must NOT be restored when ownership check fails first")
    for k in os_before:
        assert torch.equal(obj2.state_dict()[k], os_before[k]), (
            f"objective {k} must NOT be restored when ownership check fails first")


def test_optimizer_ownership_identical_fingerprint_loads(tmp_path):
    """A valid (identical) optimizer fingerprint loads successfully: the
    pre-restoration check must NOT reject an optimizer that owns exactly the
    same parameter shapes as the checkpoint."""
    path = str(tmp_path / "ckpt.pt")
    m1, obj1, opt1, sched1 = _model_and_opt(seed=0, in_feat=4, out_feat=4)
    _advance(opt1, sched1, steps=5)
    os1 = {k: v.clone() for k, v in obj1.state_dict().items()}
    save_checkpoint(path, m1, obj1, opt1, sched1, {}, global_step=7,
                    epoch=0, micro_step=0, is_epoch_end=False,
                    ema_state=collect_ema_state(m1),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None,
                    device="cpu", artifact_type="latest")

    m2, obj2, opt2, sched2 = _model_and_opt(seed=1, in_feat=4, out_feat=4)
    _mutate(m2, obj2)
    load_checkpoint(path, m2, obj2, opt2, sched2, "cpu")
    for k in os1:
        assert torch.equal(obj2.state_dict()[k], os1[k]), (
            f"objective {k} not restored (identical-fingerprint load)")


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


def test_load_checkpoint_ignores_surrogate_keys(tmp_path):
    """Phase-B → Phase-C resume: the unified objective registers the frozen
    MetaDiT surrogate as a submodule, so its state dict has `surrogate.*`
    keys that Phase-B checkpoints do not contain (and must never load, since
    the surrogate is authoritative from data/metadit/weights/). load_checkpoint
    must ignore `surrogate.*` from the checkpoint while loading everything
    else strictly.

    Cases:
    1. checkpoint WITHOUT surrogate.* loads into an objective that HAS a
       surrogate → success, surrogate untouched.
    2. checkpoint WITH fake/different surrogate.* values does NOT overwrite
       the loaded surrogate.
    3. an unrelated missing objective key still raises."""
    from losses.unified_losses import UnifiedJEPALoss

    class _FakeSurrogate(nn.Module):
        def __init__(self, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))

        def forward(self, x):
            return self.net(x)

    def _make(seed=0, surrogate=None):
        torch.manual_seed(seed)
        m = _SmallModel()
        obj = UnifiedJEPALoss(hidden=192, surrogate=surrogate)
        params = [p for p in list(m.parameters()) + list(obj.parameters())
                  if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda s: 0.5 ** (s // 10))
        return m, obj, opt, sched

    def _mutate_params(module):
        with torch.no_grad():
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.1)

    # --- Case 1: Phase-B checkpoint (no surrogate.*) -> Phase-C objective
    # (surrogate attached) must load, keeping the loaded surrogate intact.
    m1, obj1, opt1, sched1 = _make(seed=0, surrogate=None)
    # Pre-mutate the objective so restore is observable.
    _mutate_params(obj1)
    ckpt_no_surrogate = str(tmp_path / "phase_b.pt")
    save_checkpoint(ckpt_no_surrogate, m1, obj1, opt1, sched1, {},
                    global_step=10, epoch=0, micro_step=0, is_epoch_end=False,
                    ema_state=collect_ema_state(m1),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None, device="cpu", artifact_type="latest")
    expected_proj = {k: v.clone() for k, v in obj1.state_dict().items()
                     if not k.startswith("surrogate.")}

    surr = _FakeSurrogate(seed=1)
    surr.eval()
    for p in surr.parameters():
        p.requires_grad_(False)
    m2, obj2, opt2, sched2 = _make(seed=2, surrogate=surr)
    _mutate_params(obj2)                       # mutates projector AND surrogate
    surr_after_mutate = {k: v.clone() for k, v in surr.state_dict().items()}
    load_checkpoint(ckpt_no_surrogate, m2, obj2, opt2, sched2, "cpu")
    for k, v in expected_proj.items():
        assert torch.equal(obj2.state_dict()[k], v), (
            f"objective key {k} not restored from checkpoint")
    for k, v in surr_after_mutate.items():
        assert torch.equal(surr.state_dict()[k], v), (
            f"surrogate key {k} must remain authoritative (not overwritten)")

    # --- Case 2: checkpoint WITH fake surrogate.* values must NOT overwrite
    # the loaded surrogate.
    m3, obj3, opt3, sched3 = _make(seed=3, surrogate=None)
    _mutate_params(obj3)
    ckpt_with_surrogate = str(tmp_path / "phase_c_stale.pt")
    save_checkpoint(ckpt_with_surrogate, m3, obj3, opt3, sched3, {},
                    global_step=11, epoch=0, micro_step=0, is_epoch_end=False,
                    ema_state=collect_ema_state(m3),
                    best_prediction={}, best_healthy_prediction={},
                    masker_rng_state=None, device="cpu", artifact_type="latest")
    # Rewrite the saved objective_state with fake surrogate.* values.
    raw = torch.load(ckpt_with_surrogate, map_location="cpu", weights_only=False)
    fake_surr = _FakeSurrogate(seed=99)
    for k, v in fake_surr.state_dict().items():
        raw["objective_state"][f"surrogate.{k}"] = v
    torch.save(raw, ckpt_with_surrogate)

    surr2 = _FakeSurrogate(seed=5)
    surr2.eval()
    for p in surr2.parameters():
        p.requires_grad_(False)
    surr2_before = {k: v.clone() for k, v in surr2.state_dict().items()}
    m4, obj4, opt4, sched4 = _make(seed=6, surrogate=surr2)
    load_checkpoint(ckpt_with_surrogate, m4, obj4, opt4, sched4, "cpu")
    for k, v in surr2_before.items():
        assert torch.equal(surr2.state_dict()[k], v), (
            f"stale surrogate key {k} from checkpoint must be ignored")

    # --- Case 3: an unrelated missing objective key still raises.
    m5, obj5, opt5, sched5 = _make(seed=7, surrogate=None)
    raw2 = torch.load(ckpt_no_surrogate, map_location="cpu", weights_only=False)
    # Drop a NON-surrogate objective key (projector weight) -> strict load
    # must fail loudly.
    proj_keys = [k for k in raw2["objective_state"] if k.startswith("projector.")]
    assert proj_keys, "expected projector.* keys in objective_state"
    del raw2["objective_state"][proj_keys[0]]
    torch.save(raw2, str(tmp_path / "missing_proj.pt"))
    with pytest.raises(RuntimeError, match="projector"):
        load_checkpoint(str(tmp_path / "missing_proj.pt"),
                        m5, obj5, opt5, sched5, "cpu")


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