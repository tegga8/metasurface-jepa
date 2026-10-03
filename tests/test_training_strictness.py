"""Tests for strict real-data / released-weights enforcement (Fixes 5, 6, 16).

Verifies:
- Real mode with missing dataset split → RuntimeError (no silent synthetic).
- Real mode with missing released spectrum weights → RuntimeError.
- Smoke mode (explicit flag) allows synthetic data + dummy spectrum weights.
- Preflight requires real data + released weights.
- Frozen surrogate must NOT enter the optimizer (Phase-B/Phase-C fingerprint
  identity).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

import pytest
import yaml
import torch


def _load_cfg():
    with open(os.path.join(REPO_ROOT, "configs", "unified.yaml")) as f:
        return yaml.safe_load(f)


_HAS_REAL_RUN_ASSETS = all(
    os.path.exists(os.path.join(REPO_ROOT, p)) for p in (
        "data/metadit/split_data/train_set.mat",
        "data/metadit/split_data/val_set.mat",
        "data/metadit/weights/spec_encoder.pth",
    )
)


def test_config_rejects_physics_without_ste():
    """Audit B24: lambda_phys > 0 together with physics_use_ste=false is a SILENT
    NO-OP. Measured against the released surrogate (2026-09-13 probe): the
    soft-occupancy path computes a healthy-looking L_phys (18.58) while ZERO
    student parameters receive gradient, because the soft field is ~96% out of
    distribution for the surrogate (spectrum_rel_diff = 0.9599). The combination
    must be rejected rather than warned about — the failure is invisible in the
    loss logs.
    """
    from train_unified import _validate_config

    cfg = _load_cfg()
    cfg["staging"]["phase"] = "C"
    cfg["staging"]["physics_use_ste"] = False
    cfg["loss"]["lambda_phys"] = 0.1
    with pytest.raises(ValueError, match="physics_use_ste"):
        _validate_config(cfg)


def test_config_allows_physics_with_ste():
    """The STE path is the one the probe measured working (360 student params
    with gradient), so the config must remain valid."""
    from train_unified import _validate_config

    cfg = _load_cfg()
    cfg["staging"]["phase"] = "C"
    cfg["staging"]["physics_use_ste"] = True
    cfg["loss"]["lambda_phys"] = 0.1
    _validate_config(cfg)  # must not raise


def test_config_rejects_eval_samples_below_two():
    """Audit B27: the evaluation batch IS the gate's sample size. Below 2 the
    shuffled control has no valid derangement, and the historical value of 2
    (inherited from train.batch_size) was too few to settle anything."""
    from train_unified import _validate_config

    cfg = _load_cfg()
    cfg.setdefault("eval", {})["n_samples"] = 1
    with pytest.raises(ValueError, match="eval\\.n_samples"):
        _validate_config(cfg)


def test_config_accepts_the_shipped_eval_samples():
    """The shipped default must validate, and it must be independent of the
    training batch size (which is 2)."""
    from train_unified import _validate_config

    cfg = _load_cfg()
    assert cfg["eval"]["n_samples"] >= 2
    assert cfg["eval"]["n_samples"] != cfg["train"]["batch_size"], (
        "the evaluation sample size must not be the training batch size — that "
        "is the defect audit B27 fixed")
    _validate_config(cfg)



def test_checkpoint_config_differences_are_detected():
    """ckpt["cfg"] was written into every checkpoint and never read, so editing the
    YAML between save and resume silently diverged the run from its own record.
    The comparison must find nested changes and ignore runtime annotations."""
    from train_unified import _config_differences

    saved = {"loss": {"lambda_phys": 0.1, "lambda_inv": 25.0},
             "train": {"total_steps": 1500}, "_architecture_id": "x"}
    live = {"loss": {"lambda_phys": 3.32, "lambda_inv": 25.0},
            "train": {"total_steps": 70000}, "_architecture_id": "y"}

    diffs = dict((path, (a, b)) for path, a, b in _config_differences(saved, live))
    assert "loss.lambda_phys" in diffs, diffs
    assert diffs["loss.lambda_phys"] == (0.1, 3.32)
    assert "train.total_steps" in diffs
    assert not any(p.startswith("_") for p in diffs), (
        "runtime annotations are not user config and must not be reported")
    assert "loss.lambda_inv" not in diffs, "identical values must not be reported"

    # identical configs produce no differences
    assert _config_differences(saved, dict(saved)) == []


def test_objective_state_excludes_the_frozen_surrogate():
    """The objective registers the released surrogate as a submodule, so a plain
    state_dict() serialises ~25 MB of frozen weights into every checkpoint - while
    the loader deliberately ignores those keys and re-loads the surrogate from disk.
    Measured: 188 MB with the surrogate, 163 MB without."""
    from train.engine import saveable_objective_state

    class _Obj:
        def __init__(self):
            self._sd = {"projector.net.0.weight": 1, "surrogate.blocks.0.weight": 2,
                        "surrogate.head.bias": 3, "scalar_loss.dummy": 4}

        def state_dict(self):
            return self._sd

    sd = saveable_objective_state(_Obj())
    assert "projector.net.0.weight" in sd
    assert "scalar_loss.dummy" in sd
    assert not any(k.startswith("surrogate.") for k in sd), (
        f"surrogate keys must be excluded from the saved objective state: {sorted(sd)}")
    assert saveable_objective_state(None) is None


def test_shipped_gate_threshold_is_three_quarters():
    """Operator decision 2026-09-13: the primary gate demands a 3/4 majority, not a
    bare majority. At 0.5 a model right 51% of the time passes, and the scalar gates
    were doing exactly that - two_known reads 0.5938 on the 32-sample batch and
    0.5106 over the full split with a mean-error difference of 0.0001. At 0.75 both
    scalar strata fail, which is the honest reading.

    Pinned so a silent change to the bar is caught: it decides gate verdicts.
    """
    from train_unified import _validate_config

    cfg = _load_cfg()
    assert cfg["eval"]["gate_beats_fraction_min"] == 0.75
    _validate_config(cfg)

    # and the value is actually honoured by the gate
    from scripts.eval.eval_scenarios import _gate_beats_fraction
    real = torch.tensor([0.1] * 6 + [0.9] * 4)     # wins 6/10
    shuf = torch.tensor([0.5] * 10)
    assert _gate_beats_fraction(real, shuf, 0.75)["gate"] is False, (
        "a 0.6 win rate must not clear a 0.75 bar")
    assert _gate_beats_fraction(real, shuf, 0.5)["gate"] is True

def test_real_mode_missing_data_raises():
    """Fix 5: real training with a missing dataset split must raise, never
    silently fall back to synthetic data."""
    from train_unified import train
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    with pytest.raises(RuntimeError, match="real dataset split"):
        train(cfg, no_train=True, device="cpu", use_synthetic_smoke=False)


def test_trainer_initializes_ema_schedule(tmp_path, monkeypatch):
    """Audit B2: the EMA momentum ramp must be scheduled from the run's total
    steps.

    EMAEncoder defaults to total_steps=1, which saturates current_momentum() at
    its final value (0.999) from step 1 instead of ramping 0.996 -> 0.999 across
    the run — the trainer never called set_total_steps.
    """
    import train_unified
    from train_unified import train

    monkeypatch.setattr(train_unified, "REPO_ROOT", str(tmp_path))
    cfg = _load_cfg()
    report = train(cfg, no_train=True, device="cpu",
                   use_synthetic_smoke=True, max_steps=7)
    assert report.get("ema_total_steps") == 7, (
        "trainer must set the EMA schedule length from the run's total steps; "
        f"got {report.get('ema_total_steps')!r}")


def test_resume_restores_ema_state_and_reports_schedule_change(tmp_path, monkeypatch, capsys):
    """Audit B3: resuming must restore the saved EMA state and must report a
    schedule-length change loudly — never silently re-derive a different ramp.

    Run A: 3 steps (checkpoint step 2). Run B resumes from it with a longer
    schedule (5 steps): the resumed run must announce the change and adopt the
    current run's schedule so the EMA ramp stays consistent with the LR cosine
    schedule rebuilt from config.
    """
    import train_unified
    from train_unified import train

    monkeypatch.setattr(train_unified, "REPO_ROOT", str(tmp_path))
    cfg = _load_cfg()
    r1 = train(cfg, use_synthetic_smoke=True, max_steps=3, device="cpu")
    assert r1["ema_total_steps"] == 3
    ckpt = os.path.join(str(tmp_path), "checkpoints", "unified", "final.pt")
    assert os.path.exists(ckpt), "run A must write a final checkpoint"
    capsys.readouterr()  # drop run A output

    r2 = train(cfg, resume_path=ckpt, use_synthetic_smoke=True, max_steps=5,
               device="cpu")
    out = capsys.readouterr().out
    assert "Resumed at step 3" in out, out
    assert "EMA schedule length changed" in out, (
        "resuming with a different schedule length must be reported loudly, "
        f"not silently rescheduled; captured:\n{out}")
    assert r2["ema_total_steps"] == 5, (
        "the resumed run must adopt THIS run's schedule length (matching the "
        "LR cosine schedule rebuilt from config)")


def test_curriculum_rng_state_round_trips_through_checkpoints(tmp_path, monkeypatch, capsys):
    """Audit B4: the private curriculum generator (mask ratio / scalar regime /
    goal dropout) must be checkpointed and restored — otherwise a resumed run
    silently restarts the sampling stream from the seed and diverges from an
    uninterrupted run (Phase 3 MD §7/§8: mask RNG state is part of the resume
    contract)."""
    import train_unified
    from train_unified import train

    monkeypatch.setattr(train_unified, "REPO_ROOT", str(tmp_path))
    cfg = _load_cfg()
    train(cfg, use_synthetic_smoke=True, max_steps=3, device="cpu")
    ckpt_path = os.path.join(str(tmp_path), "checkpoints", "unified", "final.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "curriculum_rng_state" in ckpt, (
        "checkpoint must carry the curriculum RNG state")
    assert torch.is_tensor(ckpt["curriculum_rng_state"]), (
        f"curriculum_rng_state must be a tensor, got "
        f"{type(ckpt['curriculum_rng_state'])!r}")
    capsys.readouterr()  # drop run A output

    train(cfg, resume_path=ckpt_path, use_synthetic_smoke=True, max_steps=5,
          device="cpu")
    out = capsys.readouterr().out
    assert "curriculum RNG state restored" in out, (
        "the resumed run must restore the checkpointed curriculum RNG state; "
        f"captured:\n{out}")


def test_validate_reports_easy_and_hard_strata(tmp_path, monkeypatch, capsys):
    """Audit B6: validation must report the easy and hard strata separately,
    never as one pooled metric.

    easy: low occupancy mask + all scalars known.
    hard: full occupancy mask + all scalars unknown — the pure-inverse-design
    stratum the design's gates apply to (architecture_v5.md §8.3).
    """
    import json as _json

    import train_unified
    from train_unified import train

    monkeypatch.setattr(train_unified, "REPO_ROOT", str(tmp_path))
    cfg = _load_cfg()
    cfg["train"]["val_every_steps"] = 1
    train(cfg, use_synthetic_smoke=True, max_steps=2, device="cpu")
    out = capsys.readouterr().out
    val_lines = [ln for ln in out.splitlines() if ln.strip().startswith("[val]")]
    assert val_lines, "validation must run with val_every_steps=1"
    metrics = _json.loads(val_lines[-1].split("[val]", 1)[1].strip())
    assert "easy" in metrics and "hard" in metrics, (
        "validation must report easy and hard strata separately (never pooled); "
        f"got keys {sorted(metrics)}")
    assert metrics["hard"]["mask_ratio"] == 1.0, metrics["hard"]
    assert metrics["hard"]["scalars"] == "all_unknown", metrics["hard"]
    assert metrics["easy"]["scalars"] == "all_known", metrics["easy"]
    assert "L_total" in metrics["hard"] and "L_total" in metrics["easy"]


def test_config_validation_rejects_unknown_scalar_regime():
    """Audit B17: an unknown scalar regime must raise instead of being silently
    ignored (only ScalarMasker.REGIMES and the documented 'mixed' alias are
    valid)."""
    from train_unified import _validate_config
    cfg = _load_cfg()
    cfg["curriculum"]["scalar_regimes"] = ["bogus_regime"]
    with pytest.raises(ValueError, match="scalar_regimes"):
        _validate_config(cfg)


def test_config_validation_rejects_incoherent_staging():
    """Audit B17: a no-physics staging phase with lambda_phys > 0 is
    incoherent and must raise."""
    from train_unified import _validate_config
    cfg = _load_cfg()
    cfg["staging"]["phase"] = "B"
    cfg["loss"]["lambda_phys"] = 1.0
    with pytest.raises(ValueError, match="staging"):
        _validate_config(cfg)


def test_shipped_config_validates_cleanly():
    """The shipped config must validate with zero warnings (guards config drift)
    and the documented 'mixed' regime alias must be accepted."""
    from train_unified import _validate_config
    cfg = _load_cfg()
    assert _validate_config(cfg) == []
    cfg["curriculum"]["scalar_regimes"] = ["all_known", "mixed"]
    assert _validate_config(cfg) == []


def test_regime_logger_tracks_achieved_mask_fraction():
    """Audit B18: the requested curriculum ratio is nominal — block masking does
    not achieve it exactly (min-side clamps inflate blocks, independent
    placement overlaps them), so reports must carry the achieved fraction
    alongside the requested one."""
    from data.mask import BlockMasker
    from train_unified import RegimeLogger

    cfg = _load_cfg()
    logger = RegimeLogger(cfg)
    logger.record(0.25, "mixed", achieved_masked_fraction=0.42)
    rep = logger.report()
    assert "mask_fraction_achieved_mean" in rep, rep
    assert abs(rep["mask_fraction_achieved_mean"][0.25] - 0.42) < 1e-9, rep

    # The achieved fraction tracks the requested one after calibration
    # (operator decision 2026-09-13 / audit B20); before calibration this
    # measured ~0.40 for a requested 0.5.
    occ = torch.rand(4, 1, 64, 64)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=7)
    achieved = [float((masker.sample(occ, 0.25) < 0.5).float().mean().item())
                for _ in range(8)]
    mean_achieved = sum(achieved) / len(achieved)
    assert abs(mean_achieved - 0.25) <= 0.03, (
        "requested 0.25 vs achieved — calibrated coverage must track the "
        f"request; got {mean_achieved:.3f}")


def test_ema_gradient_guard_covers_surrogate():
    """Audit B18: the per-step frozen guard must also catch gradients that leak
    into the objective's registered surrogate."""
    import types

    import torch.nn as nn

    from train_unified import _assert_no_ema_gradients

    class _P(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))

    model = types.SimpleNamespace(
        ema=_P(), scalar_mlp_ema=_P(),
        spectrum_path=types.SimpleNamespace(released=None))
    surrogate = _P()
    surrogate.w.grad = torch.ones_like(surrogate.w)
    objective = types.SimpleNamespace(surrogate=surrogate)
    with pytest.raises(RuntimeError, match="surrogate"):
        _assert_no_ema_gradients(model, 3, objective)
    # Without the objective (or with a clean surrogate) the guard passes.
    surrogate.w.grad = None
    _assert_no_ema_gradients(model, 3, objective)


def test_scalar_masker_rng_evolves_across_batches():
    """Fix 3: scalar masking must use PERSISTENT RNG state — two mixed batches
    drawn from the SAME persistent bank must differ (RNG evolves), and the
    RNG state save → generate / restore → generate must be reproducible."""
    from train_unified import (
        _build_scalar_masker_bank, collect_scalar_masker_bank_state,
        restore_scalar_masker_bank_state, sample_scalar_known,
    )
    cfg = _load_cfg()
    cfg["curriculum"]["scalar_regimes"] = ["mixed"]
    cfg["curriculum"]["scalar_regime_probs"] = [1.0]

    # Persistent bank: RNG evolves across batches.
    bank = _build_scalar_masker_bank(cfg, seed=7)
    rng = torch.Generator().manual_seed(0)
    sk1, _ = sample_scalar_known(8, cfg, rng, device="cpu", masker_bank=bank)
    rng = torch.Generator().manual_seed(0)
    sk2, _ = sample_scalar_known(8, cfg, rng, device="cpu", masker_bank=bank)
    # Different masks: the sampler was NOT recreated with the same seed each
    # call (identical masks would indicate the RNG state was reset).
    assert not torch.equal(sk1, sk2), (
        "mixed scalar masks must differ across batches (RNG must evolve)")

    # Checkpoint/restore: save state, generate, restore, generate → identical.
    state = collect_scalar_masker_bank_state(bank)
    rng_a = torch.Generator().manual_seed(1)
    sk_a1, _ = sample_scalar_known(8, cfg, rng_a, device="cpu", masker_bank=bank)
    restore_scalar_masker_bank_state(bank, state)
    rng_b = torch.Generator().manual_seed(1)
    sk_b1, _ = sample_scalar_known(8, cfg, rng_b, device="cpu", masker_bank=bank)
    assert torch.equal(sk_a1, sk_b1), (
        "restoring scalar-masker RNG state must reproduce the next mask")

    # Preserved semantics: all_known / all_unknown unaffected by persistence.
    cfg2 = _load_cfg()
    cfg2["curriculum"]["scalar_regimes"] = ["all_known"]
    cfg2["curriculum"]["scalar_regime_probs"] = [1.0]
    bank2 = _build_scalar_masker_bank(cfg2, seed=0)
    rng2 = torch.Generator().manual_seed(0)
    sk_k, _ = sample_scalar_known(4, cfg2, rng2, device="cpu", masker_bank=bank2)
    assert sk_k.all()
    cfg2["curriculum"]["scalar_regimes"] = ["all_unknown"]
    bank3 = _build_scalar_masker_bank(cfg2, seed=0)
    rng3 = torch.Generator().manual_seed(0)
    sk_u, _ = sample_scalar_known(4, cfg2, rng3, device="cpu", masker_bank=bank3)
    assert not sk_u.any()


def test_half_sensitivity_mask_uses_surrogate_geometry():
    """Fix 2: half_sensitivity masking must receive the COMPLETE [B,3,64,64]
    MetaDiT broadcast geometry (assembled from the TRUE sample), while the
    unified model keeps receiving factorized occupancy [B,1,64,64]."""
    from train_unified import training_step
    from losses.unified_losses import UnifiedJEPALoss
    from assembly import build_unified_model
    from data.mask import BlockMasker
    from runtime.reproducibility import set_seed

    surrogate_path = os.path.join(
        REPO_ROOT, "data/metadit/weights/surrogate_model.bin")
    if not os.path.exists(surrogate_path):
        pytest.skip("surrogate weights not available")
    from physics.physics_loop import load_surrogate

    import tempfile
    tmpdir = tempfile.mkdtemp()
    cfg = _load_cfg()
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    cfg["curriculum"]["train_mask_ratios"] = [0.5]
    cfg["curriculum"]["train_mask_ratio_probs"] = [1.0]
    cfg["curriculum"]["scalar_regimes"] = ["all_known"]
    cfg["curriculum"]["scalar_regime_probs"] = [1.0]
    cfg["curriculum"]["mask_placement"] = "half_sensitivity"
    cfg["train"]["guidance_dropout"] = 0.0
    cfg["loss"]["lambda_phys"] = 0.0

    set_seed(cfg["train"]["seed"])
    device = "cpu"
    from train_unified import _ensure_spectrum_weights, make_synthetic_dataset
    spec_weights = _ensure_spectrum_weights(
        cfg["weights"]["spectrum"], device, allow_dummy=True)
    model = build_unified_model(cfg, spec_weights, device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"],
        lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"],
        lambda_phys=0.0,
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"],
    ).to(device)
    surrogate = load_surrogate(surrogate_path, device=device)
    masker = BlockMasker(
        placement="half_sensitivity", grid=16, min_side=3, k_range=(1, 4),
        seed=cfg["train"].get("seed", 42))
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))

    train_data = make_synthetic_dataset(
        max(cfg["train"].get("batch_size", 2) * 4, 8), device,
        seed=cfg["train"].get("seed", 42))
    occ, sv, spec = train_data[0]
    assert occ.shape == (2, 1, 64, 64), "unified occupancy stays factorized"

    class _Logger:
        def record(self, ratio, regime, achieved_masked_fraction=None):
            pass
    result, M, sk = training_step(
        model, objective, occ, sv, spec, cfg, device, 0, masker, rng,
        _Logger(), surrogate=surrogate)
    assert M.shape == (2, 16, 16), "mask must be [B,16,16]"
    assert torch.isfinite(result["total_loss"]), "forward must be finite"


def test_synthetic_batch_reproducible():
    """Fix 6: synthetic batch seeding must be reproducible (same seed →
    identical scalars + spectrum), not wall-clock-derived."""
    from train_unified import synthetic_batch, make_synthetic_dataset
    b1 = synthetic_batch(2, "cpu", seed=42)
    b2 = synthetic_batch(2, "cpu", seed=42)
    assert torch.equal(b1[1], b2[1]), "scalars must be identical for same seed"
    assert torch.equal(b1[2], b2[2]), "spectrum must be identical for same seed"

    # Different seeds → different data.
    b3 = synthetic_batch(2, "cpu", seed=43)
    assert not torch.equal(b1[1], b3[1]), "different seeds must differ"

    # make_synthetic_dataset is reproducible.
    d1 = make_synthetic_dataset(4, "cpu", seed=1)
    d2 = make_synthetic_dataset(4, "cpu", seed=1)
    for (a1, a2, a3), (c1, c2, c3) in zip(d1, d2):
        assert torch.equal(a2, c2) and torch.equal(a3, c3)


def test_trainer_uses_canonical_scalar_masker():
    """Cleanup item 1: the trainer's scalar-known sampling must delegate to
    the canonical ScalarMasker (src/data/scalar_mask.py), not a duplicate
    inline sampler. Verify output shape, dtype, device, and semantics."""
    from train_unified import sample_scalar_known, _build_scalar_masker_bank
    from data.scalar_mask import ScalarMasker
    cfg = _load_cfg()

    # Shape/dtype/device for each curriculum regime.
    for regime in ("all_known", "all_unknown", "mixed"):
        cfg["curriculum"]["scalar_regimes"] = [regime]
        cfg["curriculum"]["scalar_regime_probs"] = [1.0]
        rng = torch.Generator().manual_seed(42)
        sk, out_regime = sample_scalar_known(4, cfg, rng, device="cpu")
        assert sk.shape == (4, 3), f"{regime}: shape {sk.shape}"
        assert sk.dtype == torch.bool, f"{regime}: dtype {sk.dtype}"
        assert out_regime == regime
        if regime == "all_known":
            assert sk.all()
        elif regime == "all_unknown":
            assert not sk.any()
        else:
            # mixed → canonical "independent" (p=0.5): both True and False
            # must appear somewhere across a large-enough draw.
            rng2 = torch.Generator().manual_seed(1)
            sk_big, _ = sample_scalar_known(200, cfg, rng2, device="cpu")
            assert sk_big.any() and not sk_big.all()

    # The canonical implementation is ScalarMasker — verify the delegation is
    # structural (the bank-builder imports it), so a future duplicate sampler
    # cannot silently reappear.
    import inspect
    src = inspect.getsource(_build_scalar_masker_bank)
    assert "from data.scalar_mask import ScalarMasker" in src, (
        "trainer scalar masking must construct the canonical ScalarMasker "
        "via _build_scalar_masker_bank")


def test_train_mask_ratios_exclude_zero():
    """Cleanup item 3: the training mask-ratio config must exclude 0.0, while
    the eval mask-ratio config may include it as the unmasked reference."""
    cfg = _load_cfg()
    train_ratios = cfg["curriculum"]["train_mask_ratios"]
    train_probs = cfg["curriculum"]["train_mask_ratio_probs"]
    eval_ratios = cfg["curriculum"]["eval_mask_ratios"]

    assert 0.0 not in train_ratios, (
        "training mask ratios must exclude 0.0 (masked-token objective "
        "undefined with no masked tokens)")
    assert len(train_ratios) == len(train_probs), (
        "train_mask_ratios and train_mask_ratio_probs must align")
    assert abs(sum(train_probs) - 1.0) < 1e-6, (
        "train_mask_ratio_probs must sum to 1")
    assert 0.0 in eval_ratios, (
        "eval mask ratios may include 0.0 as the unmasked reference")

    # The training sampler must never return 0.0 even if a legacy config
    # leaks it into the train list (defensive filter).
    from train_unified import sample_mask_ratio
    cfg["curriculum"]["train_mask_ratios"] = [0.0, 0.5, 1.0]
    cfg["curriculum"]["train_mask_ratio_probs"] = [0.1, 0.6, 0.3]
    rng = torch.Generator().manual_seed(0)
    sampled = {float(sample_mask_ratio(cfg, rng)) for _ in range(100)}
    assert 0.0 not in sampled, "sample_mask_ratio must never return 0.0"


def test_optimizer_gradients_reset_each_step():
    """Fix 1 (spec §3): gradients must not leak from one optimizer step into
    the next. Proves the trainer's zero_grad(set_to_none=True) placement by
    checking that a second backward produces the gradient of the second loss
    ALONE, not the sum of first + second losses.

    Two-part verification:
    (a) the train() loop source calls optimizer.zero_grad(set_to_none=True)
        BEFORE the microbatch accumulation loop (the required placement);
    (b) following that exact protocol, the gradient after the second
        training_step backward is the second loss's gradient alone.
    """
    import inspect
    import tempfile
    from train_unified import train, training_step, _ensure_spectrum_weights
    from losses.unified_losses import UnifiedJEPALoss
    from assembly import build_unified_model
    from data.mask import BlockMasker
    from runtime.reproducibility import set_seed

    # (a) Static contract: the training loop must zero_grad at step start,
    # before the microbatch loop (not inside it, not implicitly via the
    # optimizer).
    src = inspect.getsource(train)
    loop_body = src[src.index("for step in range"):]
    zero_grad_pos = loop_body.index("optimizer.zero_grad(set_to_none=True)")
    microbatch_pos = loop_body.index("range(grad_accum):")
    assert zero_grad_pos < microbatch_pos, (
        "train() must zero_grad BEFORE the grad_accum microbatch loop")

    tmpdir = tempfile.mkdtemp()
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["data"]["val_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    cfg["curriculum"]["train_mask_ratios"] = [0.5]
    cfg["curriculum"]["train_mask_ratio_probs"] = [1.0]
    cfg["curriculum"]["scalar_regimes"] = ["all_known"]
    cfg["curriculum"]["scalar_regime_probs"] = [1.0]
    cfg["train"]["guidance_dropout"] = 0.0
    cfg["loss"]["lambda_phys"] = 0.0

    set_seed(cfg["train"]["seed"])
    device = "cpu"
    spec_weights = _ensure_spectrum_weights(
        cfg["weights"]["spectrum"], device, allow_dummy=True)
    model = build_unified_model(cfg, spec_weights, device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"],
        lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"],
        lambda_phys=0.0,
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": cfg["train"]["lr"]},
         {"params": objective.parameters(), "lr": cfg["train"]["lr"]}],
        weight_decay=cfg["train"].get("wd", 1e-4))
    masker = BlockMasker(
        placement="random", grid=16, min_side=3, k_range=(1, 4),
        seed=cfg["train"].get("seed", 42))
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))

    from train_unified import make_synthetic_dataset
    train_data = make_synthetic_dataset(
        max(cfg["train"].get("batch_size", 2) * 4, 8), device)
    occ, sv, spec = train_data[0]

    # (b) Behavioral contract: follow the trainer's protocol (zero_grad at
    # step start, then the training_step backward) and verify the second
    # step's gradient is NOT the first+second sum. No optimizer.step() is
    # called: AdamW does not clear .grad on step, so the residual step-1
    # gradient is what a missing zero_grad would accumulate onto.
    optimizer.zero_grad(set_to_none=True)
    result1, _, _ = training_step(
        model, objective, occ, sv, spec, cfg, device, 0, masker, rng,
        _RegimeLoggerStub(), surrogate=None)
    result1["total_loss"].backward()
    grad_after_step1 = {n: p.grad.detach().clone()
                        for n, p in model.named_parameters()
                        if p.grad is not None}

    # Step 2: zero_grad then backward — the gradient must be step-2's alone.
    optimizer.zero_grad(set_to_none=True)
    result2, _, _ = training_step(
        model, objective, occ, sv, spec, cfg, device, 1, masker, rng,
        _RegimeLoggerStub(), surrogate=None)
    result2["total_loss"].backward()
    grad_after_step2 = {n: p.grad.detach().clone()
                        for n, p in model.named_parameters()
                        if p.grad is not None}

    # If zero_grad were missing, p.grad after step 2 == step1+step2 sum.
    # Only params with a NONZERO step-1 gradient can reveal a leak (a param
    # with zero step-1 gradient trivially satisfies grad2 == grad1+grad2).
    # Use a PURE RELATIVE tolerance (atol=0): the tiny ~1e-9 gradients some
    # norm-bias params legitimately carry would otherwise be swamped by any
    # absolute tolerance; the leak signature is a ~2x inflation, which a
    # relative comparison detects robustly.
    leaked = False
    for n, g1 in grad_after_step1.items():
        if n in grad_after_step2 and g1.abs().sum() > 0:
            sum_grad = g1 + grad_after_step2[n]
            if torch.allclose(grad_after_step2[n], sum_grad, rtol=1e-3, atol=0.0):
                leaked = True
                break
    assert not leaked, (
        "gradients leaked across optimizer steps: second backward gradient "
        "equals the SUM of first+second gradients — zero_grad is missing or "
        "in the wrong scope")


class _RegimeLoggerStub:
    """Minimal RegimeLogger stand-in for training_step (records nothing)."""
    def record(self, ratio, regime, achieved_masked_fraction=None):
        pass


def test_real_mode_missing_spectrum_weights_raises():
    """Fix 6: real mode with missing released spectrum encoder must raise."""
    from train_unified import _ensure_spectrum_weights
    missing = os.path.join(REPO_ROOT, "data/metadit/weights/NONEXISTENT.pth")
    with pytest.raises(RuntimeError, match="released spectrum encoder"):
        _ensure_spectrum_weights(missing, "cpu", allow_dummy=False)


@pytest.mark.skipif(
    not _HAS_REAL_RUN_ASSETS,
    reason="real dataset splits / released spectrum weights not staged locally; "
           "this test exercises the real-mode preflight end to end")
def test_real_mode_missing_surrogate_with_physics_raises():
    """Fix 3 (spec §5): real mode with lambda_phys > 0 and a missing surrogate
    checkpoint must RAISE before training begins — never silently continue
    with a zero placeholder physics term."""
    from train_unified import train
    cfg = _load_cfg()
    cfg["loss"]["lambda_phys"] = 1.0
    cfg["weights"]["surrogate"] = "data/metadit/weights/NONEXISTENT_surrogate.bin"
    with pytest.raises(RuntimeError, match="surrogate checkpoint"):
        train(cfg, no_train=True, device="cpu", use_synthetic_smoke=False)


@pytest.mark.skipif(
    not _HAS_REAL_RUN_ASSETS,
    reason="real dataset splits / released spectrum weights not staged locally; "
           "this test exercises the real-mode preflight end to end")
def test_real_mode_missing_surrogate_without_physics_is_legal():
    """Fix 3 (spec §5): lambda_phys = 0 with a missing surrogate must remain
    legal — no physics loss requested, so no surrogate is needed (and random
    mask placement needs no sensitivity maps)."""
    from train_unified import train
    cfg = _load_cfg()
    cfg["loss"]["lambda_phys"] = 0.0
    cfg["weights"]["surrogate"] = "data/metadit/weights/NONEXISTENT_surrogate.bin"
    # Real mode with real data present; should reach no_train smoke without
    # raising about the surrogate.
    report = train(cfg, no_train=True, device="cpu", use_synthetic_smoke=False)
    assert "final_loss" in report


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available in this environment (code path statically "
           "verified: training_step moves mask and scalar_known to device)")
def test_training_step_cuda_device_correct():
    """Fix 2 (spec §4): the unified forward path must receive occupancy,
    scalars, mask, known flags, and spectrum ALL on the CUDA device. This test
    is CUDA-guarded; on CPU-only machines it is skipped but the code path is
    statically device-correct (training_step constructs sk on the device and
    moves M to it)."""
    import tempfile
    from train_unified import training_step, _ensure_spectrum_weights
    from losses.unified_losses import UnifiedJEPALoss
    from assembly import build_unified_model
    from data.mask import BlockMasker
    from runtime.reproducibility import set_seed

    tmpdir = tempfile.mkdtemp()
    cfg = _load_cfg()
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    cfg["curriculum"]["train_mask_ratios"] = [0.5]
    cfg["curriculum"]["train_mask_ratio_probs"] = [1.0]
    cfg["curriculum"]["scalar_regimes"] = ["mixed"]
    cfg["curriculum"]["scalar_regime_probs"] = [1.0]
    cfg["train"]["guidance_dropout"] = 0.0
    cfg["loss"]["lambda_phys"] = 0.0

    device = "cuda"
    set_seed(cfg["train"]["seed"])
    spec_weights = _ensure_spectrum_weights(
        cfg["weights"]["spectrum"], device, allow_dummy=True)
    model = build_unified_model(cfg, spec_weights, device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"],
        lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"],
        lambda_phys=0.0,
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"],
    ).to(device)
    masker = BlockMasker(
        placement="random", grid=16, min_side=3, k_range=(1, 4),
        seed=cfg["train"].get("seed", 42))
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))
    from train_unified import make_synthetic_dataset
    train_data = make_synthetic_dataset(
        max(cfg["train"].get("batch_size", 2) * 4, 8), device)
    occ, sv, spec = train_data[0]
    assert occ.device.type == "cuda"
    assert sv.device.type == "cuda"
    assert spec.device.type == "cuda"

    result, M, sk = training_step(
        model, objective, occ, sv, spec, cfg, device, 0, masker, rng,
        _RegimeLoggerStub(), surrogate=None)
    assert M.device.type == "cuda", "mask must be on the model device"
    assert sk.device.type == "cuda", "scalar_known must be on the model device"
    assert torch.isfinite(result["total_loss"]), "forward must be finite"
    result["total_loss"].backward()
    assert torch.isfinite(result["total_loss"]), "backward must be finite"


def test_smoke_mode_allows_dummy_spectrum_weights():
    """Fix 6: explicit smoke mode permits creating a dummy spectrum encoder."""
    from train_unified import _ensure_spectrum_weights
    import tempfile
    tmpdir = tempfile.mkdtemp()
    dummy = os.path.join(tmpdir, "dummy_spec.pth")
    path = _ensure_spectrum_weights(dummy, "cpu", allow_dummy=True)
    assert os.path.exists(path)
    os.remove(path)


def test_smoke_mode_synthetic_runs():
    """Fix 16: --use-synthetic-smoke runs with synthetic data (no real paths).

    Uses a temp directory for the dummy spectrum checkpoint so the real
    weights directory is never polluted by the smoke dummy.
    """
    import tempfile
    from train_unified import train
    tmpdir = tempfile.mkdtemp()
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["data"]["val_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    report = train(cfg, no_train=True, device="cpu",
                   use_synthetic_smoke=True)
    assert "final_loss" in report


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, "data/metadit/split_data/train_set.mat")),
    reason="real training split not present")
def test_preflight_requires_real_data():
    """Fix 17: preflight must require the real training split."""
    from train_unified import preflight
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    with pytest.raises(RuntimeError, match="requires the real training split"):
        preflight(cfg, device="cpu")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, "data/metadit/split_data/train_set.mat")),
    reason="real training split not present")
def test_preflight_passes_on_real_data():
    """Fix 17: preflight passes end-to-end on real data (shapes, finite loss,
    gradient ownership)."""
    from train_unified import preflight
    cfg = _load_cfg()
    result = preflight(cfg, device="cpu")
    checks = result["checks"]
    assert checks["occupancy_shape"] == [2, 1, 64, 64]
    assert checks["z_x_shape"] == [2, 256, 192]
    assert checks["z_hat_shape"] == [2, 256, 192]
    assert checks["scalar_pred_shape"] == [2, 3]
    assert checks["assembled_geometry_shape"] == [2, 3, 64, 64]
    assert checks["surrogate_prediction_shape"] == [2, 2, 301]
    assert checks["loss_finite"] is True
    # Cleanup item 7: geometry invariants + scalar precedence must pass.
    assert checks["geometry_invariants_ok"] is True
    assert checks["known_scalar_precedence_ok"] is True
    assert checks["unknown_scalar_precedence_ok"] is True
    own = result["gradient_ownership"]
    assert own["student_params_with_grad"] > 0
    assert own["decoder_params_with_grad"] > 0
    assert own["predictor_params_with_grad"] > 0
    assert own["surrogate_params_with_grad"] == 0
    assert own["ema_params_with_grad"] == 0
    assert own["scalar_mlp_ema_params_with_grad"] == 0
    assert own["released_params_with_grad"] == 0


def test_resume_step_is_next_unrun_step():
    """Fix (resume off-by-one): checkpoint `step` is the LAST COMPLETED
    optimizer step, so resume must continue at step+1. A run that saves a
    checkpoint at step 4 (5 steps, total_steps=5) and resumes with
    total_steps=10 must run steps 5..9 — NOT re-run step 4 (steps 4..9).

    Discriminator: the training loop prints the actual step number each log
    interval; capture stdout and verify the first executed step is 5, not 4.
    (final_step in the report is hardcoded to total_steps-1, so it cannot
    distinguish the two behaviors.)"""
    import contextlib
    import io
    import re
    import tempfile
    from train_unified import train

    def _cfg(tmpdir, total_steps):
        cfg = _load_cfg()
        cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
        cfg["data"]["val_split"] = "data/metadit/split_data/NONEXISTENT.mat"
        cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
        cfg["data"]["use_synthetic"] = True
        cfg["train"]["total_steps"] = total_steps
        cfg["train"]["grad_accum"] = 1
        cfg["train"]["ckpt_every_steps"] = 1000  # no mid-run ckpt writes
        cfg["train"]["log_every_steps"] = 1
        cfg["train"]["val_every_steps"] = 1000
        cfg["train"]["guidance_dropout"] = 0.0
        cfg["curriculum"]["train_mask_ratios"] = [0.5]
        cfg["curriculum"]["train_mask_ratio_probs"] = [1.0]
        cfg["curriculum"]["scalar_regimes"] = ["all_known"]
        cfg["curriculum"]["scalar_regime_probs"] = [1.0]
        cfg["loss"]["lambda_phys"] = 0.0
        return cfg

    def _run(cfg, resume_path=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            train(cfg, resume_path=resume_path, no_train=False,
                  device="cpu", use_synthetic_smoke=True)
        return buf.getvalue()

    def _steps_captured(out):
        # Lines like "step     0  loss=..." -> set of executed step numbers.
        return {int(m) for m in re.findall(r"step\s+(\d+)\s+loss=", out)}

    # Audit B18: never overwrite or delete live checkpoints from a test run.
    existing = [p for p in (
        os.path.join(REPO_ROOT, "checkpoints", "unified", f)
        for f in ("latest.pt", "final.pt")) if os.path.exists(p)]
    if existing:
        pytest.skip("live checkpoints present at checkpoints/unified/ — "
                    "refusing to overwrite or delete them from a test run")
    with tempfile.TemporaryDirectory() as td:
        # Run 1: 5 steps -> final checkpoint at step 4.
        cfg1 = _cfg(td, total_steps=5)
        out1 = _run(cfg1)
        assert 4 in _steps_captured(out1), "run 1 must execute step 4"
        ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "unified",
                                 "final.pt")
        assert os.path.exists(ckpt_path), "resume checkpoint must exist"

        # Run 2: resume from that checkpoint, run to total_steps=10.
        cfg2 = _cfg(td, total_steps=10)
        out2 = _run(cfg2, resume_path=ckpt_path)
        steps2 = _steps_captured(out2)
        assert 5 in steps2, (
            f"resume must start at checkpoint step+1 (step 5); captured "
            f"steps {sorted(steps2)}")
        assert 4 not in steps2, (
            f"resume must NOT re-run the already-completed step 4; captured "
            f"steps {sorted(steps2)}")
        assert 9 in steps2, (
            f"resume must run through the final step 9; captured "
            f"steps {sorted(steps2)}")

        # Clean up the smoke-run checkpoint artifacts.
        for f in ("latest.pt", "final.pt"):
            p = os.path.join(REPO_ROOT, "checkpoints", "unified", f)
            if os.path.exists(p):
                os.remove(p)


def test_frozen_surrogate_not_in_optimizer():
    """Fix (frozen surrogate in optimizer): the objective optimizer group must
    contain ONLY trainable objective parameters. An objective with a frozen
    surrogate attached must produce an optimizer whose fingerprint is
    IDENTICAL to the no-surrogate (Phase-B) case — this is the root cause of
    the Phase-B → Phase-C resume fingerprint mismatch.

    Invariants:
    1. No surrogate parameters in the optimizer.
    2. Optimizer fingerprint identical with and without frozen surrogate.
    3. Surrogate still receives no parameter gradients.
    4. Geometry input still receives a physics gradient (differentiable path)."""
    import torch.nn as nn
    from losses.unified_losses import UnifiedJEPALoss
    from train.engine import _optimizer_param_shapes

    class _FakeSurrogate(nn.Module):
        """Accepts [B,3,64,64] geometry, returns a spectrum-like prediction
        [B,2,301] (enough for the physics-loss path to compute error)."""
        def __init__(self, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(3, 8), nn.Linear(8, 2))

        def forward(self, x):
            # x: [B,3,64,64] → spatial mean [B,3] → net → [B,2] → broadcast
            # to a [B,2,301] spectrum-like prediction.
            b = x.shape[0]
            h = self.net(x.mean(dim=(2, 3)))
            class R:
                pass
            r = R()
            r.prediction = h.unsqueeze(-1).expand(b, 2, 301)
            return r

    def _objective(surrogate=None):
        return UnifiedJEPALoss(hidden=192, lambda_phys=0.0,
                               surrogate=surrogate)

    def _make_optimizer(obj, lr=3e-4, wd=1e-4):
        return torch.optim.AdamW(
            [{"params": [p for p in obj.parameters() if p.requires_grad],
              "lr": lr}],
            weight_decay=wd,
        )

    # Phase B: no surrogate.
    obj_b = _objective(surrogate=None)
    opt_b = _make_optimizer(obj_b)
    fp_b = _optimizer_param_shapes(opt_b)

    # Phase C: frozen surrogate attached.
    surr = _FakeSurrogate(seed=1)
    for p in surr.parameters():
        p.requires_grad_(False)
    obj_c = _objective(surrogate=surr)
    opt_c = _make_optimizer(obj_c)
    fp_c = _optimizer_param_shapes(opt_c)

    # 1. No surrogate params in the optimizer (any group).
    opt_param_ids = {id(p) for g in opt_c.param_groups for p in g["params"]}
    surr_param_ids = {id(p) for p in surr.parameters()}
    assert not (opt_param_ids & surr_param_ids), (
        "frozen surrogate parameters must NOT be in the optimizer")

    # 2. Phase-B and Phase-C fingerprints identical.
    assert fp_b == fp_c, (
        f"optimizer fingerprints must match with/without frozen surrogate: "
        f"{fp_b} vs {fp_c}")

    # Objective trainable group = the objective's trainable parameters, and
    # nothing else. This used to be written as "exactly the projector", which
    # silently assumed the projector was the objective's ONLY trainable module.
    # Door (a) (2026-09-13) added the summary-token read-out to the objective, so
    # the assertion is now stated as the property that actually has to hold: every
    # trainable objective parameter is optimised, and no frozen one is.
    obj_trainable_ids = {id(p) for p in obj_c.parameters() if p.requires_grad}
    obj_group_ids = {id(p) for g in opt_c.param_groups for p in g["params"]}
    assert obj_group_ids == obj_trainable_ids, (
        "objective optimizer group must be exactly the objective's trainable "
        "parameters (projector + summary read-out)")
    assert {id(p) for p in obj_c.projector.parameters()} <= obj_group_ids, (
        "the projector must be optimised")

    # 3. Surrogate still receives no parameter gradients.
    assert all(p.requires_grad is False for p in surr.parameters()), (
        "surrogate parameters must remain requires_grad=False")

    # 4. Geometry input still receives a physics gradient through the
    # differentiable surrogate (no no_grad around the forward).
    from physics.physics_loop import physics_loss_from_out

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    out = {"z_hat": torch.randn(2, 256, 192),
           "scalar_pred": torch.randn(2, 3)}
    surrogate = _FakeSurrogate(seed=2)
    for p in surrogate.parameters():
        p.requires_grad_(False)

    class _ModelStub(nn.Module):
        """decode_geometry stub: return x-shaped geometry from z_hat/scalars,
        ignoring occupancy retention (the physics path's gradient flows into
        the geometry input regardless)."""
        def __init__(self):
            super().__init__()
            self.geo = x

        def decode_geometry(self, z_hat, scalar_pred, occ_input=None,
                            mask=None, use_ste=False, scalar_known=None,
                            scalar_values=None, hard_forward=False):
            return self.geo, self.geo

    occ = torch.rand(2, 1, 64, 64)
    sv = torch.rand(2, 3)
    sk = torch.ones(2, 3, dtype=torch.bool)
    spec = torch.randn(2, 2, 301)
    M = torch.ones(2, 16, 16)
    model_stub = _ModelStub()
    L_phys, _, _ = physics_loss_from_out(
        model_stub, out, surrogate, occ, sv, sk, spec, M,
        loss_type="smooth_l1", use_ste=False, normalize=False)
    assert L_phys.requires_grad, "physics loss must be differentiable"
    L_phys.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, (
        "geometry input must receive a physics gradient (surrogate forward "
        "must not be no_grad)")


def test_surrogate_stays_eval_through_objective_train():
    """Diagnostic-protocol finding: objective.train() recursively put the
    frozen surrogate's BatchNorm2d layers into TRAIN mode, corrupting the
    physics loss (batch-stat BN over batch size 2) and drifting BN running
    stats. The surrogate must remain in eval mode through train/eval cycles,
    and its output for identical geometry must be call-order invariant."""
    import torch.nn as nn
    from losses.unified_losses import UnifiedJEPALoss

    class _BNSurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.conv = nn.Conv2d(3, 2, 1)

        def forward(self, x):
            return type("R", (), {"prediction": self.conv(self.bn(x))})()

    surr = _BNSurrogate().eval()
    for p in surr.parameters():
        p.requires_grad_(False)
    obj = UnifiedJEPALoss(hidden=192, lambda_phys=0.01, surrogate=surr)

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out_before = surr(x).prediction.clone()

    obj.train()
    assert surr.training is False, (
        "objective.train() must leave the frozen surrogate in eval mode")
    assert all(not m.training for m in surr.modules()
               if isinstance(m, nn.BatchNorm2d)), (
        "surrogate BatchNorm layers must stay in eval mode")
    with torch.no_grad():
        out_train_mode = surr(x).prediction.clone()
    assert torch.allclose(out_before, out_train_mode, atol=1e-6), (
        "surrogate output must be call-order invariant (BN must not switch "
        "to batch statistics or drift running stats during training)")

    obj.eval()
    with torch.no_grad():
        out_after = surr(x).prediction.clone()
    assert torch.allclose(out_before, out_after, atol=1e-6)

    # Gradient still flows through the eval-mode surrogate w.r.t. input.
    x2 = x.clone().requires_grad_(True)
    surr(x2).prediction.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0, (
        "eval-mode surrogate must stay differentiable w.r.t. geometry input")


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
