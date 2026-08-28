"""Tests for strict real-data / released-weights enforcement (Fixes 5, 6, 16).

Verifies:
- Real mode with missing dataset split → RuntimeError (no silent synthetic).
- Real mode with missing released spectrum weights → RuntimeError.
- Smoke mode (explicit flag) allows synthetic data + dummy spectrum weights.
- Preflight requires real data + released weights.
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


def test_real_mode_missing_data_raises():
    """Fix 5: real training with a missing dataset split must raise, never
    silently fall back to synthetic data."""
    from train_unified import train
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    with pytest.raises(RuntimeError, match="real dataset split"):
        train(cfg, no_train=True, device="cpu", use_synthetic_smoke=False)


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
        def record(self, ratio, regime):
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
    microbatch_pos = loop_body.index("for _ in range(grad_accum):")
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
    def record(self, ratio, regime):
        pass


def test_real_mode_missing_spectrum_weights_raises():
    """Fix 6: real mode with missing released spectrum encoder must raise."""
    from train_unified import _ensure_spectrum_weights
    missing = os.path.join(REPO_ROOT, "data/metadit/weights/NONEXISTENT.pth")
    with pytest.raises(RuntimeError, match="released spectrum encoder"):
        _ensure_spectrum_weights(missing, "cpu", allow_dummy=False)


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
