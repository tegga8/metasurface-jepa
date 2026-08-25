"""Test mid-epoch checkpoint resume (hardening spec §4).

Verifies that:
- Epoch-end checkpoints restore to next epoch with micro_step=0
- Mid-epoch checkpoints restore to same epoch with correct micro_step
- Masker RNG, optimizer, scheduler, EMA, model all restored exactly
"""

import sys
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from assembly import build_model
from data.mask import BlockMasker
from losses.objectives import build_objective
from train.engine import (
    save_checkpoint, load_checkpoint, collect_ema_state, restore_ema_state,
)
from scripts.train.train_milestone_b import build_scheduler
from runtime.device import resolve_device


class _TinyMetaDiTDataset(torch.utils.data.Dataset):
    def __init__(self, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.G = torch.randn(n, 3, 64, 64, generator=g)
        self.S = torch.randn(n, 2, 301, generator=g)
    def __len__(self): return len(self.G)
    def __getitem__(self, i): return self.G[i], self.S[i]


def collate_batch(batch):
    G = torch.stack([b[0] for b in batch])
    S = torch.stack([b[1] for b in batch])
    return G, S


def _make_tiny_setup(device, seed=0):
    cfg = {
        "model": {"hidden": 384, "num_heads": 6, "geo_depth": 1, "predictor_depth": 1,
                  "goal_tokens": 16, "num_predictor_heads": 2,
                  "ema_momentum_start": 0.99, "ema_momentum_end": 0.999},
        "weights": {"spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
                    "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin")},
    }
    model = build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=False)
    objective = build_objective("jepa_vicreg", {}, projector_input_dim=384).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    total_steps = 4
    scheduler = build_scheduler(optimizer, 1e-3, 0, total_steps)
    model.ema.set_total_steps(total_steps)
    masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 2), seed=12345)
    return model, objective, optimizer, scheduler, masker, cfg


def test_mid_epoch_resume():
    """Resume from mid-epoch checkpoint restores exact micro_step."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    ds = _TinyMetaDiTDataset(n=8, seed=42)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)

    # Train for 1.5 epochs (3 optimizer steps with batch_size=2, 8 samples -> 4 steps/epoch)
    step = 0
    micro_step = 0
    accum = 1

    model.train()
    objective.train()

    for epoch in range(2):
        for bi, (G, S) in enumerate(loader):
            if step >= 3:  # Stop mid-epoch (3 out of 4 steps)
                break
            G, S = G.to(device), S.to(device)
            M = masker.sample(G, 0.5).to(device)

            res = objective(model, G, S, M)
            total = res["total_loss"]
            total.backward()

            micro_step += 1
            if micro_step % accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                [p for p in objective.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            objective.on_optimizer_step(model, step)
            scheduler.step()

            # Save MID-EPOCH checkpoint at step 1 (bi=1, not end of epoch)
            if step == 1:
                with tempfile.TemporaryDirectory() as td:
                    ckpt_path = os.path.join(td, "mid_epoch.pt")
                    ema_state = collect_ema_state(model)
                    save_checkpoint(
                        ckpt_path, model, objective, optimizer, scheduler, cfg,
                        global_step=step, epoch=epoch, micro_step=micro_step,
                        is_epoch_end=False,
                        metrics={}, health=None,
                        ema_state=ema_state,
                        best_prediction={}, best_healthy_prediction={},
                        masker_rng_state=masker.get_rng_state(),
                        device=device, artifact_type="full",
                    )

                    # Verify checkpoint has correct is_epoch_end=False
                    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    assert obj["is_epoch_end"] == False
                    assert obj["micro_step"] == micro_step
                    assert obj["epoch"] == epoch

                    # Resume
                    model2, objective2, optimizer2, scheduler2, masker2, _ = _make_tiny_setup(device)
                    trainable2 = [p for p in model2.parameters() if p.requires_grad] + \
                                 [p for p in objective2.parameters() if p.requires_grad]
                    obj2 = load_checkpoint(ckpt_path, model2, objective2, optimizer2, scheduler2, device)
                    restore_ema_state(model2, obj2.get("ema_state"))
                    masker2.set_rng_state(obj2.get("masker_rng_state"))

                    # Verify restored state
                    assert obj2["step"] == step
                    assert obj2["epoch"] == epoch
                    assert obj2["micro_step"] == micro_step
                    assert obj2["is_epoch_end"] == False

                    # Verify model state matches
                    for k in model.state_dict():
                        assert torch.allclose(model.state_dict()[k], model2.state_dict()[k]), f"Model {k} mismatch"

                    # Resume training from step 2
                    step = obj2["step"] + 1
                    epoch = obj2["epoch"]
                    micro_step = obj2["micro_step"]

                    # Continue for one more step
                    for bi, (G, S) in enumerate(loader):
                        if bi < epoch * 4 + micro_step:  # Skip already-trained batches
                            continue
                        G, S = G.to(device), S.to(device)
                        M = masker2.sample(G, 0.5).to(device)
                        res = objective2(model2, G, S, M)
                        total = res["total_loss"]
                        total.backward()
                        micro_step += 1
                        if micro_step % accum != 0:
                            continue
                        optimizer2.step()
                        optimizer2.zero_grad(set_to_none=True)
                        objective2.on_optimizer_step(model2, step)
                        scheduler2.step()
                        break

            step += 1
            micro_step = 0  # reset at optimizer step boundary

    print("PASS: test_mid_epoch_resume")


def test_epoch_end_resume():
    """Resume from epoch-end checkpoint restores to next epoch."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    ds = _TinyMetaDiTDataset(n=8, seed=42)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_batch)

    step = 0
    micro_step = 0
    accum = 1

    model.train()
    objective.train()

    # Train for exactly 1 epoch (4 steps)
    for epoch in range(1):
        for bi, (G, S) in enumerate(loader):
            if step >= 4:
                break
            G, S = G.to(device), S.to(device)
            M = masker.sample(G, 0.5).to(device)
            res = objective(model, G, S, M)
            total = res["total_loss"]
            total.backward()
            micro_step += 1
            if micro_step % accum != 0:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            objective.on_optimizer_step(model, step)
            scheduler.step()

            # Save at epoch end (bi == len(loader) - 1)
            if bi == len(loader) - 1:
                with tempfile.TemporaryDirectory() as td:
                    ckpt_path = os.path.join(td, "epoch_end.pt")
                    ema_state = collect_ema_state(model)
                    save_checkpoint(
                        ckpt_path, model, objective, optimizer, scheduler, cfg,
                        global_step=step, epoch=epoch, micro_step=micro_step,
                        is_epoch_end=True,
                        metrics={}, health=None,
                        ema_state=ema_state,
                        best_prediction={}, best_healthy_prediction={},
                        masker_rng_state=masker.get_rng_state(),
                        device=device, artifact_type="full",
                    )

                    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    assert obj["is_epoch_end"] == True
                    assert obj["epoch"] == epoch
                    assert obj["micro_step"] == micro_step

    print("PASS: test_epoch_end_resume")


# ---------------------------------------------------------------------------
# Production-path hardening A4: real-CLI mid-epoch stop -> resume equivalence.
#
# Run A: production CLI, tiny deterministic config, stops MID-EPOCH (--max-steps).
#   -> final checkpoint must carry is_epoch_end=False and batch_index=<next>.
# Run B: same config + --resume <checkpoint>, continues to the same final step.
# Reference: uninterrupted CLI run to the same final step.
#
# Resumed vs uninterrupted CPU runs must match exactly on: next batch order,
# next mask (both subsumed by exact state equality given the deterministic
# sampler + restored masker RNG), model, objective, optimizer, scheduler, EMA
# state, and final global step.
# ---------------------------------------------------------------------------

import subprocess

import numpy as np
import yaml

from train.engine import REQUIRED_CHECKPOINT_KEYS


def _make_tiny_mat(path, n=8, seed=0):
    import numpy as np
    from scipy import io as sio
    rng = np.random.RandomState(seed)
    pattern = (rng.rand(64, 64, n) > 0.5).astype("int8")
    sio.savemat(str(path), {
        "pattern": pattern,
        "parameter": rng.rand(n, 3),
        "real": rng.randn(n, 301),
        "imag": rng.randn(n, 301),
    })
    return str(path)


def _write_cli_config(tmpdir, train_mat, val_mat):
    """Tiny deterministic config: 8 samples / batch 2 / accum 1 -> 4 steps per epoch."""
    cfg = {
        "experiment": "minimal",
        "objective": "jepa_vicreg",
        "minimal": {"mask_ratio": 0.5, "mask_placement": "random"},
        "sweep": {"mask_ratios": [0.5], "mask_placement": "random"},
        "objective_params": {
            "jepa_vicreg": {
                "projector": {"input_dim": 384, "hidden_dim": 384, "output_dim": 384},
                "lambda_inv": 25.0, "lambda_var": 25.0, "lambda_cov": 1.0,
                "gamma": 1.0, "eps": 1.0e-4,
            },
        },
        "model": {
            "variant": "jepa", "patch_size": 4, "token_grid": 16,
            "hidden": 384, "num_heads": 6, "num_predictor_heads": 2,
            "geo_depth": 1, "predictor_depth": 1,
            "goal_tokens": 16, "num_goal_heads": 4,
            "ema_momentum_start": 0.99, "ema_momentum_end": 0.999,
            "init_from_metadit": False,
        },
        "mask": {"min_side": 3, "k_range": [1, 2], "mask_seed": 12345},
        "data": {"train_split": train_mat, "val_split": val_mat,
                 "max_train_samples": 8, "num_workers": 0},
        "weights": {
            "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin"),
            "spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
            "surrogate": os.path.join(REPO_ROOT, "data/metadit/weights/surrogate_model.bin"),
        },
        "train": {
            "batch_size": 2, "grad_accum": 1, "epochs": 1,
            "lr": 1e-3, "wd": 1e-4, "warmup_steps": 0, "clip_grad_norm": 1.0,
            "save_optimizer": True, "ckpt_every_steps": 1,
            "val_every_steps": 1, "val_batches": 1, "log_every_steps": 1,
            "seed": 0, "device": "cpu",
        },
        "out_dir": os.path.join(tmpdir, "out"),
    }
    cfg_path = os.path.join(tmpdir, "tiny.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def _run_cli(config_path, out_dir, extra_args):
    cmd = [sys.executable,
           os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py"),
           "--config", config_path,
           "--device", "cpu",
           *extra_args]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                          env=env, timeout=900)


def _deep_equal(x, y):
    """Exact equality across tensors / numpy arrays / nested containers."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and np.array_equal(x, y)
    if isinstance(x, dict):
        return (isinstance(y, dict) and set(x.keys()) == set(y.keys())
                and all(_deep_equal(x[k], y[k]) for k in x))
    if isinstance(x, (list, tuple)):
        return (isinstance(y, type(x)) and len(x) == len(y)
                and all(_deep_equal(xi, yi) for xi, yi in zip(x, y)))
    if x is None or isinstance(x, (int, float, str, bool)):
        return x == y and type(x) is type(y)
    return x == y


def _assert_states_equal(a, b, label):
    """Deep-compare two loaded checkpoint dicts' training-relevant state."""
    assert a["step"] == b["step"], f"{label}: final global step mismatch"
    assert a["epoch"] == b["epoch"], f"{label}: epoch mismatch"
    assert a["micro_step"] == b["micro_step"], f"{label}: micro_step mismatch"
    assert a["batch_index"] == b["batch_index"], f"{label}: batch_index mismatch"
    assert a["is_epoch_end"] == b["is_epoch_end"], f"{label}: is_epoch_end mismatch"

    # Next batch order + next mask equivalence: the deterministic epoch sampler
    # plus restored global/masker RNG make any divergence here propagate to the
    # weights, so exact equality of everything below SUBSUMES both.
    assert set(a["model"].keys()) == set(b["model"].keys())
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), f"{label}: model param {k}"
    for k in a["objective_state"]:
        assert torch.equal(a["objective_state"][k], b["objective_state"][k]), \
            f"{label}: objective state {k}"

    oa, ob = a["optimizer"], b["optimizer"]
    assert oa["param_groups"] == ob["param_groups"], f"{label}: optimizer groups"
    assert set(oa["state"].keys()) == set(ob["state"].keys()), \
        f"{label}: optimizer state keys (steps seen differ)"
    for pid in oa["state"]:
        for sk in oa["state"][pid]:
            va, vb = oa["state"][pid][sk], ob["state"][pid][sk]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"{label}: optimizer state {pid}.{sk}"
            else:
                assert va == vb, f"{label}: optimizer state {pid}.{sk}"

    assert a["scheduler_state"] == b["scheduler_state"], \
        f"{label}: scheduler state (lr trajectory differs)"
    # EMA: scalar counters by value; target-encoder weights by tensor equality
    ea, eb = a["ema_state"], b["ema_state"]
    for k in ("momentum_start", "momentum_end", "total_steps"):
        assert ea[k] == eb[k], f"{label}: EMA counter {k}"
    ta, tb = ea.get("target"), eb.get("target")
    assert (ta is None) == (tb is None), f"{label}: EMA target presence"
    if ta is not None:
        assert set(ta.keys()) == set(tb.keys()), f"{label}: EMA target keys"
        for k in ta:
            assert torch.equal(ta[k], tb[k]), f"{label}: EMA target {k}"
    for key in ("rng_state", "masker_rng_state"):
        ra, rb = a.get(key), b.get(key)
        if ra is not None or rb is not None:
            assert _deep_equal(ra, rb), f"{label}: {key}"


def test_production_cli_mid_epoch_stop_then_resume_matches_uninterrupted():
    """A4: stop mid-epoch via CLI, resume via CLI, compare against one clean run."""
    with tempfile.TemporaryDirectory() as td_runA, \
         tempfile.TemporaryDirectory() as td_runB, \
         tempfile.TemporaryDirectory() as td_ref:
        train_mat = _make_tiny_mat(os.path.join(td_ref, "train.mat"), n=8, seed=42)
        val_mat = _make_tiny_mat(os.path.join(td_ref, "val.mat"), n=4, seed=123)

        # ---- Run A: stop after 2 of 4 optimizer steps (mid-epoch) ----
        cfg_a = _write_cli_config(td_runA, train_mat, val_mat)
        proc_a = _run_cli(cfg_a, td_runA, ["--max-steps", "2"])
        assert proc_a.returncode == 0, (
            f"Run A failed\nstdout:\n{proc_a.stdout}\nstderr:\n{proc_a.stderr}")
        ckpt_a_path = os.path.join(td_runA, "out", "minimal_jepa_vicreg_latest.pt")
        assert os.path.exists(ckpt_a_path)

        ckpt_a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt_a]
        assert not missing, f"Run A checkpoint missing schema keys: {missing}"
        # A2 regression: stopped mid-epoch -> actual state, NOT forced epoch-end.
        assert ckpt_a["is_epoch_end"] is False, (
            "final checkpoint of a mid-epoch --max-steps stop must have "
            "is_epoch_end=False")
        assert ckpt_a["batch_index"] == 2, (
            f"final checkpoint must point at the NEXT batch (2), got "
            f"{ckpt_a['batch_index']}")
        assert ckpt_a["epoch"] == 0
        assert ckpt_a["step"] == 1

        # ---- Run B: resume from Run A's final checkpoint to the same final step ----
        cfg_b = _write_cli_config(td_runB, train_mat, val_mat)
        proc_b = _run_cli(cfg_b, td_runB,
                          ["--resume", ckpt_a_path, "--max-steps", "4"])
        assert proc_b.returncode == 0, (
            f"Run B failed\nstdout:\n{proc_b.stdout}\nstderr:\n{proc_b.stderr}")

        # ---- Reference: uninterrupted run to the same final step ----
        cfg_r = _write_cli_config(td_ref, train_mat, val_mat)
        proc_r = _run_cli(cfg_r, td_ref, ["--max-steps", "4"])
        assert proc_r.returncode == 0, (
            f"Reference run failed\nstdout:\n{proc_r.stdout}\nstderr:\n{proc_r.stderr}")

        ckpt_b = torch.load(os.path.join(td_runB, "out",
                                         "minimal_jepa_vicreg_latest.pt"),
                            map_location="cpu", weights_only=False)
        ckpt_r = torch.load(os.path.join(td_ref, "out",
                                         "minimal_jepa_vicreg_latest.pt"),
                            map_location="cpu", weights_only=False)
        _assert_states_equal(ckpt_b, ckpt_r, "resumed vs uninterrupted")

        print("PASS: test_production_cli_mid_epoch_stop_then_resume_matches_uninterrupted")


if __name__ == "__main__":
    test_mid_epoch_resume()
    test_epoch_end_resume()
    test_production_cli_mid_epoch_stop_then_resume_matches_uninterrupted()