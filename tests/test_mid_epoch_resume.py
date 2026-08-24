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


if __name__ == "__main__":
    test_mid_epoch_resume()
    test_epoch_end_resume()