"""Test reference projector freshness (hardening spec §5).

Ensures the healthy reference is projected through the CURRENT objective's projector,
not a stale/random one.
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
from train.engine import (
    build_deterministic_reference, fixed_validation_from_loader, healthy_references,
)
from data.mask import BlockMasker
from assembly import build_model
from losses.objectives import build_objective


class _TinyMetaDiTDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.G = torch.randn(n, 3, 64, 64, generator=g)
        self.S = torch.randn(n, 2, 301, generator=g)
    def __len__(self): return len(self.G)
    def __getitem__(self, i): return self.G[i], self.S[i]


def test_reference_projector_freshness():
    """Healthy reference must use the CURRENT objective's projector."""
    device = torch.device("cpu")
    cfg = {
        "model": {"hidden": 384, "num_heads": 6, "geo_depth": 1, "predictor_depth": 1,
                  "goal_tokens": 16, "num_predictor_heads": 2,
                  "ema_momentum_start": 0.99, "ema_momentum_end": 0.999},
        "weights": {"spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
                    "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin")},
    }

    # Build reference model
    ref_model = build_deterministic_reference(
        lambda: build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=False)
    )
    ref_model.eval()

    # Build objective with a projector that has known initial state
    obj1 = build_objective("jepa_vicreg", {}, projector_input_dim=384).to(device)
    proj1_state = {k: v.clone() for k, v in obj1.projector.state_dict().items()}

    # Build fixed validation
    val_ds = _TinyMetaDiTDataset(n=4, seed=123)
    fv = fixed_validation_from_loader(val_ds, 2, 2, device, ratio=0.5)

    # Get healthy references using obj1
    refs1 = healthy_references(ref_model, fv, objective=obj1)

    # Mutate the objective's projector
    with torch.no_grad():
        for p in obj1.projector.parameters():
            p.add_(torch.randn_like(p) * 10)

    # Get healthy references again using the SAME objective (now mutated)
    refs2 = healthy_references(ref_model, fv, objective=obj1)

    # The projected stats MUST change because the projector changed
    # If they don't change, the reference is using a stale projector
    proj1_stats = refs1["proj"]
    proj2_stats = refs2["proj"]

    # At least one stat should differ (the projector changed)
    # Use token_std as a sensitive measure
    diff = abs(proj1_stats.get("token_std", 0) - proj2_stats.get("token_std", 0))
    assert diff > 1e-4, f"Projected stats did not change after projector mutation (diff={diff}). Stale projector detected!"

    print("PASS: test_reference_projector_freshness")


if __name__ == "__main__":
    test_reference_projector_freshness()