"""Smoke/debug scenario runner (NOT the scientific evaluator).

The authoritative scientific evaluator is scripts/eval/eval_scenarios.py.
This module exists only for local smoke/debug: it always uses synthetic data
and delegates to eval_scenarios.run_all_scenarios(..., smoke=True).

The only scenario helpers kept here are ScenarioInputs (mask/known-flag
construction) and diversity_metrics — pure data helpers, no duplicate
scenario-execution or gate semantics.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import yaml

from data.mask import BlockMasker


class ScenarioInputs:
    """Construct scenario-specific mask / scalar-known inputs (pure helper).

    Used by smoke tests and by eval_scenarios via the same conventions.
    """

    def __init__(self, occ, sv, spec, mask_ratio, scalar_known, b, device):
        self.occ = occ
        self.sv = sv
        self.spec = spec
        self.mask = BlockMasker(placement="random", grid=16, min_side=3,
                                k_range=(1, 4), seed=100).sample(occ, mask_ratio)
        self.scalar_known = scalar_known
        self.b = b
        self.device = device

    @classmethod
    def scenario_a(cls, occ, sv, spec, b, device, mask_ratio=1.0):
        """Pure inverse design: full mask + all scalars unknown."""
        sk = torch.zeros(b, 3, dtype=torch.bool, device=device)
        return cls(occ, sv, spec, mask_ratio, sk, b, device)

    @classmethod
    def scenario_b(cls, occ, sv, spec, b, device, mask_ratio=0.5):
        """Partial-parameter conditioning: masked occupancy + some scalars known."""
        sk = torch.tensor([[True, False, False],
                           [False, True, False]], dtype=torch.bool,
                          device=device)[:b]
        return cls(occ, sv, spec, mask_ratio, sk, b, device)

    @classmethod
    def scenario_c(cls, occ, sv, spec, b, device, mask_ratio=0.25):
        """Retrofit/constrained completion: mostly known + small masked region."""
        sk = torch.ones(b, 3, dtype=torch.bool, device=device)
        return cls(occ, sv, spec, mask_ratio, sk, b, device)


@torch.no_grad()
def diversity_metrics(generations):
    """Compute diversity and uniqueness of generated geometries.

    Args:
        generations: list of [B, 3, 64, 64] geometry tensors (one per sample)

    Returns:
        dict with pairwise distances and determinism flag.
    """
    if len(generations) <= 1:
        return {"diversity": float("nan"), "n_generations": len(generations),
                "deterministic": True}

    flats = [g.flatten(1) for g in generations]
    flats = torch.cat(flats, dim=0)

    norms = flats / (flats.norm(dim=-1, keepdim=True) + 1e-8)
    sim = norms @ norms.T
    dist = 1 - sim
    n = flats.shape[0]
    tri = dist.triu(diagonal=1)
    nnz = tri[tri > 0]
    if len(nnz) == 0:
        return {"diversity": 0.0, "n_pairs": n * (n - 1) // 2,
                "deterministic": True}

    return {
        "diversity_mean": float(nnz.mean().item()),
        "diversity_std": float(nnz.std().item()),
        "diversity_min": float(nnz.min().item()),
        "n_pairs": n * (n - 1) // 2,
        "deterministic": False,
    }


def main():
    """Smoke-only entry: delegates to the authoritative evaluator in smoke
    mode (synthetic data). Not for scientific conclusions."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Smoke/debug scenario runner (delegates to "
                    "eval_scenarios.py in smoke mode; NOT scientific)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from scripts.eval.eval_scenarios import run_all_scenarios
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    results = run_all_scenarios(cfg, args.checkpoint, args.device, smoke=True)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
