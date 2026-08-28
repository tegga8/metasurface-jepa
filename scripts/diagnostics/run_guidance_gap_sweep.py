#!/usr/bin/env python
"""Run guidance-gap sweep across mask ratios (§20.3, Phase 4 MD §3.5.1).

Produces the normalized guidance-gap curve:
    ||P(Z_x, A_goal) - P(Z_x, A_∅)|| / sigma(Z_x)

across 20/40/60/80/100% masking buckets. A curve flat near zero, especially
at low mask ratios, indicates Failure Mode 2 (predictor ignores the spectrum).

Usage:
    python scripts/diagnostics/run_guidance_gap_sweep.py \
        --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from data.mask import BlockMasker
from assembly import build_unified_model


def main():
    parser = argparse.ArgumentParser(description="Guidance gap sweep (§20.3)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    spec_path = cfg["weights"].get("spectrum")
    model = build_unified_model(cfg, spec_path, device=device)
    model.eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        from train.engine import load_checkpoint
        load_checkpoint(args.checkpoint, model, None, None, None, device,
                        strict_model=True, strict_objective=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # Synthetic test data
    torch.manual_seed(42)
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[2.5, 0.8, 4.0], [2.8, 1.0, 4.2]]).to(device)
    spec = torch.randn(2, 2, 301).to(device)

    ratios = cfg.get("curriculum", {}).get("mask_ratios",
            [0.2, 0.4, 0.6, 0.8, 1.0])
    if 0.0 in ratios:
        ratios = [r for r in ratios if r > 0.0]

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)

    from diagnostics.guidance_gap import guidance_gap_sweep
    results = guidance_gap_sweep(model, occ, sv, spec, masker, ratios,
                                  device=device)

    print("\n=== Guidance Gap Sweep (§20.3) ===")
    for ratio in sorted(results.keys()):
        print(f"  mask {ratio:.0%}: normalized_gap = {results[ratio]:.6f}")

    print(f"\n{json.dumps(results, indent=2)}")
    print("\nInterpretation:")
    print("  - Gap should increase with mask ratio")
    print("  - Flat/near-zero gap (especially at high mask) = Failure Mode 2")


if __name__ == "__main__":
    main()
