"""Which module accounts for the 2 x hidden discrepancy?

The shipped configs build 2x hidden more trainable parameters than the scaling
table records (base delta 384 at hidden 192, S1 256 at 128, S0 192 at 96, L1 512
at 256). No commit has touched src/ since the scaling study, so the cause is in
the model's module set, not a code change. List every leaf module's trainable
parameter count and flag the ones that could sum to the delta, rather than
guessing at a LayerNorm.
"""

import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from assembly import build_unified_model  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "unified.yaml"),
                          encoding="utf-8"))
spec = os.path.join(REPO, cfg["weights"]["spectrum"])
model = build_unified_model(cfg, spec, device="cpu")
hidden = cfg["hidden"]
delta = 2 * hidden

print(f"hidden={hidden}  target delta={delta}")
print("\ntrainable parameter owners (top-level modules):")
tot = 0
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    tot += n
    print(f"  {name:22s} {n:>11,}")
print(f"  {'TOTAL':22s} {tot:>11,}")

print(f"\nleaf modules whose trainable size is 2*hidden ({delta}):")
for name, mod in model.named_modules():
    n = sum(p.numel() for p in mod.parameters(recurse=False)
            if p.requires_grad)
    if n == delta:
        print(f"  {name:60s} {n:,}  "
              f"({[tuple(p.shape) for p in mod.parameters(recurse=False) if p.requires_grad]})")

print("\nall zero-or-2*hidden-sized trainable leaves near that size:")
for name, mod in model.named_modules():
    n = sum(p.numel() for p in mod.parameters(recurse=False)
            if p.requires_grad)
    if 0 < n <= 4 * hidden and delta * 0.4 <= n <= delta * 2.5:
        print(f"  {name:60s} {n:,}")
