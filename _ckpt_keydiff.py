"""Do the staged checkpoints match the current configs, key by key?

The parameter audit found the shipped configs build 192-512 params more than the
scaling table records (base v1 matches the other ledger figure exactly, so the
scaling table's counts are the stale ones). A count difference is only cosmetic if
the extra parameters are absent from the checkpoint and sit at their init -- but if
they are *trained* parameters that the checkpoint does not carry, evaluating that
checkpoint against this config measures a partly random model, and every number
from it would be suspect.

So diff the state dicts: missing keys (in config, not in checkpoint), unexpected
keys (in checkpoint, not in config), and shape mismatches. Reports per checkpoint.
"""

import os
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from assembly import build_unified_model  # noqa: E402

CASES = [
    ("base v1", "configs/unified.yaml", r"D:\metamaterials_scratch\ckpt\v1base\v1_final.pt"),
    ("S1 small", "configs/scaling/unified_s1_small.yaml", r"D:\metamaterials_scratch\s1_out\s1\seed0.pt"),
    ("S2 slim", "configs/scaling/unified_s2_slim.yaml", r"D:\metamaterials_scratch\s2_out\s2\seed0.pt"),
    ("L1 wide", "configs/scaling/unified_l1_wide.yaml", r"D:\metamaterials_scratch\l1a_ckpt\l1\seed0.pt"),
]


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def state_of(obj):
    for k in ("model", "model_state_dict", "state_dict"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            return obj[k]
    return obj


for label, cfg_path, ck_path in CASES:
    if not os.path.exists(ck_path):
        print(f"{label:10s} SKIP checkpoint missing")
        continue
    cfg = yaml.safe_load(open(resolve(cfg_path), encoding="utf-8"))
    spec = resolve(cfg["weights"]["spectrum"])
    if not os.path.exists(spec):
        print(f"{label:10s} SKIP weights missing")
        continue
    model = build_unified_model(cfg, spec, device="cpu")
    mine = model.state_dict()
    blob = torch.load(ck_path, map_location="cpu", weights_only=False)
    theirs = state_of(blob)
    theirs = {k: v for k, v in theirs.items()
              if isinstance(v, torch.Tensor)}
    missing = [k for k in mine if k not in theirs]
    extra = [k for k in theirs if k not in mine]
    shape = [(k, tuple(mine[k].shape), tuple(theirs[k].shape))
             for k in mine if k in theirs and mine[k].shape != theirs[k].shape]
    n_missing = sum(mine[k].numel() for k in missing)
    print(f"\n{label}: config {len(mine)} keys / ckpt {len(theirs)} keys")
    print(f"  missing from ckpt: {len(missing)} keys, {n_missing:,} params")
    for k in missing[:6]:
        print(f"      {k} {tuple(mine[k].shape)}")
    print(f"  unexpected in ckpt: {len(extra)}"
          + (f" e.g. {extra[:3]}" if extra else ""))
    print(f"  shape mismatches: {len(shape)}"
          + (f" e.g. {shape[:2]}" if shape else ""))
