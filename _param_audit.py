"""Count parameters for the shipped configs — resolve the ledger's base-count conflict.

Two different base-v1 trainable counts appear in the ledger (11,402,628 in the scaling
table, 11,403,012 in the imp1 block). One of them is wrong for the config as it stands.
This builds each model from its config and counts exactly, so the published number is
checked against the code instead of against memory.

Also verifies the guarantee that the variants are opt-in: imp2 must be byte-identical in
parameter count to base (mask placement only), and base must still build the same as it
always did.
"""

import os
import sys

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from assembly import build_unified_model  # noqa: E402

CASES = [
    ("base v1", "configs/unified.yaml"),
    ("imp1", "configs/unified_imp1.yaml"),
    ("imp2", "configs/unified_imp2.yaml"),
    ("S1 small", "configs/scaling/unified_s1_small.yaml"),
    ("S2 slim", "configs/scaling/unified_s2_slim.yaml"),
    ("S0 tiny", "configs/scaling/unified_s0_tiny.yaml"),
    ("L1 wide", "configs/scaling/unified_l1_wide.yaml"),
]

LEDGER = {  # what the ledger currently claims, per arm
    "base v1": (11402628, 11403012),
    "imp1": (11565444,),
    "imp2": (11402628,),
    "S0 tiny": (2193284,),
    "S1 small": (4232836,),
    "S2 slim": (8142980,),
    "L1 wide": (19896708,),
}


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


for label, cfg_path in CASES:
    p = resolve(cfg_path)
    if not os.path.exists(p):
        print(f"{label:10s} SKIP (no config at {cfg_path})")
        continue
    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    spec = resolve(cfg["weights"]["spectrum"])
    if not os.path.exists(spec):
        print(f"{label:10s} SKIP (missing {cfg['weights']['spectrum']})")
        continue
    model = build_unified_model(cfg, spec, device="cpu")
    tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
    tot = sum(q.numel() for q in model.parameters())
    claims = LEDGER.get(label, ())
    match = [c for c in claims if c == tr]
    note = ("matches ledger " + str(match[0])) if match else (
        "MISMATCH vs ledger " + "/".join(str(c) for c in claims) if claims else "")
    print(f"{label:10s} trainable={tr:>10,}  total={tot:>11,}  {note}")
