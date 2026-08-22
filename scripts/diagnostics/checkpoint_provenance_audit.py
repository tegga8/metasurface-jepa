"""Checkpoint provenance audit (validity-fix spec §6-§7, §10 Check A/B).

Prints step / objective / config for every candidate checkpoint so a genuine
trained checkpoint can be distinguished from smoke/near-init state.

Candidates are discovered by globbing checkpoints/milestone_b for *.pt
(excluding the frozen probe weights), so the audit stays valid as smoke
artifacts are cleaned out and genuine trained checkpoints land.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

BASE = "checkpoints/milestone_b"
CANDIDATES = sorted(
    str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    for p in (REPO_ROOT / BASE).rglob("*.pt")
    if p.name != "latent_geometry_probe_weights.pt"
)
if not CANDIDATES:
    print(f"no candidate checkpoints found under {BASE}/ — nothing to audit")
    sys.exit(0)

for rel in CANDIDATES:
    p = REPO_ROOT / rel
    if not p.exists():
        print(f"{rel}\n  MISSING\n")
        continue
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{rel}\n  LOAD ERROR: {e}\n")
        continue
    step = ck.get("step")
    epoch = ck.get("epoch")
    obj = ck.get("objective_name")
    if obj is None:
        obj = (ck.get("adaptive_meta") or {}).get("objective")
    cfg = ck.get("cfg") or {}
    tr = cfg.get("train", {}) or {}
    data = cfg.get("data", {}) or {}
    adapt = cfg.get("adaptive_training") or {}
    exp = cfg.get("experiment")
    seed = tr.get("seed")
    print(rel)
    print(f"  step={step} epoch={epoch} objective={obj} experiment={exp} seed={seed}")
    print(f"  batch_size={tr.get('batch_size')} epochs={tr.get('epochs')} "
          f"max_train_samples={data.get('max_train_samples')} "
          f"max_total_steps={adapt.get('max_total_steps')}")
    best = ck.get("best") or {}
    if best:
        print(f"  best_metric={best.get('primary')} at step {best.get('step')}")
    print()

# smoke heuristic per validity-fix spec §10 Check A
print("--- smoke classification (step <= 10 OR max_total_steps <= 20 OR "
      "max_train_samples <= 64) ---")
for rel in CANDIDATES:
    p = REPO_ROOT / rel
    if not p.exists():
        continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    step = ck.get("step") or 0
    cfg = ck.get("cfg") or {}
    mts = (cfg.get("adaptive_training") or {}).get("max_total_steps") or 10 ** 9
    mtsamp = (cfg.get("data") or {}).get("max_train_samples") or 10 ** 9
    is_smoke = step <= 10 or mts <= 20 or mtsamp <= 64
    print(f"{rel}: is_smoke={is_smoke} (step={step}, max_total_steps={mts}, "
          f"max_train_samples={mtsamp})")
