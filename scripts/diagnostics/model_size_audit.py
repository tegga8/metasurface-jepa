"""Param / step-cost audit for unified-JEPA size variants (Phase 6 instrument).

Operator decision 2026-10-04: the model-size scaling study needs a per-module
parameter and relative step-cost audit before any GPU run (`ROADMAP.md` Phase 6:
"Per-module param/FLOP audit + width×depth grid scored on the benchmark").

Behaviour-neutral: builds each config, counts parameters per module, and times
one CPU forward+backward step on a synthetic batch (no training, no dataset —
only the released spectrum-encoder weights are needed).

Scope note: the objective's projector and scalar-summary read-out live in the
training engine, not in the model (~0.15 M at hidden 192, same structure across
variants) — excluded from these counts.

Run:
    python scripts/diagnostics/model_size_audit.py --device cpu
    python scripts/diagnostics/model_size_audit.py \
        --configs configs/unified.yaml configs/scaling/unified_s1_small.yaml \
        --no-timing --out checkpoints/benchmark/model_size_audit.json
"""

import argparse
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
EVAL_DIR = os.path.join(REPO_ROOT, "scripts", "eval")
for _p in (REPO_ROOT, SRC_DIR, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import yaml

from assembly import build_unified_model
import eval_scenarios as es


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def module_param_counts(model):
    """Per-top-level-child (total, trainable) parameter counts + sums."""
    counts = []
    for name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        counts.append((name, int(total), int(trainable)))
    total = sum(c[1] for c in counts)
    trainable = sum(c[2] for c in counts)
    return counts, total, trainable


def time_step(model, batch_size, device, reps, warmup=1):
    """Seconds per synthetic forward+backward step (CPU-relative cost probe)."""
    occ, sv, spec = es._make_synthetic_batch(batch_size, device)
    mask = torch.ones(batch_size, 16, 16, device=device)
    mask[:, :4, :4] = 0.0
    sk = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
    model.train()
    times = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        out = model(occ, sv, sk, spec, mask, goal_mode="real", with_target=True)
        loss = out["z_hat"].pow(2).mean() + out["scalar_pred"].pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return sum(times) / len(times)


def audit_config(label, cfg_path, device, batch_size, reps, timing):
    """Build one config and return (entry dict, model)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    spec_weights = _resolve(cfg["weights"]["spectrum"])
    if not os.path.exists(spec_weights):
        raise FileNotFoundError(f"released spectrum encoder missing: {spec_weights}")
    model = build_unified_model(cfg, spec_weights, device=device)
    counts, total, trainable = module_param_counts(model)
    try:
        cfg_rel = os.path.relpath(cfg_path, REPO_ROOT)
    except ValueError:  # e.g. configs on another Windows drive
        cfg_rel = cfg_path
    entry = {
        "label": label,
        "config": cfg_rel,
        "params": {name: {"total": t, "trainable": tr} for name, t, tr in counts},
        "total": total,
        "trainable": trainable,
        "seconds_per_step": time_step(model, batch_size, device, reps)
        if timing else None,
    }
    return entry, model


def format_table(results):
    lines = [f"{'config':<26}{'total':>12}{'trainable':>12}{'s/step':>9}{'ratio':>7}"]
    for r in results:
        s = r.get("seconds_per_step")
        ratio = r.get("ratio_vs_base")
        s_str = f"{s:.3f}" if s else "-"
        r_str = f"{ratio:.2f}" if ratio else "-"
        lines.append(f"{r['label']:<26}{r['total']:>12,}"
                     f"{r['trainable']:>12,}{s_str:>9}{r_str:>7}")
    return "\n".join(lines)


def format_modules(entry):
    lines = [f"  {entry['label']}:"]
    for name, d in entry["params"].items():
        lines.append(f"    {name:<20}{d['total']:>12,}{d['trainable']:>12,}")
    return "\n".join(lines)


def default_configs():
    paths = [os.path.join(REPO_ROOT, "configs", "unified.yaml")]
    paths += sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "scaling", "*.yaml")))
    return paths


def run_audit(config_paths, device, batch_size, reps, timing):
    results = []
    for path in config_paths:
        label = os.path.splitext(os.path.basename(path))[0]
        entry, _ = audit_config(label, _resolve(path), device, batch_size, reps,
                                timing)
        results.append(entry)
    base = next((r for r in results if r["label"] == "unified"), results[0])
    if timing and base.get("seconds_per_step"):
        for r in results:
            if r.get("seconds_per_step"):
                r["ratio_vs_base"] = r["seconds_per_step"] / base["seconds_per_step"]
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Per-module parameters + relative step cost for unified-JEPA "
                    "size variants (behaviour-neutral audit).")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="config paths (default: configs/unified.yaml + "
                         "configs/scaling/*.yaml)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3,
                    help="timed steps after a warmup step")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    config_paths = args.configs or default_configs()
    results = run_audit(config_paths, torch.device(args.device), args.batch_size,
                        args.reps, timing=not args.no_timing)
    print("=== model-size audit ===")
    print(format_table(results))
    print()
    for entry in results:
        print(format_modules(entry))
    if args.out:
        out = _resolve(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        payload = {
            "meta": {"device": args.device, "batch_size": args.batch_size,
                     "reps": args.reps, "timing": not args.no_timing},
            "results": results,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
