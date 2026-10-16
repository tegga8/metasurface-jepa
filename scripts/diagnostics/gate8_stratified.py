"""Gate 8 — goal-ignoring collapse, STRATIFIED by masking regime (§8.3).

Why this exists: the validation plan's critical gap. The mandated comparison is
real / null / **shuffled**, stratified by masking regime, gated on the stratum
where occupancy AND scalars are both heavily masked. The existing suite covers
real-vs-null only and never stratifies by mask ratio — and this project already
has direct evidence of exactly that failure (real-vs-null distance ≈1.4 while
the predictor barely used the spectrum), so a pooled or null-only control would
have declared Task 2 a pass.

Strata:
  (a) easy  — mask ratio 0.25, all scalars KNOWN
  (b) hard  — mask ratio 1.00, all scalars UNKNOWN   <- the gate

Reported per stratum and per evaluation seed: the paired per-sample win rate
`real < shuffled`, the paired difference with a bootstrap CI, and the real-vs-
null gap (for continuity, never as evidence). The gate is the HARD stratum's win
rate against `eval.gate_beats_fraction_min` (0.75), per seed and pooled.

Usage:
  python scripts/diagnostics/gate8_stratified.py <config> <ckpt> \
      [--n 512] [--eval-seeds 3 4 5] [--out path.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
sys.path.insert(0, REPO)

import eval_scenarios as es  # type: ignore[import-not-found]  # noqa: E402
import _novel_modeltest as nmt  # noqa: E402
from data.dataset import MetaDiTDataset, collate_batch  # noqa: E402
from data.factorize import factorize_geometry  # noqa: E402
from data.mask import BlockMasker  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def load_batch(cfg, n, device):
    val_path = cfg["data"]["val_split"]
    if not os.path.isabs(val_path):
        val_path = os.path.join(REPO, val_path)
    ds = MetaDiTDataset(val_path, max_samples=n, seed=42)
    loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0,
                        collate_fn=collate_batch)
    G, S = next(iter(loader))
    occ, sv = factorize_geometry(G)
    return occ.to(device), sv.to(device), S.to(device)


def boot_ci(diff, n=5000, seed=0):
    d = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    threshold = float(cfg.get("eval", {}).get("gate_beats_fraction_min", 0.75))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, surrogate = nmt.load_model(args.config, args.ckpt)
    model = model.to(device)
    model.eval()
    surrogate.eval()

    occ, sv, spec = load_batch(cfg, args.n, device)
    print(f"gates 8 | {os.path.basename(args.ckpt)} | n={args.n} | "
          f"device={device} | threshold={threshold}")

    strata = {
        "easy (ratio 0.25, scalars known)": dict(ratio=0.25, known=True),
        "hard (ratio 1.00, scalars unknown)": dict(ratio=1.0, known=False),
    }
    out = {"ckpt": args.ckpt, "n": args.n, "threshold": threshold, "seeds": {}}
    per_stratum = {k: {"real": [], "shuffled": []} for k in strata}

    for seed in args.eval_seeds:
        row = {}
        for name, s in strata.items():
            masker = BlockMasker(placement="random", grid=16, min_side=3,
                                 k_range=(1, 4), seed=int(seed))
            M = masker.sample(occ, ratio=s["ratio"]).to(device)
            known = torch.full((occ.shape[0], 3), bool(s["known"]),
                               dtype=torch.bool, device=device)
            res = es.real_null_shuffled(model, surrogate, occ, sv, spec, M,
                                        device, scalar_known=known,
                                        seed=int(seed),
                                        gate_threshold=threshold)
            gap = res.get("gap", {})
            row[name] = {
                "real": res["real"], "null": res["null"],
                "shuffled": res.get("shuffled"),
                "win_rate": gap.get("real_beats_shuffled_fraction"),
                "real_minus_null": gap.get("real_minus_null"),
                "paired_diff_mean": gap.get("paired_diff_mean"),
            }
            if "per_sample_real" in gap:
                per_stratum[name]["real"] += gap["per_sample_real"]
                per_stratum[name]["shuffled"] += gap["per_sample_shuffled"]
            print(f"  seed {seed} | {name}")
            print(f"    real {res['real']:.4f}  null {res['null']:.4f}  "
                  f"shuffled {res.get('shuffled', float('nan')):.4f}  "
                  f"win-rate {gap.get('real_beats_shuffled_fraction', float('nan')):.4f}")
        out["seeds"][str(seed)] = row

    print("\n=== pooled over seeds ===")
    verdicts = {}
    for name, d in per_stratum.items():
        if not d["real"]:
            continue
        real = np.array(d["real"])
        shuf = np.array(d["shuffled"])
        diff = shuf - real
        mean, lo, hi = boot_ci(diff)
        wins = float((real < shuf).mean())
        verdicts[name] = {"pooled_win_rate": wins, "paired_diff_mean": mean,
                          "ci95": [lo, hi], "n_pooled": int(len(real)),
                          "passes": bool(wins > threshold)}
        print(f"  {name}")
        print(f"    pooled win rate {wins:.4f} (n={len(real)})  "
              f"paired diff {mean:+.4f} CI [{lo:+.4f},{hi:+.4f}]  "
              f"{'PASS' if wins > threshold else 'FAIL'} vs {threshold}")
    out["pooled"] = verdicts
    hard = verdicts.get("hard (ratio 1.00, scalars unknown)")
    print(f"\nGATE 8 (hard stratum): "
          f"{'PASS' if hard and hard['passes'] else 'FAIL'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
