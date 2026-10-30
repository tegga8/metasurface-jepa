"""Kaggle kernel: schedule-fix arm — base config with length-proportional staging.

Operator directive 2026-10-05: "physics ramp is bad — review this and fix a
better schedule." The v2 schedule (resolve_staging_steps, commit e9eb688) keeps
the 10k-era DESIGNED proportions in a 70k run: physics off 0-14k, ramp
14k-35k, fully on at 35k (50 % mark); mask ramp 21k; multi-target 7k-21k.
This arm trains the SHIPPED base config (no architecture change) under the v2
schedule for 70k x seeds {0,1,2} and runs the canonical battery.

Gate: no regression vs the v1 base (MAE 0.0490 +/- 0.0013; gates A/B/C
0.9967 / 0.9772 / 0.9303). v1 runs are NOT directly comparable to v2 arms
(different schedule); this arm carries its own control.

Writes per-seed checkpoints + JSONs under /kaggle/working/schedfix/.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "335c589"          # length-proportional staging (resolve_staging_steps)
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/schedfix")
VARIANT = "configs/unified.yaml"
TAG = "schedfix"
SEEDS = (0, 1, 2)
STEPS = 70000


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find(name, roots=("/kaggle/input",)):
    hits = []
    for r in roots:
        hits += list(Path(r).rglob(name))
    return hits


OUT.mkdir(parents=True, exist_ok=True)

run("nvidia-smi", check=False)
run(f"git clone {REPO_URL} {REPO}")
run(f"git checkout {REPO_REF}", cwd=REPO)
run("pip install -q numpy scipy PyYAML scikit-learn timm einops transformers "
    "matplotlib tqdm safetensors", cwd=REPO, check=False)
run("git rev-parse HEAD", cwd=REPO)

data_roots = sorted({
    p.parent.parent for p in find("train_set.mat")
    if (p.parent.parent / "weights" / "surrogate_model.bin").exists()
})
assert len(data_roots) == 1, f"expected one MetaDiT data root, got {data_roots}"
link = REPO / "data" / "metadit"
if link.is_symlink() or link.is_file():
    link.unlink()
elif link.exists():
    shutil.rmtree(link)
link.symlink_to(data_roots[0], target_is_directory=True)
print("data:", data_roots[0], flush=True)

run(f"python scripts/train/train_unified.py --config {VARIANT} "
    "--device cuda --preflight", cwd=REPO)

base = yaml.safe_load(open(REPO / VARIANT))
codes = {}


def battery(name, cmd, json_out):
    lg = OUT / f"{name}.log"
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"\n=== {name} (exit {r.returncode}) ===\n{r.stdout[-1000:]}", flush=True)
    lg.write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
    return r.returncode if Path(json_out).exists() else -1


for seed in SEEDS:
    cfg = copy.deepcopy(base)
    cfg["train"]["seed"] = int(seed)
    cfg_rel = f"configs/unified_{TAG}_seed{seed}.yaml"
    with open(REPO / cfg_rel, "w") as f:
        yaml.safe_dump(cfg, f)

    tr = subprocess.run(
        f"python scripts/train/train_unified.py --config {cfg_rel} "
        f"--device cuda --max-steps {STEPS}", shell=True, cwd=REPO,
        capture_output=True, text=True)
    (OUT / f"seed{seed}_train.log").write_text(
        tr.stdout + "\n===STDERR===\n" + tr.stderr)
    print(f"\n[train seed {seed}] exit {tr.returncode}\n{tr.stdout[-800:]}", flush=True)
    codes[f"train_seed{seed}"] = tr.returncode

    ckpt = OUT / f"seed{seed}.pt"
    shutil.copy(str(REPO / "checkpoints" / "unified" / "latest.pt"), str(ckpt))

    codes[f"bench_seed{seed}"] = battery(
        f"seed{seed}_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config {cfg_rel} "
        f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 4 "
        f"--nn-samples 0 --device cuda --out {OUT}/seed{seed}_scenarioA.json",
        f"{OUT}/seed{seed}_scenarioA.json")
    codes[f"eval_seed{seed}"] = battery(
        f"seed{seed}_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config {cfg_rel} "
        f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
        f"--out {OUT}/seed{seed}_eval_scenarios.json",
        f"{OUT}/seed{seed}_eval_scenarios.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nSCHED-FIX DONE", codes, flush=True)
