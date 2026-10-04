"""Kaggle kernel: split-ratio 40:40:20 (Phase-6 data-allocation study).

Operator decision 2026-10-04: the full released pool (174,883) is re-split at
40:40:20 (train:val:test) with one permutation (nested prefixes; see
scripts/data/make_split_ratio.py). Trains the shipped config for 1 epoch on the
arm's train split (steps = train // 2 at batch 2), seeds {0,1,2}, and scores
each checkpoint on the arm's own test (which IS the shared 20 % common slice —
no extra run). Scenario gates run on the arm's own val. NOT MetaDiT-comparable.

Writes per-seed JSONs under /kaggle/working/split_40_40_20/.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "fda7236"          # committed tooling + scaling configs + overrides
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/split_40_40_20")
SPLIT_ROOT = Path("/kaggle/working/split_study")
RATIO = "40:40:20"
TAG = "tr40_va40_te20"
SEEDS = (0, 1, 2)
COMMON_FRAC = 0.20


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
root = data_roots[0]
link = REPO / "data" / "metadit"
if link.is_symlink() or link.is_file():
    link.unlink()
elif link.exists():
    shutil.rmtree(link)
link.symlink_to(root, target_is_directory=True)
print("data:", root, flush=True)

run(f"python scripts/data/make_split_ratio.py --data-dir {root}/split_data "
    f"--out-base {SPLIT_ROOT} --ratios {RATIO} --perm-seed 42 "
    f"--common-test-frac {COMMON_FRAC}", cwd=REPO)
arm = SPLIT_ROOT / TAG
man = json.load(open(arm / "manifest.json"))
steps = int(man["counts"]["train"]) // 2
print("arm counts:", man["counts"], "| steps:", steps, flush=True)
assert steps > 0

run("python scripts/train/train_unified.py --config configs/unified.yaml "
    "--device cuda --preflight", cwd=REPO)

base = yaml.safe_load(open(REPO / "configs" / "unified.yaml"))
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
    cfg["data"]["train_split"] = str(arm / "train_set.mat")
    cfg["data"]["val_split"] = str(arm / "val_set.mat")
    cfg["data"]["test_split"] = str(arm / "test_set.mat")
    cfg_rel = f"configs/unified_split_{TAG}_seed{seed}.yaml"
    with open(REPO / cfg_rel, "w") as f:
        yaml.safe_dump(cfg, f)

    tr = subprocess.run(
        f"python scripts/train/train_unified.py --config {cfg_rel} "
        f"--device cuda --max-steps {steps}", shell=True, cwd=REPO,
        capture_output=True, text=True)
    (OUT / f"seed{seed}_train.log").write_text(
        tr.stdout + "\n===STDERR===\n" + tr.stderr)
    print(f"\n[train seed {seed}] exit {tr.returncode}\n{tr.stdout[-600:]}", flush=True)
    codes[f"train_seed{seed}"] = tr.returncode

    ckpt = OUT / f"seed{seed}.pt"
    shutil.copy(str(REPO / "checkpoints" / "unified" / "latest.pt"), str(ckpt))

    codes[f"bench_seed{seed}"] = battery(
        f"seed{seed}_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config {cfg_rel} "
        f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 4 "
        f"--nn-samples 0 --device cuda --out {OUT}/seed{seed}_scenarioA.json",
        f"{OUT}/seed{seed}_scenarioA.json")

    common_file = man["files"].get("common_test")
    if common_file:
        cfg_c = copy.deepcopy(cfg)
        cfg_c["data"]["test_split"] = str(common_file)
        cfg_c_rel = f"configs/unified_split_{TAG}_seed{seed}_common.yaml"
        with open(REPO / cfg_c_rel, "w") as f:
            yaml.safe_dump(cfg_c, f)
        codes[f"bench_common_seed{seed}"] = battery(
            f"seed{seed}_scenarioA_common",
            f"python scripts/benchmark/benchmark_metadit.py --config {cfg_c_rel} "
            f"--checkpoint {ckpt} --split test --scenario A --samples 0 "
            f"--candidates 4 --nn-samples 0 --device cuda "
            f"--out {OUT}/seed{seed}_scenarioA_common.json",
            f"{OUT}/seed{seed}_scenarioA_common.json")

    codes[f"eval_seed{seed}"] = battery(
        f"seed{seed}_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config {cfg_rel} "
        f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
        f"--out {OUT}/seed{seed}_eval_scenarios.json",
        f"{OUT}/seed{seed}_eval_scenarios.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nSPLIT-40-40-20 DONE", codes, flush=True)
