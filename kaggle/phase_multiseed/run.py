"""Kaggle kernel: 3-seed CI phase (best config) + battery per seed.

Trains the current best config (Phase-2 schedule + Phase-4 std-0.1 multi-target)
at three seeds and runs the benchmark + scenario battery on each, so the single-seed
Phase-2/4 gains get confidence intervals. Writes real JSON under /kaggle/working/ms/.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "7088930a3a9f4cfaf2533e46c1371de6bcd257ac"   # NN-scoping + headroom
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/ms")
SEEDS = (0, 1, 2)


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
# Kaggle ships torch; the pinned torch wheels are unavailable on its newer Python.
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

run("python scripts/train/train_unified.py --config configs/unified.yaml "
    "--device cuda --preflight", cwd=REPO)

base = yaml.safe_load(open(REPO / "configs" / "unified.yaml"))
codes = {}


def battery(name, cmd, json_out):
    lg = OUT / f"{name}.log"
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"\n=== {name} (exit {r.returncode}) ===\n{r.stdout[-2500:]}", flush=True)
    if r.stderr:
        print(f"--- stderr ---\n{r.stderr[-1500:]}", flush=True)
    lg.write_text(r.stdout)
    if not Path(json_out).exists():
        print(f"[warn] {name} produced no JSON at {json_out}", flush=True)
        return -1
    return r.returncode


for seed in SEEDS:
    cfg = copy.deepcopy(base)
    cfg["train"]["seed"] = int(seed)
    cfg["train"]["total_steps"] = 10000
    cfg_path = REPO / "configs" / f"unified_seed{seed}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # train
    tr = subprocess.run(
        f"python scripts/train/train_unified.py --config configs/unified_seed{seed}.yaml "
        f"--device cuda", shell=True, cwd=REPO, capture_output=True, text=True)
    (OUT / f"seed{seed}_train.log").write_text(tr.stdout + "\n===STDERR===\n" + tr.stderr)
    print(f"\n[train seed {seed}] exit {tr.returncode}\n{tr.stdout[-1500:]}", flush=True)
    if tr.stderr:
        print(f"--- train stderr ---\n{tr.stderr[-1500:]}", flush=True)
    codes[f"train_seed{seed}"] = tr.returncode

    ckpt = OUT / f"seed{seed}.pt"
    shutil.copy(str(REPO / "checkpoints" / "unified" / "latest.pt"), str(ckpt))

    codes[f"bench_seed{seed}"] = battery(
        f"seed{seed}_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml "
        f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 1 "
        f"--nn-samples 0 --device cuda --out {OUT}/seed{seed}_scenarioA.json",
        f"{OUT}/seed{seed}_scenarioA.json")
    codes[f"eval_seed{seed}"] = battery(
        f"seed{seed}_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config configs/unified.yaml "
        f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
        f"--out {OUT}/seed{seed}_eval_scenarios.json",
        f"{OUT}/seed{seed}_eval_scenarios.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nMULTISEED DONE", codes, flush=True)
