"""Kaggle kernel: publication-grade full-epoch baseline for the CURRENT model.

70,000 optimizer steps x training seeds {0,1,2}, CURRENT work-192d config (Phase-2
mask curriculum + Phase-4 multi-target + Phase-1/3/4/5 absent) UNCHANGED except
train.seed. Evaluates each final checkpoint with the existing benchmark protocol.
Writes per-seed JSONs under /kaggle/working/fe/.
"""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "c35196f"          # current work-192d tip
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/fe")
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

run("python scripts/train/train_unified.py --config configs/unified.yaml "
    "--device cuda --preflight", cwd=REPO)

base = yaml.safe_load(open(REPO / "configs" / "unified.yaml"))
codes = {}


def battery(name, cmd, json_out):
    lg = OUT / f"{name}.log"
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"\n=== {name} (exit {r.returncode}) ===\n{r.stdout[-1200:]}", flush=True)
    if r.stderr:
        print(f"--- stderr ---\n{r.stderr[-800:]}", flush=True)
    lg.write_text(r.stdout)
    return r.returncode if Path(json_out).exists() else -1


for seed in SEEDS:
    cfg = copy.deepcopy(base)
    cfg["train"]["seed"] = int(seed)          # ONLY change: the training seed
    cfg_path = REPO / "configs" / f"unified_fe_seed{seed}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    tr = subprocess.run(
        f"python scripts/train/train_unified.py --config configs/unified_fe_seed{seed}.yaml "
        f"--device cuda --max-steps {STEPS}", shell=True, cwd=REPO,
        capture_output=True, text=True)
    (OUT / f"seed{seed}_train.log").write_text(tr.stdout + "\n===STDERR===\n" + tr.stderr)
    print(f"\n[train seed {seed}] exit {tr.returncode}\n{tr.stdout[-800:]}", flush=True)
    if tr.stderr:
        print(f"--- train stderr ---\n{tr.stderr[-800:]}", flush=True)
    codes[f"train_seed{seed}"] = tr.returncode

    ckpt = OUT / f"seed{seed}.pt"
    shutil.copy(str(REPO / "checkpoints" / "unified" / "latest.pt"), str(ckpt))

    codes[f"bench_seed{seed}"] = battery(
        f"seed{seed}_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml "
        f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 4 "
        f"--nn-samples 0 --device cuda --out {OUT}/seed{seed}_scenarioA.json",
        f"{OUT}/seed{seed}_scenarioA.json")
    codes[f"eval_seed{seed}"] = battery(
        f"seed{seed}_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config configs/unified.yaml "
        f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
        f"--out {OUT}/seed{seed}_eval_scenarios.json",
        f"{OUT}/seed{seed}_eval_scenarios.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nFE_CURRENT DONE", codes, flush=True)
