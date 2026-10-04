"""Kaggle kernel: re-evaluate the scalar-film intervention checkpoints (fix).

The Step-3 training succeeded; its battery failed because it evaluated with
`configs/unified.yaml` (scalar_predictor_film=false) against film-enabled checkpoints
(strict load mismatch). This kernel mounts the training kernel's output and runs the
benchmark + scenario battery with scalar_predictor_film=true. No training.
"""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "1c84914"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/sfe")
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
run(f"git clone {REPO_URL} {REPO}")
run(f"git checkout {REPO_REF}", cwd=REPO)
run("pip install -q numpy scipy PyYAML scikit-learn timm einops transformers "
    "matplotlib tqdm safetensors", cwd=REPO, check=False)

data_roots = sorted({
    p.parent.parent for p in find("train_set.mat")
    if (p.parent.parent / "weights" / "surrogate_model.bin").exists()
})
assert len(data_roots) == 1, f"expected one data root, got {data_roots}"
link = REPO / "data" / "metadit"
if link.is_symlink() or link.is_file():
    link.unlink()
elif link.exists():
    shutil.rmtree(link)
link.symlink_to(data_roots[0], target_is_directory=True)

# film-enabled config (the ONLY change vs the shipped config)
cfg = yaml.safe_load(open(REPO / "configs" / "unified.yaml"))
cfg["scalar_predictor_film"] = True
cfg_path = REPO / "configs" / "unified_sf_eval.yaml"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f)

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
    hits = find(f"sf_seed{seed}.pt")
    assert hits, f"sf_seed{seed}.pt not found under /kaggle/input"
    ckpt = hits[0]
    codes[f"bench_seed{seed}"] = battery(
        f"sf_seed{seed}_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config configs/unified_sf_eval.yaml "
        f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 4 "
        f"--nn-samples 0 --device cuda --out {OUT}/sf_seed{seed}_scenarioA.json",
        f"{OUT}/sf_seed{seed}_scenarioA.json")
    codes[f"eval_seed{seed}"] = battery(
        f"sf_seed{seed}_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config configs/unified_sf_eval.yaml "
        f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
        f"--out {OUT}/sf_seed{seed}_eval_scenarios.json",
        f"{OUT}/sf_seed{seed}_eval_scenarios.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nSF_EVAL DONE", codes, flush=True)
