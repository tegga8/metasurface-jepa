"""Kaggle kernel: Phase-2 CONTROL training run + canonical battery.

Same 10k-step length as the Phase-2 run, but the OLD schedule: no mask-ratio
ramp, the old distribution (P(full mask)=0.15), and physics ramped from step 0
over 500 steps. This is the like-for-like baseline the Phase-2b/2c delta is
measured against (the roadmap's Phase-2 gate).

The control config is generated at run time from configs/unified.yaml so no
second config is committed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "16b3c525f5446453da45f7bd4ea8a9948b1a5442"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/control")
CFG = "configs/unified_control.yaml"


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

# --- stage the MetaDiT data ---
data_roots = sorted({
    p.parent.parent for p in find("train_set.mat")
    if (p.parent.parent / "weights" / "surrogate_model.bin").exists()
})
assert len(data_roots) == 1, f"expected one MetaDiT data root, got {data_roots}"
data_root = data_roots[0]
link = REPO / "data" / "metadit"
if link.is_symlink() or link.is_file():
    link.unlink()
elif link.exists():
    shutil.rmtree(link)
link.symlink_to(data_root, target_is_directory=True)
print("data:", data_root, flush=True)

# --- build the CONTROL config (old schedule, same 10k length) ---
with open(REPO / "configs" / "unified.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["curriculum"]["train_mask_ratio_probs"] = [0.25, 0.35, 0.25, 0.15]
cfg["curriculum"].pop("mask_schedule", None)          # no ramp
cfg["staging"]["lambda_phys_start_step"] = 0          # physics from the start
cfg["staging"]["lambda_phys_ramp_steps"] = 500        # the old 500-step ramp
cfg["train"]["total_steps"] = 10000
cfg["train"]["log_grad_share_every_steps"] = 500
with open(REPO / CFG, "w") as f:
    yaml.safe_dump(cfg, f)
print("wrote", CFG, flush=True)

# --- mandatory preflight ---
run(f"python scripts/train/train_unified.py --config {CFG} --device cuda --preflight",
    cwd=REPO)

# --- control training (10k steps, old schedule) ---
r = subprocess.run(
    f"python scripts/train/train_unified.py --config {CFG} --device cuda",
    shell=True, cwd=REPO, capture_output=True, text=True)
(OUT / "control_train.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
print(f"--- training exit {r.returncode} ---", flush=True)
print(r.stdout[-4000:], flush=True)
if r.stderr:
    print("--- train stderr ---\n" + r.stderr[-2000:], flush=True)

CKPT = "checkpoints/unified/latest.pt"
codes = {"train": r.returncode}


def battery(name, cmd, json_out=None):
    lg = OUT / f"{name}.log"
    rr = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"\n=== {name} (exit {rr.returncode}) ===\n{rr.stdout[-4000:]}", flush=True)
    if rr.stderr:
        print(f"--- stderr ---\n{rr.stderr[-2000:]}", flush=True)
    lg.write_text(rr.stdout)
    if json_out and not Path(json_out).exists():
        print(f"[warn] {name} produced no JSON at {json_out}", flush=True)
        return -1
    return rr.returncode


codes["benchmark"] = battery(
    "control_scenarioA",
    f"python scripts/benchmark/benchmark_metadit.py --config {CFG} "
    f"--checkpoint {CKPT} --split test --scenario A --samples 0 --candidates 4 "
    f"--nn-samples 512 --nn-pools 512,5000,20000 --device cuda "
    f"--out {OUT}/control_scenarioA.json",
    json_out=f"{OUT}/control_scenarioA.json")
codes["eval_scenarios"] = battery(
    "control_eval_scenarios",
    f"python scripts/eval/eval_scenarios.py --config {CFG} "
    f"--checkpoint {CKPT} --scenario all --samples 512 --device cuda "
    f"--out {OUT}/control_eval_scenarios.json",
    json_out=f"{OUT}/control_eval_scenarios.json")
codes["masked_fill"] = battery(
    "control_masked_fill",
    f"python scripts/diagnostics/masked_fill_check.py --config {CFG} "
    f"--checkpoint {CKPT} --samples 32 --device cuda "
    f"--out {OUT}/control_masked_fill.json",
    json_out=f"{OUT}/control_masked_fill.json")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nCONTROL DONE", codes, flush=True)
