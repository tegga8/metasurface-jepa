"""Kaggle kernel: Phase-4 multi-target training run + canonical battery.

Trains the unified JEPA for 10000 steps with the Phase-4 multi-target objective
enabled (configs/unified.yaml: lambda_cond=1 spectrum-conditioned geometry target,
lambda_scal_t=1 scalar-latent target), then runs the canonical battery. The
Phase-2 run at the same 10k schedule (no multi-target) is the like-for-like
control. Writes real JSON under /kaggle/working/phase4/.
"""

import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "979f08856b2a557098cbe16c64fa70e7320fe85d"   # Phase 4 fix + film std 0.1   # Phase 4 (fixed)
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/phase4")
CFG = "configs/unified.yaml"


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

# --- mandatory preflight (real data, end-to-end) ---
run(f"python scripts/train/train_unified.py --config {CFG} --device cuda --preflight",
    cwd=REPO)

# --- Phase-4 training (config total_steps = 10000) ---
r = subprocess.run(
    f"python scripts/train/train_unified.py --config {CFG} --device cuda",
    shell=True, cwd=REPO, capture_output=True, text=True)
(OUT / "phase4_train.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
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
    "phase4_scenarioA",
    f"python scripts/benchmark/benchmark_metadit.py --config {CFG} "
    f"--checkpoint {CKPT} --split test --scenario A --samples 0 --candidates 4 "
    f"--nn-samples 512 --nn-pools 512,5000,20000 --device cuda "
    f"--out {OUT}/phase4_scenarioA.json",
    json_out=f"{OUT}/phase4_scenarioA.json")
codes["eval_scenarios"] = battery(
    "phase4_eval_scenarios",
    f"python scripts/eval/eval_scenarios.py --config {CFG} "
    f"--checkpoint {CKPT} --scenario all --samples 512 --device cuda "
    f"--out {OUT}/phase4_eval_scenarios.json",
    json_out=f"{OUT}/phase4_eval_scenarios.json")
codes["masked_fill"] = battery(
    "phase4_masked_fill",
    f"python scripts/diagnostics/masked_fill_check.py --config {CFG} "
    f"--checkpoint {CKPT} --samples 32 --device cuda "
    f"--out {OUT}/phase4_masked_fill.json",
    json_out=f"{OUT}/phase4_masked_fill.json")
codes["guidance_gap"] = battery(
    "phase4_guidance_gap",
    f"python scripts/diagnostics/run_guidance_gap_sweep.py --config {CFG} "
    f"--checkpoint {CKPT} --device cuda")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nPHASE4 DONE", codes, flush=True)
