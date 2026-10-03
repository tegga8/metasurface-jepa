"""Kaggle kernel: Phase-2 training run + canonical battery.

Trains the unified JEPA on the **Phase-2 schedule** (configs/unified.yaml:
per-sample mask curriculum that ramps from easy to full masking with P(full)=0.35;
physics held at 0 for `lambda_phys_start_step` then ramped), then runs the
canonical battery on the resulting checkpoint. Writes real JSON under
/kaggle/working/phase2/ and keeps the training stdout (with the per-term
`[grad-share]` lines) as phase2_train.log.
"""

import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "ee23a248cbba51b36373db4ad1e75a05eeb77a10"   # Phase 2b/2c
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/phase2")


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
run("pip install -r requirements.txt -q", cwd=REPO)
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
run("python scripts/train/train_unified.py --config configs/unified.yaml "
    "--device cuda --preflight", cwd=REPO)

# --- Phase-2 training (config total_steps = 10000) ---
train_cmd = "python scripts/train/train_unified.py --config configs/unified.yaml --device cuda"
r = subprocess.run(train_cmd, shell=True, cwd=REPO, capture_output=True, text=True)
(OUT / "phase2_train.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
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
    "phase2_scenarioA",
    f"python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml "
    f"--checkpoint {CKPT} --split test --scenario A --samples 0 --candidates 4 "
    f"--nn-samples 512 --nn-pools 512,5000,20000 --device cuda "
    f"--out {OUT}/phase2_scenarioA.json",
    json_out=f"{OUT}/phase2_scenarioA.json")
codes["eval_scenarios"] = battery(
    "phase2_eval_scenarios",
    f"python scripts/eval/eval_scenarios.py --config configs/unified.yaml "
    f"--checkpoint {CKPT} --scenario all --samples 512 --device cuda "
    f"--out {OUT}/phase2_eval_scenarios.json",
    json_out=f"{OUT}/phase2_eval_scenarios.json")
codes["masked_fill"] = battery(
    "phase2_masked_fill",
    f"python scripts/diagnostics/masked_fill_check.py --config configs/unified.yaml "
    f"--checkpoint {CKPT} --samples 32 --device cuda "
    f"--out {OUT}/phase2_masked_fill.json",
    json_out=f"{OUT}/phase2_masked_fill.json")
codes["guidance_gap"] = battery(
    "phase2_guidance_gap",
    f"python scripts/diagnostics/run_guidance_gap_sweep.py --config configs/unified.yaml "
    f"--checkpoint {CKPT} --device cuda")

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nPHASE2 DONE", codes, flush=True)
