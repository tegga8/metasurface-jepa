"""Kaggle kernel: Phase 1 baseline (measure-only) — unified 192-D JEPA.

No training. Clones the repo at a pinned commit, stages the MetaDiT data and the
trained checkpoint from attached datasets, then runs the baseline battery:
  - scripts/benchmark/benchmark_metadit.py        (MAE / AAE / AAE&K, paper units)
  - scripts/eval/eval_scenarios.py                (A/B/C gates, hard stratum)
  - scripts/diagnostics/masked_fill_check.py      (seam / texture / locality)
  - scripts/diagnostics/run_guidance_gap_sweep.py (guidance gap curve)
Writes each output under /kaggle/working/baseline/<name>.json and an exit-code map.

Pin REPO_REF to the commit that carries the harness (docs/benchmarking + scripts/benchmark).
"""

import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "5d228aca721bdbd240e46d01143221fffacb12f0"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/baseline")
CHECKPOINT_NAME = "full_epoch_final.pt"


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find(name, roots=("/kaggle/input",)):
    hits = []
    for r in roots:
        hits += list(Path(r).rglob(name))
    return hits


run("nvidia-smi", check=False)
run(f"git clone {REPO_URL} {REPO}")
run(f"git checkout {REPO_REF}", cwd=REPO)
run("pip install -r requirements.txt -q", cwd=REPO)
run("git rev-parse HEAD", cwd=REPO)

# --- stage the MetaDiT data (attach the data dataset) ---
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
print("data:", data_root)

# --- stage the trained checkpoint (attach the checkpoint dataset) ---
ckpts = find(CHECKPOINT_NAME)
assert ckpts, f"{CHECKPOINT_NAME!r} not found under /kaggle/input"
(REPO / "checkpoints" / "unified").mkdir(parents=True, exist_ok=True)
target = REPO / "checkpoints" / "unified" / "latest.pt"
shutil.copy(str(ckpts[0]), str(target))
print("checkpoint:", ckpts[0], "->", target)

OUT.mkdir(parents=True, exist_ok=True)


def battery(name, cmd):
    out = OUT / f"{name}.json"
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    print(f"\n=== {name} (exit {r.returncode}) ===\n{r.stdout[-4000:]}", flush=True)
    if r.stderr:
        print(f"--- stderr ---\n{r.stderr[-2000:]}", flush=True)
    out.write_text(r.stdout)
    return r.returncode


CKPT = "checkpoints/unified/latest.pt"
codes = {
    "benchmark": battery("baseline_scenarioA",
        f"python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml "
        f"--checkpoint {CKPT} --split test --scenario A --samples 0 --candidates 4 "
        f"--nn-samples 512 --device cuda"),
    "eval_scenarios": battery("baseline_eval_scenarios",
        f"python scripts/eval/eval_scenarios.py --config configs/unified.yaml "
        f"--checkpoint {CKPT} --scenario all --samples 512 --device cuda"),
    "masked_fill": battery("baseline_masked_fill",
        f"python scripts/diagnostics/masked_fill_check.py --config configs/unified.yaml "
        f"--checkpoint {CKPT} --samples 32 --device cuda"),
    "guidance_gap": battery("baseline_guidance_gap",
        f"python scripts/diagnostics/run_guidance_gap_sweep.py --config configs/unified.yaml "
        f"--checkpoint {CKPT} --device cuda"),
}
(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nBASELINE DONE", codes, flush=True)
