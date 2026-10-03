"""Kaggle kernel: Scenario-C crossed seed study (Task §1).

3 training checkpoints (the 3-seed CI) x 3 evaluation seeds {3,4,5} = 9 Scenario-C
evaluations. No training. Protocol identical to A3; only the checkpoint and the eval
seed vary. Writes the full JSON (incl. per-sample arrays) per cell.
"""

import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "3a69527"          # A3-A5 + D (A1 --eval-seed control)
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/x")
CKPTS = ("seed0", "seed1", "seed2")
EVAL_SEEDS = (3, 4, 5)


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find(name, roots=("/kaggle/input",)):
    hits = []
    for r in roots:
        hits += list(Path(r).rglob(name))
    return hits


def swap_link(link, target):
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=True)


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
swap_link(REPO / "data" / "metadit", data_roots[0])

ckpt_paths = {}
for name in CKPTS:
    hits = find(f"{name}.pt")
    assert hits, f"{name}.pt not found under /kaggle/input"
    ckpt_paths[name] = hits[0]
print("checkpoints:", {k: str(v) for k, v in ckpt_paths.items()}, flush=True)

codes = {}
for name in CKPTS:
    for e in EVAL_SEEDS:
        out_json = OUT / f"crossed_{name}_e{e}.json"
        cmd = (f"python scripts/eval/eval_scenarios.py --config configs/unified.yaml "
               f"--checkpoint {ckpt_paths[name]} --scenario C --samples 512 "
               f"--eval-seed {e} --device cuda --out {out_json}")
        r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
        print(f"\n=== {name} eval_seed {e} (exit {r.returncode}) ===", flush=True)
        print(r.stdout[-800:], flush=True)
        if r.stderr:
            print(f"--- stderr ---\n{r.stderr[-800:]}", flush=True)
        (OUT / f"crossed_{name}_e{e}.log").write_text(r.stdout)
        codes[f"{name}_e{e}"] = r.returncode

(OUT / "exit_codes.json").write_text(json.dumps(codes, indent=2))
print("\nCROSSED DONE", codes, flush=True)
