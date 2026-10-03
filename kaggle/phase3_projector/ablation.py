"""Kaggle kernel: Phase-3b projector ablation.

Trains the unified JEPA for 10000 steps with each projector variant
(none | linear | mlp; the incumbent mlp_bn is the Phase-2 reference) on the
current schedule, and runs the canonical battery on every arm. Writes real JSON
under /kaggle/working/proj/<kind>/.

Config overrides are generated per arm from configs/unified.yaml.
"""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "b811da6f65349d87c68c6c0feba72441082b3d41"   # Phase 3a/3b
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/proj")
ARMS = ["none", "linear", "mlp"]           # mlp_bn = the Phase-2 reference
BASE_CFG = REPO / "configs" / "unified.yaml"

OUT.mkdir(parents=True, exist_ok=True)


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

CKPT_DIR = REPO / "checkpoints" / "unified"


def battery(name, kind, cfg_name, ckpt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds = {
        "benchmark": (
            f"python scripts/benchmark/benchmark_metadit.py --config {cfg_name} "
            f"--checkpoint {ckpt} --split test --scenario A --samples 0 --candidates 4 "
            f"--nn-samples 512 --nn-pools 512,5000,20000 --device cuda "
            f"--out {out_dir}/scenarioA.json"),
        "eval_scenarios": (
            f"python scripts/eval/eval_scenarios.py --config {cfg_name} "
            f"--checkpoint {ckpt} --scenario all --samples 512 --device cuda "
            f"--out {out_dir}/eval_scenarios.json"),
    }
    codes = {}
    for nm, cmd in cmds.items():
        r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
        (out_dir / f"{nm}.log").write_text(r.stdout)
        print(f"\n=== {kind}/{nm} (exit {r.returncode}) ===\n{r.stdout[-2500:]}",
              flush=True)
        codes[nm] = r.returncode
    return codes


results = {}
for kind in ARMS:
    cfg = yaml.safe_load(open(BASE_CFG))
    cfg["loss"]["projector_type"] = kind
    cfg_name = f"configs/unified_proj_{kind}.yaml"
    with open(REPO / cfg_name, "w") as f:
        yaml.safe_dump(cfg, f)

    # preflight once, on the first arm's config (mirrors the training objective).
    if kind == ARMS[0]:
        run(f"python scripts/train/train_unified.py --config {cfg_name} "
            f"--device cuda --preflight", cwd=REPO)

    print(f"\n########## ARM {kind}: training ##########", flush=True)
    r = subprocess.run(
        f"python scripts/train/train_unified.py --config {cfg_name} --device cuda",
        shell=True, cwd=REPO, capture_output=True, text=True)
    od = OUT / kind
    od.mkdir(parents=True, exist_ok=True)
    (od / "train.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
    print(f"--- {kind} training exit {r.returncode} ---", flush=True)
    print(r.stdout[-1500:], flush=True)

    ckpt_keep = CKPT_DIR / f"proj_{kind}.pt"
    if (CKPT_DIR / "latest.pt").exists():
        shutil.copy(str(CKPT_DIR / "latest.pt"), str(ckpt_keep))
    codes = battery(kind, kind, cfg_name,
                    f"checkpoints/unified/proj_{kind}.pt", od)
    codes["train"] = r.returncode
    results[kind] = codes

(OUT / "exit_codes.json").write_text(json.dumps(results, indent=2))
print("\nPROJECTOR ABLATION DONE", results, flush=True)
