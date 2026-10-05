"""Kaggle kernel: base config, seed 0, 70k steps — checkpoint for the grad-share probe.

Produces a CURRENT-code base checkpoint (configs/unified.yaml, seed 0) so the
per-term gradient-share probe can be measured under the identical 6-batch
protocol used for S1/S2 (the recovered Sep-era artifact is a different code
variant and cannot serve as the current base row). No battery — the checkpoint
is the deliverable.

Writes /kaggle/working/base/seed0.pt.
"""

import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "cafa3bd"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/base")
VARIANT = "configs/unified.yaml"
STEPS = 70000


def run(cmd, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, cwd=cwd)


OUT.mkdir(parents=True, exist_ok=True)

subprocess.run("nvidia-smi", shell=True, check=False)

run(f"git clone {REPO_URL} {REPO}")
run(f"git checkout {REPO_REF}", cwd=REPO)
subprocess.run(
    "pip install -q numpy scipy PyYAML scikit-learn timm einops transformers "
    "matplotlib tqdm safetensors", shell=True, cwd=REPO, check=False)
run("git rev-parse HEAD", cwd=REPO)

data_roots = sorted({
    p.parent.parent for p in Path("/kaggle/input").rglob("train_set.mat")
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

run(f"python scripts/train/train_unified.py --config {VARIANT} "
    "--device cuda --preflight", cwd=REPO)

tr = subprocess.run(
    f"python scripts/train/train_unified.py --config {VARIANT} "
    f"--device cuda --max-steps {STEPS}", shell=True, cwd=REPO,
    capture_output=True, text=True)
(OUT / "seed0_train.log").write_text(tr.stdout + "\n===STDERR===\n" + tr.stderr)
print(f"[train seed0] exit {tr.returncode}\n{tr.stdout[-800:]}", flush=True)

shutil.copy(str(REPO / "checkpoints" / "unified" / "latest.pt"),
            str(OUT / "seed0.pt"))
print("BASE-CHECKPOINT DONE", flush=True)
