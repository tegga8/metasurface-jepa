"""Kaggle kernel: generate the 9 symbolic meta-atom geometries.

Operator request 2026-10-05: swastika, cross, crescent moon, aum, khanda
(Sikh), dharmachakra (Buddhist), yin-yang, torii (Shinto), faravahar
(Zoroastrian). Runs scripts/data/make_symbol_designs.py from the repo
(commit 4c108aa); writes symbol_designs.npz + symbol_designs_preview.png
under /kaggle/working/out. CPU-only; staged dataset used for the D4 novelty
check.

Pushed from the WSL-side Kaggle CLI (account akashkesav).
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "4c108aa"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/out")


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


OUT.mkdir(parents=True, exist_ok=True)

run(f"git clone -q {REPO_URL} {REPO}")
run(f"git checkout -q {REPO_REF}", cwd=REPO)
run("git rev-parse HEAD", cwd=REPO)

roots = sorted({p.parent for p in Path("/kaggle/input").rglob("train_set.mat")})
assert len(roots) == 1, f"expected one data root, got {roots}"
print("data root:", roots[0], flush=True)

r = run(f"python scripts/data/make_symbol_designs.py --out {OUT} "
        f"--data {roots[0]}", cwd=REPO, check=False)

print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
print("SYMBOL-GEN DONE exit=", r.returncode, flush=True)
