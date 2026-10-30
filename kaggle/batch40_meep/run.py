"""Kaggle kernel: Meep spectra for batch40 (20 symmetric + 20 asymmetric designs).

Input: batch40_designs.npz (repo @ 443d737; shared scalars = dataset mean).
Runs _meep_novel.py under micromamba pymeep; writes
/kaggle/working/out/batch40_meep.json (per-design flux + eigenmode amplitudes,
rewritten per design). CPU only.
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "443d737"
REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/out")
MM = Path("/kaggle/working/mmbin")
MEEP_ENV = Path("/kaggle/working/meep_env")


def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


OUT.mkdir(parents=True, exist_ok=True)
run(f"git clone -q {REPO_URL} {REPO}")
run(f"git checkout -q {REPO_REF}", cwd=REPO)
run("git rev-parse HEAD", cwd=REPO)

MM.mkdir(parents=True, exist_ok=True)
run(f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | "
    f"tar -xj -C {MM} bin/micromamba")
run(f"{MM}/bin/micromamba create -y -p {MEEP_ENV} -c conda-forge pymeep "
    f"2>&1 | tail -5")
run(f"{MEEP_ENV}/bin/python -c 'import meep; print(meep.__version__)'")

r = run(f"{MEEP_ENV}/bin/python _meep_novel.py batch40_designs.npz "
        f"{OUT}/batch40_meep.json", cwd=REPO, check=False)

print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
print("BATCH40 MEEP DONE exit=", r.returncode, flush=True)
