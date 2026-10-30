"""Kaggle kernel: Meep spectra for the 36 decoded symbol geometries (CPU).

Input: symbol_decodes.npz (4 models x 9 symbols, repo @ pinned commit).
Runs _meep_novel.py under a micromamba pymeep environment; writes
/kaggle/working/out/symbol_decodes_meep.json (per-design flux + eigenmode
amplitudes; resumable). The comparison Meep(decoded) vs Meep(symbol) is
phase-convention-free (|a_s/a_e| is invariant) and runs locally afterwards.
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "6637a8d"
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

r = run(f"{MEEP_ENV}/bin/python _meep_novel.py symbol_decodes.npz "
        f"{OUT}/symbol_decodes_meep.json", cwd=REPO, check=False)

print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
print("SYMBOL-DECODES MEEP DONE exit=", r.returncode, flush=True)
