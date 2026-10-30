"""Kaggle kernel: Meep full-wave spectra for the 9 symbol designs (CPU).

Runs _meep_novel.py (repo @ 3f33a85) under a micromamba pymeep environment
(conda-forge; linux-64). Writes /kaggle/working/out/meep_symbols_out.json
(per-design flux magnitude + eigenmode amplitudes under / EVEN_Z / ODD_Z,
rewritten after every design). No GPU, no laptop load; pushed from the
WSL-side CLI (account akashkesav).

Next stage (local, light): _novel_modeltest.py symbol <scratchpad> with the
fitted phase convention -> IoU decoded vs original glyph + control.
"""

import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "3f33a85"
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

r = run(f"{MEEP_ENV}/bin/python _meep_novel.py symbol_designs.npz "
        f"{OUT}/meep_symbols_out.json", cwd=REPO, check=False)

print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
print("MEEP SYMBOLS DONE exit=", r.returncode, flush=True)
