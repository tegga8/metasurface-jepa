"""Task 1, step 1: full-wave (Meep) spectra on the fixed harness.

Two sets, one kernel:

  1. meep_calib_designs.npz - 20 MetaDiT validation geometries with their true
     CST spectra in meep_calib_targets.npz. These are the fit AND held-out
     validation set for the Meep->CST convention, which has to be refitted
     because the old one was fitted on the defective harness (residuals up to
     3898, one item excluded). Each has its own (l,h,r), so each needs its own
     empty-cell reference.
  2. batch40_designs.npz - the 40 novel geometries (20 symmetric, 20
     asymmetric) that are absent from the dataset, shared dataset-mean scalars.

Runs _meep_novel.py under a micromamba pymeep environment. The harness now
terminates on field decay with a lossy medium instead of a fixed 1100-unit run,
so each design costs ~110s rather than ~1500s and all 60 fit in one kernel.
CPU only; writes JSON after every design, so a kill still leaves the completed
designs on disk.

Pinned to the harness fix (e2997e6) and the calibration set (03a4688).
"""

import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "03a4688"
REPO = Path("/kaggle/working/repo")
WORK = Path("/kaggle/working")
OUT = WORK / "out"
MM = WORK / "mmbin"
MEEP_ENV = WORK / "meep_env"
MICROMAMBA_URL = ("https://micro.mamba.pm/api/micromamba/linux-64/latest")


def run(argv, cwd=None, check=True):
    """Run a command from an argument list. No shell, so nothing is expanded."""
    print(f"\n$ {' '.join(str(a) for a in argv)}", flush=True)
    return subprocess.run([str(a) for a in argv], check=check, cwd=cwd,
                          capture_output=True, text=True)


def fetch_micromamba():
    """Download and unpack micromamba without shelling out to curl | tar."""
    MM.mkdir(parents=True, exist_ok=True)
    tar_path = WORK / "micromamba.tar.bz2"
    print(f"\nGET {MICROMAMBA_URL}", flush=True)
    urllib.request.urlretrieve(MICROMAMBA_URL, tar_path)
    with tarfile.open(tar_path) as tf:
        member = tf.getmember("bin/micromamba")
        tf.extract(member, MM)
    (MM / "bin" / "micromamba").chmod(0o755)
    print(f"micromamba at {MM / 'bin' / 'micromamba'}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "-q", REPO_URL, str(REPO)])
    run(["git", "checkout", "-q", REPO_REF], cwd=REPO)
    head = run(["git", "rev-parse", "HEAD"], cwd=REPO)
    print("checked out", head.stdout.strip(), flush=True)

    fetch_micromamba()
    mm = MM / "bin" / "micromamba"
    r = run([str(mm), "create", "-y", "-p", str(MEEP_ENV), "-c", "conda-forge",
             "pymeep"], check=False)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    run([str(MEEP_ENV / "bin" / "python"), "-c",
         "import meep; print('meep', meep.__version__)"])

    meep_py = MEEP_ENV / "bin" / "python"

    # 1. calibration set: per-item scalars -> 20 distinct empty-cell references.
    r1 = run([str(meep_py), "_meep_novel.py", "meep_calib_designs.npz",
              str(OUT / "meep_calib_out.json")], cwd=REPO, check=False)
    print(f"CALIB exit={r1.returncode}", flush=True)

    # 2. the 40 novel geometries: shared scalars -> one empty-cell reference.
    r2 = run([str(meep_py), "_meep_novel.py", "batch40_designs.npz",
              str(OUT / "batch40_meep.json")], cwd=REPO, check=False)
    print(f"BATCH40 exit={r2.returncode}", flush=True)

    print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
    for p in sorted(OUT.glob("*.json")):
        try:
            d = json.load(open(p))
            designs = d.get("designs", [])
            bad = sum(1 for x in designs if not x.get("passivity_ok", True))
            print(f"  {p.name}: {len(designs)} designs, {bad} passivity "
                  f"failures", flush=True)
        except (ValueError, OSError) as exc:
            # Best-effort summary only; the spectra themselves are already
            # written, and a truncated file must not suppress DONE below.
            print(f"  {p.name}: could not summarise ({exc})", flush=True)
    print("MEEP TASK1 DONE", flush=True)


if __name__ == "__main__":
    main()