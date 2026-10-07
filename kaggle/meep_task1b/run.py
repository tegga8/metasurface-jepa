"""Task 1 step 1 on Kaggle: full-wave spectra for both design sets, phase-correct.

Re-run of metasurface-jepa-meep-task1, which was pinned before two fixes:

  1. The harness recorded only |T| and discarded the complex phase, but the
     Meep->CST convention maps a COMPLEX transmission onto MetaDiT's [2,301]
     target and the model is conditioned on that complex spectrum. The first
     kernel's output therefore could not have been pushed into the model.
  2. --max-ring is raised to 2500. At the old 600 cap, 2 of the 20 calibration
     designs stopped without meeting the decay criterion and 4 more finished
     above 400, i.e. their ringdowns were cut short. Unconverged spectra are
     worse than no spectra.

Both sets, one kernel:

  meep_calib_designs.npz  20 MetaDiT validation geometries, per-item scalars
                          (one empty-cell reference each). These carry the true
                          CST spectra in meep_calib_targets.npz and are what the
                          Meep->CST convention is fitted and leave-one-out
                          validated on.
  batch40_designs.npz     the 40 novel geometries (20 symmetric, 20
                          asymmetric), shared dataset-mean scalars.

Runs _meep_novel.py under a micromamba pymeep environment; JSON is rewritten
after every design, so a kill still leaves the completed designs on disk. Each
design records peak_ratio / passivity_ok, and the run prints a violation count.

Pinned to 33b4b6a.
"""

import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "33b4b6a"
REPO = Path("/kaggle/working/repo")
WORK = Path("/kaggle/working")
OUT = WORK / "out"
MM = WORK / "mmbin"
MEEP_ENV = WORK / "meep_env"
MAX_RING = 2500
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"


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
        tf.extract(tf.getmember("bin/micromamba"), MM)
    (MM / "bin" / "micromamba").chmod(0o755)
    print(f"micromamba at {MM / 'bin' / 'micromamba'}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "-q", REPO_URL, str(REPO)])
    run(["git", "checkout", "-q", REPO_REF], cwd=REPO)
    print("checked out", run(["git", "rev-parse", "HEAD"],
                             cwd=REPO).stdout.strip(), flush=True)

    fetch_micromamba()
    r = run([str(MM / "bin" / "micromamba"), "create", "-y", "-p",
             str(MEEP_ENV), "-c", "conda-forge", "pymeep"], check=False)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    meep_py = MEEP_ENV / "bin" / "python"
    run([str(meep_py), "-c", "import meep; print('meep', meep.__version__)"])

    for npz, out_name in (("meep_calib_designs.npz", "meep_calib_out.json"),
                          ("batch40_designs.npz", "batch40_meep.json")):
        print(f"\n===== {npz} -> {out_name} =====", flush=True)
        r = run([str(meep_py), "_meep_novel.py", npz,
                 str(OUT / out_name), "--max-ring", str(MAX_RING)],
                cwd=REPO, check=False)
        print(f"{npz} exit={r.returncode}", flush=True)
        print(r.stdout[-3000:], flush=True)

    print("OUTPUTS:", sorted(p.name for p in OUT.iterdir()), flush=True)
    for p in sorted(OUT.glob("*.json")):
        try:
            d = json.load(open(p))
            designs = d.get("designs", [])
            bad = sum(1 for x in designs if not x.get("passivity_ok", True))
            capped = sum(1 for x in designs if x.get("hit_cap"))
            print(f"  {p.name}: {len(designs)} designs, {bad} passivity "
                  f"failures, {capped} ringdown-capped", flush=True)
        except (ValueError, OSError) as exc:
            # Best-effort summary only; the spectra are already written, and a
            # truncated file must not suppress DONE below.
            print(f"  {p.name}: could not summarise ({exc})", flush=True)
    print("MEEP TASK1B DONE", flush=True)


if __name__ == "__main__":
    main()