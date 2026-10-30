"""MetaDiT-S decoding of the 40 novel geometries' Meep spectra, on Kaggle GPU.

Same task, same input as our JEPA decode: b40_merged.json holds the fixed
harness's Meep spectra for the novel geometries and _meep_convention.json (from
the repo, pinned) maps them to MetaDiT's [2,301] layout, so the condition
tensor is bit-identical to the one the unified 192-D JEPA received.

The released DiT checkpoint comes from the attached private dataset rather than
git, because data/metadit/* is gitignored and the upstream HuggingFace repo
(Hao-Li-131/MetaDiT-AAAI2026) is gated.

Pinned to b9121e1.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_URL = "https://github.com/tegga8/metasurface-jepa.git"
REPO_REF = "b9121e1"
WORK = Path("/kaggle/working")
REPO = WORK / "repo"
OUT = WORK / "out"


def run(argv, cwd=None, check=True):
    print(f"\n$ {' '.join(str(a) for a in argv)}", flush=True)
    return subprocess.run([str(a) for a in argv], check=check, cwd=cwd,
                          capture_output=True, text=True)


def find_weights():
    """The attached dataset lands under /kaggle/input/<slug>/."""
    for p in Path("/kaggle/input").rglob("metadit-small.bin"):
        print(f"found weights: {p}", flush=True)
        return p
    raise SystemExit("metadit-small.bin not found under /kaggle/input")


def find_conditions():
    """Locate the bundled Meep spectra.

    Do not assume /kaggle/working: the first version of this kernel did, and
    died with FileNotFoundError on the copy step. Search the working tree and
    the input mount instead.
    """
    for root in ("/kaggle/working", "/kaggle/input"):
        for p in Path(root).rglob("b40_merged.json"):
            print(f"found conditions: {p}", flush=True)
            return p
    raise SystemExit("b40_merged.json not found under /kaggle")


def main():
    import torch
    print("=" * 60, flush=True)
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "-q", REPO_URL, str(REPO)])
    run(["git", "checkout", "-q", REPO_REF], cwd=REPO)
    print("checked out", run(["git", "rev-parse", "HEAD"],
                             cwd=REPO).stdout.strip(), flush=True)

    w = find_weights()
    dest = REPO / "data/metadit/weights"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(w, dest / "metadit-small.bin")
    print(f"staged checkpoint: {dest / 'metadit-small.bin'}", flush=True)

    # b40_merged.json ships with this kernel; the convention comes from the repo.
    src = find_conditions()
    shutil.copy(src, REPO / "b40_merged.json")
    print(f"staged conditions: {REPO / 'b40_merged.json'} "
          f"({(REPO / 'b40_merged.json').stat().st_size} bytes)", flush=True)

    r = subprocess.run(
        ["python", "_metadit_decode.py", "b40_merged.json",
         "_meep_convention.json", str(OUT)],
        cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"decode failed rc={r.returncode}")

    total = time.time() - t_start
    print("=" * 60, flush=True)
    print(f"TOTAL WALL (incl. clone+weights): {total:.1f}s", flush=True)
    npz = OUT / "metadit_decoded.npz"
    if npz.exists():
        import numpy as np
        d = np.load(npz, allow_pickle=True)
        print(f"generations: {d['patterns'].shape} "
              f"mean occupancy {d['patterns'].mean():.3f}", flush=True)
        shutil.copy(npz, OUT.parent / "metadit_decoded.npz")
    print("KAGGLE MDDECODE DONE", flush=True)
    print(json.dumps({"total_wall_s": round(total, 1)}))


if __name__ == "__main__":
    main()