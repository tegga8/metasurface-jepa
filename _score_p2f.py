"""Score the corrected (realizability-screened) phase-2 arm.

Merges the per-set Meep dirs into the filenames _compare_phase2.py expects and
runs it against the screened targets, so the numbers are directly comparable to
the old phase-2 rows.

Usage: python _score_p2f.py <meep_root> <targets.npz> [which]
  which: A (feasible) | B (infeasible) | both
"""

import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def merge(indir, outpath):
    designs, refs = [], {}
    for f in sorted(glob.glob(os.path.join(indir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        designs += j.get("designs", [])
        refs.update(j.get("refs", {}))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump({"designs": designs, "refs": refs}, fh)
    return len(designs)


def main():
    root, targets = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    a_dir = sys.argv[4] if len(sys.argv) > 4 else "jepa_v1_A"
    b_dir = sys.argv[5] if len(sys.argv) > 5 else "jepa_v1_B"
    staged = os.path.join(root, "_staged")
    os.makedirs(staged, exist_ok=True)
    if which in ("A", "both"):
        d = os.path.join(root, a_dir)
        if os.path.isdir(d):
            n = merge(d, os.path.join(staged, "decoded_A_interpolated_meep.json"))
            print(f"merged A ({a_dir}): {n} designs")
    if which in ("B", "both"):
        d = os.path.join(root, b_dir)
        if os.path.isdir(d):
            n = merge(d, os.path.join(staged, "decoded_B_synthetic_meep.json"))
            print(f"merged B ({b_dir}): {n} designs")
    r = subprocess.run([PY, "_compare_phase2.py", staged, targets,
                        os.path.join(REPO, "_meep_convention.json")],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])


if __name__ == "__main__":
    main()
