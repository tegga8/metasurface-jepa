"""Score Task 1 + Task 2 physics for v1 and v2 with the identical scorers.

Validates first by re-scoring v2 (the recorded numbers must come back), then
scores v1 from the clean v1all run. Usage: python _score_all.py
"""

import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
SP = r"D:\metamaterials_scratch"
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


def split_by_npz(indir, prefix_map):
    """Split a dir of per-element JSONs into per-npz dirs."""
    counts = {k: 0 for k in prefix_map}
    for out in prefix_map.values():
        os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(indir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        npz = (j.get("npz") or "").replace(".npz", "")
        if npz not in prefix_map:
            continue
        dest = os.path.join(prefix_map[npz], os.path.basename(f))
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(j, fh)
        counts[npz] += len(j.get("designs", []))
    return counts


def run(title, argv):
    print("\n" + "#" * 78)
    print("#", title)
    print("#" * 78, flush=True)
    r = subprocess.run([PY] + argv, cwd=REPO, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
    return r.returncode


def main():
    # ---- split the v2 decoded run (runs/dec) by mask ----
    c = split_by_npz(os.path.join(SP, "dec_v2"), {
        "decoded_full": os.path.join(SP, "dec_v2_full"),
        "decoded_half": os.path.join(SP, "dec_v2_half"),
    })
    print("dec_v2 split:", c)

    # ---- merge the v1 per-set dirs ----
    for tag in ("task1_full", "task1_half", "task2_A", "task2_B"):
        n = merge(os.path.join(SP, "v1all", tag),
                  os.path.join(SP, "v1all_merged", tag + ".json"))
        print("merged", tag, n)

    # ---- Phase 2 scorer wants exact filenames ----
    p2v1 = os.path.join(SP, "v1all_p2")
    os.makedirs(p2v1, exist_ok=True)
    for tag, name in (("task2_A", "A_interpolated"),
                      ("task2_B", "B_synthetic")):
        src = os.path.join(SP, "v1all_merged", tag + ".json")
        dst = os.path.join(p2v1, f"decoded_{name}_meep.json")
        with open(src, encoding="utf-8") as fh:
            j = json.load(fh)
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(j, fh)

    orig = os.path.join(SP, "t1_orig")

    # ---- TASK 1 ----
    rc1 = run("TASK 1 physics | v1 (clean 638208 run)", [
        "_compare_decoded_meep.py", orig,
        os.path.join(SP, "v1all", "task1_full"),
        os.path.join(SP, "v1all", "task1_half")])
    rc2 = run("TASK 1 physics | v2 (sched-fix, must reproduce the ledger)",
              ["_compare_decoded_meep.py", orig,
               os.path.join(SP, "dec_v2_full"),
               os.path.join(SP, "dec_v2_half")])

    # ---- TASK 2 ----
    rs = os.path.join(SP, "random_spectra.npz")
    conv = os.path.join(REPO, "_meep_convention.json")
    rc3 = run("TASK 2 physics | v1 (clean 638208 run)",
              ["_compare_phase2.py", p2v1, rs, conv])
    rc4 = run("TASK 2 physics | v2 (sched-fix, must reproduce the ledger)",
              ["_compare_phase2.py", os.path.join(SP, "p2_out"), rs, conv])

    print("\nreturn codes:", rc1, rc2, rc3, rc4)


if __name__ == "__main__":
    main()
