"""Merge the md_t2 per-set Meep dirs into the filenames _compare_phase2 expects,
then score MetaDiT's task-2 physics against the requested targets + null.

Runs the same scorer as the v1/v2 arms, so the three models are comparable.
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


def main():
    out = os.path.join(SP, "md_t2_merged")
    n_a = merge(os.path.join(SP, "md_t2", "metadit_A_interpolated"),
                os.path.join(out, "decoded_A_interpolated_meep.json"))
    n_b = merge(os.path.join(SP, "md_t2", "metadit_B_synthetic"),
                os.path.join(out, "decoded_B_synthetic_meep.json"))
    print(f"merged A={n_a} B={n_b}")

    # passivity tally (MetaDiT's own decodes)
    for tag, path in (("A", "metadit_A_interpolated"),
                      ("B", "metadit_B_synthetic")):
        j = json.load(open(os.path.join(SP, "md_t2", path, "e_0_0.json")
                           if False else os.path.join(out, "decoded_"
                                                      + ("A_interpolated" if tag == "A" else "B_synthetic")
                                                      + "_meep.json"),
                           encoding="utf-8"))
        ds = j["designs"]
        bad = [d["name"] for d in ds if not d.get("passivity_ok")]
        print(f"set {tag}: {len(ds)} designs, {len(bad)} passivity failures "
              f"{bad[:5]}")

    r = subprocess.run([PY, "_compare_phase2.py", out,
                        os.path.join(SP, "random_spectra.npz"),
                        os.path.join(REPO, "_meep_convention.json")],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])


if __name__ == "__main__":
    main()
