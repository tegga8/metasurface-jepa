"""Build b40_merged.json (the 38-40 novel geometries' Meep spectra) from the
fetched t1_orig dir, filtered to the batch40 design names."""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:\metamaterials_scratch\t1_orig"
DST = r"D:\metamaterials_scratch\b40_merged.json"

b = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
want = set(b["names"].tolist())

designs, refs = [], {}
seen = {}
for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
    with open(f, encoding="utf-8") as fh:
        j = json.load(fh)
    refs.update(j.get("refs", {}))
    for d in j.get("designs", []):
        nm = d.get("name")
        if nm in want and nm not in seen:
            seen[nm] = d
designs = list(seen.values())

print("originals matched:", len(designs), "of", len(want))
okc = sum(1 for d in designs if d.get("passivity_ok"))
print("passivity_ok:", okc)
missing = sorted(want - {d["name"] for d in designs})
if missing:
    print("missing from Meep run:", missing)

with open(DST, "w", encoding="utf-8") as fh:
    json.dump({"designs": designs, "refs": refs}, fh)
print("saved", DST)
