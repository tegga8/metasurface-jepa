import json
import glob
import numpy as np
import os

for f in sorted(glob.glob(r"D:\metamaterials_scratch\t3res\*.json")):
    j = json.load(open(f, encoding="utf-8"))
    names = [e.get("name", f"idx{e.get('idx')}") for e in j.get("designs", [])]
    refs = list(j.get("refs", {}).keys())
    print(f"{os.path.basename(f):16s} designs={len(names)} refs={refs} names={names}")
    if j.get("designs"):
        e = j["designs"][0]
        print(f"     peak_ratio={e.get('peak_ratio'):.4f} "
              f"window={e.get('window')} ring={e.get('ringdown')} "
              f"peak_ratio_flux={e.get('peak_ratio_flux')}")
