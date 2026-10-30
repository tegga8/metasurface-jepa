import glob
import json
import os

import numpy as np

base = r"D:\metamaterials_scratch\t3"
for tag in ["jepa_v1_spec_m20", "jepa_v1_B_true", "jepa_v1_spec_p20",
            "jepa_v2_spec_p20", "jepa_v2_B_true"]:
    d = os.path.join(base, tag)
    f = sorted(glob.glob(os.path.join(d, "e_0_*.json")))[0]
    j = json.load(open(f, encoding="utf-8"))
    print(tag)
    print("  refs keys:", sorted(j.get("refs", {}).keys()))
    print("  harness:", j.get("harness"))
    for k, r in list(j.get("refs", {}).items())[:2]:
        fe = np.abs(np.asarray(r["flux"], dtype=float))
        m = r.get("meta") or {}
        print(f"  key={k} len={fe.size} argmax={int(np.nanargmax(fe))} "
              f"vals@[0,100,150,200,300]="
              f"{[round(float(fe[i]), 10) for i in (0, 100, 150, 200, 300)]} "
              f"window={m.get('window')} ring={m.get('ringdown')}")
