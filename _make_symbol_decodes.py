"""Bundle the decoded symbol geometries (4 models x 9) into a Meep input npz."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SP = sys.argv[1]

files = [
    ("S2", "symbol_dec_S2.npz"),
    ("S1", "symbol_dec_S1.npz"),
    ("L1", "symbol_dec_L1.npz"),
    ("base", "symbol_dec_base(v2,s0).npz"),
]
names = np.load(os.path.join(REPO, "symbol_designs.npz"),
                allow_pickle=True)["names"].tolist()

pats, tags = [], []
for model, f in files:
    prob = np.load(os.path.join(SP, f))["prob"]
    geom = (np.squeeze(prob) > 0.5).astype(np.uint8)
    assert geom.shape == (len(names), 64, 64), (model, geom.shape)
    for i, nm in enumerate(names):
        pats.append(geom[i])
        tags.append(f"{model}/{nm}")

scalars = np.load(os.path.join(REPO, "symbol_designs.npz"),
                  allow_pickle=True)["scalars"]
out = os.path.join(REPO, "symbol_decodes.npz")
np.savez(out, patterns=np.stack(pats), scalars=scalars,
         names=np.array(tags))
print("saved", out, np.stack(pats).shape, "designs:", tags)
