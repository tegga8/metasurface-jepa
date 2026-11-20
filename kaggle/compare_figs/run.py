import os, sys, json, glob, subprocess

import numpy as np
import torch
import yaml

OUT = "/kaggle/working"
REPO = "/kaggle/working/repo"

if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tegga8/metasurface-jepa.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "fetch", "--depth", "1", "origin", "work-192d"], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-f", "FETCH_HEAD"], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout)

sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "external", "metadit"))

from assembly import build_unified_model
from data.dataset import MetaDiTDataset
from data.factorize import factorize_geometry
from physics.physics_loop import load_surrogate


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits, key=len)[0]


TEST = find("test_set.mat")
SURR = find("surrogate_model.bin")
SPECW = find("spec_encoder.pth")
BASE_CKPT = find("full_epoch_final.pt")
IMP1_CKPT = find("imp1_seed0.pt")
for p in (TEST, SURR, SPECW, BASE_CKPT, IMP1_CKPT):
    print("using", p, os.path.getsize(p) // 1024 // 1024, "MB")

IDX = [1658, 3750, 3807, 7292, 10321, 11422, 11497, 13300, 14523, 15838]
N = len(IDX)

ds = MetaDiTDataset(TEST)
pos = {int(v): i for i, v in enumerate(ds.indices)}
rows = torch.stack([ds[pos[i]][0] for i in IDX])
spec = torch.stack([ds[pos[i]][1] for i in IDX])
occ, scal = factorize_geometry(rows)
surrogate = load_surrogate(SURR)

cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "unified.yaml"), encoding="utf-8"))
cfg["weights"]["spectrum"] = SPECW
cfg["weights"]["surrogate"] = SURR


def build(ckpt_path, cfg_path):
    c = yaml.safe_load(open(os.path.join(REPO, cfg_path), encoding="utf-8"))
    c["weights"]["spectrum"] = SPECW
    m = build_unified_model(c, SPECW, device="cpu")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(cfg_path, "missing", len(missing), "unexpected", len(unexpected))
    m.eval()
    return m


def mask_for(ratio, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(N, 16, 16, generator=g) >= ratio).float()


REG = {"A": (1.00, [False] * 3, 11),
       "B": (0.50, [True, False, False], 12),
       "C": (0.25, [True, True, True], 13)}

res = {}
for tag, ck, cfgf in (("base", BASE_CKPT, "configs/unified.yaml"),
                      ("variant", IMP1_CKPT, "configs/unified_imp1.yaml")):
    m = build(ck, cfgf)
    for reg, (ratio, known, seed) in REG.items():
        mask = mask_for(ratio, seed)
        sk = torch.tensor([known] * N, dtype=torch.bool)
        with torch.no_grad():
            o = m(occ, scal, sk, spec, mask, with_target=False)
            geom, _ = m.decode_geometry(o["z_hat"], o["scalar_pred"], occ_input=occ,
                                        mask=mask, scalar_known=sk, scalar_values=scal,
                                        hard_forward=True)
            pred = surrogate(geom)
        res[f"{tag}_{reg}"] = geom.cpu().numpy()
        np.save(os.path.join(OUT, f"spec_{tag}_{reg}.npy"), pred.cpu().numpy())
        mae = (pred - spec).abs().mean(dim=(1, 2)).numpy()
        corr = [np.corrcoef(pred[i].flatten(), spec[i].flatten())[0, 1] for i in range(N)]
        print(f"{tag} {reg}: MAE {mae.mean():.4f} corr {np.mean(corr):.4f}")
        np.save(os.path.join(OUT, f"occ_{tag}_{reg}.npy"),
                (geom[:, :1] > 0).float().numpy())

np.save(os.path.join(OUT, "idx.npy"), np.array(IDX))
print("DONE")
