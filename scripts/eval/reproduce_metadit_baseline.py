import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

import numpy as np
import torch
from scipy import io

from model.dit import DIT_MODEL
from model.spec_encoder import VanillaSpectrumEncoder
from model.surrogate import surrogate_s3
from datapipe import FreeFormDataset, SurrogateFreeFormDataset
from diffusion import create_diffusion

DATA_DIR = os.path.join(REPO_ROOT, "data", "metadit")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
SPLIT_DIR = os.path.join(DATA_DIR, "split_data")
GENERATION_DIR = os.path.join(DATA_DIR, "generation")

SPLITS = ["train", "val", "test"]


def split_path(split):
    return os.path.join(SPLIT_DIR, f"{split}_set.mat")


def verify_shapes():
    print("=" * 70)
    print("[1/3] Dataset shape verification (design doc SS7 Phase 0 / SS1.1)")
    print("=" * 70)
    for split in SPLITS:
        raw = io.loadmat(split_path(split))
        pattern = raw["pattern"]
        parameter = raw["parameter"]
        real = raw["real"]
        imag = raw["imag"]
        n = pattern.shape[-1]
        print(f"\n{split}_set.mat: N={n}")
        print(f"  pattern   {pattern.shape}  dtype={pattern.dtype}  "
              f"value range [{pattern.min():.4g}, {pattern.max():.4g}]")
        print(f"  parameter {parameter.shape}  dtype={parameter.dtype}")
        print(f"  real      {real.shape}  dtype={real.dtype}")
        print(f"  imag      {imag.shape}  dtype={imag.dtype}")
        assert pattern.shape == (64, 64, n), "pattern must be (64,64,N)"
        assert parameter.shape == (n, 3), "parameter must be (N,3)"
        assert real.shape == (n, 301) and imag.shape == (n, 301), "spectrum must be 301 pts"
        assert set(np.unique(pattern).tolist()).issubset({0, 1}), "pattern must be binary"

    print("\nRepository construction check (repo's own dataset classes):")
    for split in SPLITS:
        raw = io.loadmat(split_path(split))
        ff = FreeFormDataset.__new__(FreeFormDataset)
        ff.data = raw
        sur = SurrogateFreeFormDataset.__new__(SurrogateFreeFormDataset)
        sur.data = raw
        for idx in [0, 1, len(raw["pattern"]) // 2]:
            di = ff.__getitem__(idx)
            si = sur.__getitem__(idx)
            assert di["inputs"].shape == (3, 32, 32), "DiT grid must be 3x32x32"
            assert di["condition"].shape == (2, 301), "DiT condition must be 2x301"
            assert si["inputs"].shape == (3, 64, 64), "surrogate grid must be 3x64x64"
            assert si["labels"].shape == (2, 301), "surrogate label must be 2x301"
        print(f"  {split}: first-quadrant DiT grid (3,32,32) and full surrogate grid "
              f"(3,64,64) constructed, condition (2,301) [real, imag]")
    print("\nSHAPE VERIFICATION: PASS")


def verify_weights():
    print("\n" + "=" * 70)
    print("[2/3] Released weight loading (all three files, strict=True)")
    print("=" * 70)

    torch.set_grad_enabled(False)

    spec_path = os.path.join(WEIGHTS_DIR, "spec_encoder.pth")
    print(f"\n1) Spectrum encoder: {spec_path}")
    spec_ckpt = torch.load(spec_path, map_location="cpu")
    stripped = {k.split("context_encoder.")[1]: v for k, v in spec_ckpt.items()}
    spec_model = VanillaSpectrumEncoder()
    spec_model.load_state_dict(stripped, strict=True)
    spec_model.eval()
    n_spec = sum(p.nelement() for p in spec_model.parameters())
    print(f"   loaded into VanillaSpectrumEncoder (dim=256, num_blocks [1,1,1,1]); "
          f"prefix 'context_encoder.' stripped per train_metadit.py; params={n_spec:,}")
    with torch.no_grad():
        out = spec_model(torch.randn(2, 2, 301))
    assert out.shape == (2, 301, 256), f"unexpected spectrum-encoder output {out.shape}"
    print(f"   forward(2x2x301) -> {tuple(out.shape)} OK")

    dit_path = os.path.join(WEIGHTS_DIR, "metadit-small.bin")
    print(f"\n2) Geometry ViT / DiT: {dit_path}")
    diffusion = create_diffusion("500", learn_sigma=False)
    dit_model = DIT_MODEL["metadit_s"](diffusion=diffusion, condition_channel=301)
    dit_model.load_state_dict(torch.load(dit_path, map_location="cpu"), strict=True)
    dit_model.eval()
    n_dit = sum(p.nelement() for p in dit_model.parameters())
    print(f"   loaded into metadit_s (depth=12, hidden=384, heads=6, patch=2) "
          f"via generate.py build_model convention; params={n_dit:,}")
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([0])
        y = torch.randn(1, 301, 2).transpose(1, 2).contiguous()
        out = dit_model._forward(x, t, y)
    assert out.shape == (1, 3, 32, 32), f"unexpected DiT output {out.shape}"
    print(f"   forward(1x3x32x32) -> {tuple(out.shape)} OK")

    sur_path = os.path.join(WEIGHTS_DIR, "surrogate_model.bin")
    print(f"\n3) Forward EM surrogate: {sur_path}")
    sur_model = surrogate_s3()
    sur_model.load_state_dict(torch.load(sur_path, map_location="cpu"), strict=True)
    sur_model.eval()
    n_sur = sum(p.nelement() for p in sur_model.parameters())
    print(f"   loaded into surrogate_s3 via metric.py build_surrogate_model "
          f"convention; params={n_sur:,}")
    with torch.no_grad():
        pred = sur_model(torch.randn(1, 3, 64, 64)).prediction
    assert tuple(pred.shape) == (1, 2, 301), f"unexpected surrogate output {pred.shape}"
    print(f"   forward(1x3x64x64) -> {tuple(pred.shape)} OK")

    print("\nWEIGHT LOADING: PASS  (all three strict loads + batch-1 forward passes OK)")


def reproduce_baseline():
    print("\n" + "=" * 70)
    print("[3/3] Baseline spectrum evaluation (MetaDiT official metric.py on released seed0.json)")
    print("=" * 70)

    seed0 = os.path.join(GENERATION_DIR, "seed0.json")
    data = json.load(open(seed0))
    test_raw = io.loadmat(split_path("test"))
    print(f"seed0.json: {len(data)} items")

    table = {}
    for j in range(len(test_raw["real"])):
        key = tuple(np.round(np.concatenate(
            [test_raw["real"][j].astype(np.float32),
             test_raw["imag"][j].astype(np.float32)]), 4))
        table[key] = j
    used = set()
    for i, item in enumerate(data):
        cond = np.round(np.array(item["condition"], dtype=np.float32).reshape(-1), 4)
        hit = table.get(tuple(cond))
        assert hit is not None, f"seed0.json item {i} condition not in test_set.mat"
        assert hit not in used, f"seed0.json item {i} duplicates test row {hit}"
        used.add(hit)
    print(f"condition alignment: {len(used)}/{len(test_raw['real'])} test-set rows "
          "covered 1:1 (permuted order) -> seed0.json is MetaDiT's own forward "
          "prediction output on the TEST split (order irrelevant to the metric)")

    cmd = [
        sys.executable, os.path.join(METADIT_SRC, "metric.py"),
        "--data_path", GENERATION_DIR,
        "--model_path", os.path.join(WEIGHTS_DIR, "surrogate_model.bin"),
        "--metric_save_path", os.path.join(REPO_ROOT, "checkpoints", "phase0", "seed0_metric.json"),
        "--device", "cpu",
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=METADIT_SRC, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("metric.py stderr:", proc.stderr)
        sys.exit(proc.returncode)

    metric_path = os.path.join(REPO_ROOT, "checkpoints", "phase0", "seed0_metric.json")
    with open(metric_path) as f:
        metric = json.load(f)
    print("BASELINE REPRODUCED (official MAE/AAE definition, released weights):")
    for k, v in metric.items():
        print(f"  {k}: {v}")
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0 MetaDiT baseline reproduction")
    parser.add_argument("--skip-shapes", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-metric", action="store_true")
    args = parser.parse_args()

    if not args.skip_shapes:
        verify_shapes()
    if not args.skip_weights:
        verify_weights()
    if not args.skip_metric:
        reproduce_baseline()