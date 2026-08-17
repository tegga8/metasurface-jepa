"""Milestone A - released-weight embedding sanity checks.

Standalone CLI, inference-only, CPU-friendly. Scope per operator decision
(checkpoints/milestone_a/REPORT.md): the §1.4 CLIP checkpoint was never released
(verified against GitHub: 0 releases/tags, 1 branch; HF: weights/ holds exactly
metadit-small.bin, surrogate_model.bin, spec_encoder.pth), so this script checks
what released weights actually support:

  1. S -> z_S through the released spectrum encoder (shape/stats/determinism).
  2. Consistency of that encoder with the encoder embedded in the released DiT.
  3. Geometry-only ViT-feature clustering sanity (param-binned 2x2x2 cells and
     topology quartiles) - the operator-approved proxy for the unreleased
     geometry-side CLIP embeddings.

Usage: python scripts/eval/eval_milestone_a.py [--n-samples 1500] [--seed 0]
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
SRC_DIR = os.path.join(REPO_ROOT, "src")
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
from scipy import io
from scipy.ndimage import label as label_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from model.dit import DIT_MODEL
from model.spec_encoder import VanillaSpectrumEncoder
from datapipe import FreeFormDataset
from diffusion import create_diffusion
from encoders.spectrum_encoder import ReleasedSpectrumEncoder

DATA_DIR = os.path.join(REPO_ROOT, "data", "metadit")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
SPLIT_DIR = os.path.join(DATA_DIR, "split_data")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "milestone_a")

PARAM_NAMES = ["l_lattice", "h_atom", "r_atom"]


def load_dit():
    path = os.path.join(WEIGHTS_DIR, "metadit-small.bin")
    diffusion = create_diffusion("500", learn_sigma=False)
    model = DIT_MODEL["metadit_s"](diffusion=diffusion, condition_channel=301)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval()
    return model


def sample_test_rows(n, seed):
    raw = io.loadmat(os.path.join(SPLIT_DIR, "test_set.mat"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(raw["pattern"].shape[-1], size=n, replace=False)
    ff = FreeFormDataset.__new__(FreeFormDataset)
    ff.data = raw
    rows = [ff.__getitem__(i) for i in idx]
    grids = torch.stack([r["inputs"] for r in rows])
    conds = torch.stack([r["condition"] for r in rows])
    params = torch.tensor(raw["parameter"][idx], dtype=torch.float32)
    patterns = torch.from_numpy(raw["pattern"][:, :, idx]).permute(2, 0, 1)
    return grids, conds, params, patterns, idx


def quantile_bins(values, n_bins):
    bounds = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    bounds[0] -= 1e-12
    return np.digitize(values, bounds[1:-1]), bounds


def topology_descriptors(patterns):
    n = patterns.shape[0]
    fill = patterns.reshape(n, -1).float().mean(dim=1).numpy()
    n_comp = np.zeros(n, dtype=int)
    n_holes = np.zeros(n, dtype=int)
    for i in range(n):
        structure = np.ones((3, 3), dtype=int)
        lab, n_lab = label_components(patterns[i].numpy(), structure=structure)
        n_comp[i] = n_lab
        inv = 1 - patterns[i].numpy()
        lab_inv, n_inv = label_components(inv, structure=structure)
        border = set(np.unique(np.concatenate([lab_inv[0, :], lab_inv[-1, :],
                                               lab_inv[:, 0], lab_inv[:, -1]])))
        n_holes[i] = sum(1 for l in range(1, n_inv + 1) if l not in border)
    return {"fill_fraction": fill, "n_components": n_comp.astype(float),
            "n_holes": n_holes.astype(float)}


def dit_geometry_features(dit, grids, blocks_to_extract, chunk=32):
    n = grids.shape[0]
    features = {b: [] for b in blocks_to_extract}
    zeros = torch.zeros(n, 301, 2)
    t = torch.zeros(n, dtype=torch.long)
    start = time.time()
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = grids[i:i + chunk]
            t_b = t[i:i + chunk]
            y = zeros[i:i + chunk].transpose(1, 2).contiguous()
            x = dit.x_embedder(x) + dit.pos_embed
            t_emb = dit.t_embedder(t_b)
            y_emb = dit.y_embedder(y, train=False)
            c = t_emb + y_emb.mean(1)
            for li, block in enumerate(dit.blocks):
                x = block(x, c, y_emb)
                if li in blocks_to_extract:
                    features[li].append(x.mean(dim=1))
            if (i // chunk + 1) % 4 == 0:
                print(f"    features: {i + chunk}/{n} ({(time.time() - start) / (i // chunk + 1):.1f} s/chunk)")
    for li in blocks_to_extract:
        features[li] = torch.cat(features[li])
    return features


def clustering_report(name, features, labels, k, n_known):
    feat = features.numpy()
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    if len(feat) <= k + 1:
        return {"kmeans_ARI": float("nan"), "silhouette_kmeans": float("nan"),
                "silhouette_known": float("nan")}
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feat)
    ari = adjusted_rand_score(labels, km.labels_)
    sil_km = silhouette_score(feat, km.labels_)
    n_lab = len(np.unique(labels))
    sil_gt = silhouette_score(feat, labels) if n_known > 1 and n_lab > 1 else float("nan")
    print(f"  {name}: kmeans ARI vs groups={ari:.3f}  "
          f"silhouette(kmeans)={sil_km:.3f}  silhouette(known-groups)={sil_gt:.3f}")
    return {"kmeans_ARI": float(ari), "silhouette_kmeans": float(sil_km),
            "silhouette_known": float(sil_gt)}


def main():
    parser = argparse.ArgumentParser(description="Milestone A embedding sanity checks")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("[1/3] S -> z_S through the released spectrum encoder")
    print("=" * 70)
    spec_model = ReleasedSpectrumEncoder(os.path.join(WEIGHTS_DIR, "spec_encoder.pth"))
    print(f"  loaded spec_encoder.pth (prefix 'context_encoder.' stripped) via "
          f"src/encoders/spectrum_encoder.py, "
          f"params={sum(p.nelement() for p in spec_model.parameters()):,}")
    grids, conds, params, patterns, idx = sample_test_rows(64, args.seed)
    out1 = spec_model(conds)
    out2 = spec_model(conds)
    assert out1.shape == (64, 301, 256)
    det = (out1 - out2).abs().max().item()
    norms = out1.norm(dim=2)
    print(f"  forward(64, 2, 301) -> {tuple(out1.shape)} OK")
    print(f"  determinism: two identical forwards max|d| = {det:.3e}")
    print(f"  token-norm stats: mean={norms.mean():.3f} std={norms.std():.3f} "
          f"min={norms.min():.3f} max={norms.max():.3f}")
    sim = torch.nn.functional.cosine_similarity(
        out1.mean(1)[:, None, :], out1.mean(1)[None, :, :], dim=2)
    off = sim.fill_diagonal_(0).abs().mean().item()
    print(f"  cross-sample mean |cos(z_S_i, z_S_j)| (i!=j) = {off:.4f} "
          "(distinct spectra should not be near-identical)")

    print("\n" + "=" * 70)
    print("[2/3] Consistency with the encoder inside the released DiT")
    print("=" * 70)
    dit = load_dit()
    dit_spec = VanillaSpectrumEncoder()
    dit_keys = {k[len("y_embedder.encoder."):]: v
                for k, v in dit.state_dict().items()
                if k.startswith("y_embedder.encoder.")}
    dit_spec.load_state_dict(dit_keys, strict=True)
    dit_spec.eval()
    w_max = 0.0
    for k in dit_spec.state_dict():
        a = spec_model.encoder.state_dict()[k]
        b = dit_spec.state_dict()[k]
        if a.shape != b.shape:
            print(f"  MISMATCHED SHAPE on key {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
        w_max = max(w_max, (a - b).float().abs().max().item())
    print(f"  weight tensors (spec_encoder.pth vs y_embedder.encoder in metadit-small.bin): "
          f"max abs diff = {w_max:.3e}")
    o_a = spec_model(conds)
    o_b = dit_spec(conds)
    print(f"  output diff on 64 spectra: max abs = {(o_a - o_b).abs().max().item():.3e}")
    print("  (identity expected: released spec_encoder.pth IS the frozen encoder "
          "the released DiT conditions on)")

    print("\n" + "=" * 70)
    print(f"[3/3] Geometry-only ViT-feature clustering sanity (N={args.n_samples})")
    print("=" * 70)
    grids, conds, params, patterns, idx = sample_test_rows(args.n_samples, args.seed)
    p = params.numpy()
    print(f"  param ranges: l_lattice [{p[:, 0].min():.4f}, {p[:, 0].max():.4f}]  "
          f"h_atom [{p[:, 1].min():.4f}, {p[:, 1].max():.4f}]  "
          f"r_atom [{p[:, 2].min():.4f}, {p[:, 2].max():.4f}]")
    param_labels = np.zeros(len(p), dtype=int)
    bin_edges = []
    for col in range(3):
        lab, bounds = quantile_bins(p[:, col], 2)
        param_labels = param_labels * 2 + lab
        bin_edges.append(np.round(bounds[1], 4))
    cells, counts = np.unique(param_labels, return_counts=True)
    print(f"  quantile 2x2x2 param cells: edges {bin_edges}; occupancy {len(cells)}/8 "
          f"cells, sizes {sorted(counts.tolist())}")

    topo = topology_descriptors(patterns)
    topo_groups = {}
    for name, vals in topo.items():
        lab, bounds = quantile_bins(vals, 4)
        topo_groups[name] = lab
        print(f"  topology {name}: quartiles {np.round(bounds[1:-1], 3)}")

    print("  extracting DiT features (zero spectrum condition, t=0, clean geometry)...")
    feats = dit_geometry_features(dit, grids, blocks_to_extract=[6, 11])
    results = {}
    for li, feat in feats.items():
        print(f"  block {li} (mean-pooled 256 tokens, 384-dim):")
        row = {"param8": clustering_report(
            "param 2x2x2 cells", feat, param_labels, k=8, n_known=8)}
        for name, lab in topo_groups.items():
            row[name] = clustering_report(
                f"topology {name} quartiles", feat, lab, k=4, n_known=4)
        results[f"block{li}"] = row

    out = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "param_quantile_edges": bin_edges,
        "param_cell_occupancy": {int(c): int(cnt) for c, cnt in zip(cells, counts)},
        "topology_quartiles": {k: np.round(quantile_bins(v, 4)[1][1:-1], 4).tolist()
                               for k, v in topo.items()},
        "clustering": results,
    }
    out_path = os.path.join(OUT_DIR, "milestone_a_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
