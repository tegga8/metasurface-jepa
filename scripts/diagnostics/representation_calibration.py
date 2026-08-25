"""Representation-health calibration runner (B8, calibration spec).

One read-only pass that answers the operator's calibration questions with
numbers instead of gate verdicts:

  1. Is the trained EMA-target representation still degenerate/collapsed?
     -> mean-pooled pairwise cos (mean/p05), effective rank (+fraction),
        participation, top-eig fraction vs released / random / collapsed refs.
  2. Is it merely mimicking the released MetaDiT encoder?
     -> same table side by side; distance is reported, not judged.
  3. Does the raw EMA representation carry physical-parameter information?
     -> closed-form linear-probe R^2 for l_lattice / h_atom / r_atom
        (identical deterministic split across all compared encoders).
  4. Where does VICReg gradient actually flow?
     -> pointer to vicreg_gradient_attribution.py (run separately); this
        script reports raw-vs-projector statistics only.

Read-only contract: no parameter update anywhere — parameter checksums are
taken before and after and asserted identical. The collapse GATE itself
(classify_health/classify_failure_mode thresholds) is intentionally NOT
weakened or re-derived here (B7).

Usage:
  python scripts/diagnostics/representation_calibration.py \
      --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
      --config configs/milestone_b.yaml --device cpu --max-geoms 512 \
      --out checkpoints/milestone_b/representation_calibration.json
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for pth in (REPO_ROOT, SRC_DIR):
    if pth not in sys.path:
        sys.path.insert(0, pth)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from assembly import build_model, load_into_model  # noqa: E402
from data.dataset import MetaDiTDataset, collate_batch  # noqa: E402
from encoders.geometry_encoder import GeometryEncoder  # noqa: E402
from losses.objectives import build_objective  # noqa: E402

from diagnostics.representation_health import (  # noqa: E402
    COLLAPSED_ANCHOR, grouped_view, token_space_stats,
)
from diagnostics.representation_probes import geometry_linear_probes  # noqa: E402


def _checksum(*modules):
    total = 0.0
    for m in modules:
        for p in m.parameters():
            total += p.detach().double().sum().item()
    return total


def build_random_calibration_encoder(hidden, heads, depth, seed, device):
    """Build a random-init geometry encoder without mutating global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GeometryEncoder(
            hidden=hidden, num_heads=heads, depth=depth,
        ).to(device)


def collect_embeddings(enc, geoms, device):
    """Token embeddings (N, T, D) over a FIXED geometry list, in order."""
    expected_device = torch.device(device)
    try:
        actual_device = next(enc.parameters()).device
    except StopIteration:
        actual_device = expected_device
    if actual_device != expected_device:
        raise RuntimeError(
            f"Encoder is on {actual_device}, but calibration requested {expected_device}"
        )
    was_training = getattr(enc, "training", False)
    if hasattr(enc, "eval"):
        enc.eval()
    try:
        with torch.no_grad():
            return torch.cat([enc(G.to(device)).cpu() for G in geoms], dim=0)
    finally:
        if hasattr(enc, "train") and was_training:
            enc.train()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "milestone_b.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-geoms", type=int, default=512)
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- load checkpoint into live modules (read-only) ----
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_cfg = ck.get("cfg", cfg)
    spec_path = os.path.join(REPO_ROOT, ck_cfg["weights"]["spectrum"])
    metadit_path = os.path.join(REPO_ROOT, ck_cfg["weights"]["metadit"])

    model = build_model(ck_cfg["model"], spec_path, device=device,
                        init_from_metadit=False, metadit_weights=metadit_path)
    load_into_model(model, ck["model"], device)
    from train.engine import restore_ema_state
    restore_ema_state(model, ck.get("ema_state"))

    objective = None
    objective_name = ck.get("objective_name", ck_cfg.get("objective", "jepa_vicreg"))
    if ck.get("objective_state"):
        objective = build_objective(
            objective_name,
            (ck_cfg.get("objective_params", {}) or {}).get(objective_name, {}),
            projector_input_dim=ck_cfg["model"].get("hidden", 384),
        ).to(device)
        objective.load_state_dict(ck["objective_state"])
        objective.eval()

    checksum_before = _checksum(model, objective) if objective else _checksum(model)

    # ---- fixed geometry set + parameters (SAME order for every encoder) ----
    data_root = args.data_root or os.path.join(REPO_ROOT, cfg["data"]["val_split"])
    val_mat = (os.path.join(data_root, "split_data", "val_set.mat")
               if os.path.isdir(data_root) else data_root)
    ds = MetaDiTDataset(val_mat)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                        drop_last=False, collate_fn=collate_batch)
    geoms, params = [], []
    for G, S in loader:
        geoms.append(G)
        if sum(g.shape[0] for g in geoms) >= args.max_geoms:
            break
    geoms = geoms[:args.max_geoms]
    n_needed = sum(g.shape[0] for g in geoms)
    all_params = ds.data["parameter"][:n_needed] \
        if hasattr(ds, "data") else None
    if all_params is None:
        raise RuntimeError("dataset does not expose raw parameter array")
    params = np.asarray(all_params, dtype=np.float64)

    # ---- representations ----
    X_trained = collect_embeddings(model.ema, geoms, device)

    hidden = ck_cfg["model"].get("hidden", 384)
    heads = ck_cfg["model"].get("num_heads", 6)
    depth = ck_cfg["model"].get("geo_depth", 6)
    released = GeometryEncoder(
        hidden=hidden, num_heads=heads, depth=depth,
    ).to(device)
    released.init_from_metadit(torch.load(metadit_path, map_location="cpu"),
                               blocks_to_take=depth)
    X_released = collect_embeddings(released, geoms, device)

    random_enc = build_random_calibration_encoder(
        hidden, heads, depth, args.probe_seed, device,
    )
    X_random = collect_embeddings(random_enc, geoms, device)

    # ---- stats + grouped views ----
    stats = {
        "trained_ema": token_space_stats(X_trained),
        "released_vit": token_space_stats(X_released),
        "random_init": token_space_stats(X_random),
    }
    grouped = {name: grouped_view(s) for name, s in stats.items()}

    # raw-vs-projector measurement through the CHECKPOINT's own projector
    proj_stats = None
    if objective is not None:
        with torch.no_grad():
            X_proj = objective.projector(X_trained.to(device)).cpu()
        proj_stats = token_space_stats(X_proj)

    # ---- B3 probes: identical split for all three encoders ----
    probes = {}
    probe_inputs = {"trained_ema": X_trained, "released_vit": X_released,
                    "random_init": X_random}
    probes = {name: geometry_linear_probes(
                  X, params, ridge_lambda=args.ridge_lambda,
                  val_fraction=args.val_fraction, seed=args.probe_seed)
              for name, X in probe_inputs.items()}

    checksum_after = _checksum(model, objective) if objective else _checksum(model)
    assert checksum_after == checksum_before, (
        "parameter checksum changed during calibration — this must be a "
        "strictly read-only measurement")

    # ---- report ----
    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"step/epoch : {ck.get('step')} / {ck.get('epoch')}")
    print(f"geometries : {stats['trained_ema']['n_geoms']}")
    print(f"[OK] read-only verified (param checksum unchanged)\n")

    hdr = (f"{'metric':<26}{'EMA target':>14}{'released ViT':>14}"
           f"{'random init':>14}{'collapsed':>14}")
    print(hdr)
    print("-" * len(hdr))

    def cell(d, key):
        v = d
        try:
            for part in key.split("."):
                v = v[part]
        except (KeyError, TypeError):
            return f"{'n/a':>14}"
        if isinstance(v, float) and v != v:
            return f"{'nan':>14}"
        return f"{v:>14.6g}" if isinstance(v, (int, float)) else f"{'n/a':>14}"

    _ANCHOR_MAP = {"pairwise_cos.mean": "pairwise_cos",
                   "pairwise_cos.p05": "pairwise_p05"}
    rows = [("token_var", "token var"), ("token_std", "token std"),
            ("sample_var", "sample var"), ("sample_std", "sample std"),
            ("pairwise_cos.mean", "pairwise cos mean"),
            ("pairwise_cos.p05", "pairwise cos p05"),
            ("same_token_cos", "same-token cos"),
            ("eff_rank_frac", "eff rank frac"),
            ("eff_rank_unnorm", "entropy eff rank"),
            ("participation", "participation"),
            ("top_eig_frac", "top eig frac")]
    for key, label in rows:
        ak = _ANCHOR_MAP.get(key, key)
        anchor = cell(COLLAPSED_ANCHOR, ak)
        print(f"{label:<26}"
              f"{cell(stats['trained_ema'], key)}"
              f"{cell(stats['released_vit'], key)}"
              f"{cell(stats['random_init'], key)}"
              f"{anchor}")

    print("\nlinear-probe R^2 (physical parameters from pooled representation;"
          " identical split):")
    phdr = (f"{'encoder':<16}{'l_lattice':>12}{'h_atom':>12}{'r_atom':>12}"
            f"{'mean R2':>12}")
    print(phdr)
    print("-" * len(phdr))
    for name in ("trained_ema", "released_vit", "random_init"):
        pr = probes[name]
        print(f"{name:<16}{pr['l_lattice_r2']:>12.4f}{pr['h_atom_r2']:>12.4f}"
              f"{pr['r_atom_r2']:>12.4f}{pr['mean_r2']:>12.4f}")

    if proj_stats is not None:
        print(f"\nraw-vs-projector ({objective_name} projector, measurement only):")
        print(f"{'metric':<26}{'raw EMA':>14}{'projected':>14}")
        print("-" * 54)
        for label, k_raw, k_proj in [
                ("eff rank frac", "eff_rank_frac", "eff_rank_frac"),
                ("entropy eff rank", "eff_rank_unnorm", "eff_rank_unnorm"),
                ("token std", "token_std", "token_std"),
                ("pairwise cos p05", "pairwise_cos.p05", "pairwise_cos.p05"),
                ("same-token cos", "same_token_cos", "same_token_cos")]:
            print(f"{label:<26}{cell(stats['trained_ema'], k_raw)}"
                  f"{cell(proj_stats, k_proj)}")

    print("\n(no auto-verdict is issued — see "
          "checkpoints/milestone_b/REPRESENTATION_CALIBRATION_REPORT.md)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "step": ck.get("step"), "epoch": ck.get("epoch"),
            "objective_name": objective_name,
            "n_geoms": int(stats["trained_ema"]["n_geoms"]),
            "stats": stats,
            "grouped": grouped,
            "proj_vs_raw": proj_stats,
            "probes": probes,
            "collapsed_anchor": COLLAPSED_ANCHOR,
            "read_only_verified": True,
            "probe_settings": {"seed": args.probe_seed,
                               "ridge_lambda": args.ridge_lambda,
                               "val_fraction": args.val_fraction},
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\njson -> {args.out}")


if __name__ == "__main__":
    main()
