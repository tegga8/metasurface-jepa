"""VICReg gradient attribution (B5, calibration spec).

Answers, with numbers: when the checkpointed VICReg objective backpropagates
each component term (L_inv / L_var / L_cov) through ONE REAL MetaDiT batch,
where does gradient actually flow?

For each component, on a FRESH forward pass:
  - total L2 norm of gradients reaching the geometry encoder ("raw" branch)
  - total L2 norm of gradients reaching the objective's VICReg projector
  - total L2 norm of gradients reaching the predictor
  - hard assertion that NO gradient reached the frozen EMA target encoder

No parameter update happens anywhere: gradients are zeroed directly between
components and parameter checksums are compared before/after the whole run.

Usage:
  python scripts/diagnostics/vicreg_gradient_attribution.py \
      --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
      [--device cpu] [--batch-size 8] [--out attribution.json]
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

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from assembly import build_model, load_into_model  # noqa: E402
from data.dataset import MetaDiTDataset, collate_batch  # noqa: E402
from losses.objectives import build_objective  # noqa: E402
from train.engine import restore_ema_state  # noqa: E402


def param_checksum(*modules):
    """Exact float64 checksum over all parameters of the given modules."""
    total = 0.0
    for m in modules:
        for p in m.parameters():
            total += p.detach().double().sum().item()
    return total


def group_grad_norm(params):
    """(total_l2_norm, n_params_with_grad, n_params_total)."""
    sq_sum = 0.0
    n_grad, n_tot = 0, 0
    for p in params:
        n_tot += 1
        if p.grad is not None:
            n_grad += 1
            sq_sum += p.grad.detach().double().pow(2).sum().item()
    return (sq_sum ** 0.5, n_grad, n_tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "milestone_b.yaml"))
    ap.add_argument("--data-root", default=None,
                    help="Override dataset root (expects <root>/split_data/train_set.mat)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional json output path")
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- checkpoint -> live model/objective/EMA (mirrors the eval-only path) ----
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_cfg = ck.get("cfg", cfg)
    spec_path = os.path.join(REPO_ROOT, ck_cfg["weights"]["spectrum"])
    metadit_path = os.path.join(REPO_ROOT, ck_cfg["weights"]["metadit"])
    model = build_model(ck_cfg["model"], spec_path, device=device,
                        init_from_metadit=False, metadit_weights=metadit_path)
    load_into_model(model, ck["model"], device)
    objective_name = ck.get("objective_name", ck_cfg.get("objective", "jepa_vicreg"))
    objective = build_objective(
        objective_name,
        (ck_cfg.get("objective_params", {}) or {}).get(objective_name, {}),
        projector_input_dim=ck_cfg["model"].get("hidden", 384),
    ).to(device)
    objective.load_state_dict(ck["objective_state"])
    restore_ema_state(model, ck.get("ema_state"))

    # ---- one REAL MetaDiT batch ----
    if args.data_root:
        data_root = args.data_root
        train_mat = os.path.join(data_root, "split_data", "train_set.mat")
    else:
        train_mat = os.path.join(REPO_ROOT, ck_cfg["data"]["train_split"])
    ds = MetaDiTDataset(train_mat, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=True, collate_fn=collate_batch)
    G, S = next(iter(loader))
    G, S = G.to(device), S.to(device)

    masker_seed = (ck_cfg.get("mask", {}) or {}).get("mask_seed", 12345)
    from data.mask import BlockMasker
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=masker_seed)
    M = masker.sample(G.cpu(), args.mask_ratio).to(device)

    model.train()
    objective.train()

    groups = {
        "geometry_encoder": lambda: model.geometry_encoder.parameters(),
        "projector": lambda: objective.projector.parameters(),
        "predictor": lambda: model.predictor.parameters(),
    }
    ema_params = list(model.ema.parameters())

    checksum_before = param_checksum(model, objective)
    results = {}
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"objective : {objective_name}")
    print(f"batch     : {tuple(G.shape)} geometries, mask ratio ~{args.mask_ratio}\n")

    header = f"{'component':<12}{'geom_enc L2':>14}{'projector L2':>14}{'predictor L2':>14}{'EMA grads':>11}"
    print(header)
    print("-" * len(header))

    for comp in ("L_inv", "L_var", "L_cov"):
        model.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)

        res = objective(model, G, S, M)
        loss = res["components"][comp]
        if not torch.is_tensor(loss):  # NaN-marked scalar etc.
            raise RuntimeError(f"component {comp} is not a tensor: {loss!r}")
        loss.backward()

        # B5 hard contract: the EMA target must never receive gradients.
        leaked = [i for i, p in enumerate(ema_params) if p.grad is not None]
        assert not leaked, (
            f"{comp}: {len(leaked)} EMA target parameters received gradients — "
            "target-gradient leak invalidates the JEPA setup")

        row = {}
        for gname, get_params in groups.items():
            norm, n_g, n_t = group_grad_norm(get_params())
            row[gname] = {"l2_norm": norm, "n_with_grad": n_g, "n_total": n_t}
        results[comp] = row
        print(f"{comp:<12}"
              f"{row['geometry_encoder']['l2_norm']:>14.4e}"
              f"{row['projector']['l2_norm']:>14.4e}"
              f"{row['predictor']['l2_norm']:>14.4e}"
              f"{'none':>11}")

        model.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)

    checksum_after = param_checksum(model, objective)
    assert checksum_after == checksum_before, (
        "parameter checksum changed during gradient attribution — an "
        "unexpected parameter update occurred")
    print("\n[OK] no EMA target gradients; parameter checksums unchanged "
          "(read-only measurement)")

    if args.out:
        payload = {
            "checkpoint": args.checkpoint,
            "objective_name": objective_name,
            "batch_geometry_shape": list(G.shape),
            "mask_ratio_requested": args.mask_ratio,
            "param_checksum_before": checksum_before,
            "param_checksum_after": checksum_after,
            "ema_gradient_leak": False,
            "components": results,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"json -> {args.out}")


if __name__ == "__main__":
    main()
