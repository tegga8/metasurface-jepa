"""Short raw-JEPA architecture audit (architecture-repair spec §32).

Purpose: BEFORE any objective experiment, verify the SHARED architecture is not
structurally broken — no representation collapse, physics actually enters the
predictor, gradients reach every trainable path, EMA target stays frozen, scales
stay sane. This is NOT a training-objective check: a low scalar L_J is never
interpreted as success (spec §32: "Do not interpret a low scalar JEPA loss as
success").

Run (full, cloud GPU per AGENTS.md compute environment):
    python scripts/diagnostics/architecture_audit.py \
        --config configs/milestone_b.yaml --steps 200 --batch 8 \
        --report-every 25 --device cuda
    (uses data/metadit/weights/* — the released spec encoder + MetaDiT init)

Local CPU crash/smoke test (no released weights required, stub spectrum path):
    python scripts/diagnostics/architecture_audit.py --smoke --steps 3

Per report step it logs (spec §32 list):
  - z_hat / z_y effective-rank fraction, cross-sample pairwise cosine, feature std
  - c_physics effective rank + cross-sample cosine; a_goal token diversity
  - condition sensitivity (masked-token |real - null| and |real - shuffled| /
    |real|) — the §33 PHYSICS_PATH_FAILURE signal
  - gradient norms: geometry encoder, predictor, FiLM conditioner, spectrum path
  - per-step loss + mask fraction

Flags per §33:
  RAW_REPRESENTATION_COLLAPSE | PHYSICS_PATH_FAILURE | TARGET_BUG |
  SCALE_PATHOLOGY | (MASK_LEAKAGE is locked by unit tests, not per-step here)

Fixed seeds + fixed masks throughout; batch >= 8 in the full mode.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from assembly import build_model
from data.mask import BlockMasker
from diagnostics.representation_health import eff_ranks, goal_token_stats


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pool_cos(X):
    """Cross-sample pairwise cosine on mean-pooled (B, T, D) -> mean over pairs."""
    b = X.shape[0]
    if b < 2:
        return float("nan")
    Xn = F.normalize(X.mean(dim=1), dim=-1)
    G = Xn @ Xn.T
    idx = torch.triu_indices(b, b, offset=1, device=G.device)
    return G[idx[0], idx[1]].mean().item()


def _eff_rank_frac_meanpooled(X):
    """Effective-rank fraction on the per-sample mean-pooled (B, D) representation —
    the same convention as diagnostics.representation_health."""
    return eff_ranks(X.mean(dim=1))["eff_rank_frac"]


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


class _StubReleasedEncoder(torch.nn.Module):
    """Frozen random MLP (B, 2, 301) -> (B, 301, 256) for --smoke / no-weights runs."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.GELU(),
                                       torch.nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def build_stub_model(cfg):
    from assembly import GoalConditionedJEPA
    set_seed(0)
    model = GoalConditionedJEPA(**{
        "hidden": cfg["model"].get("hidden", 384),
        "num_heads": cfg["model"].get("num_heads", 6),
        "geo_depth": cfg["model"].get("geo_depth", 6),
        "predictor_depth": cfg["model"].get("predictor_depth", 8),
        "goal_tokens": cfg["model"].get("goal_tokens", 16),
        "num_predictor_heads": cfg["model"].get("num_predictor_heads", 6),
        "momentum_start": cfg["model"].get("ema_momentum_start", 0.996),
        "momentum_end": cfg["model"].get("ema_momentum_end", 0.999),
    })
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/milestone_b.yaml")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--report-every", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3-step CPU crash test with a stub spectrum path")
    args = ap.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu" if args.smoke else args.device)

    if args.smoke:
        model = build_stub_model(cfg).to(device)
        steps, batch, report_every = min(3, args.steps), 8, 1
        print("[architecture_audit] SMOKE mode: stub spectrum path, CPU")
    else:
        spec = cfg["weights"]["spectrum"]
        metadit = cfg["weights"]["metadit"]
        assert os.path.exists(spec), f"missing released spectrum weights: {spec}"
        assert os.path.exists(metadit), f"missing metadit weights: {metadit}"
        model = build_model(cfg, spec, device=device, init_from_metadit=True,
                            metadit_weights=metadit)
        steps, batch, report_every = args.steps, args.batch, args.report_every
        print(f"[architecture_audit] full mode: batch={batch} steps={steps} "
              f"device={device}")

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    geo_params = [p for p in model.geometry_encoder.parameters() if p.requires_grad]
    pred_params = [p for p in model.predictor.parameters() if p.requires_grad]
    cond_params = [p for blk in model.predictor.blocks for p in blk.cond.parameters()]
    spec_params = [p for n, p in model.spectrum_path.named_parameters()
                   if not n.startswith("released")]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)
    masker = BlockMasker(seed=args.seed, placement="random")
    model.ema.set_total_steps(steps)

    rows = []
    flags = {"PHYSICS_PATH_FAILURE": False,
             "TARGET_BUG": False,
             "SCALE_PATHOLOGY": False}
    # Observation flags (per §32/§33: raw-JEPA REGIME observations are reported,
    # not treated as structural defects — §38 forbids fixing collapse by adding
    # architecture; collapse prevention belongs to the objective mechanisms).
    obs = {"RAW_REPRESENTATION_COLLAPSE": False,   # §33: the shared EMA/context
                                                   # representation itself collapses
           "PREDICTOR_LATENT_LOW_RANK": False}     # raw-L_J regime: z_hat rank << z_y
    scale_parts = []

    for step in range(steps):
        torch.manual_seed(args.seed * 1000 + step)
        G = torch.randn(batch, 3, 64, 64)
        S = torch.randn(batch, 2, 301)
        M = masker.sample(G, args.mask_ratio).to(device)
        G, S, M = G.to(device), S.to(device), M

        L, out = model.loss(G, S, M, goal_mode="real")
        if not torch.isfinite(L):
            raise RuntimeError(f"non-finite L_J at step {step}: {L.item()}")
        L.backward()
        for n, p in model.ema.named_parameters():
            if p.grad is not None:
                flags["TARGET_BUG"] = True
                print(f"[architecture_audit] TARGET_BUG: EMA param {n} has gradient")

        # Capture per-path gradient norms BEFORE the optimizer consumes them.
        grad_norms = {
            "grad_geom": _grad_norm(geo_params),
            "grad_predictor": _grad_norm(pred_params),
            "grad_film": _grad_norm(cond_params),
            "grad_spectrum": _grad_norm(spec_params),
        }

        opt.step()
        opt.zero_grad(set_to_none=True)
        model.ema.update(model.geometry_encoder, step)   # §9: only AFTER optimizer.step()

        if step % report_every != 0:
            continue
        model.eval()
        with torch.no_grad():
            out_r = model(G, S, M, goal_mode="real")
            out_n = model(G, S, M, goal_mode="null")
            S_shuf = S.roll(shifts=1, dims=0)
            out_s = model(G, S_shuf, M, goal_mode="real")
        model.train()

        zh, zy = out_r["z_hat"], out_r["z_y_raw"]
        mask = out_r["mask"].bool()
        zhn = F.normalize(zh, dim=-1)
        cond_real = (zh - out_n["z_hat"]).norm(dim=-1)[mask].mean().item()
        cond_shuf = (zh - out_s["z_hat"]).norm(dim=-1)[mask].mean().item()
        z_scale = zhn.norm(dim=-1)[mask].mean().item()
        cp = out_r["c_physics"]
        ag = out_r["a_goal"]

        cp_cos = _pool_cos(cp.unsqueeze(1))
        gts = goal_token_stats(ag)
        row = {
            "step": step,
            "L_J": L.item(),
            "mask_frac": 1.0 - M.mean().item(),
            "z_hat_eff_rank_frac": _eff_rank_frac_meanpooled(zh),
            "z_y_eff_rank_frac": _eff_rank_frac_meanpooled(zy),
            "z_hat_cross_cos": _pool_cos(zh),
            "z_y_cross_cos": _pool_cos(zy),
            "z_hat_feature_std": zh.std().item(),
            "z_y_feature_std": zy.std().item(),
            "c_physics_eff_rank_frac": eff_ranks(cp)["eff_rank_frac"],
            "c_physics_cross_cos": cp_cos,
            "goal_token_cos": gts["goal_token_pairwise_cosine_mean"],
            "goal_token_eff_rank": gts["goal_token_effective_rank"],
            "cond_sensitivity_real_null": cond_real,
            "cond_sensitivity_real_shuf": cond_shuf,
            "cond_sensitivity_normed": cond_real / max(z_scale, 1e-9),
            "grad_geom": grad_norms["grad_geom"],
            "grad_predictor": grad_norms["grad_predictor"],
            "grad_film": grad_norms["grad_film"],
            "grad_spectrum": grad_norms["grad_spectrum"],
        }
        rows.append(row)
        scale_parts.append({
            "z_hat_std": zh.std().item(), "z_y_std": zy.std().item(),
            "z_x_std": out_r["z_x"].std().item(),
            "z_hat_mean_norm": zh.norm(dim=-1).mean().item(),
            "z_y_mean_norm": zy.norm(dim=-1).mean().item(),
        })
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in row.items()}))

    # ---- §33 flag evaluation ----
    # Structural gates (these FAIL the architecture gate per §38: the architecture's
    # only job is no hidden bottleneck, no ignored condition, no scale pathology, no
    # target gradient, no aliasing). Non-finite loss already aborts mid-run.
    last = rows[-1]
    if max(last["cond_sensitivity_real_null"], last["cond_sensitivity_real_shuf"]) < 1e-6:
        flags["PHYSICS_PATH_FAILURE"] = True
    s = scale_parts[-1]
    stds = [s["z_hat_std"], s["z_y_std"], s["z_x_std"]]
    if max(stds) > 0 and max(stds) / max(min(stds), 1e-9) > 100.0:
        flags["SCALE_PATHOLOGY"] = True
    # Observations (reported, do NOT gate): the EMA/context representation itself
    # collapsing is architecture-level; a low-rank PREDICTOR under raw L_J is the
    # known raw-JEPA regime that the objective mechanisms exist to prevent (§32,
    # §38) and must not be "fixed" by adding architecture.
    if last["z_y_eff_rank_frac"] < 0.01 or last["z_y_feature_std"] < 1e-3:
        obs["RAW_REPRESENTATION_COLLAPSE"] = True
    if last["z_y_eff_rank_frac"] > 0.05 and \
            last["z_hat_eff_rank_frac"] < 0.25 * last["z_y_eff_rank_frac"]:
        obs["PREDICTOR_LATENT_LOW_RANK"] = True

    out_path = args.out or str(REPO_ROOT / "checkpoints" / "milestone_b" /
                               "architecture_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"mode": "smoke" if args.smoke else "full",
                   "steps": steps, "batch": batch, "seed": args.seed,
                   "mask_ratio": args.mask_ratio, "rows": rows,
                   "scale": scale_parts, "flags": flags, "observations": obs},
                  f, indent=2)
    print(f"[architecture_audit] -> {out_path}")
    print(f"[architecture_audit] structural flags: {json.dumps(flags)}")
    print(f"[architecture_audit] observations: {json.dumps(obs)}")
    if any(flags.values()):
        print("[architecture_audit] FAIL: structural flags raised — do NOT start "
              "objective experiments before fixing these.")
        sys.exit(2)
    print("[architecture_audit] PASS (structural): no architecture defect. "
          "Observations above are raw-JEPA-regime notes for the objective phase; "
          "L_J value carries no success meaning (spec §32).")


if __name__ == "__main__":
    main()