"""B9 — Representation calibration smoke tests.

Covers:
1. grouped_view is a pure view (no recomputation) of an existing token_space_stats dict.
2. identical geometry subset: two encoders evaluated on the same geoms/order produce stats
   dicts with the same n_geoms.
3. mean-pooled uses X.mean(dim=1) — calling with (N, T, D) and pre-pooled (N, D) gives
   the same result for mean_pooled keys.
4. same-token fixed positions: same_token_cos is invariant to column permutation
   (relabeling spatial tokens) of a (N, T, D) tensor.
5. geometry_linear_probes are finite and deterministic (same inputs -> identical R^2).
6. VICReg gradient attribution: gradient norms finite for each component, EMA gradients
   always None, parameter checksums unchanged.
"""

import os
import sys
import tempfile

import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def test_grouped_view_is_pure_view():
    from diagnostics.representation_health import token_space_stats, grouped_view
    X = torch.randn(64, 16, 128)
    flat = token_space_stats(X)
    gv = grouped_view(flat)
    # mean_pooled keys must equal the expected set
    assert set(gv.keys()) == {"mean_pooled", "token_level", "n_geoms"}, gv.keys()
    assert set(gv["mean_pooled"].keys()) == {
        "pairwise_cos", "eff_rank_unnorm", "eff_rank_frac",
        "participation", "top_eig_frac"}
    assert set(gv["token_level"].keys()) == {
        "token_var", "token_std", "same_token_cos"}
    assert gv["n_geoms"] == 64
    # values are aliases, not copies — same object
    assert gv["token_level"]["token_std"] is flat["token_std"]
    assert gv["mean_pooled"]["eff_rank_frac"] is flat["eff_rank_frac"]


def test_identical_subset_same_ng():
    """Two encoders on the same geoms produce same n_geoms."""
    from diagnostics.representation_health import encoder_stats
    from encoders.geometry_encoder import GeometryEncoder
    geoms = [torch.randn(8, 3, 64, 64) for _ in range(3)]
    enc1 = GeometryEncoder(hidden=64, num_heads=4, depth=1)
    enc2 = GeometryEncoder(hidden=64, num_heads=4, depth=1)
    s1 = encoder_stats(enc1, geoms, torch.device("cpu"), 16)
    s2 = encoder_stats(enc2, geoms, torch.device("cpu"), 16)
    assert s1["n_geoms"] == s2["n_geoms"], (s1["n_geoms"], s2["n_geoms"])
    assert s1["n_geoms"] == 16


def test_mean_pooled_matches_manual_pooling():
    """token_space_stats' mean_pooled keys (pairwise_cos, eff_rank, etc.) must
    equal those computed on X.mean(dim=1) directly — the pooling is a step
    inside token_space_stats, not a property of its (N, T, D) input."""
    from diagnostics.representation_health import (
        token_space_stats, pairwise_cos_stats, eff_ranks, same_token_cos)
    X = torch.randn(32, 16, 128)
    s = token_space_stats(X)
    pooled = X.mean(dim=1)  # (32, 128)
    p_pooled = pairwise_cos_stats(pooled)
    e_pooled = eff_ranks(pooled)
    assert abs(s["eff_rank_frac"] - e_pooled["eff_rank_frac"]) < 1e-6
    assert abs(s["eff_rank_unnorm"] - e_pooled["eff_rank_unnorm"]) < 1e-6
    assert abs(s["participation"] - e_pooled["participation"]) < 1e-6
    for pk in ("mean", "p05", "min"):
        assert abs(s["pairwise_cos"][pk] - p_pooled[pk]) < 1e-6


def test_same_token_cos_column_permutation_invariant():
    """same_token_cos is a cross-sample statistic at each spatial position — it
    should be invariant to relabeling the spatial positions (column permutation
    over the T axis)."""
    from diagnostics.representation_health import same_token_cos
    X = torch.randn(8, 16, 64)
    base = same_token_cos(X)
    perm = torch.randperm(16)
    permuted = X[:, perm, :]
    permuted_val = same_token_cos(permuted)
    # Both should be the same value (average over all token positions is invariant)
    # But the per-position values change; only the mean is invariant
    assert abs(base - permuted_val) < 1e-6, (base, permuted_val)


def test_geometry_linear_probes_finite_and_deterministic():
    from diagnostics.representation_probes import geometry_linear_probes
    N, T, D = 64, 16, 128
    X = torch.randn(N, T, D)
    params = np.random.RandomState(42).randn(N, 3).astype(np.float64)
    r1 = geometry_linear_probes(X, params, ridge_lambda=1.0, seed=7)
    r2 = geometry_linear_probes(X, params, ridge_lambda=1.0, seed=7)
    for key in ("l_lattice_r2", "h_atom_r2", "r_atom_r2", "mean_r2"):
        assert key in r1, f"missing key: {key}"
        assert isinstance(r1[key], float), f"{key} not float: {type(r1[key])}"
        assert r1[key] == r1[key], f"{key} is NaN"  # NaN check (v != v)
        assert r1[key] == r2[key], f"{key} not deterministic"


def test_vicreg_gradient_attribution_ema_no_grads():
    """Build a tiny model + objective, run one forward/backward, verify EMA params
    have no gradients and parameter checksums unchanged."""
    from assembly import build_model
    from losses.objectives import build_objective
    from data.mask import BlockMasker

    hidden = 64
    model_cfg = {
        "variant": "jepa", "patch_size": 4, "token_grid": 16,
        "hidden": hidden, "num_heads": 4, "num_predictor_heads": 2,
        "geo_depth": 1, "predictor_depth": 1,
        "goal_tokens": 16, "num_goal_heads": 4,
        "ema_momentum_start": 0.99, "ema_momentum_end": 0.999,
        "init_from_metadit": False,
    }
    weights_dir = os.path.join(REPO_ROOT, "data", "metadit", "weights")
    spec_path = os.path.join(weights_dir, "spec_encoder.pth")
    model = build_model(model_cfg, spec_path, device=torch.device("cpu"),
                        init_from_metadit=False,
                        metadit_weights=os.path.join(weights_dir, "metadit-small.bin"))
    objective = build_objective(
        "jepa_vicreg",
        {"projector": {"input_dim": hidden, "hidden_dim": hidden,
                       "output_dim": hidden},
         "lambda_inv": 25, "lambda_var": 25, "lambda_cov": 1},
        projector_input_dim=hidden,
    )
    model.train()
    objective.train()

    masker = BlockMasker(placement="random", seed=12345)
    G = torch.randn(4, 3, 64, 64)
    S = torch.randn(4, 2, 301)
    M = masker.sample(G, 0.5)

    # checksum before
    def _cs():
        return sum(p.detach().double().sum().item()
                   for m in (model, objective) for p in m.parameters())
    cs_before = _cs()

    ema_params = list(model.ema.parameters())
    for comp in ("L_inv", "L_var", "L_cov"):
        model.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)
        res = objective(model, G, S, M)
        loss = res["components"][comp]
        loss.backward()
        # EMA must not have gradients
        leaked = [i for i, p in enumerate(ema_params) if p.grad is not None]
        assert not leaked, f"EMA gradients after {comp}: {leaked}"

    assert _cs() == cs_before, "parameters mutated during gradient attribution"


def test_vicreg_gradient_attribution_grad_norms_finite():
    from assembly import build_model
    from losses.objectives import build_objective
    from data.mask import BlockMasker

    hidden = 64
    model_cfg = {
        "variant": "jepa", "patch_size": 4, "token_grid": 16,
        "hidden": hidden, "num_heads": 4, "num_predictor_heads": 2,
        "geo_depth": 1, "predictor_depth": 1,
        "goal_tokens": 16, "num_goal_heads": 4,
        "ema_momentum_start": 0.99, "ema_momentum_end": 0.999,
        "init_from_metadit": False,
    }
    weights_dir = os.path.join(REPO_ROOT, "data", "metadit", "weights")
    spec_path = os.path.join(weights_dir, "spec_encoder.pth")
    model = build_model(model_cfg, spec_path, device=torch.device("cpu"),
                        init_from_metadit=False,
                        metadit_weights=os.path.join(weights_dir, "metadit-small.bin"))
    objective = build_objective(
        "jepa_vicreg",
        {"projector": {"input_dim": hidden, "hidden_dim": hidden,
                       "output_dim": hidden},
         "lambda_inv": 25, "lambda_var": 25, "lambda_cov": 1},
        projector_input_dim=hidden,
    )
    model.train()
    objective.train()

    masker = BlockMasker(placement="random", seed=12345)
    G = torch.randn(4, 3, 64, 64)
    S = torch.randn(4, 2, 301)
    M = masker.sample(G, 0.5)

    groups = {
        "geometry_encoder": list(model.geometry_encoder.parameters()),
        "projector": list(objective.projector.parameters()),
        "predictor": list(model.predictor.parameters()),
    }

    for comp in ("L_inv", "L_var", "L_cov"):
        model.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)
        res = objective(model, G, S, M)
        loss = res["components"][comp]
        loss.backward()
        for gname, params in groups.items():
            sq = sum(p.grad.detach().pow(2).sum().item()
                     for p in params if p.grad is not None)
            norm = sq ** 0.5
            assert np.isfinite(norm), f"{comp}/{gname} grad norm is not finite: {norm}"


if __name__ == "__main__":
    test_grouped_view_is_pure_view()
    test_identical_subset_same_ng()
    test_mean_pooled_matches_manual_pooling()
    test_same_token_cos_column_permutation_invariant()
    test_geometry_linear_probes_finite_and_deterministic()
    test_vicreg_gradient_attribution_ema_no_grads()
    test_vicreg_gradient_attribution_grad_norms_finite()
    print("PASS: all B9 representation calibration tests")
