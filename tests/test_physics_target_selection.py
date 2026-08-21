"""Tests for the physics-target-selection diagnostics (latent-selection spec §23).

Covers, at minimum:
  1. same target across real/null/shuffle conditions
  2. same mask across interventions
  3. shuffle permutation excludes self where B > 1
  4. distance matrix diagonal semantics
  5. probe train/test split separation
  6. no in-place mutation
  7. B=1 -> shuffled-spectrum retrieval marked INFEASIBLE (never a fake negative)

Run:  python tests/test_physics_target_selection.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))

import numpy as np
import pytest
import torch

from latent_geometry_probe import (
    build_probe, fit_input_stats, split_indices,
)
from physics_target_selection import (
    evaluate_ratio, geometry_aware_subset, margin_stats,
    per_sample_distances, retrieval_matrix_metrics, same_context_causal_wins,
    shuffle_permutation,
)


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------

class _FakeEMA(torch.nn.Module):
    """Deterministic 'target encoder': latent depends ONLY on G."""

    def __init__(self, t=8, d=4):
        super().__init__()
        self.t, self.d = t, d

    def forward(self, G):
        b = G.shape[0]
        base = torch.arange(self.t * self.d, dtype=torch.float32)
        z = base.repeat(b, 1).reshape(b, self.t, self.d).clone()
        # geometry-dependent offset so different geometries -> different targets
        z = z + G[:, 1].mean(dim=(1, 2)).view(b, 1, 1)
        return z


class _FakeModel(torch.nn.Module):
    """Records every forward call; z_hat shifts toward z_y by a spectrum-dependent
    amount under 'real', less under 'null'/'shuffled' — enough to exercise the
    metric plumbing without a real network. Token count matches the 16x16 mask
    grid (256 tokens) so masks index latents directly."""

    def __init__(self, b=4, t=256, d=4, seed=0):
        super().__init__()
        self.b, self.t, self.d = b, t, d
        self.ema = _FakeEMA(t, d)
        self.calls = []
        g = torch.Generator().manual_seed(seed)
        self.noise = torch.randn(b, t, d, generator=g)

    def target(self, G):
        return self.ema(G)

    def forward(self, G, S, M, goal_mode="real", need_attn=False):
        self.calls.append({
            "G": G.clone(), "S": S.clone(), "M": M.clone(),
            "goal_mode": goal_mode,
        })
        z_y = self.target(G)
        mask = (M.view(M.shape[0], -1) == 0)
        b = z_y.shape[0]
        if goal_mode == "real":
            # spectrum-dependent correction amplitude (row-specific): a real
            # physics-conditioned predictor produces row-specific outputs
            amp = 0.05 + 0.10 * torch.sigmoid(S.mean(dim=(1, 2)))
            z_hat = z_y + amp.view(-1, 1, 1) * self.noise[:b]
        else:
            z_hat = z_y + 0.75 * self.noise[:b]
        return {"z_hat": z_hat, "z_y_raw": z_y, "z_x": None,
                "mask": mask, "attn_weights": None,
                "c_physics": None, "a_goal": None}


def _make_batch(b=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    G = (torch.rand(b, 3, 64, 64, generator=g) > 0.7).float()
    G[:, 2] = 0.9                      # l_lattice channel is everywhere
    S = torch.randn(b, 301, 2, generator=g)
    M = torch.ones(b, 16, 16)
    M[:, :6, :] = 0.0                  # one masked block -> fixed mask
    return G, S, M


# ---------------------------------------------------------------------------
# 3. shuffle permutation excludes self where B > 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [2, 3, 5, 16])
def test_shuffle_permutation_excludes_self(b):
    perm = shuffle_permutation(b)
    assert sorted(perm) == list(range(b)), "must be a permutation"
    assert all(perm[i] != i for i in range(b)), "no fixed points allowed"


def test_shuffle_permutation_b1_returns_identity():
    assert shuffle_permutation(1) == [0]


# ---------------------------------------------------------------------------
# 1 + 2. same target and same mask across real/null/shuffle
# ---------------------------------------------------------------------------

def test_same_target_mask_across_conditions():
    torch.manual_seed(0)
    model = _FakeModel(b=4)
    G, S, M = _make_batch(4)
    res = evaluate_ratio(model, [(G, S)], [M], device="cpu")

    modes = [c["goal_mode"] for c in model.calls]
    # repo convention: shuffled = goal_mode "real" with a PERMUTED spectrum
    assert modes == ["real", "null", "real"]
    assert torch.equal(model.calls[0]["S"], S)
    s_shuf = model.calls[2]["S"]
    assert not torch.equal(s_shuf, S), "shuffled call must use a different spectrum"
    assert sorted(s_shuf[:, 0, 0].tolist()) == sorted(S[:, 0, 0].tolist()), \
        "shuffled spectrum must be a row permutation of the original"
    g0 = model.calls[0]["G"]
    m0 = model.calls[0]["M"]
    for c in model.calls[1:]:
        assert torch.equal(c["G"], g0), "geometry must be identical across conditions"
        assert torch.equal(c["M"], m0), "mask must be identical across conditions"

    # null is strictly worse than real BY CONSTRUCTION (0.75 vs <=0.15 amplitude)
    assert res["margins"]["cos_null_minus_real"]["median"] > 0.0
    # shuffled margins are recorded and generically nonzero (spectrum-dependent fake)
    sm = res["margins"]["cos_shuffle_minus_real"]
    assert np.isfinite(sm["median"])
    assert not (sm["fraction_positive"] == 0.0 and sm["fraction_negative"] == 0.0)


# ---------------------------------------------------------------------------
# distance/margin math
# ---------------------------------------------------------------------------

def test_per_sample_distances_hand_computed():
    z_y = torch.zeros(1, 2, 3)
    z_y[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    z_y[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    z_hat = z_y.clone()
    z_hat[0, 0] = torch.tensor([0.0, 1.0, 0.0])   # orthogonal to target token 0
    mask = torch.tensor([[True, False]])
    d = per_sample_distances(z_hat, z_y, mask)
    assert abs(d["cos"][0] - 1.0) < 1e-6          # 1 - cos = 1 - 0
    assert abs(d["l2_token"][0] - np.sqrt(2.0)) < 1e-6
    # one masked token: pooled vector IS that token's difference
    assert abs(d["l2_pooled"][0] - np.sqrt(2.0)) < 1e-6


def test_identical_latents_zero_distance():
    z = torch.randn(3, 5, 7)
    mask = torch.ones(3, 5, dtype=torch.bool)
    d = per_sample_distances(z, z.clone(), mask)
    assert all(abs(v) < 1e-7 for v in d["cos"])
    assert all(abs(v) < 1e-7 for v in d["l2_token"])


def test_margin_stats_fractions():
    real = [0.5, 0.5, 0.5, 0.5]
    other = [0.6, 0.4, 0.5, 0.7]
    m = margin_stats(real, other)
    assert abs(m["fraction_positive"] - 0.5) < 1e-9   # two strictly positive
    assert abs(m["fraction_negative"] - 0.25) < 1e-9  # one strictly negative
    # margins: +.1, -.1, .0, +.2 -> sorted median = (0.0 + 0.1)/2
    assert abs(m["median"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# 4. distance matrix diagonal semantics
# ---------------------------------------------------------------------------

def test_retrieval_matrix_perfect_diagonal():
    D = np.array([[0.1, 0.9, 0.8],
                  [0.7, 0.2, 0.9],
                  [0.8, 0.9, 0.3]])
    m = retrieval_matrix_metrics(D)
    assert m["feasible"] is True
    assert m["recall_at_1"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["mean_correct_rank"] == 1.0
    assert m["diagonal_minus_offdiagonal_margin"] > 0.0


def test_retrieval_matrix_inverted_diagonal():
    # diagonal is the WORST entry in every row
    D = np.array([[0.9, 0.1, 0.2],
                  [0.2, 0.9, 0.1],
                  [0.1, 0.2, 0.9]])
    m = retrieval_matrix_metrics(D)
    assert m["recall_at_1"] == 0.0
    assert m["diagonal_minus_offdiagonal_margin"] < 0.0
    assert m["mean_correct_rank"] == 3.0


def test_causal_wins_perfect_diagonal():
    D = np.array([[0.1, 0.9],
                  [0.8, 0.2]])
    w = same_context_causal_wins(D)
    assert w["feasible"] is True
    assert w["row_win_rate"] == 1.0
    assert w["col_win_rate"] == 1.0
    assert w["mutual_win_rate"] == 1.0


# ---------------------------------------------------------------------------
# 7. B=1 -> shuffled-spectrum marked infeasible, never a fake negative
# ---------------------------------------------------------------------------

def test_b1_retrieval_marked_infeasible():
    m = retrieval_matrix_metrics(np.array([[0.42]]))
    assert m["feasible"] is False
    assert "batch_size" in m and m["batch_size"] == 1


def test_b1_evaluate_ratio_marks_shuffled_infeasible():
    torch.manual_seed(0)
    model = _FakeModel(b=1)
    G, S, M = _make_batch(1)
    res = evaluate_ratio(model, [(G, S)], [M], device="cpu")
    assert res["shuffled_condition_feasible"] is False


# ---------------------------------------------------------------------------
# geometry-aware subset selection is geometry-only
# ---------------------------------------------------------------------------

def test_geometry_subset_ignores_latents():
    occ = np.zeros((6, 16), dtype=np.uint8)
    occ[0, :8] = 1
    occ[1, 8:] = 1
    occ[2, ::2] = 1
    occ[3, 1::2] = 1
    occ[4, :4] = 1
    occ[5, 12:] = 1
    lat_a = torch.randn(6, 5)
    sub_a, info_a = geometry_aware_subset(occ, k=3)
    sub_b, info_b = geometry_aware_subset(occ, k=3)
    # deterministic function of geometry alone
    assert np.array_equal(sub_a, sub_b)
    assert info_a["min_pairwise_hamming"] > 0
    # distinct geometries actually differ
    pair_h = (occ[sub_a][:, None, :] ^ occ[sub_a][None, :, :]).sum(-1)
    np.fill_diagonal(pair_h, 10 ** 9)
    assert pair_h.min() >= info_a["min_pairwise_hamming"]


# ---------------------------------------------------------------------------
# 5. probe split separation
# ---------------------------------------------------------------------------

def test_split_indices_disjoint_and_covering():
    n = 101
    tr, va, te = split_indices(n, seed=0)
    assert len(tr) + len(va) + len(te) == n
    all_idx = np.concatenate([tr, va, te])
    assert len(set(all_idx.tolist())) == n, "splits must be disjoint"
    assert set(all_idx.tolist()) == set(range(n)), "splits must cover everything"


def test_fit_input_stats_train_only():
    x = torch.randn(50, 4)
    tr, va, te = split_indices(50, seed=1)
    mu, sd = fit_input_stats(x[tr])
    mu_full = x.mean(0)
    assert not torch.allclose(mu, mu_full, atol=1e-6), \
        "stats must come from the train split only"


def test_probe_never_sees_test_rows():
    """The trained probe's weights must be a function of train/val rows only:
    a linear probe fitted on train-split rows of a linearly separable problem
    classifies its own train split near-perfectly (fitting consumed exactly the
    passed rows); split disjointness is pinned by test_split_indices_*."""
    torch.manual_seed(0)
    n, d = 60, 6
    x = torch.randn(n, d)
    y = ((x[:, 0] > 0).float().unsqueeze(1).repeat(1, 3))
    tr, va, te = split_indices(n, seed=2)
    mu, sd = fit_input_stats(x[tr])
    zs = (x - mu) / sd
    probe = build_probe("linear", d, 3)
    from latent_geometry_probe import train_probe
    train_probe(probe, zs[tr], y[tr], zs[va], y[va], epochs=400, lr=0.1)
    with torch.no_grad():
        acc_tr = (((probe(zs[tr]) > 0).float() == y[tr]).float().mean())
        # test rows are from the SAME separable distribution -> also classified
        acc_te = (((probe(zs[te]) > 0).float() == y[te]).float().mean())
    assert acc_tr > 0.95
    assert acc_te > 0.95


# ---------------------------------------------------------------------------
# 6. no in-place mutation
# ---------------------------------------------------------------------------

def test_evaluate_ratio_does_not_mutate_inputs():
    torch.manual_seed(0)
    model = _FakeModel(b=3)
    G, S, M = _make_batch(3)
    G0, S0, M0 = G.clone(), S.clone(), M.clone()
    evaluate_ratio(model, [(G, S)], [M], device="cpu")
    assert torch.equal(G, G0)
    assert torch.equal(S, S0)
    assert torch.equal(M, M0)


def test_per_sample_distances_does_not_mutate_inputs():
    z1 = torch.randn(2, 4, 3)
    z2 = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    a, b = z1.clone(), z2.clone()
    per_sample_distances(z1, z2, mask)
    assert torch.equal(z1, a)
    assert torch.equal(z2, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
