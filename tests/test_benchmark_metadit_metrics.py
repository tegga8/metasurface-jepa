"""Tests for the MetaDiT-comparable benchmark harness.

Covers:
- MAE / AAE parity with external/metadit/metric.py (the authoritative definition)
- AAE&K definition and parity with reference calculate_aaeandk
- candidate generator: K=1/sigma=0 == deterministic; seeded reproducibility
- baseline helpers (AVG1, surrogate floor) run and return finite MAE/AAE
- the recorded reproduced MetaDiT baseline (checkpoints/phase0/seed0_metric.json)

Run:  python -m pytest tests/test_benchmark_metadit_metrics.py -v
      python tests/test_benchmark_metadit_metrics.py        (standalone runner)

Data-dependent checks skip loudly when the external reference or the reproduced
artifact is absent (never pass silently).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "benchmark"))

import pytest
import torch

import metadit_metrics as mm
import candidate_sampling as cs
import baselines as bl

_HAS_REFERENCE = os.path.exists(os.path.join(mm.METADIT_SRC, "metric.py"))
_NEEDS_REFERENCE = pytest.mark.skipif(
    not _HAS_REFERENCE,
    reason="external/metadit/metric.py not staged (reference metric unavailable)")


# ---------------------------------------------------------------------------
# stubs (kept light: the generator helpers only need decode_geometry + a callable
# surrogate returning an object with .prediction)
# ---------------------------------------------------------------------------

class _StubModel:
    def decode_geometry(self, z_hat, scalar_pred, occ_input=None, mask=None,
                        scalar_known=None, scalar_values=None, use_ste=False,
                        hard_forward=False):
        b = z_hat.shape[0]
        g = z_hat.mean(dim=-1).view(b, 1, 16, 16)
        g = torch.nn.functional.interpolate(g, size=(64, 64), mode="nearest")
        return g.expand(b, 3, 64, 64).contiguous(), occ_input


class _SurrogateOut:
    def __init__(self, prediction):
        self.prediction = prediction


class _StubSurrogate:
    def __call__(self, geom):
        v = geom.mean(dim=(1, 2, 3))  # (B,)
        return _SurrogateOut(v.view(-1, 1, 1).expand(geom.shape[0], 2, 301).contiguous())


def _stub_batch(b=3, hidden=8):
    torch.manual_seed(0)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[2.7, 0.8, 4.2], [2.6, 0.7, 3.9], [2.9, 0.9, 4.8]])[:b].contiguous()
    sk = torch.zeros(b, 3, dtype=torch.bool)
    mask = torch.zeros(b, 16, 16)
    spec = torch.randn(b, 2, 301)
    z_hat = torch.randn(b, 256, hidden)
    out = {"z_hat": z_hat, "scalar_pred": torch.zeros(b, 3)}
    return occ, sv, sk, mask, spec, out


# ---------------------------------------------------------------------------
# metric parity vs the reference implementation
# ---------------------------------------------------------------------------

@_NEEDS_REFERENCE
def test_batched_mae_aae_match_reference():
    torch.manual_seed(1)
    y_true = torch.randn(4, 2, 301)
    y_pred = y_true + 0.01 * torch.randn(4, 2, 301)
    # MAE is a mean over all elements, so the batch form equals the reference.
    assert mm.mae(y_true, y_pred) == pytest.approx(
        mm.reference_mae(y_true, y_pred), rel=1e-6)
    # AAE sums the WHOLE tensor, and the reference is only ever called per
    # single item (metric.eval_loop); compare against the per-item average.
    b = y_true.shape[0]
    ref_aae = sum(mm.reference_aae(y_true[i:i + 1], y_pred[i:i + 1])
                  for i in range(b)) / b
    assert mm.aae(y_true, y_pred) == pytest.approx(ref_aae, rel=1e-6)


@_NEEDS_REFERENCE
def test_aae_and_k_matches_reference():
    torch.manual_seed(2)
    mat = torch.rand(7, 4) * 50.0
    for k in (1, 2, 4):
        # reference builds float32 tensors from python floats, so the max/mean
        # reduction order differs slightly; agreement is to float32 precision.
        assert mm.aae_and_k(mat, k) == pytest.approx(
            mm.reference_aae_and_k(mat, k), rel=1e-6)


def test_aae_and_k_k1_is_plain_aae():
    torch.manual_seed(3)
    y_true = torch.randn(5, 2, 301)
    y_pred = y_true + 0.02 * torch.randn(5, 2, 301)
    col = mm.per_item_aae(y_true, y_pred).unsqueeze(1)  # (5,1)
    assert mm.aae_and_k(col, 1) == pytest.approx(float(col.mean().item()))


def test_aae_and_k_takes_the_worst_candidate():
    mat = torch.tensor([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]])
    assert mm.aae_and_k(mat, 3) == pytest.approx((5.0 + 9.0) / 2.0)


# ---------------------------------------------------------------------------
# candidate generator
# ---------------------------------------------------------------------------

def test_latent_jitter_k1_sigma0_is_deterministic():
    occ, sv, sk, mask, spec, out = _stub_batch()
    model, surr = _StubModel(), _StubSurrogate()
    geoms = cs.latent_jitter_geometries(model, out, occ, mask, sk, sv,
                                        sigma=0.0, k=3)
    ref, _ = model.decode_geometry(out["z_hat"], out["scalar_pred"],
                                   occ_input=occ, mask=mask)
    assert torch.equal(geoms[0], ref)
    for g in geoms:
        assert torch.equal(g, geoms[0]), "sigma=0 candidates must be identical"


def test_latent_jitter_seeded_reproducible_and_moves_with_sigma():
    occ, sv, sk, mask, spec, out = _stub_batch()
    model = _StubModel()
    a = cs.latent_jitter_geometries(model, out, occ, mask, sk, sv,
                                    sigma=0.5, k=3, seed=11)
    b = cs.latent_jitter_geometries(model, out, occ, mask, sk, sv,
                                    sigma=0.5, k=3, seed=11)
    for ga, gb in zip(a, b):
        assert torch.equal(ga, gb), "fixed seed must be reproducible"
    assert not torch.equal(a[0], a[1]), "sigma>0 must move candidate 1"
    assert torch.equal(a[0], cs.latent_jitter_geometries(
        model, out, occ, mask, sk, sv, sigma=0.5, k=1, seed=11)[0]), \
        "candidate 0 is always the noiseless decode"


def test_candidate_aae_matrix_shape_and_k1_equals_deterministic():
    occ, sv, sk, mask, spec, out = _stub_batch()
    model, surr = _StubModel(), _StubSurrogate()
    mat = cs.candidate_aae_matrix(model, surr, out, occ, mask, sk, sv, spec,
                                  sigma=0.0, k=1)
    assert mat.shape == (occ.shape[0], 1)
    ref, _ = model.decode_geometry(out["z_hat"], out["scalar_pred"],
                                   occ_input=occ, mask=mask)
    assert torch.allclose(mat[:, 0], mm.per_item_aae(spec, surr(ref).prediction))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def test_avg1_and_surrogate_floor_return_finite_metrics():
    occ, sv, sk, mask, spec, out = _stub_batch()
    surr = _StubSurrogate()
    avg1 = bl.avg1_metrics(spec, bl.mean_spectrum(spec))
    floor = bl.surrogate_floor_metrics(occ, sv, spec, surr)
    for d in (avg1, floor):
        assert set(d) == {"MAE", "AAE"}
        assert all(isinstance(v, float) and v == v for v in d.values())  # finite


# ---------------------------------------------------------------------------
# reproduced MetaDiT baseline artifact
# ---------------------------------------------------------------------------

def test_seed0_metric_artifact_is_the_reproduced_baseline():
    path = os.path.join(REPO_ROOT, "checkpoints", "phase0", "seed0_metric.json")
    if not os.path.exists(path):
        pytest.skip("checkpoints/phase0/seed0_metric.json absent (reproduced "
                    "baseline not staged)")
    with open(path) as f:
        m = json.load(f)
    assert m["MAE"] == pytest.approx(0.0801, abs=5e-3)
    assert m["AAE"] == pytest.approx(48.2495, abs=0.5)


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001 - standalone runner
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
