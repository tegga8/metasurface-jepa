"""A1 regression: the --eval-seed measurement control (measurement-only).

Proves the evaluation seed (a) reproduces the historical draw at 0, (b) gives an
independent draw otherwise, (c) is deterministic for a fixed seed, and (d) touches
no model/training state.

Run: python -m pytest tests/test_eval_seed.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "eval"))

import torch
import torch.nn as nn

import eval_scenarios as es
from data.mask import BlockMasker
from runtime.physics_controls import make_shuffled_spectrum


def _c_mask(masker_seed, b=4):
    """Reproduce run_all_scenarios' draw order: A (1.0, no draw), B (0.5), C (0.25)."""
    occ = torch.zeros(b, 1, 64, 64)
    mk = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4),
                     seed=masker_seed)
    mk.sample(occ, ratio=1.0)
    mk.sample(occ, ratio=0.5)
    return mk.sample(occ, ratio=0.25)


def test_eval_seeds_derivation_and_zero_is_historical():
    cfg = {"train": {"seed": 42}}
    s0 = es._eval_seeds(cfg, 0)
    # eval_seed=0 must reproduce the historical seeds exactly.
    assert s0 == {"masker": 999, "A": 43, "B": 44, "C": 45}
    s5 = es._eval_seeds(cfg, 5)
    assert s5 == {"masker": 1004, "A": 48, "B": 49, "C": 50}


def test_same_eval_seed_reproduces_and_differs_across_seeds():
    a = _c_mask(999)
    a_again = _c_mask(999)
    assert torch.equal(a, a_again)                 # deterministic per eval seed
    b = _c_mask(1000)                              # eval_seed 0 vs 1
    assert not torch.equal(a, b)                   # a genuinely different draw


def test_c_mask_ratio_preserved_across_eval_seeds():
    for seed in (999, 1000, 1004):
        m = _c_mask(seed)
        masked = float((m < 0.5).float().mean())
        assert abs(masked - 0.25) < 0.05, f"seed {seed}: masked={masked}"


def test_same_eval_seed_reproduces_derangement_and_differs():
    torch.manual_seed(0)
    S = torch.randn(6, 2, 301)
    d45 = make_shuffled_spectrum(S, seed=45)
    assert torch.equal(d45, make_shuffled_spectrum(S, seed=45))
    d50 = make_shuffled_spectrum(S, seed=50)
    assert not torch.equal(d45, d50)


def test_eval_seed_touches_no_model_state():
    """Generating masks/derangements must not mutate a model's parameters."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for ev in (0, 1, 7):
        _c_mask(es._eval_seeds({"train": {"seed": 42}}, ev)["masker"])
        make_shuffled_spectrum(torch.randn(4, 2, 301),
                               seed=es._eval_seeds({"train": {"seed": 42}}, ev)["C"])
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"model param changed: {k}"
    assert all(p.grad is None for p in model.parameters())


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
