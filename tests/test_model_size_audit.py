"""Tests for the model-size audit instrument (scripts/diagnostics/model_size_audit.py).

Fast pure checks always run; the real model build + one timed step skip loudly
when the released spectrum encoder is not staged locally (data/metadit/weights).

Run:  python -m pytest tests/test_model_size_audit.py -v
      python tests/test_model_size_audit.py
"""

import contextlib
import copy
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO_ROOT, "scripts", "diagnostics"),
           os.path.join(REPO_ROOT, "src"),
           os.path.join(REPO_ROOT, "scripts", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
import torch
import torch.nn as nn
import yaml

import model_size_audit as msa

SPEC_WEIGHTS = os.path.join(REPO_ROOT, "data", "metadit", "weights",
                            "spec_encoder.pth")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "unified.yaml")


@contextlib.contextmanager
def _workdir():
    d = tempfile.mkdtemp(prefix="model_size_audit_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_module_param_counts_tiny():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)                 # 20
            self.b = nn.Sequential(nn.Linear(2, 3))  # 9
            self.frozen = nn.Linear(1, 1)            # 2, frozen below
    m = Tiny()
    m.frozen.weight.requires_grad_(False)
    m.frozen.bias.requires_grad_(False)
    counts, total, trainable = msa.module_param_counts(m)
    by_name = {n: (t, tr) for n, t, tr in counts}
    assert by_name["a"] == (20, 20)
    assert by_name["b"] == (9, 9)
    assert by_name["frozen"] == (2, 0)
    assert total == 31 and trainable == 29


def test_format_table_lists_rows():
    results = [
        {"label": "unified", "total": 18900000, "trainable": 11370000,
         "seconds_per_step": 1.0, "ratio_vs_base": 1.0},
        {"label": "unified_s1_small", "total": 4000000, "trainable": 3000000,
         "seconds_per_step": 0.5, "ratio_vs_base": 0.5},
    ]
    text = msa.format_table(results)
    assert "unified" in text and "unified_s1_small" in text
    assert "18,900,000" in text and "0.50" in text


@pytest.fixture(scope="module")
def base():
    if not os.path.exists(SPEC_WEIGHTS):
        pytest.skip(f"released spectrum encoder missing: {SPEC_WEIGHTS}")
    entry, model = msa.audit_config("unified", BASE_CONFIG, torch.device("cpu"),
                                    batch_size=2, reps=1, timing=False)
    return entry, model


def test_base_counts_consistent_and_sane(base):
    entry, _ = base
    assert entry["total"] == sum(v["total"] for v in entry["params"].values())
    assert entry["trainable"] == sum(v["trainable"] for v in entry["params"].values())
    assert 15_000_000 < entry["total"] < 25_000_000
    assert 8_000_000 < entry["trainable"] < 15_000_000
    for name in ("predictor", "ema"):
        assert name in entry["params"]
    # frozen targets must carry zero trainable parameters
    assert entry["params"]["ema"]["trainable"] == 0
    assert entry["params"]["scalar_mlp_ema"]["trainable"] == 0


def test_variant_builds_and_shrinks(base):
    entry, _ = base
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    small = copy.deepcopy(cfg)
    small.update({"hidden": 128, "num_heads": 4, "geo_depth": 4,
                  "predictor_depth": 6, "num_predictor_heads": 4,
                  "scalar_hidden": 96, "n_film_blocks": 4})
    with _workdir() as tmp:
        path = os.path.join(tmp, "unified_small.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(small, f)
        entry_small, _ = msa.audit_config("small", path, torch.device("cpu"),
                                          batch_size=2, reps=1, timing=False)
    assert entry_small["total"] < entry["total"]
    assert entry_small["trainable"] < entry["trainable"]


def test_timed_step_positive(base):
    _, model = base
    secs = msa.time_step(model, batch_size=1, device=torch.device("cpu"),
                         reps=1, warmup=0)
    assert secs > 0.0


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
