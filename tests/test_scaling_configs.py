"""Build + smoke tests for the Phase 6 scaling configs (configs/scaling/*.yaml).

Per config: architectural asserts hold (n_film_blocks == geo_depth, heads divide
hidden), intended key mutations are present, the model builds via the audit
instrument, one synthetic CPU forward+backward step runs, and the size relation
vs the shipped base config is as intended (S1/S2 smaller, L1 larger).

Run:  python -m pytest tests/test_scaling_configs.py -v
      python tests/test_scaling_configs.py
"""

import glob
import os
import sys

import pytest
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO_ROOT, "scripts", "diagnostics"),
           os.path.join(REPO_ROOT, "src"),
           os.path.join(REPO_ROOT, "scripts", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_size_audit as msa

SPEC_WEIGHTS = os.path.join(REPO_ROOT, "data", "metadit", "weights",
                            "spec_encoder.pth")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "unified.yaml")
SCALING_CONFIGS = sorted(glob.glob(
    os.path.join(REPO_ROOT, "configs", "scaling", "*.yaml")))

EXPECTED_MUTATIONS = {
    "unified_s1_small.yaml": {"hidden": 128, "num_heads": 4, "geo_depth": 4,
                              "predictor_depth": 6, "num_predictor_heads": 4,
                              "scalar_hidden": 96, "n_film_blocks": 4},
    "unified_s2_slim.yaml": {"hidden": 192, "predictor_depth": 4},
    "unified_l1_wide.yaml": {"hidden": 256, "num_heads": 8,
                             "num_predictor_heads": 8, "scalar_hidden": 192},
}
SIZE_RELATION = {
    "unified_s1_small": "smaller",
    "unified_s2_slim": "smaller",
    "unified_l1_wide": "larger",
}


def _cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_scaling_configs_present():
    names = {os.path.basename(p) for p in SCALING_CONFIGS}
    assert set(EXPECTED_MUTATIONS) <= names


@pytest.mark.parametrize("path", SCALING_CONFIGS, ids=os.path.basename)
def test_config_keys_and_asserts(path):
    cfg = _cfg(path)
    name = os.path.basename(path)
    for key, value in EXPECTED_MUTATIONS[name].items():
        assert cfg[key] == value, f"{name}: {key}={cfg[key]} != {value}"
    assert cfg["n_film_blocks"] == cfg["geo_depth"]
    assert cfg["hidden"] % cfg["num_heads"] == 0
    assert cfg["hidden"] % cfg["num_predictor_heads"] == 0
    # objective / staging carried over unchanged
    assert cfg["objective"] == "jepa"
    assert cfg["staging"]["physics_use_ste"] is True


@pytest.fixture(scope="module")
def audit():
    if not os.path.exists(SPEC_WEIGHTS):
        pytest.skip(f"released spectrum encoder missing: {SPEC_WEIGHTS}")
    if not SCALING_CONFIGS:
        pytest.skip("no configs/scaling/*.yaml present")
    base_entry, _ = msa.audit_config("unified", BASE_CONFIG, torch.device("cpu"),
                                     batch_size=2, reps=1, timing=False)
    entries, models = {}, {}
    for path in SCALING_CONFIGS:
        label = os.path.splitext(os.path.basename(path))[0]
        entry, model = msa.audit_config(label, path, torch.device("cpu"),
                                        batch_size=2, reps=1, timing=False)
        entries[label] = entry
        models[label] = model
    return base_entry, entries, models


def test_relative_size_and_frozen_targets(audit):
    base_entry, entries, _ = audit
    assert set(SIZE_RELATION) <= set(entries)
    for label, relation in SIZE_RELATION.items():
        entry = entries[label]
        assert entry["total"] == sum(v["total"] for v in entry["params"].values())
        if relation == "smaller":
            assert entry["trainable"] < base_entry["trainable"]
        else:
            assert entry["trainable"] > base_entry["trainable"]
        assert entry["params"]["ema"]["trainable"] == 0
        assert entry["params"]["scalar_mlp_ema"]["trainable"] == 0


def test_build_and_smoke_step(audit):
    _, _, models = audit
    for label, model in models.items():
        secs = msa.time_step(model, batch_size=1, device=torch.device("cpu"),
                             reps=1, warmup=0)
        assert secs > 0.0, label


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
