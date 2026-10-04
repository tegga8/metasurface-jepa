"""Tests for the split-ratio generator (scripts/data/make_split_ratio.py).

Synthetic .mat fixtures only — no released data needed. Checks:
- exact floor counts per ratio; the remainder goes to train
- splits are disjoint and cover the pool (unique per-item markers)
- one permutation shared by all ratios -> nested test prefixes: the 20 % tests of
  the 60/40 arms ARE the common slice; the 25 % arm's test supersets it
- determinism: same seed -> identical assignment
- unknown non-meta fields abort loudly (never silently dropped)
- MetaDiTDataset round-trips a produced file (skips loudly without torch)

Run:  python -m pytest tests/test_make_split_ratio.py -v
      python tests/test_make_split_ratio.py
"""

import contextlib
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "data"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import pytest
from scipy import io

import make_split_ratio as msr

RATIOS = ("60:20:20", "50:25:25", "40:40:20")


@contextlib.contextmanager
def _workdir():
    d = tempfile.mkdtemp(prefix="split_ratio_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fixture(root, sizes=(20, 6, 6), extra_fields=None):
    """Write three released-style .mat files; item ids live in parameter[:, 0]."""
    data_dir = os.path.join(root, "released")
    os.makedirs(data_dir, exist_ok=True)
    marker = 0
    for name, n in zip(msr.SOURCE_FILES, sizes):
        rng = np.random.RandomState(marker)
        parameter = rng.rand(n, 3)
        parameter[:, 0] = np.arange(marker, marker + n, dtype=float)
        payload = {
            "pattern": (rng.rand(64, 64, n) > 0.5).astype(np.int8),
            "parameter": parameter,
            "real": rng.rand(n, 301).astype(np.float32),
            "imag": rng.rand(n, 301).astype(np.float32),
        }
        if extra_fields:
            payload.update(extra_fields)
        io.savemat(os.path.join(data_dir, name), payload)
        marker += n
    return data_dir, marker


def _markers(path):
    return set(io.loadmat(path)["parameter"][:, 0].astype(int).tolist())


def _generate(root, ratios=RATIOS, perm_seed=42, common_frac=0.20):
    data_dir, n = _fixture(root)
    arrays, sources, n2 = msr.load_pools(data_dir, hashes=False)
    assert n2 == n
    out_base = os.path.join(root, "splits")
    mans = {}
    for text in ratios:
        mans[text] = msr.write_ratio(out_base, arrays, n, msr.parse_ratio(text),
                                     perm_seed, common_frac=common_frac,
                                     sources=sources)
    return data_dir, out_base, n, mans


def test_counts_disjoint_and_cover():
    with _workdir() as tmp:
        _, _, n, mans = _generate(tmp)
        all_ids = set(range(n))
        for text, man in mans.items():
            counts = man["counts"]
            assert (counts["train"], counts["val"], counts["test"]) == \
                msr.ratio_counts(n, *msr.parse_ratio(text))
            tr = _markers(man["files"]["train"])
            va = _markers(man["files"]["val"])
            te = _markers(man["files"]["test"])
            assert tr | va | te == all_ids
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert (len(tr), len(va), len(te)) == \
                (counts["train"], counts["val"], counts["test"])


def test_nested_prefixes_and_common_slice():
    with _workdir() as tmp:
        _, _, _, mans = _generate(tmp)
        te60 = _markers(mans["60:20:20"]["files"]["test"])
        te50 = _markers(mans["50:25:25"]["files"]["test"])
        te40 = _markers(mans["40:40:20"]["files"]["test"])
        assert te60 == te40                 # both 20 % -> identical prefix
        assert te60 < te50                  # the 25 % test supersets the 20 % one
        common = _markers(mans["50:25:25"]["files"]["common_test"])
        assert common == te60               # shared held-out slice, cross-arm
        # arms whose own test IS the common slice write no duplicate file
        assert "common_test" not in mans["60:20:20"]["files"]
        assert "common_test" not in mans["40:40:20"]["files"]


def test_determinism_same_seed():
    with _workdir() as tmp:
        _, _, _, a = _generate(os.path.join(tmp, "a"))
        _, _, _, b = _generate(os.path.join(tmp, "b"))
        for text in RATIOS:
            for tag in ("train", "val", "test"):
                assert _markers(a[text]["files"][tag]) == \
                    _markers(b[text]["files"][tag])


def test_unknown_field_fails_loudly():
    with _workdir() as tmp:
        data_dir, _ = _fixture(tmp, extra_fields={"subset": np.zeros((6, 1))})
        with pytest.raises(ValueError, match="subset"):
            msr.load_pools(data_dir, hashes=False)


def test_manifest_written_and_matches():
    with _workdir() as tmp:
        _, _, n, mans = _generate(tmp)
        for _, man in mans.items():
            with open(os.path.join(man["out_dir"], "manifest.json")) as f:
                loaded = json.load(f)
            assert loaded["counts"] == man["counts"]
            assert loaded["perm_seed"] == 42 and loaded["n"] == n
            assert loaded["ratio_text"] == man["ratio_text"]


def test_dataset_roundtrip():
    pytest.importorskip("torch")
    from data.dataset import MetaDiTDataset
    with _workdir() as tmp:
        _, _, _, mans = _generate(tmp)
        man = mans["60:20:20"]
        ds = MetaDiTDataset(man["files"]["train"])
        assert len(ds) == man["counts"]["train"]
        grid, spec = ds[0]
        assert tuple(grid.shape) == (3, 64, 64)
        assert tuple(spec.shape) == (2, 301)


def test_parse_ratio_and_validation():
    assert msr.parse_ratio("60:20:20") == pytest.approx((0.6, 0.2, 0.2))
    assert msr.parse_ratio("0.6:0.2:0.2") == pytest.approx((0.6, 0.2, 0.2))
    with pytest.raises(ValueError):
        msr.parse_ratio("60:20")
    with pytest.raises(ValueError, match="sum to 1.0"):
        msr.ratio_counts(100, 0.6, 0.3, 0.3)
    with pytest.raises(ValueError, match="degenerate"):
        msr.ratio_counts(3, 0.34, 0.33, 0.33)
    assert msr.ratio_counts(100, 0.6, 0.2, 0.2) == (60, 20, 20)


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
