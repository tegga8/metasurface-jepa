"""Re-split the released MetaDiT .mat pools into alternative train/val/test ratios.

Operator decision 2026-10-04: the split-ratio study replaces sample-count
subsampling — the total dataset is unchanged (174,883 items); only the
allocation moves (e.g. 60:20:20, 50:25:25, 40:40:20 vs the released 80:10:10).

Design (nested splits from one permutation):
- The released files (train/val/test_set.mat) are concatenated in a fixed order
  (train, then val, then test) into one pool of N items.
- ONE permutation (seed 42 by default) serves every ratio:

      test  = p[: n_te]
      val   = p[n_te : n_te + n_va]
      train = p[n_te + n_va :]

  so test sets of different ratios are nested prefixes: the smallest requested
  test slice ("common" slice, --common-test-frac) is valid held-out data for
  EVERY arm and is the cross-arm comparison set.
- Counts are floor-based per fraction; the remainder goes to train.

The .mat contract is exactly MetaDiTDataset's field set:
    pattern (64, 64, N), parameter (N, 3), real (N, 301), imag (N, 301).
Unknown non-meta fields in the source files abort loudly — they would be
silently dropped otherwise.

Run:
    python scripts/data/make_split_ratio.py --data-dir data/metadit/split_data \
        --out-base /kaggle/working/splits --ratios 60:20:20,50:25:25,40:40:20 \
        --perm-seed 42 --common-test-frac 0.20
"""

import argparse
import hashlib
import json
import os

import numpy as np
from scipy import io

META_KEYS = {"__header__", "__version__", "__globals__"}
DATA_FIELDS = ("pattern", "parameter", "real", "imag")
SOURCE_FILES = ("train_set.mat", "val_set.mat", "test_set.mat")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pools(data_dir, hashes=True):
    """Concatenate the four data fields of train/val/test_set.mat (fixed order).

    Returns (arrays, sources, n). Raises on any unexpected non-meta field.
    """
    parts = {k: [] for k in DATA_FIELDS}
    sources = []
    for name in SOURCE_FILES:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mat = io.loadmat(path)
        unknown = sorted(set(mat) - META_KEYS - set(DATA_FIELDS))
        if unknown:
            raise ValueError(
                f"{path}: unexpected non-meta fields {unknown}; refusing to drop "
                "them silently — extend DATA_FIELDS and the writer first")
        for k in DATA_FIELDS:
            parts[k].append(mat[k])
        sources.append({"file": name, "n": int(mat["parameter"].shape[0]),
                        "sha256": _sha256(path) if hashes else None})
    arrays = {
        "pattern": np.concatenate(parts["pattern"], axis=-1),
        "parameter": np.concatenate(parts["parameter"], axis=0),
        "real": np.concatenate(parts["real"], axis=0),
        "imag": np.concatenate(parts["imag"], axis=0),
    }
    n = arrays["parameter"].shape[0]
    if arrays["pattern"].shape[-1] != n:
        raise ValueError(
            f"pattern has {arrays['pattern'].shape[-1]} items, parameter {n}")
    return arrays, sources, n


def parse_ratio(text):
    """'60:20:20' -> (0.6, 0.2, 0.2); also accepts fractions already < 1."""
    parts = [p.strip() for p in str(text).split(":")]
    if len(parts) != 3:
        raise ValueError(f"ratio must be train:val:test, got {text!r}")
    vals = [float(p) for p in parts]
    if sum(vals) > 1.5:          # given as percents (60:20:20)
        vals = [v / 100.0 for v in vals]
    return tuple(vals)


def ratio_counts(n, train_frac, val_frac, test_frac):
    """Floor-based counts; the remainder goes to train. All parts must be > 0."""
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    n_te = int(np.floor(test_frac * n))
    n_va = int(np.floor(val_frac * n))
    n_tr = n - n_te - n_va
    if min(n_tr, n_va, n_te) <= 0:
        raise ValueError(f"degenerate split for n={n}: {(n_tr, n_va, n_te)}")
    return n_tr, n_va, n_te


def assign_indices(n, n_tr, n_va, n_te, perm_seed):
    """One permutation shared by all ratios -> nested prefix splits."""
    p = np.random.RandomState(perm_seed).permutation(n)
    return p[n_te + n_va:], p[n_te:n_te + n_va], p[:n_te]


def write_split(out_dir, arrays, idx, tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_set.mat")
    io.savemat(path, {
        "pattern": arrays["pattern"][:, :, idx],
        "parameter": arrays["parameter"][idx],
        "real": arrays["real"][idx],
        "imag": arrays["imag"][idx],
    })
    return path


def ratio_tag(train_frac, val_frac, test_frac):
    return (f"tr{round(train_frac * 100)}_va{round(val_frac * 100)}_"
            f"te{round(test_frac * 100)}")


def write_ratio(out_base, arrays, n, ratio, perm_seed,
                common_frac=None, sources=None):
    train_frac, val_frac, test_frac = ratio
    n_tr, n_va, n_te = ratio_counts(n, train_frac, val_frac, test_frac)
    tr, va, te = assign_indices(n, n_tr, n_va, n_te, perm_seed)
    tag = ratio_tag(train_frac, val_frac, test_frac)
    out_dir = os.path.join(out_base, tag)
    files = {
        "train": write_split(out_dir, arrays, tr, "train"),
        "val": write_split(out_dir, arrays, va, "val"),
        "test": write_split(out_dir, arrays, te, "test"),
    }
    if common_frac is not None:
        n_common = int(np.floor(common_frac * n))
        if n_common > n_te:
            raise ValueError(
                f"common frac {common_frac} exceeds this arm's test frac "
                f"{test_frac} — the common slice must be a prefix of every test")
        if n_common != n_te:     # own test already IS the common slice
            files["common_test"] = write_split(
                out_dir, arrays, te[:n_common], "common_test")
    manifest = {
        "out_dir": out_dir,
        "ratio": list(ratio),
        "ratio_text": tag,
        "perm_seed": perm_seed,
        "n": n,
        "counts": {"train": n_tr, "val": n_va, "test": n_te},
        "common_test_frac": common_frac,
        "fields": list(DATA_FIELDS),
        "files": files,
        "sources": sources or [],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description="Re-split released MetaDiT pools into alternative "
                    "train/val/test ratios (nested prefixes, one permutation).")
    ap.add_argument("--data-dir", required=True,
                    help="directory with train/val/test_set.mat")
    ap.add_argument("--out-base", required=True,
                    help="base directory for the per-ratio output folders")
    ap.add_argument("--ratios", required=True,
                    help="comma-separated train:val:test, e.g. 60:20:20,50:25:25,40:40:20")
    ap.add_argument("--perm-seed", type=int, default=42)
    ap.add_argument("--common-test-frac", type=float, default=0.20,
                    help="fraction of the pooled dataset written as the shared "
                         "common_test_set.mat (only for arms whose test is larger)")
    ap.add_argument("--no-common", action="store_true")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip source SHA256 (saves a full re-read of the .mat files)")
    args = ap.parse_args()

    arrays, sources, n = load_pools(args.data_dir, hashes=not args.no_hash)
    print(f"[make_split_ratio] pooled N={n} from "
          f"{[s['file'] for s in sources]} (perm seed {args.perm_seed})")
    common = None if args.no_common else args.common_test_frac
    for text in args.ratios.split(","):
        ratio = parse_ratio(text)
        man = write_ratio(args.out_base, arrays, n, ratio, args.perm_seed,
                          common_frac=common, sources=sources)
        c = man["counts"]
        print(f"[make_split_ratio] {text}: train {c['train']} / val {c['val']} "
              f"/ test {c['test']} -> {man['out_dir']}")
    print("[make_split_ratio] done")


if __name__ == "__main__":
    main()
