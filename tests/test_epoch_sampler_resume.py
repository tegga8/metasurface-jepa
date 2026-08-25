"""A5 — DeterministicEpochSampler resume-determinism contract.

Covers:
1. Same (seed, epoch) -> identical permutation, across repeated constructions
   AND across separate OS processes (the cloud-resume guarantee).
2. set_epoch advances the stream reproducibly (epoch N permutation is a pure
   function of (seed, N)).
3. Skip-k equivalence: consuming batches [k:] of the full epoch sees exactly
   the item sequence a mid-epoch-resumed run must observe — i.e., the sampler
   carries no hidden per-instance state that a fresh construction would lose.
4. Production wiring: train_milestone_b.py must construct this sampler with
   DataLoader(shuffle=False) — any shuffle=True would silently break exact
   resume (Bug #3 class).
5. Audit hygiene: the sampler module seeds via explicit torch.Generator only,
   never global np.random.seed / torch.manual_seed.
"""

import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def _perm(seed, epoch, n):
    from data.epoch_sampler import DeterministicEpochSampler
    s = DeterministicEpochSampler(n, seed=seed, epoch=epoch)
    return list(iter(s))


def test_same_seed_epoch_identical_permutation():
    """Repeated constructions with the same (seed, epoch) agree exactly."""
    for seed in (0, 7, 12345):
        for epoch in (0, 1, 5):
            n = 37
            p1 = _perm(seed, epoch, n)
            p2 = _perm(seed, epoch, n)
            assert p1 == p2, f"permutation differs across constructions ({seed},{epoch})"
            assert sorted(p1) == list(range(n)), "permutation must cover all indices"


def test_epoch_advances_stream_reproducibly():
    """Epoch N is a pure function of (seed, N); different epochs differ."""
    p0a = _perm(11, 0, 23)
    p0b = _perm(11, 0, 23)
    p1 = _perm(11, 1, 23)
    p1b = _perm(11, 1, 23)
    assert p0a == p0b and p1 == p1b
    assert p0a != p1, "different epochs must yield different permutations"
    # set_epoch on a live instance must match a fresh construction at that epoch
    from data.epoch_sampler import DeterministicEpochSampler
    live = DeterministicEpochSampler(23, seed=11, epoch=0)
    consumed_first = list(iter(live))
    assert consumed_first == p0a
    live.set_epoch(1)
    assert list(iter(live)) == p1, "set_epoch must reproduce the fresh-construction stream"


def test_skip_k_equivalent_to_fresh_construction():
    """Resume semantics: skipping the first k yielded items leaves the remaining
    sequence identical to items [k:] of the canonical permutation."""
    seed, n, k = 99, 40, 13
    canonical = _perm(seed, 3, n)
    # simulate a resumed consumer: iterate, drop first k, keep the rest
    from data.epoch_sampler import DeterministicEpochSampler
    s = DeterministicEpochSampler(n, seed=seed, epoch=3)
    it = iter(s)
    for _ in range(k):
        next(it)
    rest = list(it)
    assert rest == canonical[k:], (
        "skip-k consumption must equal the canonical tail — a resumed run "
        "would otherwise see different data than the uninterrupted run")


def test_sampler_is_a_pure_function_of_seed_and_epoch():
    """No dependence on ambient global RNG state: perturb the global streams,
    then confirm the permutation is unchanged."""
    seed, n = 5, 31
    baseline = _perm(seed, 2, n)
    torch.manual_seed(987654)
    import numpy as np
    np.random.rand(3)
    import random as _random
    _random.random()
    after = _perm(seed, 2, n)
    assert baseline == after, (
        "sampler output changed after global RNG perturbation — it is reading "
        "ambient RNG state instead of its own seeded generator")


def test_production_script_uses_deterministic_sampler_no_shuffle():
    """Static wiring check: the production driver must use
    DeterministicEpochSampler + DataLoader(shuffle=False)."""
    path = os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py")
    with open(path, "r") as f:
        src = f.read()
    assert "DeterministicEpochSampler" in src, \
        "production driver must use DeterministicEpochSampler"
    # find the DataLoader construction and require shuffle=False there
    idx = src.find("DataLoader(")
    assert idx != -1, "no DataLoader construction found in production driver"
    call = src[idx:src.find(")", idx) + 1]
    assert "shuffle=False" in call, \
        "DataLoader must be constructed with shuffle=False when using a sampler"


def test_production_script_fork_rng_excludes_cuda():
    """fork_rng(devices=[]) for iterator creation — only CPU RNG isolation is
    needed; forking every visible CUDA device triggers a warning."""
    path = os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py")
    with open(path, "r") as f:
        src = f.read()
    assert "torch.random.fork_rng(devices=[])" in src, \
        "production driver must use fork_rng(devices=[]) for iterator creation"


def test_process_level_permutation_stability():
    """Two independent OS processes produce byte-identical permutations for the
    same (seed, epoch) — the actual property cloud resume depends on."""
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s')\n"
        "from data.epoch_sampler import DeterministicEpochSampler\n"
        "s = DeterministicEpochSampler(64, seed=2026, epoch=1)\n"
        "print(','.join(map(str, iter(s))))\n"
    ) % (REPO_ROOT, os.path.join(REPO_ROOT, "src"))
    outs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1] and outs[0], \
        "cross-process permutation mismatch"


def test_sampler_module_has_no_global_seeding():
    """Audit hygiene: epoch_sampler.py must not touch global seeding APIs."""
    path = os.path.join(REPO_ROOT, "src", "data", "epoch_sampler.py")
    with open(path, "r") as f:
        src = f.read()
    for forbidden in ("np.random.seed(", "torch.manual_seed(", "random.seed("):
        assert forbidden not in src, f"global seeding found in epoch_sampler.py: {forbidden}"


if __name__ == "__main__":
    test_same_seed_epoch_identical_permutation()
    test_epoch_advances_stream_reproducibly()
    test_skip_k_equivalent_to_fresh_construction()
    test_sampler_is_a_pure_function_of_seed_and_epoch()
    test_production_script_uses_deterministic_sampler_no_shuffle()
    test_production_script_fork_rng_excludes_cuda()
    test_process_level_permutation_stability()
    test_sampler_module_has_no_global_seeding()
    print("PASS: all A5 epoch-sampler resume tests")
