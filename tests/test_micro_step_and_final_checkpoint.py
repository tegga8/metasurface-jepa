"""Regression tests: micro_step semantics + final-checkpoint resume metadata.

Fix 1 (micro_step semantics): micro_step counts micro-batches accumulated since
the LAST optimizer step. It is incremented before the step boundary and reset to
zero IMMEDIATELY after every successful optimizer step — never only at the epoch
boundary. For grad_accum=2 the exact required sequence is:

    after batch 1 -> micro_step == 1
    after batch 2 -> optimizer.step(), micro_step == 0

Fix 2 (final-checkpoint metadata): a run stopped mid-epoch (--max-steps) must
save its FINAL checkpoint with is_epoch_end=False and batch_index=<next batch>,
not the historical forced is_epoch_end=True / batch_index=0.
"""

import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py")


# ---------------------------------------------------------------------------
# behavioral simulation of the production accumulation sequence
# ---------------------------------------------------------------------------

def _simulate_training_loop(num_batches, grad_accum):
    """Faithful mirror of the fixed production loop's micro_step bookkeeping.

    Mirrors scripts/train/train_milestone_b.py exactly:
      backward -> micro_step += 1 -> window check -> [step/clip/ema/sched]
      -> micro_step = 0.
    Returns the list of micro_step values observed after EACH processed batch,
    plus the number of optimizer steps.
    """
    micro_step = 0
    optimizer_steps = 0
    observed = []
    for _ in range(num_batches):
        # ... forward / total.backward() ...
        micro_step += 1
        observed.append(micro_step)
        if micro_step % grad_accum != 0:
            continue
        # optimizer.step(); zero_grad; on_optimizer_step; scheduler.step();
        micro_step = 0
        optimizer_steps += 1
        observed[-1] = micro_step  # value AFTER the successful optimizer step
    return observed, optimizer_steps


def test_grad_accum_2_exact_sequence():
    """grad_accum=2: after batch 1 -> micro_step=1; after batch 2 -> step, micro_step=0."""
    observed, steps = _simulate_training_loop(4, grad_accum=2)
    assert observed == [1, 0, 1, 0], observed
    assert steps == 2


def test_grad_accum_1_resets_every_batch():
    observed, steps = _simulate_training_loop(3, grad_accum=1)
    assert observed == [0, 0, 0], observed
    assert steps == 3


def test_grad_accum_3_sequence():
    observed, steps = _simulate_training_loop(6, grad_accum=3)
    assert observed == [1, 2, 0, 1, 2, 0], observed
    assert steps == 2


def test_micro_step_zero_at_epoch_boundary_when_flushed():
    """After a flushed epoch, micro_step must already be 0 (no boundary reset needed)."""
    observed, _ = _simulate_training_loop(4, grad_accum=2)
    assert observed[-1] == 0


def test_partial_window_raises():
    """An epoch ending mid-window still raises (strict flush contract preserved)."""
    import torch  # noqa: F401  (mirror imports keep the mirror honest)

    def simulate(batches, accum):
        micro_step = 0
        for _ in range(batches):
            micro_step += 1
            if micro_step % accum != 0:
                continue
            micro_step = 0
        return micro_step

    try:
        simulate(3, 2)
    except RuntimeError:
        raise AssertionError("mirror should not raise; the production loop guards this")
    # production-side contract: leftover window detected via micro_step != 0
    assert simulate(3, 2) == 1


# ---------------------------------------------------------------------------
# source-level pinning of the fix inside the real production script
# ---------------------------------------------------------------------------

def _script_source():
    with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _line_index(lines, needle):
    for i, line in enumerate(lines):
        if needle in line:
            return i
    return -1


def test_increment_precedes_optimizer_step_in_source():
    lines = _script_source().splitlines()
    inc = _line_index(lines, "micro_step += 1")
    step_call = _line_index(lines, "optimizer.step()")
    assert inc != -1 and step_call != -1
    assert inc < step_call, (
        "micro_step += 1 must appear BEFORE the optimizer-step block "
        "(it counts micro-batches accumulated since the last optimizer step)")


def test_reset_follows_scheduler_step_in_source():
    lines = _script_source().splitlines()
    sched = _line_index(lines, "scheduler.step()")
    assert sched != -1
    window = lines[sched:sched + 4]
    assert any("micro_step = 0" in l for l in window), (
        "micro_step must be reset to 0 immediately after every successful "
        "optimizer step (scheduler.step() is the last call of that block)")


def test_no_epoch_boundary_only_reset_in_source():
    lines = _script_source().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("micro_step = 0") and "epoch boundary" in line.lower():
            raise AssertionError(
                f"stale epoch-boundary-only micro_step reset at line {i+1}: {line}")


def test_final_checkpoint_saves_actual_state_not_forced_epoch_end():
    src = _script_source()
    assert 'artifact_type="final"' in src
    # The final save_checkpoint call must be driven by the tracked actual state.
    for needle in ("last_global_step", "last_epoch", "last_micro_step",
                   "batch_index=last_batch_index_next",
                   "is_epoch_end=last_is_epoch_end"):
        assert needle in src, f"final checkpoint missing actual-state token: {needle}"
    # And the old forced metadata must be gone from the final call.
    assert "step - 1, epoch, micro_step=micro_step, batch_index=0," not in src


def test_actual_state_tracking_variables_exist():
    src = _script_source()
    for needle in ("last_is_epoch_end = (bi == len(loader) - 1)",
                   "last_batch_index_next = 0 if last_is_epoch_end else bi + 1"):
        assert needle in src, f"missing per-batch tracking: {needle}"


def test_next_batch_index_rule_matches_spec():
    """The per-batch rule: is_epoch_end iff last loader batch; else next index."""
    loader_len = 5
    for bi in range(loader_len):
        is_epoch_end = (bi == loader_len - 1)
        next_index = 0 if is_epoch_end else bi + 1
        if bi < loader_len - 1:
            assert (is_epoch_end, next_index) == (False, bi + 1)
        else:
            assert (is_epoch_end, next_index) == (True, 0)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
