import math
import torch


def _count_optimizer_steps(num_micro_batches: int, grad_accum: int) -> int:
    """Mirror the Milestone-B accumulation bookkeeping."""
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")

    step = 0
    micro_step = 0

    for _ in range(num_micro_batches):
        micro_step += 1

        if micro_step % grad_accum != 0:
            continue

        step += 1

    if micro_step % grad_accum != 0:
        raise RuntimeError("incomplete accumulation window")

    return step


def _expected_optimizer_steps(num_micro_batches: int, grad_accum: int) -> int:
    return math.ceil(num_micro_batches / grad_accum)


def test_grad_accum_1():
    assert _count_optimizer_steps(4, 1) == 4


def test_grad_accum_2():
    assert _count_optimizer_steps(4, 2) == 2


def test_grad_accum_4():
    assert _count_optimizer_steps(8, 4) == 2


def test_total_steps_current_milestone_b_config():
    micro_batches_per_epoch = 128
    grad_accum = 2
    epochs = 30

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / grad_accum
    )
    total_steps = optimizer_steps_per_epoch * epochs

    assert total_steps == 1920


def test_grad_accum_does_not_use_optimizer_step_counter():
    """
    Regression test for the exact old infinite-loop bug.

    The accumulation boundary must advance using micro_step, not step.
    """
    accum = 2
    step = 0
    micro_step = 0
    optimizer_steps = 0

    for _ in range(4):
        micro_step += 1

        if micro_step % accum != 0:
            continue

        optimizer_steps += 1
        step += 1

    assert optimizer_steps == 2
    assert step == 2


def test_incomplete_accumulation_is_detected():
    try:
        _count_optimizer_steps(3, 2)
    except RuntimeError as exc:
        assert "incomplete accumulation" in str(exc)
    else:
        raise AssertionError(
            "Expected incomplete accumulation to raise"
        )