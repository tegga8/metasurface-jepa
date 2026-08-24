"""Test device contract utilities (hardening spec §1)."""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from torch import nn
from runtime.device import (
    resolve_device, assert_module_device, assert_tensor_device, assert_same_device, move_to_device
)


def test_resolve_device_auto():
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device)


def test_resolve_device_cpu():
    dev = resolve_device("cpu")
    assert dev.type == "cpu"


def test_resolve_device_cuda():
    dev = resolve_device("cuda")
    assert dev.type == "cuda"
    # Should resolve to current CUDA device


def test_resolve_device_explicit_cuda():
    dev = resolve_device("cuda:0")
    assert dev.type == "cuda"
    assert dev.index == 0


def test_resolve_device_torch_device():
    dev = resolve_device(torch.device("cpu"))
    assert dev.type == "cpu"


def test_assert_module_device_ok():
    m = nn.Linear(4, 4).to("cpu")
    assert_module_device(m, "cpu", "test_module")


def test_assert_module_device_fail():
    m = nn.Linear(4, 4).to("cpu")
    with pytest.raises(RuntimeError, match="test_module parameter"):
        assert_module_device(m, "cuda", "test_module")


def test_assert_tensor_device_ok():
    t = torch.randn(2, 3, device="cpu")
    assert_tensor_device(t, "cpu", "test_tensor")


def test_assert_tensor_device_fail():
    t = torch.randn(2, 3, device="cpu")
    with pytest.raises(RuntimeError, match="test_tensor is on cpu, expected cuda"):
        assert_tensor_device(t, "cuda", "test_tensor")


def test_assert_same_device_ok():
    a = torch.randn(2, 3, device="cpu")
    b = torch.randn(2, 3, device="cpu")
    assert_same_device(a, b, names=["a", "b"])


def test_assert_same_device_fail():
    a = torch.randn(2, 3, device="cpu")
    b = torch.randn(2, 3, device="cpu")
    # Can't test CUDA fail on CPU-only, skip
    if torch.cuda.is_available():
        b = b.cuda()
        with pytest.raises(RuntimeError, match="b is on cuda"):
            assert_same_device(a, b, names=["a", "b"])


def test_move_to_device_module():
    m = nn.Linear(4, 4)
    m2 = move_to_device(m, "cpu")
    assert m2 is m
    assert next(m.parameters()).device.type == "cpu"


def test_move_to_device_tensor():
    t = torch.randn(2, 3)
    t2 = move_to_device(t, "cpu")
    assert t2.device.type == "cpu"


def test_move_to_device_dict():
    d = {"a": torch.randn(2, 3), "b": torch.randn(2, 3)}
    d2 = move_to_device(d, "cpu")
    assert all(v.device.type == "cpu" for v in d2.values())


def test_move_to_device_list():
    lst = [torch.randn(2, 3), torch.randn(2, 3)]
    lst2 = move_to_device(lst, "cpu")
    assert all(v.device.type == "cpu" for v in lst2)


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


if __name__ == "__main__":
    import sys
    # Simple manual test runner
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                print(f"FAIL {name}: {e}")
                sys.exit(1)
    print("All device tests passed")