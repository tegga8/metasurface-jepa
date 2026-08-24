"""Shared runtime device utilities.

Single source of truth for device resolution and device-contract assertions.
All first-party training/evaluation code must use these utilities.
"""

import torch


def resolve_device(device_spec: str | torch.device | None = None) -> torch.device:
    """Resolve a device specification to a concrete torch.device.

    Args:
        device_spec: One of:
            - None or "auto": use CUDA if available, else CPU
            - "cpu": CPU device
            - "cuda": current CUDA device (respects CUDA_VISIBLE_DEVICES)
            - "cuda:N" or torch.device("cuda:N"): explicit CUDA device index
            - torch.device: passed through

    Returns:
        A concrete torch.device. "cuda" resolves to the current CUDA device;
        explicit "cuda:N" is respected.
    """
    if device_spec is None or device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device_spec, torch.device):
        return device_spec

    if isinstance(device_spec, str):
        s = device_spec.strip().lower()
        if s == "cpu":
            return torch.device("cpu")
        if s == "cuda":
            return torch.device("cuda")
        if s.startswith("cuda:"):
            return torch.device(s)

    raise ValueError(f"Unrecognized device specification: {device_spec!r}")


def assert_module_device(module: torch.nn.Module, expected: torch.device | str,
                         module_name: str = "module") -> None:
    """Assert that all parameters and buffers of a module are on the expected device.

    Args:
        module: The module to check.
        expected: Expected device (torch.device or string).
        module_name: Name for error messages.

    Raises:
        RuntimeError: If any parameter or buffer is on a different device.
    """
    expected = resolve_device(expected)
    for name, param in module.named_parameters():
        if param.device != expected:
            raise RuntimeError(
                f"{module_name} parameter {name!r} is on {param.device}, "
                f"expected {expected}"
            )
    for name, buf in module.named_buffers():
        if buf.device != expected:
            raise RuntimeError(
                f"{module_name} buffer {name!r} is on {buf.device}, "
                f"expected {expected}"
            )


def assert_tensor_device(tensor: torch.Tensor, expected: torch.device | str,
                         tensor_name: str = "tensor") -> None:
    """Assert that a tensor is on the expected device.

    Args:
        tensor: The tensor to check.
        expected: Expected device (torch.device or string).
        tensor_name: Name for error messages.

    Raises:
        RuntimeError: If the tensor is on a different device.
    """
    expected = resolve_device(expected)
    if tensor.device != expected:
        raise RuntimeError(
            f"{tensor_name} is on {tensor.device}, expected {expected}"
        )


def assert_same_device(*tensors: torch.Tensor, names: list[str] | None = None) -> None:
    """Assert that all tensors are on the same device.

    Args:
        *tensors: Tensors to check.
        names: Optional names for error messages.

    Raises:
        RuntimeError: If tensors are on different devices.
    """
    if not tensors:
        return
    device = tensors[0].device
    for i, t in enumerate(tensors[1:], 1):
        name = names[i] if names and i < len(names) else f"tensor[{i}]"
        if t.device != device:
            raise RuntimeError(
                f"{name} is on {t.device}, expected {device} (like {names[0] if names else 'tensor[0]'})"
            )


def move_to_device(obj, device: torch.device | str):
    """Move a module, tensor, or dict/list/tuple of tensors to device.

    Args:
        obj: Module, tensor, or nested container of tensors.
        device: Target device.

    Returns:
        The object (for modules) or moved container.
    """
    device = resolve_device(device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj