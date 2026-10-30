"""GPU smoke test for the anosvol path: verify a CUDA session from the WSL-pushed CLI.

Pushed from WSL (`anosvol` account) per the 2026-10-06 account-routing override.
Prints nvidia-smi + a torch CUDA matmul; no repo clone or dataset needed.
"""

import subprocess

subprocess.run("nvidia-smi", shell=True, check=False)
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2048, 2048, device=dev)
    y = x @ x
    print("matmul ok | sum:", float(y.sum()))
except Exception as e:  # noqa: BLE001
    print("ERR", repr(e))
print("GPU SMOKE DONE")
