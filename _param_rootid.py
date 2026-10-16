import os
import sys

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from assembly import build_unified_model  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "unified.yaml"),
                          encoding="utf-8"))
model = build_unified_model(cfg, os.path.join(REPO, cfg["weights"]["spectrum"]),
                            device="cpu")

root = [(n, tuple(p.shape), p.requires_grad, p.numel())
        for n, p in model.named_parameters() if "." not in n]
print("root-level parameters (not inside a submodule):")
for n, shp, rg, num in root:
    print(f"  {n:28s} shape={shp} requires_grad={rg} n={num}")
print(f"  submodule sum (named_children, trainable) = "
      f"{sum(p.numel() for m in model.children() for p in m.parameters() if p.requires_grad):,}")
print(f"  full model.parameters() trainable = "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"  full model.parameters() total     = "
      f"{sum(p.numel() for p in model.parameters()):,}")

# where do they come from in the code?
import inspect  # noqa: E402
import assembly  # noqa: E402
src = inspect.getsource(assembly)
for line_no, line in enumerate(src.splitlines(), 1):
    if "Parameter" in line and "(1, 1" in line.replace(" ", ""):
        print(f"  assembly.py:{line_no}: {line.strip()}")
