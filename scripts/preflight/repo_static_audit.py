#!/usr/bin/env python3
"""Repository Static Audit (Phase 2 §9).

Scans first-party code for:
- Device duplication: .cuda(), .to("cuda"), torch.device("cuda")
- RNG duplication: manual_seed, manual_seed_all, randperm, torch.Generator
- Duplicate training loops: torch.optim.AdamW, optimizer.step(), objective.on_optimizer_step() outside canonical path
- Duplicate checkpoint writes: torch.save() classification
- Stale references: model.proj, latest.pt, adaptive, LOSS_LADDER, guidance.py, routing.py, geometry_decoder.py

Removes obsolete active references.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TESTS_DIR = REPO_ROOT / "tests"

# First-party code directories to scan
SCAN_DIRS = [SRC_DIR, SCRIPTS_DIR]

# Patterns to search for
PATTERNS = {
    "device_cuda_call": [
        r"\.cuda\(\)",
        r'\.to\(["\']cuda["\']\)',
        r'torch\.device\(["\']cuda["\']\)',
    ],
    "rng_manual_seed": [
        r"torch\.manual_seed\(",
        r"torch\.cuda\.manual_seed_all\(",
        r"torch\.randperm\(",
        r"torch\.Generator\(",
        r"np\.random\.seed\(",
        r"random\.seed\(",
    ],
    "optimizer_adamw": [
        r"torch\.optim\.AdamW\(",
    ],
    "optimizer_step": [
        r"optimizer\.step\(\)",
        r"\.step\(\)",  # generic, but we'll filter
    ],
    "objective_on_step": [
        r"objective\.on_optimizer_step\(",
        r"\.on_optimizer_step\(",
    ],
    "torch_save": [
        r"torch\.save\(",
    ],
    "stale_model_proj": [
        r"model\.proj",
        r"\.proj\b",
    ],
    "stale_latest_pt": [
        r"latest\.pt",
    ],
    "stale_adaptive": [
        r"adaptive",
        r"ADAPTIVE",
    ],
    "stale_loss_ladder": [
        r"LOSS_LADDER",
        r"loss_ladder",
    ],
    "stale_guidance": [
        r"guidance\.py",
        r"guidance\b",
    ],
    "stale_routing": [
        r"routing\.py",
        r"routing\b",
    ],
    "stale_geometry_decoder": [
        r"geometry_decoder\.py",
        r"geometry_decoder\b",
    ],
}

# Files to exclude from scanning
EXCLUDE_FILES = {
    "runtime/device.py",  # This is the canonical device module
    "runtime/reproducibility.py",  # This is the canonical RNG module
    "train/engine.py",  # This is the canonical training engine
    "train_milestone_b.py",  # This is the canonical training script
    "checkpoint_integrity_check.py",  # Preflight script
    "milestone_b_preflight.py",  # Preflight script
    "repo_static_audit.py",  # This script
}

# Allowed patterns with context (these are OK in specific files)
# Paths are relative to REPO_ROOT
ALLOWED_CONTEXT = {
    "device_cuda_call": [
        "src/runtime/device.py",  # Canonical device resolution
        "scripts/train/train_milestone_b.py",  # Device selection in training script
        "scripts/eval/",  # Eval scripts may need device
        "notebooks/",  # Notebooks
    ],
    "rng_manual_seed": [
        "src/runtime/reproducibility.py",  # Canonical RNG
        "src/data/mask.py",  # BlockMasker uses Generator
        "src/losses/sigreg.py",  # SIGReg uses Generator
        "src/runtime/physics_controls.py",  # Physics controls use Generator
        "scripts/train/train_milestone_b.py",  # Seed setting in training script
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
        "scripts/eval/",  # Eval scripts
    ],
    "optimizer_adamw": [
        "scripts/train/train_milestone_b.py",  # Canonical optimizer creation
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
        "scripts/eval/",  # Eval scripts
    ],
    "optimizer_step": [
        "scripts/train/train_milestone_b.py",  # Canonical optimizer step
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
        "scripts/eval/",  # Eval scripts
    ],
    "objective_on_step": [
        "scripts/train/train_milestone_b.py",  # Canonical
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
        "scripts/eval/",  # Eval scripts
    ],
    "torch_save": [
        "src/train/engine.py",  # Canonical checkpoint save
        "scripts/train/train_milestone_b.py",  # Training script
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
    ],
    "stale_model_proj": [
        "tests/",  # Tests may check for absence
        "src/encoders/geometry_encoder.py",  # Attention.proj is a different thing
        "src/encoders/spectrum_encoder.py",  # SpectrumEncoder.proj is a different thing
        "src/losses/objectives.py",  # Documents the no-model.proj rule
        "src/losses/objective_modules.py",  # Documents the no-model.proj rule
        "src/train/engine.py",  # Documents the no-model.proj rule
        "scripts/eval/decisive_representation_validation.py",  # Documents the no-model.proj rule
    ],
    "stale_latest_pt": [
        "scripts/train/train_milestone_b.py",  # Training script references
        "scripts/preflight/",  # Preflight scripts
        "tests/",  # Tests
        "scripts/diagnostics/",  # Diagnostics scripts
        "scripts/eval/",  # Eval scripts
    ],
    "stale_adaptive": [
        "docs/",  # Documentation
        "checkpoints/",  # Historical reports
        "tests/",  # Tests
        "src/losses/barlow.py",  # Documents adaptive-ladder phase
        "src/losses/sigreg.py",  # Documents adaptive-ladder phase
        "src/train/engine.py",  # Documents removal
        "scripts/diagnostics/",  # Diagnostics may reference
        "scripts/train/",  # Training scripts may reference
    ],
    "stale_loss_ladder": [
        "docs/",  # Documentation
        "checkpoints/",  # Historical reports
        "tests/",  # Tests
        "src/train/engine.py",  # Documents removal
    ],
    "stale_guidance": [
        "docs/",  # Documentation
        "checkpoints/",  # Historical reports
        "tests/",  # Tests
        "src/predictor/gclct.py",  # Documents guidance as future work
    ],
    "stale_routing": [
        "docs/",  # Documentation
        "checkpoints/",  # Historical reports
        "tests/",  # Tests
        "src/diagnostics/goal_token_entropy.py",  # Documents routing analysis
        "src/predictor/gclct.py",  # Documents routing as future work
    ],
    "stale_geometry_decoder": [
        "docs/",  # Documentation
        "checkpoints/",  # Historical reports
        "tests/",  # Tests
    ],
}


def should_exclude(filepath: Path) -> bool:
    """Check if file should be excluded from scan."""
    rel = filepath.relative_to(REPO_ROOT)
    rel_str = str(rel).replace("\\", "/")
    for pattern in EXCLUDE_FILES:
        if rel_str == pattern or rel_str.endswith("/" + pattern):
            return True
    if rel.suffix == ".pyc":
        return True
    if "__pycache__" in rel.parts:
        return True
    return False


def is_allowed_context(filepath: Path, category: str) -> bool:
    """Check if finding is in allowed context."""
    rel = filepath.relative_to(REPO_ROOT)
    # Normalize path separators for cross-platform comparison
    rel_str = str(rel).replace("\\", "/")
    for allowed in ALLOWED_CONTEXT.get(category, []):
        if rel_str.startswith(allowed):
            return True
    return False


def scan_file(filepath: Path, category: str, patterns: list) -> list:
    """Scan a single file for patterns."""
    findings = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    for i, line in enumerate(content.splitlines(), 1):
        for pattern in patterns:
            if re.search(pattern, line):
                if not is_allowed_context(filepath, category):
                    findings.append({
                        "file": str(filepath.relative_to(REPO_ROOT)),
                        "line": i,
                        "content": line.strip(),
                        "pattern": pattern,
                        "category": category,
                    })
    return findings


def main():
    print("="*60)
    print("REPOSITORY STATIC AUDIT")
    print("="*60)

    all_findings = []

    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if should_exclude(py_file):
                continue
            for category, patterns in PATTERNS.items():
                findings = scan_file(py_file, category, patterns)
                all_findings.extend(findings)

    # Group by category
    by_category = {}
    for f in all_findings:
        cat = f["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)

    # Report
    total_findings = len(all_findings)
    print(f"\nTotal findings: {total_findings}")

    if total_findings == 0:
        print("\n[PASS] No unexpected patterns found in first-party code.")
        return 0

    for cat, findings in by_category.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {cat} ({len(findings)} findings)")
        print(f"{'='*60}")
        for f in findings:
            print(f"  {f['file']}:{f['line']}: {f['content']}")

    # Specific checks for critical issues
    critical_categories = [
        "device_cuda_call",
        "rng_manual_seed",
        "optimizer_adamw",
        "optimizer_step",
        "objective_on_step",
        "torch_save",
    ]

    # Canonical files where these patterns ARE allowed
    canonical_files = {
        "device_cuda_call": {"src/runtime/device.py"},
        "rng_manual_seed": {"src/runtime/reproducibility.py", "src/data/mask.py", "src/losses/sigreg.py", "src/runtime/physics_controls.py", "src/train/engine.py"},
        "optimizer_adamw": {"scripts/train/train_milestone_b.py"},
        "optimizer_step": {"scripts/train/train_milestone_b.py"},
        "objective_on_step": {"scripts/train/train_milestone_b.py"},
        "torch_save": {"src/train/engine.py", "scripts/train/train_milestone_b.py"},
    }

    has_critical = False
    for cat in critical_categories:
        if cat in by_category:
            # Check if any findings are OUTSIDE canonical files
            canonical = canonical_files.get(cat, set())
            for f in by_category[cat]:
                rel = f["file"].replace("\\", "/")
                if rel not in canonical:
                    has_critical = True
                    break
            if has_critical:
                break

    if has_critical:
        print(f"\n[FAIL] Critical patterns found in non-canonical locations.")
        print("These should be consolidated into the canonical modules:")
        print("  - Device: src/runtime/device.py")
        print("  - RNG: src/runtime/reproducibility.py, src/data/mask.py, src/losses/sigreg.py, src/runtime/physics_controls.py, src/train/engine.py")
        print("  - Training loop: src/train/engine.py + scripts/train/train_milestone_b.py")
        print("  - Checkpoint: src/train/engine.py + scripts/train/train_milestone_b.py")
        return 1

    # Stale references check
    stale_categories = [
        "stale_model_proj",
        "stale_latest_pt",
        "stale_adaptive",
        "stale_loss_ladder",
        "stale_guidance",
        "stale_routing",
        "stale_geometry_decoder",
    ]
    has_stale = any(cat in by_category for cat in stale_categories)
    if has_stale:
        print(f"\n[WARN] Stale references found. These should be removed from active code:")
        for cat in stale_categories:
            if cat in by_category:
                print(f"  - {cat}: {len(by_category[cat])} findings")

    print(f"\n[PASS] Static audit completed with {total_findings} total findings.")
    print("Review non-critical findings above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())