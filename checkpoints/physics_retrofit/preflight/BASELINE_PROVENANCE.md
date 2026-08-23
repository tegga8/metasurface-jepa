# Baseline provenance record — Physics-Guided Masked Retrofit, Phase 1 (Preflight)

Recorded: 2026-08-23. Machine: local dev workstation (RTX 3050 4 GB — dev-only per
AGENTS.md Compute Environment; CPU-only torch build installed). All hashes are SHA-256.

## 1. Code state

| Item | Value |
|---|---|
| Repository | `tegga8/metasurface-jepa` (origin), branch `main` |
| Git commit SHA | `9774b7becc7c9ca3dd371d74d19adb81bc89ebf9` |
| Working tree | clean (`git status --porcelain` = 0 lines) at record time |
| Baseline config | `configs/milestone_b.yaml` — SHA-256 `3ffc926d375659abc549e3bf9dd7ae68b0166ce3cdbbaf9a7a9bd1c81b5cdb17` |
| Architecture test suite | 230 passed / 6 skipped (CUDA-only skips), pytest, this commit |

## 2. JPA checkpoint — **DOES NOT EXIST (blocking finding)**

- **Exact JPA checkpoint path:** none found on this machine.
- **Checkpoint file SHA-256:** N/A.
- **Resumable or raw-state-only:** N/A.

An exhaustive recursive search of the repository for `*.pt`, `*.pth`, `.bin`, `*.ckpt`,
`*.tar` model artifacts found only:

| File | Size | Role |
|---|---|---|
| `data/metadit/weights/metadit-small.bin` | 141.95 MB | released MetaDiT ViT init (frozen reference) |
| `data/metadit/weights/spec_encoder.pth` | 59.54 MB | released spectrum encoder (frozen) |
| `data/metadit/weights/surrogate_model.bin` | 24.24 MB | released ConvSurrogate (frozen) |
| `checkpoints/milestone_b/physics_validation/latent_geometry_probe_weights.pt` | 0.05 MB | linear probe weights (diagnostic only, not a JPA model) |

This confirms and extends the standing finding in
`checkpoints/milestone_b/physics_validation/PHYSICS_TARGET_SELECTION_TRAINED_REPORT.md`
("BLOCKED — no genuine trained checkpoint exists on this machine … zero checkpoints
satisfy §6's genuine-trained criteria"):

- The only genuine cloud-trained JPA checkpoint (Kaggle, `minimal_jepa_latest.pt`,
  step 2687) exhibited the collapsed EMA-target representation
  (`checkpoints/milestone_b/REPORT.md`, "CURRENT STATUS — BLOCKED"). It was explicitly
  retired ("The collapsed checkpoint is not reused.") and is not present locally.
- All local smoke/adaptive/synthetic-proxy checkpoints referenced by older reports
  (`minimal_smoke_latest.pt`, `synthetic_collapsed.pt`, `adaptive/phase_00_jepa_*.pt`, …)
  have since been removed from disk; they were smoke-scale anyway and are excluded as
  baseline candidates by the wired-in checkpoint-validity guards
  (`scripts/diagnostics/checkpoint_provenance_audit.py`,
  `validate_checkpoint_provenance()`).

**Consequence for Phase 1:** work items requiring a candidate JPA checkpoint —
target-representation health validation (item 3) and Gate-1 physics-utility
reproduction on the trained conditioning path (item 5), plus the representation-
separation half of Gate-2 — are **BLOCKED pending a genuine trained checkpoint**.
Per the existing directive (same PHYSICS_TARGET_SELECTION report): run the genuine
Milestone-B training on Kaggle per `CLOUD_TRAINING.md`, sync the winning HEALTHY
§30 checkpoint back into the repo, then complete those items. Per the Phase-1
document itself: do NOT train a decoder against a collapsed or unavailable target.

## 3. Frozen released components (verified present)

| Component | Path | SHA-256 |
|---|---|---|
| Spectrum encoder | `data/metadit/weights/spec_encoder.pth` | `21b833d0d46fb5a4f5c17a859f914c0f90216f0156679465ae20f613359254d9` |
| Forward EM surrogate | `data/metadit/weights/surrogate_model.bin` | `765fbf5e5d73a036fb17b70ac0648ad20f40786e528a54b54371239073e5ce98` |
| MetaDiT ViT init | `data/metadit/weights/metadit-small.bin` | `48289374f45db1421acceb00e0b3383b655a6f816f0ddfdaf16af0f7709852d4d` |

## 4. Dataset splits

| Split | Path | SHA-256 |
|---|---|---|
| Train | `data/metadit/split_data/train_set.mat` | `1c0cf8eaaf253df91744bec6419d4510913238a992f2a1df0001f8ee673d456d` |
| Validation | `data/metadit/split_data/val_set.mat` | `6cf1bb67d6381c6eee4e7e594bcc912cca49a5fd8bf0f2ab634873e097388a8b` |
| Test | `data/metadit/split_data/test_set.mat` | `7d7f4abf53b2e6beae022c6eedbd0d737f79112d75cf63d0acfab5a492e1dc1b` |

Tensor conventions verified in `src/data/dataset.py`: `G` = 3×64×64
(ch0 = r_atom/5 on occupied pixels, ch1 = h_atom on occupied pixels, ch2 = l_lattice/3
everywhere), `S` = 2×301 ([real; imag]).

## 5. Environment (local dev)

| Item | Value |
|---|---|
| Python | 3.11.0 |
| PyTorch | 2.13.0+cpu |
| CUDA (local) | not available (CPU-only build; training is cloud-side per AGENTS.md) |
| numpy / scipy / pyyaml | 2.4.6 / 1.17.1 / 6.0.3 |
| OS | Windows (win32), PowerShell 5.1 |

Cloud runs (Kaggle/Colab) must record their own versions in the training report when
the genuine Milestone-B checkpoint is produced.

## 6. Checkpoint acceptance contract (for the future synced-back checkpoint)

When the genuine checkpoint lands in this repo it must be:

1. A full §30 resumable dict (`save_checkpoint`: objective_name, objective_state,
   optimizer state + param-shape ownership, scheduler, EMA counters, RNG states, cfg,
   step, best) — raw state dicts are rejected as baselines;
2. Accepted by `validate_checkpoint_provenance()` (not smoke/near-init/synthetic);
3. HEALTHY-gated: target-representation health verdict from
   `scripts/diagnostics/check_ema_target_diversity.py` must be non-degenerate against
   both anchors (released-ViT reference AND collapsed anchor), per the two-anchor
   convention in `src/diagnostics/representation_health.py`.

This file freezes the pre-training provenance baseline. It must be updated (not
replaced) when the checkpoint arrives, appending §7 with the checkpoint's own
provenance block.
