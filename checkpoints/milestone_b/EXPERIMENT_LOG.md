# Milestone B — Experiment Log (scientific screening history)

Chronological record of the adaptive anti-collapse screening (Kaggle runs and their
outcomes). Code-correctness items live in `BUGLOG.md`; this file records the
experimental kinematics — hyperparameters, health trajectories, collapse votes,
cos_err, and decisions per phase.

Never overwrite or edit past entries; append new phases below. Original failed runs
are retained as evidence, not rewritten as blanket claims.

## Phase 0 — L_J (EMA-JEPA baseline) — Kaggle, real run

- starting checkpoint: jepa base initialization
- step range: 100–300+
- cos_err trajectory: very small (~0.000261) despite collapse
- health trajectory: WARNING @ step 100 (2 votes), COLLAPSED @ step 200 (3 votes),
  COLLAPSED @ step 300
- collapse votes: escalating (2 -> 3)
- final status: COLLAPSED with negligible prediction error —

  **Interpretation (per screening directive):** persistent representation collapse
  (cross-sample cosine ≈ 1, effective rank very low, dominant eigenvalue large →
  dimensional/redundancy collapse), NOT a prediction-quality failure.

## Phase 1 — prediction-only VICReg (`jepa_vicreg`, historical single-branch) — Kaggle, real run

- starting checkpoint: reset from base initialization
- hyperparameters: lambda_var=0.1, lambda_cov=0.04, gamma=1.0
- step range: 400–500
- health trajectory: COLLAPSED @ step 400, COLLAPSED @ step 500
- cos_err: ≈ 0.00324
- final status: not healthy under the tested configuration and short phase budget

  **Interpretation (per screening directive):** the existing prediction-only VICReg
  variant did not remain healthy under lambda_var=0.1, lambda_cov=0.04, gamma=1.0 and
  a short phase budget. NOT a blanket claim that VICReg fails — the corrected
  branch-symmetric form (`jepa_vicreg2`) is the next controlled test.

## Phase 2 — LeJEPA/SIGReg (`lejepa`) — Kaggle, first real CUDA execution

- status: **NOT scientifically evaluated** — the first real CUDA execution crashed
  from a CPU/CUDA device mismatch (z on CUDA, slice directions U and randperm
  subsampling indices created on default CPU) before the objective could run.
- local fix applied and tested: `src/losses/sigreg.py` now creates the generator,
  slice directions, and subsample indices on `z.device` explicitly (SIGReg math,
  seed behavior, num_slices/num_points unchanged). 6 passed / 5 CUDA-only skipped
  locally (`tests/test_sigreg.py`).
- cloud evaluation: pending — do not resume the crashed phase; start `lejepa` fresh
  from base initialization per the screening directive.

## Planned screening ladder (global 800-step budget, max_total_steps: 800, val @ 50)

Phases to be appended below as they run (fresh base-initialization start per method;
never from a collapsed checkpoint):

1. `jepa`            — EMA-JEPA baseline (rerun under screening budget)
2. `jepa_var`        — variance floor only
3. `jepa_vicreg`     — historical prediction-only VICReg (unchanged, for comparison)
4. `jepa_vicreg2`    — corrected branch-symmetric VICReg
5. `jepa_barlow`     — Barlow-style redundancy reduction
6. `lejepa`          — SIGReg (fresh start, post device fix)

## Pre-training verification state (2026-08-18, operator-directed continuation)

No new Kaggle screening phases have run since Phase 2. The ladder above remains the pending
decision run. Before it launches, the pre-training cleanup was verified locally (smoke scale,
4 samples — verification only, NOT a scientific screening result):

- Two repeated six-objective smoke runs (`adaptive/_smoke_A/`, `adaptive/_smoke_B/`) are
  reproducible: identical health trajectories (HEALTHY, votes=0), identical ladder behavior,
  identical winner (jepa_vicreg2, best_healthy 0.061089852593553255), identical final
  winner evaluation, byte-identical run artifacts, semantically identical checkpoints
  (engineering detail: `checkpoints/milestone_b/BUGLOG.md` Tier 5).
- Do NOT infer method superiority from this smoke; it establishes determinism, objective
  execution, and validation consistency only. The scientific comparison requires the
  800-step global screening on Kaggle.