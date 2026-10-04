# AGENTS.md — Unified Occupancy–Parameter–Spectrum JEPA (metasurface-jepa)

Operational playbook for any coding agent working on this repository. It does not restate the
architecture — that lives in `docs/implementation/unified_jepa/architecture_v5.md`
(architectural authority), with `00_MASTER_EXECUTION.md` as execution controller and
`01_…`–`05_…` as the phase instructions. All five phases are implemented; see
`IMPLEMENTATION_REPORT.md` (build record) and `AUDIT_REPORT_192D.md` (retirement + defect audit).

## Authority hierarchy

1. `docs/implementation/unified_jepa/architecture_v5.md` — architecture and design rationale.
2. `docs/implementation/unified_jepa/00_MASTER_EXECUTION.md` — execution controller.
3. `docs/implementation/unified_jepa/01_…`–`05_…` — phase-level instructions.
4. The repository itself — exact APIs, interfaces, conventions, and config keys come from code
   and tests, never from memory or assumption.

For every implementation detail not fixed by the MDs, resolve it from the repository first. If a
phase MD conflicts with the repository and `architecture_v5.md` does not resolve the conflict,
stop and report the exact conflict instead of inventing a workaround.

## Single active architecture (post 2026-09-13 retirement)

The legacy 384-D "Milestone-B" path (3-channel broadcast geometry input, 384-D hidden/predictor,
the `jepa_vicreg`/Barlow/LeJEPA objective ladder, the phase-1 decoder, `train_milestone_b.py`)
was **retired by operator decision on 2026-09-13 and deleted**. Exactly one live architecture
remains:

- **Unified 192-D JEPA** — single-channel occupancy + explicit scalar parameters
  (scalar-encoder FiLM + scalar-summary token) + conditioned spectrum:
  `src/assembly.py::UnifiedJEPA` / `build_unified_model`, `src/encoders/occupancy_encoder.py`,
  `src/encoders/scalar_encoder.py`, `src/fusion/fusion_encoder.py`,
  `src/decoders/occupancy_decoder.py`, `src/decoders/scalar_decoder.py`,
  `src/losses/unified_losses.py`, `src/physics/physics_loop.py`,
  `src/data/factorize.py`, `src/data/scalar_mask.py`, `scripts/train/train_unified.py`,
  `scripts/eval/eval_scenarios.py`, `configs/unified.yaml`.
- The `[3,64,64]` broadcast tensor exists **only** at the frozen MetaDiT surrogate boundary
  (`assemble_metadit_geometry`). It is never a representation to mask or predict against.
- Frozen components, never trained and never restarted: the released spectrum encoder
  (`SpectrumPath`), the EM surrogate, both EMA targets (`ema`, `scalar_mlp_ema`).
- `scalar_mlp_ema` is target-side FiLM conditioning only — never decoded, never a loss target.
- Historical 384-D design material remains under `checkpoints/**` and `docs/design_doc.md`
  (banner: historical/superseded). Do not resurrect code from it; the retired tree is
  recoverable from git at commit `f557eb6`.

## Standing rules (always in effect)

1. **One change, one commit.** Each behavior change lands as its own commit with a regression
   test that fails before the fix and passes after. Never bundle unrelated refactors.
2. **No unmotivated mechanisms.** Do not add a loss term, module, or architecture piece without
   a specific observed failure it addresses (cite the check in `architecture_v5.md` §8 or the
   audit report). "This might help" is not sufficient.
3. **Stop and ask on ambiguity.** If a spec section is ambiguous or a threshold is unspecified,
   ask the human operator rather than guessing.
4. **Re-verify external assumptions.** MetaDiT paths/APIs and released-weight formats are
   hypotheses to re-check against the live `external/metadit` tree before relying on them.
5. **Local machine is dev-only.** The local machine (RTX 3050, 4 GB VRAM / 16 GB RAM) is for
   code, unit tests, shape/forward smokes on tiny batches, and debugging. All gradient-based
   training runs happen on the cloud GPU (Kaggle/Colab) per `CLOUD_TRAINING.md`. Never silently
   shrink model sizes to fit local VRAM — that is a deviation to flag.
6. **Tests are the gate.** `python -m pytest tests/ -q --tb=line` must be green (skips for
   missing CUDA / not-staged dataset are expected locally). Run it before declaring anything done.
7. **Data-dependent tests skip loudly, never pass silently.** Tests needing dataset splits or
   released weights must guard with an explicit skip reason.
8. **No scientific claim from "the code runs".** The unified path has no trained checkpoint yet;
   per-scenario evaluation (`eval_scenarios.py`) on the hard stratum — full occupancy mask + all
   scalars unknown — is the acceptance gate. Pooled metrics are not a gate.
9. **Dated operator overrides.** Any decision that deviates from the MDs is recorded here with a
   date so later sessions do not re-derive or reverse it (see below).

## Dated operator overrides

- **2026-09-13 — legacy 384-D path retired (deletion, not archiving).** Operator directive:
  "retire the old 384d and go through the newer 192d with the scalar architecture", with "use
  this commit only, the newer ones are broken" (pinned at `f557eb6`). This supersedes the
  legacy-retention clauses in `00_MASTER_EXECUTION.md` ("Keep old Milestone-B code/checkpoints as
  reproducible historical reference"), `03_training_and_objective.md` ("Keep the old
  Milestone-B trainer intact…"), and `05_tests_evaluation_cleanup.md` ("Do not delete the old
  Milestone-B implementation…"). Historical reports under `checkpoints/**` and
  `docs/design_doc.md` are kept; all legacy code, scripts, tests, and configs are deleted.
- **2026-09-13 — goal-dropped steps skip the physics term.** On null-goal (CFG dropout) batches,
  `L_phys` is skipped: its target is the sample's true spectrum, i.e. the very condition that was
  dropped, so training it there would push the unconditional branch toward outputs it cannot
  infer (goal-ignoring / mode-collapse pressure). Latent (VICReg) and scalar objectives still
  train that branch.
- **2026-09-13 — decode-time FiLM takes explicit known/unknown flags.** The occupancy decoder's
  scalar conditioning uses the same 6-dim `[value, known-flag]` convention as the scalar encoder
  (`architecture_v5.md` §3.2), so a known value and a predicted value are distinguishable.
- **2026-10-04 — Phase 6 capacity study: width/depth re-opened (scoped exception).** Operator
  directive: "check if there are any other models or dataset which we can work on — shrink the
  model size or increasing". `architecture_v5.md` §11 ("occupancy latent dimension and depth …
  not reopened as a general hyperparameter search") is scoped to **this controlled study only**:
  three variants — S1 small (`hidden 128 / geo 4 / pred 6`), S2 slim (`pred 4`), L1 wide
  (`hidden 256`) — each scored by the canonical battery at 70k × 3 seeds against the existing
  full-fidelity baseline. Not a general search; no variant ships without its recorded gate result.
- **2026-10-04 — split-ratio study: the dataset is re-split (data-contract change).** Operator
  directive: "no change in the dataset size … train and test split like 50:25:25 or 60:20:20 or
  40:40:20 rather than 80/20". The released 80:10:10 split remains the paper-referenced protocol;
  the re-splits are a separate ladder (`scripts/data/make_split_ratio.py`; one permutation,
  nested prefixes, a shared 20 % test slice) whose numbers are **not MetaDiT-comparable**. Arms
  are driven by per-run configs only (absolute `data.*_split` paths, `data.test_split` honored by
  the benchmark); the shipped `configs/unified.yaml` paths are unchanged.

## Repo layout

```
repo/
  AGENTS.md                     # this file
  CLOUD_TRAINING.md             # canonical Kaggle/Colab runbook (setup, sync, resume)
  configs/unified.yaml          # the shipped training config (architecture_id: unified_oc_…_v1)
  configs/scaling/              # Phase-6 model-size variants (S1 small / S2 slim / L1 wide)
  docs/
    design_doc.md               # historical (v2, 384-D era) — superseded
    implementation/unified_jepa/ # architecture_v5.md + 00-05 MDs + reports (authority)
  external/metadit/             # released MetaDiT reference implementation (read-only)
  data/metadit/                 # dataset splits + released weights (staged, not committed)
  src/                          # the unified model (see "Single active architecture")
  scripts/
    train/train_unified.py      # standalone CLI trainer (--config/--resume/--device/…)
    eval/eval_scenarios.py      # per-scenario evaluation (A/B/C reported separately)
    data/make_split_ratio.py    # Phase-6 split-ratio re-splits (nested prefixes, one permutation)
    run_scenarios.py            # smoke-only scenario runner (synthetic data)
    diagnostics/protocol_v1/    # protocol-v1 diagnostic steps (unified path)
    diagnostics/run_guidance_gap_sweep.py
    preflight/repo_static_audit.py
  checkpoints/                  # training outputs (checkpoints/unified/) + historical reports
  tests/                        # unified + shared-contract suites
```

## Compute environment (decided — do not re-litigate)

- **Local:** dev-only (see Standing Rule 5).
- **Cloud:** Kaggle T4/P100 16 GB, or Colab T4/A100 — all gradient-based training, and any
  evaluation needing real batch sizes. `CLOUD_TRAINING.md` is the single canonical workflow; do
  not re-derive cloud steps inline anywhere else.
- **Session loop:** (1) coding session writes/updates code + config, commits; (2) cloud session
  pulls and runs per `CLOUD_TRAINING.md`, producing `checkpoints/unified/` artifacts; (3) a new
  coding session pulls, reads the produced reports/metrics, and decides next steps. An agent must
  not self-certify a training result from a session that did not run the training.

## Validation gates (what "done" means)

Implemented from `architecture_v5.md` §8:

- real/null/**shuffled** comparisons stratified by regime — the gate is the **hard stratum**
  (full occupancy mask + all scalars unknown): real must beat shuffled in physics-consistency.
  Never report a pooled gap.
- scalar-dependence stratified the same way (varying scalars must change outputs *correctly*,
  not merely change them).
- occupancy collapse: IoU/F1 on the occupied class and predicted occupancy-fraction variability
  — not pixel accuracy.
- generative diversity: perturbing the target spectrum must move the decoded design.
- §8.2 diagnostic re-runs on the 192-D encoder (ported; see audit report deferrals).

## If something fails

1. Stop; do not add mechanisms to force a gate to pass.
2. Consult `architecture_v5.md` §8/§13-style failure framing and `AUDIT_REPORT_192D.md`.
3. Record the failure and observed numbers in the relevant report before deciding:
   (a) diagnose and retry within scope, (b) escalate to the operator for a scope/threshold
   decision, or (c) stop the line if the failure criteria are met.
4. Never respond to a failed gate by silently loosening the gate.
5. Compute/environment failures (OOM, session disconnect, staging problems) are operational —
   consult `CLOUD_TRAINING.md` (resume, reduce batch size, switch platform), not a research
   conclusion.
