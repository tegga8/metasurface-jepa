# Physics Target Selection — TRAINED-CHECKPOINT REPORT (validity-fix spec §16)

Date: 2026-08-21.
Scope: rerun of Diagnostic B (`scripts/diagnostics/physics_target_selection.py`)
on a GENUINE trained JEPA/VICReg checkpoint, per the validity-fix directive
("VALIDITY FIX BEFORE ANY NEW PHYSICS OBJECTIVE").

## VERDICT UP FRONT

**BLOCKED — no genuine trained checkpoint exists on this machine. The
trained-model rerun has NOT been performed. No T1–T4 classification is issued,
and the earlier Case-B interpretation remains UNCONFIRMED smoke-state evidence
only.**

Per spec §6/§7/§18: an arbitrary short local training run must NOT be
manufactured to claim a checkpoint, and a smoke-scale state must never be
reported as a trained-model result. This report therefore documents (a) the
exhaustive provenance audit that found zero genuine checkpoints, (b) the
checkpoint-validity guards now permanently wired into the diagnostic so the
refusal is automatic, and (c) the exact conditions under which the rerun can
proceed.

## 1. Checkpoint provenance audit (spec §20 step 1–2)

Audit tool: `scripts/diagnostics/checkpoint_provenance_audit.py`.
Smoke heuristic (recorded in every provenance block):
`step <= 10 OR max_total_steps <= 20 OR max_train_samples <= 64`.

| candidate checkpoint | step | objective | max_total_steps | max_train_samples | batch | verdict |
|---|---|---|---|---|---|---|
| `minimal_jepa_vicreg_smoke_latest.pt` | 2 | jepa_vicreg | – | 8 | 1 | SMOKE |
| `minimal_smoke_latest.pt` / `_best_model.pt` | 2 | jepa | – | – | – | SMOKE |
| `sweep_smoke_latest.pt` / `_best_model.pt` | 2 | jepa | – | – | – | SMOKE |
| `adaptive/phase_00_jepa_{latest,best}.pt` | 4 | jepa | 6 | 8 | 1 | SMOKE |
| `adaptive/phase_00_jepa_best_repr.pt` | 0 | jepa | 6 | 8 | 1 | SMOKE (near-init) |
| `adaptive/_smoke*/**` (all phases 00–05) | ≤4 | various | 6 | 8 | 1 | SMOKE |
| `synthetic_collapsed.pt` | 2687 (fabricated) | – | – | – | – | SYNTHETIC PROXY — not a real training artifact |

Additional decisive facts from `checkpoints/milestone_b/REPORT.md` and
`checkpoints/milestone_b/EXPERIMENT_LOG.md`:

- The ONLY genuine cloud-trained checkpoint (Kaggle, `minimal_jepa_latest.pt`,
  step 2687, epoch 20) **collapsed** (EMA target pairwise cosine ≈ 0.99987,
  effective rank 2.6/384) and **is not on this machine** — only its measured
  anchor stats survive. It was explicitly retired ("The collapsed checkpoint is
  not reused").
- Kaggle screening Phase 0 (`jepa`) COLLAPSED by step 200–300; Phase 1
  (`jepa_vicreg`, λ_var=0.1/λ_cov=0.04) COLLAPSED by step 400–500; Phase 2
  (`lejepa`) crashed pre-device-fix and was never evaluated.
- The planned 800-step global screening ladder (jepa → jepa_var → jepa_vicreg →
  jepa_vicreg2 → jepa_barlow → lejepa) **has not run yet**
  (EXPERIMENT_LOG.md, "Pre-training verification state", 2026-08-18).

Conclusion: **zero checkpoints satisfy §6's genuine-trained criteria.**

## 2. Guards implemented (spec §10) — now mandatory in the diagnostic

`scripts/diagnostics/physics_target_selection.py`:

- **Check A/B (metadata gate)** — `validate_checkpoint_provenance()`: refuses
  any checkpoint missing `step`/`objective_name`/`cfg`, any smoke-scale signal
  (thresholds above), or any objective outside the approved set
  (`jepa, jepa_var, jepa_vicreg, jepa_vicreg2, jepa_barlow, lejepa`). Fires
  BEFORE weights touch the model. Explicit operator override via
  `--allow-smoke-reason "<why>"` proceeds but stamps the JSON
  `is_smoke_checkpoint: true`, `genuinely_trained: false`,
  `run_status: reference_only_not_a_trained_model_result`.
- **Check C (shape)** — `check_target_shape()`: z_y_raw must be
  `(B, 256, hidden)` per batch and in the retrieval matrix.
- **Checks D/E (frozen references)** — `assert_reference_modules_frozen()`:
  EMA target and released spectrum encoder must have zero trainable params;
  recorded in `provenance.*.freeze_checks`.
- **Check F (target identity)** — `check_same_target_across_conditions()`:
  z_y_raw must be bit-identical across real/null/shuffled conditions, every
  batch (negative control tested: spectrum-leaking targets are caught).

Provenance block written to the JSON (spec §7): top-level
`checkpoint_provenance` with `path, step, epoch, objective_name, seed,
config_sha256_16, smoke_signals, is_smoke_checkpoint, genuinely_trained`.

Demonstrated refusal (actual run output):

```
ValueError: checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt:
SMOKE/near-init checkpoint refused (signals={'step': 2, ..., 'max_train_samples':
8, 'train_max_steps': 3, 'batch_size': 1}; ...). A trained-model result cannot be
produced from this state.
```

## 3. Tests (spec §17)

`tests/test_physics_target_selection.py`: **33 passed** (was 20; +13 guard
tests), including all five required:

- `test_accepts_genuine_checkpoint_metadata`
- `test_objective_name_checked`
- `test_ema_frozen`
- `test_spectrum_encoder_frozen`
- `test_target_identical_across_conditions`

plus loud-failure tests: smoke rejection (3 parametrized signals), missing-
metadata rejection, raw-state-dict rejection, override labeling, shape check,
leaky-target detection, end-to-end guard enforcement in `evaluate_ratio`.

## 4. Smoke vs trained comparison (spec §16 table)

Trained column: **PENDING — cannot be filled without a genuine checkpoint.**
Smoke/reference columns below are from the existing (now clearly labeled)
smoke-state runs; note `trained_r0.75` was byte-identical in behavior to
`init_reference_r0.75` — i.e., the step-2 smoke model IS the initialization
distribution, which is precisely why §18 forbids presenting it as evidence.

| metric (ratio 0.75, n=32) | smoke ckpt (step 2) | init reference | GENUINE TRAINED |
|---|---|---|---|
| cos margin null−real (median) | +0.0043 (97% pos) | identical | PENDING |
| cos margin shuffle−real (median) | +0.0002 (53% pos) | identical | PENDING |
| l2_pooled margin shuffle−real | −0.0048 (47% pos) | identical | PENDING |
| retrieval R@1 / R@5 | 0.000 / 0.000 | identical | PENDING |
| mean correct rank (B=8) | 8.00 (chance) | identical | PENDING |
| mutual causal win rate | 0.000 | identical | PENDING |

(At ratio 1.00 the earlier smoke run likewise gave R@1=0.125, mutual wins 0.0,
shuffle-margin fraction-positive ≈ 0.53.)

## 5. Decision: T1 / T2 / T3 / T4

**NONE can be decided yet.** The decision tree requires trained-model behavior;
every number currently in hand comes from smoke/near-init states whose
predictor output is statistically indistinguishable from the released-init
reference. Issuing a T-verdict now would repeat exactly the epistemic error the
validity-fix directive exists to prevent (the earlier near-zero `cos_err`
incident).

## 6. ONE recommended next action

**Run the pending genuine Milestone-B training on Kaggle per
`CLOUD_TRAINING.md` (the already-planned 800-step adaptive screening ladder,
fresh base-initialization start per method), sync the winning HEALTHY checkpoint
back into `checkpoints/milestone_b/`, then rerun:**

```
python scripts/diagnostics/physics_target_selection.py \
    --config configs/milestone_b.yaml \
    --checkpoint <genuine_trained_checkpoint>.pt \
    --ratios 1.00 0.75 --n-samples 32 --retrieval-batch 8 --device cpu
```

The guards will verify provenance automatically; only then fill Section 4's
trained column and issue the T1–T4 verdict. No CLIP alignment work, no new
physics objective, and no architecture change before that verdict exists.

## 7. What was NOT done (by design)

- No local training run was launched to manufacture a checkpoint (§6).
- No smoke checkpoint was evaluated as "trained" (§18); the demonstration run
  above was refused before weight loading.
- No CLIP alignment, no architecture changes, no new physics objective.
