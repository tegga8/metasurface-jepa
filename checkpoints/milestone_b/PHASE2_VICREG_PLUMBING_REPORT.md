# Phase 2 — Fix VICReg Training/Validation Plumbing Before Cloud Run — REPORT

Date: 2026-08-24. Scope: plumbing-only (objective device placement + eval-mode
hygiene in validation/reference paths), per the operator's phase directive and the
dated 2026-08-24 override entry in `AGENTS.md`. No architecture, GCLCT, bottleneck,
VICReg-mathematics, coefficient (25/25/1, gamma=1, eps=1e-4), projector-layout,
EMA-semantics, dataset/mask/hyperparameter, or checkpoint-strictness changes.

## Files changed and why

| File | Location | Change | Why |
|---|---|---|---|
| `scripts/train/train_milestone_b.py` | objective construction (~L233) | `build_objective(...).to(device)` + guarded `next(objective.parameters()).device == device` assertion | **Fix A** — verified runtime failure: projector on CPU while z_hat/z_y_raw on CUDA (`Expected all tensors to be on the same device`). Assertion is name-independent; parameterless objectives would skip safely |
| `src/train/engine.py` | new `_eval_mode_restore()` helper after `_objective_projection` | duck-typed eval-enter/restore for (model, objective); returns zero-arg restore callable | **Fix B** core — one implementation for all validation/reference paths |
| `src/train/engine.py` | `FixedValidation._acc_stats` | mode handling via `_eval_mode_restore` inside try/finally (was model-only, no finally) | VICReg projector contains BatchNorm1d×2 → train-mode validation forwards consumed batch statistics AND updated running statistics; metric nondeterministic + stats contaminated |
| `src/train/engine.py` | `FixedValidation.null_gap` | same | same defect on the real/null goal-intervention diagnostic |
| `src/train/engine.py` | `healthy_references` | same (previously NO mode management at all — callers passed a fresh train-mode objective) | reference statistics are projected through the candidate objective's own projector (spec §17) → the released-init reference itself was BN-contaminated |
| `scripts/eval/eval_vicreg_sanity.py` | `_audit_row` | diagnostic forward now runs with BOTH modules eval, restored via finally | short-audit measurement rows ran while training loop held both modules in train mode; eval-mode measurement is deterministic and never touches BN running stats. `_term_grad_norms` keeps its existing snapshot/restore (train-mode semantics unchanged). Checkpoint-validation path already set `model.eval(); objective.eval()` (unchanged) |
| `tests/test_vicreg_plumbing_mode_device.py` | new | Tests A–D regression suite | locks both fixes |

Duck-typing note: `_eval_mode_restore` leaves bare stub objects without
`.eval/.train/.training` untouched (pre-fix behavior), so existing engine tests
using plain-object models (`test_tier3_fixes.py`) keep passing unmodified. Real
`nn.Module`s always get full semantics.

## Verification (local, CPU-only machine)

Ordered per §8 (suite = repo's own `tests/`; bare `pytest -q` additionally sweeps
`external/lejepa/tests/`, which requires the uninstalled Milestone-G `lejepa`
package — collection pollution, not a code failure):

1. `pytest -q tests/test_vicreg_collapse.py tests/test_vicreg_objective.py tests/test_vicreg_gradients.py tests/test_vicreg_plumbing_mode_device.py` → **39 passed, 1 skipped** (skip = CUDA-only Test A)
2. `pytest -q tests/test_architecture.py` → **43 passed**
3. `pytest -q tests` → **238 passed, 7 skipped** (baseline before phase: 230 passed / 6 skipped at provenance commit; delta = 4 Gate-0 helper tests from Phase-1 closure + 4 plumbing regressions + 1 CUDA-skip)

Entry-point smokes:
- `python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml --smoke` → completed 3 steps, device assertion held, in-loop validation + final null-gap eval + checkpoints OK.
- `python scripts/eval/eval_vicreg_sanity.py --short-audit --smoke --config configs\milestone_b.yaml` → 6 steps, no aborts, all components/per-term grads reported.

## Confirmations

- **VICReg mathematics untouched**: `src/losses/vicreg.py` not modified this phase;
  hinge `mean(relu(gamma - std))` non-squared and unhalved covariance sum verified by
  the untouched `test_vicreg_objective.py::test_components_present_and_finite`,
  `test_weighted_terms_sum_to_total`, `test_custom_lambdas_change_weighted_terms`.
- **EMA optimizer-excluded and gradient-free**: unchanged order backward → EMA-grad
  assert → clip → step → `objective.on_optimizer_step(model, step)`; guards
  `_assert_no_ema_gradients` (train script) and `assert not (opt_ids & ema_ids)`
  (both scripts) intact; `test_vicreg_gradients.py` green.
- **Validation/reference now switch BOTH model and objective to eval and restore
  prior states**: proven by `test_b_*` (train→train, eval→eval) and `test_c_*`
  (BN running stats byte-identical across evaluate/null_gap/healthy_references)
  in the new test file.
- **Strictness untouched**: `load_checkpoint` strict objective-name match,
  optimizer param-shape ownership fingerprint, missing-objective_state refusal —
  all green under `test_checkpoint_resume.py`.

## Cloud gate (§9) — PENDING OPERATOR RUN

Not started locally by design. Command:

```
python scripts/eval/eval_vicreg_sanity.py \
    --short-audit \
    --steps 200 \
    --report-every 25 \
    --subset 32 \
    --batch-size 8 \
    --config configs/milestone_b.yaml
```

Real MetaDiT data, current main, obsolete Perceiver checkpoint excluded (none used).
Expected report: `checkpoints/milestone_b/vicreg_sanity/eval_jepa_vicreg_short_audit.json`
with per-report rows (loss components incl. weighted/ratio terms, raw+projected
rank/pairwise-cos/feature-std/cov-RMS, projector SV audit, per-term gradient norms)
and abort-on-violation guards (NaN/Inf loss or grads, EMA leakage, raw/projected
target-or-predictor collapse vs released-init baseline, persistent single-term
domination >99.9% ×5 reports).

## Verdict

**LOCAL PASS — PENDING CLOUD GATE.** All local §8 gates green; no stop condition
(§10) fired locally. Full Milestone-B run is unlocked CONDITIONAL on the operator's
200-step cloud audit completing with zero hard-guard violations. Do not start
decoder/inverse-design work regardless until that gate is recorded.
