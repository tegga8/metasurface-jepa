# VICReg Objective — Phase 1 Report (mathematical faithfulness repair)

Status: **GATES 1–4 PASSED locally (exact math, collapse, gradients, smoke); real-data short training run pending on cloud GPU** (compute policy: gradient training on cloud per `CLOUD_TRAINING.md`). Not yet at the §33 decision boundary until the cloud short run verifies.

## 2. Current repository implementation before repair

`src/losses/vicreg.py` already had the correct high-level VICReg structure:
- objective-owned 3-layer projector (Linear+BN+ReLU ×2 + Linear(bias=False), 384→384→384→384),
- invariance (MSE), variance, covariance terms,
- EMA target frozen, raw 384-D latent visible outside the objective space,
- coefficients 25 / 25 / 1, gamma 1, eps 1e-4.

The `jepa_vicreg` objective (`src/losses/objectives.py`, `VICRegObjective`) applied these terms over masked geometry tokens with geometry-level pooled statistics computed as diagnostic-only components.

## 3. Discrepancies found

| # | Severity | Location | Discrepancy vs canonical VICReg |
|---|----------|----------|--------------------------------|
| 1 | CRITICAL | `variance_loss` | `torch.relu(gamma - std).pow(2).mean()` — squared hinge. Canonical: `torch.relu(gamma - std).mean()` (official implementation `mean(F.relu(1 - std))`; paper `v(Z) = mean_j max(0, gamma - sqrt(Var(z_j)+eps))`, no square). |
| 2 | CRITICAL | `vicreg_branch_terms` | `L_cov = 0.5 * (cov_penalty(p_hat) + cov_penalty(p_y))` — accidental 0.5 on the branch sum. Canonical: `cov_loss = cov_x + cov_y` (paper `lambda_cov * [c(Z) + c(Z')]`). Variance's 0.5 branch average was already correct and was kept. |
| 3 | NUMERICAL | `invariance_loss` | `_require_n2` on both branches — an N>=2 restriction on plain MSE, which is mathematically valid for any N>=1. N>=2 is required only for variance/covariance. |
| 4 | DEAD CODE | `vicreg_loss` | Historical single-branch helper (old `jepa_var`/`jepa_vicreg`/`jepa_vicreg2` ladder rungs) — zero references anywhere in the repo; a second, competing formula (same squared-hinge bug) left in `src/losses/vicreg.py`. |
| 5 | EDGE | `VICRegObjective.forward` | Geometry-level diagnostic terms crashed the objective on batch size 1 (pooled statistics over 1 geometry are undefined). Diagnostics must NaN-mark (Bug #21 convention), never raise or zero — the hard N>=2 raise is reserved for the actual loss terms. |

No discrepancy: invariance was already plain MSE (no L2 norm, temperature, stop-gradient, or detach); the shared projector on both branches was already correct.

## 4. Exact mathematical corrections

`src/losses/vicreg.py` (canonical forms, unchanged by coefficient tuning):

```
L_inv = MSE(p_hat, p_y)                                          (invariance)
L_var = 0.5 * (var_penalty(p_hat) + var_penalty(p_y))            (variance)
L_cov = cov_penalty(p_hat) + cov_penalty(p_y)                    (covariance)

var_penalty(Z) = mean_d relu(gamma - sqrt(var_d + eps))          (hinge, NOT squared)
cov_penalty(Z) = sum_{i != j} C_ij^2 / D,  C = Zc^T Zc / (N-1)   (unbiased)
```

- #1: removed `.pow(2)` — hinge-mean restored in `variance_loss` (and the dead `vicreg_loss` copy is gone, so no competing formula remains).
- #2: removed the `0.5 *` from `L_cov` branch aggregation; `L_var` keeps its canonical `0.5 *` branch average (official implementation divides each branch by 2).
- #3: `invariance_loss` is now `F.mse_loss(p_hat, p_y)` with no `_require_n2`; `_require_n2` remains in `variance_loss`/`covariance_loss` (undefined statistics raise — never silently zeroed).
- #4: `vicreg_loss` deleted (was dead code; docstring + module docs updated).
- #5: geometry-level diagnostics NaN-marked at B<2; loss terms still raise at N<2.

## 5. JEPA-specific adaptation

This is a **deliberate JEPA adaptation of VICReg**, not a reproduction of the original training topology:

```
masked geometry -> context encoder + physics-conditioned predictor -> z_hat
full geometry   -> EMA target encoder (FROZEN)                     -> z_y_raw
z_hat, z_y_raw  -> VICReg-owned projector (shared, trainable)      -> p_hat, p_y
```

Canonical VICReg receives two trainable views of the same image through a shared encoder; this project's pair is **predictor output ↔ frozen EMA target output**. The mathematics of the objective terms are canonical; the pair construction is JEPA-specific. This distinction is stated in the code (`VICRegObjective` docstring) and here; the objective is never described as "official VICReg unchanged."

## 6. Projector ownership

- `self.projector = VICRegProjector(...)` — objective-owned module (there is no `model.proj` anywhere; `test_objective_owns_projector_no_model_proj_needed`).
- One shared projector applied to both branches: `p_hat_full = self.projector(z_hat)`, `p_y_full = self.projector(z_y)`. No `projector_target`, no weight tying to Barlow/LeJEPA heads (separate classes per objective, `objective_modules.py`).
- Layout kept: `Linear(384,384,bias=True) → BN → ReLU → Linear(384,384,bias=True) → BN → ReLU → Linear(384,384,bias=False)` — official VICReg-style; final layer bias-free. 384-dims are a project-specific capacity choice, not a replacement of official 8192-d ImageNet defaults.
- The projector is trainable on both branches: no `projector(z_y.detach())`. The EMA **encoder** stays gradient-free (gradients reach the projector through both branches, never the frozen encoder — tested).
- The scientific representation remains `z_y_raw`; `p_y` is the VICReg objective space only. All diagnostics report raw AND projected health separately.

## 7. Sample axis / token-level statistics

VICReg statistics are computed over **masked geometry tokens** — `p_hat = p_hat_full[mask]`, `p_y = p_y_full[mask]` — with samples as rows (N = masked tokens across the batch), features as columns (D). This is the project's JEPA token-level adaptation, not canonical image-level VICReg; the projector's BatchNorm is computed over the same flattened token/sample axis. **Known consequence, stated explicitly**: variance/covariance regularization can encourage diversity across spatial token positions. Whether token-level regularization improves the raw geometry representation is an empirical question to be evaluated after this correctness phase, not assumed.

The projector runs BEFORE masking (never project-only-masked-tokens): BatchNorm's sample population is the full-batch flattened token axis on both branches, keeping the two branches' statistics symmetric.

## 8. Gradient boundaries

- `L_inv`/`L_var`/`L_cov` each flow student → projector (per-term backward tests).
- The target branch reaches the objective-owned projector but NEVER the frozen EMA encoder (per-term target-branch tests, `tests/test_vicreg_gradients.py`).
- The frozen EMA encoder is updated only via `objective.on_optimizer_step(model, step)` → `model.ema.update(...)`; a bare `optimizer.step()` does not touch it.
- Documented path for the target-side terms: `z_y` → target projector (shared) → target-side VICReg statistics → projector gradients. **Target-side variance/covariance does not directly update the frozen EMA encoder** — do not claim it does.

## 9. Collapse tests

`tests/test_vicreg_collapse.py` (35 VICReg tests total across the three VICReg test files):

- **V1 exact variance formula**: `variance_loss` == `mean relu(gamma - std)` reference; a squared hinge measurably differs — this test would have caught the historical bug.
- **V2 exact covariance formula**: `covariance_loss` == `off-diag(C)^2.sum() / D` with unbiased C.
- **V3 branch aggregation**: `L_var == 0.5*(var_hat + var_y)`; `L_cov == cov_hat + cov_y`; the accidental-0.5 form is asserted to differ.
- **V4 total objective**: with default coefficients `total == 25*L_inv + 25*L_var + 1*L_cov`, no hidden normalization.
- **V5 collapsed target branch**: p_y constant, p_hat healthy → `L_var > 0`, `total > 0`.
- **V6 near-collapse gradient**: nonzero variance gradient for small nonzero near-collapsed representations (gradient at exact zero is degenerate by construction — not required).
- **V7 adversarial projector recovery** (spec §18): full-rank raw input, final projection layer initialized NEAR rank-1 (unit row × 0.5 + 1e-3 asymmetry noise), optimize ONLY variance+covariance. Recorded numbers:
  - before: projected eff-rank fraction **0.125**, variance **0.603**, covariance **0.173**
  - after (800 Adam steps): eff-rank fraction **0.993**, variance **0.000**, covariance **0.000**
  - both penalties decrease; the space recovers from the exact pathology that damaged the old pipeline.
  - NOTE (documented in the test): exact rank-1 initialization is a gradient-symmetric saddle — every row receives the identical gradient, the rank-1 manifold is invariant, and the joint var+cov objective has an interior minimum on it (dL/ds = -1 + (D-1)/2·s³ = 0 → s ≈ 0.66). Hence the physically relevant test is NEAR rank-1, as the spec itself says.
- Retained: constant-input max penalty, healthy-Gaussian near-zero penalty, covariance response + decorrelation-on-minimization, zero-invariance-on-identical-branches.
- Raw and projected collapse are monitored separately everywhere (raw target / raw predictor / proj target / proj predictor).

## 10. Training integration

`scripts/train/train_milestone_b.py` unchanged for VICReg (already correct):
- optimizer owns model trainable params + objective trainable params; EMA parameters explicitly excluded (`ema_ids & opt_ids` assert, line 267-270);
- update order: total.backward() → `_assert_no_ema_gradients` → clip → optimizer.step() → zero_grad → `objective.on_optimizer_step` (EMA) → scheduler.step();
- resumable checkpoints with objective state (§30); standalone CLI; per `CLOUD_TRAINING.md`.

**Optimizer**: AdamW (lr 1e-4, wd 0.05, warmup + cosine) is kept. The official VICReg implementation uses LARS for its large-scale ImageNet setup; this project's task is a faithful objective at project scale, not a reproduction of the official training stack, and AdamW was not replaced.

## 11. Results of unit tests

- VICReg files: **35 passed** (`test_vicreg_collapse.py` 20, `test_vicreg_objective.py` 10, `test_vicreg_gradients.py` 5).
- Full suite: **192 passed, 6 skipped** (CUDA-only skips), ~18 s, local CPU.
- `test_n_below_two_raises_not_silently_zeroed` updated: invariance computes at N=1; variance/covariance/objective still raise at N<2.

## 12. Results of real-data smoke training

Local dev-only smoke runs (batch 1 / few steps, CPU — per compute policy, NOT a substitute for the cloud run):

1. `scripts/eval/eval_vicreg_sanity.py --short-audit --smoke` (6 optimizer steps, real MetaDiT val data, released-init model): completed with no abort. Step-0 row: `L_inv 0.242`, `L_var 0.710`, `L_cov 0.519` (unweighted); weighted `6.06 / 17.75 / 0.52`, total `28.03`; ratios `0.249 / 0.730 / 0.021`; per-term gradient norms all nonzero (`1.28 / 1.01 / 5.79`) — every anti-collapse term has a live optimization path; raw and projected covariance off-diagonal RMS reported (`0.0020 / 0.0174`); EMA gradient-free guard passed. Artifact: `checkpoints/milestone_b/vicreg_sanity/eval_jepa_vicreg_short_audit.json`.
2. `scripts/train/train_milestone_b.py --smoke` (3 steps, real MetaDiT forward/backward/optimizer step/EMA update): **exit 0**, final metrics written (fixed-validation cos_err and null-gap paths exercised; health `UNAVAILABLE` in the smoke subset is the expected Bug-#21 guard at n<2 samples, not a collapse signal).

**Pending (cloud GPU, per `CLOUD_TRAINING.md`)**: the real-data short training experiment (200 steps, batch 8, real subset) verifying variance/covariance behavior, raw/projected non-collapse, and EMA gradient-freeness over a longer trajectory; then record in this report and update §14.

## 13. Remaining deviations from canonical VICReg

Every deviation is explicitly labeled:

- **[DELIBERATE JEPA ADAPTATION]** Pair construction: predictor output ↔ frozen EMA target output, instead of two views of the same image (the mathematics of the terms are canonical).
- **[DELIBERATE JEPA ADAPTATION]** Sample unit: masked geometry tokens (token-level statistics), not image-level samples; projector BN over the flattened token axis.
- **[DELIBERATE JEPA ADAPTATION]** Geometry-level pooled statistics (`geo_inv`/`geo_var`/`geo_cov`) computed and reported — **diagnostic-only, never part of the loss** (the loss is exactly `lambda_inv*L_inv + lambda_var*L_var + lambda_cov*L_cov` over the declared sample unit; `test_token_level_stats_are_the_loss_geometry_level_is_health_only`).
- **[PROJECT CHOICE, NOT A MATH CHANGE]** Projector capacity 384→384→384→384 (official ImageNet defaults are 8192-d).
- **[PROJECT CHOICE, NOT A MATH CHANGE]** AdamW instead of LARS (official LARS is for large-scale ImageNet; coefficients 25/25/1, gamma 1, eps 1e-4 unchanged and canonical).
- **[PROJECT CHOICE]** NaN-marker convention for undefined B<2 geometry diagnostics (Bug #21); the loss terms keep the hard N≥2 raise.
- **No mathematical deviations remain** in the term formulas or branch aggregation (verified by V1–V4 exact tests and the §30 audit below).

## 14. Final status

- §32 acceptance checklist: variance averaged across both branches ✓; covariance = SUM of both branch penalties ✓; no accidental 0.5 ✓; invariance = plain MSE ✓; projector shared ✓; projector objective-owned ✓; EMA encoder receives no gradient ✓; target projector trainable ✓; coefficients 25/25/1, gamma 1, eps 1e-4 ✓; token-level adaptation explicitly documented ✓; geometry-level diagnostics not in the loss ✓; exact formula tests pass ✓; collapse tests pass ✓; projector-collapse recovery passes ✓; gradient-isolation tests pass ✓; real MetaDiT smoke (forward/finite loss/backward/step/EMA) passes ✓; raw and projected collapse monitored separately ✓.
- §30 final code audit: `variance_loss`/`covariance_loss`/`vicreg_branch_terms` defined once in `src/losses/vicreg.py` and consumed only by `VICRegObjective`; `vicreg_loss` dead helper removed (zero references); `jepa_vicreg` registered once (`OBJECTIVES`); one VICReg projector class, one shared instance. Exactly one active canonical implementation.
- §33 decision boundary NOT yet reached: the real-data short training experiment on cloud GPU is pending (compute policy). Upon its PASS, VICReg is declared **mathematically correct + explicitly documented JEPA adaptation + empirically tested anti-collapse mechanism**, and the objective can proceed to controlled comparison against Barlow/LeJEPA — without modifying the shared architecture.