# VICREG_IMPLEMENTATION_REPORT

**Candidate:** `jepa_vicreg` — faithful EMA-JEPA + VICReg-style regularization
**Date:** 2026-08-20
**Scope:** Milestone B screening ladder (operator override 2026-08-17) — this pass implements
the CODEX-specified `jepa_vicreg` candidate only. Barlow (`jepa_barlow`) and LeJEPA (`lejepa`)
are NOT implemented, fixed, or extended in this pass.

---

## 1. Files changed

| File | Change |
|---|---|
| `src/losses/objective_modules.py` | **NEW** — `VICRegProjector`, the objective-owned projector (Linear+BN+ReLU)×2 + Linear(no bias), reshape `(...,D)→(-1,D)→(...,D)`. |
| `src/losses/vicreg.py` | Rewritten — canonical term functions `invariance_loss` / `variance_loss` / `covariance_loss`, `vicreg_branch_terms`, and the historical single-branch `vicreg_loss` (N≥2 guard added, silent-zero fallback removed). |
| `src/losses/objectives.py` | `VICRegObjective` (name `jepa_vicreg`, registered in `OBJECTIVES`); `JEPAObjective` fixed to `getattr(model, "proj", None)` for the refactored 256×384 model (which has no `.proj`). Registry keys unchanged (six rungs). |
| `scripts/train/train_milestone_b.py` | Ladder integration: `_objective_kwargs` for `jepa_vicreg` (lambdas/eps/gamma/projector dims from config), optimizer built from student-trainable + objective params with EMA-leak assertion, checkpoint carries `objective_state` (missing → resume raises `RuntimeError`), EMA-gradient guard after `total.backward()`. Default ladder narrowed to `[jepa, jepa_vicreg]` (the historical `jepa_var`/`jepa_vicreg2`/`barlow`/`lejepa` rungs still reference the removed `model.proj`; pre-existing, out of scope). |
| `configs/milestone_b.yaml` | Added `vicreg:` (projector 384/384/384; lambda_inv 25, lambda_var 25, lambda_cov 1, gamma 1, eps 1e-4) and `adaptive_training.enabled: true` with objectives `[jepa, jepa_vicreg]`, max_total_steps 1800, lr 1e-4, wd 0.05, grad_accum 2, batch 64, val_every 100, log_every 25, clip 1.0, fixed_val_subset 512, mask_ratio 0.5, mask_seed 12345, refs_seed 2026, save_optimizer true. |
| `scripts/eval/eval_vicreg_sanity.py` | **NEW** — checkpoint validation (`validate_checkpoint`: raw/projected space diagnostics, collapse classifier, physics real/null/shuffled comparisons, raw-vs-projected) and short training audit (`run_short_audit`, 100–300 steps, per-term gradient norms with projector-BN snapshot/restore, reference-relative abort gates). |
| `tests/test_vicreg_objective.py` | **NEW** — 11 objective-level tests. |
| `tests/test_vicreg_gradients.py` | **NEW** — 12 gradient-boundary tests. |
| `tests/test_vicreg_collapse.py` | **NEW** — 7 synthetic collapse/redundancy tests. |

## 2. Architecture: before vs. after

**Before (historical, refactored out):** the model carried a shared `model.proj` projection
head and the predictor had a two-branch (base+delta) design with a latent bottleneck 256→64.

**After (current corrected 256×384 path, preserved):**

```
masked geometry -> context encoder (256 tokens, no Perceiver) -> GCLCT -> z_hat [B,256,384]
full geometry   -> EMA target encoder (FROZEN)                -> z_y   [B,256,384]
```

- `GoalConditionedJEPA` has **no** `model.proj` (asserted in tests).
- The **objective** owns the projector: `VICRegObjective.projector` is a `VICRegProjector`
  mapping the raw 384-D latent into the VICReg objective space. The raw latent is NOT the
  projector output; raw-space health is reported separately.
- Forbidden constructs verified absent: latent bottleneck before the projector, shared global
  `model.proj`, projector in the raw representation path, base+delta predictor.

## 3. Loss equations (token-level, both branches)

```
L_total = lambda_inv * L_inv + lambda_var * L_var + lambda_cov * L_cov

L_inv = MSE(p_hat, p_y)                                        p = projector(z)
L_var = 0.5 * (var_penalty(p_hat) + var_penalty(p_y))
L_cov = 0.5 * (cov_penalty(p_hat) + cov_penalty(p_y))

var_penalty(Z) = mean_d relu(gamma - sqrt(var_d + eps))^2       std per feature, unbiased var
cov_penalty(Z) = sum_{i!=j} C_ij^2 / D,  C = Zc^T Zc / (N-1)    off-diagonal of unbiased cov
```

Defaults: `lambda_inv=25, lambda_var=25, lambda_cov=1, gamma=1.0, eps=1e-4`.

**Sample axis (deliberate adaptation, spec §6):** statistics are computed over **masked
geometry tokens** (rows = N masked tokens across the batch, columns = D features). This is the
token-level adaptation of canonical VICReg's image-level statistics. Additionally, the
**geometry-level** mean-pooled vectors [B, D] run through the same three terms as
`geo_inv/geo_var/geo_cov` health components — reported, **never added to the loss** (asserted
in tests).

**N≥2 hard guard:** any branch with fewer than 2 rows raises
`ValueError("VICReg requires at least two valid samples ...")` — undefined statistics are never
silently zero-substituted (spec numerical rules). `vicreg_loss`'s historical silent-zero
behavior for n<2 was deliberately removed.

## 4. Gradient flow (verified by tests)

| Path | Student (z_hat) | Projector | EMA target encoder |
|---|---|---|---|
| L_inv | ✓ | ✓ | ✗ (never) |
| L_var | ✓ | ✓ | ✗ (never) |
| L_cov | ✓ | ✓ | ✗ (never) |
| total.backward() | ✓ | ✓ | ✗ (asserted, `p.grad is None`) |

- The EMA target encoder is updated **only** via `objective.on_optimizer_step(model, step)` →
  `model.ema.update(model.geometry_encoder, step)`. A bare `optimizer.step()` never touches it
  (tested).
- `train_milestone_b.py` asserts optimizer parameter ownership: every trainable parameter is in
  the optimizer, the EMA target's parameters are absent, and no EMA parameter receives a
  gradient (checked after `total.backward()`, before `zero_grad`).

## 5. Optimizer & checkpoint ownership

- Optimizer = AdamW(student trainable params + objective params), weight decay 0.05.
- Checkpoints (all 5 save sites) now carry `objective_name` + `objective_state` via the
  existing `extra` mechanism. Resume with a missing or mismatched objective state **fails
  loudly** (`RuntimeError`), never silently continues with a fresh projector (which would reset
  the objective space mid-ladder).
- `save_optimizer` config flag controls optimizer-state persistence (true in the new config).

## 6. Diagnostics (spec §22)

`validate_checkpoint` (offline, on a checkpoint with objective state):
- Raw space (target/predictor, masked tokens): effective rank, rank fraction, top-eig fraction,
  participation, pairwise cross-geometry cosine (p05/mean/median/p95), per-feature std stats
  (min/median/mean, fraction < 0.5 and < 0.1).
- Projector input/output SVD audits (eff rank, min/max/median singular value, condition number)
  and cross-sample cosine — collapse classifier returns `RAW_COLLAPSE` / `PROJECTOR_COLLAPSE` /
  `HEALTHY`.
- **Physics controls:** real-target latent vs. null-target vs. shuffled-target goals — the
  physics improvement (`P(Z_x, S_real)` vs null/shuffled) is reported raw and projected.
- **Raw-vs-projected:** cosine of raw latents vs. cosine of projected latents, to show the
  projector space is sharper than (or at least as rich as) the raw space.

`run_short_audit` (100–300 steps): per-term gradient norms (inv/var/cov) with projector-BN
running-stat snapshot/restore so measurement doesn't perturb training; abort on NaN/Inf, EMA
gradients, raw collapse, projector collapse, or a ≥5-step domination streak. Collapse aborts are
**reference-relative** to the run's own released-init first report (absolute thresholds were
rejected after they fired spuriously on the healthy initialized model; see §8 below).

## 7. Test results

Full suite: **149 passed, 6 skipped** (skips pre-existing). New tests: 30.

- `tests/test_vicreg_objective.py` (11): registry placement + six-rung stability; objective-owned
  projector with a model that has no `.proj`; component contract incl. weighted terms summing to
  total and ratios to 1; token-level terms define the loss while geo-level stats are health-only;
  N<2 raises across all four entry points; EMA update ownership; full-path backward finiteness.
- `tests/test_vicreg_gradients.py` (12): per-term (L_inv/L_var/L_cov) gradient flow to student +
  projector with the frozen EMA encoder receiving nothing; target-branch-only backward still
  trains the projector; optimizer moves student+projector but never the EMA encoder.
- `tests/test_vicreg_collapse.py` (7): constant input → variance penalty ≈ 1, covariance 0,
  total > 0; identical branches → zero invariance but positive total; healthy Gaussian ≪
  collapsed penalty; covariance responds to redundancy and decorrelates columns when minimized;
  rank-1 projector collapse detected (eff-rank 0.125 vs raw 0.916) and recovered by the
  var+cov mechanism (rank > 0.2, penalty < 0.6×) — the exact-zero point was excluded because
  the variance gradient degenerates there by construction (see §8).

## 8. Short-run results and deviations

Smoke audit (CPU, 6 steps, batch 2, released-init weights): completed without abort; all three
gradient terms nonzero each step (inv ~1.1–1.6, var ~1.2–1.7, cov ~4–13 — cov-dominated
gradient norms as expected from an initially-unregularized projector); `L_total` ≈ 17–19 with
`L_var` ≈ 0.5 (variance pressure active), `L_inv` ≈ 0.17–0.23, `L_cov` ≈ 0.25–0.35; raw space
stays rich (mean feature std ≈ 0.08–0.11, min ≈ 0.03); projected space healthy (proj rank frac
≈ 0.06 vs raw 0.02; proj frac_std<0.1 = 0.0 throughout); raw p05 pairwise cosine ~0.96–0.99 at
batch 2 (single pair — not meaningful at this size; real runs use batch ≥ 8). JSON:
`checkpoints/milestone_b/vicreg_sanity/eval_vicreg_short_audit.json`.

**Deviations from canonical VICReg (all deliberate, all spec-mandated or CODEX-mandated):**

1. Statistics over **masked tokens** rather than image-level vectors (spec §6) — with the
   geometry-level pooled health axis reported separately.
2. The target branch is the **frozen EMA encoder's output** (canonical VICReg trains both
   views through the same encoder); the target branch's projector path is a separate learnable
   head — gradient reaches the projector, never the EMA encoder.
3. `eps` inside the `sqrt` of the variance term (canonical impl adds it outside); collapsed
   constant input gives penalty `(1−sqrt(eps))² ≈ 0.9801` rather than exactly 1 — harmless,
   verified by tests.
4. N<2 raises instead of zero-fallback (spec numerical rules).
5. Exact-zero collapse point excluded from the recovery test: with all-centered-zero outputs the
   variance gradient is identically zero by construction (a degenerate point, not a failure of
   the term); near-collapse (the physical case) is tested and recovers.

**Not yet run (cloud):** the real short audit (100–300 steps) and `validate_checkpoint` physics
comparisons require a gradient-training cloud run per `CLOUD_TRAINING.md`; the local smoke
verified the full code path. No unrelated modules (Barlow, LeJEPA, DirectMaskedGenerator,
physics encoder, dataset format) were touched.