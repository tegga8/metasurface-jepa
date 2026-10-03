# Phase 4 diagnosis — why the multi-target objective collapsed

Phase 4 (commit `72195e6`) failed: at 10k the model collapsed vs the Phase-2
control (MAE 0.0792 → 0.3214; occupancy IoU 0.70 → 0.02; A/B/C gates 0.982/0.910/
0.736 → 0.512/0.512/0.266). This is the code + results review that explains it and
the fix.

## Evidence

- Live `[grad-share]` (Phase 2a probe) from the Phase-4 run:
  `L_cond` = **0.909 at step 0**, and **0.95–0.99 for the whole run**; every other
  term (L_inv, L_var, L_cov, L_scalar, L_occ, L_phys, L_summary, L_scal_t) sat
  below 2 %.
- Prediction: `z_hat_occ_spec = spec_proj(occupancy_pred)` — a randomly-initialized
  `Linear(192,192)`.
- Target: `z_y_occ_spec = EMA_occ(full occ, scalar FiLM, spectrum_film_ema(c_physics))`.
- Loss: `L_cond = masked mean over 256 tokens of ||z_hat_occ_spec − z_y_occ_spec||²`.

## Root causes

**1. Scale dominance (proximate).** `L_cond` is an un-normalized MSE over 192-D
vectors. At init the prediction is uncorrelated with the target, so its per-element
squared error is ~Var(z) and its gradient dwarfs the others. `λ_cond = 1` with **no
ramp** turned it on at step 0. Meanwhile the VICReg terms' gradients are tiny early
(near-constant latents → the variance hinge is ~0), so the budget was ~entirely
`L_cond`.

**2. The mechanism is nullifiable (deeper).** `SpectrumFilm` is zero-initialized to
identity (γ=1, β=0), so at step 0 `z_y_occ_spec == z_y_raw` exactly — `L_cond` starts
as a *redundant copy of the stable-target regression*. Because the film was
**trainable and shared** with the student encoder, the loss is *minimized by driving
the film back toward identity* (target → stable target, which the student can already
predict). So even in principle this construction does **not force** spectrum use; it
can collapse to "target = stable target" and be satisfied.

**3. Student-side conditioning is a confound.** Conditioning the student occupancy
encoder on the spectrum changed the validated model's `z_x` and is not needed for the
target-side claim.

**4. `L_scal_t` shares the design issues** (raw MSE, no ramp), though its share stayed
small (~0.3–0.7 %).

## Fix

1. **Scale-free losses.** Use **cosine distance** (`1 − mean cos`) for `L_cond` and
   `L_scal_t` — bounded in [0, 2], normalized by vector norms, so they cannot hijack
   the gradient budget the way raw MSE does.
2. **Ramp the new terms in.** `λ_cond`, `λ_scal_t` start at 0 and ramp after a warmup
   (`staging.multi_target_start_step`, `multi_target_ramp_steps`), so the
   representation forms on the proven Phase-2/3 objective first.
3. **Make the conditioning non-nullifiable.** **Freeze** the spectrum film
   (non-trainable) and give it a **small non-zero** init so the target is genuinely
   spectrum-dependent at step 0 and *cannot* be optimized back to identity. The target
   is then a fixed function of (occ, spectrum) and the student must use the goal to
   match it.
4. **Target-only conditioning.** Revert the student encoder to the validated
   (spectrum-agnostic) form; the conditioning lives only on the target encoder. Drop
   the film's EMA (a frozen module needs none).
5. **Keep the stable spectrum-free target unchanged**, so Scenario A/B/C behaviour has
   a no-regression baseline.

## Expected check before re-examining gates

- `[grad-share]` for `L_cond`/`L_scal_t` must stay a *modest* fraction (target: each
  well under the VICReg invariant term) — i.e. no repeat of the 91–99 % hijack.
- The **no-shortcut probe** must show `z_hat_occ_spec` depends on the goal.
- Scenario A/B/C must not regress vs the Phase-2 control.
