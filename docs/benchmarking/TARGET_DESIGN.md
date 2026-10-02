# Target design — multi-target JEPA (Phase 4)

Authority for Phase 4 of `ROADMAP.md`. Resolves the two structural weaknesses the
current architecture has, using the decisions locked this cycle.

## 0. Why the current target has to change

`architecture_v5.md` §0.2 (lines 79-99): the JEPA target `z_y_raw` is built from
occupancy + true scalars only and **never sees the spectrum**, by design — and the
doc concedes the loss "can only ever *reward* … never *require* spectrum use." The
teacher is also only a **partial** EMA: `occupancy_encoder` → `self.ema` and
`scalar_encoder` → `self.scalar_mlp_ema`; `fusion_encoder` and the GCLCT `predictor`
(the only spectrum-facing modules) have **no EMA counterpart**. So the teacher is
spectrum-blind, and the predictor has no obligation to use the goal.

## 1. Decisions locked

- **Multi-target objective.**
- **Include the spectrum in the target side.**
- **Keep scalar FiLM on the geometry target** (so the student must infer unknown
  scalars to match it).

## 2. Correction to the earlier sketch — the naive spectrum target is shortcut-able

`z_y_spec = frozen spectrum encoder latent of the TRUE spectrum` **forces nothing**:
the student already receives the goal spectrum as input, so a head predicting the
encoder latent of that same spectrum can be satisfied by copying the input — the
geometry never has to change. The target must therefore be a **spectrum-conditioned
*geometry* latent**, which cannot be copied and which changes with the spectrum.

## 3. Targets (multi-target)

| target | construction | role |
|---|---|---|
| `z_y_occ_stable` | `EMA_occ(full occ, FiLM=EMA_scalar(true scalars))` | stable answer key; clean real/null/shuffled control (**existing — keep**) |
| `z_y_occ_spec` | `EMA_occ(full occ, FiLM=EMA_scalar(true scalars), spec_film(c_physics_true))` | **NEW** — spectrum-conditioned; forces spectrum use |
| `z_y_scal` | `EMA_scalar(true scalars).summary` (`[B,192]`) | **NEW** — forces scalar inference (door (b)) |

Student predicts all three from `(masked occ, goal spectrum, partial scalars)`:
`z_hat_occ` (existing predictor tokens), `z_hat_occ_spec` (**new** head off the
predictor trunk), `z_hat_scal` (existing scalar-summary query).

## 4. Objective

```
L = λ_inv·L_inv + λ_var·L_var + λ_cov·L_cov + λ_scalar·L_scalar
  + λ_occ·L_occ + λ_phys·L_phys + λ_summary·L_summary          # existing, unchanged
  + λ_cond·JEPA(z_hat_occ_spec, z_y_occ_spec)                  # NEW
  + λ_scal_t·JEPA(z_hat_scal, z_y_scal)                        # NEW
```
`JEPA` = masked-token MSE/cosine (reuse `src/losses/jepa_loss.py`, `vicreg.py`).

Why it works where the sketch did not: `z_y_occ_spec` is a *geometry* latent that
depends on the true spectrum, so the only way to match it is to transform
`(context, goal)` into that latent — i.e. to use the goal. The stable target stays
for the unentangled control.

## 5. Implementation sketch

| file | change |
|---|---|
| `src/assembly.py` | `spectrum_film` module; second EMA target `z_y_occ_spec`; `z_hat_occ_spec` head; `z_y_scal` / `z_hat_scal` exports |
| `src/encoders/occupancy_encoder.py` | accept optional `spectrum_film_params` (applied like the scalar FiLM) |
| `src/losses/unified_losses.py` | `L_cond`, `L_scal_t` terms + weights |
| `configs/unified.yaml` | `λ_cond`, `λ_scal_t` with their **own ramp, on before physics** (representation-first) |

`spectrum_film` = small MLP `c_physics 384 → 2·hidden` per block, **zero-init to
identity** (repo AdaLN-zero convention). Reuses `SpectrumPath`'s frozen `c_physics`;
no change to the frozen spectrum encoder.

## 6. Gates (Phase 4)

1. **Forcing happened:** scalar-dependence win rate moves off ~0.5 toward 0.94–0.97;
   hard-stratum guidance gap rises. Otherwise the target still isn't forcing — **stop,
   do not loosen the gate**.
2. **No shortcut:** `z_hat_occ_spec` must change materially under null vs real goal at
   fixed occupancy; if not, the head is copying the input — redesign, do not ship.
3. **No regression:** Scenario-A MAE/AAE does not regress; the stable target / real-null-
   shuffled control still behaves.

## 7. Risks / open questions

- **Two geometry targets** double the occupancy-latent heads (cheap: shared trunk). If
  heavy, fallback = replace the stable target with the conditioned one (loses the
  unentangled control).
- **EMA scope:** target #2 needs its own EMA of the spectrum-conditioned encoder. The
  fusion/predictor stay student-only (as in I-JEPA the predictor is not in the teacher).
- **Not a literal "EMA copy of the whole student"** — the predictor stays student-only.
  A literal symmetric teacher (EMA of the full stack incl. fusion) is a larger change
  that would entangle the real/shuffled control; not adopted.
