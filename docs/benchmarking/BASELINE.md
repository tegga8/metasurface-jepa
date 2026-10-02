# Phase 1 — current-architecture baseline

Purpose: answer **"is the current architecture working well?"** with recorded
numbers *before any change*. Every later phase in `ROADMAP.md` is judged against
this baseline. No code in `src/` is touched here.

---

## 1. Recorded run — provenance

- platform: **Kaggle**, GPU notebook `tejaspbiradar/metasurface-jepa-baseline-phase-1`
  (kernel v1, `enable_gpu`/`enable_internet` on), commit **`5d228ac`**.
- checkpoint: `full_epoch_final.pt` (`anosvol/metasurface-jepa-192d-full-epoch-ckpt`).
- data: `tejaspbiradar/metadit-aaai2026` (test split for the benchmark; val for scenarios).
- artifacts: `checkpoints/benchmark/baseline_*.json`.
- **deviation:** `masked_fill_check.py` crashed in kernel v1 (a CLI bug — `eval()`/
  `no_grad` missing and the wrong decode return bound). It was fixed in commit
  `5d27bc5` and re-run **locally on CPU (n=32, same checkpoint)**; identical numbers
  are expected on any device (deterministic, no training).

## 2. Results

| quantity | value | note |
|---|---|---|
| tests | 328 passed / 8 skipped / 1 failed | the 1 failure is pre-existing & unrelated (`test_phase5_contracts::test_physics_gradient_regression`) |
| **MAE** — ours, Scenario A, test (n=17,489) | **0.0673** | **beats MetaDiT-S 0.0801** |
| **AAE** — ours | **40.535** | beats MetaDiT-S 48.2495 |
| AAE&2 / AAE&4 | 42.14 / 43.95 | latent-jitter analogue; paper 58.80 / 68.73 |
| AVG1 MAE / AAE | 0.2574 / 154.966 | eval-split mean (paper 0.5860 / 352.7424 is the train mean — not the same predictor) |
| NN retrieval MAE / AAE (n=512) | 0.0551 / 33.161 | **better than ours** — the model is not yet beating retrieval on raw spectrum error |
| surrogate floor MAE | 0.0065 | achievable floor |
| **A / B / C win rate** (n=512) | **0.9941 / 0.9668 / 0.8984** | gate is ≥ 0.75 → **PASS** |
| **scalar-dep. one-known / two-known** | **0.5059 / 0.4512** | → **FAIL** (at chance) |
| guidance gap, hard stratum (all-unknown, mask 1.0) | **185.28** | **large — NOT Failure Mode 2** (spread 183–189 across ratios) |
| occupancy IoU/F1, Scenario A masked region | 0.7481 / 0.8559 | raw-sigmoid definition |
| predicted occupancy fraction (A) | 0.4270 vs true 0.4247; std 0.0673 | healthy, not collapsed (min 0.026, max 0.542) |
| spectrum-sensitivity A (scale 0.10) | 3.94 % pixels flipped, `design_moves`=True, monotone | design tracks the target spectrum |
| masked-fill filled IoU / F1 — A / B / C | 0.7446 / 0.8053 / 0.8616 (F1 0.854/0.892/0.926) | n=32 |
| masked-fill `visible_identity_maxdiff` | **0.0** (all scenarios) | only the masked part is filled — hard guarantee holds |
| masked-fill `seam_mismatch` B / C | 0.0684 / 0.0587 | ~6–7 % of boundary pixels disagree (mild seam effect) |
| masked-fill `filled_roughness_ratio` A/B/C | 1.055 / 1.117 / 0.979 | texture ≈ the true pattern's (not blurred) |
| masked-fill `locality_ratio` B / C | 0.496 / 0.385 | near context matters ~2–2.6× more than far → partially localised; A undefined (full mask) |

## 3. Reading — is the current architecture "working"?

- **Spectrum fidelity: yes, and it beats the paper baseline** on the same metric
  and surrogate (MAE 0.0673 vs 0.0801; AAE 40.5 vs 48.2), with A/B/C gates passing.
- **The suspected Failure Mode 2 is NOT present:** the guidance gap is large
  (~185), and real-vs-shuffled wins on 99.4 % of Scenario-A samples. The spectrum IS
  used.
- **Masked-fill is sound:** the visible region is byte-identical to the input
  (only the hole is generated), the fill's texture matches the true pattern, and the
  latent is partially localised (near context weighs ~2–2.6× far).
- **The one clear failure is the scalar path** (both strata at chance), confirming
  `REPORT.md` §17: the scalar-summary → predictor route is effectively dead. This is
  what Phase 4 (multi-target) and Phase 5 (scalar capacity) target.
- **Open caveat:** NN retrieval (0.0551) still beats our MAE — the model is not yet
  ahead of a simple retrieval baseline on raw spectrum error, even though it beats
  MetaDiT-S.

## 4. Gate to proceed

Baseline recorded; failing aspect = **scalar dependence**; passing aspects are
no-regress constraints. → proceed to Phase 2. No architecture change lands until a
phase's own before/after battery is recorded in `RESULTS.md`.
