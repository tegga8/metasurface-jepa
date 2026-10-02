# Results ledger (append-only)

One block per phase: the **before** (baseline) numbers and the **after** numbers,
same battery, same split / config / seed. Never overwrite a block — append.
Numbers come from `checkpoints/benchmark/*.json` + the eval-scenario output.

See `ROADMAP.md` §"Standard phase protocol" and `BASELINE.md`.

---

## Baseline — current architecture (Phase 1)

> **Corrections pending re-run (review A2/A4/A5).** The numbers below were produced
> by the *pre-fix* driver: the NN row is **not** like-for-like (512-item pool, first
> 512 test items vs the model's full 17,489) and the AVG1 row used the evaluated
> split's own mean (transductive), not the train mean. The driver has been fixed;
> this block must be **regenerated** before it is cited. The MAE/AAE/AAE&K headline
> and the A/B/C gates are unaffected by these fixes.

- date: 2026-10-02   commit: `5d228ac` (kernel) / `5d27bc5` (masked-fill fix)
- platform: Kaggle GPU kernel `tejaspbiradar/metasurface-jepa-baseline-phase-1`
- checkpoint: `full_epoch_final.pt` (`anosvol/metasurface-jepa-192d-full-epoch-ckpt`)
- artifacts: `checkpoints/benchmark/baseline_scenarioA.json`,
  `baseline_eval_scenarios.json`, `baseline_masked_fill.json`, `baseline_guidance_gap.json`

| quantity | value | notes |
|---|---|---|
| tests | 328 passed / 8 skipped / 1 failed | failure pre-existing & unrelated |
| MAE (ours, Scenario A, test, n=17,489) | **0.0673** | beats MetaDiT-S 0.0801 |
| AAE (ours) | **40.535** | beats 48.2495 |
| AAE&2 / AAE&4 | 42.14 / 43.95 | analogue; paper 58.80 / 68.73 |
| AVG1 MAE / AAE | 0.2574 / 154.966 | eval-split mean |
| NN MAE / AAE (n=512) | 0.0551 / 33.161 | **better than ours** |
| surrogate floor MAE | 0.0065 | |
| A / B / C win rate (n=512) | 0.9941 / 0.9668 / 0.8984 | gate ≥ 0.75 → PASS |
| scalar-dep one-known / two-known | 0.5059 / 0.4512 | **FAIL** |
| guidance gap (hard stratum) | 185.28 | large → not Failure Mode 2 |
| occupancy IoU/F1 (A, masked) | 0.7481 / 0.8559 | |
| pred occupancy fraction (A) | 0.4270 (true 0.4247); std 0.0673 | not collapsed |
| masked-fill filled IoU A/B/C | 0.7446 / 0.8053 / 0.8616 | n=32 |
| masked-fill visible_identity_maxdiff | 0.0 | only the hole is filled |
| masked-fill seam_mismatch A/B/C | n/a / 0.0684 / 0.0587 | |
| masked-fill locality_ratio A/B/C | n/a / 0.496 / 0.385 | partially localised |
| physics gradient share | not measured in this run | train-time; ~66 % per REPORT §13.1 |

**Gate:** baseline recorded; failing = scalar dependence; the rest are no-regress
constraints → proceed to Phase 2.

---

## Phase 2 — representation-first training schedule

- date: _TBD_   commit: _TBD_   change: _
- before: (copy Baseline)
- after: _
- delta / gate: _

---

## Phase 3 — representation hygiene

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 4 — multi-target objective (spectrum + scalar in the target side)

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
- shortcut probe (null vs real at fixed occupancy): _TBD_

---

## Phase 5 — scalar capacity & bounds

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 6 — encoder sizing

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
