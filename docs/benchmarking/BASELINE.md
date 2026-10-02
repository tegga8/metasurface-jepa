# Phase 1 — current-architecture baseline

Purpose: answer **"is the current architecture working well?"** with recorded
numbers *before any change*. Every later phase in `ROADMAP.md` is judged against
this baseline. No code in `src/` is touched here.

Prerequisite: the trained checkpoint (`checkpoints/unified/latest.pt`) — a Kaggle
artifact (`CLOUD_TRAINING.md`). Run the battery on Kaggle (or wherever the
checkpoint lives); the local machine cannot run it (no checkpoint, dev-only).

---

## 1. Battery (run and record every line)

```
# tests
python -m pytest tests/ -q --tb=line

# MetaDiT-comparable benchmark (paper units) — Scenario A, full test split
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
    --samples 0 --candidates 4 --nn-samples 512 \
    --out checkpoints/benchmark/baseline_scenarioA.json

# per-scenario gates (A = hard stratum = the gate)
python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --scenario all \
    --samples 512 > checkpoints/benchmark/baseline_eval_scenarios.json

# masked-fill verification (seam / texture / locality)
python scripts/diagnostics/masked_fill_check.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --samples 32 \
    > checkpoints/benchmark/baseline_masked_fill.json

# guidance gap sweep
python scripts/diagnostics/run_guidance_gap_sweep.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt
```

## 2. Results table (fill from the run above)

| quantity | value | notes |
|---|---|---|
| tests | __ passed / __ skipped / __ failed | |
| **MAE** (ours, Scenario A, test) | | vs MetaDiT-S 0.0801 |
| **AAE** (ours) | | vs 48.2495 |
| **AAE&2 / AAE&4** (ours) | | analogue; paper 58.80 / 68.73 |
| AVG1 MAE / AAE | | paper 0.5860 / 352.7424 |
| NN MAE / AAE | | |
| surrogate floor MAE | | paper ≈0.0084 |
| A / B / C win rate | | gate ≥ 0.75 |
| scalar-dep. one-known / two-known win rate | | **expected FAIL ≈0.50** |
| guidance gap (hard stratum) | | |
| physics gradient share | | **expected ≈66 %** |
| masked-fill: seam_mismatch (A/B/C) | | |
| masked-fill: filled_roughness_ratio (A/B/C) | | |
| masked-fill: locality_ratio (A/B/C) | | |

## 3. What "working well" means (from `checkpoints/unified/REPORT.md`)

- A/B/C real-vs-shuffled gates **pass** (A 0.9939 full split).
- **Scalar-dependence gates FAIL** (~0.50) — the known weak point.
- Physics is **≈66 %** of the gradient budget — too high for representation-first.
- Guidance gap small on the hard stratum — Failure Mode 2 suspected.
- Masked-fill seam/texture/locality: **previously unmeasured** — this run establishes
  them.

## 4. Record

- Fill §2 above and paste the same numbers as the **baseline block** in `RESULTS.md`
  (commit = the pre-change commit, date, config hash).
- The failing aspects go into the Phase 2+ work items; the passing ones become
  no-regress constraints.

## 5. Gate

Baseline recorded and failing aspects listed → proceed to Phase 2. Until then no
architecture change lands.
