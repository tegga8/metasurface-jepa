# Overfitting check — train vs held-out

**Verdict: the model is NOT overfit on typical samples.** Train and held-out
medians/p90/p99 are essentially identical. The difference between train and
held-out *means* is a rare catastrophic tail (~0.24 % of samples), not fitting.

## Splits (MetaDiT's own, shared by both models)

| split | N | used for |
|---|---|---|
| train | **139,906** | gradient updates only |
| val | **17,488** | training-time validation (easy/hard strata) |
| test | **17,489** | untouched held-out (the baseline + comparison split) |

The model trained on `train_set.mat` **only**; `val` was used for validation, `test`
was never touched → clean held-out.

## Protocol

Scenario A (full occupancy mask + all scalars unknown — the hardest stratum),
per-item MAE through the frozen surrogate (`mean|surrogate(design) − target|` over
the 2×301 spectrum), **n = 2048 random subsample (seed 0) per split**, eval mode.

## Results

| split | mean | median | p90 | p99 | max | frac > 1 |
|---|---|---|---|---|---|---|
| train | 0.0528 | **0.0426** | 0.1026 | 0.1992 | 0.28 | 0.0000 |
| val | 0.0785 | **0.0425** | 0.1086 | 0.1980 | 13.12 | 0.0024 |
| test | 0.0824 | **0.0432** | 0.1108 | 0.1909 | 13.00 | 0.0024 |

- **Medians match** (0.0426 / 0.0425 / 0.0432) — a typical held-out design is
  predicted as well as a typical training design.
- **p90 / p99 match** — the bulk distribution does not shift on held-out data.
- **The mean gap is entirely the tail**: train `max = 0.28`, `frac>1 = 0`; val/test
  have `frac>1 ≈ 0.24 %` with `max ≈ 13`. A handful of catastrophic samples inflate
  the held-out mean. This is the same heavy tail flagged in `REPORT.md` (27 of
  17,489 samples = 0.15 % on Scenario A, no input property predicts it).

## Why memorisation is structurally unlikely

- **~1 epoch:** 70,000 steps × batch 2 = 140,000 samples ≈ the 139,906 training
  samples — each design was seen roughly once, so there is little opportunity to
  memorise.
- **11.4 M trainable params** against 139,906 designs, with an EMA target encoder
  and VICReg variance/covariance penalties (anti-collapse regularisation).

## Caveat on the mean

The mean is unstable at small n (list below); report the median/p90 for the
typical-case claim and the tail fraction for the failure claim.

| source | test MAE |
|---|---|
| n=1024 subsample (driver, seed 0) | 0.0985 |
| n=2048 subsample (seed 0) | 0.0824 |
| **full test, n=17,489** (Phase 1 baseline) | **0.0673** |

## Conclusion

Overfitting is **not** the problem. The open issue is the **catastrophic tail**
(~0.2 % of held-out designs), which is separate from generalisation and is the
thing worth diagnosing next.
