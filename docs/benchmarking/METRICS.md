# Metric catalog

Every metric the benchmark reports: formula, units, source, and paper reference.
Split into **CREATE** (new, MetaDiT-comparable) and **REUSE** (already in the repo).

Notation: `Ŝ = surrogate(G)` is the frozen surrogate's spectrum prediction for the
deployed geometry `G`; `S` is the ground-truth target spectrum. Both are
`[B, 2, 301]` (`[real, imag]` × 301 frequencies). `C = 2`, `F = 301`.

---

## CREATE — MetaDiT-comparable (primary)

### MAE — mean absolute error
```
MAE = (1/N) Σ_i (1/(C·F)) Σ_{c,f} |Ŝ_i[c,f] − S_i[c,f]|
```
Mean over all `2×301` values, then over items. Units: same as the spectrum.
Mirrors `external/metadit/metric.py::mean_absolute_error`.
**Paper:** MetaDiT-S 0.0801 · vanilla DiT 0.1677 · AVG1 0.5860.
**Reproduced:** 0.0803 (`checkpoints/phase0/seed0_metric.json`).

### AAE — accumulated absolute error
```
AAE = (1/N) Σ_i Σ_{c,f} |Ŝ_i[c,f] − S_i[c,f]|
```
Sum over all `2×301` values, then mean over items. Mirrors
`external/metadit/metric.py::accumulate_absolute_error`.
**Paper:** MetaDiT-S 48.2495 · vanilla DiT 100.9437 · AVG1 352.7424.
**Reproduced:** 48.34.

### AAE&K — worst-of-K robustness (analogue)
```
AAE&K = (1/N) Σ_i max_{j ∈ [0,K)} AAE_ij
```
where `AAE_ij` is the AAE of candidate `j` for condition `i`. Mirrors
`external/metadit/metric.py::calculate_aaeandk`.
**Caveat:** MetaDiT draws candidates from **diffusion seeds**; ours come from
latent jitter (see `README.md` §3). Reported with
`is_diffusion_seed_diversity = false`. `K=1, σ=0` reproduces the deterministic
AAE by construction (asserted).
**Paper (not reproducible — only seed0 released):** AAE&2 = 58.80, AAE&4 = 68.73.

### AVG1 baseline
Predict the dataset-mean spectrum for every item; score MAE/AAE. The floor for
"learned nothing about the condition".
**Paper:** 0.5860 / 352.7424.

### NN retrieval (in MAE/AAE units)
Existing `eval_scenarios.nearest_neighbor_baseline` returns normalized-L1 units.
The harness recomputes the same retrieval (L1 nearest train spectrum → its real
geometry → frozen surrogate) and scores it with **MAE/AAE** so it sits in the same
table. This is the baseline `architecture_v5.md` §10 names for the retrofit
scenario.

### Surrogate floor
The surrogate's own error vs the true spectrum, in MAE units: `MAE(surrogate(G_true), S_true)`
over a sample of the split. No geometry→spectrum method can beat it.
**Paper Table 1 / local estimate:** ≈ 0.0084 (local p50 0.0061).

---

## REUSE — secondary suite (already implemented)

| Metric | Formula / definition | Source |
|---|---|---|
| Normalized-L1 spectrum error | `mean_{c,f} |Ŝ−S| / clamp(std(S_i), 1e-6)` | `eval_scenarios._spectrum_error` |
| Occupancy IoU / F1 / precision / recall | on occupied class, `pred > 0.5`; split masked vs visible region | `eval_scenarios._occupancy_metrics` |
| Occupancy fraction stats | per-sample `mean(pred>0.5)`, mean/std/min/max, all-empty/all-occupied | `eval_scenarios._collapse_metrics` |
| Scalar MAE (known / unknown) | `mean|scalar_pred − sv|` over known and over unknown positions | `eval_scenarios.evaluate_scenario` |
| Real/null/shuffled gate | paired per-sample win rate `real < shuffled`; threshold `eval.gate_beats_fraction_min` (0.75) | `eval_scenarios.real_null_shuffled`, `_gate_beats_fraction` |
| Scalar-dependence gate | same win rate with deranged scalar conditioning | `eval_scenarios.scalar_dependence` |
| Spectrum-sensitivity probe | target-spectrum perturbation → geometry move / pixels flipped, monotonicity | `eval_scenarios.spectrum_sensitivity_probe` |
| CFG guidance sweep | normalized-L1 of the guided design vs guidance weight `w` | `eval_scenarios.cfg_guidance_sweep` |
| Diversity / determinism | pairwise spectrum diversity (deterministic at scale 0) | `eval_scenarios.diversity_check` |
| Guidance gap | `mean_i ‖z_real_i − z_null_i‖₂ / std(z_real_i)`, per scalar stratum | `src/diagnostics/guidance_gap.py::normalized_gap_stats` |
| MetaDiT baseline reproduction | official `metric.py` on released `seed0.json` | `scripts/eval/reproduce_metadit_baseline.py` |

### Masked-fill metrics (Phase 0 diagnostic)

Source: `scripts/diagnostics/masked_fill_check.py` (tests: `tests/test_masked_fill_check.py`).
Convention: mask `(B,16,16)`, 1 = visible, 0 = masked; the *filled region* is the masked pixels.

| Metric | Definition | Reads |
|---|---|---|
| `visible_identity_maxdiff` | max `abs(deployed − input)` over **visible** pixels | hard retention guarantee (must be 0) |
| `filled_region.{iou,f1,precision,recall}` | pred vs true over the filled region | completion quality |
| `seam_band` | same IoU/F1 restricted to the boundary band (filled pixels within 1 px of visible) | seam quality |
| `seam_mismatch` | fraction of boundary-band pixels where pred binary ≠ true | discontinuity at the seam |
| `filled_roughness_ratio` | mean adjacent-difference inside the fill, pred / true | <1 oversmoothed, >1 noisy |
| `filled_agreement_{pred,true}` | fraction of adjacent filled pairs sharing a value | periodicity / continuity proxy |
| `locality_ratio` | change when flipping the k **farthest** vs **nearest** visible pixels | ≈0 localised, ≈1 global |

---

## Reporting rules (from `architecture_v5.md` §8 / phase MDs)

- Scenarios A / B / C reported **separately and never pooled**.
- Easy / hard strata reported separately; the **hard stratum** (Scenario A) is the gate.
- The primary gate statistic is the **paired win rate**, not batch means (heavy
  error tail — 27 / 17,488 samples score `1–30` while the median is 0.075).
- No single pooled headline score.
- Every number is attached to its `split`, `n_samples`, and data mode (REAL vs SMOKE).
