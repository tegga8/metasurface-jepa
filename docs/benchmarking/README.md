# Benchmarking the Unified JEPA against MetaDiT

Design document for the MetaDiT-comparable benchmark harness. This folder defines
**what we measure, in what units, on what data, and why the numbers are directly
talliable to the MetaDiT paper** (`arXiv:2508.05076`, AAAI 2026).

Authority: `docs/implementation/unified_jepa/architecture_v5.md` §10 states the
benchmarking stance — "any paper claim should be benchmarked against (a) the
existing MetaDiT diffusion transformer on the pure-inverse-design scenario …
not against 'no architecture' as a strawman." This folder is where that stance
becomes executable.

Status: harness **implemented** (tests pass locally; the real run awaits the trained
checkpoint on Kaggle — see `BASELINE.md`). **No `src/` changes.** Measurement roadmap:
`ROADMAP.md`; current-architecture baseline: `BASELINE.md`; append-only results:
`RESULTS.md`; Phase 4 target design: `TARGET_DESIGN.md`.

---

## 1. Why MAE / AAE and not our existing metric

MetaDiT reports three numbers (paper Table 2):

| model | MAE | AAE |
|---|---|---|
| MetaDiT-S (released) | 0.0801 | 48.2495 |
| vanilla DiT baseline | 0.1677 | 100.9437 |
| AVG1 (mean-spectrum) | 0.5860 | 352.7424 |

plus a robustness metric **AAE&K** (AAE&2 = 58.80, AAE&4 = 68.73).

We already **reproduce** the MetaDiT-S row: `checkpoints/phase0/seed0_metric.json`
holds MAE **0.0803** / AAE **48.34** (Δ +0.24% / +0.18%), produced by
`scripts/eval/reproduce_metadit_baseline.py` running the official
`external/metadit/metric.py` on the released `generation/seed0.json`.

Our own evaluator (`scripts/eval/eval_scenarios.py`) instead headlines a
**target-std-normalized L1** spectrum error. That is a good internal metric but it
is **scale-free** — it is not the paper's unit, so it cannot be put in the same
column as 0.0801 / 48.25.

**Decision:** the benchmark harness makes **MAE / AAE / AAE&K primary** (paper
units) and keeps the existing normalized-L1 / IoU / scalar / gate suite as
**secondary** in the same table.

---

## 2. Comparability contract

1. **Same surrogate.** Both arms run the released frozen forward EM surrogate
   `data/metadit/weights/surrogate_model.bin` (`surrogate_s3`, 6.33M params).
   Ours is loaded by `src/physics/physics_loop.py::load_surrogate`.
2. **Same input convention.** The surrogate consumes `[B,3,64,64]` with
   `ch0 = occ·r_atom/5`, `ch1 = occ·h_atom`, `ch2 = l_lattice/3` (dense). This is
   exactly `src/data/factorize.py::assemble_metadit_geometry` and exactly
   `external/metadit/datapipe.py:36-51`. No conversion is needed.
3. **Same metric.** `MAE = mean|Ŝ − S|`, `AAE = sum|Ŝ − S|` over the `2×301`
   tensor, per item then averaged — identical to
   `external/metadit/metric.py::mean_absolute_error` / `accumulate_absolute_error`.
4. **Same split.** MetaDiT's released numbers are on the **test** split (17,489).
   The harness takes `--split {val,test}`; the paper tally uses `test`. The model
   was trained on `train_set.mat` only, so test is a clean held-out set.
5. **Same scenario.** MetaDiT does `spectrum → geometry` (pure inverse design),
   which maps to **Scenario A** of the unified model: full occupancy mask +
   all scalars unknown. Scenario B/C are reported but **not** tallied against the
   paper (MetaDiT has no partial-conditioning mode).

### 2.1 What we deliberately do NOT do

- **We do not call `restore_structure`.** That function in
  `external/metadit/metric.py` is MetaDiT's decoder for its own generated
  `3×32×32` tensor (threshold-at-mean, quadrant mirroring). Our deployment path
  (`UnifiedJEPA.decode_geometry(..., hard_forward=True)`) already yields a native
  `3×64×64` geometry in the surrogate's convention. Mirroring/thresholding a
  geometry that is already at the surrogate's resolution would corrupt it.
- **We do not compare against "no architecture".** The baselines are the real
  ones: MetaDiT-S (reproduced), AVG1, and nearest-neighbor retrieval.

### 2.2 The one known convention difference (must be stated in any write-up)

`AUDIT_REPORT_192D.md` §3.3: MetaDiT's own classifier-free guidance feeds a
**constant `0.5`-filled spectrum through the released encoder**, whereas this
repo **zeroes the conditioning and skips the frozen encoder** on null-goal steps
(`src/encoders/spectrum_encoder.py:104-110`). This affects *CFG / unconditional*
comparisons only. The **MAE/AAE headline uses the real conditioned output**, so
it is unaffected — but the difference must be noted wherever guided numbers are
shown.

---

## 3. AAE&K for a deterministic model

`AAE&K = (1/N) Σ_i max_{j∈[0,K)} AAE_ij` — the worst of K samples per condition
(robustness). MetaDiT varies the **diffusion seed**. The unified JEPA is
deterministic: one geometry per input.

**Choice (locked):** add a minimal **candidate generator** — `K` candidates per
condition from fixed-generator Gaussian jitter σ injected into the predictor's
masked-token latents before hard decode (`scripts/benchmark/candidate_sampling.py`).

- `K = 1, σ = 0` **must** reproduce the deterministic single-shot MAE/AAE
  (asserted in tests).
- Every AAE&K row carries metadata:
  `candidate_generator = "latent-jitter"`, `deterministic_model = true`,
  `is_diffusion_seed_diversity = false`.
- It measures **decode-robustness to injected noise only** and is never presented
  as MetaDiT-equivalent seed diversity (`AGENTS.md` rule 8: no scientific claim
  from "the code runs").
- σ is reported as a small curve (`[0, 0.01, 0.05, 0.10]`) so the reader sees the
  whole response, mirroring the discipline of
  `eval_scenarios.spectrum_sensitivity_probe`.

---

## 4. The reference table the harness must produce

Every benchmark run emits one table with all arms in the **same units**:

| arm | MAE | AAE | notes |
|---|---|---|---|
| Unified JEPA (ours), Scenario A | … | … | `<split>`, N=`<n>` |
| NN retrieval | … | … | recomputed in MAE/AAE units (§ METRICS) |
| AVG1 (dataset mean spectrum) | … | … | paper 0.5860 / 352.7424 |
| MetaDiT-S (reproduced) | 0.0803 | 48.34 | `seed0_metric.json` |
| MetaDiT-S (paper) | 0.0801 | 48.2495 | reference |
| vanilla DiT (paper) | 0.1677 | 100.9437 | reference |
| surrogate floor | ≈0.0084 | — | achievable lower bound (paper Table 1) |

plus **AAE&2 / AAE&4** rows for ours (analogue) beside the paper's 58.80 / 68.73.

The secondary suite (normalized-L1, IoU/F1 by masked/visible region, scalar MAE
known/unknown, occupancy-fraction variability, real/null/shuffled win rate,
guidance gap, sensitivity curve, CFG sweep) is printed alongside, never pooled
with the primary table.

---

## 5. Harness specification (to implement)

```
docs/benchmarking/            # this folder: README, METRICS, ROADMAP, BASELINE, RESULTS, TARGET_DESIGN
scripts/benchmark/
  metadit_metrics.py          # mae() / aae() / aae_and_k() mirroring external/metadit/metric.py
  candidate_sampling.py       # K-candidate latent-jitter generator (+ CFG-grid variant)
  baselines.py                # AVG1, NN-in-MAE-units, surrogate floor
  benchmark_metadit.py        # CLI driver, writes checkpoints/benchmark/<name>.json
scripts/diagnostics/
  masked_fill_check.py        # seam / texture / locality diagnostics
tests/
  test_benchmark_metadit_metrics.py
  test_masked_fill_check.py
```

Reused (do not reimplement): `eval_scenarios._load_eval` /
`_load_val_batch` / `_occupancy_metrics` / `evaluate_scenario` /
`real_null_shuffled` / `cfg_guidance_sweep` / `spectrum_sensitivity_probe` /
`nearest_neighbor_baseline`; `src/diagnostics/guidance_gap.py`;
`src/physics/physics_loop.py::load_surrogate`.

```
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
  --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
  --samples 512 --candidates 4 --device cpu --out checkpoints/benchmark/scenarioA.json
```

Metric definitions and formulas: `METRICS.md`.
Measurement roadmap (baseline → phases): `ROADMAP.md`.
Current-architecture baseline protocol: `BASELINE.md`.
Results ledger: `RESULTS.md`.
Phase 4 target design: `TARGET_DESIGN.md`.
