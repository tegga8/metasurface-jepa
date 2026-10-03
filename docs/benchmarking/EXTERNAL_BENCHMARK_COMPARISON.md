# External benchmark comparison — MetaDiT (AAAI 2026) vs unified 192-D JEPA

Separated from model changes. Audit performed 2026-10-03 against the live repository;
no model/training change accompanies this document.

## 1. Metric definition (parity VERIFIED)

Authoritative source: `external/metadit/metric.py`.

| metric | definition (reference) | per-item meaning | direction |
|---|---|---|---|
| `MAE` | `mean(abs(y_pred - y_true))` over all elements | mean L1 over the 2×301 condition | lower better |
| `AAE` | `sum(abs(y_pred - y_true))` — called **per item** in `eval_loop` | **sum** L1 over the 2×301 condition | **lower is better** |
| `AAE&K` | over the first K seeds: per-condition **max** AAE across seeds, then mean | worst-case over K stochastic draws | lower better |

`eval_loop` builds, per item, `predicted_spec = surrogate(restore_structure(generation))`
and compares to `item["condition"]`; the reported `MAE`/`AAE` are the **means over
seed0's items** of the per-item values.

**Our parity is enforced by real tests** (`tests/test_benchmark_metadit_metrics.py`):
`test_batched_mae_aae_match_reference` (batched `mae`/`aae` vs the reference functions,
with the per-item averaging the reference relies on) and `test_aae_and_k_matches_reference`
(vs `calculate_aaeandk`). Not vacuous. **AAE is lower-is-better.**

## 2. Evaluation-set parity

| | MetaDiT | ours |
|---|---|---|
| dataset / split | released `split_data/{train,val,test}_set.mat` (8:1:1) | **same files**, same 139,906 / 17,488 / 17,489 |
| test conditions | `condition` = real/imag `(2, 301)` | same `(2, 301)` from `real`/`imag` |
| geometry at the surrogate | 3×32×32 generation → `restore_structure` (threshold + quadrant mirror) → 3×64×64 | native 3×64×64 (`assemble_metadit_geometry`, same scalar→channel convention) |

The surrogate boundary therefore matches; the **generation protocol differs** (below).

## 3. Surrogate parity — like-for-like

- The MetaDiT **reference number in the paper** (`0.0801 / 48.2495`) was produced with
  the paper's StarNet-MLP surrogate (1.90 M params, paper Table 1).
- **Both our model and our reproduced MetaDiT baseline are scored by the same released
  surrogate** `surrogate_s3` (`data/metadit/weights/surrogate_model.bin`, 6.33 M), via
  `physics.physics_loop.load_surrogate`. So the *model-vs-reproduced-MetaDiT* comparison
  is scored by one and the same surrogate — **like-for-like at the surrogate level**.

## 4. MetaDiT result provenance

- `0.0801 / 48.2495` = **paper Table 2 citation only** (not a released artifact we run).
- The repository additionally holds a **local reproduction**, `0.0803 / 48.34`
  (`checkpoints/phase0/seed0_metric.json`), produced by
  `scripts/eval/reproduce_metadit_baseline.py` — the **official `metric.py`** on the
  released `seed0.json` generation with the **released surrogate**.
- **Anchor for the comparison: the reproduced `0.0803 / 48.34`**, not the paper citation.
- Only **seed0** is released ⇒ `AAE&K` for K>1 cannot be reproduced.

## 5. AAE&K — an analogue, NOT equivalent

| | MetaDiT | ours |
|---|---|---|
| candidate source | independent **diffusion** generations (seeds {0,7,42,3407}) | deterministic model + **latent-jitter** (`σ`-perturbed decodes) |
| flag | — | `candidate_generator: latent-jitter`, `is_diffusion_seed_diversity: false` |

Our `AAE&K` must **never** be presented as equivalent to MetaDiT's stochastic
multi-sample metric; it is an *analogue*.

## 6. Claim policy

- **Supportable:** "our model **beats MetaDiT-S on the paper metric, under the released
  surrogate, on the same test split**" — ours MAE **0.0490 ± 0.0013 (3 training seeds)**
  vs the reproduced MetaDiT-S **0.0803** (and the paper's 0.0801).
- **Not supportable:** "beats MetaDiT" unqualified — the generation protocols differ
  (ours deterministic, MetaDiT diffusion), our AAE&K is an analogue, and the exact
  headline number is a citation plus a close local reproduction.
- **Also note:** NN retrieval (same split/surrogate) scores MAE ≈ 0.0296, still below
  ours; the comparison to MetaDiT does **not** imply beating retrieval.

## 7. Provenance summary

| | model / source | checkpoint | split | surrogate | n | generation | provenance |
|---|---|---|---|---|---|---|---|
| ours | unified 192-D JEPA | `metasurface-jepa-fe-current` (70k, seeds 0-2) | test (17,489) | released `surrogate_s3` | 17,489 | deterministic single decode | established 3-seed |
| MetaDiT-S (reproduced) | MetaDiT-S | released `seed0.json` generation | test | released `surrogate_s3` | 17,489 | diffusion (seed 0) | `checkpoints/phase0/seed0_metric.json` |
| MetaDiT-S (paper) | MetaDiT-S | — | test | paper StarNet-MLP | — | diffusion | AAAI 2026 Table 2 (citation) |
| retrieval | NN (pool 20k) | — | test subset (n=512) | released `surrogate_s3` | 512 | retrieval | `metadit_metrics` / `nn_scoping.py` |
