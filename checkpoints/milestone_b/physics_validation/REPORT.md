# Physics–Geometry Validation — Gates 0 & 0.5 Report

**Spec:** PHYSICS_GEOMETRY_VALIDATION_V2 §8–§10 · **Date:** 2026-08-21 · **Executed:** local dev machine (CPU), per Compute Environment section (inference + toy-scale training only)

**Status: COMPLETE — classified 0D (corrected 2026-08-21, see §Classification correction below). STOP called per spec; no further validation work in this line.**

---

## What was run

| Script | Output |
|---|---|
| `python scripts/diagnostics/gate0_occupancy_audit.py` | `gate0_occupancy_audit.json` |
| `python scripts/diagnostics/gate0_5_trivial_baseline.py` | `gate0_5_trivial_baseline.json` |

Tests: `tests/test_gate0_diagnostics.py` — 5/5 pass after final edits.

Both gates use the raw `.mat` splits exactly as `src/data/dataset.py` reads them (train 139,906 / val 17,488 / test 17,489) and the repo's own `BlockMasker` semantics. No JEPA component was trained or modified.

---

## Gate 0 — Is occupancy a deterministic rasterization of (l_lattice, h_atom, r_atom)?

**Answer: NO.** The 64×64 binary occupancy mask is a one-to-many function of the three scalars.

- **G0-A (count diagnostic):** occupied-pixel count vs `a·r_atom² + b`: **R² = 2.9e-06**, RMSE 324.8 px on a mean count of 1725.7. Correlations of count with each scalar: |r| ≤ 0.0032. Occupied area carries essentially zero signal about the parameters.
- **G0-B (exact-bucket identity, decisive):** bucketing all 174,883 samples by exact (l,h,r) at the data's native 0.01 quantization gives 124,587 buckets, 34,799 with ≥2 members → 73,339 within-bucket pairs compared:
  - IoU mean **0.571** (p10 0.363, p90 0.746)
  - Hamming distance mean **938.7 px** (of 4096)
  - near-identical pairs (IoU ≥ 0.95): **2 of 73,339 = 2.7e-05**
  - fully identical pairs: **0**; buckets whose members are all mutually identical: **0 / 34,799**
- **G0-C (learned generator check):** tiny MLP (l,h,r) → 64×64 occupancy reaches only val IoU **0.690 ± 0.128** (precision 0.770, recall 0.870) — consistent with G0-B: the residual variability is real data structure, not noise a generator could nail.
- **G0-D (observability under actual BlockMasker):** fraction of samples with each scalar recoverable from visible pixels — 25% mask: 100/100/100%; 50%: 100/99.9/99.9%; 75%: 100/95.7/95.7%; 100%: 0/0/0. (`l` is readable from any visible pixel; `h`,`r` need a visible *occupied* pixel.)

---

## Gate 0.5 — Trivial end-to-end ceiling baseline

TrivialMLP (608 → 256 → 128 → 64 → 3, ReLU) regressing (l,h,r) from `[visible-param values + observability flags (dataset's own grid encoding), raw (2,301) spectrum]`, under the **same BlockMasker masks** the JEPA sees. Pre-registered threshold: near-solving ⇔ **NRMSE ≤ 0.10 per scalar** on held-out val (n=1024; train n=8192).

| mask | NRMSE l | NRMSE h | NRMSE r | all ≤ 0.10 | best val MSE |
|---|---|---|---|---|---|
| 0.25 | **0.034** | **0.047** | **0.027** | **YES** | 6.2e-05 |
| 0.50 | **0.053** | **0.063** | **0.075** | **YES** | 3.5e-04 |
| 0.75 | 0.078 | 0.075 | 0.132 | no (r) | 1.0e-03 |
| 1.00 | 0.809 | 0.241 | 0.518 | no | 1.9e-02 |

- Occupancy lookup-render IoU ≈ 0.57–0.58 at every ratio — the ≥0.95 IoU criterion is **not applicable** (Gate 0 found non-determinism; recorded as such in the JSON), and would fail regardless.
- Zero-context note: at 100% masking (no geometry at all) the trivial MLP still recovers h_atom to NRMSE 0.24 from the raw spectrum alone — relevant when interpreting any future zero-context claims.

---

## Classification: **0D** (corrected 2026-08-21 — was incorrectly labeled 0B)

> **Classification correction (2026-08-21, operator-directed).** The original report
> labeled this result 0B. That label was wrong under the specified matrix. The
> measured facts are: occupancy deterministic? **NO**; scalar baseline near-solves?
> **YES at 25–50%**. Under the authoritative matrix
>
> | | scalar baseline near-solves | does not near-solve |
> |---|---|---|
> | **occupancy deterministic** | (n/a) | **0B** |
> | **occupancy not deterministic** | **0D — this result** | **0C** |
>
> the correct branch is **0D**: non-deterministic occupancy + scalar subproblem
> near-solved. **0D does NOT mean the complete product is solved.**

Correct interpretation wording:

> **The scalar physical-parameter subproblem is near-solved at low/moderate masking,
> while independent spatial occupancy remains unsolved.**

The previously written sentence "the product problem is solved by a trivial MLP" is
retracted as unsupported: the Gate 0.5 MLP predicts only `[l_lattice, h_atom, r_atom]`
— it is NOT a geometry-completion baseline and provides no evidence about spatial
occupancy prediction.

Original mapping table retained for the record (superseded by the matrix above):

| | trivial baseline near-solves | does not near-solve |
|---|---|---|
| **occupancy deterministic** | 0A — everything trivial; premise collapses | 0B |
| **occupancy not deterministic** | ~~0B~~ → **0D (correct)** | 0C |

Observed: occupancy **not deterministic** (G0-B decisive) **and** the scalar
parameter subproblem **near-solved** by a trivial MLP at ≤50% masking under identical
mask semantics. The full spatial geometry task remains unsolved by this baseline.

## Implications for the JEPA validation

1. **Geometry is a rich object, not a scalar readout.** Same (l,h,r) → many distinct layouts (IoU 0.57 among exact-parameter twins). The spatial degrees of freedom are real, so a JEPA over geometry is not redundant with the 3 scalars. This half of the premise survives.
2. **Scalar-regression accuracy at ≤50% masking is a saturated metric.** A 3-hidden-layer MLP already sits at the pre-registered near-solving bar there. Any downstream comparison claiming JEPA value via param NRMSE at low masking is measuring nothing. Headroom exists only at **≥75% masking** (r_atom fails at 0.13) and at 100% (spectrum-only).
3. **Recommended protocol change for later milestones:** benchmark predictive value at 75–100% masking; treat 25–50% scalar NRMSE as a sanity floor, not an objective.

## Deviations & fixes (recorded per standing rules)

1. **Optimizer schedule (not a ceiling change):** fixed-LR Adam stalls at ~3e-3 MSE — an optimization artifact; the input literally contains the targets when observable (verified max feature/target error < 1e-6). Cosine annealing over 2400 epochs + best-val-state restore reaches the reported numbers; a linear probe independently corroborates recoverability (4.7e-5 MSE).
2. **Val-set construction bug fixed:** val features previously paired params of row `j` with the pattern of sampled index `vi[j]` (features remained self-consistent, but sample identity was mixed across pattern/spectrum/params). Now built consistently from `va_params[vi]`.
3. **Early stopping disabled by default:** patience-based stops were killing runs mid-plateau, hundreds of epochs before the annealed fine-tune phase produced the true optimum. Full schedule + best-state restore instead.
4. **Input standardization** (train-set per-feature z-score of the 608 dims) added purely for optimizer conditioning; invertible, representation unchanged.

---

**STOP.** Per spec §10: classification **0D** recorded (corrected 2026-08-21); no
further gates in this validation line without operator direction. Follow-up work is
governed by the physics-geometry latent-selection specification (spatial latent probe +
physics target selection diagnostics).
