# Validation plan — what was run, what holds, and what must run next

Written 2026-10-08. Grounded in `architecture_v5.md` §8 (the mandated gates),
`ACCEPTANCE_PROTOCOL.md` (the replication rules), `METRICS.md` (metric
definitions) and the live `tests/` suite. Supersedes nothing; the ledger
(`RESULTS.md`) remains the record of numbers.

---

## 0. The rule this plan exists to enforce

**Training optimises through the frozen CST surrogate. Meep is an independent
validator, never a training signal.** From `publishability_assessment.md`
(commit `d64cf93`, an ancestor of HEAD): the physics loss "optimises through the
*same* frozen surrogate that scores the headline metric", and surrogate
validation "does not substitute for" real-device validity.

Consequences that are easy to get wrong and were got wrong:

- Meep spectra must never enter the loss or the target. They did not.
- **For a geometry whose spectrum the model must be conditioned on, use the
  surrogate, not Meep.** The dataset was built that way (`real`/`imag` are CST
  outputs of the released forward solver). Conditioning on a Meep spectrum forces
  a Meep→CST bridge whose error becomes a floor on every downstream number.
- Meep-vs-Meep comparisons are the only place Meep belongs, and there the
  convention cancels identically on both sides.

---

## 1. Audit of what I actually ran

Validity verdict per step. "Holds" means the number can be quoted.

| step | what | verdict |
|---|---|---|
| 1 | Meep harness passivity audit | **holds** — 36/43 designs violated \|T\|≤1 |
| 2 | Harness fix (loss + decay + gate) | **holds** — re-verified, 0 violations where claimed |
| 3 | Meep→CST convention refit | **holds** — 0.1138 LOO, but see §4.2 |
| 4 | Pipeline verification: surrogate(true G) vs true CST | **holds** — L1 0.0099, corr 0.9976 |
| 5 | v1 base retrain (70k, cluster A100) | **holds as a checkpoint; PRELIMINARY as a result** — single training seed |
| 6 | Task 1 IoU / style stats (v1) | **holds as preliminary** — single seed, scalar regime A only |
| 7 | Task 1 physics (v1) | **NOT YET RUN** — every attempt contaminated; see §3 |
| 8 | Task 2 decode (v1) | **holds as preliminary** |
| 9 | Task 2 physics (v1) | **NOT YET RUN** |
| 10 | Task 3 scalar sweep (JEPA) | **holds as preliminary** |
| 11 | Task 3 scalar sweep (MetaDiT) | **CONFUNDED** — MetaDiT got a surrogate target, ours got a Meep→CST target. Not comparable at face value |
| 12 | Degeneracy check | **holds** — no model involved |
| 13 | Style stats vs real corpus | **holds** — no model involved |
| 14 | Ledger write-up | **STALE** — records v2; v1/Phase 2/Task 3 unwritten |

### Errors that produced wrong numbers

Recorded so they are not repeated:

1. **Wrong checkpoint used throughout** — every Task 1/2 result until today used
   the `sched-fix` **v2-schedule** checkpoint, whose gate FAILED (0.0531 vs 0.0490)
   and which gives `lambda_phys` only 50% of the gradient budget (93% in v1).
   Part of the reported "fragmentation" was this, not the decoder.
2. **Meep fed into the model's input path** for Task 1 — added an avoidable
   convention bridge (0.114 error floor) where the surrogate would have been the
   correct physics tool.
3. **Output-directory collision + resume logic** — two cluster jobs wrote
   `~/runs/p2` with identical per-worker filenames. The harness resume check saw
   the other job's entries as done and **skipped 24 of 38 designs while exiting 0**.
   Leftovers were **v2** data. Anything from that directory is void.
4. **Silently swallowed failures** — `run()` captured stderr without printing it,
   so a `FileNotFoundError` surfaced as an unexplained 0.0004 s "success".
5. **Ran diffusion on the operator's laptop CPU** after an explicit instruction
   not to. ~2.5 CPU-hours burned.
6. **Wrong symmetry group** — measured D4 where the generator enforces only
   mirror symmetry, producing a spurious "corpus is only 0.75 symmetric" reading
   when it is exactly 1.000.
7. **Wrong tensor axis** — `pattern` is stored `(64,64,N)`; indexing `pat[idx]`
   reads the wrong axis. Bit me three times.

---

## 2. The 11 mandated gates (§8) — coverage map

`architecture_v5.md` §8.1–8.3. "Covered" means a test exists in `tests/`;
"Run on v1" means executed against the current checkpoint.

| # | gate | covered by | run on v1? | status |
|---|---|---|---|---|
| 1 | Cross-recovery both directions | — (no `cross_recovery` in `tests/`) | ❌ | **GAP** |
| 2 | FiLM ablation (train w/o scalar-FiLM) | `test_scalar_predictor_film.py` (unit only) | ❌ | **PARTIAL — no trained ablation** |
| 3 | MetaDiT-init transfer | — | ❌ | **GAP** |
| 4 | Soft-vs-hard surrogate (STE) | `test_phase4_physics.py` | ❌ | **PARTIAL — unit only** |
| 5 | Per-scenario gates reported separately | `eval_scenarios.py` | ❌ | **GAP on v1** |
| 6 | VICReg grad attribution re-run | `test_grad_share.py` | ❌ | **GAP on v1** |
| 7 | Within-bucket shape probe vs +0.225 | `test_representation_calibration.py` | ❌ | **GAP on v1** |
| 8 | Goal-ignoring collapse: real/**null**/**shuffled**, **stratified** | `test_goal_mode_validation.py` (real/null only) | ❌ | **CRITICAL GAP** |
| 9 | Scalar-conditioning collapse, stratified | — | partial | **GAP** |
| 10 | Generative (output) diversity collapse | — | ❌ | **GAP** |
| 11 | Majority-class collapse in decoder | partial in eval | ❌ | **PARTIAL** |

**The single most important gap is gate 8.** The mandated comparison is
real/**shuffled** (not real/null) **stratified by masking regime**, gated on the
stratum where occupancy *and* scalars are both heavily masked. Existing tests
cover real-vs-null only, and no test stratifies by mask ratio. The whole point of
the redesign is the pure-inverse-design scenario; a pooled real/null gap says
almost nothing about it — and this project already has direct evidence of exactly
that failure (real-vs-null distance ≈1.4 while the predictor barely used the
spectrum).

---

## 3. Immediate blocker

`638208` (cluster) is re-running all four v1 decode sets into **per-set output
directories with a clean-slate delete** before each set, so resume cannot fire
against foreign data. Required checks on completion, in order:

1. `designs == expected` **and** names all carry that set's prefix — if any name
   from another set appears, the directory is contaminated again; discard.
2. Every design has `passivity_ok == true`; count the failures and name them.
3. `hit_cap` count — a capped design is not converged and must be excluded.
4. Only then compute any metric.

---

## 4. Tests to run

### 4.1 Preconditions — cheap, do these first

| test | what it proves | why it matters now |
|---|---|---|
| **P1 surrogate round-trip** | `surrogate(assemble(G_true)) ≈ S_true` on ≥200 real items, L1 ≤ 0.02 | Already passes (0.0099). Must pass on *every* future run before anything else is believed |
| **P2 decode-path health** | real geometry + its true spectrum → decode, IoU ≥ 0.7 full-mask | Already passes (0.757). Distinguishes "model bad" from "pipeline wrong" |
| **P3 passivity gate** | every Meep design \|T\|≤1.02 | Catches harness regressions before any comparison |
| **P4 artifact isolation** | assert per-set output dirs; assert name-prefix purity | Directly prevents error #3 |

### 4.2 The mandated gates, as runnable checks

**G8 — goal-ignoring collapse, stratified (CRITICAL).** For two strata —
(a) mask ratio ≤ 0.3 with scalars known, (b) ratio = 1.0 with all scalars unknown —
report real / null / **shuffled** physics-consistency. Gate: real must beat
**shuffled** in stratum (b). Use ≥3 evaluation seeds and report the paired
margin with a CI. *A real-vs-null gap is not evidence.*

**G10 — generative diversity.** Perturb the target spectrum ±1%, hold everything
else; measure the decoded-geometry change. A deterministic predictor can pass
every latent-variance check and still emit a constant design. Report mean
per-sample IoU between perturbed decodes (must be well below 1.0) and the
occupancy-fraction spread.

**G11 — majority-class collapse.** Predicted occupancy fraction must track ground
truth across samples (correlation, spread), not cluster near the mean. IoU/F1 on
the *occupied* class, not pixel accuracy.

**G9 — scalar conditioning, stratified.** Vary `(l,h,r)`; assert the geometry
moves in the *correct* direction (a surrogate-predicted target for the perturbed
geometry), not merely that it moves. Stratify as G8.

**G1 — cross-recovery.** Occupancy masked + scalars known → does occupancy beat
the dataset-frequency prior? Occupancy known + scalars masked → does scalar
regression beat the dataset mean? Both directions.

**G5 — per-scenario gates.** A/B/C reported separately, never pooled.

**G6/G7 — re-establish on v1.** VICReg gradient attribution and the shape probe.
§8.2 explicitly says these were measured on the old 384-dim encoder and *nothing
carries over automatically*.

### 4.3 Ablations that would settle open questions

| ablation | question it answers |
|---|---|
| **A1 λ_phys on/off at fixed steps** | Is the physics term doing the work, or the JEPA losses alone? Direct read on the v1-vs-v2 fragmentation question |
| **A2 symmetrise the decode** (done) | Already run: made fragmentation *worse* → structural, not symmetry |
| **A3 known vs unknown scalars** (done) | Regime A provably ignores scalars. Now: does regime B improve **physics**, not just IoU? Untested |
| **A4 STE soft vs hard surrogate input** | Gate 4 as a trained comparison, not a unit test |
| **A5 multi-seed (3 seeds, 70k)** | Required by the acceptance protocol before *any* result is established |
| **A6 MetaDiT multi-seed decode** | MetaDiT is stochastic; all our MetaDiT numbers are seed 0 |

### 4.4 Corrections owed to the ledger

1. Fragmentation was **partly a v2 artefact** (comps 3.50 → 1.26 on v1). Holes
   remain ~3× the corpus.
2. Every prior decode used **scalar regime A** (all three scalars unknown,
   provably ignored). This was not stated.
3. All earlier numbers are the **v2** checkpoint; label them or supersede them.
4. Phase 2's negative result stands on v1, but its Meep came from a contaminated
   run and must be redone.
5. Task 3's MetaDiT comparison is **confounded** and must not be quoted as a
   model-vs-model result until both sides get the same target path.

---

## 5. Standing rules

- **3 training seeds** before any phase-level result is "established"
  (`ACCEPTANCE_PROTOCOL.md`). Evaluation seeds quantify measurement noise on a
  *fixed* checkpoint and never substitute.
- Every number states its replication unit: `x ± y (N training seeds)` or
  `(N evaluation seeds; fixed checkpoint)`.
- Prohibited, because each has produced a wrong conclusion here:
  - reusing a checkpoint whose gate failed
  - assuming resume/output paths are isolated
  - a control that is weaker than the claim (real/null where shuffled is required)
  - pooling across masking strata or symmetry classes
  - treating surrogate-agreement as physical validity
