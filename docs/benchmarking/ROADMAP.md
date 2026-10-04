# Unified JEPA — architecture roadmap (measure-first)

Sequenced plan for the 192-D unified JEPA. **No architecture change lands until the
current architecture is measured and recorded.** Every phase follows the standard
protocol; each change is one commit with a regression test (`AGENTS.md` rule 1).
Heavy training runs happen on Kaggle/Colab per `CLOUD_TRAINING.md` — local is dev-only.

**Decisions locked**
- Multi-target objective (`TARGET_DESIGN.md`); include the spectrum in the target side
  via a *spectrum-conditioned geometry target*; keep scalar FiLM on the geometry target.
- **Removed:** I-JEPA-compliance doc; geometry-only target (superseded by multi-target).

## Status board

| item | status |
|---|---|
| Phase 0 — benchmarking harness | **DONE** |
| Phase 1 — current-architecture baseline (Kaggle) | **DONE**, but the NN/AVG1 rows must be **regenerated** (pre-fix driver) |
| Review P0 hygiene (A1, B1, A2, A4, A5, B2, C1, C2, C6, C3, A3, A6) | **DONE** (suite: 330 passed / 0 failed) |
| Phase 2 — representation-first training schedule (2a/2b/2c) | **DONE** — gate MET at 10k (Scenario C 0.736→0.779; MAE −5.4 %); see `RESULTS.md` |
| Branch divergence (review C4 / door (b)) | **DECISION: do not merge here** — see Step 0 |
| Step 1 baseline regeneration | **DONE** with the Phase-2 kernel (NN/AVG1 rows regenerated: NN pool 20k = 0.0296) |
| Phase 3 — representation hygiene (3a/3b) | **DONE** — 3a behaviour-neutral; 3b projector ablation keeps `mlp_bn` (best MAE/gate); see `RESULTS.md` |
| Phase 4 — multi-target objective | **UNESTABLISHED under CIs** — the single-seed wins (MAE 0.0749/0.0725) did not survive 3 seeds; Scenario C's "pass" was single-seed |
| Phases 5–7 below | **TODO** — reframed by the NN-scoping probe: **beat NN is the bar** (NN wins all novelty quartiles; Scenario C is a tie). Phase 5–7 become retrieval-competitive |

## Full-epoch (70k) baseline — CURRENT config, 3 training seeds  ·  DONE
- MAE **0.0490 ± 0.0013** (3 training seeds) [CI 0.0459–0.0521]; AAE 29.51; gates
  A **0.9967**, B **0.9772**, **C 0.9303** (CI 0.903–0.958 — robustly above 0.75).
- **70k materially beats the 10k 3-seed baseline** (MAE 0.0816→0.0490; C 0.748→0.930).
- **Scalar dependence unchanged (~0.505 / 0.529)** — the surviving weakness.
- Kernel `metasurface-jepa-fe-current`, commit `c35196f`. See `RESULTS.md`.

## Scalar conditioning — predictor-FiLM intervention at 70k × 3 seeds  ·  NEGATIVE
- `scalar_predictor_film=true` vs baseline: MAE 0.0490 → **0.0506** (slightly worse,
  within noise), A/B/C ~unchanged, **scalar one/two 0.505/0.529 → 0.525/0.555**
  (still ~chance). **STOP the scalar architecture line** per the plan's stop condition;
  reassess the task/objective rather than stacking modules. See `RESULTS.md`.

## Scenario C — crossed seed study (3 training ckpts × 3 eval seeds): UNRESOLVED
- Checkpoint means (3 eval seeds each): seed0 0.7480±0.0039, seed1 0.7839±0.0063,
  seed2 0.7572±0.0205. **Overall (replication unit = checkpoint): 0.7630 ± 0.0186
  (3 training seeds); 95 % t-CI [0.7168, 0.8093] → straddles 0.75.**
- Failure population mostly checkpoint-specific: only 3.5 % of items fail for all 3
  checkpoints → the ~16.6 % failing at all is largely training-seed-specific, not a
  common structural set. See `RESULTS.md`.

## Multi-seed CI (3 × 10k) — the phase gains did NOT survive
- MAE **0.0816 ± 0.0025** (seeds 0.0800/0.0803/0.0845) — the single-seed Phase-2 (0.0749)
  and Phase-4 (0.0725) runs were **lucky draws**; the honest 10k MAE is at the control's
  level (0.0792) → **no established improvement from Phases 2/3/4**.
- Scenario A/B robust (0.990 / 0.926); **Scenario C NOT robust** (0.748 ± 0.037 < 0.75,
  one seed fails); scalar dependence still fails (0.614 / 0.557).
- Seed std ≈ 2.5 % of the mean — the same order as the effects being chased.
- **Retraction:** Phase-2/4 gains are unestablished; the robust results are A/B gates,
  the NN loss, and the scalar failure.

## Standard phase protocol (every phase)

See [`ACCEPTANCE_PROTOCOL.md`](./ACCEPTANCE_PROTOCOL.md): no model/training result is
**established** without ≥3 training seeds; evaluation seeds are separate and quantify
measurement noise on a fixed checkpoint; every number states its replication unit.

1. **Baseline** — record current numbers before touching code.
2. **Change** — one behaviour change + a regression test that fails before / passes after.
3. **Measure** — same battery, same split / config / seed.
4. **Record** — append before/after + delta to `RESULTS.md`.
5. **Gate** — proceed only if the gate is met (or no regression); else **STOP and report**.

**Canonical battery** (run at step 1 and step 3):
```
python -m pytest tests/ -q --tb=line
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
    --samples 0 --candidates 4 --nn-samples 512 --nn-pools 512,5000,20000 \
    --out checkpoints/benchmark/scenarioA.json
python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --scenario all --samples 512 \
    --out checkpoints/benchmark/eval_scenarios.json
python scripts/diagnostics/masked_fill_check.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --samples 32 \
    --out checkpoints/benchmark/masked_fill.json
python scripts/diagnostics/run_guidance_gap_sweep.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt
```
Reported **separately, never pooled** (easy/hard strata; scenarios A/B/C). Every row
carries its **n** and provenance. Co-primary with MAE/AAE: occupancy IoU/F1, seam/
locality, gate win rates — metrics that are **not** the training objective (review A6).

---

## Next steps (in order)

### Step 0 — Branch divergence (review C4) · DECISION: do not merge here
The door-(b)/preflight line is **not in this clone**, and door (b) is an
**unvalidated** remedy. door (b) = route the scalar signal through the predictor
(`scalar_predictor_film`), because door (a) showed the read-out works but the scalar
gates still sit below the 0.75 bar (`checkpoints/unified/REPORT.md` §21.3). Decision:
**do not promote it to the mainline by merging.** Adopt the **provenance fallback** —
every doc and kernel pin states the commit + config that produced each number, and
door (b) stays an ablation branch. When door (b) is validated and merged by the
operator, fix **C5** (the `scalar_predictor_film` construction vs its comment) and
**B2-local** (the `checks[...]`-before-`checks` `NameError` + the missing empty-sample
guard) on that branch.

### Step 1 — Regenerate the baseline NN/AVG1 rows · DONE
The Phase-2 kernels ran the fixed driver, so the rows are now correct: AVG1 uses the
train mean; **NN is scored like-for-like** — pool-20000 MAE **0.0296** vs ours 0.0707
at the same 512 test items (NN beats us ~2.4×; pool curve 512→0.0551, 5000→0.0377,
20000→0.0296). `RESULTS.md` updated.

### Phase 2 — Representation-first training schedule (issues 5, 6, 7) · DONE
- **2a instrument — DONE** (`3064c70`): the objective exposes per-term weighted losses;
  `per_term_grad_share` logs the budget share every `train.log_grad_share_every_steps`.
- **2b mask curriculum — DONE** (`ee23a24`): per-sample ratios, a distribution ramp
  (`curriculum.mask_schedule`), and P(full mask) raised 0.15 → 0.35.
- **2c physics ramp — DONE** (`ee23a24`): `staging.lambda_phys_start_step = 2000`,
  ramp 3000.
- **Gate MET at 10k** (control vs Phase-2, `RESULTS.md`): MAE −5.4 %, Scenario C
  0.736 → 0.779 (crosses the 0.75 gate), scalar dependence +4–7 pp. Open follow-up: a
  λ re-sweep to land the gradient share at ~10–25 % — the probe now measures it live.

### Phase 3 — Representation hygiene (issues 2, 8) · DONE
- **3a — DONE** (`b811da6`): removed the redundant pre-patch-embed pixel mask; masking
  is now token-level only. Behaviour-neutral (masked-pixel-invariance test).
- **3b — DONE** (`b811da6`): projector is ablable (`loss.projector_type`); the
  ablation keeps **`mlp_bn`** (best MAE/AAE and Scenario-A gate; the non-BN arms win
  only on Scenario C). See `RESULTS.md`.

### Phase 4 — Multi-target objective (issues 4, 11) · ATTEMPTED — FAILED
Implemented (commit `72195e6`): a trainable `SpectrumFilm` (c_physics → per-block
FiLM), shared by the student occupancy encoder and — via an EMA copy — the
spectrum-conditioned geometry target `z_y_occ_spec`; the scalar-latent target
`z_y_scal`; `z_hat_occ_spec` (a `spec_proj` head off the predictor trunk) and
`z_hat_scal`; losses `L_cond` / `L_scal_t`. The stable spectrum-free target is kept.
**Deviation from `TARGET_DESIGN.md`:** the spectrum FiLM is shared with the *student*
encoder (not target-only), because a target-only conditioning module receives no
gradient and its zero-init keeps the target spectrum-free forever.
- **Result: FAILED.** `L_cond` took **91–99 % of the gradient budget** (an
  un-normalized masked MSE through a random head, degenerate with the stable target
  at init), collapsing the representation and the occupancy decoder; every gate
  failed. See `RESULTS.md`.
- **Fix applied** (`904856e`, per `PHASE4_DIAGNOSIS.md`): cosine (scale-free) losses,
  ramped in from step 1000, and a **FROZEN non-identity** spectrum film
  (non-nullifiable). The corrected 10k run **does not collapse and does not regress**
  (MAE 0.0789 vs 0.0792; A/B/C 0.988/0.934/0.771 — C crosses the gate; scalar
  dependence +2–6 pp) — **but** the multi-target terms are now only **~0.1 %** of the
  gradient (inert; over-corrected) and the hard-stratum guidance gap **fell**
  (~41→27), so the forcing is **not demonstrated**. Status: **PARTIAL**.
- **Resolved** (`979f088`): before retuning the weights, the target-separation probe
  (`scripts/diagnostics/spectrum_film_separation.py`) showed the conditioning was
  near-redundant (~0.037 cross-spectrum cosine distance at the shipped init), so the
  film init was strengthened **0.02 → 0.1** (not the loss weight) → **MAE 0.0725 (best
  10k), guidance gap rose 41→52, A/B/C 0.994/0.947/0.779** — the mechanism works.
  Scalar dependence improves (0.650/0.596) but is still short of 0.94–0.97 → partial.
  See `RESULTS.md`.

### Phase 5 — Scalar capacity & bounds (issues 3, 9)
- **5a** bound decoded scalars to verified ranges (config-sourced).
- **5b** more scalar tokens + predictor routing (L_scalar gives the encoder 0.0036 vs
  the decoder 0.6932 — `REPORT.md` §17/§19/§21).
- **Gate:** scalar-dependence gates; MAE/AAE no regress.

### Phase 6 — Encoder sizing (issue 1)
Per-module param/FLOP audit + width×depth grid scored on the benchmark and the hard
stratum; decide from **measured deltas** (`AGENTS.md` rule 2).

### Phase 7 — Evidence & publication-readiness (review P1) · the "is it real?" phase
- **7a — ≥3 seeds** (cheap at the 10k-step λ-sweep scale) → median/IQR + bootstrap CI;
  the headline becomes a **paired statement** (win rate + median), not a mean (B3).
- **7b — ≥2 epochs** with the canonical battery recorded per checkpoint — answers the
  1-epoch-vs-500 budget question (B5); either the gap holds or it closes.
- **7c — full-wave / second-surrogate validation** on ~32 stratified designs incl. tail
  cases (B4), plus an OOD guard for the catastrophic tail.
- **7d — circularity**: promote a **non-objective** metric (IoU/F1, seam, win rate) to
  co-primary and state the "physics loss optimises the scorer" asymmetry in the text (A6).
- **7e — diversity curve**: run the σ/CFG grid [0, 0.01, 0.05, 0.10] so the deterministic
  vs one-to-many trade-off is shown, not asserted. Guidance-gap stays a diagnostic.
- **Gate:** beat NN retrieval on the hard stratum; scalar gates pass; results stable
  across seeds/epochs; at least one non-surrogate check agrees.

---

## Phase→verification contract
- Each fix: regression test that fails before and passes after.
- End of each phase: full suite green **and** the canonical battery re-run, recorded in
  `RESULTS.md`, easy/hard separate, never pooled, with n and provenance on every row.
- Data-dependent tests skip loudly with a reason, never pass silently.
