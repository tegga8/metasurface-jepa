# Unified JEPA — architecture roadmap (measure-first)

Sequenced plan for the 192-D unified JEPA, revised this cycle. **No architecture
change lands until the current architecture has been measured and recorded.** Every
phase follows the standard protocol below; each change is one commit with a
regression test (`AGENTS.md` rule 1). Heavy training runs happen on Kaggle/Colab
per `CLOUD_TRAINING.md` — the local machine is dev-only.

**Decisions locked this cycle**
- **Multi-target objective** — add target-side conditioning so the spectrum and
  scalars are *required*, not merely rewarded (see `TARGET_DESIGN.md`).
- **Include the spectrum in the target side** — via a *spectrum-conditioned geometry
  target* (the naive "predict the input spectrum's latent" target is shortcut-able;
  see `TARGET_DESIGN.md` §2).
- **Keep scalar FiLM on the geometry target.**

**Removed (was in the previous draft)**
- ~~Phase 2c — I-JEPA compliance / alignment doc~~ (dropped).
- ~~Phase 3c — geometry-only JEPA target~~ (dropped; superseded by the multi-target
  design, which keeps the stable spectrum-free target *and* adds conditioned targets).

---

## Standard phase protocol (applies to EVERY phase below)

1. **Baseline** — record the current numbers *before* touching code: benchmark
   table + A/B/C gates + guidance gap + masked-fill + full test suite.
2. **Change** — one behaviour change + a regression test that **fails before** and
   **passes after** (`AGENTS.md` rule 1).
3. **Measure** — re-run the **same** battery, same split / config / seed.
4. **Record** — append before/after + delta to `RESULTS.md`.
5. **Gate** — proceed only if the phase gate is met (or no regression). Otherwise
   **STOP and report** — do not loosen a gate to pass it (`AGENTS.md` §"If
   something fails").

**Canonical battery** (run at every step 1 and step 3):
```
python -m pytest tests/ -q --tb=line                                  # tests green
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
    --samples 0 --candidates 4                                        # MAE/AAE/AAE&K
python scripts/eval/eval_scenarios.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt --scenario all         # A/B/C + gates
python scripts/diagnostics/masked_fill_check.py --config configs/unified.yaml \
    --checkpoint checkpoints/unified/latest.pt                        # seam/texture/locality
```
Reported **separately, never pooled** (easy / hard strata; scenarios A / B / C).

---

## Phase 0 — Benchmarking harness  ·  status: implemented, run pending
MetaDiT-comparable MAE/AAE/AAE&K + secondary suite + masked-fill diagnostic.
Files: `docs/benchmarking/` (this folder), `scripts/benchmark/`,
`scripts/diagnostics/masked_fill_check.py`, `tests/test_benchmark_metadit_metrics.py`,
`tests/test_masked_fill_check.py` (all tests pass locally; cloud run pending).

## Phase 1 — Current-architecture baseline  ·  **the gate to everything else**
Goal: answer *"is the current architecture working well?"* with recorded numbers,
before changing anything. No code change.

- Run the canonical battery on the **real checkpoint** (Kaggle; `checkpoints/unified/latest.pt`).
- Record in `BASELINE.md` (protocol + results table) and open `RESULTS.md`.
- Known expectations from `checkpoints/unified/REPORT.md` to confirm/replace:
  A/B/C gates pass; **scalar-dependence gates FAIL (~0.50)**; physics ≈ **66 %** of
  the gradient budget; guidance gap small; masked-fill seam/locality **unmeasured**.
- **Gate to proceed:** baseline recorded, and the failing aspects explicitly listed.
  Every later phase is judged against these numbers.

## Phase 2 — Representation-first training schedule (issues 5, 6, 7)
Let the representation form before physics is heavy.
- **2a — instrument (no behaviour change).** Log per-term gradient share live during
  training (today the 66 % is post-hoc only). Pattern: `protocol_v1/step3_4_...:175-184`.
- **2b — mask-ratio curriculum.** Draw the ratio **per sample** (not per batch);
  config-driven ramp (start low, raise total-masking probability over training);
  raise P(ratio=1.0) above 0.15. Config: `curriculum.mask_schedule`.
- **2c — physics ramp.** `staging.lambda_phys_start_step` + longer ramp; re-run the
  λ sweep (`REPORT.md` §11) targeting a **deliberate** share (~10–25 %).
- **Gate:** Scenario-A MAE/AAE holds or improves; hard-stratum gate holds; logged
  gradient share in the target band.

## Phase 3 — Representation hygiene (issues 2, 8)
- **3a — remove the redundant pixel mask.** `src/assembly.py:280` masks raw pixels
  before patch-embed *and* `occupancy_encoder.py:81-83` replaces masked tokens with
  a learned `mask_token`. Keep the token-level path; drop `apply_mask_to_pixels` on
  the student forward (predictor-query construction untouched).
- **3b — projector ablation (issue 8).** Latent loss is on `P(ẑ) vs P(z_y)` with a
  **BatchNorm** MLP projector (`src/losses/vicreg.py:153`). Ablate
  {none, linear, MLP, MLP+BN} on the hard-stratum gate; cross-check `validate()`'s
  raw-space `raw_mse / raw_cos_err`.
- **Gate:** Scenario-A MAE/AAE + hard-stratum gate.

## Phase 4 — Multi-target objective: spectrum + scalar in the target side (issues 4, 11)
Full design: **`TARGET_DESIGN.md`** (authority).
- **4a — spectrum-conditioned geometry target** (`z_y_occ_spec`): EMA occupancy
  encoder conditioned on the TRUE spectrum. Non-shortcut-able, and the only way to
  match it is to use the goal spectrum.
- **4b — scalar-latent target** (`z_y_scal`): EMA scalar encoder summary of the true
  scalars — door (b) after the door-(a) readout (`lambda_summary`).
- **Keep** the stable, spectrum-free geometry target (`z_y_occ_stable`) for the
  unentangled real/null/shuffled control; **keep** scalar FiLM on the geometry target.
- **Gate:** scalar-dependence win rate moves off ~0.5 toward 0.94–0.97; hard-stratum
  guidance gap rises; **target-dependence (no-shortcut) probe passes**; MAE/AAE no regress.

## Phase 5 — Scalar capacity & bounds (issues 3, 9)
- **5a — bound decoded scalars.** `ScalarDecoder` emits raw unbounded values
  (`scalar_decoder.py:31-54`); parameterise into verified ranges
  (l∈[2.5,3.0], h∈[0.5,1.0], r∈[3.5,5.0]; `datapipe.py:60-64`), config-sourced,
  applied on the decode→assembly path; keep raw values in the loss.
- **5b — scalar tokens & routing (issue 9).** L_scalar gives **0.0036** gradient to
  the scalar encoder vs **0.6932** to the decoder (`REPORT.md` §17/§19/§21). Increase
  scalar **tokens** (per-parameter + summary) and route them through the predictor.
- **Gate:** scalar-dependence gates; MAE/AAE no regress.

## Phase 6 — Encoder sizing (issue 1)
- Per-module param/FLOP audit (now 18.9M total / 11.37M trainable; geo depth 6,
  predictor depth 8, hidden 192 — `ARCHITECTURE_AUDIT_192D.md` §1.1), then a
  width×depth grid scored on the benchmark + hard stratum. Decide from **measured
  deltas**, not intuition (`AGENTS.md` rule 2).

---

## Phase→verification contract
- Each fix: regression test that fails before and passes after.
- End of each phase: full suite green **and** the canonical battery re-run, recorded
  in `RESULTS.md`, easy/hard separate, never pooled.
- Data-dependent tests skip loudly with a reason, never pass silently.
