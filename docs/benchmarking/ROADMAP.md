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
| Review P0 hygiene (A1, B1, A2, A4, A5, B2, C1, C2, C6, C3, A3, A6) | **DONE** (suite: 328 passed / 0 failed) |
| Branch merge (review C4) | **PENDING** — door-(b)/preflight line is not in this clone; C5 + B2-local fixed at merge |
| Phases 2–7 below | **TODO** |

## Standard phase protocol (every phase)

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

### Step 0 — Merge the two lines (review C4) · small
Pull the door-(b)/preflight line into `work-192d`; at the merge fix **C5** (the
`scalar_predictor_film` construction vs its comment) and **B2-local** (the
`checks[...]`-before-`checks` `NameError` plus the missing empty-sample guard). Re-run
`--preflight` and the full suite on the merged branch. Everything below assumes one branch.

### Step 1 — Regenerate the Phase-1 baseline block · cheap cloud run
Re-run the canonical battery with the **fixed** driver (review A2/A4/A5) and overwrite
the `RESULTS.md` baseline NN/AVG1 rows, adding the NN pool curve. MAE/AAE/AAE&K and the
A/B/C gates are unchanged by these fixes. Record in `RESULTS.md`.

### Phase 2 — Representation-first training schedule (issues 5, 6, 7)
- **2a instrument** (no behaviour change): log per-term gradient share live
  (`scripts/diagnostics/protocol_v1/step3_4_...:175-184`).
- **2b mask-ratio curriculum**: draw the ratio **per sample**; config-driven ramp;
  raise P(ratio=1.0) above 0.15. Config: `curriculum.mask_schedule`.
- **2c physics ramp**: `staging.lambda_phys_start_step` + longer ramp; re-sweep λ
  targeting ~10–25 % gradient share (not 66 %).
- **Gate:** Scenario-A MAE/AAE holds/improves; hard-stratum gate holds; share in band.

### Phase 3 — Representation hygiene (issues 2, 8)
- **3a** remove the redundant pixel mask (`assembly.py:280`); keep token masking.
- **3b** projector ablation {none, linear, MLP, MLP+BN} on the hard stratum.
- **Gate:** Scenario-A MAE/AAE + hard-stratum gate.

### Phase 4 — Multi-target objective — spectrum + scalar in the target side (issues 4, 11)
`TARGET_DESIGN.md` is the authority. **4a** spectrum-conditioned geometry target
(`z_y_occ_spec`); **4b** scalar-latent target (`z_y_scal`); **keep** the stable
spectrum-free target and scalar FiLM on the geometry target.
- **Gate:** scalar-dependence win rate moves off ~0.5 toward 0.94–0.97; hard-stratum
  guidance gap rises; **no-shortcut probe passes**; MAE/AAE no regress.

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
