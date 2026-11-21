# Independent review — unified 192-D JEPA for metasurface inverse design

**Reviewer scope:** the project as of remote `tegga8/metasurface-jepa` `work-192d` @ `ad78d54` (the
benchmarking/publishability line) plus the local branch @ `1478026` (the door-(b)/preflight line),
and its own claims as recorded in `docs/publishability_assessment.md`, `docs/comparison/`,
`docs/benchmarking/`, `checkpoints/unified/REPORT.md`, and the cloud baseline run.

**Date:** 2026-10-02 · **Review type:** adversarial validation for research-readiness
(claims checked against code, records, and executable probes — not a literature review).

**What was run (not just read):**

| check | result |
|---|---|
| Full local suite (door-b branch, Windows Py 3.14 + torch 2.14 CPU) | **300 passed / 23 skipped / 0 failed** |
| Full remote-snapshot suite (same env, data absent) | **311 passed / 24 skipped / 2 failed** — 337 collected, same collection as the cloud run's 328+8+1; the 2 failures are my env missing `tensorboard` (the reference-import parity tests); the cloud failure is different and is finding A1 |
| Parameter counts, live-built | MetaDiT-S `37,197,644` (incl. its internal spectrum encoder `4,526,144` → `32,671,500` without it); surrogate `6,328,698`; ours `18,900,100` total / `11,365,956` trainable — matches the comparison doc exactly |
| `objective.physics_loss` exists? | **No → `AttributeError`** (proven by executing the actual test against a structurally-valid surrogate) |
| Determinism (same input twice, eval, no_grad) | `z_hat` and `scalar_pred` bit-identical ✓ |
| B=1 forward; `make_shuffled_spectrum` at B=1 | forward OK; shuffled silently returns the input (documented, guard-required) |
| Benchmark harness smoke (untrained model, synthetic) | runs end-to-end, prints the full table ✓ |
| Released-component data provenance (`external/metadit/scripts/*`) | surrogate/encoder/DiT training scripts all use `split_data/train_set.mat` (+ val) — **test split not used** |

**TL;DR.** The publishability assessment is fundamentally honest and its two headline blockers are
real. This review **confirms both, sharpens one, and resolves one**: the NN gap is measured
unfairly *in both directions* and must be recomputed like-for-like; the "surrogate saw the test
split" leakage fear is **substantially defused by the released training scripts** (train/val only),
leaving a reward-hacking/circularity concern rather than a data-leak one. Beyond that I found
**six additional issues** the project's own documents do not record — the most important being that
the **only automated guard of the physics-gradient claim has been broken for this entire line of
work** (skipped locally, failing in every environment that can run it). None of the findings is
fatal to the direction; all are fixable; most are small. Verdict unchanged from the assessment:
**workshop-ready after the P0 fixes below; not yet a top-tier paper.**

---

## 1. Findings

### A. Result-integrity findings (these touch the numbers in the tables)

**A1 — The physics-gradient regression test is broken everywhere, and the failure was
mis-recorded as "pre-existing & unrelated" (VERIFIED).**
`tests/test_phase5_contracts.py:330` is the Phase-5-MD-§6 guard for the project's headline
mechanism ("gradients flow from the surrogate through the geometry to the student"). It calls
`objective.physics_loss.enable()` (line 350). `UnifiedJEPALoss` has **no such attribute**
(`src/losses/unified_losses.py` exposes `self.surrogate` only). I proved it by building a
structurally-valid surrogate checkpoint in a sandbox copy and running the test:

```
AttributeError: 'UnifiedJEPALoss' object has no attribute 'physics_loss'
```

- Locally (no surrogate staged) the test **skips**, so it has never been green locally either.
- On the Kaggle baseline run it **failed** — recorded in `BASELINE.md` §2 as "the 1 failure is
  pre-existing & unrelated". It is not unrelated: it is *the* guard for the claim that STE physics
  reaches the student, and it is *inert in every environment*: skipped where weights aren't
  staged, broken where they are.
- Fix (small): rewrite the test against the current API (`objective.surrogate = surrogate`; the
  loss activates when `lambda_phys>0 and surrogate is not None and training and goal_mode!='null'`),
  and make it runnable with a *stub* surrogate so CI exercises it without the released weights.

**A2 — The NN-retrieval comparison is not like-for-like — in two ways, one of which makes the
reported gap *understate* the problem (VERIFIED in code).**
`scripts/benchmark/benchmark_metadit.py:206-252`: the NN arm retrieves over
`MetaDiTDataset(train, max_samples=512, seed=0)` — i.e. **512 randomly chosen designs out of
139,906** — and is evaluated on the **first 512 test items** (`all_spec[:512]`), while the model's
number is the **full 17,489-item** test mean. The project's own `docs/comparison/overfitting.md`
shows the model's mean is n-unstable: **0.0985 @ n=1024 → 0.0824 @ n=2048 → 0.0673 @ full**.
Consequences:
- The table row "NN 0.0551 vs ours 0.0673" compares a *handicapped* baseline (1/273 of the pool)
  at n=512 against our full-split number. A full-pool NN can only be *better* than 0.0551, and
  our n=512-equivalent number is likely ~0.10 — so **the true gap is probably worse than
  stated**, but the stated numbers are not defensible as-is.
- Fix: one command at equal n (e.g. n=512 for both, or full split for both) + a pool-size curve
  (512 / 5k / 20k / full) for the NN arm. Until then, the table row must carry "n=512 vs n=17,489,
  pool=512" footnotes.

**A3 — "Predicts all parameters" is conflated with "conditioning on parameters" (VERIFIED).**
Two different measurements are being merged in the assessment:
- *Scalar prediction accuracy* from the spectrum: **scalar MAE 0.088** (comparison doc §5) —
  this is measured and it is fine; it supports "the model predicts the parameters".
- *Scalar conditioning use* (given a known scalar, does the output change correctly): **at
  chance** (0.506 / 0.451). This is the failing gate.
The honest statement: *"the model predicts scalars accurately from the spectrum, but does not
use user-supplied scalars; therefore scenario-B/C-style partial **parameter** conditioning is not
demonstrated (partial **occupancy** conditioning is)."* The current phrasing "predicts all
parameters is false" over-claims the failure, and "flexible partial-observation conditioning"
over-claims the capability — both need the split.

**A4 — The AVG1 baseline is transductive and not the paper's baseline (VERIFIED).**
`benchmark_metadit.py:201`: `bl.avg1_metrics(all_spec, bl.mean_spectrum(all_spec))` — the mean
spectrum is computed **from the evaluated split itself** (test-set statistics). The standard
AVG1 baseline predicts the **train** mean. The docs note the paper's 0.5860 is "not the same
predictor" but print both in one table (0.2574 vs 0.5860, a 2.3× gap needing explanation).
Fix: compute the mean from `train_set.mat` (one line); if the paper's number still differs,
investigate before keeping both rows.

**A5 — Several diagnostic numbers are single-batch / synthetic but presented as split-level
(VARIOUS).**
- *Surrogate floor* (`0.0065`): computed on the **first batch** of the eval loop (≤8 items;
  `benchmark_metadit.py:203-205`). Cheap to compute over all items (the NN arm already renders
  true-geometry spectra).
- *Guidance gap* (`185.28`): from `run_guidance_gap_sweep.py:73-77` — **2 synthetic samples**
  (fixed seed, random occ/spectrum). It is a model-behavior probe, not a split statistic; label
  it, or run it on real val items.
- *Masked-fill* numbers: n=32, and by the project's own note re-run on CPU after the kernel crash
  — fine, but "n=32" belongs in the table row, not only in the prose.

**A6 — Reward-hacking / metric-circularity remains the real methodological risk (ANALYSIS).**
The physics loss optimizes **through the same frozen surrogate** used to compute the headline
metric. MetaDiT's number is scored by that surrogate too, so the *comparison* is symmetric — but
our model *trains against the scorer*, which MetaDiT never does. The failure mode is exploiting
the surrogate's blind spots (the soft-occupancy path already demonstrated the surrogate has huge
OOD regions: 96% off-distribution, measured). Recommendations: (i) report at least one metric
that is *not* the training objective (occupancy IoU/F1, seam metrics, win rates — already
computed; use them as co-primary); (ii) full-wave (or second-surrogate) validation on N≈32
designs; (iii) state the asymmetry explicitly in the paper.

### B. Verification gaps (what stops this being a "proper research project")

**B1 — Parts of the test suite do not test what their names claim (VERIFIED, partial list).**
- `tests/test_phase5_contracts.py::test_masking_does_not_modify_scalars` — compares `sv` to a
  copy of `sv`; **no masking call occurs**. Same for `test_scalar_masking_does_not_modify_occupancy`
  (asserts `occ == occ_copy` with nothing modifying `occ`).
- `test_scalar_flags_match_regime` — constructs a tensor and asserts its own shape/dtype.
- `test_evaluator_flags_all_empty` / `_all_occupied` — assert `0.0 < 0.01` and `1.0 > 0.99` on
  hand-made constants; the comment says "the collapse check would flag this" but no check runs.
- `test_target_uses_scalar_mlp_ema` — compares `out["z_y_raw"]` with `out2["z_y"]` (an alias of
  the same tensor in the same dict) under an unchanged input; it cannot detect a live-vs-EMA mixup.
- `test_contract_surrogate_spectrum_shape` — never runs a surrogate; asserts a geometry shape.
The suite count (337) is real, but these entries inflate apparent protection. Replace with
behavioral versions (e.g. mutate the live encoder and show `z_y` unchanged vs a shuffled EMA
target; call the actual mask functions and compare outputs).

**B2 — The preflight scalar-precedence block is broken on **both** branches (VERIFIED, two
distinct bugs).**
- Remote (`ad78d54`, `train_unified.py:1324-1327`): the h/r check reads the pixel index from the
  **true** occupancy while reading the value from the **decoded** geometry — on any model whose
  decoded occupancy misses that particular true-occupied pixel, assembly is *falsely* reported in
  violation (this is exactly what crashed the first door-(b) run locally:
  `"known-scalar precedence violated for h (sample 0: got 0.0, expected 0.85)"`). Also
  `assert occ_idx.shape[0] >= b` does not guarantee one pixel *per sample*; a legitimately
  empty true pattern → `IndexError` in the loop.
- Local (`1478026`): fixed the *source* of the index (decoded occupancy — correct) but writes
  `checks["precedence_unverifiable_samples"]` **~50 lines before `checks` exists** → `NameError`
  on the empty-decoded path, and the unknown-case loop lacks the same empty-sample guard
  (`IndexError` once the NameError is fixed).
- Correct fix: build `checks = {...}` before the block; maintain a per-sample `empty` list from
  the decoded support; skip empty samples in **both** loops with a recorded note (never assert on
  them); keep the local version's decoded-occupancy index source.

**B3 — Single seed, no uncertainty on the headline numbers (RECORDED GAP).**
The win rates are paired statistics with ~17k samples (good), but MAE/AAE are single-checkpoint;
no CI/seed variance. The assessment lists this; nothing in the ROADMAP phases adds seeds. Add a
phase: ≥3 seeds (cheap at the 10k-step scale used for the λ sweep), report median/IQR + bootstrap
CI, and make the paper's headline a paired statement (win rate + median), not a mean.

**B4 — No independent physical validation (RECORDED GAP).** Everything is surrogate-scored.
Full-wave on a small stratified sample (32 designs incl. tail cases) is the single highest-value
addition for credibility.

**B5 — Budget asymmetry unanswered (RECORDED GAP).** 1 epoch / 1×T4 vs 500 epochs / 4×A100.
The strongest answer is empirical: keep training (2–4 epochs) and show the same battery at each
checkpoint — either the gap holds (great paper) or it closes (still a solid workshop result).

### C. Reproducibility & engineering

**C1 — The recorded "artifacts" are stdout captures, not machine-readable results (VERIFIED).**
`kaggle/baseline_eval/baseline_eval.py` writes each battery command's **stdout text** to
`baseline_<name>.json`; the benchmark command does **not** pass `--out` (which would emit real
JSON). Some captures are parseable JSON (eval_scenarios prints JSON), some are formatted tables.
Also none of these artifacts are committed to the repo (`checkpoints/**/*.json` is gitignored).
Fix: pass `--out` for every battery command, keep the small JSONs (or paste key blocks into
`RESULTS.md`), and add a sync-back step for them.

**C2 — `--scenario` is parsed and ignored (VERIFIED, still present in remote).**
`eval_scenarios.py:751` defines it; `:769` calls `run_all_scenarios(...)` without it. Harmless in
the kernel (which passes `all`), misleading for anyone else. Either wire it or drop the flag.

**C3 — Parameter accounting has a double-count (VERIFIED, arithmetic).**
`metadit_vs_unified_jepa.md` §2 says "37.20 M DiT (+4.53 M spectrum encoder)" and an ecosystem
"MetaDiT ≈ 48 M vs ours ≈ 30 M". The 37.20 M **already includes** the 4.53 M encoder (measured:
37,197,644 total incl. 4,526,144 encoder). Correct ecosystems: **43.5 M vs 25.2 M**; the
assessment's "32.57 M" is the encoder-excluded DiT (measured 32,671,500 — reconcile the small
paper-vs-live delta). The headline "3.3× smaller trainable, ~half the total" survives untouched.

**C4 — Two divergent lines of work, neither containing the other (VERIFIED).**
Remote `work-192d` has the benchmarking line; local `work-192d` has door (b) + the preflight fix
(2 commits, unpushed); 8 remote commits are unpulled. Both preflight bugs (B2) exist *because* of
this split. Merge (or rebase) early — the publishability docs describe a config that does not
contain door (b), while the local config (the one the next run will use) does; the comparison
tables must state which commit/config produced them (they do via kernel pins — keep that).

**C5 — `scalar_predictor_film` checkpoint-interchange comment is false (local branch, VERIFIED).**
`src/assembly.py:152-154` claims the door-(b) projection is "constructed either way (zero-init)
so checkpoints stay interchangeable"; `src/predictor/gclct.py:131-136` constructs it only when
enabled → toggling the flag breaks strict resume. Fix either the comment or the construction.

**C6 — Smoke-mode dummy weights can contaminate real runs (VERIFIED, both branches).**
`train_unified.py:174-187`: in explicit smoke mode a **random-init spectrum encoder is written to
the configured released-weights path**, and real mode later accepts it on a bare
`os.path.exists`. For a project whose central object is a frozen released encoder, this is a
foot-gun. Fix: write dummies to a clearly separate path (or temp dir) and record a hash/marker.

### D. Disposition of the assessment's own threat list

1. **NN beats us** — confirmed; the equal-n re-measurement (A2) will likely make it *worse*, so
   treat it as the top must-fix, exactly as the assessment says.
2. **Scalar dependence fails** — confirmed; plus the A3 conflation to fix in the wording.
3. **Possible surrogate leakage** — **substantially resolved**: the released training scripts
   (`train_surrogate.sh`, `train_clip.sh`, `train_metadit.sh`) train on `train_set.mat` with
   `val` for selection; test is untouched. Residual = A6 (circularity/reward-hacking), not data
   leakage. State this evidence in the paper; one sentence in the harness docs would preempt the
   reviewer question.
4. **1 epoch vs 500** — unresolved; see B5.
5. **AAE&K not apples-to-apples** — confirmed; keep the label (they do) and demote it from any
   headline.
6. **No full-wave** — unresolved; see B4.
7. **Single seed / no ablations** — unresolved; note the roadmap's Phase 2c/3b *are* two of the
   needed ablations (physics-share; projector) — the λ=0 sweep arm already exists as a physics-off
   ablation, use it in the paper's ablation table rather than re-running.
8. **Determinism vs one-to-many** — framing; the candidate-sampling machinery (σ/CFG grid) is the
   honest diversity story; run the σ curve [0, 0.01, 0.05, 0.10] (specified but not yet executed)
   so the paper shows the trade-off rather than asserting it.

---

## 2. What the project gets right (so the review is not read as one-sided)

- **The two headline comparisons are true as measured**: 11.37 M trainable vs 37.20 M; MAE 0.0673
  vs 0.0801; AAE 40.54 vs 48.25; deterministic 1 forward. All independently recomputed here.
- **Not overfit** — the median match across train/val/test (0.0426/0.0425/0.0432) is a clean,
  well-executed check, and the "mean gap = tail" explanation is correct.
- **Metric parity is guarded** — the new benchmark metric module is tested against the *reference
  implementation* (`external/metadit/metric.py`) and passes in the cloud env. This is the right
  pattern.
- **The masked-fill diagnostic is real science-grade instrumentation** (visible-identity hard
  guarantee, seam, texture, near/far locality) with stub-based unit tests that genuinely test the
  helpers (`test_masked_fill_check.py` is the *good* example of test design in this repo).
- **The failure analysis is unusually honest** (the report contains four self-corrections of
  earlier claims). Keep that voice — it is an asset.
- **Task novelty (partial-observation occupancy conditioning) is demonstrated**: visible-identity
  0.0, seam mismatch 6–7%, locality ratio 0.39–0.50, B/C gates pass. (Scalar conditioning, the
  other half of the claim, is not — see A3.)

---

## 3. Suggested solutions — ordered action plan

**P0 — before the next cloud run (hours-to-a-day each, all local):**
1. Fix the physics-gradient regression test (A1) and make it stub-based so it runs everywhere.
2. Fix the preflight precedence block correctly (B2) — single implementation covering empty
   decoded/true samples, `checks` defined before use.
3. Equal-n NN comparison + pool curve; footnotes on the current table (A2).
4. AVG1 from the train mean; state the paper-number difference or drop the row (A4).
5. Label/regenerate the ≤8-item and synthetic-n diagnostics (A5).
6. Replace the vacuous tests with behavioral ones (B1).
7. Pass `--out` in the kernel; commit (or paste) the small result JSONs (C1); merge the two
   lines of work and re-pin (C4).

**P1 — the research items (already largely scoped by ROADMAP phases):**
8. Door (b) at full epoch; then the multi-target design (ROADMAP Phase 4 / TARGET_DESIGN.md —
   the spectrum-conditioned *geometry* target is the right call; note its gate 2 "no shortcut"
   probe is exactly the check this review would ask for).
9. Train ≥2 epochs with the canonical battery recorded per checkpoint (answers B5); add a
   ≥3-seed phase with CIs (B3).
10. Full-wave (or second-surrogate) validation on ~32 designs (B4) + an OOD guard for the tail.
11. Run the σ/CFG diversity curve; keep guidance-gap as a diagnostic, not a headline.

**P2 — positioning:** unchanged from the assessment: lead with the task (arbitrary
partial-observation conditioning across occupancy + parameters) and the determinism/efficiency
result; JEPA is the method, not the headline. Workshop first. The "first JEPA for
metasurface/EM inverse design" claim is defensible but narrow — pair it with the task
contribution and the benchmark artifact.

---

## 4. Bottom line

The project is in good scientific shape for its stage: the architecture works, the headline
numbers are real, the instrumentation is above average, and the self-assessment is honest. What
stands between it and a credible submission is not one big flaw but a set of **verification
hygiene** issues — one broken guard, one unfair comparison, one transductive baseline, a handful
of vacuous tests, and two branches that need merging. Fix P0, finish the measured scalar work
(P1), and the difference between "interesting direction" and "publishable result" becomes an
empirical question rather than a documentation question.

*Prepared against `tegga8/metasurface-jepa` @ `ad78d54` and local `1478026`; all executable
checks were run in a sandbox copy (`D:\_review_snapshot_tmp`, deleted after use). No repository
files were modified.*
