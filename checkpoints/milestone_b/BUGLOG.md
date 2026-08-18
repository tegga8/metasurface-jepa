# Milestone B — Bug Log (single source of truth for remaining fix work)

Status of all confirmed audit findings for the adaptive `L_J -> VICReg -> LeJEPA/SIGReg`
controller. Read this file first in any session touching Milestone B fixes; do not re-verify
already-FIXED rows unless something looks inconsistent.

## Bug table

| # | Item | Tier | File | Status |
|---|------|------|------|--------|
| 1 | Target projection not stop-gradient (proj(target) has no torch.no_grad) | 1 | src/losses/jepa_loss.py | FIXED (tests/test_jepa_loss.py) |
| 2 | Effective-rank naming/math wrong (eff_rank_unnorm is entropy H, not exp(H)) | 1 | src/diagnostics/representation_health.py (eff_ranks) | FIXED (tests/test_eff_rank_math.py) |
| 3 | Goal-token "effective rank" has same naming bug | 1 | goal_token_stats() | FIXED (tests/test_eff_rank_math.py) |
| 4 | FixedValidation prediction metric uses need_attn=True | 1 | src/train/engine.py (FixedValidation._acc_stats) | FIXED (tests/test_fixed_validation_metric_path.py) |
| 5 | WARNING phases can be selected as winner (only COLLAPSED excluded) | 1 | winner ranking / _phase_ok() | FIXED (tests/test_winner_phase_ok.py) |
| 6 | Legacy evaluator (run_eval/evaluate_jepa/_eval_ratio_masks) still active alongside FixedValidation, uses own masks + need_attn=True | 1 | scripts/train/train_milestone_b.py | FIXED (run_eval asserts direct-only; JEPA goes through _jepa_fixed_val_metrics) |
| 7 | healthy_references() CPU/GPU device hazard (target moved to .cpu() before projection, proj stays on device) | 2 | src/train/engine.py (healthy_references) | FIXED (projection runs on the model's device; raw-only tensors go to CPU; .cuda() hack removed — tests/test_tier2_fixes.py) |
| 8 | fixed_validation_from_loader doesn't honor exact n_samples when not divisible by batch_size | 2 | fixed_validation_from_loader() | FIXED (final batch trimmed to the exact remaining count; n_samples < batch_size now yields n samples, not batch_size — tests/test_tier2_fixes.py) |
| 9 | Mask coverage std uses default unbiased (NaN risk at n=1) | 2 | FixedValidation.__init__ | FIXED (population std, finite and 0.0 at n_batches=1 — tests/test_tier2_fixes.py) |
| 10 | global_step increments per micro-batch, not per optimizer.step() (breaks under grad_accum>1) | 2 | scripts/train/train_milestone_b.py | FIXED (one optimizer step == one global step; micro-batch counter gates stepping; log/val once per optimizer step — verified by integration runs, see Tier 2 verification below) |
| 11 | Checkpoint loading uses strict=False silently | 2 | src/assembly.py (load_into_model) | FIXED (strict=True default; released keys filtered on BOTH sides; loud RuntimeError listing missing/unexpected keys — tests/test_tier2_fixes.py) |
| 12 | Adaptive resume incomplete (only restores phase/objective/global_step/best_metric/best_step, not plateau/collapse counters, health history, scheduler state) | 2 | resume path | FIXED (full controller state_dict/load_state_dict: counters, histories, transitions, phase bookkeeping, best_health; python RNG saved too; verified by unit + integration resume runs — tests/test_tier2_fixes.py) |
| 13 | Prediction-health stats built from raw z_hat, not projected ph_ | 3 | src/train/engine.py (_pooled_pred_stats caller, zh_pooled population) | FIXED (tests/test_tier3_fixes.py — pooled pred mean_std scales with proj k; same convention in healthy_references) |
| 14 | healthy_references uses a separately-initialized refs_model's own proj head, not the candidate model's — different coordinate systems | 3 | src/train/engine.py (healthy_references), scripts/train/train_milestone_b.py (refs_model construction) | FIXED (proj_source param; candidate's head defines the coordinate system; legacy default kept — tests/test_tier3_fixes.py, test_tier2_fixes.py #7 identity contract extended) |
| 15 | No separation between best-prediction and best-healthy checkpoints; WARNING/COLLAPSED can become best_metric | 3 | adaptive controller / winner selection | FIXED (best_healthy is HEALTHY-gated only; phase_ok requires a HEALTHY-gated best AND HEALTHY provenance; select_winner returns explicit no-clean-winner — tests/test_tier3_fixes.py + tests/test_winner_phase_ok.py) |
| 16 | itertools.cycle(loader) used for adaptive training data iteration | 3 | scripts/train/train_milestone_b.py | FIXED (explicit `while enough_budget: for G, S in loader:` epochs; budget checks at epoch AND step; regression test asserts no cycle in code lines — tests/test_tier3_fixes.py) |
| 17 | Checkpoint resume does not save/restore CUDA RNG state | 3 | checkpoint save/load path | FIXED (collect_rng_state/restore_rng_state: torch+numpy+python, CUDA optional, CPU-safe skip; saved via save_phase_checkpoint, restored in load — tests/test_tier3_fixes.py) |
| 18 | Adaptive train-loss logging divides accumulated loss by val_every, not actual interval count | 3 | scripts/train/train_milestone_b.py | FIXED (IntervalLossAccumulator sum/count exact means; two independent instances for log vs val records; empty interval reports 0.0 — tests/test_tier3_fixes.py) |
| 19 | FixedValidation._acc_stats averages per-batch means instead of aggregating globally — not invariant to batch partitioning | 3 | src/train/engine.py | FIXED (global float64 loss_sum/mask_count in _acc_stats and null_gap, identical rule both paths; partition-invariance unit-tested — tests/test_tier3_fixes.py) |
| 20 | jepa_loss() silently falls back to full-token mean when zero masked tokens, contradicting strict masked-only objective | 3 | src/losses/jepa_loss.py | FIXED (explicit ValueError "mask contains no masked tokens"; also under proj — tests/test_tier3_fixes.py) |
| 21 | No explicit handling for n_samples < 2 in health diagnostics — can silently emit NaN into classification | 3 | representation_health.py / classify_health | FIXED (token_space_stats/_pooled_pred_stats emit NaN markers for n<2; classify_health returns UNAVAILABLE with reason; pairwise_cos_stats guards empty pair set; var_stats unbiased=False at n=1 — tests/test_tier3_fixes.py) |

## DEFERRED — needs operator decision (do NOT act without explicit go-ahead)

- Mask-ratio policy A/B (block-mask placement/evaluation policy split).
- VICReg/SIGReg formulation placement (which losses attach to which blocks).
- Best-prediction vs best-healthy separation criteria for winner selection.
- Goal-token ranking criteria.
- Base-init immutability hashing.
- Repo hygiene (probe-file cleanup, gitignore tightening) — DONE 2026-08-18 (operator-delegated
  hygiene pass): repo-root probe files removed (68 `out_*.txt/.json`; `/out_*` gitignored);
  `checkpoints/**/adaptive/` artifacts gitignored, incl. untracked `_smoke/LOSS_LADDER_SUMMARY.md`
  (git rm --cached — smoke-scale summaries are verification-only, not results); `logs/.gitkeep`
  added (layout dir was missing); `requirements.txt` created — it was absent despite being
  referenced by CLOUD_TRAINING.md and the cloud notebook (`pip install -r requirements.txt`),
  pinned to the locally-verified versions (torch 2.13.0, numpy 2.4.6, scipy 1.17.1, PyYAML 6.0.3,
  scikit-learn 1.9.0, timm 1.0.28, einops 0.8.2, transformers 5.11.0, matplotlib 3.11.0,
  tqdm 4.68.2, safetensors 0.8.0, pytest 9.1.1).

## Tier 1 — suite result

FULL SUITE PASS 2026-08-18 **after all Tier 1 + Tier 2 fixes** (46 tests, 7 files, all PASS;
`py_compile` clean on all changed files): tests/test_jepa_loss.py (4), tests/
test_eff_rank_math.py (6), tests/test_fixed_validation_metric_path.py (4), tests/
test_winner_phase_ok.py (6), tests/test_same_token_cos.py (5), tests/
test_null_gap_metric_consistency.py (9), tests/test_tier2_fixes.py (11 — new, covers Bugs
#7/#8/#9/#11/#12; #10 by integration, see below).

## Tier 2 — integration verification (#10, #12; 2026-08-18)

Three temp-config runs of the adaptive ladder (configs in `%TEMP%\opencode\tier2_cfg_a/b.yaml`,
outputs in `checkpoints/milestone_b/adaptive/`; the canonical `--smoke` was also re-run and
`adaptive/_smoke/` artifacts refreshed):

- **accum=1 (cfg a)**: 6 optimizer steps, `val @ 0, 2, 4` — the exact pre-fix cadence is
  preserved; in-loop best `0.13656535744667053` == final winner eval (bit-identical);
  `null_gap=0.0341` non-collapsed.
- **accum=2 (cfg b)**: 6 OPTIMIZER steps consuming 12 micro-batches — global_steps and the
  budget now count optimizer steps, not micro-batches (pre-fix, 12 micro-batches would have
  been counted as steps 0–11 and burned the budget twice as fast). In-loop best
  `0.14166349545121193` == final winner eval (bit-identical).
- **resume (cfg b, `--resume phase_00_jepa_latest.pt`)**: restored at "global step 5",
  executed step 5, ended at budget 6; best metric `0.14166349545121193` carried across the
  resume boundary, final winner eval identical. Restored plateau/collapse counters,
  histories, transitions, optimizer + python RNG all load without error (unit-level
  decision-continuation proven by `test_restored_counters_drive_future_decisions`).
- **strict-load compatibility**: legacy `synthetic_collapsed.pt` still loads under
  strict=True and the collapse diagnostic reproduces the same STILL COLLAPSED verdict.

## Tier 3 — suite result (2026-08-18)

FULL SUITE PASS **after all Tier 3 fixes** (69 tests, 9 files, all PASS; `py_compile` clean on
all 7 changed files): the 46 pre-Tier-3 tests remain green plus `tests/test_tier3_fixes.py`
(21 tests covering #13-#21) and `tests/test_winner_phase_ok.py` re-worked for the #15 semantics
(HEALTHY-gated best + HEALTHY provenance required; no-clean-winner instead of fallback).
`tests/test_tier2_fixes.py` #7 extended: the EMA-output identity contract now coexists with the
#13 projected-prediction path (4 proj calls for 2 batches: ema z_y + z_hat each).

**Adaptive smoke re-run** (`configs/milestone_b_adaptive.yaml --smoke`, `adaptive/_smoke/`,
budget 6, val @ 0/2/4): HEALTHY at every validation (votes=0); in-loop best (`best_healthy`)
`0.09906762448175228` at step 4 == final winner eval `0.09906762448175228` (bit-identical);
winner checkpoint = `phase_00_jepa_best_healthy.pt`; `null_gap=0.06618` on the winner —
goal utilization intact; no crash, clean exit. Rows #13-#21 all FIXED above.

## Tier 4 — six-objective screening ladder, Batch 4 (2026-08-18)

Batch 4 = Batch 3 approved + six-rung config + full component logging + LeJEPA smoke.

**Changes.**
- `configs/milestone_b_adaptive.yaml`: ladder now exactly `[jepa, jepa_var, jepa_vicreg,
  jepa_vicreg2, jepa_barlow, lejepa]`; screening budget `max_total_steps: 800`,
  `val_every_steps: 50` (per EXPERIMENT_LOG "Planned screening ladder"); per-rung
  `objective_params` for all six.
- `src/losses/objectives.py`: every regularized rung now reports `*_weighted`
  (lambda-scaled regularizer) and `var_ratio`/`cov_ratio`/`barlow_ratio`/`sigreg_ratio`
  (regularizer's share of total loss) as TENSORS so the loop's tensor-only accumulator
  picks them up; `jepa_vicreg2` also reports combined `L_var`/`L_cov`. Plain `jepa`
  stays L_J-only (tested).
- Batch 4 logging fix: `barlow_twins_loss` info returns python floats, which the
  loop's tensor-only accumulator silently dropped — `bt_diag`/`bt_off_diag` never
  reached the phase report. `JEPABarlowObjective` now re-wraps them as 0-dim tensors
  (`zh.new_tensor(...)`); `tests/test_ladder_ratios.py` asserts they are tensors.
- `src/train/engine.py` `write_ladder_summary`: fixed a crash when a phase ends
  before its first validation (`best_cos_err` None → `:.6g` TypeError; observed in
  the smoke when jepa_var hit the global budget at 1 step, 0 vals — reachable in
  real runs too, e.g. a phase starting at step >= 800-50). Also writes the .md with
  `encoding="utf-8"` (was platform-default cp1252 on Windows, corrupting the em-dash).
  Regression tests: `tests/test_ladder_summary.py`.
- `scripts/train/train_milestone_b.py`: `_objective_kwargs` covers all six rungs
  (defaults == class defaults; config `objective_params` overrides). `--smoke` no
  longer forces `objectives: [jepa]` — it runs the CONFIGURED ladder, so the smoke
  exercises all six rungs. Smoke cycling is now deterministic via `min_delta=1e6`:
  a released-init model on the 4-sample fixed val set improves at EVERY validation
  (jepa improved through 39 straight steps in the first attempt, burning the whole
  budget and starving the other five rungs); with min_delta=1e6 only a phase's FIRST
  validation can register a best, so the plateau switch always fires at
  `plateau_patience` valuations after warmup (6 phases × 3-4 steps, budget 28).

**New tests (Batch 4): `tests/test_lejepa_objective.py` (6) + `tests/test_ladder_ratios.py`
(7) + `tests/test_ladder_summary.py` (2) = 15. LeJEPA contract verified: student
geometry_encoder supplies z_y (forward called with `with_target=False`), `jepa_loss`
called with `stop_grad_target=False` (backward reaches the student encoder),
`on_optimizer_step` never touches the EMA (counting stub), `L = L_J + lambda_sigreg
* L_SIGReg` exactly, finite loss + gradients. Weighted/ratio components verified for
all five regularized rungs; plain `jepa` verified clean.

**FULL SUITE PASS (110 tests, 13 files, all PASS; 5 skipped = CUDA-only SIGReg);
`py_compile` clean on all changed files.**

**Six-objective local smoke** (`--smoke`, `adaptive/_smoke/`, batch 1, CPU): all six
objectives instantiated and executed (jepa 3 steps, others 4 each, budget 28), zero
objective-specific crashes, zero unstable steps, collapse_detected=False, votes=0 at
every validation. Two runs recorded:
- Run A: HEALTHY at every validation; winner = lejepa (best_healthy 0.06919831360435916
  @ step 20); final winner eval cos_err 0.06919831360435916 — **bit-identical**
  in-loop == final eval; null_gap 0.140 on the winner (goal utilization intact).
- Run B: same ladder, but status=WARNING at every validation (votes=0) — the health
  status at the 4-sample smoke scale flips run-to-run because `refs_model` is freshly
  randomly built per run (pre-existing: refs_model is NOT loaded from
  base_initialization.pt; only the candidate is). No collapse signals either way. Run B
  exercised the no-clean-winner path end-to-end (healthy-only winner semantics hold:
  no HEALTHY checkpoint → no deployment candidate, correct per Bug #15).

**Component logging observed (Run B phase reports, per-objective means):**
`L_J` (all rungs); `L_var`/`L_var_weighted`/`var_ratio` (jepa_var: 0.89/0.81,
jepa_vicreg: 0.92/0.09/0.50); `L_cov`/`L_cov_weighted`/`cov_ratio` (jepa_vicreg
0.0003/1.2e-5/5.8e-5); `L_var_pred`/`L_var_target`/`L_cov_pred`/`L_cov_target` +
combined + weighted + ratios (jepa_vicreg2); `L_BT`/`bt_diag`/`bt_off_diag`/
`L_BT_weighted`/`barlow_ratio` (jepa_barlow); `L_SIGReg`/`L_SIGReg_weighted`/
`sigreg_ratio` (lejepa 0.0093/0.00093/0.013).

**Screening-relevant observation (operator decision needed before the real run):**
with `lambda_bt=1.0`, barlow_ratio ≈ 0.9997 — the Barlow term (~200) swamps L_J
(~0.05), i.e. the jepa_barlow phase is effectively Barlow-only training with a
cosmetic JEPA term. Whether lambda_bt=1.0 is the intended screening configuration
is an operator call; the ratio logging now makes this visible per phase.

**Remaining risks (Batch 4):** (1) refs_model nondeterminism above — flag for a
decided seed or base-init load for refs_model if health status at screening scale
matters; (2) lejepa sigreg_info (dict) is not in loss_components (tensor-only
accumulator) — hyperparameters are recorded via `loss_components_config` instead;
(3) `objective_params` values for the real 800-step run (lambda_bt in particular)
need operator sign-off before Kaggle.

## Tier 5 — pre-training cleanup (2026-08-18, operator-directed continuation)

Final cleanup pass before the 800-step global screening; everything below is verification,
not new mechanism. Do not redo any of these fixes unless a regression is demonstrated.

**Barlow dimension-normalization fixed** (`src/losses/barlow.py`): the Barlow loss is now
dimension-normalized (off-diagonal cross-correlation terms divided by D, matching the
diagonal/off-diagonal scale). Measured scale on identical synthetic D=384 data: NEW L_BT ≈
0.997 — the previous D=384 raw-sum domination is gone (the old raw-sum behavior is preserved
as the historical Batch-4 observation above: barlow_ratio ≈ 0.9997 at lambda_bt=1.0). Do NOT
change lambda_bt again without a new demonstrated regression. Regression coverage: Barlow
normalization/scaling tests + Barlow D=1 guard in the suite.

**Healthy reference made deterministic** (`src/train/engine.py` healthy_references): the
candidate model's own `proj` head now defines the projected-reference coordinate system
(Bug #14 semantics), reference stats are built without perturbing the ambient RNG, and the
reference stats are reproducible run-to-run. Regression coverage:
`tests/test_healthy_reference_determinism.py`.

**SIGReg device + phi shape verified** (`src/losses/sigreg.py`): generator, slice directions,
and subsample indices are created on `z.device` (the Phase-2 Kaggle CPU/CUDA crash fix, see
EXPERIMENT_LOG); ECF phi shape corrected to (num_slices, num_points) as expected by the Epps-
Pulley machinery; mathematical formulation unchanged. CPU tests pass; CUDA-only tests skip
cleanly when CUDA is unavailable. Regression coverage: `tests/test_sigreg.py`.

**LeJEPA contract verified** (extends the Tier-4 LeJEPA verification): forward/backward
contract tests, no-EMA-update contract (`on_optimizer_step` never touches the EMA),
`stop_grad_target=False` contract (backward reaches the student encoder), and metadata
coverage (sigreg hyperparameters recorded via `loss_components_config`). Regression coverage:
`tests/test_lejepa_objective.py`.

**Metadata complete**: all six objectives' loss components and ratios are logged
(jepa_var, jepa_vicreg, jepa_vicreg2 branch-specific, jepa_barlow incl. bt_diag/bt_off_diag,
lejepa incl. sigreg metadata); the earlier Barlow float-component logging bug was fixed by
re-wrapping as 0-dim tensors (Tier 4 above).

**FULL SUITE PASS: 121 passed, 6 skipped** (13+ test files; the 6 skips are CUDA-only SIGReg
tests on the local CPU machine). `py_compile` clean on all changed files. The suite grew from
the 110-pass Batch-4 state with: healthy-reference determinism tests, Barlow
normalization/scaling tests, Barlow D=1 guard, frozen-EMA boundary tests for VICReg2 and
Barlow, SIGReg CUDA-gradient coverage, LeJEPA metadata coverage.

**Two repeated six-objective smoke runs reproducible** (`--smoke`, `adaptive/_smoke_A/` and
`adaptive/_smoke_B/`, CPU, budget 28, six rungs each): both runs are observably identical —
identical health classifications (HEALTHY at every validation, votes=0), identical collapse
votes, identical objective-ladder behavior (all 5 transitions by plateau, restart
best_healthy), identical winner (jepa_vicreg2, best_healthy 0.061089852593553255 @ step 12),
identical final winner evaluation (cos_err_r0.5 0.061089852593553255, null_gap
0.10827767001317802, in-loop best == final eval bit-identical). All JSON/MD/JSONL run
artifacts are byte-identical between the two runs. All 24 .pt checkpoint pairs are
semantically identical (top-level keys, model state dicts incl. all 438 tensors, optimizer
state, scheduler/cfg, torch RNG, numpy RNG, best metrics, health metadata, adaptive
metadata) with one benign exception: `python_rng` (Python `random` module state) differs
from index 0 because `set_seed()` seeds torch+numpy only and Python's `random` is never
drawn during training (saved/restored for completeness only, `src/train/engine.py`) — its
state is per-process ambient entropy and training-irrelevant. base_initialization.pt is
byte-identical. This is the determinism/objective-execution/validation-consistency
verification for the real run; it is NOT a scientific screening result (smoke scale, 4
samples) — the global 800-step screening on Kaggle is still the decision run.

## Tier 1 — collapsed-checkpoint diagnostic re-run

DONE 2026-08-18 (no retrain; existing checkpoint `checkpoints/milestone_b/synthetic_collapsed.pt`,
step 2687, 128 held-out geometries, exp-scale effective rank):

| metric | EMA target | released ViT | random init |
|---|---|---|---|
| pairwise cos mean | 0.999947 | 0.996079 | 0.999288 |
| pairwise cos p05 | 0.999802 | 0.986828 | 0.997634 |
| entropy eff rank | **1.00489** | 2.33418 | 2.2653 |
| participation | 1.00105 | 1.62977 | 1.70611 |
| top eig frac | 0.999477 | 0.771541 | 0.746362 |

Verdict: **STILL COLLAPSED / DEGENERATE** — eff-rank ratio vs collapsed anchor 0.22x (need >5x),
p05 margin vs anchor −0.0002. The synthetic collapsed checkpoint is more degenerate than the
recorded Kaggle step-2687 anchor (eff rank 1.00 vs 13.44), as intended: it exists to prove the
detector fires. The fixed exp-scale math (row 2/3) reports it correctly, where the old
entropy-scale value (2.6 nats) read as "healthy-looking".
JSON: `checkpoints/milestone_b/collapse_diag_rerun.json`.

## Tier 1 — blocker: in-loop vs winner-eval metric divergence (RESOLVED 2026-08-18)

**Symptom.** FixedValidation wrapper paths reported `cos_err ≈ 0.888873` while the canonical
`_acc_stats` reported `0.087273` on byte-identical forwards (proj inputs/outputs/weights
fingerprinted identical; model params and RNG unchanged).

**Root cause.** `torch.nn.functional.normalize(x, -1)` binds the second positional argument
to `p` (norm order), NOT `dim` (`dim` defaults to 1 — the 256-token dim). Every probe/wrapper
metric variant passed `-1` positionally → p=-1 normalization across the token dim → garbage
cosine distances (0.888873). The canonical `_acc_stats` used `normalize(..., dim=-1)` correctly
(0.087273). Verified exactly: probe re-run of the buggy wrapper reproduces 0.888873; the
`dim=-1` wrapper equals the canonical path to 1e-12 (0.000000e+00 abs diff).

**Real-code instance fixed.** `FixedValidation.null_gap()` in `src/train/engine.py` had the same
positional `-1` in all three normalize calls (lines 172–177) — the null-goal diagnostic was
returning garbage cos_errs (0.888873 vs the fixed 0.087273). Corrected to `dim=-1`; also added
the `proj=None` guard to match `_acc_stats`, and removed the `DEBUG_ACC_STATS` debug blocks from
`_acc_stats`.

**Verification.**
- `tests/test_null_gap_metric_consistency.py` (new, 9 tests, all PASS): positional-vs-keyword
  pitfall; `null_gap` real-mode == `evaluate` cos_err (bit-exact); null-mode == evaluate(null);
  order-invariance (canonical→diagnostic→canonical); input immutability; proj-weight invariance;
  mode restoration; `need_weights` branch does not change predictions.
- Adaptive smoke rerun: in-loop best `0.23583755269646645` == final winner eval
  `0.23583755269646645` (bit-identical; previously ~0.230 in-loop vs ~0.9376 winner-eval).
- `null_gap` on the winner checkpoint: real 0.23584 / null 0.23598 / gap 0.04085 — gap
  non-collapsed, goal utilization intact.