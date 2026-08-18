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