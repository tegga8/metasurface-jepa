# ARCHITECTURE_CLEANUP_AUDIT.md

Status: DRAFT (written before any destructive edit, per the Architecture Repair spec)
Date: 2026-08-20
Head: 5346b93 ("Implement VICReg Projector and associated tests for jepa_vicreg objective")
Working tree: clean at time of audit.

This document records the full-file inventory, classifies every component, and catalogs the
confirmed defects before any file is modified. It is the pre-cleanup snapshot the final report
(§36, sections A–K) is written against.

## 0. Classification legend

- **ACTIVE** — part of the live jepa pipeline (`build_model("jepa")` + train/eval scripts). Must
  be repaired in place, not deleted.
- **REQUIRED-BUT-NOT-ACTIVE-YET** — declared in AGENTS.md repo layout for a later milestone
  (C–I) or not yet reachable from the current pipeline. Leave untouched this pass.
- **OBSOLETE** — dead code from the prior adaptive-ladder design (six-rung registry). Remove.
- **REFERENCE-ONLY** — kept for provenance / later ablation (e.g. Baseline 2 direct masked
  generator, §10.1) but must not be reachable from the active training path.

## 1. Defect catalog (confirmed by reading the files)

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| D1 | Predictor ignores `c_physics`. `GCLCTBlock.forward(x, kv, c=None, need_weights=False)` never uses `c`; `c_physics` is computed in `assembly.py` forward then discarded. | `src/predictor/gclct.py`, `src/assembly.py` | FiLM conditioning per spec §7 (6 groups, zero-init `cond`), `forward(x, kv, c_physics, ...)`. |
| D2 | Predictor docstring is stale: claims AdaLN-Zero(`c_physics`), 64-token Perceiver bottleneck, base+delta. None of these exist in code. | `src/predictor/gclct.py` module docstring | Rewrite to the true spec: 256 context tokens + 16 structured physics/goal tokens + global `c_physics` -> 256x384 `z_hat`. |
| D3 | Dead predictor params: `bottleneck_tokens` (assembly passes it; gclct ignores it), pixel-head `decoder_head` + `DirectMaskedGenerator`. Direct variant is unreachable anyway (`build_model` raises for `variant != "jepa"`). | `src/assembly.py`, `src/predictor/gclct.py` | Remove params; move `DirectMaskedGenerator` + pixel head to `src/reference/` (REFERENCE-ONLY, Baseline 2 provenance). |
| D4 | `context_encoder.py` docstring claims a Perceiver-IO-style 64-token bottleneck that does not exist in code. | `src/encoders/context_encoder.py` | Rewrite docstring to actual implementation. |
| D5 | Stale objective classes all reach the predictor through `model.proj` fallback: `JEPAObjective`, `JEPAVarianceObjective`, `JEPAVICRegObjective`, `JEPAVICRegDualObjective`, `JEPABarlowObjective`, and the current `LeJEPAObjective` (uses `model.proj`). Real model has no `.proj`. | `src/losses/objectives.py` | Registry pruned to exactly `{jepa_vicreg, jepa_barlow, lejepa}`; every objective owns its projector; no `model.proj` anywhere. |
| D6 | Evaluator reaches the projector via `getattr(model, "proj", None)` — silent None fallback masks a broken objective/projector wiring. | `src/train/engine.py` lines 157, 225, 330-331 | Evaluator must receive the objective's projector explicitly; hard error if absent. |
| D7 | Ladder machinery: six-rung registry (`jepa`, `jepa_var`, `jepa_vicreg`, `jepa_vicreg2`, `jepa_vicreg2/debug`, `lejepa`), `select_winner`/`phase_ok`/`goal_score`, adaptive controller, adaptive config, ladder tests. Superseded: final registry is exactly three objectives. | `src/train/adaptive.py`, `scripts/train/train_milestone_b.py`, `configs/milestone_b_adaptive.yaml`, tests | Delete ladder code/config/tests. |
| D8 | Training script embeds a legacy direct-masked-generator path plus adaptive-ladder branches; the training interface is `model.loss(...)`, not the shared `objective(model, G, S, M)`. | `scripts/train/train_milestone_b.py` line ~390 | Single-objective script calling the shared engine; interface `objective(model, G, S, M)`. |
| D9 | `decisive_representation_validation.py` lists `jepa_vicreg2` in MODELS and reads the projector off `model.proj`. | `scripts/eval/decisive_representation_validation.py` | MODELS = 3 objectives; projector loaded from checkpoint `objective_state`. |
| D10 | Checkpoints do not yet record objective name/objective_state; resume can silently swap objective or re-init a projector. | `src/train/engine.py` | §30 checkpoint integrity: `objective_state`, `optimizer_state`, `scheduler_state`, EMA state, RNG state, `objective_name`, config; strict load + optimizer-ownership check. |
| D11 | Collapse classification is single-dimensional (`HEALTHY/WARNING/COLLAPSED`) with no attribution; cannot distinguish projector collapse from raw latent collapse from physics-conditioning failure. | `src/diagnostics/representation_health.py` | Five-way classification: `PROJECTOR_COLLAPSE` / `RAW_COLLAPSE` / `PHYSICS_CONDITIONING_FAILURE` / `TARGET_GRADIENT_LEAK` / `INVALID_IMPLEMENTATION`. |
| D12 | No physics-conditioning audit exists; nothing measures whether `c_physics`/`a_goal` embeddings carry signal. | — | New `scripts/eval/physics_conditioning_audit.py` (Cases A/B/C interpretation). |

## 2. File-by-file classification

### src/
| File | Class | Notes |
|------|-------|-------|
| `assembly.py` | ACTIVE (repair) | `build_model("jepa")`; remove `bottleneck_tokens`; move `DirectMaskedGenerator` + pixel head to `src/reference/` (REFERENCE-ONLY); keep direct-raise guard. |
| `data/dataset.py` | ACTIVE | MetaDiT loader, shapes (3,64,64)/(2,301). Unchanged. |
| `data/mask.py` | ACTIVE | §2 block masking, random + half_sensitivity. Unchanged. |
| `diagnostics/goal_token_entropy.py` | ACTIVE | Goal-token utilization entropy, §20.2. Verify signature still used after ladder removal. |
| `diagnostics/representation_health.py` | ACTIVE (repair) | Fix D11: add five-way classification; keep `eff_ranks`, `goal_token_stats`, `same_token_cos` semantics pinned by tests. |
| `encoders/geometry_encoder.py` | ACTIVE | MetaDiT ViT init. Unchanged. |
| `encoders/context_encoder.py` | ACTIVE (repair) | Fix D4 docstring. |
| `encoders/spectrum_encoder.py` | ACTIVE | MetaDiT spec encoder -> 16 goal tokens. Unchanged (verify no stale bottleneck claims). |
| `encoders/target_encoder.py` | ACTIVE | EMA target, frozen, no `.proj`. Unchanged. |
| `losses/barlow.py` | ACTIVE (extend) | Keep `barlow_twins_loss`; build `BarlowObjective` on it with objective-owned projector (canonical Barlow, not a VICReg copy). |
| `losses/jepa_loss.py` | ACTIVE | `ProjectionMLP`, `jepa_loss`, stop-grad contract. Unchanged. |
| `losses/objective_modules.py` | ACTIVE | `VICRegProjector` (kept), add `BarlowProjector`, `LeJEPAProjector`. |
| `losses/objectives.py` | ACTIVE (rewrite) | Fix D5: exactly `VICRegObjective`, `BarlowObjective`, `LeJEPAObjective`; remove all stale classes; drop the six-rung registry; `OBJECTIVE_REGISTRY` = 3 names; interface `objective(model, G, S, M)`. |
| `losses/sigreg.py` | ACTIVE (extend) | Keep `sigreg_loss`; `LeJEPAObjective` uses it. Compare against upstream `galilai-group/lejepa` when network allows; document match/deviation (spec §16). |
| `losses/vicreg.py` | ACTIVE | Canonical VICReg terms. Unchanged. |
| `predictor/gclct.py` | ACTIVE (repair) | Fix D1/D2/D3: FiLM conditioning, docstring, remove dead params/pixel head. |
| `train/adaptive.py` | OBSOLETE | Fix D7: delete whole file. |
| `train/engine.py` | ACTIVE (repair) | Fix D6/D10: objective-aware evaluation, checkpoint integrity. |

### src/reference/ (new)
| File | Class | Notes |
|------|-------|-------|
| `direct_masked_generator.py` | REFERENCE-ONLY | Moved `DirectMaskedGenerator` + pixel head (Baseline 2, §10.1 provenance). Not reachable from active pipeline. |

### scripts/
| File | Class | Notes |
|------|-------|-------|
| `train/train_milestone_b.py` | ACTIVE (rewrite) | Fix D8: single-objective shared-engine script, CLI + `--resume`, resumable, cloud-invokable. |
| `eval/eval_vicreg_sanity.py` | ACTIVE (extend) | Becomes the shared offline validation + short-audit runner for all three objectives. |
| `eval/decisive_representation_validation.py` | ACTIVE (repair) | Fix D9. |
| `eval/physics_conditioning_audit.py` | NEW | Fix D12. |
| `eval/reproduce_metadit_baseline.py` | ACTIVE | Baseline reproduction. Unchanged. |
| `eval/eval_milestone_a.py` | ACTIVE | Milestone A retrieval eval. Unchanged. |
| `eval/unseen_multimask_generalization.py` | ACTIVE | Multi-mask generalization eval. Check for `proj`/ladder references, update if found. |
| `eval/compare_milestone_b_candidates.py` | ACTIVE | Candidate comparison. Check for ladder refs, update if found. |
| `diagnostics/check_ema_target_diversity.py` | ACTIVE | `same_token_cos` source (pinned by `test_same_token_cos`). Unchanged. |
| `diagnostics/make_synthetic_collapsed_ckpt.py` | ACTIVE | Test-fixture generator for collapsed checkpoints. Verify it doesn't rely on ladder schema; update to §30 checkpoint schema. |

### configs/
| File | Class | Notes |
|------|-------|-------|
| `milestone_b.yaml` | ACTIVE (clean) | Remove ladder/bottleneck keys; single-objective schema (`objective: jepa_vicreg`), training/validation/audit sections. |
| `milestone_b_adaptive.yaml` | OBSOLETE | Delete. |

### tests/
| File | Class | Notes |
|------|-------|-------|
| `test_eff_rank_math.py` | ACTIVE | Keep. |
| `test_same_token_cos.py` | ACTIVE | Keep. |
| `test_jepa_loss.py` | ACTIVE | Keep. |
| `test_sigreg.py` | ACTIVE | Keep. |
| `test_vicreg_collapse.py` | ACTIVE | Keep. |
| `test_vicreg_gradients.py` | ACTIVE | Keep (uses `VICRegObjective` + fake model, no `model.proj`). |
| `test_vicreg_objective.py` | ACTIVE (update) | Registry test must pin exactly 3 rungs; `objective(model,G,S,M)` interface. |
| `test_lejepa_objective.py` | ACTIVE (rewrite) | Objective-owned projector; no `model.proj`. |
| `test_tier2_fixes.py` | ACTIVE (update) | References to `AdaptiveController`/ladder removed; items kept that still apply. |
| `test_tier3_fixes.py` | ACTIVE (update) | `proj` references -> objective projector. |
| `test_healthy_reference_determinism.py` | ACTIVE (update) | Projected-space path -> objective projector. |
| `test_fixed_validation_metric_path.py` | ACTIVE (update) | `FixedValidation` signature after engine refactor. |
| `test_null_gap_metric_consistency.py` | ACTIVE (update) | Same signature updates. |
| `test_ladder_extension.py` | OBSOLETE | Six-rung registry pin + Barlow-loss fold-in; delete; Barlow loss tests re-homed into `test_barlow_gradients.py` / `test_barlow_collapse.py`. |
| `test_ladder_summary.py` | OBSOLETE | Delete. |
| `test_ladder_ratios.py` | OBSOLETE | Delete. |
| `test_winner_phase_ok.py` | OBSOLETE | Delete (ladder winner selection). |
| `test_jepa_var_objective.py` | OBSOLETE | Delete (`jepa_var` rung removed). |
| `test_jepa_vicreg2_objective.py` | OBSOLETE | Delete (`jepa_vicreg2` rung removed). |
| `test_predictor_conditioning.py` | NEW | Tests A–E per spec §22 (cond in range, frozen-cond-identity, cond-dropout, c/z independence sanity, gradient flow). |
| `test_barlow_gradients.py` | NEW | Barlow diag/off-diag gradients reach student + objective projector; never EMA target. |
| `test_barlow_collapse.py` | NEW | Constant input -> cross-corr off-diag -> penalty responds; minimization decorrelates; rank restoration. |
| `test_lejepa_gradients.py` | NEW | Teacher-free: gradients reach student through target path (`stop_grad_target=False`), no EMA target exists. |
| `test_lejepa_collapse.py` | NEW | SIGReg penalty responds to collapse; assert no EMA target in model wiring. |
| `test_checkpoint_resume.py` | NEW | §30 integrity: strict load, optimizer ownership, objective mismatch error, RNG continuity. |

## 3. Not-yet-existing components (REQUIRED-BUT-NOT-ACTIVE-YET — not built, do not build this pass)

Per AGENTS.md Standing Rule 1 (one milestone at a time) and the cleanup spec's scope (architecture
repair + cleanup + audit, not new capability):

- `src/predictor/guidance.py`, `src/predictor/routing.py` (Milestones D/E)
- `src/decoders/`, `src/surrogate/`, `src/stochastic/`
- `src/losses/geometry_loss.py`, `physics_loss.py`, `alignment_loss.py`, `goal_infonce.py`, `curriculum.py`
- `src/model.py`
- `configs/milestone_{c,d,e,f,g,h,i}.yaml`, `scripts/train/train_milestone_{c..i}.py`
- `scripts/eval/eval_hypotheses.py`, `scripts/eval/run_ablation.py`
- `scripts/diagnostics/run_routing_jacobian_check.py`, `run_guidance_gap_sweep.py`
- `external/lejepa/` (Milestone G; attempted clone blocked on slow network — see §5)

## 4. Non-goals / explicit exclusions this pass

- No changes to `data/` loaders or `mask.py` (correct as-is).
- No changes to `losses/vicreg.py`, `losses/jepa_loss.py` contracts (pinned by tests).
- No new capability (no guidance, routing, decoder, physics loss) — repair only.
- No retraining. Local machine is CPU-only (4GB VRAM too small); the GPU short audit (§15/§34)
  and multi-seed gates are cloud work per `CLOUD_TRAINING.md`.

## 5. External-verification caveat

`https://github.com/galilai-group/lejepa` was attempted (2026-08-20) but the clone did not
complete (TCP reachable, transfer too slow/blocked from this network; partial `.git` removed).
The spec's §16 "compare against upstream" requirement is therefore recorded as
**PENDING-NETWORK**. Local `sigreg.py` is a custom implementation and must not be presented as
canonical LeJEPA until the upstream comparison completes.

## 6. Cleanup order (dependency-respecting)

1. `src/predictor/gclct.py` (D1/D2/D3) — conditioning is upstream of everything.
2. `src/assembly.py` + `src/reference/` move (D3).
3. `src/encoders/*.py` docstrings (D4).
4. `src/losses/objective_modules.py` + `barlow.py` + `objectives.py` (D5) + `sigreg.py`.
5. `src/train/engine.py` (D6/D10).
6. `src/train/adaptive.py` deletion + `train_milestone_b.py` rewrite (D7/D8).
7. `src/diagnostics/representation_health.py` (D11).
8. `scripts/eval/*` (D9, D12, shared short audit).
9. `configs/` cleanup.
10. `tests/` update + new.
11. Full suite in §33 order; §32 diff audits; final report.