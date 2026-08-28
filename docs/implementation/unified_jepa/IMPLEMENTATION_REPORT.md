# Unified Occupancy–Parameter–Spectrum JEPA — Implementation Report

**Date:** 2026-08-27
**Scope:** Phase 1 → Phase 5 of the Unified JEPA restructure
**Authority:** `docs/implementation/unified_jepa/architecture_v5.md` (architectural spec) + `00_MASTER_EXECUTION.md` (execution controller) + `01`–`05` phase MDs
**Test status:** `473 passed, 12 skipped` (skips are CUDA-only tests on a CPU machine)

---

## 1. Final architecture (what was built)

The legacy Milestone-B representation (3-channel broadcast geometry `[B,3,64,64]`, 384-D
hidden) is replaced by a factorized internal representation that keeps the broadcast
tensor **only at the MetaDiT surrogate boundary**:

```
Inputs (each independently maskable):
  occupancy M[64,64]              l_lattice, h_atom, r_atom        target spectrum [2,301]
        │                                │                                │
        ▼                                ▼                                ▼
  OccupancyEncoder              ScalarEncoder (MLP)              SpectrumPath (frozen)
  (patch-4 ViT, 192-D,          FiLM (γ,β) per block             c_physics [384]
  single-channel input)         + scalar summary [B,1,192]       a_goal [B,16,384]
        │                                │                                │
        └──────────┬─────────────────────┴────────────┬───────────────────┘
                   ▼                                 ▼
            FusionEncoder (2-layer, 192-D)
            256 occ + 16 goal(proj 384→192) + 1 scalar → [B,273,192]
                   │
                   ▼
            GCLCT predictor (192-D, dense attention)
            256 occ mask-queries + 1 scalar-summary query,
            FiLM-conditioned by c_physics (384→192 projected)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  occupancy token pred      scalar-summary pred
  z_hat [B,256,192]         → scalar decoder → (l̂, ĥ, r̂) [B,3]
        │
        ▼
  GeometryDecoder (192-D) → soft occupancy sigmoid(logits)
        │
        ▼
  assemble_metadit_geometry(occ, l̂, ĥ, r̂) → [B,3,64,64]   ← broadcast boundary only
        │
        ▼
  frozen MetaDiT surrogate (ConvSurrogate, eval, params frozen)
        │
        ▼
  spectrum prediction Ŝ [B,2,301] → L_phys (normalized SmoothL1, ramped)

EMA target path (frozen, both):
  full occupancy + true l/h/r
    → EMA OccupancyEncoder (occupancy EMA, JEPA target) → z_y_raw [B,256,192]
    ← FiLM from scalar_mlp_ema (target-side conditioning ONLY, never a loss target)

Loss:
  L = λ_inv·L_inv + λ_var·L_var + λ_cov·L_cov + λ_scalar·L_scalar + λ_phys·L_phys
  (VICReg invariance/variance/covariance on masked occupancy tokens via an
   objective-owned projector; scalar L1/Huber on unknown positions only;
   λ_phys ramped from 0 per Phase 4 MD §4.1)
```

Key architectural decisions (from `architecture_v5.md`, all enforced):

- **192-D throughout** except `c_physics`/`a_goal` which stay 384-D at the frozen
  SpectrumPath and are projected down to 192.
- **No BI-JEPA, no second physics predictor, no second geometry decoder, no spectrum
  decoder, no generic 3-channel reconstruction loss.**
- **No scalar EMA latent loss target** — `scalar_mlp_ema` exists solely to stabilize
  the EMA target encoder's FiLM conditioning (architecture_v5.md §3.6).
- **No slicing/cropping of old 384-D Milestone-B weights** — the old implementation is
  preserved intact as a legacy/reference path.
- **Classifier-free goal guidance** (§3.5.1): ~10% goal dropout during training;
  `Ẑ_guided = P(Z_x, A_∅) + w·[P(Z_x, A_goal) − P(Z_x, A_∅)]` at inference.
- **STE for physics loss** (Phase 4 MD §3): the frozen surrogate's ReLU6 activations
  have a zero Jacobian on soft occupancy fields (verified empirically by
  `soft_hard_occupancy_test`); the documented, configurable `physics_use_ste` flag
  defaults to true.

---

## 2. Phase-by-phase changes

### Phase 1 — Data contract and new modules (`01_data_contract_and_new_modules.md`)

**Created:**

| File | Purpose |
|---|---|
| `src/data/factorize.py` | `factorize_geometry` (legacy `[B,3,64,64]` → occupancy `[B,1,64,64]` + scalars `[B,3]`), `assemble_geometry` (exact inverse, l/3, h raw, r/5), `assemble_metadit_geometry` (Phase-4 wrapper), `SCALAR_CONVENTION` metadata |
| `src/data/scalar_mask.py` | `ScalarMasker` — independent scalar known/unknown sampling; regimes: `all_known`, `all_unknown`, `independent`, `correlated`; explicit binary known flags (never a numeric sentinel); RNG state save/restore |
| `src/encoders/occupancy_encoder.py` | Single-channel patch-4 ViT: `Conv2d(1,192,k=4,s=4)` → 16×16 grid → 256 tokens of dim 192; fresh 192-D sincos pos-embed; per-block scalar FiLM (`x = γ⊙x + β`); FiLM identity init (γ=1, β=0); no weight transfer from 384-D MetaDiT blocks |
| `src/encoders/scalar_encoder.py` | `Linear(6,128)→GELU→Linear(128,128)` trunk over `[l_val, l_known, h_val, h_known, r_val, r_known]`; per-block FiLM heads (zero-init → identity); pooled scalar-summary token `[B,1,192]` |
| `src/fusion/fusion_encoder.py` | 2-layer 192-D fusion transformer; `goal_proj = Linear(384,192)`; token layout 256 occ \| 16 goal \| 1 scalar → `[B,273,192]` |
| `tests/test_unified_data_contract.py` | 25 tests: factorization, round-trip (synthetic + real), scalar flags/regimes, occupancy/scalar encoder shapes, FiLM identity init, fusion token count/width, scalar_mlp_ema independence |

**Files modified:** none (Phase 1 is additive; Milestone-B training loop untouched).

**Result:** 25/25 Phase-1 tests pass.

### Phase 2 — 192-D model integration (`02_model_integration_192d.md`)

**Modified:**

| File | Change |
|---|---|
| `src/predictor/gclct.py` | Added `c_physics_dim` param to `GCLCT`: when `c_physics_dim != hidden`, a `Linear(c_physics_dim, hidden)` projects c_physics (384→192); otherwise `nn.Identity()` (backward compatible — old 384-D path unchanged). `forward` now applies the projection before the blocks. |
| `src/encoders/target_encoder.py` | `EMAEncoder.forward(x, **kwargs)` — transparent kwargs passthrough so the EMA target can receive `film_params` for the occupancy encoder or scalar-MLP input for the scalar EMA. |
| `src/encoders/occupancy_encoder.py` | Added optional `mask`/`mask_token` to forward: masked patch positions are replaced with the learned placeholder before pos-embedding (§2 masking convention). |
| `src/train/engine.py` | `collect_ema_state`/`restore_ema_state` extended to carry `scalar_mlp_ema` (momentum counters + target weights) so exact resume preserves target-side FiLM conditioning. |

**Created:**

| File | Purpose |
|---|---|
| `src/decoders/scalar_decoder.py` | 3 small MLP heads (Linear→GELU→Linear) reading the scalar-summary query → `(l̂, ĥ, r̂)`. Final-layer bias initialized to dataset means `[2.75, 0.75, 4.25]` (verified from `external/metadit/datapipe.py`) so initial geometry is nonzero and inside the surrogate's training distribution — zero-init would collapse geometry and kill the surrogate's Jacobian (Phase 4 MD §3 "NO zero-init on this one"). |
| `src/assembly.py` additions | `UnifiedJEPA` (nn.Module) + `build_unified_model(cfg, spec_weights)` + `UNIFIED_ARCHITECTURE_ID = "unified_occ_param_spectrum_jepa_v1"`. Forward contract: `(occupancy, scalar_values, scalar_known, spectrum, mask, goal_mode, with_target)`. EMA init from students, never from old 384-D checkpoints. `loss()` = L_JEPA + scalar L1. |
| `configs/unified.yaml` | Unified config: 192-D model sizes, loss weights, curriculum, staging, weights paths, training hyperparams. |
| `tests/test_unified_model_phase2.py` | 25 tests: architecture ID, forward shapes, mask convention, gradient ownership (student yes / EMA + released no), EMA isolation/update, scalar input masking, goal-mode real/null/shuffled, GCLCT 384→192 projection, checkpoint round-trip, old-checkpoint incompatibility. |

**Checkpoint schema** (Phase 2 MD §8): distinct `architecture_id` stored in cfg; a
384-D Milestone-B checkpoint load into the 192-D model fails loudly
(`test_old_checkpoint_not_compatible`).

**Result:** full suite green at this point (408 passed, 12 skipped); Phase 2 tests 25/25.

### Phase 3 — Unified training loop and objective (`03_training_and_objective.md`)

**Created:**

| File | Purpose |
|---|---|
| `src/losses/unified_losses.py` | `UnifiedJEPALoss` — owns the VICReg projector (no `model.proj`, spec §17); combines L_inv/L_var/L_cov (VICReg on masked tokens), L_scalar (L1/Huber on unknown positions), L_phys (delegated to `physics_loop.physics_loss` — single authoritative implementation). `on_optimizer_step` updates BOTH EMA targets. `OccupancyTokenLoss`, `ScalarPredictionLoss`, `PhysicsSpectrumLoss` submodules. |
| `scripts/train/train_unified.py` | Standalone CLI trainer (`--config`, `--resume`, `--no-train`, `--device`). Curriculum sampling (mask ratios incl. 1.0, scalar regimes all-known/all-unknown/mixed, logged via `RegimeLogger`). Cosine warmup scheduler. Per-step EMA-frozen guard. Checkpointing via `train.engine.save_checkpoint` (resumable, EMA state incl. scalar_mlp_ema, masker RNG). Surrogate loading for physics loss and for `half_sensitivity` mask placement. Goal dropout (Phase 4 §3.5.1). λ_phys ramp. `--no-train` forward smoke. Synthetic data fallback for local dev. |
| `tests/test_unified_losses.py` | 14 tests: loss dict/components, backward gradient ownership, scalar-loss-only-unknown (L1 + Huber), physics disabled-by-default, curriculum config, single-channel masker, projector objective-owned, JEPA-on-masked-only. |

**Verified:** 5-step training run + checkpoint resume (LR continues at 1.5e-5 → 2.1e-5,
EMA restored) — resume-equivalence test passed.

**Result:** Phase 3 tests 14/14; smoke + resume verified.

### Phase 4 — Physics loop and scenarios (`04_physics_loop_and_scenarios.md`)

**Created:**

| File | Purpose |
|---|---|
| `src/physics/physics_loop.py` | `load_surrogate` (frozen ConvSurrogate, eval, params frozen, autograd flows through input). `physics_loss` — the single authoritative physics path: `z_hat → decode_geometry → assembled geometry → surrogate → Ŝ → normalized L1/SmoothL1/MSE`; `normalize` divides prediction AND target by per-sample spectrum std (wide dynamic range). `surrogate_gradient_test` — tries soft occupancy FIRST, falls back to documented STE only if soft yields zero student gradients (Phase 4 MD §3 "do not silently choose STE"). `soft_hard_occupancy_test` — quantifies soft-vs-binary surrogate OOD-ness, recommends STE. `preservation_loss` — L_preserve for known occupancy/scalars (Phase 4 MD §6). |
| `src/assembly.py` addition | `UnifiedJEPA.geometry_decoder` (GeometryDecoder, 192-D hidden, occupancy head) + `decode_geometry(z_hat, scalar_pred, occ_input, mask, use_ste)` — soft occupancy (sigmoid), optional STE hard-forward/soft-backward, visible-pixel retention from input occupancy, assembles `[B,3,64,64]` via `assemble_metadit_geometry`. |
| `src/predictor/guidance.py` | `cfg_combine(z_real, z_null, w)`, `goal_dropout(goal_mode, p, rng)`, `cfg_forward(...)` — two-pass CFG inference. |
| `src/diagnostics/guidance_gap.py` | `compute_guidance_gap` (‖z_real − z_null‖ / σ(z_real), §20.3) and `guidance_gap_sweep` across mask ratios. |
| `scripts/diagnostics/run_guidance_gap_sweep.py` | CLI producing the §20.3 gap curve. |
| `scripts/run_scenarios.py` | Scenario A (pure inverse), B (partial-parameter), C (retrofit); `real_null_shuffled_evaluation` — gate on the **hard stratum** (full mask + all scalars unknown), easy stratum reported separately (never pooled); NN retrieval baseline (per-sample L1 nearest training spectrum → retrieved geometry); diversity metrics (deterministic reporting). |
| `tests/test_phase4_physics.py` | 14 tests: assembly invariants (support, constancy, constants), round-trip, decoder shapes, visible-pixel retention, STE path, physics loss finite, surrogate gradient test, soft/hard characterization, scenario input shapes, diversity determinism. |
| `tests/test_phase4_guidance.py` + `test_phase4_guidance_gap.py` | 14 tests: cfg_combine formula/w=0/w=1, goal_dropout probability/idempotence, cfg_forward shapes/equality, guidance-gap dict/sweep/nonnegativity. |

**Result:** Phase 4 tests 28/28 (incl. real-surrogate gradient test using
`data/metadit/weights/surrogate_model.bin`).

### Phase 5 — Tests, evaluation, cleanup (`05_tests_evaluation_cleanup.md`)

**Created:**

| File | Purpose |
|---|---|
| `tests/test_phase5_contracts.py` | 23 tests: architecture contract shapes (§2), data invariants (§3), mask isolation (§4), EMA stability incl. `scalar_mlp_ema` no-gradient and momentum schedule (§5), physics-gradient regression (surrogate frozen, decoder/predictor/encoder receive gradient — automated, no `no_grad` around surrogate) (§6), spectrum dependence real/null/shuffled easy stratum (§7), scalar dependence hard stratum (§8), occupancy fraction / collapse flags (§10). |
| `scripts/eval/eval_scenarios.py` | Per-scenario evaluation (A/B/C reported separately, never pooled): spectrum error, occupancy IoU/F1, scalar MAE, occupancy fractions, real/null/shuffled gap, diversity, NN baseline. Runs from a trained checkpoint. |

**Repository hygiene (Phase 5 MD §13):** no raw dataset modifications, no
`external/metadit` edits, no generated checkpoints committed, no stale `_*.py` debug
files, `git diff --check` clean.

**Result:** full suite `473 passed, 12 skipped`.

### Review pass (post-implementation, this session)

After all five phases, a systematic review against the MDs found and fixed:

1. **Physics-loss normalization bug** (`physics_loop.py`): the `smooth_l1` branch
   computed `diff` but never used it — the `normalize` flag was silently ignored for
   the default loss type. Now both prediction and target are normalized by the
   target's per-sample std consistently across all loss types (Phase 4 MD §5).
2. **Wrong surrogate path** (`configs/unified.yaml`): `weights.surrogate` pointed at
   `data/metadit/dit.pt` (the DiT) instead of
   `data/metadit/weights/surrogate_model.bin`. Physics loss would have been silently
   inactive on cloud.
3. **Staging-phase inconsistency** (`configs/unified.yaml`): config said `phase: D`
   while `lambda_phys: 0` (stage B). Reset to `phase: B`, `lambda_phys_ramp_steps: 0`.
4. **Physics-loss DRY violation** (`unified_losses.py`): inlined a second copy of the
   physics path; now delegates to `physics_loop.physics_loss` (single authoritative
   implementation).
5. **Silent STE choice** (Phase 4 MD §3): `surrogate_gradient_test` now tries soft
   first and only falls back to STE when soft yields zero gradients; the training
   objective's STE usage is a documented, configurable `physics_use_ste` flag.
6. **Hard-stratum gate** (Phase 4 MD §10): `run_scenarios.py` now evaluates the
   real/null/shuffled gate on the hard stratum (full mask + all scalars unknown) and
   reports the easy stratum separately — no pooled gate.
7. **Mask placement hardcoded** (architecture_v5.md §2): `train_unified.py` now reads
   `curriculum.mask_placement` (random / half_sensitivity) and loads the surrogate for
   sensitivity maps when needed.
8. **Scalar convention metadata** (Phase 1 MD §3): `scalar_convention` added to
   `configs/unified.yaml` (carried into checkpoints via cfg), satisfying the
   "store it in config/checkpoint metadata" requirement.

---

## 3. Files created vs. modified

**Modified (tracked):**

| File | Nature |
|---|---|
| `src/assembly.py` | +337 lines — `UnifiedJEPA`, `build_unified_model`, `decode_geometry`, imports |
| `src/predictor/gclct.py` | +17 −3 — `c_physics_dim` projection |
| `src/encoders/target_encoder.py` | +7 −2 — kwargs passthrough |
| `src/train/engine.py` | +35 −1 — scalar_mlp_ema checkpoint state |

**Created (untracked):**

```
configs/unified.yaml
src/data/factorize.py            src/data/scalar_mask.py
src/encoders/occupancy_encoder.py   src/encoders/scalar_encoder.py
src/fusion/fusion_encoder.py
src/decoders/scalar_decoder.py
src/losses/unified_losses.py
src/physics/physics_loop.py
src/predictor/guidance.py
src/diagnostics/guidance_gap.py
scripts/train/train_unified.py
scripts/run_scenarios.py
scripts/eval/eval_scenarios.py
scripts/diagnostics/run_guidance_gap_sweep.py
tests/test_unified_data_contract.py
tests/test_unified_model_phase2.py
tests/test_unified_losses.py
tests/test_phase4_physics.py
tests/test_phase4_guidance.py
tests/test_phase4_guidance_gap.py
tests/test_phase5_contracts.py
```

**Preserved (untouched):** all Milestone-B code (`GoalConditionedJEPA`,
`train_milestone_b.py`, `configs/milestone_b.yaml`, `src/reference/`), `external/metadit`,
raw dataset files, `src/losses/objectives.py` registry, the phase1-decoder path.

---

## 4. Tests

| Suite | Count | Result |
|---|---|---|
| `test_unified_data_contract.py` (Phase 1) | 25 | pass |
| `test_unified_model_phase2.py` (Phase 2) | 25 | pass |
| `test_unified_losses.py` (Phase 3) | 14 | pass |
| `test_phase4_physics.py` (Phase 4) | 14 | pass (incl. real-surrogate gradient test) |
| `test_phase4_guidance.py` | 10 | pass |
| `test_phase4_guidance_gap.py` | 4 | pass |
| `test_phase5_contracts.py` (Phase 5) | 23 | pass |
| Legacy Milestone-B + hardening suites | ~358 | pass (no regressions) |
| **Full suite** | **473 passed, 12 skipped** | skips = CUDA-only tests |

Additional runtime verification (not in pytest): `train_unified.py --no-train` smoke
forward+backward with finite loss; 5-step training with checkpoint save + resume
equivalence; `run_scenarios.py --scenario all` end-to-end on a smoke checkpoint;
`run_guidance_gap_sweep.py` end-to-end.

---

## 5. Status of the two paths

**Legacy Milestone-B path — fully preserved and green.** All historical tests
(architecture, masking, VICReg/Barlow/LeJEPA objectives, checkpoint/resume, gate
tests) pass unchanged. The old 384-D architecture remains a reproducible reference;
configuration selection between paths is explicit (`variant: unified` in
`unified.yaml` vs. the Milestone-B trainer/config).

**New unified path — implemented, unit/smoke-tested, NOT yet trained.** Forward,
backward, EMA isolation, physics gradient flow, checkpoints/resume, and scenario
evaluators all run. No actual gradient-based training has been executed — per
AGENTS.md compute rules, training must run on cloud GPU (Kaggle/Colab) via
`CLOUD_TRAINING.md`, and the Phase 5 §11 scenario evaluation must be re-run against a
trained checkpoint before any inverse-design claim can be made.

---

## 6. Unresolved items / next steps

1. **Cloud training** (`python scripts/train/train_unified.py --config
   configs/unified.yaml`) on Kaggle/Colab per `CLOUD_TRAINING.md` — resume-safe,
   checkpoints to `checkpoints/unified/`.
2. **Scenario evaluation on a trained checkpoint**:
   `python scripts/eval/eval_scenarios.py --config configs/unified.yaml --checkpoint
   checkpoints/unified/latest.pt` — the real/null/shuffled hard-stratum gate
   (`real must outperform shuffled`) is the actual acceptance test; it currently
   fails on an untrained model (expected).
3. **Guidance-gap curve** on a trained model (`run_guidance_gap_sweep.py`) — should
   rise with mask ratio; flat/near-zero at high mask would indicate Failure Mode 2
   (§13).
4. **Diagnostic re-runs against the new encoder** (architecture_v5.md §8.2):
   `vicreg_gradient_attribution.py` and the within-bucket spatial-structure probe must
   be re-run on the 192-D encoder — prior results were on the old 384-D encoder and do
   not transfer automatically.
5. **NN baseline** in `run_scenarios.py` currently uses synthetic training data for
   smoke purposes; point it at real train-set spectra for the actual §12 baseline.

---

*This report describes code state and local verification only. It does not claim
inverse-design scientific success — that requires the cloud-trained checkpoint and the
per-scenario acceptance gates above.*
