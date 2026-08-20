# FINAL ARCHITECTURE REPORT — shared-architecture hardening

Date: 2026-08-20. Scope: architecture only. No VICReg/Barlow/LeJEPA mathematics,
no SIGReg, no objective coefficients, no objective selection, no winner selection,
no long training. This report covers both the architecture-integrity repair pass
(pre-cleanup snapshot in `ARCHITECTURE_INTEGRITY_AUDIT.md`) and the final
shared-architecture hardening pass (spec §0–§28). Section letters follow hardening
spec §26.

**Hardening spec §28 stop condition:** the shared architecture is hereby FROZEN in
the form documented below. No backbone changes while objectives are evaluated
unless a test demonstrates a concrete architecture defect.

---

## B. Geometry path

- 3x64x64 → Conv4x4/stride4 → 256 tokens x 384 dims (`GeometryEncoder`, patch-4 ViT,
  6 pre-norm blocks, affine-less LN eps 1e-6, qk-norm attention, GELU-tanh MLP).
- MetaDiT init (`init_from_metadit`) strict: pos_embed shape assert, patch kernel
  center-init, per-block `load_state_dict(strict=True)`. Untouched this pass.
- §25 residual-scale drift measured (batch 8, depth 6): post-embed std 2.008 →
  2.143 (+6.7%), mean norm 40.3 → 43.2 (+7.4%). Not substantial → no final norm
  added (authorized only if substantial).
- Shared-encoder identity locked: `model.context_encoder.geo is
  model.geometry_encoder` (A6) — the EMA cannot silently track a different network
  than the one trained by context encoding.

## C. Masking path

- Block masking per §2: 1–4 axis-aligned blocks, min side 3 tokens, random or
  half-sensitivity placement; pixel-space `apply_mask_to_pixels` (M upsampled 4x
  per axis), masked patch embeddings replaced by `mask_token + pos` (never
  zero-fill semantics).
- Mask protocol in the audit is reported explicitly (placement, min_side, k_range,
  token_grid, ratio) — the actual protocol used, per hardening §9.
- Locked by `tests/test_architecture_masking.py` (9 tests): M1 masked-value
  invariance (now strengthened per hardening §13: masked regions filled with
  random noise, exact ones, and valid-looking channel values — r_atom/5, h_atom,
  l_lattice/3 uniform draws — all produce bit-identical z_x; this is also the §21
  leakage guard), M2 visible sensitivity, M3 mask sensitivity, M4 alignment
  (patch (r,c) → token r*16+c in context token, predictor query, EMA target
  token, loss selection; loss ignores unmasked tokens).
- Hardening §14 channel-validity test: visible pixels retain EXACT original
  values, masked pixels exactly zero, channel order unchanged (binary material
  channel never mixed with the geometry channels by masking).
- Hardening §17 full-model spatial-alignment test: unique per-patch identifiers
  survive transpose/reshape/mask.view — flipping a visible patch moves exactly its
  own context token (argmax == token index), flipping a masked patch leaves z_x
  bit-identical, mask vector == `(M.view(B,-1) == 0)`, masked queries use
  mask_token+pos (differ from context tokens at the same index).

## D. Target/EMA path

```
FULL GEOMETRY
     │
     ▼
 EMA TARGET (deepcopy of shared geometry encoder, ALL params frozen)
     │
  z_y_raw  (B, 256, 384)
     │
 feature-wise F.layer_norm(z_y_raw, (384,))   — no params, no running state
     │
  z_y_normalized
```

- Update order locked by a deterministic test (hardening §6): loss → backward →
  gradient clipping → `optimizer.step()` → `EMA.update(student, step)` →
  zero_grad. The test clones student+target, runs a real optimizer step, verifies
  the student changed, the target did NOT change until `EMA.update`, and target
  after update == `m·target + (1−m)·student` within 1e-6.
- Never updated during forward/backward/inside the optimizer (A10, gradient
  ownership tests). No gradient ever reaches target params (A11). Target absent
  from optimizer param groups (A8).
- EMA momentum 0.996 → 0.999 linear ramp (`EMAEncoder.set_total_steps`).
- No hidden representation map exists between z_hat, z_y_raw, z_y_normalized
  except this explicitly named normalization boundary (§27).

## E. Physics path

- Frozen released `VanillaSpectrumEncoder` (requires_grad_(False); eval mode
  ENFORCED structurally, see I). `SpectrumPath` keeps `released` out of the
  optimizer, checkpoints (SAVED_EXCLUDES), and gradients.
- `c_physics` (B,384) = mean-pooled summary → FiLM-conditioning every predictor
  block (Route B); `a_goal` (B,16,384) = 16 structured tokens retained in the
  cross-attention kv (Route A). `goal_mode='null'` zeroes both.
- Physics NEVER enters the target encoder: `z_y_raw` is `ema(G)` only (§19).
- A16: gradients reach all trainable spectrum-pooling weights after the FiLM
  zero-init thaws.

## F. Predictor

- GCLCT depth 8, dense attention only (no sparse routing / CFG — Milestones E/D).
  Per block: affine-less LN → FiLM(c_physics) → self-attn / cross-attn (kv = 256
  z_x + 16 a_goal) / MLP with residuals. Final affine-less LN + Linear(384→384).
- No Perceiver, no 256→64 bottleneck, no base+delta, no shared model.proj, no
  objective projector inside the model (A2–A5, §22 greps: only attention-internal
  `CrossAttention.proj` / `SpectrumPath.proj` / patch projection, which are
  ordinary network layers and allowed).
- Architecture does not know objectives (hardening §21 test: no vicreg/barlow/
  lejepa/sigreg/objectives references in `src/encoders`, `src/predictor`,
  `src/data`).

## G. Target normalization

- `z_y_normalized = F.layer_norm(z_y_raw, (384,))` — per-token feature-wise,
  no learnable parameters, no running state, never mutates z_y_raw, never mutates
  EMA weights.
- Output contract (hardening §2/§1): `out["z_y_raw"]` and `out["z_y_normalized"]`
  are the ONLY public target representations. `out["z_y"]` remains as a
  backward-compatible raw alias ONLY; active code must not consume it — enforced
  by test `test_h1_active_code_never_consumes_z_y_alias` (greps src/ + scripts/
  for `["z_y"]` consumption; sole allowed occurrence is the alias-creation line
  in `src/assembly.py`).
- T1–T5 tests: shape (B,256,384); per-token feature std ≈ 1 (1.0000 up to fp32
  rounding with `unbiased=False`); raw unchanged; no trainable params; no added
  state on the EMA target.

## H. Parameter ownership

- Trainable: student geometry encoder, predictor (incl. zero-init FiLM conds),
  SpectrumPath pooling (proj_g, proj_goal, goal queries, q/kv/proj).
- Frozen (never in optimizer): EMA target (A8 + hardening §20 test asserts
  released and EMA params absent from optimizer param groups), released spectrum
  encoder.

## I. Frozen module modes

- `GoalConditionedJEPA.enforce_frozen_reference_modes()` sets
  `ema.target.eval()` and `spectrum_path.released.eval()` (safe when released is
  absent in stubs). `model.train()` override re-enforces after every mode switch;
  `forward()` calls it on entry, covering checkpoint load / resume / any
  train/eval switch (hardening §4/§5).
- R1–R4 tests: after `model.train()`, `ema.target` and `released` are NOT in
  training mode; released params `requires_grad=False`; both remain eval across
  checkpoint restore + train/eval/train cycling. This prevents silent stochastic
  behavior from frozen components (e.g. BatchNorm in a "frozen" module).

## J. Shape contract

Every forward enforces (asserts raise immediately on deviation):

| tensor | shape |
|---|---|
| `z_hat` | (B, 256, 384) |
| `z_y_raw` | (B, 256, 384) |
| `z_y_normalized` | (B, 256, 384) |
| `mask` | (B, 256) |
| `c_physics` | (B, 384) |
| `a_goal` | (B, 16, 384) |

## K. Leakage tests

- M1 (strengthened): masked content can be noise / ones / valid-looking channel
  values — z_x bit-identical. §21: masked content never enters the context.
- §22 unique-identifier alignment + hardening §17 full-model alignment (above).
- §14 channel-validity masking (above).
- EMA target never receives gradients or optimizer steps (A11, A8).

## L. Conditioning tests

- A14/A15/A16 + `test_predictor_conditioning.py` A–E: changing c_physics changes
  predictor output after activation; condition module receives gradient after
  activation; null condition changes output; shuffled physics changes output.
  Zero-init caveat respected: FiLM output is zero at step 0 by design (§15), so
  the tests distinguish "identical at init" from "identical after activation".

## M. Real-data audit

- `scripts/diagnostics/architecture_audit.py --real-data` loads the ACTUAL
  configured validation split (`data/metadit/split_data/val_set.mat`, no invented
  split), fixed subset 0..batch-1, fixed masks, and reports: actual mask protocol,
  objective-independent raw-JEPA loss, geometry statistics (per-channel mean/std,
  occupancy fraction), spectrum statistics, condition sensitivity. Header prints
  "REAL DATA ARCHITECTURE AUDIT".
- Local CPU real-data run (20 steps, batch 8, real released weights):
  **PASS, no structural flags**; last row L_J 0.071, z_y eff_rank_frac 0.28,
  cond_sensitivity_normed 0.81, physics diagnosis CASE_3.
- Synthetic mode is explicitly named SYNTHETIC in the JSON and printed banner —
  never reportable as evidence of real-distribution behavior (hardening §10).
- Physics representation diagnostics evaluated separately for c_physics and
  a_goal (cross-sample pairwise cosine, effective rank, rank denominator), with a
  CASE_1/2/3 diagnosis kept separate from the structural flags: CASE_1 embedding
  collapsed, CASE_2 diverse-but-ignored, CASE_3 diverse-and-responded (hardening
  §16). All local runs diagnose CASE_3.

## N. Scale audit

Per report step: std(z_hat), std(z_y_raw), std(z_y_normalized), std(z_x),
std(mask_token+pos) (query init), mean_norm(z_x / z_hat / z_y_raw /
z_y_normalized). Real-data run: z_hat_std 0.53, z_y_raw_std 2.52,
z_y_normalized_std 1.00, z_x_std 4.00, mask_token+pos_std 0.54 — no gross
mismatch (max spread < 10x, normalized boundary exactly ~1). No projection added
to fix scale — measured only, per hardening §6.

**§8 rank interpretation:** D = 384, batch = 8, so the pooled-representation SVD
has at most min(batch, D) = 8 singular values; every effective-rank fraction is
printed alongside its rank denominator and must NOT be read as rank/384.

## O. Problems found (this pass)

1. Active code consumed the ambiguous `out["z_y"]` alias in three places
   (`src/train/engine.py` x2, `src/losses/objectives.py` x2, three eval scripts) —
   the hardening §2 rule risk. Fixed (P.1).
2. `model.train()` recursively re-enabled training mode on the frozen EMA target
   and released spectrum encoder — freezing params does not guarantee inference
   behavior (hardening §4).
3. No enforcement existed that frozen reference modules stay out of the optimizer
   (hardening §20) or that the architecture never references objectives
   (hardening §21).
4. Audit had no real-data mode, no rank denominator, no mask-token+pos scale, no
   physics case classification (hardening §8/§9/§11/§16).
5. Test-bug: a 0.5-valued patch flip is a no-op (1−0.5=0.5) and mask=1 means
   hidden — both inverted-probe errors in the new alignment test (fixed).

## P. Problems fixed (this pass)

1. All active consumers moved to `z_y_raw` explicitly (engine `evaluate`/
   `null_gap`, VICReg/Barlow/LeJEPA objectives, eval scripts, test stubs).
   `src/assembly.py` now only CREATES the alias (never consumes it); `model.loss`
   uses `z_y_raw`. Enforced by `test_h1_active_code_never_consumes_z_y_alias`.
2. Added `enforce_frozen_reference_modes()` + `train()` override + `forward()`
   call (hardening §4) with R1–R4 tests.
3. Added hardening §20 optimizer-exclusion test, §21 architecture-doesn't-know-
   objectives test, §22 no-hidden-projection test.
4. Audit: `--real-data` mode (val split + mask protocol + REAL DATA banner),
   `rank_denominator` on every eff-rank fraction, extended scale surface
   (z_y_normalized std ~1 gate, mask_token+pos), physics CASE_1/2/3 diagnosis,
   `data_mode` in JSON.
5. M1 strengthened (§13 fills), §14 channel-validity test, §17 full-model
   alignment test, §12 T1–T5 normalization tests, §6 deterministic EMA-order
   test, shape contract asserts for c_physics/a_goal/mask in forward.
6. Full suite after hardening: **185 passed, 6 skipped** (CUDA-only skips) in
   ~25s.

## Q. Remaining risks

1. **Real GPU audit gate (execution-pending by policy, not code):** per AGENTS.md
   compute environment, gradient-based runs execute on the cloud GPU
   (`CLOUD_TRAINING.md`). The §23 architecture-freeze gate:
   `python scripts/diagnostics/architecture_audit.py --config
   configs/milestone_b.yaml --steps 200 --batch 8 --device cuda` and the real-data
   variant with `--real-data` — both must pass on cloud GPU and be recorded here
   before objective experiments start. Until then the status below is FAIL per
   hardening §26.
2. Raw-L_J regime observations (logged, NOT defects): z_hat effective rank trends
   below z_y under the unregularized short run; VICReg/Barlow/LeJEPA not yet
   active. Per hardening §25/§38 these are objective-level questions and must not
   be "fixed" with more architecture.
3. a_goal/c_physics pairwise cosines start high at random init (mean-pooled
   summaries; goal queries near-zero at init); gated only on "not identical";
   real diversity is a trained-checkpoint question.
4. `out["z_y"]` compat alias remains in the model output; the h1 test guarantees
   no active consumer, but future objective code must follow the §2 rule.

## R. Final status

**FAIL — pending the real GPU architecture gate** (hardening §26: PASS only after
the real GPU gate passes; otherwise FAIL).

- All code-level and local-execution gates pass: 185 tests green; smoke audit,
  synthetic CPU audit (20 steps batch 8), and REAL DATA ARCHITECTURE AUDIT (CPU,
  20 steps batch 8, real released weights on the actual val split) all PASS with
  no structural flags (PHYSICS_PATH_FAILURE / TARGET_BUG / SCALE_PATHOLOGY all
  false) and physics diagnosis CASE_3.
- Freeze in effect (§28): the shared architecture is FROZEN as documented above;
  objective work proceeds separately on VICReg / Barlow / LeJEPA without backbone
  changes unless a test demonstrates a concrete architecture defect.
- Remaining action: run the two GPU audit commands above on the cloud GPU per
  `CLOUD_TRAINING.md`, append the JSONs, and flip this status to PASS.