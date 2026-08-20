# FINAL ARCHITECTURE REPORT — architecture-only integrity repair pass

Date: 2026-08-20. Scope: architecture only (no VICReg/Barlow/LeJEPA mathematics,
no objective coefficients, no winner selection, no long training). Pre-cleanup
state is snapshotted in `ARCHITECTURE_INTEGRITY_AUDIT.md`; this report records the
final state.

---

## B. Tensor shapes

| tensor | shape | produced by | notes |
|---|---|---|---|
| `G` | (B, 3, 64, 64) | dataset | pixel geometry |
| `G_c` | (B, 3, 64, 64) | `apply_mask_to_pixels` | pixel-space masked copy |
| `M` | (B, 16, 16) | `BlockMasker` | 1 = visible, 0 = masked |
| `z_x` | (B, 256, 384) | context encoder | full 256-token context, no bottleneck |
| `queries` | (B, 256, 384) | `torch.where(mask, mask_token+pos, z_x)` | masked queries = mask token + OWN pos |
| `kv` | (B, 272, 384) | `cat([z_x, a_goal], 1)` | 256 context + 16 goal tokens |
| `a_local` | (B, 301, 256) | frozen released spectrum encoder | eval(), params excluded from optimizer |
| `c_physics` | (B, 384) | `proj_g(mean_301(a_local))` | global FiLM condition |
| `a_goal` | (B, 16, 384) | 16 learned queries over a_local -> `proj_goal` | structured goal tokens |
| `z_hat` | (B, 256, 384) | GCLCT predictor | final affine-less LN + Linear(384->384) |
| `z_y_raw` | (B, 256, 384) | EMA target encoder | `z_y` alias kept for backward compat |
| `z_y_normalized` | (B, 256, 384) | `F.layer_norm(z_y_raw, (384,))` | feature-wise, no learnable params |
| `mask` (out) | (B, 256) | `(M.view(B,-1) == 0)` | row-major, 1 = masked |
| `attn_weights` | optional | per-block cross-attn weights | `need_attn=True` only |

`model.forward` now exposes the full §30 contract: `z_x, z_hat, mask, attn_weights,
c_physics, a_goal, z_y(=raw), z_y_raw, z_y_normalized`.

## C. Masking

- Block masking per §2: 1–4 axis-aligned blocks, min side 3 tokens, random or
  half-sensitivity placement (`src/data/mask.py`, unchanged this pass).
- Pixel-space masking (`apply_mask_to_pixels` = M upsample 4x per axis, multiply)
  then masked patch embeddings replaced by learned `mask_token + pos` — never
  zero-fill semantics, no patch-boundary leakage (patch size 4 == mask cell size 4).
- Locked by `tests/test_architecture_masking.py`: M1 masked-value invariance
  (bit-identical z_x for arbitrary masked content — this is also the §21 leakage
  guard), M2 visible sensitivity, M3 mask sensitivity, M4 alignment (patch (r,c)
  -> token r*16+c in context token, predictor query, EMA target token, and loss
  selection; loss ignores errors at unmasked tokens).
- `jepa_loss` averages masked positions ONLY (masked-only contract, raises on
  empty mask).

## D. Geometry encoder

- Patch-4 ViT, 3x64x64 -> 256 x 384, 6 pre-norm blocks (affine-less LN, eps 1e-6,
  qk-norm attention, GELU-tanh MLP). Unchanged this pass.
- §25 residual-scale drift measured (batch 8, depth 6): post-embed std 2.008 ->
  2.143 (+6.7%), mean norm 40.3 -> 43.2 (+7.4%). Not substantial -> NO final
  affine-less geometry-encoder norm added (spec §25 authorizes it only if growth
  is substantial).
- MetaDiT init (`init_from_metadit`) verified strict: pos_embed shape assert,
  patch kernel center-initialization, per-block `load_state_dict(strict=True)`.
  Untouched this pass.

## E. EMA target

- `EMAEncoder`: deepcopy of the shared geometry encoder, all params frozen,
  momentum 0.996 -> 0.999 linear ramp. Updated ONLY after `optimizer.step()`
  (audit script and `train_milestone_b.py` via `objective.on_optimizer_step`).
- Invariants locked by tests: forward never touches the target (A10),
  update moves target toward student and leaves it frozen (A10), no gradient ever
  reaches target params (A11), target absent from optimizer param groups (A8).
- Student == target at build time (A9) — `build_model` re-syncs after MetaDiT init.

## F. Physics encoder

- Frozen released `VanillaSpectrumEncoder` (eval mode; `requires_grad_(False)`).
  `SpectrumPath` keeps `released` OUT of the optimizer; only `proj_g`, `proj_goal`,
  goal queries, q/kv/proj pooling weights are trainable (verified A16: gradients
  reach all of them after the FiLM zero-init thaws).
- c_physics = mean-pooled summary (coarse global); a_goal = structured tokens
  (fine spectral detail). Roles documented in `spectrum_encoder.py` docstring.

## G. Physics conditioning

- Route B: `c_physics` FiLM-modulates EVERY predictor block (6 groups
  gamma1/beta1..gamma3/beta3 across self-attn/cross-attn/MLP sublayers).
- Route A: `a_goal` retained in the cross-attention kv sequence.
- `goal_mode='null'` zeroes both (Failure-Mode-2 cheap proxy).
- Physics NEVER enters the target encoder: `z_y` is `ema(G)` only (§19).
- A14: every block's cond has nonzero weight+bias grad after one backward.
- A15: real vs null and real vs shuffled spectra change the predictions.
- §20 C1–C5 tests exist as A14/A15/A16 + `test_predictor_conditioning.py` A–E.

## H. Predictor

- GCLCT depth 8, dense attention only (no sparse routing, no CFG — those are
  Milestones E/D). Final affine-less LN + Linear(384->384). No objective
  mathematics anywhere in the predictor (§29).
- A3/A4/A5: no Perceiver, no pooling, no bottleneck, no shared model.proj, no
  base+delta parameters (structural + state-dict checks).

## I. Initialization

- FiLM cond zero-initialized (identity modulation at step 0); verified A9:
  `cond(c) == 0` for ANY c, and the predictor head keeps its normal (non-zero)
  init — no accidental constant-zero parameters outside the FiLM output layer.
- All parameters finite at init (A9). Student/target identical at init.
- MetaDiT init path unchanged (spec §4: not altered unless a test demonstrates a
  mismatch — none did).

## J. Gradient ownership

- EMA target: never in optimizer, never receives gradients (A8, A11).
- Frozen released spectrum encoder: eval, excluded from optimizer and from
  checkpoints (SAVED_EXCLUDES).
- Trainable spectrum pooling: receives gradients (A16, after activation).
- Geometry/predictor/FiLM: all receive gradients (audit §32 grad norms > 0).

## K. Architecture deviations from canonical I-JEPA

1. Dense 256-token masked-placeholder context encoder (canonical I-JEPA encodes
   only visible patches; predictor queries carry mask tokens). Deliberate,
   documented project choice: preserves all-256-token alignment for later
   geometry reconstruction; verified leak-free (M1/§21: masked content never
   enters z_x) — the spec's condition for keeping it.
2. Physics-conditioned predictor (FiLM + structured goal tokens) — this is the
   project's whole point, absent from canonical I-JEPA.
3. Explicit feature-wise target normalization boundary (`z_y_normalized`,
   spec §10) mirroring the official I-JEPA target LayerNorm practice, while the
   raw target remains accessible for objective-space mapping.
4. No hierarchical multi-block prediction (deferred), no predictor projection
   head beyond the final Linear back to encoder dim.

## L. Tests

- NEW `tests/test_architecture_masking.py`: 9 tests (M1–M4, §21 leakage, §22
  unique-identifier alignment, loss-selection ordering).
- NEW `tests/test_architecture.py`: A2–A20 (+ A9/A10/A16 extra cases, §15
  sequence, §20 C1–C5), 26 tests.
- Full suite: **168 passed, 6 skipped** (skips = CUDA-only tests on the local
  CPU machine) in ~19s. Existing engine/checkpoint/objective tests untouched and
  still green (forward-output contract extended additively: `z_y` remains the raw
  alias; new keys are additions).

## M. Problems found

1. Model output contract incomplete vs §30: `c_physics`/`a_goal` computed but
   discarded; no explicit target-normalization boundary (§10).
2. No masking-integrity tests existed (M1–M4/§21/§22), despite the mask-topology
   being the experiment's core validity guard.
3. No consolidated architecture test file; EMA-ownership and physics-in-target
   invariants were only implicit.
4. No short raw-JEPA architecture audit runner (§32).
5. First audit implementation flagged PREDICTOR low-rank as architecture collapse:
   artifact of pooled-token effective rank (2048 correlated rows); switched to the
   repo's per-sample mean-pooled convention, consistent with
   `representation_health`.
6. Audit grad-norms initially read post-`zero_grad` (always 0) — capture moved
   before `opt.step()`.

## N. Problems fixed

1. `src/assembly.py`: `_encode` now returns `c_physics, a_goal`; `forward`
   exposes the full §30 contract including `z_y_raw` and `z_y_normalized`
   (`F.layer_norm(z_y_raw, (384,))`, no learnable params, raw never overwritten).
   `z_y` kept as raw alias — all downstream consumers unchanged.
2. Added `tests/test_architecture_masking.py` (M1–M4, §21, §22).
3. Added `tests/test_architecture.py` (A2–A20) locking every §37 code-level
   invariant.
4. Added `scripts/diagnostics/architecture_audit.py` (§32 runner; `--smoke` for
   local CPU, full mode for cloud GPU; fixed seeds/masks; structural flags +
   raw-JEPA regime observations).
5. Audit eff-rank metric fixed to per-sample mean-pooled convention.
6. Audit grad-norm capture fixed (before optimizer step).

## O. Remaining risks

1. **§32 GPU audit execution**: the audit PASSED on local CPU with real released
   weights (20 steps, batch 8: z_y eff_rank_frac 0.79->0.77 healthy, physics
   sensitivity ~1.0, gradients on all paths, no structural flags). The done
   criteria item "short GPU architecture audit passes" is the same script on a
   cloud GPU (`--steps 200 --device cuda`), pending per AGENTS.md compute
   environment (dated operator decision: cloud for all gradient-based runs).
2. Raw-L_J regime observation (logged, NOT a structural defect): z_hat effective
   rank trends down (0.56 -> 0.24 over 20 steps) while z_y stays ~0.77 — the
   known raw-JEPA predictor compression that the objective mechanisms
   (VICReg/Barlow/LeJEPA) exist to counter. Per §38 this must NOT be fixed with
   more architecture.
3. c_physics/a_goal diversity at random init is modest (mean-pooled summaries;
   goal queries near-zero at init). The audit/`goal_token_entropy` monitor this
   per-step; real diversity is a trained checkpoint question (Milestone B runs).
4. a_goal pairwise cosine 0.98–0.99 at init (attention near-uniform): expected
   random-init regime, gated only on "not identical" in tests; monitor on real
   training.

## P. Final status

**PASS** (architecture code-level + local real-weights structural audit).

- All §37 code-level items pass: 256x384 preserved; no Perceiver/bottleneck/
  base+delta/model.proj; single shared student geometry encoder; EMA frozen,
  outside optimizer, updated only after optimizer.step; explicit
  `z_y_normalized` boundary with raw `z_y_raw` retained; predictor output raw
  384-D; c_physics reaches every block; a_goal structured; physics embeddings
  sample-specific; condition-sensitivity and condition-gradient tests pass after
  activation; physics absent from target; spatial indices aligned (unique-
  identifier test); init finite/deterministic (FiLM zero excepted); scale audit
  passes; 168 tests green.
- One §37 item remains EXECUTION-pending by policy, not code: "short GPU
  architecture audit passes" — run
  `python scripts/diagnostics/architecture_audit.py --config configs/milestone_b.yaml
  --steps 200 --batch 8 --device cuda` on the cloud GPU per `CLOUD_TRAINING.md`
  and record the JSON here before the objective phase starts.