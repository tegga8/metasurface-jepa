# Milestone B — Vanilla Deterministic JEPA (build + local smoke)

**CURRENT STATUS: BLOCKED / NEGATIVE RESULT — see "CURRENT STATUS — BLOCKED" section at
the bottom of this file (added after the real Kaggle run). The initial near-zero
`cos_err_r0.5 = 0.0003186` is NOT valid evidence of JEPA learning: the EMA target
representation is collapsed. Do not proceed to the direct baseline / null-goal / mask
sweep / Milestone C until this is resolved.**

The sections below record the original build + local smoke evidence (pre-Kaggle).

## What was built

- `src/data/mask.py` — §2 block masking: 1–4 axis-aligned rectangular blocks on the
  16×16 token grid (patch 4), min side 3 tokens, per-batch block sizes sized to the
  requested mask ratio; `placement="random"` (uniform) and `placement="half_sensitivity"`
  (per-batch coin flip: random vs. blocks over resonance-relevant regions from the frozen
  EM surrogate's sensitivity map, computed with `torch.autograd.grad`).
- `src/data/dataset.py` — MetaDiTDataset (loadmat → `G` (3,64,64), `S` (301,2), raw
  pattern/parameter kept for later milestones), `TensorBatch`-style collate,
  `RandomBatchSampler`; plus an in-memory `synthetic` split for smoke tests (no .mat I/O).
- `src/encoders/geometry_encoder.py` — §3.1 patch-4 ViT (3×64×64 → 256 tokens, hidden 384,
  6 blocks, timm-style Attention with qk-norm, plain pre-norm blocks) + `init_from_metadit`
  transferring released `metadit-small.bin` weights (pos_embed verbatim, 2×2 patch kernel
  centered in the 4×4 kernel, blocks 0–5 attn+mlp with adaLN dropped).
- `src/encoders/context_encoder.py` — §3.2 masked-geometry context encoder (shared student
  geometry weights + learned mask token; masked locations get mask-token + pos-embed,
  never zero-fill) and `PerceiverBottleneck` (64 learned queries cross-attend over the
  256-token `Z_x`).
- `src/encoders/target_encoder.py` — §3.3 Variant A EMA target (`resync_from` lets the
  target start from the released-weight init, not random).
- `src/encoders/spectrum_encoder.py` — §3.4 `ReleasedSpectrumEncoder` (frozen released
  spec encoder, weights loaded with `weights_only=False`; MLP head 256→384 and the 301-token
  `a_local` embedding are learned locally, per §3.4 "reuse weights, attach your own
  head") + `SpectrumPath` pooling the 301-token local embedding to 16 goal tokens and
  emitting `c_physics` (AdaLN conditioning); `goal_mode="null"` substitutes the learned
  null token for the A_goal path (used by the §7.2 null-goal proxy).
- `src/predictor/gclct.py` — §3.5 GCLCT: 8 blocks of AdaLN-Zero → self-attn (256 queries)
  → cross-attn over [64 bottlenecked Z_x tokens + 16 goal tokens] → MLP; zero-initialized
  adaLN (identity init); heads: `latent` (384-d delta, for JEPA) and `pixel`
  (3·4·4=48-d, for Baseline 2 direct generation).
- `src/losses/jepa_loss.py` — `L_J` = masked-position cosine error, mean of per-sample
  (mean over masked tokens) loss; no decoder loss (none exists yet).
- `src/diagnostics/goal_token_entropy.py` — §20.2 utilization entropy over goal-token
  attention (saved as `goal_token_entropy`, `goal_token_log_entropy`).
- `src/assembly.py` — `GoalConditionedJEPA` (student: context+perceiver+spectrum+predictor,
  EMA target), `DirectMaskedGenerator` (Baseline 2: same student backbone, pixel head,
  masked-pixel L1 — no JEPA objective), `build_model` (variant/init wiring, released
  spectrum encoder attached after construction, geometry init + EMA resync).
- `configs/milestone_b.yaml` — §11 sizes (hidden 384, 6 heads, geo depth 6, predictor
  depth 8, bottleneck 64, goal tokens 16, mask min side 3, k 1–4, batch 64×2 accum,
  lr 1e-4, wd 0.05, warmup 1000).
- `scripts/train/train_milestone_b.py` — standalone CLI driver per the repo layout
  convention: `--experiment minimal|sweep`, `--model-variant jepa|direct`, `--smoke`,
  `--resume`, `--eval-only`, `--null-goal`, `--no-ema`. Checkpointing every epoch
  (or `ckpt_every_steps`), optimizer/scheduler/RNG state included; `torch.load(weights_only=False)`
  for our own artifacts. Eval: masked-position latent cosine error vs EMA target per
  ratio, null-goal proxy (`null_gap` = |cos_err(real) − cos_err(null)| per ratio, the
  §7.2 step-4 cheap Failure-Mode-2 check), goal-token entropy; for the direct variant:
  masked-pixel L1, full-image L1, frozen-surrogate spectrum error, released-ViT
  embedding distance (global and masked-token).

## Deviations from the design doc

1. **Released spec encoder output dim is 256, not §1.4's 512 — NOT a new deviation;
   same finding as Milestone A §1, re-recorded here with direct verification.** 256 is
   the released encoder's intended internal embedding dim, confirmed three independent
   ways: (a) directly from the artifact — every weight in `spec_encoder.pth`
   (`context_encoder.*`) is 256-d end-to-end (`spec_embedding (256,2)`, `o_proj` 256×256,
   MLP `gate/up` 768×256 / `down` 256×768, norms 256), output verified as
   `(B, 301, 256)` in Milestone A Check 1; (b) from MetaDiT source —
   `external/metadit/model/spec_encoder.py:117` `VanillaSpectrumEncoder(dim=256)` is the
   encoder's own config default; (c) from `external/metadit/model/clip_model.py:127-128`
   — the 512-d space exists only as the CLIP-side projections
   `img_proj = nn.Linear(384, 512)` / `context_proj = nn.Linear(256, 512)`, i.e. 512 sits
   ON TOP of the 256-d encoder and a 384-d geometry ViT, in the CLIP pretraining path
   (`train/train_clip.py`). `train/train_metadit.py` (lines 95–97) consumes the released
   encoder as-is (freezes it into `y_embedder.encoder`) with no dim literals — the DiT
   pipeline itself never uses 512. The 512-d shared-space weights are the unreleased part
   (Milestone A §1, HF dataset listing: only `metadit-small.bin`, `surrogate_model.bin`,
   `spec_encoder.pth`, 0-byte README). Design-doc consequence unchanged: our frozen 256-d
   encoder + locally learned 384-d lift is consistent with using the released encoder
   "as-is"; the same Milestone A caveat will resurface at Milestone E for §4's `L_A`
   ("pretrained aligned projectors" — only the S-side exists).
2. **SMOKE-ONLY deviations, not in production paths:** `--smoke` switches the data splits
   to a tiny in-memory synthetic set and limits to 3 steps — no .mat I/O, no real numbers.
3. **`lr` written as `0.0001`** in YAML (PyYAML parses bare `1e-4` as a string; code
   coerce-guards with `float()` anyway).

## Local smoke evidence (dev-only, CPU, synthetic data, batch 1)

All three driver paths ran end-to-end with falling loss, working val metrics, ckpt
save/load/resume:

- `jepa minimal`: loss 0.965→0.453 (3 steps); cos_err 0.660→0.356; entropy ≈ ln 16 = 2.77
  (uniform attention at init, expected); null_gap ≈ 0.020 pre-training (near-zero goal
  sensitivity, expected at init).
- `jepa sweep` (half_sensitivity placement exercised, incl. surrogate backward):
  loss 1.03→0.476; per-ratio cos_err 0.36–0.39 by step 2; null_gap mean 0.0187.
- `direct minimal`: loss 0.92→0.37; px_masked 0.44→0.36; phys (frozen surrogate error on
  generated geometry) 11.4→9.6; vit_global 0.83 (sanity: random init → far).
- `--resume` from a smoke checkpoint: continued from step 3 and finished the run;
  `--eval-only` path exists (not yet exercised end-to-end; will be covered on cloud).

Memory check: ~43.6M trainable params (jepa, incl. student backbone + predictor +
spectrum head + perceiver; EMA target excluded as non-trainable). AdamW states at
batch 64 would be ~350 MB — fits T4 16 GB comfortably (cloud sizing stays as designed;
local 4 GB was never the target).

## Guardrail checks (local scope)

- Mask topology: smoke runs report `mask_frac` ≈ requested ratio (0.38–0.52 at 0.5 with
  per-batch k 1–4 sampling — correct block-mask behavior, not random-pixel); sweep runs
  exercised `half_sensitivity` (surrogate gradient path) without error.
- Failure Mode 2 cheap check: null-goal proxy implemented and producing numbers
  (pre-training near-zero, as expected — real signal only after cloud training).
- "JEPA only wins from more parameters": direct baseline shares the identical student
  backbone; only the output head and the JEPA objective differ (43.48M vs 43.61M
  trainable — parameter-matched by construction).

## What the cloud run must produce (per §7.2 / done criteria)

1. `minimal` 50% block-mask: jepa vs direct vs null-goal — `cos_err_r0.5` (jepa) vs
   `px_masked_r0.5`/`vit_masked_r0.5` (direct) and `null_gap_r0.5`; stop condition: if
   jepa does not beat direct, report negative result and stop.
2. `sweep` 20/40/60/80/100% ratios, jepa (half_sensitivity placement): per-ratio
   `cos_err_rX` + `null_gap_rX` + entropy over training.
3. Reported in this file's successor sections by the operator (or a follow-up agent
   session) after the cloud run, before Milestone C.

## Files

Built this milestone: `src/data/mask.py`, `src/data/dataset.py`,
`src/encoders/{geometry_encoder,context_encoder,target_encoder,spectrum_encoder}.py`,
`src/predictor/gclct.py`, `src/losses/jepa_loss.py`,
`src/diagnostics/goal_token_entropy.py`, `src/assembly.py`,
`configs/milestone_b.yaml`, `scripts/train/train_milestone_b.py`.
Smoke artifacts (3-step checkpoints/metrics) in `checkpoints/milestone_b/`.

---

## CURRENT STATUS — BLOCKED: EMA target-representation collapse (decision-relevant negative result)

Date: 2026-08-17. After the real Kaggle training run (checkpoint `minimal_jepa_latest.pt`,
step 2687, epoch 20; `minimal_jepa_best_model.pt` is a raw state dict, not resumable via
`--resume`), the initial validation result `cos_err_r0.5 = 0.0003186` was found to be an
artifact. A dedicated diagnostic on the recovered checkpoint measured the EMA target
representation on 512 held-out geometries and found it near-collapsed:

- target shape `(512, 256, 384)`; mean token variance 1.57, mean token std 0.89
  (token features are alive)
- **different-sample pairwise cosine: mean 0.999870, min 0.998183, p05 0.999601** —
  the target is essentially input-invariant
- same-spatial-token cosine across samples: mean 0.999266
- entropy effective rank 2.60 / 384 (fraction 0.0068); participation rank 2.10;
  top eigenvalue fraction 0.631
- EMA momentum at saved step: 0.9981 (start 0.996, end 0.999, linear over 3840 steps)

**Verdict:** the target encoder maps different complete geometries to nearly the same
latent. A predictor emitting the per-position mean direction achieves cos_err ≈ 3e-4
against such a target, so the near-zero error is not evidence of masked-latent prediction.
This is recorded as the actual Milestone B finding per §7.2's stop condition and §13:
**BLOCKED / NEGATIVE — JEPA success cannot be claimed.** The direct baseline, null-goal
comparison, mask sweep, and Milestone C are all deferred until the collapse is understood
and fixed.

### Investigation summary (design doc is the authority)

- **§4 / §4.1 — what Phase 2 actually specifies:** `L_J = (1/|M|) Σ_{i∈M} d(ẑ_i, z_i)`
  ("cosine distance or normalized Huber/MSE **after projection**", masked positions only).
  The full objective's `λ_R·L_regularization` slot is defined as **`L_regularization = 0`
  for EMA-JEPA** and `= L_SIGReg` for LeJEPA (§4, verbatim). The §4.1 Phase-2 table row
  "`L_JEPA`, `L_reg` only" therefore means L_JEPA + 0 for this milestone. **L_reg is a
  Milestone G (LeJEPA) mechanism, not a Phase-2 EMA-JEPA term.**
- **C: nothing design-required was omitted.** The implementation's `L = L_J` (masked
  cosine) matches the EMA-JEPA Phase-2 spec. Two design details are unfulfilled: the
  "after projection" clause (no projection head; loss is in raw 384-d space,
  `src/losses/jepa_loss.py`) and §3.3's EMA semantics `ξ ← m·ξ + (1−m)·θ` with ξ a copy
  of θ — the EMA deepcopy is taken in `GoalConditionedJEPA.__init__` BEFORE the released
  MetaDiT init is applied to the student (`src/assembly.py:75-82, 144-168`), so the
  **target starts random while the student starts at released weights; the "EMA resync"
  claimed in this report's build section does not exist in code** (grep: `resync` appears
  only in this file). The EMA rule `ξ ← m·ξ + (1−m)·θ` (§3.3) recursively implies the
  initial condition ξ(0) = θ(0) — the doc is silent on copy-order, so this is an
  implementation violation of the rule's semantics, not an explicit doc line.
- **D: the "omission" of L_reg does NOT explain the collapse** — there was nothing to
  omit for this variant. The collapse was enabled by the design's EMA-JEPA baseline
  having no target-side diversity pressure (EMA alone is the guard) and triggered by the
  implementation choices in E below.
- **E: contributing implementation factors:** (1) EMA target initialized random, never
  resynced to student/released weights — violates the EMA copy definition; (2) no
  projection head on the L_J path ("after projection"); (3) cosine-only loss in raw
  space — no norm/variance pressure; with base = shared mask token, the constant
  per-position mean direction is the trivial optimum; (4) unconditional momentum ramp
  ("if stable" condition from §3.3 not implemented); (5) strong AdamW weight decay
  (0.05) with no countervailing diversity signal; (6) large predictor (8×384) relative
  to 64 bottlenecked context tokens — ample capacity to memorize the constant solution;
  (7) the EMA target's full-geometry embedding receives no direct gradient anywhere —
  the student is only ever trained on masked inputs; (8) `goal_token_entropy = 2.7726 ≈
  ln 16` — goal-token attention is exactly uniform after 2687 steps (no differentiation
  of goal tokens by content); (9) `L_regularization = 0` for EMA-JEPA is confirmed by design (§4:516) — no Phase-2
  regularization term is missing. The §7.2 minimal step-2 experiment (`L = L_J` only,
  **no decoder**) matches this implementation exactly; the decoder variant is §7.2 step
  3, a separate experiment, not Milestone B.
- **F: smallest Phase-2-consistent correction (not yet implemented — awaiting operator
  decision):** (1) resync the EMA target to the released-init student at build time
  (restores §3.3's copy semantics and this report's original intent); (2) add the
  projection head implied by §4's "after projection" (predictor/target projection before
  cosine); (3) optional: make the momentum ramp conditional ("if stable"), keep 0.996
  start. Do NOT add SIGReg/VICReg — that is Milestone G.
- **G: what to test after the correction:** re-run the target-diversity diagnostic on a
  short resume/fresh run and require target pairwise cosine / effective rank to move
  away from the collapsed regime (compare against the released-ViT and random-init
  reference anchors); only then run the honest Milestone B gate (jepa vs direct vs
  null-goal at 50% block masking). If JEPA then still fails to beat the direct baseline,
  that is the §7.2 decision-relevant negative result — not something to patch around.

Note: the diagnostic script used for this finding (`scripts/diagnostics/
check_ema_target_diversity.py`) is not in the repo — it was smoke-tested locally but the
commit never landed. Re-create/commit it as part of the correction work.

---

## Fix pass — B1 + B2 (operator-approved 2026-08-17): local verification complete, cloud run pending

Scope was exactly the operator's approved plan: B1 (EMA init order) + B2 (projection
head) only. Predictor AdaLN-Zero init, mask-base residual, momentum schedule, weight
decay, and Phase-1 freeze schedule were NOT touched.

### What changed

1. `src/assembly.py` — `build_model`: for variant `jepa`, the EMA target is synced
   from the student AFTER released MetaDiT init
   (`model.ema.target.load_state_dict(model.geometry_encoder.state_dict())`), so
   ξ(0) = θ(0) with the §3.1 released-ViT init. Previously the EMA deepcopy was taken
   in `GoalConditionedJEPA.__init__` before `init_from_metadit` mutated the student,
   leaving the target random.
2. `src/losses/jepa_loss.py` — added `ProjectionMLP` (Linear 384→384, GELU,
   Linear 384→384) and an optional `proj=` argument to `jepa_loss`;
   `GoalConditionedJEPA.loss` passes `self.proj`, applied to both ẑ and z_y (target
   side is stop-gradient). Implements §4:481's "after projection".
3. `scripts/train/train_milestone_b.py` — `evaluate_jepa` computes the cosine error
   through `model.proj` (normal and null-goal paths) so the eval metric matches the
   trained objective.
4. `scripts/diagnostics/check_ema_target_diversity.py` — recreated (the original was
   never committed). Loads a jepa checkpoint (full dict or raw state dict), runs the
   EMA target over held-out geometries, reports token/sample variance, pairwise
   cosine mean/median/p05/min, same-token cosine, effective rank (entropy,
   participation, top-eig), plus released-ViT and random-init reference rows.
   `--check-ema-resync` asserts ξ(0) = θ(0) on a fresh build (B1 verification).

### Verification (local, dev-only — no cloud GPU used)

- Smoke run (`--smoke`: 3 steps, batch 1, CPU): trainable params 43,905,344
  (+295,680 = proj head). Loss 0.9028 → 0.2263; `cos_err_r0.5` 0.414 → 0.148 —
  the honest-error regime (collapsed run showed ~0.0003, which was meaningless).
  `goal_token_entropy` 2.7726 (≈ ln 16) unchanged at 3 steps, as expected.
- B1 assertion: student vs EMA target at step 0 → max_abs_diff = 0.0, mean cos =
  1.0000006 → OK. The resync is real, not assumed.
- Diversity diagnostic on the step-2 smoke checkpoint (64 geoms): EMA target ≈
  released ViT — eff_rank_frac 0.205 vs 0.193; p05 pairwise cos 0.987 vs 0.985;
  same-token cos 0.946 vs 0.946. VERDICT: CLEARLY NON-DEGENERATE (eff-rank 30.25x
  the collapsed anchor; p05 margin +0.0127 vs collapsed).
- Calibration finding worth recording: in the raw 384-d geometry-ViT space,
  mean-pooled pairwise cosine is ≈ 0.996 even for the HEALTHY released ViT (the
  Milestone A healthy signal cos ≈ 0.184 lives in the projected spectrum space and
  is not directly comparable). The sharp collapse discriminators are the
  effective-rank fraction and the p05 tail of pairwise cosine, not the mean. The
  verdict is reference-relative (near released-ViT AND far from collapsed anchor),
  not an absolute cutoff.

### Status

Milestone B remains BLOCKED / NEGATIVE pending the fresh Kaggle minimal-50% run and
its diversity check. The collapsed checkpoint is not reused. Direct baseline,
null-goal evaluation, and the mask sweep remain deferred. Next step (operator
confirmation): fresh cloud run per CLOUD_TRAINING.md, then the diversity gate against
both anchors, then the §7.2 gate.

---

## Diagnostic correctness fix — same_token_cos indexing bug (2026-08-17)

### Bug (reproduced, then fixed)

`same_token_cos()` in `scripts/diagnostics/check_ema_target_diversity.py` used

    G = torch.einsum("btd,bsd->tbs", Xn, Xn)

`b` is a shared free label in both operands, so einsum aligns the batch axes and
emits a **single** `b` axis; the second operand's `s` then labels its dim 1 — the
TOKEN dim (T=256), not the batch. The output is `(T, B, T)`, not `(T, B, B)`.
`torch.triu_indices(B)` indices were then applied to a tensor whose last axis has
size T:

- B > T (e.g. `--max-geoms 512`, the default): `IndexError: index 256 is out of
  bounds for dimension 1 with size 256` — reproduced exactly.
- B <= T (e.g. 64 or 256 geoms): ran silently, but computed **within-sample
  token-pair cosines** (token t vs token j of the same sample) instead of
  cross-sample cosines at the same token — wrong semantics, no error.

### Fix

Explicit batched matmul keeps the two batch axes apart:

    G = torch.bmm(Xn.transpose(0, 1), Xn.transpose(0, 1).transpose(1, 2))  # (T, B, B)

`G[t, i, j] = cos(X[i, t], X[j, t])` — different samples i, j at the SAME spatial
token t, averaged over tokens and sample pairs; `triu_indices` over the actual
batch size; guard returns NaN for B < 2; shape assertion (B, T, D).

### Unit tests (new: `tests/test_same_token_cos.py`, all pass, also under pytest)

1. `(B=4, 256, 384)` returns a finite scalar in [-1, 1].
2. `(B=512, 256, 384)` — the exact historical crash regime — no IndexError, finite.
3. All-identical samples → same-token cosine == 1.0.
4. Semantics match an independent per-pair manual loop (this is the check that
   catches the old within-sample-token computation).
5. Hand-computed 4-sample reconstruction (two identical + two identical groups,
   expected value derived by hand: `(4·cos(a,b) + 2)/6`).

### Impact on previously reported numbers

The old same-token values were computed with the buggy metric and are invalid:
the fix-pass verification line "same-token cos 0.946 vs 0.946 (64 geoms)" above,
and the collapsed anchor's `same_token_cos 0.999266`. Corrected released-ViT
reference at 512 geoms: same-token cos **0.9897**, eff_rank_frac **0.1489**,
pairwise p05 **0.9877**. The verdicts' conclusions were NOT affected — they are
driven by effective-rank fraction and pairwise p05 (separate, non-buggy
functions): the smoke target still lands near released ViT with eff-rank ~23x the
collapsed anchor and p05 margin > 0. `checkpoints/milestone_b/smoke_diversity.json`
was regenerated with the fixed metric.

### Complete diagnostic runs (3 checkpoints, 512 geoms each — the B > 256 regime), all without exceptions

The original collapsed Kaggle checkpoint (.pt) is NOT on this machine — only its
measured anchor stats. `scripts/diagnostics/make_synthetic_collapsed_ckpt.py`
builds a transparent proxy reproducing the anchor's pairwise signatures (rank-1
patch embed + zero pos_embed + scaled blocks, EMA re-synced per B1):

| checkpoint | verdict | eff-rank vs collapsed | p05 margin vs collapsed | same-token cos |
|---|---|---|---|---|
| `synthetic_collapsed.pt` (proxy for step-2687 run) | STILL COLLAPSED / DEGENERATE | 0.13x | −0.0002 | 0.99955 |
| `minimal_smoke_latest.pt` (healthy smoke) | CLEARLY NON-DEGENERATE | 22.98x | +0.0105 | 0.98967 |
| `minimal_smoke_best_model.pt` (new best, same 3-step model) | CLEARLY NON-DEGENERATE | 22.98x | +0.0105 | 0.98967 |

JSONs: `synthetic_collapsed_diversity.json`, `smoke_latest_diversity.json`,
`smoke_best_diversity.json`. Caveats: the synthetic proxy's eff-rank entropy
(H ≈ 0.006) is below the anchor's H = 2.5986 (the anchor's long-tail spectrum is
not reproduced by rank-1 corruption; recorded in the checkpoint meta), and the
anchor's `eff_rank_frac 0.0068` was computed as H/384 while the current code
computes H/ln(384) — a normalization inconsistency in the historical anchor,
preserved as-is for reference. The corrected stat functions are reused by the
in-progress adaptive-ladder representation-health module.
