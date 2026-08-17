# Milestone B — Vanilla Deterministic JEPA (build + local smoke)

Status: **code complete, local smoke-tested. Full training not yet run.** Per the
Compute Environment section of AGENTS.md, the actual §7.2 minimal experiment and the
20/40/60/80/100% sweep must be run on cloud GPU via `CLOUD_TRAINING.md`, and this
report's done criteria must be re-verified by the human operator from that run's
checkpoints before Milestone C starts. This report records what was built and what the
local (dev-only) smoke tests proved — it does not self-certify the research gate.

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
