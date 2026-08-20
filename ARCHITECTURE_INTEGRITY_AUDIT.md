# ARCHITECTURE INTEGRITY AUDIT — pre-cleanup snapshot

Snapshot taken BEFORE the architecture-only integrity repair pass. Records what was
found in the working tree at commit `cec984d`, what the active architecture looked
like, and the gaps identified against the repair spec. The FINAL state of this pass
is reported in `FINAL_ARCHITECTURE_REPORT.md`.

## 1. Working tree state

- Branch: `main`, HEAD `cec984d` "Add comprehensive tests for Barlow and LeJEPA
  objectives and massive cleanup". Working tree clean (no uncommitted changes).
- Active architecture files inspected: `src/assembly.py`, `src/encoders/{geometry,
  context,target,spectrum}_encoder.py`, `src/predictor/gclct.py`,
  `src/data/mask.py`, `src/data/dataset.py`, `src/reference/direct_masked_generator.py`.
- Forbidden-pattern search over `src/`: `Perceiver|perceiver|bottleneck|base_delta|
  base+delta|model.proj|model_proj` — matches are PROSE ONLY (docstrings asserting
  absence). `model.proj` itself does not exist. The only `.proj` modules are
  standard attention output projections (`Attention.proj`, `CrossAttention.proj`)
  and the spectrum-path goal-pooling projection — not a shared model projection.

## 2. Active architecture (as found)

- Geometry encoder: patch-4 ViT, 3x64x64 -> 256 tokens x 384-D, 6 pre-norm blocks.
- Context encoder: shared geometry encoder + learned mask token (masked positions
  -> mask_token + pos; never zero-fill), all 256 tokens, no bottleneck.
- EMA target: `deepcopy` of the geometry encoder, frozen (`requires_grad=False`),
  momentum 0.996 -> 0.999, updated after optimizer.step by the training loop.
- Spectrum path: frozen released `VanillaSpectrumEncoder` -> `a_local (B,301,256)`;
  `c_physics (B,384)` = proj_g(mean over 301); `a_goal (B,16,384)` = 16 learned
  queries cross-attending over a_local. `goal_mode='null'` zeroes both.
- Predictor: GCLCT depth 8, dense attention only; every block = affine-less LN ->
  FiLM(c_physics) -> self-attn -> cross-attn(kv=[z_x; a_goal]) -> MLP, residual
  summed per sublayer; final affine-less LN + Linear(384->384). FiLM cond
  `SiLU -> Linear(384, 6*384)` zero-initialized.
- Output dict today: `z_hat, z_x, mask, attn_weights, z_y`. `c_physics`/`a_goal`
  are computed inside `_encode` but NOT exposed.
- `query_predictions` accessor exists (unused by active scripts; kept for the
  Milestone-E counterfactual interface).
- `src/reference/direct_masked_generator.py` is self-contained, imports FROM the
  active module (never the reverse), and is not importable by the active model.

## 3. Gaps found (to fix in this pass)

1. **No explicit target-normalization boundary** (spec §10): model exposes only raw
   `z_y`; no `z_y_normalized = F.layer_norm(z_y_raw, (384,))`. Must expose both,
   without a learnable projector and without overwriting the raw target.
2. **Output contract incomplete** (spec §30): `c_physics` and `a_goal` are not
   exposed by `forward`.
3. **No architecture-masking tests** (spec §6 M1–M4, §21 leakage, §22 alignment).
4. **No consolidated architecture test file** (spec §35 A2–A20); existing coverage
   lives in `test_predictor_conditioning.py` (C-sensitivity/Route tests) and tier2/3
   (engine/checkpoint), but no EMA-ownership, shared-encoder-identity,
   target-no-gradient, physics-diversity, scale-audit, or source-integrity tests.
5. **No short raw-JEPA architecture audit runner** (spec §32).
6. **No final architecture report** (spec §36/§37).

## 4. Measurements already taken

- **§25 residual-scale drift** (pre-norm stack, batch 8, hidden 384, depth 6,
  random init): post-embed std 2.008 -> 2.143 after 6 blocks (+6.7%); mean token
  norm 40.27 -> 43.24 (+7.4%). Mild, monotonic growth; NOT "substantial" per §25.
  Decision: do NOT add a final affine-less geometry-encoder normalization; the
  §10 target-normalization boundary is added independently for the I-JEPA-style
  masked-target regression boundary.
- **EMA/student init equality**: `build_model` copies student -> EMA target after
  MetaDiT init (`model.ema.target.load_state_dict(model.geometry_encoder.state_dict())`),
  so A9 holds at build time.
- **Shared encoder identity**: `model.context_encoder.geo is model.geometry_encoder`
  is True by construction; A6 holds.

## 5. Cleanup decisions

- `DirectMaskedGenerator` stays in `src/reference/` (explicit legacy location,
  not importable by the active model — satisfies spec §34).
- No `model.proj`, no Perceiver, no bottleneck, no base+delta in the active path —
  already true; the new tests lock this.
- `query_predictions` retained (documented accessor, not dead architecture state).