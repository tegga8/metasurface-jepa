# Unified Occupancy–Parameter–Spectrum JEPA
## Architecture & Implementation Plan (v5)

Status: design document, not yet implemented. Grounded against the actual repository
(`tegga8/metasurface-jepa`) as of this writing and against the MetaDiT paper/dataset
convention. No code changes are made by this document; it is the reference an
implementation pass should follow.

**v3 changes:** incorporated a red-team pass that found one internal contradiction
(scalar EMA vs. scalar loss), five concrete robustness fixes, and a sequencing
recommendation, plus a set of collapse-mode checks raised in follow-up review. §0.1
holds the sequencing decision; §8 held the (then-new) gate list.

**v4 changes:** a second pass caught four places where the v3 fixes hadn't been
swept through every section describing the same mechanism (§6.3, §9 ×2, §10 cross-
reference) — fixed below, same content, no new decisions. One substantive upgrade:
checks 8 and 9 in §8.3 tested sensitivity ("does output change") rather than
necessity ("does it change correctly") — both now use the same real/null/shuffled
comparison already validated for the spectrum path earlier in this project, not a
weaker two-way version. One implementation note added to §3.5 on `GCLCT`'s width
assumption. No architectural decisions changed from v3 — this is a consistency and
rigor pass, not a redesign.

**v5 changes:** two substantive fixes, both real gaps rather than drift. (1) §0.2
(new) states explicitly why the target latent excludes physics and what that
implies — the JEPA loss can only *reward*, never *require*, spectrum/scalar
dependence, because "correct" is defined without reference to either. (2) §8.3
checks 8 and 9 are now stratified by masking regime instead of pooled — a pooled
real/null/shuffled gap can look healthy while genuine physics-dependence is absent
specifically in the pure-inverse-design stratum, the one case the redesign exists to
serve; gated explicitly on the hard stratum, not the easy one. (3) §3.6 adds a
second, narrowly-scoped EMA (`scalar_mlp_ema`) for target-side FiLM conditioning
only — the previous drafts left unspecified whether the target encoder's scalar
conditioning came from a stable or a live-training source, which would have made the
"stable answer key" property of the EMA target silently false. (4) Corrected the
framing of the scalar-FiLM shortcut in check 9: it is not a *new or worse* leak than
the original merged-channel leakage (which was already maximally cheap), it is a
*more reliably available* instance of the same underlying cause — the target
excluding physics — so removing FiLM alone would not fix it.

---

## 0.1 Sequencing decision — read this before building anything in this document

**This architecture is conditional, not immediate.** The dimension change (384→192),
single-channel occupancy input, and new init path mean every piece of representation-
health evidence gathered on the existing encoder — collapse diagnostics, VICReg
gradient attribution, the within-bucket spatial-structure probe — would need to be
re-run from scratch against a differently-shaped encoder. That is a larger cost than
anything else currently scoped in this project, including the decoder work already
agreed as the immediate next step.

**Recommended sequencing:**
1. Finish the decoder-first plan on the **existing** encoder (already decided, cheap,
   reuses everything as-is).
2. While doing so, specifically check the "partial-parameter conditioning" scenario
   (§7) for symptoms attributable to scalar leakage: completions that look
   shortcut-driven, or that get *worse specifically when scalars are masked* relative
   to when they're given (a gap larger than expected noise).
3. **If that symptom shows up**, this document is the designed, ready fix — proceed
   with it, starting at §1.
4. **If it doesn't show up**, the cheaper path has been validated first, and this
   redesign is deferred until there is demonstrated evidence it's needed rather than
   a sound-but-unconfirmed theoretical concern.

Everything below assumes step 3 has been triggered.

## 0.2 Why the target latent excludes physics — stated explicitly, not left implicit

This is a deliberate design choice with a real consequence, and it's easy to
mistake for an oversight if it isn't spelled out. Two latent spaces exist:

- **Target latent (`z_y_raw`, §3.6)**: built from the complete, true occupancy and
  complete, true scalars only. Never sees the spectrum.
- **Predicted latent (`ẑ_hat`, §3.5)**: built from whatever's visible, fused with
  the target spectrum, via the predictor.

The target is kept spectrum-free on purpose: its job is to be a single, stable
definition of "correct geometry" that doesn't change depending on which spectrum
happens to be conditioning the predictor. If the target shifted per spectrum, there
would be no one fixed thing to regress toward — the entire EMA-target mechanism
depends on there being a stable answer key. The true spectrum is also fully
determined by the true occupancy and scalars already, so including it in the target
would add no new information, only a dependency that breaks stability.

**The consequence, not just the justification:** because the target never depends on
spectrum, the JEPA loss can only ever *reward* the predictor for using the spectrum
when doing so happens to help reach that target — it can never *require* spectrum use,
because "correct" was defined without reference to it. This is why §8.3 checks 8 and
9 must be stratified by masking regime rather than pooled: the loss only forces
genuine spectrum (or scalar) dependence in the regime where nothing else available is
sufficient, and a model can satisfy the loss everywhere else without ever developing
that dependence.

---

## 0. Problem statement, stated precisely

A sample is exactly:

```
occupancy  M(x,y) ∈ {0,1}^{64×64}      — binary meta-atom placement
l_lattice  ∈ R                          — unit-cell period (global, one value/sample)
h_atom     ∈ R                          — atom thickness  (global, one value/sample)
r_atom     ∈ R                          — atom refractive index (global, one value/sample)
spectrum   (real[301], imag[301])       — the resulting optical response
```

`l_lattice`, `h_atom`, `r_atom` are NOT independent per-pixel fields — they are single
scalars that jointly set the boundary conditions of the same electromagnetic resonance
problem that the occupancy pattern's shape then modulates. The repository's current
`[3,64,64]` tensor is a *broadcast encoding* of this: channel 0 = occupancy × r_atom/5,
channel 1 = occupancy × h_atom, channel 2 = l_lattice/3 everywhere — confirmed both from
`src/data/dataset.py` and from MetaDiT's own paper, which uses the same three-channel
convention. This is not a bug; it is how the field packages the object for CNN input.
It is, however, the wrong representation to *mask and predict against*, because it
conflates two independent things: "is the shape known here" (local, spatial) and "is the
scalar known at all" (global, all-or-nothing).

**The actual problem worth solving** (established over this project's prior discussion,
and consistent with the field's own stated gap — MetaDiT explicitly frames prior work as
generating only a subset of design parameters, not a flexible arbitrary subset): given an
arbitrary combination of known/unknown occupancy region, known/unknown scalar values, and
a known target spectrum, produce the missing pieces such that the resulting design's true
spectrum (checked via the frozen physics surrogate) matches the target. This covers, as
special cases:

- **Pure inverse design**: spectrum known, occupancy + all scalars unknown.
- **Retrofit / constrained editing**: some occupancy region and some/all scalars fixed
  (e.g. fabrication constraints), rest generated to hit a new target spectrum.
- **Metrology / parameter estimation**: occupancy known (e.g. from imaging), scalars
  unknown, recovered against a measured spectrum.

---

## 1. What changed from earlier drafts, and why

Three prior iterations of this design were discussed and rejected in favor of this one.
Recorded here so the reasoning isn't silently lost:

1. **Full 3-channel image, patch-masked as-is** (original Milestone-B design) — rejected
   because masking cannot create genuine scalar uncertainty this way: any visible occupied
   pixel already reveals the exact scalar values, and any visible pixel at all reveals
   `l_lattice` (constant everywhere). Spatial masking and scalar masking need independent
   control, which this representation cannot give.
2. **Three separate scalar tokens + a joint-scalar token, fused via a shared transformer
   with occupancy and spectrum tokens** — architecturally expressive, but larger than
   necessary, and the joint-scalar-interaction question is better answered by direct
   joint-processing (see below) than by hoping attention layers reconstruct it from
   marginal tokens.
3. **This document (v2)**: occupancy is a single-channel CNN stream (no broadcast
   redundancy at all); the three scalars are processed *together* by one small MLP
   (preserving their true joint interaction natively) and injected into the occupancy
   CNN via FiLM; spectrum keeps its own encoder; a fusion transformer combines the two
   remaining streams (occupancy tokens, spectrum tokens). This is smaller, and the
   masking granularity problem from (1) is resolved by construction: spatial and scalar
   masking are now two independent mechanisms, not one overloaded one.

---

## 2. Full architecture

```
┌─────────────────────────────── INPUTS (each independently maskable) ──────────────────────────────┐
│                                                                                                       │
│   Occupancy M[64,64]              (l, h, r)                    Spectrum (real,imag)[2,301]          │
│        │                              │                                 │                            │
│  masked patches → learned      any subset masked →          existing ReleasedSpectrumEncoder         │
│  placeholder (per visible/      learned "unknown"           (frozen, from repo) → A_local[301,256]   │
│  masked 4×4 block, same         embedding substituted                                                │
│  convention as existing               │                                 │                            │
│  BlockMasker/apply_mask_        MLP_scalar(l,h,r) → γ,β      pooled -> c_physics [384] (FiLM-style,   │
│  to_pixels)                     (FiLM scale/shift,             reused from SpectrumPath)               │
│        │                        one pair per CNN layer)      + 16 goal tokens a_goal [16,384]         │
│        ▼                              │                                 │                            │
│  ┌─────────────────────┐              │                                 │                            │
│  │ Occupancy CNN         │◄────────────┘                                 │                            │
│  │ encoder (1-channel    │  FiLM modulation at every conv block          │                            │
│  │ input, patch-4 →       │                                                                            │
│  │ 16×16 grid)            │                                                                            │
│  └──────────┬────────────┘                                                                            │
│             │ occupancy tokens [B,256,192]                                                             │
└─────────────┼────────────────────────────────────────────────────────────┼────────────────────────────┘
              │                                                             │
              ▼                                                             ▼
     ┌─────────────────────────────────────────────────────────────────────────┐
     │             FUSION / CONTEXT ENCODER (transformer, joint attention)       │
     │             occupancy tokens ⊕ spectrum-goal tokens (a_goal)              │
     └──────────────────────────────────┬────────────────────────────────────────┘
                                        │ z_x  (contextual embeddings, visible input only)
                                        ▼
                       ┌───────────────────────────────────┐
                       │  PREDICTOR (GCLCT, reused/extended) │
                       │  mask queries (occupancy positions   │
                       │  + one "scalar summary" query) attend │
                       │  to z_x, FiLM'd by c_physics           │
                       └──────────────────┬────────────────────┘
                                        │ ẑ_hat  (predicted occupancy-token latents
                                        │         + predicted scalar-summary latent)
                                        │
     ┌──────────────────────────────────┴───────────────────────────────────────┐
     │  TARGET ENCODER (EMA copy of OCCUPANCY CNN → z_y_raw, the loss target;      │
     │  + separate scalar_mlp_ema, conditioning-only, never a loss target —        │
     │  see §3.6 — full unmasked input) → z_y_raw (stop-grad, existing EMAEncoder) │
     └──────────────────────────────────┬───────────────────────────────────────┘
                                        │
                        JEPA loss (masked occupancy tokens) + VICReg
                        + scalar regression loss (direct L1, not latent)
                                        │
                                        ▼
     ┌────────────────────────────────────────────────────────────────────────┐
     │  DECODERS                                                                │
     │  • occupancy: CNN upsampler, FiLM-conditioned by (known-or-predicted)     │
     │    (l,h,r) at every decode layer → occupancy LOGITS [B,1,64,64]          │
     │  • scalars: 3 small MLP heads reading off z_hat's scalar-summary query    │
     │    → (l̂, ĥ, r̂)                                                          │
     │  • spectrum: NOT decoded by any network. Assemble (occupancy, l̂,ĥ,r̂)     │
     │    into the repo's existing [3,64,64] broadcast convention and pass       │
     │    through the FROZEN MetaDiT surrogate (external/metadit) → Ŝ.          │
     └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component-by-component detail

### 3.1 Occupancy encoder (new)

- Input: `[B,1,64,64]` — occupancy only, never the r/h broadcast channels.
- `nn.Conv2d(1, 192, kernel_size=4, stride=4)` → `[B,256,192]`, same 16×16 grid
  convention as the existing `GeometryEncoder`.
- Positional embedding: reuse `get_2d_sincos_pos_embed` from
  `src/encoders/geometry_encoder.py` unchanged (grid size is unchanged; only the
  hidden dim and patch-embed input channels differ).
- Transformer depth: same `TransformerBlock` class, reused verbatim, `hidden=192`
  instead of 384 (see §5 for the dimension-reduction rationale).
- **FiLM injection**: after each `TransformerBlock`, apply
  `x = γ ⊙ x + β`, where `(γ, β) = MLP_scalar(l, h, r)` (see 3.2). One `(γ,β)` pair
  per block, produced by independent linear heads off a shared 2-layer MLP trunk, so
  the scalar MLP is one small module, not one per layer.
- **Initialization caveat (must be stated explicitly, not glossed over):** the
  existing `init_from_metadit` pathway initializes the released 3-channel patch
  embed into this repo's patch embed. A 1-channel occupancy encoder cannot reuse that
  init directly. Options: (a) sum the released 3-channel kernel's input-channel axis
  into a single-channel kernel as an approximate init, or (b) initialize from scratch.
  Recommendation: try (a) first since it costs nothing and may transfer useful
  spatial filters; validate with the same forward/backward preflight tests already in
  the repo before trusting it further. Do not assume it works without checking.

### 3.2 Scalar MLP (new)

- **Input is 6 numbers, not 3.** For each of `(l, h, r)`: the value if known, else
  `0.0` as a placeholder — **plus an explicit binary known/unknown flag per scalar**,
  concatenated. Do not rely on a single learned "unknown" sentinel value substituted
  for the raw scalar: if that sentinel drifts during training into the range of
  plausible real values, the missingness signal degrades silently, with no way to
  detect it short of noticing the model quietly get worse. An explicit flag is the
  standard, robust pattern for missing-data handling and costs nothing extra.
- `MLP_scalar`: `Linear(6, 128) → GELU → Linear(128, 128)` trunk, feeding N small
  linear heads (one per occupancy-encoder block) each producing `(γ, β) ∈ R^{192×2}`.
- **Zero-init the FiLM heads**: `γ` initialized to output 1, `β` initialized to
  output 0 (i.e. the head starts as the identity transform, matching this repo's
  existing `SpectrumPath`/predictor AdaLN-zero convention). Un-zero-initialized FiLM
  heads would inject random, non-identity scale/shift into every CNN block at step 0
  — a destabilization risk this exact codebase already knows to avoid elsewhere, so
  the new heads should follow the same established pattern rather than introduce a
  new one.
- This directly answers the "will separating the 3 params lose their interaction"
  question from earlier discussion: the MLP always sees all three simultaneously
  whenever they are known, so the true joint interaction is available to the network
  natively — no attention-based reconstruction of the interaction from marginal
  tokens is required.
- Also produces a single pooled 192-dim "scalar summary" vector (mean of the
  pre-head trunk output) that is inserted as one extra token into the fusion
  transformer's sequence — this is what lets the predictor's mask-query mechanism
  target and predict the scalars jointly, and is what the scalar decode heads read
  from at inference (see §3.6 for why this is decoded directly, with no EMA target).

### 3.3 Spectrum path (reused, one new projection, width pinned)

Reuse `src/encoders/spectrum_encoder.py`'s `SpectrumPath` exactly as built:
`c_physics [B,384]` for FiLM-style global conditioning of the predictor, `a_goal
[B,16,384]` as structured tokens joining the predictor's key/value sequence.

**Pinned decision (was left open in v2, now settled):** project `a_goal` down to
192-dim via one new `Linear(384, 192)` layer, and run the fusion transformer and
predictor at **192 throughout**, not 384. Leaving this open invited scope creep
before the core mechanism is even tested, and up-projecting occupancy to 384 to
match `a_goal` instead would quietly undo §5's entire justification for shrinking
the representation in the first place. Practical note for implementation: 192 dims
with 6 attention heads gives 32 dims/head, which is fine but worth stating
explicitly.

### 3.4 Fusion / context encoder (new, small, depth pinned)

A small transformer that lets the occupancy tokens and the spectrum's `a_goal`
tokens attend to each other before the predictor stage. This is the piece that lets,
e.g., "which resonance the spectrum implies" inform "which occupancy tokens matter,"
and vice versa, prior to the predictor's mask-filling job. Reuses the same
`Attention`/`TransformerBlock` classes already in the repo, just a fresh, smaller
stack.

**Pinned decision (was "2–4 layers, not fixed" in v2):** start at **2 layers**.
Open-ended depth ranges invite premature hyperparameter search before the core
mechanism is validated — consistent with this project's own established discipline
of not adding capacity without a task-level reason. Increase depth only if a
specific validation gate (§8) motivates it.

### 3.5 Predictor (reused, lightly extended)

Reuse `src/predictor/gclct.py`'s `GCLCT` unchanged in mechanism, at the new 192-dim
width — with one explicit check before assuming this "just works": `c_physics`
stays 384-dim (unchanged, from `SpectrumPath`), so if `GCLCT`'s internal FiLM
projection from `c_physics` is already parametrized as a linear layer into
`2 × hidden_dim`, a width change is just a config value; if it's hardcoded to 384
anywhere internally, that needs fixing first. Verify this alongside the existing
MetaDiT-init transfer test (§8.1 item 3) rather than assuming it, since it's the
same category of risk — reused code meeting an unfamiliar width. Mask-token queries
(one per masked occupancy patch, same convention as
`context_encoder.mask_token + pos` in `assembly.py`) plus **one additional query
token** for the scalar summary, cross-attending to the fused context `z_x`,
FiLM-conditioned by `c_physics`. This is a small, additive change to the existing
`_JEPAForwardMixin._encode` method's query-construction logic, not a rewrite.

### 3.6 Target encoder (occupancy EMA for the loss target; a second, narrower EMA for target-side conditioning only)

Same EMA mechanism as `src/encoders/target_encoder.py`'s `EMAEncoder`, tracking
**the occupancy CNN encoder**, run on the **complete, unmasked** ground truth every
step. Produces `z_y_raw` for occupancy tokens, exactly mirroring the existing
raw/normalized-alias contract in `assembly.py`'s `forward()`. This is the only
latent compared against the predictor's output in the JEPA loss.

**Correction from v2 (unchanged from v3):** v2 additionally described a second EMA
shadow copy tracking the scalar MLP, "producing a target scalar summary vector" for
a latent-space comparison — while §4.2 says the scalar loss is direct L1 regression
against ground truth, not a latent JEPA-style loss. Those two statements
contradicted each other. **There is still no EMA-tracked scalar latent used as a
loss target** — three scalars have no high-dimensional unpredictable detail to
protect against, so the reasoning that motivates a JEPA-style EMA-latent loss for
occupancy doesn't apply to them, and the direct-regression loss in §4.2 stands as
written.

**New in v5 — a gap the v3/v4 passes missed:** the target encoder's occupancy CNN
uses FiLM, conditioned by `(l, h, r)` (§3.1). That conditioning has to come from
*some* scalar MLP forward pass — and if it's computed from the **live, currently-
training** scalar MLP, the target latent would shift every single training step as
those weights change, even though the occupancy CNN's own weights are EMA-stabilized.
That defeats the entire point of an EMA target: it's supposed to be a slow-moving,
stable answer key precisely so the predictor has something fixed to regress toward
step-to-step. A target whose *conditioning* is non-stationary is not stable just
because its *backbone* is.

**Fix:** maintain a second EMA shadow copy — call it `scalar_mlp_ema` — tracking
only the scalar MLP's weights, updated on the *same* momentum schedule as the
occupancy EMA. Its **sole role** is producing the `(γ, β)` FiLM parameters consumed
inside the target encoder's forward pass. It is never decoded, never compared
against anything as a loss, and never referenced by the predictor or the scalar
regression heads in §4.2 — those continue to use the live, training scalar MLP, as
already specified. This does not reopen the contradiction fixed in v3: that fix was
about removing an EMA scalar representation used as a *loss target*; this is a
different role (stabilizing the *conditioning* used inside an already-existing EMA
computation), analogous to why the occupancy backbone itself needs EMA'ing in the
first place. State this explicitly in code comments at implementation time — it is
easy to accidentally wire the live scalar MLP into the target path and not notice,
since both paths are otherwise structurally identical.

---

## 4. Decoding — the part to get right

This project's most concrete prior mistake risk was conflating pixel-space
reconstruction with genuine understanding. The decoding design here is deliberately
asymmetric across the three outputs, on purpose:

### 4.1 Occupancy decoding

```
predicted occupancy tokens [B,256,192]
  → reshape [B,192,16,16]
  → ConvTranspose/upsample block, FiLM-conditioned by (l,h,r) at EVERY layer
  → occupancy LOGITS [B,1,64,64]
```

**Decode-time FiLM conditioning — train/inference consistency (fixed from v2):**
v2 said decode-time FiLM uses "known values if given at inference, predicted
otherwise" but didn't specify training-time behavior — leaving a real risk that
training always teacher-forces true scalars into the decoder regardless of that
step's scalar-masking curriculum (§6.2), which would create a train/inference gap
specifically in the "partial-parameter conditioning" deployment scenario (§7), where
scalars are genuinely unknown at inference. **Fixed rule, both phases:** decode-time
FiLM always uses exactly whatever that step's scalar-masking curriculum already
determined — the true value when a scalar is marked known, the predicted value
(`l̂, ĥ, r̂` from §4.2) when marked unknown. Training and inference use the identical
rule; there is no separate inference-time branch.

- Loss: `BCEWithLogits(logits, true_occupancy)`. No per-pixel scalar-value loss here
  — there is no scalar-value channel anymore, because occupancy is single-channel.
  This removes the redundant-reconstruction problem at the root, rather than
  patching it with loss reweighting as earlier drafts attempted.
- At deployment, a hard threshold (`logits > 0`) gives the final binary design.
  During training, the surrogate needs a **differentiable** occupancy — use the
  sigmoid probability directly as a soft occupancy field for the surrogate's
  physics-consistency pass, with a straight-through estimator (forward: hard
  threshold; backward: gradient of the sigmoid) if training reveals the soft field
  is too far out-of-distribution for the surrogate (the surrogate was trained on
  hard binary patterns — this needs to be checked empirically, not assumed to work
  either way; see §6 gate).

### 4.2 Scalar decoding

Three small linear/MLP heads reading the predicted scalar-summary latent →
`(l̂, ĥ, r̂)`. Loss: plain L1 (or Huber) regression against ground truth, **not** a
latent JEPA-style loss — three scalars have no high-dimensional unpredictable detail
to protect against, so the reconstruction-avoidance argument that motivates JEPA's
latent loss for occupancy does not apply here, and direct regression is simpler and
equally principled.

### 4.3 No spectrum decoder

Assemble the decoded `(occupancy, l̂, ĥ, r̂)` into the repository's existing
`[3,64,64]` broadcast convention (`occupancy × r̂/5`, `occupancy × ĥ`, `l̂/3`
everywhere — reuse `src/data/dataset.py`'s exact construction, in reverse) and pass
through the frozen, already-integrated MetaDiT surrogate (`external/metadit`) to get
`Ŝ`. Compare to the target spectrum. Gradients flow backward through the surrogate
(parameters frozen, but `requires_grad` NOT disabled on the forward pass — the
existing `sensitivity_masks` function in `src/data/mask.py` already demonstrates
this exact frozen-but-differentiable pattern via `torch.enable_grad()` and
`torch.autograd.grad`, so this is a proven mechanism in this codebase already, not
new machinery).

---

## 5. Sizes

| Component | Old (v1 discussions) | New (this doc) |
|---|---|---|
| Occupancy tokens | 256 × 384-dim, 3-channel input | 256 × **192-dim**, 1-channel input |
| Scalar representation | 3 separate tokens (+ joint token, since dropped) | 1 MLP, FiLM output, 1 pooled summary token |
| Spectrum tokens | 384-dim (existing) | 384-dim (existing SpectrumPath, unchanged) or projected to 192 — decide empirically |
| Fusion trunk | ~290 tokens, 384-dim | ~257 tokens (256 occupancy + 1 scalar summary, plus 16 a_goal), 192-dim |

The reduction is justified specifically because the redundant broadcast channels
that the original 384-dim width was partly compensating for no longer enter the
occupancy stream at all — this is not a width cut made without a task-level reason
(which earlier guidance explicitly warned against); it is removing capacity that was
never doing useful work.

---

## 6. Masking curriculum and training procedure

### 6.1 Reused masking infrastructure

`src/data/mask.py`'s `BlockMasker` already supports random and
surrogate-sensitivity-guided block placement, and already handles `ratio >= 0.999`
(full masking) as a first-class case. **Reuse this unchanged for occupancy
masking.** It does not need to change at all under this redesign — its output `M`
now gates the occupancy CNN's patch placeholders exactly as it previously gated
`apply_mask_to_pixels`.

### 6.2 New: scalar masking

Independent of occupancy masking, sample a per-scalar Bernoulli "known/unknown" flag
each step (can be correlated — e.g. "all three known" or "all three unknown" more
often than independent flips, to reflect realistic deployment scenarios — this
should be a configurable curriculum, not a fixed 50/50).

### 6.3 Training loop, one step

1. Sample a batch; sample an occupancy mask `M` via `BlockMasker` (existing);
   sample a scalar-known mask independently (new).
2. Build masked occupancy input (existing `apply_mask_to_pixels`, operating now on
   the single occupancy channel) and masked scalar input (zero the value and set
   the known/unknown flag per §3.2 for any flagged scalar).
3. Run occupancy CNN + scalar MLP on the masked input → visible tokens.
4. Run spectrum path on the (always fully known, during training) target spectrum
   → `c_physics`, `a_goal`.
5. Fusion transformer over occupancy tokens + `a_goal` → `z_x`.
6. Predictor: mask queries (occupancy + scalar-summary) attend to `z_x`,
   FiLM'd by `c_physics` → `ẑ_hat`.
7. EMA target encoder on the **full, unmasked** ground truth occupancy (FiLM'd by
   `scalar_mlp_ema`'s output, not the live scalar MLP — §3.6) → `z_y_raw` for
   occupancy tokens. No scalar EMA *loss target* exists (scalar loss is direct
   regression, §4.2) — `scalar_mlp_ema` exists solely to stabilize this step's FiLM
   conditioning.
8. Losses:
   - JEPA latent loss (existing `losses/jepa_loss.py`, reused) on masked occupancy
     token positions only.
   - VICReg (existing `losses/vicreg.py`, reused) on the occupancy latent space.
   - Scalar L1/Huber regression loss, decoded directly from the predictor's
     scalar-summary query output (§4.2), on masked scalar positions only — no
     latent comparison, no EMA target involved.
   - Decode predicted occupancy + scalars (using the §4.1 train/inference-consistent
     FiLM rule) → assemble broadcast tensor → frozen surrogate → physics-consistency
     loss against true spectrum (weight ramped from 0, per the gradient-explosion
     risk flagged in earlier review).
9. Backprop through occupancy CNN, scalar MLP, fusion transformer, predictor,
   decoders. No gradient into EMA target or the frozen surrogate.
10. EMA update (existing momentum schedule, reused).

### 6.4 Masking-ratio curriculum across training, not fixed at one setting

Do not train at a single fixed occupancy-mask ratio. Sample it per batch across the
full range including `ratio → 1.0` (full masking, ⇒ pure spectrum-conditioned
generation) and low ratios (⇒ retrofit/completion). This is what makes the same
trained model usable across all three deployment scenarios in §7, and is a direct
continuation of what `BlockMasker` already supports.

---

## 7. Deployment — three scenarios, reported separately (do not pool)

**Pure inverse design** (spectrum known; occupancy + all scalars masked): hardest
case, and the underlying inverse mapping is not unique — many designs can produce
similar spectra. Report accuracy for this case on its own; do not average it together
with the easier cases below, since pooling would mask a weak showing here behind a
strong showing on retrofit.

**Partial-parameter conditioning** (spectrum known; some scalars known, occupancy
masked): intermediate difficulty.

**Retrofit / constrained completion** (spectrum known; occupancy mostly known with a
masked region, scalars known): easiest and most industrially realistic — most tokens
are ground truth, not predictions.

For all three: decode → assemble broadcast tensor → frozen surrogate → compare `Ŝ`
to target spectrum, as the actual acceptance test. A design that decodes cleanly but
fails this check is not a usable result.

**Known limitation to flag, not hide:** this architecture produces one deterministic
latent per input. For the pure-inverse-design case specifically, where the mapping
is genuinely one-to-many, a single point estimate may not be an adequate deployment
story. If multiple candidate designs per target spectrum are required, this needs
either sampling noise injected into the predictor or a downstream gradient-based
refinement loop against the frozen surrogate — decide this explicitly before
treating the base architecture as deployment-complete for that scenario.

---

## 8. Validation gates (run before scaling up)

### 8.1 Core mechanism checks (carried from v2)

1. **Cross-recovery check**: with occupancy 100% masked and scalars known, does the
   model recover occupancy structure meaningfully better than a trivial baseline
   (dataset-frequency prior)? With occupancy known and scalars 100% masked, does
   scalar regression beat predicting the dataset mean? Both directions must pass
   before trusting the joint representation is doing real work.
2. **FiLM ablation**: train with and without the scalar-FiLM injection into the
   occupancy CNN; confirm occupancy reconstruction/completion is measurably worse
   without it, which validates that the joint scalar interaction (§3.2) is actually
   being used, not just present in principle.
3. **MetaDiT-init transfer check** (§3.1 caveat): confirm the summed-kernel init
   actually helps versus training from scratch on a small run before committing to
   it as the default.
4. **Soft-vs-hard occupancy surrogate check** (§4.1): confirm the frozen surrogate
   produces sensible, non-degenerate output on soft (sigmoid) occupancy fields
   before relying on straight-through gradients through it; if the surrogate is too
   sensitive to non-binary input, fall back to hard thresholding with
   straight-through gradient estimation only at the final loss, not throughout
   training.
5. **Per-scenario spectrum-consistency gate** (§7): report separately, gate
   progression to "deployment-ready" claims independently per scenario.

### 8.2 Re-run the existing diagnostic suite against the NEW encoder — do not assume it transfers

Every piece of representation-health evidence gathered so far (collapse diagnostics,
VICReg gradient attribution, the within-bucket spatial-structure probe) was measured
against the old 384-dim, 3-channel encoder. Nothing about those results automatically
carries over to a differently-shaped, differently-initialized, differently-conditioned
encoder. Specifically:

6. **Re-run `vicreg_gradient_attribution.py`** against the new occupancy encoder.
   The original conclusion ("VICReg gradients reach the raw encoder") needs to be
   re-established here, not assumed.
7. **Re-run the within-bucket spatial-structure probe** against the new encoder, and
   compare its trained-vs-trivial-shape margin to the old encoder's recorded +0.225.
   Since §5's entire justification for this redesign is that removing scalar leakage
   should make the "does it need to learn shape" pressure more genuine, this is
   directly testable: the new encoder should clear the shape-aware trivial baseline
   by a **larger** margin than before. This reuses infrastructure already built and
   gives a quantitative answer to whether the redesign delivers on its own stated
   rationale, rather than assuming it does.

### 8.3 Collapse-mode checks — VICReg and the EMA target do not cover these

VICReg/EMA mitigate classic one-vector representational collapse. They do nothing
for at least four other, more specific collapse modes this architecture can hit
silently. All four need their own explicit check; none should be assumed safe by
extension from the VICReg gradient result.

8. **Goal-ignoring collapse — real/null/shuffled, stratified by masking regime, not
   pooled.** This repo's own `SpectrumPath` docstring already names this failure
   mode and ships a `goal_mode="null"` ablation for it. Extend it with a third
   condition — a **shuffled** target spectrum (another sample's true spectrum, still
   present, still non-null) — because real-vs-null alone is not enough: this project
   already has direct evidence that a real/null gap can appear (real-vs-null distance
   ≈1.4) while the predictor still barely benefits from the spectrum in practice.
   The gate is "real produces measurably better physics-consistency than shuffled,"
   not "real ≠ null."
   **Critical addition — do not report one pooled gap across the whole masking
   curriculum.** The target latent (`z_y_raw`) is built from true occupancy + true
   scalars only — it never depends on the spectrum (§0.2 explains why this is
   intentional, not an oversight). That means the loss can only ever *reward*
   spectrum-dependence when the spectrum is *necessary* to reach the target — i.e.,
   when occupancy and scalars are both heavily masked. Whenever occupancy or scalars
   are mostly visible, the predictor can reach a good latent without the spectrum at
   all, and a pooled gap — dominated by the many easy-regime batches in a typical
   curriculum — can look healthy while genuine spectrum-dependence is absent exactly
   where it matters. **Report the real/null/shuffled gap separately for at least two
   strata: (a) low occupancy-mask + scalars known, and (b) full occupancy-mask + full
   scalar-mask.** Gate specifically on (b) holding up — a pass in (a) alone says
   almost nothing about the pure-inverse-design scenario this redesign exists to serve.
9. **Scalar-conditioning collapse — same upgrade, same stratification.** (New failure
   mode, no existing test — the FiLM-from-scalars mechanism is new.) Testing only
   "vary `(l,h,r)`, confirm the output changes" checks sensitivity, not whether the
   change is in the *correct* direction. Use the same three-way real/null/shuffled
   comparison as check 8, stratified the same way. **Framing correction:** this
   pathway is not a *new or worse* shortcut compared to the original merged-channel
   leakage — the original leakage was already about as cheap as a shortcut gets (an
   occupied pixel's channel value *equals* the scalar exactly; a linear probe
   recovered it at R² ≈ 0.96 from a random-init encoder, before any training). What's
   different here is *reliability of availability*: FiLM guarantees the scalar
   pathway is present, by construction, at every layer, whenever scalars are known —
   rather than something the model has to happen to discover in pixel statistics.
   Removing FiLM would not fix the underlying issue, because the underlying issue is
   that the target excludes physics by design (§0.2), not that FiLM specifically is
   the leak. Gate: real must produce lower decode error and better physics-
   consistency than shuffled, specifically in the stratum where occupancy and
   scalars are both heavily masked.
10. **Generative mode collapse** (expected to some degree, not just a risk to rule
    out). Given the field's own non-uniqueness observation, a deterministic
    predictor will plausibly gravitate toward an "average" plausible design rather
    than tracking fine target-spectrum differences — and this can happen even when
    checks 6–9 all pass, because it's collapse in output diversity across varying
    conditions, not collapse of the latent's internal variance. Test by perturbing
    the target spectrum slightly with everything else fixed and confirming the
    decoded design changes proportionally, not just that raw latent variance looks
    healthy.
11. **Majority-class collapse in the occupancy decoder.** Occupancy masks are
    imbalanced (not 50/50 filled). A decoder can achieve deceptively good pixel
    accuracy by predominantly predicting the majority class almost everywhere.
    Check IoU/F1 on the occupied class specifically, not overall pixel accuracy, and
    confirm predicted occupancy *fraction* varies across samples the way ground
    truth does, rather than clustering near one value.

None of checks 8–11 are optional or deferrable to "if something looks wrong later" —
add them to the same pre-scaling gate as 8.1/8.2, since several of them (especially
8 and 10) are specifically most severe in the pure-inverse-design scenario, which is
also the scenario this whole redesign is ultimately meant to serve.

---

## 9. Minimal repository change list

**Reused, unchanged:** `src/data/mask.py` (`BlockMasker`, `apply_mask_to_pixels`,
`sensitivity_masks`), `src/encoders/spectrum_encoder.py` (`SpectrumPath`,
`ReleasedSpectrumEncoder`), `src/predictor/gclct.py` (`GCLCT`, extended not
rewritten), `src/losses/jepa_loss.py`, `src/losses/vicreg.py`, `src/encoders/
target_encoder.py`'s EMA mechanism, instantiated twice: once for the occupancy CNN
(the loss target, `z_y_raw`) and once as `scalar_mlp_ema` (target-side FiLM
conditioning only, never a loss target — see §3.6), same momentum schedule for both;
the frozen MetaDiT
surrogate integration, checkpointing/production-hardening infrastructure.

**New:**
- `src/encoders/occupancy_encoder.py` — single-channel CNN + FiLM injection points
  (adapted from `geometry_encoder.py`'s `GeometryEncoder`/`TransformerBlock`).
- `src/encoders/scalar_encoder.py` — `MLP_scalar`, per-layer FiLM head projections,
  pooled scalar-summary token, explicit per-scalar known/unknown flags (§3.2).
- `src/fusion/fusion_encoder.py` — small transformer combining occupancy tokens and
  `a_goal`.
- `src/decoders/occupancy_decoder.py` — FiLM-conditioned CNN upsampler, occupancy
  logits output.
- `src/decoders/scalar_decoder.py` — 3 small MLP heads.
- `src/data/scalar_mask.py` — independent scalar known/unknown sampler.
- `scripts/train/train_unified.py` — new training loop per §6.3.
- `scripts/eval/eval_scenarios.py` — separate per-scenario evaluation per §7.

**Not touched:** dataset loading/parsing logic in `src/data/dataset.py` (the
broadcast-tensor assembly is still needed, only in reverse, at decode time for
surrogate input — reuse its exact construction, don't reimplement it).

---

## 10. Where the novelty actually is (stated plainly, for a paper framing)

Not "cross-modal JEPA" (an active, populated category already — X-JEPA, M3-JEPA,
CrossJEPA) and not "spectrum-conditioned metasurface generation" alone (MetaDiT's own
diffusion transformer already does this). The specific, checkable contributions are:

1. A representation that separates true spatial degrees of freedom (occupancy) from
   global scalar design parameters via FiLM rather than broadcast-channel packing,
   validated by the ablation in §8.1 item 2 against naive concatenation/broadcast
   baselines.
2. A single framework handling arbitrary partial-observation combinations across
   occupancy, scalar parameters, and spectrum — directly addressing MetaDiT's own
   stated limitation of methods that generate only a subset of parameters at a time.
3. Physics-consistency training with a frozen, differentiable surrogate providing
   gradient into the predictor/decoder through soft-occupancy straight-through
   estimation — a specific mechanism, not just "we used a surrogate loss."
4. Per-scenario (not pooled) evaluation that honestly separates the ill-posed
   pure-inverse-design case from the better-constrained retrofit case, which the
   literature's own non-uniqueness observations say should not be reported together.

Any paper claim should be benchmarked against: (a) the existing MetaDiT diffusion
transformer on the pure-inverse-design scenario, and (b) a nearest-neighbor retrieval
baseline on the retrofit scenario — not against "no architecture" as a strawman.

---

## 11. What must not change from here

- Frozen MetaDiT surrogate stays frozen in every phase; no second physics predictor.
- Occupancy latent dimension and depth are reduced with the specific justification
  in §5, not reopened as a general hyperparameter search.
- No return to the flat 3-channel broadcast tensor as a *masking/representation*
  target (it remains the correct *surrogate input* convention — those are different
  roles and should not be conflated).
- Every deployment claim reported per-scenario (§7), never pooled across
  ill-posed and well-constrained cases.
