# Goal-Conditioned Physics JEPA for High-Degree-of-Freedom Metasurface Inverse Design

Project design document — v2

---

## 0. Executive framing

This project should not be framed as "JEPA for metamaterial generation." That framing is too
weak and collides with existing physics-conditioned generative design methods.

The actual research problem is:

Learn a predictive latent state of a high-dimensional physical structure such that the same model
can (1) complete a partially specified metasurface toward a desired electromagnetic response,
(2) edit an existing structure toward a new response, and (3) generate new structures from
physics alone when no structural context is provided.

The core mapping is:

```
G_context + S_goal  →  Ẑ_future  →  Ĝ
```

where `G_context` is a partially observed metasurface geometry, `S_goal` is the desired complex
electromagnetic spectrum, `Ẑ_future` is the predicted latent state of a complete structure, and
`Ĝ` is a realizable geometry.

The zero-context limit is explicitly part of the design: setting `G_context = ∅` collapses the
mapping to `S_goal → Ẑ_future → Ĝ`, i.e. unconditional-in-geometry, physics-conditioned inverse
design, as one operating point of the same model rather than a separate system.

**The intended scientific claim is not** "a JEPA can generate metasurfaces" — that is too close
to existing conditional generative methods. **The intended claim is**: a goal-conditioned
predictive latent model can learn a reusable structural state space in which partial observation,
physical-goal conditioning, structural completion, targeted editing, and physics-only discovery
are different context regimes of one model, and that this predictive-latent formulation confers
measurable advantages — in low-data generalization, out-of-distribution target robustness,
multimodal solution coverage, and partial-structure editing — over generating geometry directly.

This document specifies the architecture, losses, training schedule, reuse of MetaDiT
components, two predictor-level mechanisms, novelty position, ablations, failure modes, a compute
budget, and the evaluation plan. Every external citation in this version has been checked against
its primary source (arXiv, AAAI, RSC, or the relevant GitHub repository) as of 2026-08-16.

---

## 1. Why MetaDiT is the starting point

### 1.1 MetaDiT is a strong benchmark, not merely a dataset

MetaDiT — *"MetaDiT: Enabling Fine-grained Constraints in High-degree-of-Freedom Metasurface
Design,"* Hao Li and Andrey Bogdanov (Qingdao Innovation and Development Center, Harbin
Engineering University; School of Physics and Engineering, ITMO University), AAAI 2026 — addresses
a high-dimensional electromagnetic inverse-design problem. The paper frames its contribution
against two specific limitations it identifies in prior high-DoF work: existing approaches often
restrict the model to generating only a subset of design parameters, and they rely on heavily
downsampled spectral targets, which compromises both the novelty and the accuracy of the
resulting structures. Its dataset (adopted, per the MetaDiT paper's own citation, from An et al.
2020) contains 170k+ metasurface designs. Each unit cell uses a 64×64 binary geometric pattern
together with three continuous design parameters: atom refractive index (`r_atom`), atom
thickness (`h_atom`), and lattice constant (`l_lattice`). The substrate refractive index and
thickness are held fixed in the released benchmark.

MetaDiT represents every structure as a `3×64×64` tensor. In each channel, locations occupied by
the meta-atom carry the corresponding continuous parameter; background locations are zero. The
target is the full complex transmission spectrum at 301 frequency points, represented as
`S ∈ ℝ^(301×2)` with channels for real and imaginary parts.

This is exactly the benchmark profile this project needs: spatially structured geometry,
nontrivial degrees of freedom, a rich physical response instead of one scalar, a many-to-one
inverse relationship, an existing state-of-the-art generative baseline, and released
implementation and weights.

**Verified sources:**
- Li, H. & Bogdanov, A. *MetaDiT: Enabling Fine-grained Constraints in High-degree-of-Freedom
  Metasurface Design.* AAAI 2026. arXiv:2508.05076. https://arxiv.org/abs/2508.05076
- Code: https://github.com/JessePrince/metadit
- Released data/model assets: https://huggingface.co/datasets/Hao-Li-131/MetaDiT-AAAI2026

**Precision note.** The MetaDiT abstract frames its own contribution as spectral fidelity and
full-parameter generation relative to prior HDoF work; it does not itself claim to solve
partial-context completion, editing, or physics-only discovery under a masking curriculum. That
gap is real and is where this project's novelty needs to sit (§14), not merely adjacent to it.

### 1.2 What MetaDiT actually does

MetaDiT is **not** a VAE latent-space generator. Its released implementation directly uses a
Diffusion Transformer (DiT) over geometry/image tokens: `G → patch tokens` and
`S → spectrum tokens`, followed by coarse-to-fine conditioning inside the DiT. There is no
conventional geometry VAE bottleneck in the released implementation.

The geometry input `U ∈ ℝ^(3×64×64)` encodes the spatially distributed values of
`(r_atom, h_atom, l_lattice)` over the meta-atom pattern; a ViT/DiT-style patch embedding with
fixed 2-D sine/cosine positional embeddings converts this into a token sequence
(`model/dit.py`, `train/train_metadit.py` in the repository above). These exact module paths
should be re-confirmed against the current repository state at the start of Phase 0 (§7) — GitHub
repositories accompanying recently accepted papers are routinely restructured after
camera-ready, and AAAI-26 acceptance is recent enough that the repository may still be settling.
Treat the file-path references throughout this document as a starting hypothesis to verify, not
fixed fact.

### 1.3 The spectrum encoder — reused with the least modification

This is the component reused most directly. MetaDiT does not flatten the 301-point complex
spectrum into an arbitrary vector. Every frequency point is represented by its two values
`[Re S(f_i), Im S(f_i)]`, projected to a 256-dimensional embedding (`301×2 → 301×256`).
Transformer blocks then apply both sequence/frequency attention and channel attention, followed
by an FFN, so the network can learn relationships between distant frequencies, resonant
structures across a spectral band, coupling between real and imaginary response, and local
spectral patterns — without manually defining resonances or bandwidths.

```
complex spectrum
      |
      v
301 frequency tokens
      |
      +--> sequence attention
      |
      +--> channel attention
      |
      v
301 x 256 spectral representation
```

(`model/spec_encoder.py` in the repository above.) This is exactly what is needed for the
"physics/goal" side of JEPA.

### 1.4 Contrastive pretraining — reused as initialization

Before training its diffusion generator, MetaDiT trains a spectrum encoder jointly with a
geometry ViT using a CLIP-style contrastive objective: `G → E_G → z_G`, `S → E_S → z_S`, with
`z_G ≈ z_S` for matching pairs and separation for mismatched pairs, via a Vision Transformer for
geometry, the spectrum encoder above, projection heads into a shared 512-dimensional space, and a
symmetric contrastive loss (`model/clip_model.py`). This means MetaDiT has already learned a
representation in which geometry and electromagnetic response are semantically related — these
released weights should be used as initialization, not discarded.

### 1.5 Coarse-to-fine conditioning — reused as the conditioning template

MetaDiT conditions its DiT in two complementary ways: (a) the 301 spectrum tokens are averaged
into a global embedding, combined with the diffusion timestep, and injected via AdaLN
conditioning ("what kind of physical response are we trying to achieve?"); (b) the full sequence
of spectrum tokens is concatenated with geometry tokens so self-attention can mix structural and
spectral information directly. This coarse-to-fine idea is one of the strongest design decisions
in MetaDiT and remains central to the predictor here.

### 1.6 The forward physics surrogate — reused as the closed-loop validator

The released implementation also provides a forward EM surrogate, `G → F_EM(G) → S`, used in the
paper to validate generated structures. This gives an essential closed loop:
`Ĝ → F_EM → Ŝ`, so a generated geometry is accepted only if it satisfies the requested physics,
not merely because it looks plausible.

### 1.7 Why MetaDiT is not itself JEPA

MetaDiT and JEPA share a broad philosophy — learn a representation related to physical state,
then use it to guide structure generation — but the objectives differ. MetaDiT learns
`p(G | S)` via diffusion directly over the geometry representation, denoising corrupted geometry
toward the target. This project instead learns a predictor
`P_θ : (E_G(G_context), E_S(S_goal), z) → Ẑ_future`, where the target is the representation of the
*complete* structure, `Z_future = E_target(G_complete)`. The generator operates in predictive
latent state space, not directly in observed geometry space. The central prediction task is:

```
partial structural state + desired physical state  →  future structural state
```

That difference is the core of this project.

---

## 2. Problem formulation

A training sample is `(G, S)` with `S = F_EM(G)`. A context mask `M ∈ {0,1}^N` over geometry
tokens defines the visible geometry `G_c = M ⊙ G` and the missing geometry
`G_m = (1-M) ⊙ G`. The model receives `(G_c, S_g)`, where `S_g` is the desired spectrum
(`S_g = S` in the standard case; counterfactual `S_g ≠ S` pairings are used in controlled
goal-sensitivity experiments — see §9). The target is the latent of the complete geometry,
`Z_y = E_target(G)`. The JEPA predictor produces `Ẑ_y = P(E_context(G_c), E_S(S_g), z)`, trained
against `D(Ẑ_y, sg(Z_y))` in the EMA formulation, or the corresponding non-stop-gradient
objective for LeJEPA (§7.3).

**On mask topology.** Mask *shape*, not just mask *ratio*, materially affects what the task
measures. Uniform random-pixel masking of a 64×64 grid is largely solvable by local interpolation,
because neighboring meta-atom parameters vary smoothly in most physically realizable designs — a
context encoder can average nearby visible patches and never need the physical goal at all. That
would silently undercut the central hypothesis (that goal conditioning helps under heavy masking)
without ever showing up as an explicit bug. Block masking is therefore used:

- Sample 1–4 axis-aligned rectangular blocks in token space (16×16 grid at patch size 4), sized
  to cover roughly `mask_ratio / num_blocks` of the grid each, with a minimum side length of 3
  tokens so blocks can't degenerate into single-token holes.
- Half of training batches place blocks uniformly at random (tests general completion); the other
  half deliberately mask a **resonance-relevant region**, identified via the frozen EM surrogate's
  sensitivity map, so the model must recover physically load-bearing structure specifically
  rather than just plausible-looking texture.

Block masking should be treated as a gating requirement, not a stylistic choice: a model that
"solves" 60% masking under unstructured random masking has not demonstrated what this project
needs it to demonstrate.

---

## 3. Architecture overview

```
                         TRAINING SAMPLE
                              |
                 +------------+-------------+
                 |                          |
          COMPLETE GEOMETRY              SPECTRUM
                 |                          |
                 v                          v
        Target Geometry Encoder      MetaDiT Spectrum Encoder
                 |                          |
                 v                          v
             Z_target                 A_local tokens (301x256)
                                          |
                                          v
                                 Global pool + 16 learned
                                 goal-query cross-attention
                                          |
                                    A_g, A_goal(16 tokens)
                                          |
                  +-----------------------+
                  |
                  | JEPA target
                  v
            +-----------+
            | PREDICTOR |  <-- context bottleneck, goal guidance,
            +-----------+      sparse spectral routing (all below)
                  ^
                  |
        +---------+----------+
        |                    |
    Z_context             A_g, A_goal
        ^
        |
Masked Geometry (block mask)
        |
        v
Context Geometry Encoder
        |
        v
    Z_context (256 tokens) --> Perceiver bottleneck (64 tokens) --> predictor

                Predictor output
                       |
                       v
                    Z_pred
                       |
                 +-----+------+
                 |            |
                 v            v
         Geometry Decoder   (latent physics alignment probe)
                 |
                 v
              G_pred
                 |
                 v
          EM Surrogate (frozen)
                 |
                 v
              S_pred  →  physics comparison against S_goal
```

### 3.1 Geometry encoder

Initialized from the MetaDiT ViT, since the released model has already learned geometry-spectrum
alignment. Input `G ∈ ℝ^(3×64×64)`; patch size 4 → 256 spatial tokens; hidden dimension 384;
6–12 Transformer blocks initially; fixed 2-D sine/cosine positional encoding. Output
`Z_G ∈ ℝ^(N×d)`, `N=256`, `d=384` initially. This is **not** collapsed to a single vector — the
model needs a spatial latent field because the missing structure is spatial.

### 3.2 Context encoder

Receives `G_c` plus the binary mask `M`. Known patches receive their normal patch embedding;
masked locations receive a learned MASK token plus positional embedding — this tells the encoder
"this location is unknown," as opposed to "this location is physically zero" (which zero-filling
would imply). Output: `Z_x = E_context(G_c, M)`.

A **Perceiver-IO-style bottleneck** sits between the context encoder and the predictor: a fixed
set of ~64 learned latent queries cross-attend once into the full 256-token `Z_x`, producing a
compressed context summary that the predictor's masked-query blocks attend to, instead of the raw
256 tokens, in every block. This is a pure efficiency measure — full-resolution `Z_x` remains
available separately for the decoder's masked-replacement step (§3.6), which genuinely needs
pixel-level fidelity. Roughly a 4× reduction in keys/values per predictor cross-attention call at
the sizes in §11, which is what makes the hierarchical prediction and sparse routing mechanisms
below affordable.

### 3.3 Target encoder

Sees the complete geometry, `Z_y = E_target(G)`. Two variants, both implemented:

- **Variant A — EMA-JEPA.** `ξ ← m·ξ + (1-m)·θ`, stop-gradient through `Z_y`, initial momentum
  `m = 0.996` with a schedule toward `0.999` if stable.
- **Variant B — LeJEPA.** Remove the EMA/teacher copy; impose SIGReg-style distribution
  regularization on the embeddings directly.

**Verified sources:**
- Balestriero, R. & LeCun, Y. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the
  Heuristics.* arXiv:2511.08544.
- Reference implementation: https://github.com/galilai-group/lejepa. LeJEPA introduces Sketched
  Isotropic Gaussian Regularization (SIGReg), which constrains learned embeddings toward an
  optimal isotropic Gaussian distribution using a family of univariate and multivariate
  statistical tests (e.g. the Epps–Pulley test) applied over randomly sliced projections of the
  embedding space, rather than a single fixed closed-form penalty. This matters practically:
  "SIGReg" names a family of tests, and the specific test choice and number of slices
  (`num_slices`, `num_points`) are themselves hyperparameters that should be reported in the
  eventual Ablation E (§10) writeup, not left implicit.

EMA vs. LeJEPA is a required ablation (§10, Ablation E), decided empirically, not philosophically.

The target encoder must see only the target geometry, never the target spectrum — otherwise the
predictor could trivially solve a geometry-spectrum alignment problem instead of representing
future structure (Failure Mode 5, §13).

### 3.4 Spectrum / physics encoder

Reused from MetaDiT, initialized from its pretrained weights, kept frozen initially (limited
fine-tuning is a later experiment). Input `S ∈ ℝ^(301×2)`, output
`A_local ∈ ℝ^(301×256)`. The 301-point spectrum is not reduced to a hand-designed scalar
vector (resonance frequency, bandwidth, Q, peak transmission, phase slope, etc.) before the
predictor — the network learns which spectral regions matter. Optional physics descriptors may be
provided as auxiliary inputs but must not replace the raw spectrum.

From `A_local` a global goal token is derived, `A_g = MeanPool(A_local)`, and 16 learned goal
tokens via cross-attention, `A_goal ∈ ℝ^(16×256)`, using `K=16` learned query vectors. So the
physics condition carries a global token for overall response, 16 goal tokens for fine spectral
structure, and (transiently, during the cross-attention step) all 301 source tokens — avoiding
the false choice between one over-compressed vector and exposing all 301 tokens everywhere
downstream.

**Precision note.** "The network learns which spectral regions matter" is the design intent, not
a guarantee — it is an empirical claim to verify, not an automatic property of the architecture.
Cross-attention pooling into a fixed number of learned queries can itself collapse, where a few
of the 16 slots absorb nearly all attention and the rest carry near-uniform, low-information
weight — a known failure mode in Perceiver-style and mixture-of-experts-style query pooling. §20.2
below adds a concrete, cheap diagnostic for this (goal-token utilization entropy) that should be
logged starting in Phase 2, rather than discovered post-hoc during the Ablation J routing analysis
in Phase 5.

### 3.5 Predictor: Goal-Conditioned Latent Completion Transformer (GCLCT)

Four operations per predictor stage:

1. **Encode known structure** → bottlenecked `Z_x` (§3.2).
2. **Create missing-state queries.** For every masked spatial location `i`:
   `q_i = e_mask + p_i`, where `p_i` is spatial positional information.
3. **Inject global physics.** `A_g` controls AdaLN-Zero modulation — "what physical state should
   the complete design achieve?"
4. **Fine-grained structural-physics attention.** Masked structural queries attend jointly to
   `[Z_visible, A_goal]` (or, under sparse routing — §3.5.2 — to a top-k subset of `A_goal`), so
   each missing region can use its own spatial location, surrounding visible geometry, spectral
   target tokens, and the global physical goal simultaneously.

Each predictor block:

```
masked latent queries
        |
        v
LayerNorm
        |
        +------ global physics AdaLN modulation
        |
        v
Self-attention among latent structural queries
        |
        v
Cross-attention:
    queries = missing structural tokens
    keys   = visible geometry (bottlenecked) + goal tokens (or top-k subset)
    values = visible geometry (bottlenecked) + goal tokens (or top-k subset)
        |
        v
FFN
        |
        v
Residual
```

**Residual future-state prediction.** Rather than generating the complete latent from scratch:
for visible positions, `ẑ_i = z_i^context + Δz_i`; for masked positions,
`ẑ_i = z_i^mask-base + Δz_i`, where the predictor learns the correction. Easy parts are
copied/stabilized; uncertainty concentrates on the missing part. For the strict masked-region
JEPA loss, only masked target tokens contribute to the principal prediction loss.

**Hierarchical prediction.** Two latent scales: a global structural state
`Z^global` (overall topology, fill fraction, broad material organization) predicted first,
conditioned on the physics goal; then local masked tokens `Z^local` predicted conditioned on
`(Ẑ^global, Z_x, A)`. This mirrors MetaDiT's coarse-to-fine physical conditioning at the
structural-prediction level.

#### 3.5.1 Classifier-free goal guidance

The single biggest risk in this design is Failure Mode 2 (§13): the predictor learns
`P(Z_y | Z_x) ≈ P(Z_y | Z_x, A)` and ignores the physical goal. Beyond training-time
regularization (§8.6), the predictor is given an explicit, controllable mechanism for goal
strength at inference, borrowed from classifier-free guidance in diffusion models:

During training (from the physics-loop phase onward, §7 Phase 4+), `A_goal` is replaced with a
learned null token `A_∅` with probability ~10%, so the predictor learns both the goal-conditioned
map `P(Z_x, A_goal)` and the goal-unconditioned map `P(Z_x, A_∅)`. At inference the two are
combined:

```
Ẑ_guided = P(Z_x, A_∅) + w · [ P(Z_x, A_goal) − P(Z_x, A_∅) ]
```

with `w > 1` amplifying the goal's effect on the predicted latent. This gives three things a
pure training-time regularizer does not: a tunable goal-strength dial usable in the
out-of-distribution target experiments (§12); a single-forward-pass diagnostic for goal-ignoring
collapse (`‖P(Z_x,A_goal) − P(Z_x,A_∅)‖` small and roughly constant across very different targets
is a direct symptom, not something you only discover via the full counterfactual experiment); and
a mechanism worth stating explicitly as a contribution (§14) — guidance-based conditioning
strength is well established in diffusion but, per the novelty search in §14, not established in
JEPA-style predictive-latent inverse design.

A sharper, normalized version of this diagnostic is given in §20.3.

#### 3.5.2 Sparse spectral-to-spatial routing

Different geometry regions plausibly should attend to different parts of the desired spectrum
(region A responsible for a mid-band resonance, region B for a separate band, region C
contributing broadband response) — but this should be discovered, not hard-coded, and it should
be a falsifiable structural claim rather than just a post-hoc attention-weight visualization
(dense attention always produces *some* nonzero correlation to point at).

This is implemented as **top-k sparse routing**: a small gate scores each masked spatial query
against all 16 goal tokens, `scores_i = q_i^T W a_j` for `j = 1..K`, and only the top `k` (2 or 3)
goal tokens are used in that region's cross-attention, mixture-of-experts style, rather than a
dense softmax over all 16. This makes routing testable: with hard top-k routing, a geometry
region either structurally depends on a given spectral-token subset or it does not, so
routing-consistency across independently sampled geometries with similar goals becomes a
checkable claim (does region A consistently route to the same frequency band across many
samples, or does routing fail to stabilize?) — see §12, Hypothesis H6, §10 Ablation J, and the
externally-validated extension in §20.1. It also reduces cross-attention cost from
`O(N_masked × K)` dense to `O(N_masked × k)` sparse, which matters once hierarchical prediction
(§3.5) doubles the effective token budget.

### 3.6 Geometry decoder

Necessary because JEPA itself only produces latent representations.

```
Z_pred
   |
   v
Transformer decoder blocks
   |
   v
spatial token grid
   |
   v
upsampling / patch projection
   |
   v
3 x 64 x 64
```

Output channels match MetaDiT's representation: atom refractive-index, atom thickness,
lattice-constant. A final projection/constraint layer ensures physically valid values: bounded
activation/normalization for `r_atom`, `h_atom`, `l_lattice`; binary/sigmoid-like treatment for
occupancy if decoded explicitly; geometry post-processing only when the benchmark requires it.
No unconstrained regression.

The decoder is context-aware: `Ĝ = D_G(Ẑ_y, G_x)`, and at inference known pixels are
retained/enforced while missing pixels take the decoder's output, so the model never
unnecessarily rewrites already-known geometry.

---

## 4. Losses

Modular, staged rather than all active from step one (see §7 for why).

**JEPA latent prediction** (masked positions only):
`L_J = (1/|M|) Σ_{i∈M} d(ẑ_i, z_i)`, cosine distance or normalized Huber/MSE after projection.

**Latent distribution regularization**: `L_SIGReg` for LeJEPA; not used in the EMA-JEPA baseline
unless explicitly testing a hybrid.

**Geometry realization**: BCE/L1/L2 depending on channel type — L1 for continuous parameter
channels, BCE or masked classification if occupancy is decoded explicitly.

**Physics response**: `D_S(F_EM(Ĝ), S_goal)`, e.g.
`α·‖Re Ŝ − Re S‖₁ + β·‖Im Ŝ − Im S‖₁` or a complex-valued equivalent.

**Latent physics alignment**: using the pretrained MetaDiT common embedding,
`1 − cos(P_G(Ẑ_y), P_S(S_goal))`, where the projectors map both sides into the previously
learned MetaDiT geometry-spectrum aligned space. Secondary loss.

**Goal-sensitivity**, as an InfoNCE contrastive loss rather than a margin loss:

```
L_goal = −log[ exp(sim(Ẑ_a, Z_a)/τ) / Σ_b exp(sim(Ẑ_a, Z_b)/τ) ]
```

over a batch of `(context, goal, target-latent)` triples, using other goals' target latents in
the same batch as in-batch negatives. This reuses the InfoNCE machinery already present for
MetaDiT's contrastive pretraining (§1.4) instead of introducing a second, differently-tuned loss
family with its own margin hyperparameter and negative-hardness sensitivity. It also yields a
free evaluation metric — top-1/top-5 goal-to-latent retrieval accuracy — alongside the training
signal. Applied only to genuine counterfactual/paired samples, not as a blanket requirement that
all different goals produce distant embeddings.

**Full objective**:

```
L = L_JEPA + λ_R·L_regularization + λ_G·L_G + λ_S·L_S + λ_A·L_align + λ_C·L_goal
```

with `L_regularization = 0` for EMA-JEPA and `= L_SIGReg` for LeJEPA.

### 4.1 Loss curriculum

`λ_JEPA = 1` is never annealed — it holds for the entire run. All other λ's are activated in
stages that match the training phases (§7), not summed from step one:

| Phase | Active losses | Rationale |
|---|---|---|
| 2 — JEPA pretrain | `L_JEPA`, `L_reg` only | isolate the pure prediction question before any decoder exists |
| 3 — decoder | + `L_G`, ramped 0 → target over first 20% of the phase | let latent prediction stabilize before decoder gradients touch it |
| 4 — physics loop | + `L_S`, `L_A`, both ramped | augment, never dominate — `λ_JEPA` stays fixed at 1 throughout |
| 5 — goal-sensitivity | + `L_goal` | depends on an already reasonably well-behaved latent space |

This directly targets Failure Mode 1 (§13): a fixed-weight sum from step one, with `λ_S` set
high enough to matter for the physics loop, tends toward physics-loss dominance by construction,
especially once decoder gradients can flow back through `Ẑ_y`. During weight tuning, check that
removing the physics losses entirely (Ablation D, §10) changes generation quality without
changing latent-prediction accuracy beyond a small tolerance — catch weight drift while tuning,
not only in the final ablation table.

---

## 5. What must not happen

- **Physics loss dominates** (`λ_S` too large): the model degenerates into a direct generator and
  the JEPA latent becomes decorative. Guarded against by the staged curriculum in §4.1.
- **Predictor ignores the spectrum**: `P(Z_x, S) ≈ P(Z_x)`. Guarded against by the InfoNCE
  goal-sensitivity loss (§4) and directly diagnosable via the classifier-free guidance gap
  (§3.5.1, sharpened in §20.3).
- **Decoder does all the work**: if the decoder becomes powerful enough to map generic latents to
  plausible structures without meaningful predictor information, the latent objective loses
  significance. Checked via the latent-ablation and probe experiments in §12.
- **Zero-context works only via memorization**: use genuinely unseen target spectra and geometry
  splits (§12, OOD evaluation).
- **Target encoder leaks the answer**: it must see only the target geometry, never the target
  spectrum (§3.3), or the predictor could trivially solve geometry-spectrum alignment without
  representing future structure at all.

---

## 6. Compute budget

Rough sizing at the §11 recommended model sizes, one dataset pass over ~170k structures, relative
to Phase 2 as the baseline unit:

| Phase | Components active | Relative cost |
|---|---|---|
| 1 — reproduction | frozen MetaDiT components only, inference only | ~0.1× |
| 2 — JEPA pretrain | context enc. + EMA target enc. + predictor | 1× |
| 3 — + decoder | + geometry decoder | ~1.3× |
| 4 — + physics loop | + frozen EM surrogate forward per step (no backward if frozen) | ~1.6× |
| 5 — + goal-sensitivity | + in-batch negatives, larger effective batch | ~1.8× |
| 6 — context curriculum | same model, more epochs across mask ratios | ~1× per epoch, more epochs |
| H — stochastic latent | + noise pathway, K-sample eval | ~1× train, ~K× at eval |
| I — flow-matching (optional) | replaces one-shot predictor | ~3–5× (multi-step sampling) |

Practical implication: Milestones A–F (§7) must succeed before spending compute on the LeJEPA
ablation or flow-matching extension — both roughly multiply cost without changing the underlying
prediction question, so they should not be run in parallel with the core pipeline's first pass.

---

## 7. Training pathway

**Phase 0 — dataset verification.** Load the MetaDiT dataset; verify tensor shape `3×64×64` and
spectral shape `301×2`; load released spectrum-encoder, geometry-ViT, and surrogate weights;
reproduce MetaDiT forward prediction and spectrum evaluation on held-out structures; establish
baseline metrics. Nothing else proceeds until this reproducibility gate passes. This is also the
point at which the repository file-path assumptions flagged in §1.2 should be re-verified against
the live repository state.

**Phase 1 — reproduce MetaDiT representation learning.** Use the released encoders; verify
`G → z_G`, `S → z_S`, and confirm geometry-spectrum retrieval. Provides pretrained
initialization, a common embedding reference, and a sanity check on weight loading. Encoders are
not modified yet.

**Phase 2 — JEPA representation pretraining.** Block masks (§2) at ratios 20/40/60/80/100%,
initially excluding 0% from the main analysis to measure true prediction. Train
`E_context(G_c) → P(Z_x, A) → Ẑ_y` against `E_target(G)`. No geometry decoder yet — this isolates
the fundamental question: can a physical goal-conditioned predictor infer the latent state of a
complete structure from incomplete structure, under a masking scheme that is not locally
solvable? Begin logging goal-token utilization entropy (§20.2) from this phase onward.

**Phase 3 — geometry realization.** Attach the decoder, `Ẑ_y → D_G → Ĝ`, introduce `L_G` (ramped,
§4.1). Establish latent prediction → actual structure, without yet letting the EM surrogate
dominate.

**Phase 4 — physics loop.** Add the frozen MetaDiT surrogate, `Ĝ → Ŝ`, introduce `L_S` (ramped).
The system must now demonstrate that latent predictions correspond to structures that actually
approach the requested electromagnetic target. Goal-dropout training for classifier-free guidance
(§3.5.1) begins here.

**Phase 5 — goal-sensitivity training.** Introduce counterfactual goal pairs — same or similar
context `G_c`, different goals `S_a, S_b` — and require `Ĝ_a`, `Ĝ_b` to move toward different
physical responses where the dataset supports valid alternatives. `L_goal` (InfoNCE, §4) is
active. This phase directly tests whether the predictor actually uses the physical goal, and is
where the Jacobian-validated routing check (§20.1) should be run alongside Ablation J.

**Phase 6 — context curriculum.** Train across `100% → 75% → 50% → 25% → 0%` visible geometry,
via a curriculum rather than abrupt introduction of all masks: early on 100/75/50% visible,
middle 75/50/25%, late 50/25/0%. Retain some examples from earlier regimes throughout to prevent
catastrophic specialization. Report the normalized guidance-gap curve (§20.3) across this entire
curriculum as a single plot.

**Phase 7 — zero-context generation.** At inference, `G_c = ∅`; the only explicit input is
`S_goal`, giving `S_goal → A → P(A, z) → Z_future → D_G → G`. The generated structure passes
through the frozen EM surrogate and is scored against `S_goal` — a true physics-conditioned
discovery experiment.

### 7.1 Milestone sequencing

```
A. Reproduce MetaDiT (dataset, encoders, surrogate, baseline eval)
B. Vanilla deterministic JEPA (masked geometry, EMA target, predictor, latent loss)
C. Geometry decoder (predicted latent -> complete geometry)
D. Physics loop (geometry -> surrogate spectrum)
E. Goal-sensitivity (counterfactual goals, InfoNCE loss)
F. Zero-context curriculum (progressive masking, explicit 100% mask training)
G. LeJEPA ablation (replace EMA with SIGReg)
H. Stochastic latent (multiple structural futures)
I. Latent flow/diffusion (only if H shows a clear need for more expressive multimodality)
```

Do not implement everything simultaneously; do not add a component unless it solves a measurable
failure observed in an earlier milestone (§13).

### 7.2 Minimal first experiment

Before writing a full training pipeline:

1. Block-masked (not random-pixel) 50% context, blocks placed uniformly at random for this first
   test.
2. MetaDiT geometry + frozen spectrum encoders, EMA target encoder, 4–6 block predictor, simple
   geometry decoder.
3. Loss: `L = L_J + 0.1·L_G` only — no `L_S` yet, matching the staged activation in §4.1 (the
   physics loop is Phase 4, not part of this minimal test).
4. Compare: direct masked generator vs. JEPA vs. JEPA with the goal replaced by the null token
   (a cheap proxy for Failure Mode 2, available without the full goal-dropout/guidance machinery).

If JEPA does not beat the direct masked baseline at 50% **block** masking, stop and diagnose
before adding more complexity — and note that failing under block masking is a stronger, more
decision-relevant negative result than failing under unstructured random masking would have been,
since it implicates the representation rather than the mask topology.

### 7.3 Minimal zero-context experiment

Once completion works: train with `P(M=100%) ≈ 0.1`, increasing later. At inference, `G_c = ∅`;
generate 32 structures for one target spectrum; evaluate all 32 with the released surrogate;
report mean error, best error, diversity, uniqueness, validity. First proof-of-concept for
physics-only structural discovery.

### 7.4 Stochastic latent interface

For one-to-many inverse design (`G_1 ≠ G_2` can both satisfy `S(G_1) ≈ S(G_2)`), the model should
eventually support `P(Z_y | Z_x, S)` as a distribution, not a deterministic map `Z_y = f(Z_x, S)`.
Introduce a latent seed `ε ~ N(0, I)`, injected through a learned pathway into the predictor's
masked-query blocks (not simply concatenated to the spectrum vector): `c_z = f_z(ε)`, used via
`AdaLN(x; c_physics, c_z)`. Start with low-dimensional `z` (`d_z = 64`) rather than a large random
tensor.

The first stochastic model should be the one-shot form `Ẑ_y = P(Z_x, S, ε)`, not latent
diffusion — easier to interpret, easier to isolate JEPA from generative complexity, cheaper, and
a direct test of whether the predictive-latent idea works at all. Only if this fails to capture
multimodality should a continuous latent generation path be introduced: a flow
`z_t` from noise to target latent, conditioned on `(Z_x, A)`, learning `v_θ(z_t, t | Z_x, A)` —
preserving the principle that generation happens in latent structural state space, not raw
geometry space. This is an extension, not part of the minimum viable research model.

---

## 8. The unified interpretation of the model

One model, four points on a context-availability continuum, not four separate models:

- **Case A — reconstruction**: `G + S(G) → G`
- **Case B — completion**: `G_partial + S_goal → G_complete`
- **Case C — redesign/editing**: `G_existing + S_new → G_edited`
- **Case D — discovery**: `S_goal → G_new`

```
100% context  ->  reconstruct
 75%          ->  small edits
 50%          ->  structural completion
 25%          ->  large redesign
  0%          ->  physics-only discovery
```

This context-continuum figure is the clearest visual explanation of the research idea and should
anchor the paper.

**Completion mode**: given `G_partial` and target `S*`, generate `G_1..G_K` with the known region
fixed and only the missing region varying — arguably a more practically constrained and
industrially relevant problem than unrestricted inverse design ("given an existing manufacturable
scaffold, modify only the unknown/free region to achieve a desired optical response").

**Editing mode**: given an existing structure `G_0` and target `S*`, mask only selected regions;
the system predicts the smallest/most appropriate structural change toward `S*`, evaluable with a
structural-change penalty `‖G_pred − G_0‖_changed` subject to physics constraints.

**Zero-context mode**: `S* → G` after explicit training (§7, Phase 7). A stochastic seed produces
multiple possibilities, `G_k = D_G(P(A, z_k))`, each independently evaluated by the EM surrogate.

### 8.1 Limitation of zero-context generation

If the dataset has limited design diversity, zero-context generation will only explore the
support of the training distribution — it does not guarantee genuinely new physics. "Novel
geometry" must therefore be measured via nearest-neighbour structural distance, topology
statistics, latent distance from the training set, component density, and spectral-target
novelty — not claimed merely because a generated image looks visually different.

---

## 9. Goal-sensitivity as a first-class safeguard

A well-documented JEPA failure mode is conditioning collapse: the predictor reconstructs the
correct future while ignoring the goal.

**Verified source and precise relevance.** Pendharkar, A. *Predictive Objectives Discard
Exogenous Control-Relevant Features: A Controlled Mechanistic Study.* arXiv:2606.30068 (submitted
29 June 2026). The paper isolates this failure mode in a controlled 2×2 experimental design that
varies feature controllability and relevance independently, comparing six objectives —
reconstruction, JEPA, action-conditioned JEPA, controllability-based JEPA, inverse dynamics under
a random policy, and reward-grounded JEPA — and finds that reward-free predictive objectives
systematically fail on exactly the cell of that 2×2 where a feature is uncontrollable but
control-relevant. This is a precise match for the risk named here: the electromagnetic spectrum
is not something the geometry "controls" in a reinforcement-learning sense, it is an *exogenous
target*, which is the specific case this paper shows predictive objectives are worst at
retaining. It is worth naming this specific 2×2 cell in the eventual paper's related-work
section, rather than gesturing at the paper generally, since it makes the goal-sensitivity design
below look like a targeted fix to a documented and precisely analogous failure mode rather than a
defensive addition.

Goal-sensitivity is therefore not an optional decoration; it is implemented as three coordinated
mechanisms rather than one:

1. **Training-time**: the InfoNCE goal-sensitivity loss (§4) over counterfactual goal pairs.
2. **Architecture-time**: goal-dropout training that enables classifier-free guidance (§3.5.1),
   giving both an inference-time control knob and a single-forward-pass diagnostic.
3. **Structure-time**: sparse top-k spectral-to-spatial routing (§3.5.2), which makes
   goal-dependence a checkable structural property (does routing stabilize across samples?)
   rather than only a scalar loss value.

For the same context `G_x`, two different goals `S_a, S_b` should produce different predicted
latents `Ẑ_a ≠ Ẑ_b` when the corresponding target geometries differ, with decoded structures
producing their respective spectra — this is not a requirement that *all* different goals must
be distant in latent space, only that known counterfactual/paired targets behave this way.

---

## 10. Baselines and ablations

### 10.1 Main baselines

- **Baseline 1 — MetaDiT.** Original `S → G`, released checkpoint.
- **Baseline 2 — deterministic masked generator.** `G_c + S → G`, no JEPA latent objective.
- **Baseline 3 — standard autoencoder latent generator.** `G → Z → G` with conditional generation
  in `Z`.
- **Baseline 4 — proposed EMA-JEPA.** `G_c + S → Ẑ_y → G`.
- **Baseline 5 — proposed LeJEPA.** Same architecture, SIGReg instead of EMA target.
- **Optional Baseline 6 — latent diffusion**, if required for the multimodal comparison (§7.4).

### 10.2 Ablations

| ID | Removed / varied | Question |
|---|---|---|
| A | Pretrained MetaDiT spectrum encoder | Does the learned physics representation matter? |
| B | Fine spectral tokens → one vector | Does fine-grained frequency conditioning matter? |
| C | Goal-sensitivity loss | Does the predictor ignore the requested physics? |
| D | Physics closed-loop loss | Does latent matching alone produce physically correct geometry? |
| E | EMA vs. LeJEPA | Which representation regularization is better? |
| F | Hierarchical prediction | Is coarse-to-fine structural latent prediction useful? |
| G | Stochastic latent | How much multimodality is actually gained? |
| H | Zero-context training | Does explicit training on fully masked structures matter? |
| I | Goal dropout / classifier-free guidance, sweep `w` | Does guidance strength measurably affect goal adherence, especially OOD? |
| J | Dense vs. sparse top-k spectral-to-spatial routing | Physics accuracy and routing-consistency across seeds |
| K | Block masking vs. unstructured random masking | Does mask topology change what the completion task actually measures? |

Ablation K should be run early — a masking-strategy bug would invalidate every downstream
ablation that depends on the completion task being genuinely hard.

---

## 11. Recommended initial architecture sizes

Start modestly; the model should be deliberately smaller than MetaDiT-L at first.

- **Geometry encoder**: input `3×64×64`, patch size 4, 256 tokens, hidden dim 384, 6 Transformer
  blocks, 6 heads.
- **Spectrum encoder**: MetaDiT pretrained configuration — 301 tokens, input dim 2, hidden dim
  256.
- **Physics bottleneck**: 16 learned goal tokens, 1–4 global tokens.
- **Context bottleneck**: 64 learned Perceiver-style latent queries (§3.2).
- **Predictor**: 8 Transformer blocks, hidden dim 384, 6 heads, AdaLN-Zero global conditioning,
  missing-token cross-attention, top-k = 2–3 sparse spectral routing.
- **Geometry decoder**: 4–8 Transformer/MLP upsampling blocks, spatial reconstruction to 64×64.
- **Stochastic latent**: `d_z = 64`.

### 11.1 Exact data flow for one training example

```
Given G, S, sample block mask M
  G_c = M ⊙ G
  Z_x = E_c(G_c, M)                          # 256 tokens
  Z_x' = Perceiver(Z_x)                      # 64 bottlenecked tokens
  Z_y = E_t(G)                               # target, EMA or LeJEPA
  A_local = E_S(S)                           # 301x256, frozen initially
  A_g   = pool(A_local)
  A_goal = learnedPool(A_local)              # 16 tokens
  A_goal_used = top_k(A_goal) per masked query   # sparse routing
  A_used = A_∅ with prob 0.1 (goal-dropout), else A_goal_used

  Ẑ_y = P(Z_x', A_g, A_used, z)
  L_J = D(Ẑ_y, Z_y)                          # masked positions only

  Ĝ = D_G(Ẑ_y, G_c)                          # known pixels retained
  Ŝ = F_EM(Ĝ)                                # frozen surrogate
  L_S = D_S(Ŝ, S)
  L_A = D_A(E_S(Ŝ), A_g)

  L = L_J + λ_R L_reg + λ_G L_G + λ_S L_S + λ_A L_A + λ_C L_goal    # staged, §4.1
```

---

## 12. Hypotheses and evaluation

**H1.** JEPA latent prediction should outperform direct reconstruction under heavy **block**
masking (20/40/60/80/100%), measured on geometry reconstruction, spectral error, and validity.

**H2.** The predictive latent should generalize better to unseen physical targets. Split the
target spectrum space so some target spectra/spectral regions are held out; compare MetaDiT, a
deterministic conditional generator, and JEPA.

**H3.** JEPA should have a sample-efficiency advantage. Train at 10/25/50/100% of available
structures; plot physics error vs. dataset size. This is one of the most important experiments,
since representation learning is supposed to help most when examples are scarce.

**H4.** The model should support meaningful multimodal generation. For a single target `S*`,
generate `G_1..G_K`; evaluate average and best-of-K spectral accuracy, geometry diversity, latent
diversity, pairwise structural distance. A good model should not simply produce the same
structure `K` times.

**H5.** The latent space itself should contain physical information. Freeze the geometry
encoder; train small probes from `Z` to spectrum, resonance features, spectral distance, geometry
statistics; compare against random initialization, a supervised autoencoder, MetaDiT's geometry
latent, and an ordinary ViT.

**H6.** Goal sensitivity should be visible in latent space and in routing structure. For fixed
context `G_x`, `S_1 ≠ S_2` should produce `Ẑ_1 ≠ Ẑ_2` when the target structures differ, with
distance correlating with `D_S(S_1, S_2)` rather than arbitrary noise; and, under sparse routing
(§3.5.2), a given geometry region should route consistently to a stable subset of goal tokens
across independently sampled geometries sharing a similar goal. §20.1 adds an externally-validated
form of this claim.

### 12.1 What "physics-aware latent space" should mean, measurably

- **Structural locality**: nearby latent tokens correspond to nearby spatial structures.
- **Physical locality**: nearby global latent representations correspond to similar spectra.
- **Goal-directed displacement**: changing the requested spectrum induces a latent displacement
  whose direction predicts the resulting structural changes — directly testable via the
  classifier-free guidance gap (§3.5.1, §20.3).
- **Realization consistency**: decoding a latent and passing it through the surrogate produces
  the intended physical behavior.

Only if these are demonstrated should the phrase be used at all.

### 12.2 Latent geometry analysis

For a set of structures `{G_i}`, compute `Z_i = E_G(G_i)`, pairwise latent distances `d_Z(i,j)`,
and compare against physical distance `d_S(i,j) = D_S(F_EM(G_i), F_EM(G_j))` and structural
distance `d_G(i,j)`. Report correlations and rank agreement to test whether the latent geometry
reflects physical similarity.

### 12.3 Out-of-distribution evaluation

Construct target spectra deliberately outside the training distribution — hold out frequency
regimes, hold out resonance combinations, interpolate between clusters, or extrapolate modestly
beyond target statistics — then check whether `S_OOD → G → F_EM(G)` actually moves toward the
unseen target. This is a substantially stronger test of representation quality than
in-distribution reconstruction, and the natural place to report the classifier-free guidance
sweep from Ablation I.

---

## 13. Failure modes

This project should be considered a failure if: MetaDiT's direct diffusion baseline is
consistently better; JEPA only helps because of more parameters (guard against by matching
compute/parameters in the comparison); physics loss overwhelms latent prediction (§4.1); the
predictor ignores the spectrum (§9); zero-context generation collapses to trivial/memorized
shapes; stochastic samples fail to improve mode coverage; latent probes show no stronger physics
organization than baseline encoders; or gains disappear when parameter count and compute are
matched. If that happens, the correct and scientifically valuable conclusion is that JEPA is
unnecessary for this task.

**What would make it a convincing paper instead**: similar-or-lower compute than MetaDiT with
better performance under structured context constraints; better low-data scaling (H3); better OOD
target generalization (§12.3); better completion/editing (§8); better multimodality (H4); latent
interpretability, including stable and externally-validated routing structure (H6, §20.1); and
one unified model supporting reconstruction, completion, editing, and discovery — substantially
more convincing than "JEPA works on a metamaterial dataset."

---

## 14. Novelty audit

A search (JEPA + inverse design, JEPA + metamaterials, JEPA + metasurfaces, physics-conditioned
JEPA, goal-conditioned JEPA, predictive latent inverse design, material JEPA, physics-informed
latent world models, masked geometry completion with physical targets) found strongly adjacent
work but no exact match for the full proposed combination in high-DoF EM metasurface design.
Absence from this search is not proof of priority; a formal publication search against IEEE,
SPIE, Optica, APS, ACS, Nature, arXiv, Google Scholar/Scopus/Web of Science, and patent literature
must still be performed before any novelty claim is finalized. Verifying the five citations below
against their primary sources confirms they say what this section claims — it is not a substitute
for that broader systematic search, which is a categorically larger task.

**Closest existing work: MetaDiT.** High-resolution spectral conditioning, a pretrained spectrum
encoder, geometry-spectrum contrastive alignment, diffusion-based inverse generation, coarse-to-
fine physical conditioning — "use a spectrum encoder and a learned physical latent to generate
metasurfaces" is not novel. MetaDiT is the primary baseline. (Li & Bogdanov, AAAI 2026,
arXiv:2508.05076, https://arxiv.org/abs/2508.05076)

**Closest JEPA work in materials: Polymer-JEPA.** Piccoli, F., Vogel, G. & Weber, J. M. *Joint
embedding predictive architecture for self-supervised pretraining on polymer molecular graphs.*
Digital Discovery, 2026, 5, 819–834. DOI: 10.1039/D5DD00308C (also arXiv:2506.18194). The paper
pretrains a JEPA model on a large dataset of conjugated copolymer photocatalysts, then fine-tunes
on two downstream tasks — predicting electron affinity in the same chemical space and classifying
phase behavior in diblock copolymers, a different chemical space — reporting improved downstream
performance under label scarcity and gains from cross-domain fine-tuning. This establishes that
JEPA in materials science is already a real direction, but it is graph representation learning
for downstream property prediction, not high-DoF physics-conditioned structural generation —
"first JEPA for materials" cannot be claimed.

**Closest modern JEPA physics work: Phys-JEPA.** *Phys-JEPA: Physics-Informed Latent World Models
for Multivariate Time-Series Forecasting*, arXiv:2606.16076. Latent states are decomposed into
physical and residual components, with physical consistency imposed directly on latent states and
latent transitions rather than only on decoded forecasts — demonstrated on Jena Climate, Traffic,
and Electricity forecasting, with modest but consistent MSE improvements over supervised
baselines. This supports the conceptual approach here (structure the *latent*, not just the
output, with physics), but is temporal forecasting of coupled scalar/vector time series, not
inverse structural design.

**Recent physical-state JEPA work: PSG-JEPA.** Yan, H. et al. (15 authors, corresponding author
Haoang Li). *Is Forward Prediction Enough? Physical State Grounding for JEPA World Models.*
arXiv:2608.06799 (submitted 7 August 2026), cs.RO. PSG-JEPA shapes JEPA latents with two
training-only grounding objectives — grounding individual latents in robot proprioceptive state
(joint angles), and grounding latent pairs in multi-horizon joint-angle changes — evaluated via
latent probing, goal-conditioned planning on frozen latents, and real-robot policy learning,
consistently outperforming baseline latent world models at all three evaluation levels.

**Precision note.** This is a robotics paper about proprioceptive grounding, not a
metasurface- or materials-adjacent paper. The relevance here is structural, not domain-adjacent:
PSG-JEPA's core move — add training-only auxiliary losses that force the latent to be identifiable
in a target physical basis, without changing inference-time architecture or cost — is exactly what
`L_align` (§4) does in this design, substituting the electromagnetic spectrum for joint angles.
It should be cited as an analogous mechanism from an unrelated domain, motivating why plain
`L_JEPA` alone is unlikely to be sufficient (its own motivation: "their forward-prediction
objectives do not explicitly enforce reliable identifiability... which can limit downstream
planning and policy performance"), not as prior physics-JEPA-for-materials work.

**Goal-conditioned world models.** Action-conditioned and goal-conditioned JEPA world-model work
already establishes `current state + goal/action → future latent`
(https://github.com/facebookresearch/eb_jepa). This project borrows that conceptual grammar but
replaces the physical "action" with the desired electromagnetic response, and replaces future
video state with future structural state.

**Warning from recent predictive-objective literature.** Pendharkar, A.
*Predictive Objectives Discard Exogenous Control-Relevant Features: A Controlled Mechanistic
Study.* arXiv:2606.30068. Shows predictive objectives can discard exogenous but control-relevant
information — directly motivating the three-mechanism goal-sensitivity design in §9, rather than
treating goal-sensitivity as optional. See §9 for the precise 2×2-cell match to this project's
risk.

### 14.1 What is plausibly novel here

The defensible novelty is the combination of: a JEPA-style predictive structural latent; a rich
frequency-domain physical goal representation; partial-to-complete structural prediction; explicit
conditioning over a continuum of context availability; physics-closed-loop latent prediction;
goal-sensitivity enforced through training loss, inference-time guidance, *and* structural
routing together (§9); zero-context physics-only discovery as the limiting case; reuse of a
pretrained geometry–spectrum aligned representation; stochastic latent future-state prediction for
one-to-many inverse design; classifier-free goal guidance ported into a predictive-latent inverse
design setting (§3.5.1); sparse, falsifiable spectral-to-spatial routing as a mechanism rather
than a visualization (§3.5.2); and — new in this version — externally validating that routing
against the frozen surrogate's own Jacobian rather than the model's self-reported attention
weights (§20.1). The claim is about the problem formulation and predictive mechanism, not the
mere use of JEPA.

### 14.2 What must not be claimed as novel

Physics-conditioned inverse design; spectrum encoders; contrastive geometry-physics alignment;
diffusion-based metasurface generation; latent generative modeling in materials; one-to-many
inverse design; conditional generation from spectra alone; JEPA in materials science generally;
classifier-free guidance itself (well established in diffusion — only its application here is
new).

---

## 15. Recommended title direction

Not simply "JEPA for Metasurface Inverse Design" — that oversells JEPA as the entire novelty.
Better candidates:

- *Predictive Physical-State Representations for Goal-Conditioned Metasurface Design*
- *Goal-Conditioned Joint-Embedding Prediction for Partial-to-Complete Metasurface Inverse
  Design*
- *Predicting the Physical Design State: A Joint-Embedding Framework for Goal-Conditioned
  Metasurface Generation*

The first is probably safest, since it does not oversell JEPA as the entire contribution.

---

## 16. The core conceptual statement

JEPA is not used here because embeddings are fashionable. It is used because the object we want
to predict is not raw pixels. The physically meaningful prediction is the latent structural state
that a completed design must occupy. The spectrum defines the goal. The partial geometry defines
the current structural state. The predictor infers the future structural state — using a
bottlenecked context, a goal it can be guided toward or away from at inference, and sparse,
checkable routing between spatial regions and spectral bands. The decoder realizes that state as
a geometry. The electromagnetic surrogate verifies that the realized geometry actually achieves
the goal. This is the conceptual backbone, and every mechanism in this document exists to either
predict that state, realize it, verify it, or prevent the predictor from quietly ignoring the
goal while doing so.

---

## 17. Implementation order

```
MetaDiT reproduction
        |
deterministic JEPA (block masking)
        |
decoder
        |
physics loop
        |
goal-sensitivity (InfoNCE + guidance + sparse routing)
        |
zero-context
        |
stochastic latent
        |
latent diffusion/flow  -- only if the stochastic latent shows a clear need for it
```

At every stage, ask: what failure does this component solve? If no measurable failure is being
solved, do not add the component.

---

## 18. Reference links (verified 2026-08-16)

- MetaDiT paper: https://arxiv.org/abs/2508.05076 (AAAI 2026)
- MetaDiT PDF: https://arxiv.org/pdf/2508.05076
- MetaDiT GitHub: https://github.com/JessePrince/metadit
- MetaDiT dataset and weights: https://huggingface.co/datasets/Hao-Li-131/MetaDiT-AAAI2026
- LeJEPA: https://arxiv.org/abs/2511.08544 · https://github.com/galilai-group/lejepa
- Action-conditioned JEPA / world-model reference: https://github.com/facebookresearch/eb_jepa
- Phys-JEPA (physics-informed JEPA for time series): https://arxiv.org/abs/2606.16076
- PSG-JEPA (physically grounded latent objectives; robotics, see §14 precision note):
  https://arxiv.org/abs/2608.06799
- Polymer-JEPA: https://doi.org/10.1039/D5DD00308C · https://arxiv.org/abs/2506.18194
- Predictive objectives discarding control-relevant information:
  https://arxiv.org/abs/2606.30068

**Note on one unverified citation carried over from the original draft**: an earlier version of
this document also listed arXiv:2606.27014 as "Generalization theory for JEPA world models." That
citation was not independently re-checked against its primary source in this revision. Before
citing it in any paper draft, verify it the same way every other source above was verified —
title, authors, abstract, and actual relevance to the specific claim it's attached to.

---

## 19. One-sentence project definition

Build a Goal-Conditioned Physics JEPA that predicts, under block-structured context masking and
guidance-controllable goal conditioning, the latent structural state required to reach a desired
electromagnetic response from any amount of available geometry context, then decodes that
predicted state into a valid metasurface and verifies it through a learned electromagnetic
forward model.

---

## 20. Additional falsifiable contributions

### 20.1 Externally-validated routing consistency (extends H6 / Ablation J)

The routing-consistency claim in §3.5.2 and H6 validates itself only against its own attention
weights: "does region A consistently route to the same frequency band across many samples." This
is a real improvement over dense-attention post-hoc visualization, but it is still an internal
consistency check — a routing table can be perfectly stable and still be routing to the wrong
frequencies for a reason unrelated to physics (e.g. a positional shortcut the model latched onto
early in training).

**Addition**: compute an independent, model-free sensitivity signal via the frozen EM surrogate's
input-output Jacobian, `J_i = ∂S(f_i)/∂G` evaluated at each training geometry, aggregated per
16×16 spatial block to give a "which frequencies does this region actually affect" ground-truth
map, entirely independent of the learned routing gate. Correlate the learned top-k routing
assignment against this Jacobian-derived map (rank correlation per region, aggregated over the
dataset). This turns H6 from "is routing internally stable" into "is routing stable *and*
correct," which is the stronger and more publishable claim, and it is nearly free — the surrogate
is already frozen and differentiable, so this is a handful of backward passes, not new training.

### 20.2 Goal-token utilization entropy (extends §3.4)

Log `H(softmax(mean attention weight into each of the 16 goal tokens))` across a validation batch
at every Phase-2+ checkpoint. Low entropy (a few tokens absorbing nearly all attention) is a cheap
early warning for the query-pooling collapse flagged in §3.4, and it is observable from epoch 1
of Phase 2 — long before the Ablation J routing analysis would otherwise catch it in Phase 5.
This is the kind of catch-it-early instrumentation the loss-curriculum weight-drift check in
§4.1 already applies to the loss weights; it was missing for the goal-token pooling mechanism
and should be added on the same principle.

### 20.3 Sharper classifier-free guidance diagnostic (extends §3.5.1)

The guidance gap is naturally reported as a scalar norm, `‖P(Z_x,A_goal) − P(Z_x,A_∅)‖`. Two
structures with very different context masks will have very different absolute latent-norm
scales, which makes a raw norm noisy to compare across mask ratios — 20% context and 80% context
are not directly comparable this way. Report instead the **guidance gap as a fraction of
context-conditioned variance**: `‖P(Z_x,A_goal) − P(Z_x,A_∅)‖ / σ(Z_x)`, computed per mask-ratio
bucket. This gives a single normalized number that is comparable across the entire context
curriculum (§7, Phase 6) and turns the classifier-free guidance mechanism into a diagnostic that
can be plotted as one curve across 20/40/60/80/100% masking, directly showing whether
goal-sensitivity degrades as more context becomes available (the intuitively expected direction:
with more visible geometry, the model has less need for the goal, and the design should show —
and report — that trend explicitly, rather than leaving it as an implicit assumption).