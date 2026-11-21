# Unified Occupancy–Parameter–Spectrum JEPA (192-D)

## Architecture, Problem Statement, Innovations, Results & Open Issues

**Document date:** 2026-10-02
**Repository:** `D:\projects\metamaterials_rewrite` (WSL: `/mnt/d/projects/metamaterials_rewrite`), branch `work-192d`, HEAD `1478026`
**Prepared from:** `docs/implementation/unified_jepa/architecture_v5.md` (architectural authority), `ARCHITECTURE_AUDIT_192D.md`, `AUDIT_REPORT_192D.md`, `checkpoints/unified/REPORT.md` (cloud run record), `configs/unified.yaml`, a direct read of the live source tree, and the operator-supplied comparison tables of 2026-10-02.
**Verification performed this session:** full test suite executed on 2026-10-02 — **300 passed, 23 skipped, 0 failed** (Windows Python 3.14 + torch 2.14.0+cpu; the 23 skips are the loud data/CUDA guards, see §6.3).

---

## 0. Executive summary (the two questions at the top)

**What problem are we tackling?**
Inverse design of metasurface unit cells: given a target optical spectrum (301 wavelengths, real + imaginary), produce the 64×64 binary geometry **and** the three physical parameters of the cell — deterministically, in one pass, and in partial-information regimes (masked geometry, partially known parameters) that the diffusion baseline mathematically cannot address.

**What solution are we making?**
A single frozen-encoder, 192-D **joint-embedding predictive architecture (JEPA)** — occupancy encoder + scalar encoder + frozen released spectrum encoder + fusion + predictor — trained with a VICReg objective, a **frozen differentiable EM surrogate in the loss loop** (physics consistency, no spectrum decoder), EMA targets, and an explicit masking/known-scalar curriculum. One forward pass at inference, deterministic, ~3.3× fewer trainable parameters than the MetaDiT-S diffusion baseline, and it currently beats that baseline on every official test metric (MAE 0.0673 vs 0.0801, AAE 40.54 vs 48.25, AAE&2 42.14 vs 58.80, AAE&4 43.95 vs 68.73), while passing all three scenario gates.

**The honest headline.** Everything above is real and measured — except one thing, which is the current open scientific problem: **scalar dependence is not demonstrated.** With the geometry fully masked, varying the known scalar parameters moves the predicted design no more than chance on a paired per-sample win-rate test (0.506 / 0.451 one/two-known on test; 0.5004 / 0.5106 on the validation split before the first fix). Conditioning reaches the encoder, the predictor attends it heavily, but the effect does not propagate into the output. Two designated fixes have been built: door (a) — a direct read-out objective on the scalar summary token — moved the gates decisively off chance (0.639 / 0.569) but not past the 0.75 bar; door (b) — scalar FiLM into the predictor — is wired, enabled, and **not yet measured**. This is stated plainly rather than hidden: the model is a genuine, competitive inverse-design engine whose scalar-conditioning path is its known weak point.

---

## 1. Problem statement

### 1.1 The physical task

A metasurface unit cell is described by a binary `64×64` pixel pattern (which pixels are filled) plus three global physical parameters:

| symbol | name | meaning in the broadcast encoding |
|---|---|---|
| `l_lattice` | lattice constant | channel 2, divided by 3.0, spatially dense |
| `h_atom` | atom height | channel 1, divided by 1.0, on occupied pixels |
| `r_atom` | atom radius | channel 0, divided by 5.0, on occupied pixels |

The forward problem — geometry → spectrum — is solved by a released, frozen electromagnetic surrogate (6.33 M parameters, `surrogate_model.bin`). The **inverse problem** — spectrum → geometry + parameters — is what this project builds, and it is one-to-many by nature (multiple geometries can produce similar spectra), which is why it has historically been attacked with diffusion.

### 1.2 The dataset

MetaDiT (AAAI 2026) dataset, staged for cloud runs:

- `pattern` `(64, 64, N)` int8 binary — the geometry
- `parameter` `(N, 3)` — `[l_lattice, h_atom, r_atom]`
- `real` / `imag` `(N, 301)` — the spectrum, consumed as `[2, 301]`
- Splits: **train 139,906 / validation 17,488 / test 17,489** samples
- Shared frozen assets: released spectrum encoder (4.53 M) + EM surrogate (6.33 M)

### 1.3 Why the baseline is not the end state (the gap this project fills)

The MetaDiT-S baseline is a DiT-12L diffusion model (`3×32×32` condition grid → `64×64` output) with a spectrum encoder conditioning. It is:

1. **Stochastic and slow at inference** — ~500–1000 denoising evaluations per design.
2. **S→G only** — it cannot consume partially observed geometry, so it cannot perform masked completion, and it cannot infer partial parameters jointly with the geometry.
3. **Not instrumented for conditioning necessity** — "the output looks right" is not the same as "the conditioning caused it".

A previous 384-D JEPA attempt in this repo (retired 2026-09-13, recoverable at commit `f557eb6`) collapsed; the current architecture is the designed replacement, with the failure modes it must be able to *detect* written into the validation gates from the start.

### 1.4 The three deployment scenarios (never pooled)

| scenario | mask ratio | scalars | what it means |
|---|---|---|---|
| **A — pure inverse design** | 1.0 (full mask) | all unknown | spectrum → complete design. The hard stratum; the acceptance gate. |
| **B — partial parameter** | 0.5 | exactly one known (rotating l/h/r) | design with partial parameter knowledge |
| **C — retrofit / completion** | 0.25 | all known | fill in a mostly-known design |

The gate is defined on **scenario A** (the hard stratum: full occupancy mask + all scalars unknown). Pooled metrics are explicitly not a gate.

---

## 2. The solution: unified occupancy–parameter–spectrum JEPA

### 2.1 What it is in one paragraph

One model, one forward pass: factorize the geometry into a **single-channel occupancy** grid and **three explicit scalars with known/unknown flags**; encode both (scalars producing per-block FiLM parameters for the occupancy encoder, plus one summary token); condition on the target spectrum through a **frozen released spectrum encoder** (global mean-pooled `c_physics` + 16 learned goal tokens); fuse, then predict masked occupancy tokens and the scalar summary from a 257-query predictor (GCLCT); decode to occupancy logits and scalar values. Training signals: VICReg latent matching against EMA targets, scalar L1 on unknown positions, occupancy BCE on masked pixels, plus a **physics-consistency loss** that decodes a geometry, runs the frozen differentiable EM surrogate, and compares the predicted spectrum to the true one. No spectrum decoder exists anywhere — physics is the reward, never a generator.

### 2.2 Data contract (Phase 1)

```
geometry [3,64,64]  ──factorize──►  occupancy [1,64,64] ∈ {0,1}
                                   scalars   [3] raw values (l, h, r)
                                   (known/unknown flags sampled per batch)
target spectrum [2,301] ──────────►  [frozen released spectrum encoder]
```

- `factorize_geometry`: `occ = (ch0≠0) | (ch1≠0)`; `l = G[:,2,0,0]·3` (dense channel), `h = G[:,1].amax()`, `r = G[:,0].amax()·5`. Exact inverse: `assemble_geometry`. Round-trip is tested on real data.
- The `[3,64,64]` broadcast tensor exists **only** at the frozen surrogate boundary (`assemble_metadit_geometry`). It is never a representation to mask or predict against — that was the retired design's mistake.
- **Missingness is an explicit binary flag**, never a sentinel value: the network input is the 6-dim `[value, known]×3` vector; decode-time FiLM uses the same convention so a known value and a predicted value are distinguishable.

### 2.3 Architecture, end to end

```
occupancy (1,64,64) ──mask──► [OccupancyEncoder: conv k4/s4 → 16×16 tokens ×192,
                               sincos pos-embed, 6 transformer blocks, FiLM per block]
                                        │  ▲ FiLM (γ,β) per block, identity at init
scalars ──[ScalarEncoder: 6→128→128]────┘
         │  └─► scalar_summary token (1,192)
         └─► deep-copied into scalar_mlp_ema (target-side FiLM only, never decoded)

target spectrum (2,301) ─► [released spectrum encoder — FROZEN, 4.53M] ─► A_local (301,256)
         ├─ mean-pool ─► c_physics (384)              (FiLM conditioning, every predictor block)
         └─ 16 learned queries cross-attend ─► a_goal (16,384)
              (goal_mode="null" ⇒ both zeros, encoder not even run — CFG branch)

z_x (256,192) ⊕ a_goal ↓192 ⊕ scalar_summary → [FusionEncoder, 2 blocks] → fused (273,192)

queries (visible tokens + mask_token + 1 scalar query = 257)
   → [GCLCT predictor, 8 blocks: self-attn + cross-attn over fused KV + c_physics FiLM]
   → z_hat (257,192)
        ├─ 256 occupancy tokens → [OccupancyDecoder: conv up 16→32→64, FiLM from 6-dim scalars]
        │                       → logits (1,64,64)  → BCE on masked pixels
        └─ 1 scalar query token → [ScalarDecoder: 192→64→1 ×3]
                                → scalar_pred (3)   → L1 on unknown positions

EMA targets (frozen, momentum 0.996→0.999):
   ema(occupancy encoder)(full unmasked occupancy) + scalar_mlp_ema(true scalars) → z_y (256,192)

physics: decode_geometry(z_hat, scalar_pred | known-substituted) → geometry (3,64,64)
         → FROZEN EM surrogate (differentiable, STE) → Ŝ (2,301)
         → per-sample-std-normalized SmoothL1 vs true spectrum        (L_phys)
```

**Inference reality:** 1 forward pass produces the occupancy and the scalars; assembly of the deployed geometry is a deterministic decode (threshold + retention of visible pixels + exact scalar substitution). No sampling, no denoising loop.

### 2.4 Component and parameter table `[code, built from shipped config]`

| module | params | trainable |
|---|---|---|
| `occupancy_encoder` | 2,668,608 | 2,668,608 |
| `scalar_encoder` | 339,392 | 339,392 |
| `spectrum_path` (released encoder + goal read-out) | 4,991,040 | 464,896 |
| `fusion_encoder` | 962,368 | 962,368 |
| `predictor` (GCLCT, 8 blocks) | 6,630,272 | 6,630,272 |
| `occupancy_decoder` | 262,785 | 262,785 |
| `scalar_decoder` | 37,251 | 37,251 |
| `ema` (target copy) | 2,668,608 | **0** (frozen) |
| `scalar_mlp_ema` (target copy) | 339,392 | **0** (frozen) |
| **total** | **18,899,716** | **~11.37 M** |

### 2.5 Shipped configuration (`configs/unified.yaml`, verified live)

```
hidden 192 · heads 6 · geo_depth 6 · predictor_depth 8 · goal_tokens 16 · spec_dim 256
EMA 0.996 → 0.999
loss: λ_inv 25 · λ_var 25 · λ_cov 1 · λ_scalar 1 (L1, unknown only)
      λ_occ 1 (BCE, masked pixels) · λ_summary 1 (door (a)) · λ_phys 3.32 (ramped 500 steps)
curriculum: mask ratios {0.25, 0.5, 0.75, 1.0} probs {0.25, 0.35, 0.25, 0.15}
            scalar regimes all_known / all_unknown / mixed (equal thirds)
            strata: easy = 0.25 mask + all known; hard = 1.0 mask + all unknown
staging: phase D, physics_use_ste true (enforced), scalar_predictor_film true (door (b))
train: batch 2, grad_accum 1, AdamW lr 3e-4, wd 1e-4, warmup 100, clip 1.0
       full epoch ≈ 70,000 steps · guidance_dropout 0.1
eval: n_samples 32 (batch gate) · primary gate threshold 0.75
```

### 2.6 The physics loop (L_phys) — design and measured facts

- The released EM surrogate is frozen (`requires_grad_(False)`, `.eval()`) but **not** wrapped in `no_grad` — gradient flows *through its input* (the decoded geometry) back into the model. This is the core trick that makes physics a training signal without any spectrum decoder.
- **STE is load-bearing** `[measured]`: the soft-occupancy path is a silent dead end — `L_phys` looks healthy (18.58) while **zero** student parameters receive gradient, because the soft field is ~96% out of distribution for the surrogate. With the straight-through estimator (hard occupancy forward, soft backward), `L_phys = 0.336` and 360 student parameters receive gradient. Config validation hard-rejects `λ_phys > 0` with STE off.
- Normalization: both predicted and target spectra are divided by the **per-sample** target std (clamped at 1e-3, hard error below), then SmoothL1. This handles the wide dynamic range of spectral amplitudes.
- `L_phys` is **skipped on goal-dropped (null) batches** — its target is the very spectrum that was dropped, so training it there would push the unconditional branch toward outputs it cannot infer.
- At the final trained state, `L_phys` is **66.15%** of the gradient budget `[measured]`; the weight 3.32 came from a calibrated 5-arm sweep and is recorded as a **lower bound** (the curve was still improving at the top of the swept range).

### 2.7 Validation gates (what "done" means — `architecture_v5.md` §8)

1. **Real vs null vs shuffled, stratified.** The primary gate: on the hard stratum, the per-sample win rate `real < shuffled` must exceed **0.75** (paired statistic — one catastrophic sample cannot flip it; the mean criterion is reported alongside but is not decisive). Never pooled across scenarios.
2. **Scalar dependence, stratified the same way** — varying the known scalars must change the output *correctly* (win rate against a deranged-scalar control), not merely change it.
3. **Occupancy collapse** — IoU/F1 on the occupied class and per-sample occupancy-fraction variability (not pixel accuracy).
4. **Generative diversity** — perturbing the target spectrum must move the decoded design monotonically and proportionally.
5. **§8.2 diagnostic re-runs on the 192-D encoder** — deferred (the legacy scripts were retired; need porting).

---

## 3. Innovations (what is actually new here)

1. **One-shot deterministic inverse design instead of diffusion.** 1 forward pass vs ~500–1000 denoising evaluations; deterministic outputs; ~3.3× fewer trainable parameters than MetaDiT-S (11.37 M vs 37.20 M) — while beating it on all official test metrics (§4).
2. **Occupancy + explicit scalar factorization with explicit known/unknown flags.** The geometry is a single binary channel plus three raw scalars; missingness is a first-class binary input, not a sentinel. This is what makes scenarios B and C possible at all — partial geometry and partial parameters are the same machinery as full inverse design.
3. **A frozen differentiable EM surrogate inside the training loop.** Physics is used as a *loss* (with a straight-through estimator for the discrete geometry), never as a decoder or generator. This gives the model a dense, physically grounded training signal without learning a spectrum decoder, and it is measurably what makes the conditioning real — STE is load-bearing, and the soft path is a measured dead end.
4. **Dual EMA targets with a deliberately physics-free answer key.** An EMA of the occupancy encoder plus a second small EMA (`scalar_mlp_ema`) that conditions the *target-side* FiLM on all-known scalars. The target latent never sees the spectrum — documented with its consequence (the JEPA loss can only *reward*, never *require*, conditioning) rather than left implicit.
5. **A masking/conditioning curriculum built for the failure modes being probed.** Calibrated block masking (±2% of requested ratio, measured 0.2462/0.5002/0.7582/1.0000 achieved), optional sensitivity-weighted placement, three scalar regimes, goal dropout (10%) for the CFG branch — each stratum reported separately, never pooled.
6. **Measurement methodology as a first-class artifact.** Real/null/shuffled tri-control; **paired per-sample win rate** as the primary statistic (a mean can be and was flipped by one 25× outlier); stratification on the hard stratum; attention-mass probes over the scalar summary; gradient attribution per loss term per module; an EMA-tracking audit; mask calibration audit. Several earlier readings were reversed *by* this instrumentation (e.g. "the objective degrades at scale" was a BatchNorm eval-mode artifact, 759×/364×).
7. **Door (a): an auxiliary read-out objective that gives the scalar encoder a direct gradient path.** The scalar summary token previously had no objective of its own; a small head reconstructs the *known* scalars from it (`λ_summary = 1`). Measured effect: the encoder's gradient from this term alone is 4.2× what the scalar loss delivers (0.17418 vs 0.04174), zero gradient to any other module, and the scalar gates moved from chance (0.5004/0.5106) to 0.639/0.569 `[measured, 17,488 samples]`.
8. **Door (b): scalar conditioning into the predictor's per-block FiLM** (zero-initialized ⇒ exact identity at step 0). Wired, enabled, not yet measured — the structural counterpart to door (a), addressing the barrier downstream of the encoder.

---

## 4. Results vs MetaDiT-S

### 4.1 System-level comparison

| | **MetaDiT-S (diffusion)** | **ours (unified JEPA)** |
|---|---|---|
| backbone | DiT 12L, hidden 384, heads 6, patch 2 | enc 6 + fusion 2 + pred 8, hidden 192, heads 6, patch 4 |
| token grid | 3×32×32 → 256 | 64×64 → 256 |
| trainable params | 37.20 M | **11.37 M (~3.3× smaller)** |
| total own weights | 37.20 M (+4.53 M spec enc) | 18.90 M (incl. 3.01 M frozen EMA) |
| shared frozen | spectrum encoder 4.53 M + surrogate 6.33 M | same two |
| inference | ~500–1000 denoising evals | **1 forward** |
| test MAE / AAE | 0.0801 / 48.25 | **0.0673 / 40.54** |
| AAE&2 / AAE&4 | 58.8 / 68.7 | **42.1 / 44.0** |
| partial-context (B/C) | no (S→G only) | **yes** |
| determinism | stochastic | **deterministic** |
| scalar dependence | n/a | **FAIL (at chance)** |

### 4.2 Baseline results table (the point of Phase 1)

| | **ours** | **MetaDiT-S** | verdict |
|---|---|---|---|
| MAE (Scenario A, test, n=17,489) | **0.0673** | 0.0801 | we beat it |
| AAE | **40.535** | 48.2495 | we beat it |
| AAE&2 / AAE&4 | **42.14 / 43.95** | 58.80 / 68.73 | we beat it |
| A / B / C gate win rate | **0.994 / 0.967 / 0.898** | — | **PASS** |
| scalar-dep. one/two-known | 0.506 / 0.451 | — | **FAIL (at chance)** |

**Provenance and context for these numbers:**

- The MetaDiT-S MAE/AAE figures are its paper Table 2 line (0.0801 / 48.2495); the local Phase-0 reproduction ran the released baseline on the released seed-0 generation through the unmodified `external/metadit/metric.py` and reproduced it at **0.080295 / 48.3375** (+0.24% / +0.18%) — i.e. the comparison target is verified, not assumed. AAE&2/AAE&4 need 4 seeds; only seed-0 is released, so those two are paper figures.
- Our numbers are the operator-supplied test-split readings of 2026-10-02. The in-repo cloud record (`checkpoints/unified/REPORT.md`) holds the matching validation-split measurements at comparable values: full-split A win rate **0.9939** (n=17,488), full-split means real 0.1196 vs shuffled 0.6084 (median 0.0749 vs 0.6023), and it now **beats the nearest-neighbour retrieval baseline on the mean** (0.1196 vs 0.1254). The 0.506/0.451 scalar numbers differ slightly from the recorded 0.5004/0.5106 (different split/power) but land on the same conclusion.
- Scenario gate at n=32: A 0.9688, B 0.9375, C 0.9063 — PASS. A's n=32 mean (0.8779) is one pathological sample (real 25.06); the win rate is unmoved.
- All numbers above are `[documented]` (cloud-run records) except where marked `[measured]`/`[code]` in this document; the local machine cannot re-run data-dependent evaluation because the splits/weights are not staged (§6.3).

---

## 5. Current issues and open problems

### 5.1 Scientific issues (ranked by importance)

**1. Scalar dependence is not demonstrated — the single open scientific problem.**
With the occupancy fully masked, varying the known scalars moves the predicted design no more than chance: one-known **0.5004** (95% CI [0.4930, 0.5078] — contains 0.5), two-known **0.5106** on the full 17,488-sample validation split `[measured]`. The instrumentation localises it precisely:

- The conditioning **reaches** the scalar encoder (summary token moves ~11% with the scalars).
- The predictor **attends it heavily** — the scalar query puts 34.3% of its attention mass on the summary token, **93.6×** its uniform share `[measured]`.
- It does not **propagate**: `scalar_pred` moves 0.000145 under scalar perturbation vs 0.246 under spectrum perturbation (~1,700×), and `L_scalar` delivers only 0.0036 of gradient to the scalar encoder vs 0.6932 to the decoder.
- A 100× scale-up of the token moved the effect only ~5×, which **ruled out magnitude** as the binding constraint ⇒ the problem is **structural** (how scalar information enters the network), not a scale or weighting tweak.

What has been done: **door (a)** (summary read-out, `λ_summary = 1`) — moved both strata decisively off chance to **0.639 / 0.569**, still below the operator-set 0.75 bar `[measured]`; the pre-registered rule was "gates warranted if gradient rises but gates do not" — the gradient rose and gates moved, partially. **Door (b)** (scalar FiLM into the predictor, zero-init identity) is enabled in the shipped config but has **no measured result yet**. Also unmeasured: door (a) at one full epoch (the 10k-step run is not the shipped 70k-step model).

Constraint any fix must respect: the scalar prediction must not be able to *read* the conditioning values, or a partially-known sample could copy a known scalar into an unknown position (leakage). "Let the scalar head read the summary token directly" is therefore not automatically safe.

**2. The catastrophic tail is a spectrum-space coverage gap, and the guard for it is not implemented.**
27 of 17,488 validation samples (0.15%) score 1–30 on spectrum error (median 0.0749). They are **extrapolations**: their distance to the nearest training spectrum is 6.28 (median 5.73, min 2.99) vs 2.17 for random controls — every one farther than the control mean `[measured]`. The identified fix is a read-out guard at inference (nearest-training-spectrum distance flagging OOD inputs instead of silently emitting an unusable design) — suggested, **not implemented**.

**3. `λ_phys = 3.32` is a lower bound, and physics already holds 66% of the gradient budget.**
The sweep's improvement was still rising at the top of the swept range. A further bounded sweep upward is the recorded next step; the config deliberately does not raise it silently. `λ_cov` (3.65% share) is explicitly not to be touched `[measured: per-term gradient attribution at the final state]`.

**4. The projector-absorption hypothesis is consistent but not ablated.**
A shared VICReg projector with BatchNorm across the student and target branches (running stats fit neither; measured train-vs-eval gap 759×/364×) could be absorbing representational differences. The trainer works around the measurement artifact (`projector_train_mode`), but the architectural question — should the branches share a projector — is **not settled**.

**5. CFG/guidance adds nothing.** `w = 1` (no guidance) is optimal in every measured arm; `w > 1` diverges by 45× with physics off and is non-monotonic with physics on. The machinery works and is wired (`cfg_forward` is exercised by the evaluator's sweep), but the real/null difference is not a direction worth amplifying `[measured]`.

**6. Deferred spec items.** The §8.2 diagnostic re-runs on the 192-D encoder (VICReg gradient attribution; within-bucket spatial-structure probe beating the old +0.225 margin) are deferred — the legacy scripts were retired with the 384-D path and need porting. The evaluator's `diversity_check` is a determinism check only; the spec's actual spectrum-perturbation probe now exists and passes (scale 0 → exactly 0 movement; 0.01 → 6.19% geometry change, monotone to 10%).

### 5.2 Engineering defects verified in the live tree (2026-10-02)

These are code-level defects I confirmed by reading the current source; none affect the headline results above, but they are real:

| # | defect | where | impact |
|---|---|---|---|
| 1 | `checks["precedence_unverifiable_samples"] = empty` executes **before** `checks` is defined, and the unknown-scalar precedence loop lacks the empty-sample guard | `scripts/train/train_unified.py:1343` vs `1393`, guard missing at `1380` | preflight raises `NameError` (then `IndexError` once fixed) exactly when a decoded occupancy is empty — the case the branch exists to handle |
| 2 | Smoke mode **writes a random-init spectrum encoder to the real configured weights path**, and real mode accepts it on a bare `os.path.exists` | `train_unified.py:174–187` | a smoke run can silently contaminate a later real training run |
| 3 | `--scenario` flag parsed and never forwarded; the authoritative evaluator writes **no artifacts** (stdout JSON only) | `scripts/eval/eval_scenarios.py:751` vs `769` | gate results are not durably recorded by the tool that produces them |
| 4 | Comment claims the door-(b) projection is "constructed either way so checkpoints stay interchangeable" — it is constructed **only when enabled** | `src/assembly.py:152–154` vs `src/predictor/gclct.py:131–136` | toggling `scalar_predictor_film` breaks strict resume/checkpoint interchange |
| 5 | `goal_mode="shuffled"` is a no-op alias for `"real"` (only `"null"` is special-cased); `validate_goal_mode` rejects `"shuffled"` | `src/encoders/spectrum_encoder.py:104` | harmless in practice (the evaluator deranges the spectrum tensor itself) but the documented mode set is inconsistent |
| 6 | Config validation accepts `independent`/`correlated` scalar regimes; the masker bank raises on anything but `all_known`/`all_unknown`/`mixed` | `train_unified.py:88` vs `273–283` | late failure instead of validation failure |
| 7 | Dead code / stale references: `goal_token_entropy` (no callers, legacy 64-token KV assumption), `epoch_sampler.py` (test-only), 384-D anchor in `representation_health.py`, `__pycache__` for deleted legacy scripts, unused imports in the trainer | various | maintenance hazard; none reachable in the live path |
| 8 | Trainer validation does not report `L_occ`/`L_summary`; checkpoint `best_prediction` uses a `"ratio"` metric the trainer never passes (always 0.0); `--resume` with a missing path silently trains from scratch | `train_unified.py` / `train/engine.py` | instrumentation gaps, not correctness bugs |

### 5.3 Measurement / local-environment limitations

- **Local machine is dev-only** (16 GB RAM, 4 GB VRAM): it cannot run real-data training or the acceptance evaluation. Runtime path chosen: cloud (Kaggle) for all training/eval; the local suite is the regression gate.
- **Local staging is partial**: only `data/metadit/weights/spec_encoder.pth` is staged; the dataset splits and the surrogate are absent. Consequences: 23 tests skip loudly (by design — rule 7 of the repo), and the preflight / any data-dependent path cannot run locally.
- **No trained checkpoint in the repo**: the full-epoch model lives as a private Kaggle dataset (`anosvol/metasurface-jepa-192d-full-epoch-ckpt`, ~188 MB); `logs/` is empty locally; the canonical run record is `checkpoints/unified/REPORT.md`.
- **A local test-suite run is green as of 2026-10-02**: **300 passed / 23 skipped / 0 failed** (Windows-side Python 3.14 + torch 2.14.0+cpu; the WSL interpreter has no torch). The 23 skips are the expected data/CUDA guards. This is a regression signal only — it is not a scientific gate.

---

## 6. Status ledger (claim → evidence grade)

| claim | grade |
|---|---|
| Architecture implemented end-to-end and runnable; 300-test regression suite green | `[measured locally 2026-10-02]` |
| Beats MetaDiT-S on test MAE 0.0673 vs 0.0801, AAE 40.54 vs 48.25, AAE&2/4 42.14/43.95 vs 58.80/68.73 | `[documented, operator tables 2026-10-02]`; MetaDiT-S line independently reproduced in Phase 0 within 0.24% |
| Scenario gates A/B/C pass the 0.75 win-rate bar | `[documented]` — A 0.9688@n=32 / 0.9939 full split; B 0.9375 / 0.967; C 0.9063 / 0.898 |
| Deterministic 1-forward inference; ~3.3× fewer trainable params | `[code + documented]` |
| STE physics path is load-bearing; soft path is a measured dead end | `[measured]` — 0 vs 360 student params with gradient |
| `L_phys` = 66.15% of gradient budget; λ=3.32 a lower bound | `[measured]` — per-term attribution at final state; sweep curve still rising |
| Mask calibration ±2%; EMA tracking ≤0.8% | `[measured]` |
| Door (a) works as designed (gradient 0.17418 to the scalar encoder, gates 0.5004→0.639 / 0.5106→0.569) | `[measured, 17,488 samples]` |
| Scalar dependence fails the 0.75 bar in both strata | `[measured]` — the open problem |
| Door (b) enabled in shipped config, unmeasured | `[code / config]` |
| Tail = spectrum-space extrapolation; OOD guard unimplemented | `[measured]` for the finding; guard absent |
| §8.2 diagnostic re-runs on 192-D | **deferred** (legacy scripts retired, need porting) |
| Local dev cannot run data-dependent evaluation | `[verified]` — staging absent, 23 skipped tests |

---

## 7. Next steps (in order)

1. **Measure door (b)** — the enabled-but-unmeasured scalar-FiLM route — on the full validation split with the standard gate; this is the cheapest remaining test of the scalar hypothesis.
2. **Measure door (a) at one full epoch** (70k steps) — the 10k reading is not the shipped model.
3. If the scalar gates still fail: escalate to the structural question with the measured constraint in hand (prediction must not read the conditioning), e.g. a design review of how scalar information enters the token set.
4. **Extend the λ_phys sweep upward (bounded)** — the recorded next step; watch the 66% gradient share.
5. **Implement the OOD read-out guard** (nearest-training-spectrum distance) so the 0.15% tail is flagged rather than silently served.
6. **Port the §8.2 diagnostics** to the 192-D encoder and re-run.
7. **Fix the verified engineering defects** (§5.2), each as its own commit with a regression test — preflight `NameError` and the smoke-dummy-checkpoint contamination first.
8. **Sync the 2026-10-02 test-split tables into `checkpoints/unified/REPORT.md`** so the canonical record carries the MAE/AAE/gate numbers, and give the evaluator a `--out` artifact.
9. Bounded ablation of the shared projector (architectural question, needs operator sign-off).

---

## Appendix A — metric definitions (so the numbers are unambiguous)

- **MAE / AAE / AAE&K** — the official MetaDiT metrics, computed by the unmodified `external/metadit/metric.py` over the released surrogate's spectra; AAE&K = amplitude-area error under K-fold binning.
- **Spectrum error (internal gates)** — per-sample L1 between the surrogate-rendered predicted spectrum and the true target spectrum, divided by the per-sample target std (clamped ≥ 1e-6).
- **Primary gate statistic** — paired per-sample win rate: fraction of samples where `real` conditioning achieves a strictly lower spectrum error than the `shuffled` control; gate is `> 0.75`. The paired mean criterion is reported alongside but is not decisive (one outlier can flip a mean; it did: 0.8779 vs a 0.1196 full-split mean).
- **Scalar dependence** — same win-rate construction, comparing true conditioning scalars against a deranged control (`scalar_pred` error metric under the same hard stratum).
- **Occupancy collapse** — IoU/F1 on the occupied class and per-sample predicted occupancy-fraction variability (never pixel accuracy).
- **Guidance gap** — `||z_real − z_null|| / std(z_real)`, per sample; near-zero ⇒ the predictor ignores the conditioning.

## Appendix B — sources and where to look further

| artifact | path | role |
|---|---|---|
| Architecture (authority) | `docs/implementation/unified_jepa/architecture_v5.md` | design intent, §8 gates |
| Execution controller | `docs/implementation/unified_jepa/00_MASTER_EXECUTION.md` | phase order and rules |
| Phase instructions | `01_…` – `05_…` in the same directory | per-phase spec |
| Component/wiring audit | `docs/implementation/unified_jepa/ARCHITECTURE_AUDIT_192D.md` | what the code does, with evidence grades |
| Retirement + defect audit | `docs/implementation/unified_jepa/AUDIT_REPORT_192D.md` | legacy retirement, B1–B21 fixes |
| Cloud run record (current state) | `checkpoints/unified/REPORT.md` | §11 λ sweep · §14 full-split · §16 scalar · §19–21 door (a)/tail |
| Baseline reproduction | `checkpoints/phase0/REPORT.md` | MetaDiT-S reproduced within 0.2% |
| Live config | `configs/unified.yaml` | shipped values with rationale comments |
| Trainer / evaluator | `scripts/train/train_unified.py`, `scripts/eval/eval_scenarios.py` | the code that produces every number |
| Cloud runbook | `CLOUD_TRAINING.md` | how runs are staged, resumed, synced |

*No scientific claim in this document is made from "the code runs". Where a claim is a cloud-run record it is marked `[documented]`; where it was measured this session or in the audit records it is marked `[measured]`; where it is a statement about the code it is marked `[code]`. The one genuinely open scientific item is scalar dependence (§5.1.1).*
