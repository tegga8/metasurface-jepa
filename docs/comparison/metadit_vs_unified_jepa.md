# MetaDiT (diffusion) vs the Unified 192-D JEPA — full comparison

Compares the released **MetaDiT-S** diffusion transformer (arXiv:2508.05076, AAAI
2026) against this repo's **unified occupancy–parameter–spectrum JEPA** on the same
task, data, and frozen EM surrogate.

Every model-size figure below was **measured live** in this session (built from the
released weights / shipped config), not copied from a paper:

```
ours  (build_unified_model, configs/unified.yaml):  TOTAL 18,900,100  TRAINABLE 11,365,956
MetaDiT-S (DIT_MODEL['metadit_s']):                 37,197,644
released spectrum encoder:                          4,526,144
released EM surrogate (surrogate_s3):               6,328,698
```

Performance numbers: ours from the Phase 1 baseline (`docs/benchmarking/BASELINE.md`,
Kaggle, full-epoch checkpoint); MetaDiT-S from its paper Table 2 and the local
reproduction (`checkpoints/phase0/seed0_metric.json`).

---

## 1. Paradigm

| | MetaDiT-S | ours (unified JEPA) |
|---|---|---|
| family | generative **diffusion** transformer (DDPM/DiT) | predictive representation learning (**JEPA**, EMA-teacher self-supervised) |
| mapping | `spectrum → geometry`, **stochastic** (sampled) | `spectrum + partial context → geometry`, **deterministic** |
| inference | ~500 denoising steps (×2 with CFG) | **1 forward pass** |
| objective | denoising score matching (+ frozen contrastive spectrum encoder as condition) | JEPA/VICReg on masked tokens + scalar L1 + occupancy BCE + physics (frozen surrogate) |
| teacher | — (no EMA teacher) | EMA copy of the occupancy/scalar encoders (stop-grad target) |
| conditioning | spectrum (CFG dropout) | spectrum (goal tokens + physics FiLM), occupancy mask, scalar known/unknown |
| partial-context edits | not supported (S→G only) | supported — retrofit, metrology, constrained edits (scenarios B/C) |

## 2. Architecture and size

| | MetaDiT-S | ours |
|---|---|---|
| backbone | DiT, depth **12**, hidden **384**, heads **6**, patch **2** | occupancy encoder depth **6** + fusion depth **2** + predictor depth **8**, hidden **192**, heads **6**, patch **4** |
| latent/token grid | 3×32×32 (symmetric quadrant) → **256 tokens** | 64×64, single-channel occupancy → **256 tokens** |
| latent width | 384 | 192 |
| trainable params | **37.20 M** (DiT) | **11.37 M** |
| total params (own weights) | 37.20 M DiT (+4.53 M spectrum encoder) | 18.90 M total (incl. 3.01 M frozen EMA targets + 4.53 M frozen released encoder) |
| frozen shared components | spectrum encoder (inside `y_embedder`) | spectrum encoder (inside `spectrum_path`) — same released 4.53 M weights |
| physics scorer | released `surrogate_s3` (6.33 M), frozen | same surrogate, frozen |

**Size takeaway:** our trainable footprint is **~3.3× smaller** (11.4 M vs 37.2 M)
at the same 256-token grid, and our *total* is ~half the DiT alone. Both share the
same frozen spectrum encoder and surrogate, so the *effective* ecosystem is
MetaDiT ≈ 48 M (DiT + spec enc + surrogate) vs ours ≈ 30 M (model + spec enc +
surrogate).

## 3. Data and training

| | MetaDiT-S | ours |
|---|---|---|
| training split | 139,906 designs (same MetaDiT release) | **same** 139,906 designs |
| geometry input | 3-channel broadcast tensor (r/5, h, l/3) | single-channel occupancy + 3 explicit scalars |
| spectrum | 301-point complex, `[2,301]` | same |
| steps run here | not reproduced (released weights) | **70,000 steps** (1 epoch, batch 2) |
| GPUs | Kaggle/Colab T4/P100/A100 | Kaggle T4 |

## 4. Spectral fidelity — the one apples-to-apples axis

Identical metric, identical frozen surrogate, both on the MetaDiT **test** split.

| arm | MAE ↓ | AAE ↓ | AAE&2 ↓ | AAE&4 ↓ |
|---|---|---|---|---|
| **Ours (Scenario A, n=17,489)** | **0.0673** | **40.535** | **42.14** | **43.95** |
| MetaDiT-S (paper) | 0.0801 | 48.2495 | 58.80 | 68.73 |
| MetaDiT-S (reproduced, seed0) | 0.0803 | 48.34 | — | — |
| vanilla DiT (paper) | 0.1677 | 100.9437 | — | — |
| AVG1 mean-spectrum (paper) | 0.5860 | 352.7424 | — | — |
| NN retrieval (ours, n=512) | **0.0551** | **33.161** | — | — |
| surrogate floor (ours) | 0.0065 | — | — | — |

- **Ours beats MetaDiT-S on every column** (−16 % MAE, −16 % AAE, −28 %/ −36 % AAE&K).
- Honest caveat: **NN retrieval still beats ours** (0.0551 vs 0.0673) on raw spectrum
  error — the model is not yet ahead of a trivial retrieval baseline.
- The AAE&K rows for ours use the **latent-jitter analogue**, not diffusion seeds
  (`is_diffusion_seed_diversity: false`) — see `docs/benchmarking/README.md` §3.

## 5. Axes beyond spectral fidelity

A JEPA and a diffusion model should not be judged on one number. These are the
standard axes that apply to *both* families, and where each stands:

| axis | MetaDiT-S | ours |
|---|---|---|
| structural fidelity (occupancy IoU/F1) | not reported (geometry is generated, not scored structurally) | **IoU 0.748 / F1 0.856** (Scenario A, masked region) |
| condition adherence | CFG scale sweep (paper); no null/shuffled control | **real>shuffled on 99.4 %** of hard-stratum samples; guidance gap ≈ 185 |
| determinism / reproducibility | stochastic — each seed differs | **deterministic** — one output per input |
| generative diversity / mode coverage | multi-seed sampling gives diversity (AAE&K captures worst-of-K) | **deterministic** → diversity only via injected jitter/CFG (a structural JEPA limitation) |
| robustness (worst-of-K) | AAE&2 58.8, AAE&4 68.7 | 42.1 / 44.0 (analogue) |
| parameter fidelity | parameters implicit in the generated 3-channel grid | explicit scalar heads, **scalar MAE 0.088** (≈ 2 % of the r≈4.25 range; 0.5–1.0 range for h) |
| partial / constrained generation | S→G only | B (partial parameters, gate 0.967), C (retrofit, gate 0.898) |
| inference cost | ~500–1000 net evals per design | **1 net eval per design** |
| collapse behaviour (JEPA-specific) | n/a | occupancy fraction std 0.067, not collapsed; **scalar path at chance (0.51/0.45)** |

## 6. What must NOT be over-claimed

- **Resolution differs.** MetaDiT operates on the 32×32 symmetric quadrant; ours
  generates native 64×64. The MAE/AAE comparison is fair only because both are
  scored by the *same* frozen surrogate on the *same* 2×301 target. It is not a
  claim that a 64×64 generation is "harder".
- **Generative vs deterministic.** MetaDiT is a sampler (many valid designs per
  spectrum); ours returns a single design. Diversity metrics are therefore not
  symmetric — a diffusion model is expected to win on mode coverage by construction.
- **Paper vs measured.** The MetaDiT numbers are the paper's (reproduced within
  ±0.25 % for seed0 only). Our numbers are measured locally on one checkpoint.
- **Same frozen surrogate for both** — so a systematic surrogate error cancels in the
  *comparison*, but neither number is a physical measurement.
- **NN retrieval beats ours** on raw MAE — do not report our MAE as SOTA.

## 7. Headline summary

| | MetaDiT-S | ours |
|---|---|---|
| trainable params | 37.20 M | **11.37 M** |
| inference cost | ~500–1000 evals | **1 eval** |
| test MAE (paper metric) | 0.0801 | **0.0673** |
| test AAE | 48.25 | **40.54** |
| AAE&2 / AAE&4 | 58.8 / 68.7 | **42.1 / 44.0** |
| partial-context support | no | **yes (B/C)** |
| scalar dependence | n/a | **FAIL (at chance)** |
| generative diversity | **strong (by construction)** | limited (deterministic) |

## 8. Reproduce

```
# ours (this repo, test split, Scenario A)
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
  --checkpoint checkpoints/unified/latest.pt --split test --scenario A \
  --samples 0 --candidates 4 --nn-samples 512
# MetaDiT-S baseline (paper metric, released weights)
python scripts/eval/reproduce_metadit_baseline.py
```

Checkpoint (this stage) saved to Kaggle:
**`tejaspbiradar/metasurface-jepa-192d-full-epoch-ckpt`** (`full_epoch_final.pt`, 188 MB).
