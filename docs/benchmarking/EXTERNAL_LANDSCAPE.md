# External landscape — who else solves this task, on what data, and where we stand

Compiled 2026-10-04 by web search + repository audit. Every number carries its
provenance: **[measured]** = produced in this repo, **[cited]** = from the published
paper, **[web]** = verified on the live web page/repo today.

Task under comparison: **inverse metasurface design** — map a target transmission
spectrum to a unit-cell geometry (pattern + continuous parameters).

---

## 0. Bottom line

1. **There is exactly one public dataset that matches our task exactly** — the
   free-form meta-atoms of An et al. 2020 (the one we already train on; MetaDiT,
   MetaSR, MetaDiffusion, I-P DM all sit on it). There is **no second public
   spectrum→64×64-pixel dataset** to test on today.
2. On the only apples-to-apples protocol (MetaDiT's MAE/AAE on the shared test
   split, released surrogate), **our 70k JEPA leads every published number** —
   ours MAE **0.0490** [measured] vs MetaDiT-S **0.0801**/**0.0803**, DiT 0.1677,
   MetaDiff-HR 0.1315, MetaDiff 0.1861 [cited].
3. **But it loses to plain NN retrieval** (0.0296 vs 0.0490) on the same split —
   that is the real bar, and it is ours alone (nobody else reports retrieval).
4. A 2025–2026 wave (MetaSR, I-P DM, masked-diffusion) works on the **same data
   with different metric protocols** → cross-ranking is currently impossible
   without re-running them; their headline MAEs are not the same MAE.
5. Our model is **the only non-generative, single-pass, partially-conditionable**
   entry in the field; every comparator is a diffusion/GAN sampler that cannot
   accept masked occupancies or known scalars as input.

---

## 1. Our task and dataset (verified)

| item | value | source |
|---|---|---|
| task | spectrum `[2,301]` (real/imag) → unit cell | [web] An 2020, OE 28(21):31932 |
| geometry | 64×64 binary pattern (1 = dielectric) | [web] `SensongAn/Meta-atoms-data-sharing` |
| continuous params | `l_lattice` 2.5–3 µm, `h_atom` 0.5–1 µm, `r_atom` (refractive index) 3.5–5 | [web] same |
| spectrum | 30–60 THz, 0.1 THz spacing, 301 points, complex transmission | [web] same |
| simulator | CST Microwave Studio (frequency domain), substrate 2 µm n=1.4 | [web] same |
| scale / splits | 174,883 total → 139,906 / 17,488 / 17,489 (MetaDiT release, 8:1:1) | [measured] `docs/benchmarking/EXTERNAL_BENCHMARK_COMPARISON.md` |
| scoring surrogate | released `surrogate_s3` (6.33 M, frozen) — floor ≈0.0065 MAE | [measured] `BASELINE.md` |
| paper surrogate (for reference) | StarNet-MLP 1.90 M, floor 0.0084 | [cited] MetaDiT Table 1 |

The dataset's home is Sensong An's repo; the actual files are behind a Google
Drive link. MetaDiT released the split files + weights on HuggingFace
(`Hao-Li-131/MetaDiT-AAAI2026`), which is what this repo stages.

**Important:** the field has effectively converged on this one dataset as *the*
benchmark for spectrum-conditioned metasurface inverse design.

---

## 2. The model landscape

### 2.1 Same dataset (freeform meta-atoms, 174,883)

| model | venue / year | family | data use | metrics (their own protocol) | comparable to our MAE? |
|---|---|---|---|---|---|
| **MetaDiT-S** | AAAI 2026 | diffusion transformer + contrastive spectrum encoder | full split, 301 pts, all params | MAE 0.0801 / AAE 48.25 / AAE&2 58.80 / AAE&4 68.73 | **YES** — same metric, same surrogate family, reproduced at 0.0803 |
| MetaDiff (reprod.) | Nanophotonics 2023, re-encoded by MetaDiT | diffusion | same split, coarse spectra | MAE 0.1861 / AAE 112.06 | YES (under MetaDiT protocol) |
| MetaDiff-HR | same, high-res condition | diffusion | 301 pts | MAE 0.1315 / AAE 79.14 | YES |
| vanilla DiT | MetaDiT baseline | diffusion | 301 pts | MAE 0.1677 / AAE 100.94 | YES |
| **MetaSR** | Opt. Laser Technol. 2026 | U-Net DDPM/DDIM + Transformer PIM + TPM | same 174,883, 26 or 301 pts | MAE **0.005261** / MSE 0.0005637 ("optimal designs"); DDIM: hundreds of candidates sub-second | **NO** — their `complex_components_v1` protocol; different denominator; do not compare |
| I-P DM | IEEE 2024 | image+parameter diffusion | same family | not extracted | NO |
| MetaDiffusion | Nanophotonics 12(20):3871, 2023 | DDPM | same family, amplitude+phase condition, 12×-downsampled spectra | own units | NO |
| An PNN-1 / PNN-2 | OE 2020 / Nanophotonics 2023 | **forward** surrogate MLPs | same data | forward error 0.0539 / 0.0426 | NO (forward, not inverse) |

### 2.1a Runnable-artifact recon — "can they be tried under our protocol?" (verified 2026-10-04)

| model | code | weights | what running it under our protocol would take | verdict |
|---|---|---|---|---|
| MetaDiT | ✓ (`JessePrince/metadit`) | ✓ (HF) | already done — reproduced 0.0803 on our split/surrogate | **runnable** |
| MetaSR | ✓ MIT (`HIT-SudoMaker/MetaSR`) | **✗** ("assets supplied separately, no public download") | train 3–5 modules from scratch on our data: 200 epochs × ~1,093 steps/module (batch 128, `sample_limit` full); order-of-magnitude estimate for the 64×64 DDPM U-Net + 2 transformers ≈ **80–250 T4-class GPU-h ≈ 3–8 weeks of quota** (sessions ≤ 12 h; their trainer supports resume) | **only as a multi-week project, or a reduced-schedule probe (weaker comparator)** |
| MetaDiffusion | ✗ (no repo found) | ✗ | reimplement + retrain; their metric is a 26-point MAE through their own PNN — not our units | not runnable |
| I-P DM (JLT 2024) | ✗ | ✗ | — | not runnable |
| XGAN | ✗ | ✗ | different data anyway (20–35 GHz reflection, 32×32 ternary) | not runnable |
| masked spectrum-guided diffusion (2026) | ✗ | ✗ | — | not runnable |
| MetasurfaceViT | ✓ | n/a | different modality (Jones matrices, Si pillars, visible) | not applicable |

The tractable stand-in for "their model under the shared protocol" is what already
exists: MetaDiT's own re-encoded baselines (MetaDiff 0.1861 / MetaDiff-HR 0.1315 /
vanilla DiT 0.1677 on the same split and metric, paper Table 2).

### 2.2 Same task, different data

| model | venue | data | why not comparable |
|---|---|---|---|
| XGAN | Neural Networks 2024 (arXiv 2401.02961) | own dataset: 5.4 mm² unit cells, 32×32 **ternary** patterns, 20–35 GHz **reflection magnitude**, 100 pts, PEEC sim, 200k samples | band, modality (no phase), resolution, simulator all differ; metrics ACC 0.9734 / MAE 0.0533 own-definition |
| SLMGAN | Applied Soft Computing 2022 | symmetric free-form, own GHz data | same as above |
| MetasurfaceViT | arXiv 2504.14895, 2025 | Si nanopillar pairs, visible, 20×6 Jones matrices, 60 M samples | parametric unit cell, visible band, Jones-matrix condition; reports 99 % / 85 % parameter accuracy, own protocol |
| Tanriover et al. | ACS Photonics 2022 | free-form dielectric metasurfaces, optical | different data + metric |
| Liu 2018 / So & Rho 2019 / Yeung 2021 | Nano Lett / Nanophotonics / AOM | basic shapes, photonic crystals | older, different tasks |
| MetaE-former | PhotoniX 2026 | 25×25 continuous-RI pillars → 100×100 **field maps**, 1064 nm, ~260k in-house | it is a *surrogate solver*, not spectrum→geometry; field-map metric |

### 2.3 Novelty check

Web search for JEPA/self-supervised-predictive methods on metasurfaces returns
**nothing domain-specific** — the JEPA angle appears unoccupied in this field.

---

## 3. Head-to-head on the shared protocol (MetaDiT metric, test split n=17,489)

| arm | MAE ↓ | AAE ↓ | AAE&2 ↓ | AAE&4 ↓ | provenance |
|---|---|---|---|---|---|
| **Ours — unified 192-D JEPA, 70k, 3 seeds** | **0.0490 ± 0.0013** | **29.51** | **30.14** | **30.79** | [measured] `RESULTS.md` (fe-current, seeds 0/1/2) |
| MetaDiT-S (reproduced, seed0) | 0.0803 | 48.34 | — | — | [measured] official `metric.py`, released surrogate |
| MetaDiT-S (paper) | 0.0801 | 48.2495 | 58.8007 | 68.7275 | [cited] Table 2 |
| MetaDiff-HR (paper) | 0.1315 | 79.14 | 100.25 | 125.49 | [cited] Table 2 |
| vanilla DiT (paper) | 0.1677 | 100.94 | 138.07 | 187.77 | [cited] Table 2 |
| MetaDiff (paper) | 0.1861 | 112.06 | 170.63 | 258.99 | [cited] Table 2 |
| AVG1 (paper) | 0.5860 | 352.74 | 352.74 | 352.74 | [cited] Table 2 |
| NN retrieval (ours, n=512, pool 20k) | **0.0296** | — | — | — | [measured] `nn_scoping.py` |
| surrogate floor | ≈0.0065 | — | — | — | [measured] |

**Reading:** ours beats every published model on this protocol (including a
−96 % margin vs MetaDiff, −71 % vs vanilla DiT). It does **not** beat trivial
retrieval, which remains the honest headline. Our AAE&K rows are a latent-jitter
analogue (`is_diffusion_seed_diversity: false`), not diffusion seeds.

**Compute asymmetry (context, not excuse):** ours = 70k steps at batch 2
(≈1 epoch) on a Kaggle T4, 11.37 M trainable params; MetaDiT = 500 epochs on
4×A100, 32.57 M (paper) / 37.20 M live-built [measured]. MetaSR/others varied.

**Not-over-claimable:** different resolutions (theirs 32×32 quadrant, ours native
64×64), different generative paradigms, one local reproduction of MetaDiT only,
no third-party evaluation of our model exists.

---

## 4. Other axes (where the model families differ structurally)

| axis | us | diffusion/GAN field |
|---|---|---|
| inference cost | 1 forward pass | MetaDiT ~500–1000 net evals (steps×CFG); MetaSR DDIM "sub-second for hundreds" |
| determinism | deterministic (one design per input) | stochastic — diversity by construction |
| mode coverage / diversity | weak structurally (jitter/CFG only) | XGAN 1000 patterns/s; MetaSR hundreds/sub-second — they win this axis |
| partial observation (masked occupancy, known scalars) | **supported — unique** (scenarios B/C: gates 0.977/0.930 at 70k) | none accept partial context |
| parameter *conditioning* (use given scalars) | **fails** — scalar-dependence 0.505/0.529 ≈ chance | n/a (they generate params, never receive them) |
| structural fidelity (IoU/F1) | IoU 0.748 / F1 0.856 | not reported by comparators |
| worst-of-K robustness | 30.1 / 30.8 (analogue protocol) | MetaDiT 58.8 / 68.7 (diffusion seeds) |

---

## 5. Datasets "of the same nature" — candidates for an external test

| # | dataset | same nature? | testability with our current stack | effort |
|---|---|---|---|---|
| 1 | **An freeform (ours)** | identical | already used | — |
| 2 | **An cylinder + H families** (same repo) | same simulator, same band, same 301-pt complex spectra; **parametric** shapes (cyl: permittivity/gap/thickness/radius; H: 7 params) | convertible: rasterize to 64×64 occupancy; cylinder maps cleanly onto our 3 scalars (`l = gap+2r`, `h`, `n = √ε` → 3.46–5 ≈ our 3.5–5). Needs a forward surrogate for scoring (none released) or CST. **Data-contract change → operator decision per AGENTS.md** | medium |
| 3 | **MetaSR's `freeform.mat`** | identical to ours (174,883, same SHA-level description) | nothing new — same data | — |
| 4 | **DFlat datasets** (TiO2/SiN nanocylinders, nanoellipse, nanofins; visible; amplitude/phase) | same task family, different band/modality/parameterization | requires new conditioning pipeline + retraining + a visible-band forward model (pretrained ones ship with DFlat) | high |
| 5 | **MetasurfaceViT data** (Si nanopillars, Jones matrices, 60 M) | different modality (Jones, visible, parametric) | new encoders + surrogates; 60 M pipeline | high |
| 6 | **MetaE-former subset** (25×25 RI → field maps, 1064 nm) | field maps, not spectra | not a spectrum→geometry benchmark for us | n/a |
| 7 | XGAN data (32×32 ternary, 20–35 GHz) | different band/modality/resolution; release unverified | not comparable | — |

**Finding:** a genuine *external validity* test (train here, test elsewhere) is only
available at #2 (medium effort) — everything else is either the same data or a
different task requiring pipeline changes.

---

## 6. What must not be claimed

- Not "state of the art" — **NN retrieval beats us** on the same split (0.0296).
- Not "beats MetaSR" — metric protocols differ; their 0.005261 is not our MAE.
- Not "beats MetaDiT" unqualified — the win holds **on the paper's own metric,
  split, and (released) surrogate**, reproducible only via our pipelines; the
  paper's number is a citation plus one seed0 reproduction.
- Not "generalizes to metasurfaces" — only one dataset has ever been evaluated.
- No independent third party has run our artifacts; everything is self-reported.

## 7. Suggested next steps (ordered, falsifiable)

1. **Close the NN gap** (0.0296 vs 0.0490) — this is the only established loss on
   our own benchmark; no external model is closer. (In-repo work, no data change.)
2. **Publish the artifact for third-party evaluation** (checkpoint + eval harness
   + split hashes) — the field has no independent number for us; MetaSR/XGAN made
   code available, we should be eval-able by others.
3. **Decide on an external-validity test**: recommended **option #2 (An cylinder
   family)** — it is the cheapest genuinely-different data with the same spectra
   format and a natural scalar mapping; requires operator sign-off (data contract).
4. Optional: reproduce one 2026 comparator (MetaSR or I-P DM) under the shared
   MetaDiT protocol so a cross-ranking exists; blocked for MetaSR (weights not
   public) → I-P DM or the 2026 masked-diffusion paper instead.
5. Keep MetaDiT's protocol as the anchor in all future tables; never mix their
   metric with ours or MetaSR's.

## 8. Sources

- An, S. et al. *Deep learning modeling approach for metasurfaces with high
  degrees of freedom*, Opt. Express 28(21):31932 (2020) + dataset repo
  `github.com/SensongAn/Meta-atoms-data-sharing` (Google Drive link).
- Li, H., Bogdanov, A. *MetaDiT: Enabling Fine-grained Constraints in
  High-degree-of Freedom Metasurface Design*, AAAI 2026 (`arXiv:2508.05076`;
  code `github.com/JessePrince/metadit`; weights/dataset HF `Hao-Li-131/MetaDiT-AAAI2026`).
- Lao, J. et al. *A novel stepwise generative-discriminative reasoning framework
  for metasurface inverse design* (MetaSR), Opt. Laser Technol. 196:114646 (2026);
  code `github.com/HIT-SudoMaker/MetaSR` (reproducibility notes incl. dataset
  description).
- Zhang, Z. et al. *Diffusion probabilistic model based accurate and
  high-degree-of-freedom metasurface inverse design* (MetaDiffusion), Nanophotonics
  12(20):3871 (2023).
- *Rapid Inverse Design of High Degree of Freedom Meta-Atoms Based on the
  Image-Parameter Diffusion Model* (I-P DM), IEEE (2024).
- *Inverse design of discrete pixelated metasurfaces via masked spectrum-guided
  diffusion*, Opt. Laser Technol. 204:116514 (2026) — new, unverified dataset.
- Dai, M. et al. *A Surrogate-Assisted Extended Generative Adversarial Network for
  Parameter Optimization in Free-Form Metasurface Design* (XGAN), Neural Networks
  (2024), `arXiv:2401.02961`.
- Yan, J. et al. *MetasurfaceViT: A generic AI model for metasurface inverse
  design*, `arXiv:2504.14895` (2025).
- Kuang, S. et al. *Rapid inverse design of large-scale freeform meta-optics with
  the neighborhood-attention transformer* (MetaE-former), PhotoniX (2026).
- Hazineh, D. et al. *DFlat*, `arXiv:2207.14780`; datasets via
  `github.com/DeanHazineh/DFlat`.
- Internal: `docs/benchmarking/*`, `docs/comparison/metadit_vs_unified_jepa.md`,
  `checkpoints/unified/REPORT.md`, `RESULTS.md`.
