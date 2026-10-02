# Publishability & novelty assessment — unified 192-D JEPA for metasurface inverse design

Date: 2026-10-02. Status of the model: single trained checkpoint (full-epoch,
70k steps), measured against MetaDiT-S with the shared frozen surrogate.

This is a deliberately conservative, evidence-based read. Where something is not
established, it says so.

---

## 1. TL;DR verdict

**Not yet publishable as a top-tier result.** There are genuine, interesting
elements (a non-generative, deterministic, ~3× smaller model that beats a AAAI-2026
diffusion baseline on its own metric), but two **result-level problems** currently
block a strong claim, and several validation gaps make the headline fragile:

1. **NN retrieval beats us** on the primary metric (test MAE 0.0551 vs our 0.0673).
   A reviewer's first question is "why not nearest-neighbour?" — unanswered.
2. **Scalar dependence fails** (win rate 0.51 / 0.45, i.e. at chance). Any claim of
   "predicts all parameters" is false as the model stands.

Plus: ~1 epoch on 1 GPU (vs MetaDiT's 500 epochs on 4×A100) makes our win
*implausibly* large and therefore suspicious; the physics loss optimises through the
**released surrogate**, whose training split we have not verified (possible leakage);
and there is no full-wave validation, no multi-seed variance, and no ablation.

**What is defensible today:** a well-instrumented, reproducible system + benchmark
harness, an honest failure analysis, and a *direction* (predictive-latent inverse
design with flexible partial observations). That is workshop / short-paper material
with careful framing — not a main-track AAAI/NeurIPS result yet.

---

## 2. What we actually have (measured)

| item | value | source |
|---|---|---|
| test MAE / AAE (Scenario A, n=17,489) | **0.0673 / 40.535** | `docs/benchmarking/BASELINE.md` |
| AAE&2 / AAE&4 | 42.14 / 43.95 | same |
| NN-retrieval MAE / AAE (n=512) | **0.0551 / 33.161** | same |
| surrogate floor MAE | 0.0065 | same |
| A / B / C win rate (n=512) | 0.994 / 0.967 / 0.898 | same |
| scalar-dep one/two-known | **0.506 / 0.451 (FAIL)** | same |
| trainable params | 11.37 M | live count |
| training | 70k steps × batch 2 ≈ **1 epoch**, 1× T4 | config / report |
| catastrophic tail | ~0.2 % of held-out samples | `docs/comparison/overfitting.md` |
| overfitting | **not overfit** (medians match) | `docs/comparison/overfitting.md` |

Baseline to beat (AAAI 2026, MetaDiT-S): 32.57 M, MAE 0.0801, AAE 48.2495,
AAE&2 58.80, AAE&4 68.73, 500 epochs, 4×A100. Vanilla DiT 32.80 M, 0.1677/100.94.

---

## 3. Field landscape (what the neighbours actually claim)

- **MetaDiT** (Li & Bogdanov, AAAI 2026, arXiv:2508.05076): diffusion **Transformer**
  (DiT) over a 3-channel geometry image, conditioned by a contrastively-pretrained
  spectrum encoder (coarse AdaLN + fine in-context tokens). Claims: (i) generates
  **all** unit-cell parameters (not a fixed subset), (ii) **full-resolution** spectral
  constraints, (iii) new metrics **AAE / AAE&K**, (iv) an ablation study + scaling
  study. It is **S→G only** — no partial/observed-subset conditioning.
- **MetaDiff / MetaDiff-HR** (Zhang et al. 2023/2024): earlier diffusion for
  metasurface inverse design (the direct predecessor MetaDiT beats).
- **DiffMeta / spectrum-to-shape conditional diffusion** (Cell Rep. Phys. Sci. 2026,
  arXiv:2506.07083): conditional diffusion, **one-to-many** design, manufacturing
  guidance, thermal camouflage.
- **3D-CDM** (Adv. Mater. Technol. 2025): 3D conditional diffusion for voxel
  metamaterials.
- **Physics-guided fabrication-aware diffusion** (Seo et al. 2025, arXiv:2504.17077).
- **JEPA is already in scientific/engineering ML**, but not (found) in metasurface/EM:
  - JEPA for **polymer graphs** (RSC Digital Discovery 2026);
  - **Mol-JEPA** — multimodal JEPA for molecules (arXiv:2608.22642);
  - **AeroJEPA** — predictive latent representations for **aerodynamic** surrogates
    (arXiv:2605.05586), incl. "constrained design latent-optimization".
  - Broad JEPA tutorials/reviews (OpenReview, TechRxiv 2026); I-JEPA/V-JEPA lineage.
- **Self-supervised disordered metamaterials**: GNDM (Science Advances 2026) —
  physics-guided **generative** SSL for disordered metamaterials (closest
  self-supervised neighbour, but generative, not JEPA).

**Reading:** metasurface inverse design is currently **diffusion-dominated**. JEPA
has crossed into molecular/structural and aerodynamic surrogates but (in our search)
**not into metasurface / EM inverse design**. That is the opening — and its limit:
it is a *domain transfer*, not yet a *method*.

---

## 4. Where we stand, axis by axis

| axis | ours | verdict |
|---|---|---|
| spectral fidelity (paper metric) | **beats MetaDiT-S** (0.0673 vs 0.0801) | strong *if* real (§6 leakage) |
| vs trivial retrieval baseline | **loses** (0.0673 vs 0.0551) | **blocking** |
| parameter prediction (from spectrum) | accurate (scalar MAE 0.088) | fine — not a blocker |
| parameter conditioning (use given scalars) | **fails** (at chance) | **blocking** |
| partial-observation capability | A/B/C gates pass; B/C are not S→G | potential novelty |
| efficiency | 11.4 M, 1 forward | strong |
| determinism / diversity | deterministic (no mode coverage) | liability vs the field |
| robustness (worst-of-K) | 43.95 vs 68.73 | **not apples-to-apples** (§5) |
| generalisation | not overfit; median matches train | good |
| failure tail | ~0.2 % catastrophic | open problem |
| validation | surrogate only; no full-wave | weak vs published work |

---

## 5. Novelty candidates — and how defensible each really is

1. **First JEPA (predictive-latent) inverse design for metasurfaces/EM.** No prior
   work found. *Defensible but narrow:* domain transfer of an existing paradigm;
   reviewers discount pure "first to apply X to Y" unless paired with a method
   insight. **Keep, but do not lead with it as the contribution.**
2. **Flexible partial-observation conditioning** — arbitrary known/unknown occupancy
   region **and** known/unknown scalars **with** a target spectrum (scenarios B/C),
   versus the field's S→G or fixed-subset conditioning. *Most defensible as a task
   contribution* — but must be shown to actually work (scalar path must be fixed).
3. **Physics-in-the-loop JEPA** — a frozen differentiable EM surrogate as a term in a
   JEPA objective. *Interesting*, but risks being a training trick; needs an ablation
   to show it is load-bearing.
4. **Efficiency result** — deterministic single forward, ~3× smaller, matching/beating
   a diffusion SOTA. *Strong if it survives §6.*
5. **Rigour artefact** — a MetaDiT-comparable benchmark + honest failure analysis.
   *Valuable, not headline.*

**Not novel / must not be claimed:** "generates all parameters" (MetaDiT already
claims this, and ours fails on scalars), "full-resolution spectrum" (MetaDiT),
"AAE/AAE&K" (MetaDiT's metrics), "first self-supervised metamaterial design" (GNDM).

---

## 6. Threats to validity (the honest part)

These must be resolved before any publication claim:

1. **NN retrieval beats us.** The single most damaging fact. Either beat it, or
   reframe the task (e.g. retrofit/partial where retrieval is inapplicable) and show
   the gap there.
2. **Scalar conditioning is dead** (§17 of the report). Blocks claims (2) and "all
   parameters".
3. **Surrogate training-split leakage — RESOLVED by the independent review.** The
   reviewer inspected the released training scripts (`external/metadit/scripts/
   train_*.sh`): the surrogate, spectrum encoder and DiT all train on
   `train_set.mat` (+ val for selection) — **the test split is not used.** So there
   is no data leakage. The residual risk is **reward-hacking / metric circularity**:
   our physics loss optimises through the *same* frozen surrogate that scores the
   headline metric (MetaDiT only evaluates with it). Mitigation: report a co-primary
   metric that is not the training objective (occupancy IoU/F1, seam/locality, gate
   win rates — already computed), add full-wave validation on ~32 designs, and state
   the asymmetry explicitly in the paper.
4. **Released spectrum encoder** was contrastively pretrained on the dataset (both
   models use it) — shared, but it means the "input" already encodes dataset structure.
5. **1 epoch vs 500 epochs.** Our win over MetaDiT is too large for the training
   budget. Either we found something real and must show it endures with more training,
   or the comparison is confounded (see 3). Reviewers will probe this.
6. **Worst-of-K is not apples-to-apples.** MetaDiT's AAE&K samples diverse structures
   (some bad by construction); a deterministic model trivially wins worst-of-K. Do not
   report AAE&K as a like-for-like win.
7. **No full-wave validation.** Everything is surrogate-scored. MetaDiT's neighbours
   increasingly fabricate/measure or use full-wave solvers.
8. **Single seed, no ablations, no variance.** Cannot support significance.
9. **Determinism vs one-to-many.** The field frames inverse design as one-to-many;
   a deterministic model is a feature (efficiency) *and* a liability (no diversity).

---

## 7. What is claimable *right now* (conservative)

- "A deterministic, single-forward, 11.4 M-parameter predictive-latent model that
  reaches competitive spectral fidelity on the MetaDiT test split under the shared
  surrogate, at ~1 epoch of training." *(Report MAE 0.0673 / AAE 40.5 with the NN
  caveat printed in the same table.)*
- "A reproducible, MetaDiT-comparable benchmark harness and an honest evaluation of a
  new architecture class on this task."
- "Evidence that predictive-latent (non-generative) modelling is a viable and efficient
  alternative *direction* for metasurface inverse design." *(framed as a direction,
  not a result.)*

Everything stronger (SOTA, "generates all parameters", "first JEPA") is **not**
supportable yet.

---

## 8. What would make it publishable (checklist)

**Must-fix (result-level):**
- [ ] **Beat NN retrieval** on the hard stratum (or show a task where retrieval cannot
      apply and win there).
- [ ] **Fix scalar conditioning** (Phase 4/5 roadmap) and pass the scalar gates.
- [ ] **Resolve surrogate leakage**: verify the released surrogate's training split;
      retrain/repair if it saw test; report the physics loss's data provenance.
- [ ] **Train properly** (many epochs, ≥3 seeds) and show the result is stable.

**Should-fix (evidence-level):**
- [ ] Ablations: physics term, EMA target, masking, projector, scalar FiLM.
- [ ] Full-wave validation of a handful of designs.
- [ ] Report median/p90 + tail, never the mean alone (already our policy).
- [ ] Diversity study: deterministic O(1) vs diffusion multi-sample trade-off.

**Positioning:**
- [ ] Lead with the **task** (flexible partial-observation conditioning) + the
      **efficiency/determinism** result; JEPA is the *method*, not the headline.

---

## 9. Recommended framing (if pursued)

*"Predictive-latent inverse design: a deterministic, single-forward JEPA replaces
iterative diffusion for metasurface design, and generalises S→G to arbitrary
partial-observation constraints (observed regions + observed parameters)."* The
contribution is the **setting + the efficiency/paradigm shift**, not "we used JEPA".

Target: a strong workshop first (e.g. an ML-for-science / physics-ML workshop), then a
full paper once §8's must-fix items clear.

---

## 10. Sources

- [MetaDiT (AAAI 2026) — paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37025/40987) · [arXiv:2508.05076](https://arxiv.org/abs/2508.05076) · [repo](https://github.com/JessePrince/metadit)
- [DiffMeta: manufacturing-guiding spectrum-to-structure conditional diffusion (Cell Rep. Phys. Sci. 2026)](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(26)00080-9) · [arXiv:2506.07083](https://arxiv.org/abs/2506.07083)
- [3D conditional diffusion for metamaterial inverse design (Adv. Mater. Technol. 2025)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/admt.202500293)
- [AeroJEPA — predictive latent representations for aerodynamic surrogates (2026)](https://arxiv.org/abs/2605.05586)
- [Mol-JEPA — multimodal JEPA for molecules (2026)](https://arxiv.org/abs/2608.22642)
- [JEPA for self-supervised pretraining of polymer graphs (RSC Digital Discovery 2026)](https://pubs.rsc.org/dd/article/5/2/819/889150/Joint-embedding-predictive-architecture-for-self)
- [GNDM — self-supervised disordered metamaterials (Science Advances 2026)](https://www.science.org/doi/pdf/10.1126/sciadv.adx7389)
- [Tutorial on JEPA (2026)](https://openreview.net/pdf?id=Zr4PUe0ZNl)
- [AI-enabled metasurface design review (2026)](https://www.oejournal.org/ioe/article/doi/10.67704/ioe.2026.260011)
- [Review of deep learning in metasurface modeling & design (Prog. Quantum Electron. 2026)](https://www.sciencedirect.com/science/article/pii/S0079672725000023)

---

## Appendix — independent review (2026-10-02) and disposition

An adversarial review of this project confirmed both headline blockers, sharpened
one (the NN gap is measured unfairly *in both directions*), and resolved one (the
surrogate leakage fear). It also found eight verification-hygiene issues. Disposition
of the P0 items — all now fixed in the repo, each as its own commit:

- **A1 (broken physics-gradient guard) — FIXED.** `test_physics_gradient_regression`
  called the removed `objective.physics_loss.enable()` and was `skipif`-ed without
  the weights → inert everywhere. Rewritten to the current API with a differentiable
  stub fallback. The full suite is now **328 passed / 8 skipped / 0 failed** (the
  previously-reported "pre-existing failure" is gone).
- **B1 (vacuous tests) — FIXED.** Six tests rewritten behaviourally.
- **A2 (unfair NN comparison) — FIXED.** Same items / same n as the model + a
  training-pool-size curve.
- **A4 (transductive AVG1) — FIXED.** AVG1 uses the exact TRAIN-split mean.
- **A5 (small-n diagnostics) — FIXED.** Surrogate floor over all items; n labelled.
- **B2 (preflight precedence) — FIXED.** Pixel index from the *delivered* occupancy;
  empty-support samples recorded, not asserted. Real preflight passes.
- **C1/C2/C6 — FIXED.** Real JSON via `--out`; `--scenario` wired; smoke dummy to a
  `.smoke_dummy` sibling.
- **A3 — wording FIXED** (prediction vs conditioning, above). **C3 — parameter
  accounting FIXED** (`docs/comparison/metadit_vs_unified_jepa.md`).

**Still open (P1 — needs cloud runs / real research):** A6 circularity (co-primary
non-objective metrics + full-wave), B3 multi-seed CIs, B4 full-wave validation, B5 the
1-epoch-vs-500 budget question, and the research items (fix scalar conditioning; beat
NN retrieval).

**C4 (branch divergence):** this clone is the benchmarking line only. The reviewer's
local door-(b)/preflight branch is not present here, so its two findings (C5 comment;
B2-local `NameError`) could not be fixed in this tree and must be handled at the
merge. **Consequence for the numbers:** the recorded Phase-1 baseline (MAE 0.0673,
NN 0.0551, AVG1 0.2574) predates the A2/A4/A5 corrections — the NN row and AVG1 row
must be regenerated with the fixed driver before publication.
