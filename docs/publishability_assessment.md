# Publishability & novelty assessment — unified 192-D JEPA for metasurface inverse design

**Date: 2026-10-04 (consolidation pass).** This **supersedes** the 2026-10-02
assessment; stale conclusions (single-checkpoint, no ablation, no multi-seed, MAE 0.0673
headline) are **replaced**, not appended to.

**Scope of this pass:** documentation / provenance / literature only. **No training, no
runs, no data-dependent compute, no changes under `src/`, no architecture/config changes.**
Every numerical claim below is traceable (§9 provenance map). Claims that could not be
traced were removed rather than carried forward.

---

## 1. Current verdict

The project now has a **stronger, experimentally supported result** than the previous
assessment, and it is still **not fully validated or publication-final**:

- the baseline has **3-training-seed** validation (§3);
- the **JEPA-vs-conventional objective ablation supports the methodological claim** — the
  latent objective contributes material value beyond the architecture alone (§3);
- the MetaDiT comparison has been **audited and corrected for like-for-like scoring**
  (same surrogate, same split, same metric implementation);
- **Scenario C is resolved** at 70k, clearly above the 0.75 gate;
- the **scalar-conditioning line has been properly tested and stopped** after two
  independent negative results.

What changed the verdict from "not publishable" to "credible direction with a supported
method claim" is (a) multi-seed establishment and (b) the ablation. What still blocks a
strong claim is the absence of independent physical validation and the nearest-neighbour
result (§5, §6).

---

## 2. Provisional headline framing (pending operator confirmation)

> **Provisional headline framing — pending operator confirmation:** a deterministic,
> single-forward, flexible partial-observation inverse-design model whose JEPA objective
> is demonstrably doing real work relative to conventional supervision, competitive in the
> retrofit/constrained-editing regime, and better than MetaDiT-S on the shared benchmark
> metric — while explicitly losing to nearest-neighbour retrieval (`0.0296`) on the
> easiest full spectrum-to-geometry task.

This is a **draft only**; this consolidation pass is **not** making the final headline
decision (§6B). The nearest-neighbour result is **not softened or reinterpreted**: on pure
S→G retrieval beats us.

---

## 3. Evidence table

Frozen values, each with a repository source (§9).

**Benchmark result (70k, 3 training seeds)** — `RESULTS.md` "Full-epoch (70k) baseline",
kernel `metasurface-jepa-fe-current` @ `c35196f`:

| metric | value | n |
|---|---|---|
| MAE (test, n=17,489) | **0.0490 ± 0.0013** | 3 training seeds |
| AAE | 29.51 ± 0.74 | 3 |
| Scenario A / B / C | **0.9967 / 0.9772 / 0.9303** | 3 |

**Objective ablation (70k, 3 training seeds each)** — `RESULTS.md` "Controlled ablation —
JEPA vs conventional", kernel `metasurface-jepa-objective-ablation` @ `2218d70`:

| arm | MAE | A / B / C | occ IoU |
|---|---|---|---|
| JEPA | **0.0490 ± 0.0013** | 0.9967 / 0.9772 / 0.9303 | ~0.75 |
| conventional | **0.0922 ± 0.0195** | 0.9792 / 0.9460 / 0.8288 | ~0.64 |

Every conventional seed (0.0730–0.1121) is worse than every JEPA seed (0.0476–0.0500);
≈1.9× worse on the mean. **The JEPA objective contributes material value beyond the
architecture alone.**

**Retrofit result (Scenario C)** — resolved at 70k: **0.9303** (95 % t-CI
consistently above 0.75). At 10k it was borderline; the crossed seed study
(`RESULTS.md` "Scenario C — crossed seed study") localised that to training-seed variance.

**Scalar conditioning — negative** — closed after **two independent** architectural
interventions: door-(a) auxiliary read-out loss (`checkpoints/unified/REPORT.md` §21) and
the predictor-FiLM (`RESULTS.md` "Step 3 — scalar-predictor FiLM"). Neither moved scalar
dependence meaningfully off chance, in JEPA or conventional training (all ≈ 0.49–0.55).

**Nearest-neighbour retrieval baseline** — **0.0296** (pool 20k, n=512), `RESULTS.md`
"NN-scoping probe" / `scripts/diagnostics/nn_scoping.py`. **It still beats JEPA on the
easiest pure spectrum→geometry task.** NN's scored error ≈ its retrieval distance; the
dataset is dense and the surrogate is near-exact.

**MetaDiT comparison** — audited in `docs/benchmarking/EXTERNAL_BENCHMARK_COMPARISON.md`:
like-for-like at metric/split/surrogate; beats the **reproduced** MetaDiT-S **0.0803**
(`checkpoints/phase0/seed0_metric.json`); the paper's 0.0801 is a citation only; the
generation protocol differs (ours deterministic vs diffusion), and our AAE&K is an
analogue, never equivalent.

---

## 4. What is established

- A 3-training-seed **fidelity/gate baseline**: MAE 0.0490 ± 0.0013; A/B/C
  0.9967/0.9772/0.9303.
- The **JEPA latent objective materially outperforms conventional supervised training**
  of the same architecture/data/optimizer (MAE 0.0490 vs 0.0922; complete per-seed
  separation), on this benchmark.
- **MetaDiT-S is beaten on the shared metric under the released surrogate** on the same
  test split (0.0490 vs the reproduced 0.0803) — with the protocol caveats above.
- **Scenario C (retrofit) passes** the 0.75 gate at the established checkpoint.
- **Scalar conditioning does not work** — established as a *negative* result.
- **The scalar architecture line is closed** (two negative interventions, no rescue
  tuning).

---

## 5. What is not established

- **No independent full-wave EM validation.** Everything is scored by the frozen *learned*
  surrogate; no claim of real-device / experimental physical validity is made.
- **Scalar conditioning is not solved.**
- **Nearest-neighbour retrieval remains stronger** on the pure S→G task (0.0296 vs
  0.0490).
- **Literature novelty is based on a targeted search, not proof of absence** (§8).
- The surrogate-circularity caveat stands: our physics loss optimises through the *same*
  frozen surrogate that scores the headline metric (MetaDiT only evaluates with it).

---

## 6. Two operator-reserved decisions (left open)

### A. Full-wave EM validation
Blocked on external EM software/solver and physical-stack setup not present in this
repository (there is no full-wave solver in-tree; the only "physics" is the learned
surrogate; the dataset spectra arrived pre-computed).

> This is the single biggest remaining publication-critical validation gap.

Surrogate validation does **not** substitute for it.

### B. Final headline framing
The §2 framing is a draft. The operator may choose a narrower or differently scoped claim
(e.g. leading with the retrofit/constrained-editing task, or with the efficiency/
determinism result). **Not decided in this pass.**

---

## 7. Named low-cost open item — Ground-truth scalar-sensitivity ceiling check

A diagnostic that would quantify how much the *true* spectrum changes with scalar changes
(not with occupancy changes), to bound whether scalar conditioning is even learnable from
this data. It would: match/bucket samples by similar occupancy; measure true spectrum
variation as scalar variation changes; compare with spectrum variation caused by occupancy
changes; use the **existing frozen surrogate**; **no model training**.

> **Not run in this consolidation phase; operator decision pending.**

---

## 8. Novelty / literature positioning

Targeted search (2026-10-04; WebSearch). Calibrated language only.

**Directly prior / adjacent work (metasurface inverse design):**
- **MetaDiT** (AAAI 2026) — diffusion transformer, S→G only; the direct baseline.
- **DiffMeta / conditional diffusion** (Cell Rep. Phys. Sci. 2026; arXiv:2506.07083),
  **3D-CDM** (Adv. Mater. Technol. 2025), **physics-guided conditional diffusion**
  (arXiv:2605.19611) — the field is **diffusion-dominated**.
- **MetasurfaceViT** (Nanophotonics 2026) — a generic AI model for metasurface inverse
  design (different method).
- **"Inverse Design in Nanophotonics via Representation Learning"** (arXiv:2507.00546) —
  the closest *representation-learning* neighbour; **not** JEPA.
- **GNDM** (Science Advances 2026) — physics-guided **generative** SSL for disordered
  metamaterials; self-supervised, but generative, not JEPA.

**JEPA in science/engineering (other domains):** polymer graphs (RSC Digital Discovery
2026), **Mol-JEPA** (arXiv:2608.22642), **AeroJEPA** (aerodynamics, arXiv:2605.05586),
**UniJEPA** (arXiv:2608.07409).

**Finding 1 — JEPA + metasurface/EM inverse design:** *we found no prior work in the
targeted search* that combines JEPA (predictive-latent SSL) with metasurface/metamaterial/
photonic inverse design. The claim is **defensible but narrow** — a domain transfer of an
existing paradigm; do not lead with it as the contribution.

**Finding 2 — retrofit / constrained-editing inverse design:** *we did not identify* prior
work that demonstrates the exact combination of (i) an existing geometry, (ii) protected/
fixed regions or parameters, (iii) a target spectral correction, and (iv) generating only
the unconstrained portion. The nearest hits are **conditional inverse design** (full
generation conditioned on constraints — a *different* setting) and one spectral-imaging
study that optimises only the GSST geometry with a **fixed** set of crystallization states
(adjacent, not the same). We did **not** equate ordinary conditional inverse design with
this retrofit setting. This narrow setting **appears more specific — hence potentially more
defensible — than the broad "JEPA for inverse design" claim**, but it is **not** proven
absent.

---

## 9. Provenance map (claim → source)

| claim | source |
|---|---|
| 70k baseline MAE 0.0490 ± 0.0013; A/B/C 0.9967/0.9772/0.9303 (3 seeds) | `docs/benchmarking/RESULTS.md` "Full-epoch (70k) baseline"; kernel `metasurface-jepa-fe-current` @ `c35196f` |
| conventional MAE 0.0922 ± 0.0195; per-seed 0.0914/0.0730/0.1121 | `RESULTS.md` "Controlled ablation — JEPA vs conventional"; kernel `metasurface-jepa-objective-ablation` v2 @ `2218d70`; results commit `7858d87` |
| MetaDiT like-for-like; reproduced 0.0803 / 48.34 | `docs/benchmarking/EXTERNAL_BENCHMARK_COMPARISON.md`; `checkpoints/phase0/seed0_metric.json`; `scripts/eval/reproduce_metadit_baseline.py` |
| NN retrieval 0.0296 | `RESULTS.md` "NN-scoping probe"; `scripts/diagnostics/nn_scoping.py` |
| Scenario C 0.9303 (resolved); crossed study | `RESULTS.md` "Full-epoch …" + "Scenario C — crossed seed study" |
| scalar negative — door (a) | `checkpoints/unified/REPORT.md` §21 |
| scalar negative — predictor-FiLM | `RESULTS.md` "Step 3 — scalar-predictor FiLM … NEGATIVE" (kernel `metasurface-jepa-scalar-film` / `-sf-eval`) |
| no full-wave solver in-repo | repo search: only `src/physics/physics_loop.py` (learned surrogate) and `external/metadit/model/surrogate.py` |
| overfitting: not overfit (medians match) | `docs/comparison/overfitting.md` |
| parameter counts (11.37 M trainable) | `docs/comparison/metadit_vs_unified_jepa.md` |
| surrogate-leakage resolved (train/val only) | `external/metadit/scripts/train_*.sh` + the independent review |

---

## 10. Sources (literature)

- [MetaDiT (AAAI 2026) — paper](https://ojs.aaai.org/index.php/AAAI/article/download/37025/40987) · [arXiv:2508.05076](https://arxiv.org/abs/2508.05076)
- [DiffMeta — spectrum-to-structure conditional diffusion (Cell Rep. Phys. Sci. 2026)](https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(26)00080-9) · [arXiv:2506.07083](https://arxiv.org/abs/2506.07083)
- [Physics-guided conditional diffusion for metasurfaces (arXiv:2605.19611)](https://arxiv.org/abs/2605.19611)
- [3D conditional diffusion for metamaterials (Adv. Mater. Technol. 2025)](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/admt.202500293)
- [Inverse Design in Nanophotonics via Representation Learning (arXiv:2507.00546)](https://arxiv.org/abs/2507.00546)
- [MetasurfaceViT — generic AI model for metasurface inverse design (Nanophotonics 2026)](https://onlinelibrary.wiley.com/doi/10.1002/nap2.70001)
- [Inverse Design of Metasurface for Spectral Imaging (arXiv:2510.21924)](https://arxiv.org/pdf/2510.21924)
- [AeroJEPA — predictive latent representations for aerodynamic surrogates (arXiv:2605.05586)](https://arxiv.org/abs/2605.05586)
- [Mol-JEPA — multimodal JEPA for molecules (arXiv:2608.22642)](https://arxiv.org/abs/2608.22642)
- [UniJEPA — a unified JEPA (arXiv:2608.07409)](https://arxiv.org/html/2608.07409v1)
- [JEPA for self-supervised pretraining of polymer graphs (RSC Digital Discovery 2026)](https://pubs.rsc.org/dd/article/5/2/819/889150/Joint-embedding-predictive-architecture-for-self)
- [GNDM — self-supervised disordered metamaterials (Science Advances 2026)](https://www.science.org/doi/pdf/10.1126/sciadv.adx7389)
- [Awesome-JEPA (curated list)](https://github.com/AbdelStark/awesome-jepa) · [Tutorial on JEPA (TechRxiv 2026)](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176469421.19270944/v1)
