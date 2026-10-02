# Model comparisons

Side-by-side comparisons of this repo's models against published baselines.

- **[metadit_vs_unified_jepa.md](./metadit_vs_unified_jepa.md)** — the unified 192-D
  JEPA vs MetaDiT-S (diffusion): paradigm, architecture and measured parameter
  counts, spectral-fidelity benchmarks, and the JEPA-vs-diffusion axes
  (determinism, diversity, partial-context, inference cost).
- **[overfitting.md](./overfitting.md)** — train vs held-out check on Scenario A:
  **not overfit** (medians match); the train/held-out mean gap is a ~0.2 %
  catastrophic tail, not fitting.

Headline: ours is **~3.3× smaller trainable** (11.4 M vs 37.2 M, same 256-token grid),
**~500× cheaper at inference** (1 forward vs ~500–1000 denoising evals), and
**beats MetaDiT-S on the paper metric** (test MAE 0.0673 vs 0.0801; AAE 40.5 vs 48.2) —
with two honest caveats: **NN retrieval still beats ours** on raw MAE, and our
**scalar path fails** its dependence gate (at chance).

Size figures are measured live; performance is from
`docs/benchmarking/BASELINE.md` (Phase 1) and the MetaDiT paper / local reproduction.
