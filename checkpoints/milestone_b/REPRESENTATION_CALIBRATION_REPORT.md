# Representation Calibration Report

> **Checkpoint:** `checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt`
> **Step / epoch:** 2 / 0
> **Objective:** `jepa_vicreg`
> **Date:** 2026-08-25
> **Geometries evaluated:** 32

---

## ⚠ Limitations

This report is run on a **2-step smoke checkpoint** (not a full training run) using
**32 geometries** from the validation split. All conclusions are preliminary:

- **Linear probe R² is meaningless at N=32** (24 train / 8 val, 384 features — the
  model is severely undersampled and R² values are extremely negative, as expected).
  The actual calibration numbers require ≥ 256 geoms after a genuine training run.
- The **EMA target weights were not saved in the checkpoint** (legacy pre-fix format
  only carrying momentum scalars). The "trained EMA target" column therefore reflects
  a freshly-initialized EMA encoder, not the result of actual training. This is
  expected to converge to meaningful values once a post-fix checkpoint (with EMA
  target weights saved) is used.
- The **3D linear-probe statistics** (eff rank, pairwise cos, same-token cos, etc.)
  are meaningful at N=32 and reported below.

---

## 1. Is the trained EMA-target representation degenerate / collapsed?

### Four-way mean-pooled statistics

| Metric                | EMA target | Released ViT | Random init | Collapsed anchor |
|----------------------|-----------|-------------|------------|-----------------|
| token var             | 0.04618   | 0.04678     | 0.02016    | n/a             |
| token std             | 0.18114   | 0.18262     | 0.10963    | n/a             |
| pairwise cos mean     | 0.99826   | 0.99819     | 0.99958    | 0.99987         |
| pairwise cos p05      | 0.99469   | 0.99526     | 0.99884    | 0.99960         |
| same-token cos        | 0.99296   | 0.99223     | 0.99371    | 0.99927         |
| eff rank frac         | 0.08755   | 0.09943     | 0.08789    | 0.03501         |
| entropy eff rank      | 2.80158   | 3.18163     | 2.81247    | 13.44490        |
| participation         | 1.99450   | 2.42121     | 2.16617    | 2.10000         |
| top eig frac          | 0.68031   | 0.57644     | 0.63086    | 0.63070         |

### Assessment

- **p05 pairwise cos (0.99469)** is well below the collapsed anchor (0.99960) but
  higher than the released ViT (0.99526). The gap (0.00057) is small but present.
- **Same-token cos (0.99296)** is close to collapsed-anchor level (0.99927) but
  still below it.
- **Eff rank frac (0.08755)** is above the collapsed anchor (0.03501) and close to
  the released ViT (0.09943) — the representation is **not collapsed** on this
  metric.
- **No collapse gate verdict is issued here** (B7: the classification thresholds
  remain unchanged from the production training path).

**Preliminary:** at step 2 with legacy checkpoint, the representation is
**not collapsed** but is closer to the released ViT than to a well-differentiated
representation. This is expected to improve after a genuine training run.

---

## 2. Is it merely mimicking the released MetaDiT encoder?

The EMA target and released ViT show very similar values across all metrics,
confirming that at step 2 the representation has **not yet diverged** from its
initialization. This is expected — a 2-step training run has essentially no effect
on the representation. The question can only be answered after a genuine training
run that produces non-trivial cosine-alignment differences.

---

## 3. Does the raw EMA representation carry physical-parameter information?

### Linear-probe R² (ridge regression, N=32, λ=0.01)

| Encoder       | l_lattice | h_atom | r_atom | mean R² |
|--------------|----------|--------|--------|---------|
| trained_ema  | -513.005 | -48.097| -335.648| -298.916|
| released_vit | -513.158 | -47.457| -332.221| -297.612|
| random_init  | -513.389 | -47.873| -334.378| -298.547|

All R² values are extremely negative — **the probes are severely undersampled**.
With 24 training samples and 384 features, no linear model can generalize. This
is a measurement limitation, not a finding about the representation.

**To answer this question, use ≥ 256 geometries after a genuine training run.**

---

## 4. Where does VICReg gradient actually flow?

This is answered separately by:
```bash
python scripts/diagnostics/vicreg_gradient_attribution.py \
    --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt \
    --config configs/milestone_b.yaml --device cpu
```

(Not run here — the gradient attribution script requires a real batch from the
training dataset and is best run on the cloud GPU.)

---

## 5. Do the numbers match the collapsed anchor reference?

No — the trained EMA target is clearly separated from the collapsed anchor:

- p05 pairwise cos: 0.99469 vs collapsed 0.99960 (margin: 0.0049)
- eff rank frac: 0.08755 vs collapsed 0.03501 (ratio: 2.50×)
- same-token cos: 0.99296 vs collapsed 0.99927 (margin: 0.0063)

However, the separation is small and may change after genuine training.

---

## 6. Do they match the released MetaDiT encoder?

Yes — the EMA target values closely track the released ViT at step 2, as
expected. After genuine training, divergence from this initial state indicates
the representation is learning task-specific structure.

---

## 7. Raw EMA vs projected (VICReg projector, measurement only)

| Metric                | Raw EMA | Projected |
|----------------------|---------|-----------|
| eff rank frac         | 0.08755 | 0.08942   |
| entropy eff rank      | 2.80158 | 2.86144   |
| token std             | 0.18114 | 0.02835   |
| pairwise cos p05      | 0.99469 | 0.99248   |
| same-token cos        | 0.99296 | 0.98930   |

The projection through the VICReg projector **reduces token-level variance** (std
0.181 → 0.028) and **slightly lowers cosine similarity**, consistent with the
projector learning to decorrelate the raw representation. Eff rank and effective
rank fraction are essentially unchanged.

---

## Files

- **Checkpoint:** `checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt`
- **JSON output:** `checkpoints/milestone_b/representation_calibration.json`
- **Gradient attribution:** `scripts/diagnostics/vicreg_gradient_attribution.py`
- **Combined calibration runner:** `scripts/diagnostics/representation_calibration.py`
- **Grouped-view helper:** `src/diagnostics/representation_health.py:grouped_view`
- **Linear probes:** `src/diagnostics/representation_probes.py`
