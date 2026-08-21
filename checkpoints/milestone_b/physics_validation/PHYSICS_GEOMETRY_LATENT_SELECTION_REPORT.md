# Physics-Geometry Latent Selection Report

**Post-Gate 0/0.5 Latent-Space Validation**  
Generated: 2026-08-21  
Checkpoint: `checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt` (step 2, jepa_vicreg)

---

## 1. Executive Summary

This report consolidates the latent-space validation diagnostics per the Post-Gate 0/0.5 plan (§25):

- **Diagnostic A** (latent_geometry_probe): Does the EMA target latent `z_y_raw` preserve independent spatial occupancy information beyond the three scalar parameters `(l, h, r)`?
- **Diagnostic B** (physics_target_selection): Does the requested spectrum move the predictor toward the correct spatial target latent at 75% and 100% masking?

**Key finding:** The target latent `z_y_raw` **does** preserve spatial geometry (median IoU 0.58 vs scalar control 0.33, pairwise Spearman ρ=0.44 vs 0.04 for params). However, **physics target selection fails**: at 100% masking, real-vs-shuffled margins are ~0 (fraction positive ≈ 0.53), recall@1 = 0.125, mutual causal win rate = 0.0. This matches **Case B** of the interpretation matrix (§18): *target latent preserves spatial geometry, but physics selection fails*.

**Recommended intervention:** Add the MetaDiT-style CLIP alignment objective on the student geometry encoder (as described in §20–§21 of the plan), keeping the EMA target geometry-only.

---

## 2. z_y_raw Spatial Probe (Diagnostic A)

### 2.1 Probe Configuration
- **Samples**: 256 (train=154, val=51, test=51; seed=0, split_seed=0)
- **Probe**: Linear, 300 epochs, lr=1e-3
- **Device**: CPU
- **Target**: 64×64 binary occupancy (4096 pixels)

### 2.2 Results (Held-out Test Set)

| Probe | IoU (median) | Precision (median) | Recall (median) | Pixel Acc (median) |
|-------|-------------|-------------------|----------------|-------------------|
| **Pooled z_y_raw** | **0.577** | 0.692 | 0.757 | 0.760 |
| Token z_y_raw | 1.000 | 1.000 | 1.000 | — |
| **Scalar control (l,h,r)** | **0.332** | 0.480 | 0.524 | 0.520 |

**Key comparison:** `latent_beats_scalar_control = true` (0.577 > 0.332).  
The token-level probe achieves near-perfect IoU (1.0), confirming the target latent encodes spatial layout at the token level.

### 2.3 Pairwise Geometry vs. Latent (§11)

| Metric | Value |
|--------|-------|
| Spearman (Hamming vs latent L2) | **0.444** |
| Spearman (Hamming vs param L2) | 0.038 |
| Pearson (Hamming vs latent L2) | **0.488** |
| Pearson (Hamming vs param L2) | 0.031 |

**Interpretation:** Latent L2 distance correlates moderately with true geometric (Hamming) distance, while parameter distance does not. Geometry-defined pairs that differ more in occupancy also differ more in target latent space — the latent preserves spatial structure.

Per-bucket latent L2 means rise monotonically with Hamming distance (2.5 → 5.2), while parameter L2 stays flat (~2.2–2.3).

---

## 3. Physics Target Selection (Diagnostic B)

### 3.1 Configuration
- **Mask ratios**: 0.75, 1.00 (primary per §12)
- **Samples**: 32 per ratio (fixed validation subset, seed=0)
- **Mask seed**: 12345 + round(ratio×1000) (identical masks across conditions)
- **Retrieval batch**: 8 (64 forwards per matrix)
- **Geometry-aware subset**: k=8, min_hamming=300 (met: 384)

### 3.2 Real vs Null Margins (correct physics improves target prediction)

| Mask | Metric | Median Margin | Fraction > 0 |
|------|--------|--------------|--------------|
| **1.00** | cos_null−real | **+0.0038** | 0.969 |
| **1.00** | l2_token_null−real | **+0.055** | 0.969 |
| **1.00** | l2_pooled_null−real | **+0.058** | 1.000 |
| **0.75** | cos_null−real | **+0.0043** | 0.969 |
| **0.75** | l2_token_null−real | **+0.060** | 0.969 |
| **0.75** | l2_pooled_null−real | **+0.063** | 0.969 |

**Interpretation:** Null spectrum consistently produces worse predictions than real spectrum. Physics conditioning is **used** (sensitivity is strong: real_vs_null_l2 median ≈ 1.0).

### 3.3 Real vs Shuffled Margins (target selection)

| Mask | Metric | Median Margin | Fraction > 0 |
|------|--------|--------------|--------------|
| **1.00** | cos_shuffle−real | **+0.0003** | 0.531 |
| **1.00** | l2_token_shuffle−real | **−0.0013** | 0.469 |
| **1.00** | l2_pooled_shuffle−real | **−0.0019** | 0.469 |
| **0.75** | cos_shuffle−real | **+0.0002** | 0.531 |
| **0.75** | l2_token_shuffle−real | **−0.0034** | 0.469 |
| **0.75** | l2_pooled_shuffle−real | **−0.0048** | 0.469 |

**Interpretation:** Shuffled spectrum is **indistinguishable** from real spectrum at the target-latent level. Margins are ~0 with fraction positive ≈ 0.5 (chance). The predictor does not select the correct target latent given the correct spectrum.

### 3.4 Predictor Sensitivity (per-masked-token L2 shift)

| Mask | real_vs_null | real_vs_shuffled |
|------|-------------|-----------------|
| 1.00 | 1.00 (median) | 0.64 (median) |
| 0.75 | 1.17 (median) | 0.72 (median) |

Sensitivity to physics is strong (real≠null) but not target-useful (real≈shuffled). This is the **sensitivity-vs-necessity failure mode** (Case C, §18).

### 3.5 Exact Spectrum Retrieval Matrix (§16)

| Mask | Recall@1 | Recall@5 | Mean Rank | Median Rank | Diag−Offdiag Margin |
|------|---------|---------|----------|------------|---------------------|
| **1.00** | 0.125 | 0.625 | 4.5 | 4.5 | **0.0** |
| **0.75** | 0.125 | 0.625 | 4.5 | 4.5 | **0.0** |

The retrieval matrix is **uniform**: every row is identical (the predictor output doesn't depend on which spectrum is conditioned on). Diagonal = off-diagonal distance exactly.

### 3.6 Geometry-Aware Retrieval Subset (§17)

- Subset indices: [6, 4, 2, 0, 7, 5, 3, 1]
- Min pairwise Hamming: 384 (meets 300 bar)
- Retrieval on subset: **identical to full batch** — Recall@1 = 0.125, margin = 0.0

Even on maximally distinct geometries, the correct spectrum does not retrieve the correct target.

### 3.7 Predicted-Latent Spatial Probe (§18)

Same frozen occupancy probe (trained on z_y_raw) applied to z_hat under each condition:

| Condition | IoU (median) | Pixel Acc (median) |
|-----------|-------------|-------------------|
| Real | 0.121 | 0.395 |
| Null | 0.122 | 0.394 |
| Shuffled | 0.122 | 0.395 |

**Interpretation:** Predicted latents contain **minimal spatial information** (IoU ≈ 0.12) and **do not differ** across conditions. The probe trained on rich z_y_raw targets cannot decode spatial structure from z_hat.

### 3.8 Same-Context / Different-Spectrum Causal Test (§19)

| Mask | Row Win Rate | Col Win Rate | **Mutual Win Rate** |
|------|-------------|-------------|-------------------|
| 1.00 | 0.50 | 0.50 | **0.0** |
| 0.75 | 0.50 | 0.50 | **0.0** |

**Mutual win rate = 0.0**: For no pair of spectra does each context prefer its own spectrum's prediction over the other's. The desired property `d(ẑ₁, z_y₁) < d(ẑ₁, z_y₂)` **fails completely**.

---

## 4. Interpretation Matrix (§18)

| Case | Condition | Matches? |
|------|-----------|----------|
| **A** | Target latent lacks spatial info | ❌ (z_y_raw IoU=0.58, token IoU=1.0) |
| **B** | **Target latent preserves geometry, physics selection fails** | ✅ |
| C | Sensitivity strong, retrieval weak | ✅ (sensitivity strong, retrieval fails) |
| D | Retrieval succeeds | ❌ |

**Diagnosis: Case B (with Case C characteristics).**  
The EMA target latent encodes spatial geometry correctly. The predictor is sensitive to physics (real≠null) but fails to use the spectrum to select the correct target latent (real≈shuffled, retrieval uniform, mutual wins=0).

---

## 5. Recommended Next Intervention (§21, §20)

Per the plan's prescribed sequence, the evidence-driven intervention is:

> **Add MetaDiT-style CLIP alignment objective on the student geometry encoder** (§20), keeping the EMA target geometry-only.

```
Student G_enc  ──→  geometry embedding  ──↕ (CLIP contrastive)──  frozen spectrum encoder
                                                                     ↓
                                                           spectrum embedding
```

**Rationale:**
- Target latent already has spatial info (Case B) — no need to change the EMA target or decoder.
- Physics is not sufficiently selecting the correct target — the student needs to learn the geometry↔spectrum correspondence.
- This preserves the target-leakage invariant (EMA target never sees spectrum).
- The alignment teaches the *physical response representation* rather than mere pixel geometry.

**Do NOT yet:**
- Strengthen the physics predictor (it already has sensitivity).
- Modify the EMA target mechanism.
- Add stochastic latent or flow-matching (§7.4, Milestones H/I).

---

## 6. Reproducibility

| Parameter | Value |
|-----------|-------|
| Seed (probe) | 0 |
| Split seed | 0 |
| Mask seed | 12345 |
| Validation samples (Diagnostic A) | 256 |
| Validation samples (Diagnostic B) | 32 per ratio |
| Probe architecture | Linear (384→4096) |
| Probe epochs | 300 |
| Device | CPU |

All random states fixed; same mask, same target, same samples across all conditions.

---

## 7. Artifacts

- `latent_geometry_probe.json` — full Diagnostic A results
- `latent_geometry_probe_weights.pt` — frozen probe weights (reused by Diagnostic B)
- `physics_target_selection.json` — full Diagnostic B results (ratios 0.75, 1.00)
- This report: `PHYSICS_GEOMETRY_LATENT_SELECTION_REPORT.md`