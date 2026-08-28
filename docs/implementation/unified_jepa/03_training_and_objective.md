# Phase 3 — Unified Training and Objective

## Objective

Implement the training loop defined by `architecture_v5.md`.

Create:

```text
scripts/train/train_unified.py
```

Keep the old Milestone-B trainer intact as a reproducible reference.

## 1. Training data construction

Every batch independently samples:

```text occupancy mask
scalar known/unknown mask
target spectrum
```

Occupancy masking uses the existing `BlockMasker` after verifying its output convention against the new single-channel input.

Scalar masking uses `src/data/scalar_mask.py`.

Do not let occupancy masking accidentally overwrite scalar inputs.

Do not let scalar masking alter occupancy.

## 2. Forward path

Student:

```text
masked occupancy
+
masked l/h/r + flags
        ↓
occupancy encoder + live scalar FiLM
        ↓
fusion with target-spectrum goal tokens
        ↓
GCLCT predictor with c_physics
        ↓
occupancy latent predictions + scalar prediction
```

Target:

```text full occupancy + true l/h/r
        ↓
EMA occupancy encoder
        ↑
scalar_mlp_ema for target FiLM
        ↓
z_y_raw
```

## 3. Losses

Use:

```text
L_jepa
+ L_vicreg
+ L_scalar
+ lambda_phys * L_phys
```

Initially:

```text lambda_phys = 0
```

until the no-physics unified architecture is numerically stable.

Scalar loss:

```text direct L1/Huber
```

only on unknown scalar positions.

JEPA loss:

```text only masked occupancy tokens
```

VICReg:

```text on occupancy representation
```

Do not add a scalar EMA latent loss.

## 4. Mask curriculum

Occupancy ratios must include:

```text low
medium
high
full / ~1.0
```

The exact distribution must be configurable.

Scalar regimes must include:

```text all known
all unknown
mixed
```

Log the actual regime frequencies.

The curriculum must not be evaluated as one pooled metric.

At minimum report:

```text easy:
low occupancy mask + scalars known

hard:
full occupancy mask + all scalars unknown
```

The hard regime is the key inverse-design regime.

## 5. Physics loss staging

Use staged training:

```text A. forward-only smoke
B. tiny training with physics loss disabled
C. tiny training with small physics loss
D. controlled run
E. scale only after gates pass
```

Do not immediately turn on a large physics loss.

Ramp `lambda_phys` from zero.

## 6. Frozen references

During student backprop:

```text released spectrum encoder: frozen/no parameter gradient
occupancy EMA: frozen/no gradient
scalar_mlp_ema: frozen/no gradient
MetaDiT surrogate: frozen parameters
```

However, if physics loss is active:

```text MetaDiT forward MUST remain differentiable with respect to geometry input.
```

Never use `torch.no_grad()` around the surrogate in a physics-loss step.

## 7. Checkpoint/resume

Save:

```text global step
epoch
optimizer
scheduler
occupancy EMA
scalar_mlp_ema
mask RNG/state
physics-loss schedule
architecture configuration
```

Do a tiny resume-equivalence test.

## 8. Acceptance

Phase passes only when:

```text forward works
backward works
expected student gradients exist
EMA gradients absent
scalar masking works
full-mask batches occur
losses finite
resume works
pytest passes
```

No large Kaggle run yet.
