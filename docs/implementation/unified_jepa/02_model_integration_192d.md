# Phase 2 — Integrate the 192-D Architecture

## Objective

Integrate the new factorized representation into the active model.

## 1. Read actual interfaces first

Inspect:

```text
src/assembly.py
src/predictor/gclct.py
src/encoders/spectrum_encoder.py
src/encoders/target_encoder.py
src/losses/jepa_loss.py
src/losses/vicreg.py
```

Verify:
- current GCLCT conditioning dimensions;
- spectrum encoder output dimensions;
- EMA API;
- loss assumptions about token count/dimension.

Adapt code to the actual interfaces rather than assuming the old Milestone-B signatures.

## 2. New forward contract

The active unified model must accept semantically:

```text
occupancy [B,1,64,64]
scalar_values [B,3]
scalar_known [B,3]
spectrum [B,2,301]
occupancy_mask [B,16,16]
```

A compatibility adapter may accept legacy batches, but internally the new path must remain factorized.

## 3. Spectrum path

Keep the released spectrum encoder frozen.

Existing outputs:

```text
c_physics [B,384]
a_goal    [B,16,384]
```

Add:

```text
a_goal_192 = Linear(384,192)(a_goal)
```

Keep `c_physics` at its verified width if required by the actual GCLCT API.

If GCLCT currently assumes `c_physics == hidden`, modify only its conditioning projection so:

```text
384-D c_physics
→ 192-D conditioning
```

Do not arbitrarily change SpectrumPath's released encoder.

## 4. Fusion

Feed:

```text
256 occupancy tokens
+ 16 projected goal tokens
+ 1 scalar summary token
```

into the new 192-D fusion/context stack.

The output must preserve a clear separation between:

```text
occupancy context
scalar summary
goal tokens
```

## 5. Predictor

Adapt GCLCT to 192-D mechanism while preserving:

```text self-attention
cross-attention
FiLM conditioning
MLP
```

Configuration from authority:

```text hidden = 192
heads  = 6
depth  = 8
```

unless actual `architecture_v5.md` values differ; the authority wins.

Queries:

```text 256 occupancy mask queries
+ 1 scalar-summary query
```

The predictor must expose:

```text occupancy_pred [B,256,192]
scalar_summary_pred [B,192]
```

Do not include the scalar summary in the occupancy JEPA token loss.

## 6. Target encoder

Create an EMA copy of the new occupancy encoder.

Target input:

```text full unmasked occupancy
+
true l/h/r
```

Target FiLM conditioning:

```text scalar_mlp_ema
```

Target output:

```text z_y_raw [B,256,192]
```

No gradient through target.

No scalar EMA loss target.

`scalar_mlp_ema` exists only to make target-side FiLM stable.

## 7. JEPA/VICReg objective integration

Reuse existing mechanisms after verifying they support 192-D:

```text masked occupancy z_hat vs z_y_raw
VICReg on occupancy representation
```

Scalar objective:

```text direct L1/Huber regression
```

on scalar positions marked unknown.

Do not create a scalar latent regression target.

Do not add physics loss until Phase 3.

## 8. Checkpoint schema

Introduce a distinct architecture ID, e.g.:

```text unified_occ_param_spectrum_jepa_v1
```

Store:

```text architecture_id
configuration
occupancy EMA state
scalar_mlp_ema state
spectrum projection state
mask configuration
```

Do not load old Milestone-B checkpoints into the new architecture as if they were compatible.

## 9. Smoke test

Use a tiny synthetic/real batch:

```text occupancy [2,1,64,64]
scalars [2,3]
known [2,3]
spectrum [2,2,301]
mask [2,16,16]
```

Verify:

```text occupancy latent [2,256,192]
predicted occupancy latent [2,256,192]
scalar output [2,3]
target latent [2,256,192]
```

Verify gradient ownership:

```text live student modules: gradients
occupancy EMA: none
scalar_mlp_ema: none
released spectrum encoder: none
```

Run:

```bash
python -m pytest -q
```

Do not train at scale until this passes.
