# Phase 1 — Data Contract and New Modules

## Objective

Implement the factorized representation and standalone modules. Do not replace the training loop yet.

## 1. First inspect the repository

Before editing, inspect:

```text
src/data/dataset.py
src/data/mask.py
src/encoders/geometry_encoder.py
src/encoders/context_encoder.py
src/encoders/spectrum_encoder.py
src/encoders/target_encoder.py
src/predictor/gclct.py
tests covering these files
configs/milestone_b.yaml
```

Confirm all dimensions/normalization directly from source.

Do not rely on prose assumptions where source code can answer the question.

## 2. Semantic data contract

Expose the new model's semantic inputs as:

```text
occupancy      [B,1,64,64]
scalar_values  [B,3]      # l,h,r
scalar_known   [B,3] bool
spectrum       [B,2,301]
occupancy_mask [B,16,16]
```

Do not use the legacy `[B,3,64,64]` broadcast image as the internal masked representation.

Do not alter raw MAT schema or corrupt existing dataset interfaces.

If compatibility requires reading the legacy tensor, factor it back into:

```text
occupancy
l
h
r
```

using the exact current repository convention.

Add tests proving round-trip equivalence:

```text
factor(legacy_geometry)
→ assemble(occupancy,l,h,r)
```

matches the original legacy geometry exactly on real and synthetic samples.

## 3. Scalar masking

Create:

```text
src/data/scalar_mask.py
```

Return:

```text
masked_values [B,3]
known_flags   [B,3] bool
```

For an unknown scalar:

```text
value = 0
flag  = 0
```

For a known scalar:

```text
value = exact configured representation
flag  = 1
```

Support configurable regimes:

```text
all known
all unknown
independent masking
correlated masking
```

Do not infer missingness from the numeric sentinel alone.

Do not silently normalize scalars twice. Use the repository's verified scalar convention and store it in config/checkpoint metadata.

## 4. Occupancy encoder

Create:

```text
src/encoders/occupancy_encoder.py
```

Contract:

```text
[B,1,64,64] → [B,256,192]
```

Use the same spatial tokenization convention as the existing geometry encoder:

```text
Conv2d(1,192,kernel=4,stride=4)
→ 16×16
→ 256 tokens
```

Use fresh 192-D positional embeddings generated with the existing sin/cos implementation.

Use a 192-D transformer stack with the depth/head count explicitly confirmed by `architecture_v5.md`.

Apply scalar FiLM after each block:

```text
x = gamma * x + beta
```

Initialize FiLM to identity:

```text
gamma = 1
beta = 0
```

### Weight transfer rule

Do not copy 384-D transformer blocks into 192-D.

The old `GeometryEncoder` is a structural template, not a compatible state dict.

The released MetaDiT patch embed may be adapted only if the exact source/target tensor shapes and semantics are verified. Otherwise initialize from scratch.

Do not invent a "center/sum/slice" transfer rule without checking actual shapes first.

## 5. Scalar encoder

Create:

```text
src/encoders/scalar_encoder.py
```

Input:

```text
[B,6]
```

representing:

```text
[l_value, l_known, h_value, h_known, r_value, r_known]
```

Use:

```text
Linear(6,128)
→ GELU
→ Linear(128,128)
```

Produce:

```text
per-occupancy-block FiLM parameters
scalar_summary [B,192]
scalar predictions only through a dedicated downstream head
```

Instantiate two copies:

```text
live scalar MLP
scalar_mlp_ema
```

The EMA copy is used ONLY to condition the EMA occupancy target.

It is never a scalar JEPA loss target.

## 6. Fusion encoder

Create:

```text
src/fusion/fusion_encoder.py
```

Inputs:

```text
occupancy tokens [B,256,192]
goal tokens      [B,16,192]
scalar summary   [B,1,192]
```

Project the existing released spectrum goal tokens:

```text
[B,16,384] → Linear(384,192)
```

Use the 2-layer 192-D fusion transformer specified by `architecture_v5.md`.

Keep token roles explicit:

```text
256 occupancy tokens
16 spectrum goal tokens
1 scalar-summary token
```

## 7. Unit tests

Add tests for:

```text
data factorization
legacy round-trip
scalar known/unknown flags
occupancy encoder shape
scalar encoder shape
FiLM identity initialization
fusion token count
fusion width
scalar_mlp_ema independence
```

Before completing this phase:

```bash
python -m pytest -q
```

Do not modify the historical Milestone-B training loop.
