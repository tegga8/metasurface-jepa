# Phase 4 — Physics Loop and Scenario Evaluation

## Objective

Connect the factorized design outputs to the existing frozen MetaDiT surrogate and implement the three deployment scenarios.

## 1. One authoritative geometry assembler

Create one helper for surrogate-boundary conversion:

```python
assemble_metadit_geometry(
    occupancy,
    l_hat,
    h_hat,
    r_hat,
)
```

Before coding, inspect the current repository dataset and MetaDiT preprocessing and verify the exact transformation.

The intended convention in the authority document is:

```text
G0 = occupancy * r_hat/5
G1 = occupancy * h_hat
G2 = l_hat/3 everywhere
```

Only use these exact constants if confirmed by source.

Return:

```text
[B,3,64,64]
```

The broadcast geometry exists only at this boundary.

## 2. Structural invariants

The assembler must guarantee:

```text support(G0) == support(G1)
G0 occupied value is constant per sample
G1 occupied value is constant per sample
G2 is constant over every pixel per sample
```

Add tests for every invariant.

## 3. Occupancy for physics

Default training representation:

```text occupancy_soft = sigmoid(logits)
```

Use soft occupancy for differentiability.

Before relying on it, run a surrogate-input sanity test comparing:

```text real binary geometry
soft occupancy geometry
```

If the surrogate is materially out-of-distribution on soft fields, implement a documented straight-through estimator.

Do not silently choose STE without the check.

## 4. Frozen MetaDiT surrogate

Use the existing integrated surrogate.

Expected contract MUST be verified from actual source:

```text geometry [B,3,64,64]
→ spectrum [B,2,301]
```

Freeze parameters:

```python
for p in surrogate.parameters():
    p.requires_grad_(False)
```

Keep it in eval mode.

But do NOT disable autograd through the input.

Physics path:

```text L_phys
→ surrogate output
→ dS/dG
→ assembled geometry
→ occupancy/scalar decoder
→ predictor
→ student encoder
```

Verify this with a real gradient test.

## 5. Spectrum loss

Inspect actual MetaDiT spectrum preprocessing before selecting the normalization.

Use the same normalized real/imag convention where appropriate.

Default:

```text normalized L1 or SmoothL1
```

Do not use magnitude/phase as the primary loss without strong evidence.

No second spectrum predictor.

## 6. Preservation loss

For partial/retrofit scenarios:

```text L_preserve
```

must penalize changes to known occupancy/scalars.

Do not permit physics loss alone to overwrite observed geometry.

## 7. Scenario A — pure inverse design

Inputs:

```text occupancy fully unknown
all l/h/r unknown
target spectrum known
```

This is one-to-many.

Report separately:

```text spectrum error
scalar error
occupancy metrics when paired GT exists
diversity
```

Do not make paired pixel reconstruction the sole criterion.

## 8. Scenario B — partial-parameter conditioning

Inputs:

```text occupancy masked/unknown
some l/h/r known
target spectrum known
```

Report independently.

## 9. Scenario C — retrofit/constrained completion

Inputs:

```text occupancy mostly known, region masked
l/h/r known
target spectrum known
```

Report independently.

Never pool A/B/C into one score.

## 10. Real/null/shuffled dependence

Evaluate:

```text real spectrum
null spectrum
shuffled spectrum from another sample
```

Separately for:

```text easy:
low occupancy mask + scalars known

hard:
full occupancy mask + scalars unknown
```

Primary spectrum-dependence gate is hard-stratum:

```text real must outperform shuffled
```

not merely real > null.

Perform the analogous condition test for scalar conditioning using shuffled scalars.

## 11. Generative diversity

If stochastic sampling is implemented:

```text fixed target
→ repeated generations
→ pairwise occupancy distance / IoU
```

If deterministic:

```text report determinism explicitly
```

Do not claim multimodal inverse design unless multiple outputs are genuinely sampled.

## 12. Baselines

Implement nearest-neighbor retrieval:

```text target spectrum
→ nearest training spectrum
→ associated training geometry
```

Evaluate through the same surrogate and spectrum metric.

Where feasible, compare against the original MetaDiT generative baseline on pure inverse design.

## 13. Acceptance

Do not scale until:

```text geometry assembly tests pass
surrogate gradient test passes
soft/hard occupancy behavior is characterized
real/null/shuffled hard-stratum evaluation works
all three scenario evaluators work
nearest-neighbor baseline works
```
