# OpenCode Master Execution — Unified Occupancy–Parameter–Spectrum JEPA

## Authority hierarchy

Use this hierarchy exactly:

1. `architecture_v5.md` = **architectural authority**
2. This file = **execution controller**
3. `01_...` through `05_...` = **phase execution instructions**
4. Actual repository code/tests/configs = **implementation/API source of truth**

The v5 document defines the intended architecture and scientific reasoning. The phase MDs tell you how to implement it. The repository determines the exact existing APIs and interfaces.

## Execution order

Execute strictly:

```text
01_data_contract_and_new_modules.md
02_model_integration_192d.md
03_training_and_objective.md
04_physics_loop_and_scenarios.md
05_tests_evaluation_cleanup.md
```

For every phase:

1. Read the phase MD completely.
2. Inspect all relevant existing repository files before editing.
3. Implement only that phase.
4. Run the phase's required tests/smoke checks.
5. Inspect `git diff`, `git diff --check`, and affected imports/configs.
6. Do not continue if acceptance criteria fail.
7. Only then read and execute the next phase.

## Critical repository-verification rule

For every implementation detail that is not explicitly fixed by `architecture_v5.md`, inspect the actual repository and resolve it from code/tests/config.

Verify before implementing:

```text
dataset channel semantics
occupancy convention
scalar normalization
spectrum preprocessing
MetaDiT geometry assembly
MetaDiT surrogate contract
SpectrumPath contract
GCLCT conditioning interface
EMA implementation
mask representation
checkpoint schema
config conventions
```

If a phase MD conflicts with the repository, do NOT silently invent a workaround. Determine whether `architecture_v5.md` resolves the conflict. If not, stop at that phase and report the exact conflict before proceeding.

## Scientific architecture

Semantic sample:

```text
M[64,64] binary occupancy
l_lattice scalar
h_atom scalar
r_atom scalar
spectrum[2,301]
```

Internal representation:

```text
occupancy only
+
explicit l/h/r values + known flags
+
spectrum condition
```

The legacy `[3,64,64]` broadcast tensor is used only at the MetaDiT surrogate boundary.

New path:

```text
occupancy + scalar condition + spectrum
            ↓
occupancy encoder + scalar MLP/FiLM
            ↓
192-D fusion/context encoder
            ↓
192-D GCLCT
            ↓
occupancy token prediction + scalar summary prediction
            ↓
occupancy decoder + scalar heads
            ↓
deterministic MetaDiT geometry [B,3,64,64]
            ↓
frozen MetaDiT surrogate
            ↓
spectrum
```

EMA:
- occupancy EMA = JEPA occupancy target only;
- `scalar_mlp_ema` = target-side FiLM conditioning only;
- no scalar EMA latent loss target.

## Old Milestone-B checkpoint

The old architecture is:

```text
3-channel geometry
384-D hidden
384-D predictor
```

The new architecture is:

```text
1-channel occupancy
192-D hidden
192-D predictor
```

Do NOT claim the old 384-D checkpoint is shape-compatible with the new transformer.

Do not slice/crop 384-D tensors into 192-D tensors.

Keep old Milestone-B code/checkpoints as reproducible historical reference.

## Forbidden

```text
no BI-JEPA
no second physics predictor
no second geometry decoder
no internal 3-channel broadcast representation for new masking
no generic 3-channel decoder loss
no spectrum decoder
no modification of external/metadit source
no modification of raw MAT files
no large training run before smoke gates
no large hyperparameter sweep
```

## Final definition of done

The active new path must support:

```text
occupancy-only spatial input
independent scalar masking
target spectrum conditioning
192-D occupancy latent
EMA occupancy target
stable target-side scalar FiLM conditioning
scalar direct regression
one occupancy decoder
three scalar outputs
deterministic MetaDiT geometry assembly
differentiable frozen MetaDiT physics loss
full occupancy masking
full scalar masking
separate pure-inverse / partial-parameter / retrofit evaluation
real/null/shuffled tests
occupancy majority-collapse test
generative diversity reporting
nearest-neighbor baseline
```

Do not declare scientific success merely because code runs. The hard pure-inverse-design scenario and domain baseline must be evaluated.
