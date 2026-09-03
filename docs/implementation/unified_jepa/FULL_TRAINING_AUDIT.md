# Unified Occupancy-Parameter-Spectrum JEPA: Full Training Audit

**Audit date:** 2026-09-03
**Repository state audited:** `d7a4ef3` (`docs: record corrected collapse and null-goal evaluation`)
**Architecture:** `unified_occ_param_spectrum_jepa_v1`
**Scope:** data contract, masking, conditioning, objective, decoder, physics loop, validation, and the completed Kaggle comparison runs.

## 1. Executive conclusion

The unified pipeline is mechanically connected end to end. I found no direct ground-truth geometry leak into the student predictor, no exact train/validation/test sample duplication, and no physics-surrogate bypass in the active physics path.

The main failure is not that the released spectrum encoder is intrinsically bad. The trained model is learning to ignore the spectrum condition because the active objective does not require the prediction to depend on that condition. Two implementation/design issues make this worse:

1. The unified occupancy decoder has no active occupancy reconstruction loss. With `lambda_phys=0`, its parameters receive no training signal, so the baseline decoder is effectively untrained.
2. `lambda_phys=0.1` is small relative to the VICReg/JEPA terms, while the target latent is deliberately spectrum-free. Physics improves decoded spectrum error, but does not force real-goal output to differ from null-goal output.

The current recommendation is to keep the released spectrum encoder, fix the decoder/objective training contract, add a hard-stratum real-versus-shuffled goal-utility loss or gate, and only then consider changing the spectrum-input architecture.

## 2. Architecture audited

The active model separates the data into three semantic inputs:

    occupancy [1, 64, 64]       three global scalars [l, h, r]
              \                         /
               \                       /
                OccupancyEncoder + ScalarEncoder
                             ↓
                     masked context tokens

    target spectrum [real, imag, 301]
                             ↓
                    Released SpectrumEncoder
                             ↓
            c_physics [384] + a_goal [16, 384]
                             ↓
                    FusionEncoder [192-D]
                             ↓
            GCLCT predictor, conditioned by c_physics
                             ↓
           occupancy latent prediction + scalar prediction
                             ↓
        OccupancyDecoder + ScalarDecoder → reconstructed geometry
                             ↓
                  frozen MetaDiT EM surrogate
                             ↓
                 predicted spectrum versus target

The implementation locations are:

- [`src/data/factorize.py`](../../../src/data/factorize.py): occupancy/scalar separation and MetaDiT broadcast reconstruction.
- [`src/encoders/occupancy_encoder.py`](../../../src/encoders/occupancy_encoder.py): single-channel occupancy ViT.
- [`src/encoders/scalar_encoder.py`](../../../src/encoders/scalar_encoder.py): six-value scalar input with known flags and FiLM parameters.
- [`src/encoders/spectrum_encoder.py`](../../../src/encoders/spectrum_encoder.py): released spectrum encoder plus learned pooling adapters.
- [`src/fusion/fusion_encoder.py`](../../../src/fusion/fusion_encoder.py): joint occupancy/goal/scalar fusion.
- [`src/predictor/gclct.py`](../../../src/predictor/gclct.py): masked-token predictor with global physics FiLM conditioning.
- [`src/assembly.py`](../../../src/assembly.py): top-level forward path and geometry assembly.
- [`src/physics/physics_loop.py`](../../../src/physics/physics_loop.py): frozen surrogate and spectrum loss.
- [`src/losses/unified_losses.py`](../../../src/losses/unified_losses.py): active training objective.

## 3. Data and split audit

The staged MetaDiT data contains:

| Split | Samples | Raw fields |
|---|---:|---|
| train | 139,906 | `pattern`, `parameter`, `real`, `imag` |
| validation | 17,488 | `pattern`, `parameter`, `real`, `imag` |
| test | 17,489 | `pattern`, `parameter`, `real`, `imag` |

The loader uses the same sample index for pattern, parameters, real spectrum, and imaginary spectrum. It creates the legacy surrogate input only at the dataset/surrogate boundary:

    G0 = occupancy * r_atom / 5
    G1 = occupancy * h_atom
    G2 = l_lattice / 3 everywhere

### Duplicate checks

The audit found:

- no exact full-spectrum duplicates across train/validation/test;
- no exact `pattern + parameter` duplicates across the split pairs;
- repeated scalar parameter combinations across splits;
- repeated occupancy patterns across splits: 676 train/validation, 684 train/test, and 170 validation/test pattern hashes.

The repeated pattern counts are not exact sample leakage because the associated scalar values differ, but they are a family-overlap risk. Partial-mask completion metrics may be optimistic because an occupancy pattern family can occur in training and evaluation. This matters less for the pure inverse-design case where occupancy is fully masked, but a strict follow-up should use a grouped split by occupancy pattern or pattern family.

## 4. Masking audit

The training configuration is in [`configs/unified.yaml`](../../../configs/unified.yaml):

    train_mask_ratios: [0.25, 0.5, 0.75, 1.0]
    train_mask_ratio_probs: [0.25, 0.35, 0.25, 0.15]

The sampler is [`scripts/train/train_unified.py`](../../../scripts/train/train_unified.py), function `sample_mask_ratio()`. It samples one ratio for each batch and excludes `0.0` from training.

The exact full-mask behavior is in [`src/data/mask.py`](../../../src/data/mask.py): ratio `1.0` returns an all-zero mask, where zero means masked. Therefore 100% means all 256 occupancy tokens are hidden.

The completed 1,500-step runs used batch size 2:

| Quantity | Value |
|---|---:|
| optimizer steps | 1,500 |
| batch size | 2 |
| training examples drawn | 3,000 |
| configured full-mask probability | 15.0% |
| observed full-mask batch frequency | 15.333% |
| full-mask batches | 230 |
| full-mask examples | **460** |

The training loader shuffles the 139,906-sample train set without replacement within an epoch, so these 460 examples are fresh in this run. Validation is intentionally fixed: the same 16 validation geometries are evaluated repeatedly so changes in metrics are attributable to the model rather than changing validation samples.

One instrumentation gap remains: the logger records mask ratio and scalar regime separately, not their joint cross-tabulation. Consequently, the exact number of `100% occupancy + all scalars unknown` training examples was not recorded. Given the approximately one-third all-unknown scalar probability, the expected number is about 150 of the 460 full-mask examples.

## 5. Ground-truth leakage audit

### 5.1 Student input

The student receives:

- masked occupancy;
- scalar values zeroed where unknown, with explicit known flags;
- target spectrum through `SpectrumPath`;
- no complete geometry tensor.

At 100% occupancy masking, `apply_mask_to_pixels()` gives the occupancy encoder no visible occupancy pixels. At all-unknown scalar masking, all three scalar values are zeroed and only the unknown flags remain.

### 5.2 EMA target

The EMA target encoder receives complete occupancy and true scalars under `torch.no_grad()`. This is intentional JEPA target construction. It is not a predictor input and does not create a direct geometry leak into `z_hat`.

### 5.3 Physics path

`physics_loss_from_out()` receives `z_hat` and the scalar predictions, decodes them, assembles the surrogate geometry, and then calls the frozen surrogate. Known occupancy pixels and known scalars are retained by design. At 100% occupancy masking there are no known occupancy pixels to retain.

The surrogate does not receive the ground-truth geometry. It receives the decoded geometry, with only explicitly known inputs substituted.

### 5.4 Spectrum used as both input and target

The target spectrum is used twice:

1. as the requested condition through `SpectrumPath`;
2. as the target for `L_phys`.

That is correct for conditional inverse design. It is not a leak. However, because there is no counterfactual or shuffled-goal training constraint, the model can satisfy the objective without using the condition.

## 6. Physics-input audit

### 6.1 Released spectrum encoder

The released MetaDiT spectrum encoder was previously verified against the released MetaDiT model:

- output shape: `[B, 301, 256]`;
- deterministic output for identical input;
- no cross-sample representation collapse;
- bitwise consistency with the encoder embedded in the released MetaDiT DiT model.

This makes replacing the released encoder premature.

### 6.2 Current conditioning routes

The unified model has two spectrum routes:

- `a_goal`: 16 learned-query pooled spectrum tokens, projected from 384-D to 192-D, then placed in the fusion/predictor key-value sequence;
- `c_physics`: mean-pooled spectrum features projected to 384-D, then linearly projected to 192-D and used by every GCLCT block's FiLM conditioner.

Both routes are structurally present. Both can also be bypassed:

- the predictor can use occupancy/scalar priors from the fusion sequence;
- the FiLM conditioner is zero-initialized;
- JEPA's target is geometry-only and spectrum-free;
- no active loss says that changing the spectrum must change the prediction.

### 6.3 What the Kaggle result says

The corrected hard-stratum result was:

| Model | Real-goal hard MAE | Null-goal hard MAE | Null minus real |
|---|---:|---:|---:|
| baseline, `lambda_phys=0` | 0.8248 | 0.8244 | -0.000410 |
| physics-fixed, `lambda_phys=0.1` | 0.3452 | 0.3452 | -0.000003 |

At the most important case—100% occupancy mask plus all scalars unknown—the physics-fixed model had:

    real hard spectrum MAE: 0.280814
    null hard spectrum MAE: 0.280821
    goal gain:             0.000006

The result is strong evidence of goal-ignoring collapse. It is not evidence that the spectrum encoder has collapsed. The downstream network is producing a generic physics-compatible answer that is nearly unchanged when the requested spectrum is removed.

### 6.4 Measurement gap

The current unified `validate_suite()` reports real/null comparisons but does not report a shuffled-spectrum condition. The existing standalone `physics_conditioning_audit.py` targets the legacy architecture and is not a valid audit of the unified checkpoint. A unified-specific audit is still required to measure:

- cross-sample variance and effective rank of `c_physics`;
- cross-sample variance and effective rank of `a_goal`;
- real/null prediction sensitivity;
- real/shuffled prediction sensitivity;
- real versus shuffled physics error in the hard stratum.

## 7. Objective and loss audit

The active objective is defined in [`src/losses/unified_losses.py`](../../../src/losses/unified_losses.py):

    L = 25.0 L_inv
      + 25.0 L_var
      +  1.0 L_cov
      +  1.0 L_scalar
      + lambda_phys L_phys

For the physics-fixed model at 100% mask plus all scalars unknown, the final validation components were approximately:

| Term | Raw value | Weighted contribution |
|---|---:|---:|
| `L_inv` | 4.3572 | 108.93 |
| `L_var` | 0.2449 | 6.12 |
| `L_cov` | 242.5027 | 242.50 |
| `L_scalar` | 0.2013 | 0.20 |
| `L_phys` | 18.2726 | 1.83 at `lambda_phys=0.1` |

The physics term is therefore small compared with the representation objective. This can teach the decoder to produce a generally plausible spectrum without creating a strong incentive for the predictor to encode target-specific spectral differences.

## 8. Confirmed implementation defects and risks

### P0: missing active occupancy reconstruction loss

The architecture document specifies `BCEWithLogits` on predicted occupancy. The unified objective does not call `GeometryReconstructionLoss`, `BCEWithLogits`, or an equivalent occupancy loss. The only active decoder training signal is the physics term.

Consequences:

- the `lambda_phys=0` baseline's occupancy decoder is not trained;
- the baseline/fixed Kaggle comparison is not a clean ablation of physics loss;
- occupancy quality is not directly measured by the active training objective;
- a physics model can learn a surrogate-compatible occupancy without learning the correct geometry.

### P0: observed goal-conditioning collapse

Real/null hard-stratum differences are essentially zero. This is a task-level failure, not a tensor-plumbing failure. The model can minimize its current losses while ignoring the requested spectrum.

### P1: no shuffled-goal training gate in the unified run

The design calls for real/null/shuffled evaluation, specifically gated on the full-mask/all-unknown stratum. The current unified report contains real/null controls but no shuffled-goal result. This prevents a complete conditionality claim.

### P1: unconstrained scalar decoder

`ScalarDecoder` outputs unconstrained real values. The dataset ranges are:

    l_lattice: [2.5, 3.0]
    h_atom:    [0.5, 1.0]
    r_atom:    [3.5, 5.0]

The decoder is initialized near the data mean but is not bounded afterward. Out-of-range scalars can send the frozen surrogate out of distribution.

### P1: hard/soft surrogate mismatch

The released surrogate was trained on binary occupancy. The training path uses a straight-through estimator: hard occupancy in the forward pass and sigmoid gradients in the backward pass. This is a documented approximation, not a leak. The validation diagnostics show a large soft-versus-hard spectrum difference, so deployed hard metrics must remain the primary metric.

### P2: spectrum loss and reported metric are different

`L_phys` is normalized SmoothL1 over all 602 real/imag values using each sample's global spectrum standard deviation. The report's main spectrum metric is raw hard MAE. This is acceptable for a first experiment, but it does not specifically emphasize resonance locations or peak errors.

### P2: partial-mask family overlap

Repeated occupancy patterns across split files can make partial-mask completion results optimistic. This is a split-design risk rather than an exact sample leak.

### P2: unified spectrum adapter is not independently audited

The released encoder is known to carry signal, but the learned `proj_g`, `proj_goal`, fusion projection, and predictor projection are not currently reported for rank, variance, or gradient magnitude on a unified checkpoint.

## 9. Kaggle execution record

### 9.1 Data staging

The Kaggle Dataset used for the unified runs was:

    akashkesav/metadit-aaai2026-staging

It contained the MetaDiT train/validation/test split files and released weights, including:

    split_data/train_set.mat
    split_data/val_set.mat
    split_data/test_set.mat
    weights/spec_encoder.pth
    weights/metadit-small.bin
    weights/surrogate_model.bin

### 9.2 Execution environment

The corrected run used:

    Kaggle GPU: Tesla P100 16 GB
    PyTorch:    2.5.1+cu124
    Torchvision: 0.20.1+cu124
    CUDA:       available

The local RTX 3050 machine was used only for code inspection, data checks, syntax checks, and small smoke tests, consistent with the repository's compute policy.

### 9.3 Comparison kernel

Kernel:

    akashkesav/metasurface-unifiedjepa-baseline-vs-physics-fixed

The corrected run used the unified code pinned to commit `5c7f485`, which added:

- 16-sample validation for meaningful sample-level rank;
- corrected rank calculation over all validation batches;
- matched real/null controls;
- isolated baseline and physics-fixed output directories.

The kernel trained:

    baseline:       lambda_phys=0.0
    physics-fixed:  lambda_phys=0.1

The physics-fixed run used a 200-step ramp from zero to `0.1` in the initial comparison. Both runs used the same seed, data, step count, validation subset, and frozen surrogate.

### 9.4 Training procedure

For each optimizer step:

1. Load a shuffled batch from the real train split.
2. Factorize `[B,3,64,64]` into occupancy `[B,1,64,64]` and three scalars.
3. Sample one scalar visibility regime.
4. Sample one occupancy mask ratio from the configured distribution.
5. Generate a random block mask on the 16x16 token grid.
6. Encode masked occupancy and masked/known scalars.
7. Encode the target spectrum through the released spectrum encoder.
8. Fuse occupancy, spectrum-goal, and scalar-summary tokens.
9. Predict masked occupancy latents and scalar values.
10. Encode complete geometry through the EMA target path.
11. Compute VICReg/JEPA and scalar losses.
12. If enabled, decode predicted geometry, run the frozen surrogate, compute `L_phys`, and backpropagate through the surrogate input.
13. Clip gradients, update the optimizer and scheduler, then update the EMA targets.
14. Checkpoint every 100 steps and validate every 50 steps.

Validation evaluated the same 16 geometries over:

    occupancy mask ratios: 0.0, 0.25, 0.5, 0.75, 1.0
    scalar regimes:        all_known, all_unknown, mixed

Each scenario reported latent diagnostics, scalar error, soft/hard physics loss, soft/hard spectrum MAE/MSE, and real/null goal comparisons.

### 9.5 First comparison result

The first 2-sample validation run showed:

| Run | Final loss | Hard spectrum MAE, all | 50% mask | 100% mask |
|---|---:|---:|---:|---:|
| baseline | 12.8869 | 0.3568 | 0.4313 | 0.4970 |
| physics fixed, 0.1 | 12.8740 | 0.4062 | 0.5699 | 0.3416 |

This run was later superseded for rank interpretation because a two-sample effective-rank estimate is not reliable.

### 9.6 Corrected comparison result

The corrected 16-sample run found:

| Run | Sample-rank trend | Final sample rank | Real hard MAE | Null hard MAE |
|---|---|---:|---:|---:|
| baseline | stable/rising | 0.3727 | 0.8248 | 0.8244 |
| physics fixed, 0.1 | stable/rising | 0.3527 | 0.3452 | 0.3452 |

Rank did not collapse. Goal usage did collapse. The physics term materially improved hard spectrum accuracy, especially at high masking, but the model produced almost the same result with the requested spectrum removed.

### 9.7 Physics-weight sweep

The separate sweep kernel was:

    akashkesav/metasurface-unifiedjepa-physics-lambda-sweep

It tested `lambda_phys` values `0.2`, `0.4`, `0.8`, and `1.0`. The single-seed sweep favored `0.8` and `1.0` for raw surrogate spectrum accuracy, but this is not sufficient to select a production weight because the baseline comparison is confounded by the missing occupancy reconstruction loss and the sweep did not establish goal dependence.

## 10. Recommended repair sequence

The repairs should be staged and re-evaluated after each material change.

### Repair 1: make the decoder comparison valid

Add an occupancy reconstruction loss to the active unified objective, applied at least to the missing occupancy pixels. Report occupied-class IoU/F1 and predicted occupancy fraction, not only overall pixel accuracy.

The baseline and physics runs must then both train the decoder before physics is compared.

### Repair 2: record the joint training curriculum

Extend `RegimeLogger` to record `(mask_ratio, scalar_regime)` pairs. This will give the exact number of pure inverse-design training examples and ensure that a claimed increase in 100% masking actually increases the hard stratum rather than only the easier all-known cases.

### Repair 3: enforce spectrum utility

For the full-mask/all-unknown stratum, add a real-versus-shuffled goal utility constraint or an equivalent goal-sensitivity loss. The acceptance test should be:

    real-goal hard physics error < shuffled-goal hard physics error

and the decoded geometry should change when the target spectrum changes.

Null-goal sensitivity alone is insufficient because a model can respond to a null token without correctly using the spectrum identity.

### Repair 4: constrain scalar outputs

Use bounded output transforms or explicit range constraints for `l_lattice`, `h_atom`, and `r_atom`. Log predicted minima and maxima during validation.

### Repair 5: strengthen the goal interface only if needed

Keep the released spectrum encoder, but consider adding a direct gated cross-attention path from `a_goal` into masked occupancy queries and/or the occupancy decoder. This should be tested only after Repairs 1-3; otherwise a larger goal pathway can remain unused while making the model harder to diagnose.

### Repair 6: improve spectral scoring

After goal dependence is demonstrated, add resonance-aware diagnostics and, if justified by measured error concentration, frequency weighting or peak/derivative terms. Do not change the physics input representation solely because the current model ignores it.

## 11. Acceptance criteria for the next run

The next cloud run should not be considered successful unless all of the following are recorded:

- exact joint counts for mask ratio and scalar regime;
- occupancy IoU/F1 and predicted occupancy-fraction statistics;
- bounded scalar predictions within the physical ranges;
- real/null/shuffled evaluation on the same hard-stratum examples;
- real-goal hard physics error lower than shuffled-goal error;
- measurable decoded-geometry change under different target spectra;
- nonzero unified `c_physics`/`a_goal` variance and gradients;
- no gradients in the frozen surrogate or EMA target;
- sample-rank trend stable or improving;
- baseline and physics-fixed models trained with the same decoder supervision;
- all results recorded in a new cloud-run report with the exact commit and staged dataset identifier.

## 12. Audit verdict

The data factorization and frozen-physics plumbing are fundamentally sound. The model is not currently a reliable spectrum-conditioned inverse designer because the objective permits goal ignoring, and the no-physics baseline does not train its occupancy decoder.

The correct next move is not to discard the spectrum encoder. The correct next move is to repair the training/evaluation contract so that:

    geometry quality is supervised fairly
    and
    target-spectrum dependence is required and measured

Only if the unified-specific conditioning audit then shows that the spectrum embeddings themselves lose information should the spectrum-input architecture be redesigned.

## 13. Evidence and artifact locations

Tracked evidence:

- [`checkpoints/unified/REPORT.md`](../../../checkpoints/unified/REPORT.md)
- [`configs/unified.yaml`](../../../configs/unified.yaml)
- [`src/assembly.py`](../../../src/assembly.py)
- [`src/physics/physics_loop.py`](../../../src/physics/physics_loop.py)
- [`src/losses/unified_losses.py`](../../../src/losses/unified_losses.py)
- [`docs/implementation/unified_jepa/architecture_v5.md`](architecture_v5.md)

Local raw Kaggle artifacts retained outside the Git history:

    .kaggle_corrected_results/
    .kaggle_compare_results/
    .kaggle_sweep_results/
    .kaggle_unified_compare2/
    .kaggle_unified_sweep2/
    .metadit_stage_20260903/

These local directories contain downloaded Kaggle result JSON, kernel metadata, staged data, and the captured run material. Large datasets, checkpoints, and raw logs are intentionally not committed to the source repository.
