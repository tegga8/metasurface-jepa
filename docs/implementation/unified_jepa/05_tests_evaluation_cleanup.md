# Phase 5 — Tests, Evaluation, and Repository Cleanup

## Objective

Make the refactor reproducible, testable, and unambiguous without deleting useful historical work.

## 1. Full test suite

Run:

```bash
python -m pytest -q
```

Fix all failures introduced by the refactor.

Preserve historical Milestone-B tests/reference behavior where it is still intended.

## 2. Architecture contract tests

Verify exact shapes:

```text occupancy input             [B,1,64,64]
occupancy latent                 [B,256,192]
goal tokens                      [B,16,192]
scalar summary                  [B,1,192]
predicted occupancy latent       [B,256,192]
predicted scalar values          [B,3]
assembled surrogate geometry     [B,3,64,64]
surrogate spectrum               [B,2,301]
```

If actual repository APIs expose a different but authority-compatible boundary, document it rather than silently forcing a mismatch.

## 3. Data invariants

For the assembled MetaDiT geometry verify:

```text support(channel0) == support(channel1)
channel0 occupied values constant per sample
channel1 occupied values constant per sample
channel2 spatially constant per sample
```

Use both synthetic and real samples.

## 4. Mask isolation

Verify:

```text occupancy masking does not modify scalar values/flags
scalar masking does not modify occupancy
```

For full masking:

```text no visible occupancy remains
known flags exactly match the intended scalar regime
```

## 5. EMA stability

Verify:

```text occupancy EMA gets no student-backprop gradient
scalar_mlp_ema gets no student-backprop gradient
live scalar MLP receives gradients
EMA updates occur at configured momentum
```

Verify target-side FiLM actually uses `scalar_mlp_ema`, not the live scalar MLP.

## 6. Physics gradient

With real MetaDiT surrogate:

```text surrogate parameter grad = None
geometry input grad exists
geometry input grad finite
decoder receives physics gradient
predictor receives physics gradient
```

This must be an automated regression test.

Do not use `torch.no_grad()` around the surrogate in this test.

## 7. Spectrum dependence

Automate:

```text real
null
shuffled
```

for both:

```text easy regime
hard regime
```

Hard regime:

```text occupancy fully masked
all scalars unknown
```

The evaluator must refuse to treat a pooled curriculum score as proof of spectrum use.

## 8. Scalar dependence

Repeat the real/null/shuffled concept with scalar conditions.

Require that correct scalar conditions beat shuffled conditions on the selected task metric in the hard/relevant regime.

## 9. Generative mode-collapse check

For repeated or controlled-perturbation generation:

```text report design diversity
report spectrum similarity
report whether different target spectra cause different outputs
```

Do not substitute raw latent-rank statistics for this output-level test.

## 10. Occupancy-majority-collapse check

Report:

```text occupied-class IoU
occupied-class F1
predicted occupancy fraction
ground-truth occupancy fraction
```

Add synthetic tests for:

```text all-empty prediction
all-occupied prediction
```

and confirm the evaluator flags both.

## 11. Scenario evaluation

Create/finish:

```text
scripts/eval/eval_scenarios.py
```

Report separately:

```text pure inverse design
partial-parameter conditioning
retrofit
```

For each include:

```text spectrum error
occupancy IoU/F1 where applicable
scalar MAE
occupancy fraction statistics
real/null/shuffled gap
diversity information
nearest-neighbor baseline
```

Do not publish one pooled headline score across these scenarios.

## 12. Legacy path

Do not delete the old Milestone-B implementation unless it is completely unused and documented as reproducible elsewhere.

Prefer:

```text legacy/reference path
+
new unified active path
```

Make configuration selection explicit.

The old generic 3-channel decoder, if present from the abandoned Phase-1 work, must not accidentally become the default unified decoder.

## 13. Repository hygiene

Check:

```text no raw dataset modifications
no external/metadit source edits
no generated checkpoints committed
no duplicate scalar masking
no duplicate surrogate
no stale active imports
no ambiguous default architecture
```

Run:

```bash
git status
git diff --stat
git diff --check
python -m pytest -q
```

## 14. Final report

Print:

```text final architecture summary
created files
modified files
tests run
test results
legacy path status
new path status
unresolved issues
```

Do not claim inverse-design success until the hard pure-inverse-design scenario has been evaluated against the required baselines.
