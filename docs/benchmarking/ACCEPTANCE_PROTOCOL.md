# Acceptance protocol (prospective)

This encodes the lesson that a single-seed number is not evidence. It applies
**prospectively** from 2026-10-03; historical single-seed results are retained as
preliminary measurements and are **not** rewritten.

## Training-seed rule
No phase-level model/training result is **established** until reproduced on **at least
3 independent training seeds**. A single training seed is allowed only as
**preliminary signal**, to decide whether a 3-seed confirmation run is worth the
compute.

## Evaluation-seed rule
Evaluation seeds are **separate** from training seeds and exist to quantify stochastic
*measurement* variability on a **fixed checkpoint**. They do **not** replace
training-seed replication. `scripts/eval/eval_scenarios.py --eval-seed E` reseeds the
mask draw and the shuffled control only (`E=0` reproduces the historical numbers);
items, split, `n_samples`, ratio, placement and gate are unchanged.

## Reporting rule
Every number states its replication unit explicitly:

```
0.XXX ± 0.XXX (N training seeds)
0.XXX ± 0.XXX (N evaluation seeds; fixed checkpoint)
```

Never mix the two, and never present a single-seed number as an improvement.

## Statistical rule
For small-N repeated-seed experiments report: mean, **sample** standard deviation, a
**95 % two-sided Student-t CI on the mean**, and the full per-seed values. Do **not**
treat individual samples inside one evaluation seed as independent seed-level
replicates.

## Historical-results rule
Old single-seed results stay as historical/preliminary measurements. This protocol
applies from now on and does not retroactively invalidate them — but no *new* claim
may rest on them.

## Provenance rule
Record, with every result: the exact checkpoint, split, sample count, mask ratio,
metric, protocol, commit, and (where relevant) the kernel/artifact that produced it.
Never silently change any of these.
