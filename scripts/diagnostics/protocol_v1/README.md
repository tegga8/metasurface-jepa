# Diagnostic Protocol v1 — Unified JEPA (raw-vs-projected / physics-weighting)

Runbook for the diagnostic protocol that isolates the training/validation
anomalies WITHOUT architecture changes. Each script is standalone:
`python scripts/diagnostics/protocol_v1/<script>.py`

Execution order (do not reorder):

1. Baseline suite first: `pytest -q` must be green.
2. `step2_raw_vs_projected.py` — 500-step run; paired raw (z_hat/z_y) vs
   projected (p_hat/p_y) metrics on fixed in-training + held-out batches at
   steps {0, 50, 100, 200, 500}. Decision: projected improves while raw stays
   flat → projector absorption (run step 5).
3. `step3_4_geometry_and_gradients.py` — (a) decoder fed the EMA target latent
   with true/predicted scalars and a perturbed latent; (b) per-term gradient
   norms (structural-zero check for JEPA terms into the decoders) + gradient
   cosine between weighted L_phys and the JEPA terms.
4. `step5_projector_ablation.py` — ONLY if step 2 shows absorption.
   A=current, B=projector frozen at init, C=+ auxiliary raw-latent term
   (lambda_raw=1.0, diagnostic-only).
5. `step6_physics_normalization.py` — normalization audit (per-sample std,
   both sides, all three loss branches, gradient flow).
6. `step7_lambda_sweep.py` — lambda_phys in {0.01, 0.1, 1, 10}, short runs;
   decoder/physics gradient scaling, L_var/L_cov stability, occupancy
   collapse, real-vs-shuffled on the fixed real batch.
7. Hard-stratum necessity gate (step 8) uses the authoritative evaluator —
   requires a TRAINED checkpoint (not runnable from a fresh model):
   `python scripts/eval/eval_scenarios.py --config configs/unified.yaml --checkpoint <ckpt> --scenario all`
   Gate: real spectrum error must beat shuffled on Scenario A
   (mask=1.0, all scalars unknown). Never pool A with the easy reference.

Key finding recorded by this protocol (2026-08-29): `UnifiedJEPALoss.train()`
now pins the frozen surrogate to eval mode — `objective.train()` previously
flipped the surrogate's 38 BatchNorm2d layers into train mode, corrupting the
physics loss (batch-stat BN) and drifting running stats.

Results are written to stdout; run from the repo root (real data/weights are
required for steps 3-7; step 2 falls back to synthetic).

NOTE: these scripts are diagnostic tooling, not part of the training pipeline.
