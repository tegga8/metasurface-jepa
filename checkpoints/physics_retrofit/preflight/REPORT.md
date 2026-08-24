# Retrofit Phase 1 — Preflight, Baseline Lock, Gate Setup — REPORT

Date: 2026-08-24. Code state: `main` @ `9774b7becc7c9ca3dd371d74d19adb81bc89ebf9`
(clean at provenance time; plumbing fixes for the 2026-08-24 re-scope landed after,
see §5). Full provenance: `BASELINE_PROVENANCE.md` (same directory).

## Verdict summary

| Phase-1 item | Status | Evidence |
|---|---|---|
| 1. Baseline provenance record | DONE | `BASELINE_PROVENANCE.md` (git SHA, config/weights/dataset SHA-256s, env, checkpoint-acceptance contract) |
| 2. Architecture invariants via existing tests | PASS | `python -m pytest tests -q` → **230 passed / 6 skipped** at provenance commit (EMA frozen + optimizer-excluded, EMA/student init equality, no model.proj / shared projection, mask alignment + leakage guards, physics conditioning path, strict checkpoint provenance) |
| 4. Gate-0 scalar-vs-shape audit | **PASS** | `gate0_scalar_vs_shape.json` — see §2 |
| 3. Target-representation health validation | **BLOCKED** | requires a genuine trained JPA checkpoint; none exists on this machine (§3) |
| 5. Gate-1 physics utility (real/null/shuffled × 3–5 seeds) | **BLOCKED** | same blocker (§3) |
| 6. Gate-2 ambiguity audit | **DEFERRED** | candidate-freezing deferred by operator re-scope (§4); representation-separation half additionally checkpoint-blocked |

## 1. Checkpoint situation (gating fact)

No genuine trained JPA checkpoint exists locally — only released MetaDiT weights and
a diagnostic linear probe. The only genuine cloud-trained candidate was retired as
collapsed (`checkpoints/milestone_b/REPORT.md`). Consequently every
checkpoint-gated item above stays blocked until a HEALTHY §30 checkpoint from a
genuine Milestone-B run is synced back per `CLOUD_TRAINING.md`. This preflight did
NOT train anything and did NOT touch any architecture/objective file.

## 2. Gate-0 — does masked-region SHAPE carry spectral information beyond scalars?

Script: `scripts/diagnostics/gate0_scalar_vs_shape.py` (+ unit tests
`tests/test_gate0_scalar_vs_shape.py`). Arms on identical held-out val subsets
(train n=16384, val n=4096, seed 0), metric rel-L2 on stacked (real, imag):

| Arm | rel-L2 | R² |
|---|---|---|
| mean_spectrum (floor) | 0.5474 | ≈0 |
| scalar_knn (1-NN in standardized scalars) | 0.4909 | 0.046 |
| scalar_mlp (6→256→256→602, 60 ep) | 0.3536 | 0.527 |
| **surrogate_shape (frozen ConvSurrogate, full 3×64×64)** | **0.0533** | **0.620** |

Proposed verdict rule (operator-confirmable per Standing Rule 3): shape is
informative iff surrogate rel-L2 < 0.8 × best scalar arm. Observed ratio **0.151**
→ **shape_informative = TRUE** by ~6.6× margin. The swapped-target retrofit route
has real physics signal to exploit; no stop condition fires.

## 3. Blocked items — exact remedy

Run genuine Milestone-B training on cloud GPU per `CLOUD_TRAINING.md`, sync the
winning HEALTHY §30 checkpoint into `checkpoints/milestone_b/` (acceptance contract
in `BASELINE_PROVENANCE.md` §6), then execute items 3 and 5 and the
representation-separation half of item 6.

## 4. Operator re-scope recorded (Standing Rule 9)

On 2026-08-24 the operator replaced the planned "Phase 2 (Geometry Decoder)"
retrofit phase with "Phase 2 — Fix VICReg Training/Validation Plumbing Before Cloud
Run", deferring Gate-2 candidate-freezing and forbidding decoder/inverse-design/
swapped-spectrum work until the plumbing-gated cloud training decision resolves.
The dated entry lives in `AGENTS.md`; the executed plumbing phase is reported in
`checkpoints/milestone_b/PHASE2_VICREG_PLUMBING_REPORT.md`.

## 5. Post-provenance code changes in this session

Diagnostics/tests only plus the operator-directed plumbing fixes:
- added `scripts/diagnostics/gate0_scalar_vs_shape.py`, `tests/test_gate0_scalar_vs_shape.py`,
  artifact `gate0_scalar_vs_shape.json`;
- Phase-2 plumbing (see PHASE2 report): `scripts/train/train_milestone_b.py`,
  `src/train/engine.py`, `scripts/eval/eval_vicreg_sanity.py`,
  `tests/test_vicreg_plumbing_mode_device.py`.
No encoder/predictor/objective-math/checkpoint-loading changes. Suite after all
changes: **238 passed / 7 skipped**.
