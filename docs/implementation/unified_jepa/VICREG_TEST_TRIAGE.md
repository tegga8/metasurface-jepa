# VICReg collapse-recovery test — triage note (Fix 3)

**Date:** 2026-08-27
**Test:** `tests/test_vicreg_collapse.py::test_projector_collapse_detected_and_recovered`
**Reported:** deterministic failure at `variance_loss(p_fixed) = 0.6705818176269531`
(ratio ≈ 0.685 vs the 0.6 threshold) under torch 2.5.1 and 2.13.0.

## Triage performed (per fix spec's decision tree)

| Step | Check | Result |
|---|---|---|
| 1 | Seed sweep (0–10), 600 Adam steps, lr=1e-2 | All 11 seeds PASS; ratio clusters at **0.576–0.589** (comfortably < 0.6). Not miscalibrated. |
| 2 | Seed sensitivity | None: every seed converges to the same fixed point `v_after = 0.566126`; only `v_before` varies with the input draw. |
| 3 | Genuine recovery defect? | No: variance restores from ~0.98 to 0.566 (ratio 0.578), projected eff-rank goes from <0.2 to 0.25. VICRegProjector recovers correctly. |

Additional checks:
- **Thread-count independent**: identical results at 1/2/4/12 `torch.set_num_threads()` on CPU.
- **Variance-only variant** (no covariance): converges to 0.0 (trivially passes) — the covariance term is not the blocker.
- **torch version**: passes on torch 2.13.0+cpu (the installed version).
- **Git history**: test is pre-existing, last touched by `e4cbe99` (an older VICReg fix), unrelated to the Phase 1–5 unified-JEPA changes.

## Conclusion

The reported failure is **not reproducible** in this environment. The 0.6×
threshold is correctly calibrated for the test's optimizer configuration
(observed ratio 0.578, with margin). No code change was made: per the fix
spec, the threshold must not be loosened without evidence, and step 3 (a real
projector defect) does not apply. If the failure reproduces on the reviewer's
machine, the next step is to capture the full reproduction (torch build,
CPU microarchitecture/BLAS backend, git tree state) rather than change the
test, since the same seed+optimizer+steps converge to the same fixed point
here across thread counts.

Full suite at time of writing: **477 passed, 12 skipped** (CUDA-only skips).
