# Results ledger (append-only)

One block per phase: the **before** (baseline) numbers and the **after** numbers,
same battery, same split / config / seed. Never overwrite a block — append.
Numbers come from `checkpoints/benchmark/*.json` + the eval-scenario output.

See `ROADMAP.md` §"Standard phase protocol" and `BASELINE.md`.

---

## Baseline — current architecture (Phase 1)

> **Corrections pending re-run (review A2/A4/A5).** The numbers below were produced
> by the *pre-fix* driver: the NN row is **not** like-for-like (512-item pool, first
> 512 test items vs the model's full 17,489) and the AVG1 row used the evaluated
> split's own mean (transductive), not the train mean. The driver has been fixed;
> this block must be **regenerated** before it is cited. The MAE/AAE/AAE&K headline
> and the A/B/C gates are unaffected by these fixes.

- date: 2026-10-02   commit: `5d228ac` (kernel) / `5d27bc5` (masked-fill fix)
- platform: Kaggle GPU kernel `tejaspbiradar/metasurface-jepa-baseline-phase-1`
- checkpoint: `full_epoch_final.pt` (`anosvol/metasurface-jepa-192d-full-epoch-ckpt`)
- artifacts: `checkpoints/benchmark/baseline_scenarioA.json`,
  `baseline_eval_scenarios.json`, `baseline_masked_fill.json`, `baseline_guidance_gap.json`

| quantity | value | notes |
|---|---|---|
| tests | 328 passed / 8 skipped / 1 failed | failure pre-existing & unrelated |
| MAE (ours, Scenario A, test, n=17,489) | **0.0673** | beats MetaDiT-S 0.0801 |
| AAE (ours) | **40.535** | beats 48.2495 |
| AAE&2 / AAE&4 | 42.14 / 43.95 | analogue; paper 58.80 / 68.73 |
| AVG1 MAE / AAE | 0.2574 / 154.966 | eval-split mean |
| NN MAE / AAE (n=512) | 0.0551 / 33.161 | **better than ours** |
| surrogate floor MAE | 0.0065 | |
| A / B / C win rate (n=512) | 0.9941 / 0.9668 / 0.8984 | gate ≥ 0.75 → PASS |
| scalar-dep one-known / two-known | 0.5059 / 0.4512 | **FAIL** |
| guidance gap (hard stratum) | 185.28 | large → not Failure Mode 2 |
| occupancy IoU/F1 (A, masked) | 0.7481 / 0.8559 | |
| pred occupancy fraction (A) | 0.4270 (true 0.4247); std 0.0673 | not collapsed |
| masked-fill filled IoU A/B/C | 0.7446 / 0.8053 / 0.8616 | n=32 |
| masked-fill visible_identity_maxdiff | 0.0 | only the hole is filled |
| masked-fill seam_mismatch A/B/C | n/a / 0.0684 / 0.0587 | |
| masked-fill locality_ratio A/B/C | n/a / 0.496 / 0.385 | partially localised |
| physics gradient share | not measured in this run | train-time; ~66 % per REPORT §13.1 |

**Gate:** baseline recorded; failing = scalar dependence; the rest are no-regress
constraints → proceed to Phase 2.

---

## Phase 2 — representation-first training schedule  ·  DONE

- date: 2026-10-02   commit: `ee23a24` (+ `16b3c52` logger fix)
- change: 2a live per-term gradient-share probe; **2b** per-sample mask ratios + a
  distribution ramp (start easy → target; P(full mask) 0.15 → 0.35); **2c** physics
  held at 0 for the first 2000 steps, then ramped over 3000.
- **Gate:** like-for-like **10k control vs Phase-2** (same length, same battery,
  Kaggle kernels `…-phase-2-train` / `…-phase-2-control`, T4):

| Scenario-A / gates | control (old) | **Phase-2 (new)** | delta |
|---|---|---|---|
| MAE (test, n=17,489) | 0.0792 | **0.0749** | −5.4 % |
| AAE | 47.675 | **45.116** | −5.4 % |
| MAE @ n=512 | 0.0967 | **0.0707** | −27 % |
| normalized-L1 (A) | 0.1378 | **0.1298** | −5.8 % |
| occupancy IoU / F1 (A, masked) | 0.7025 / 0.8244 | **0.7209 / 0.8372** | up |
| scalar MAE (unknown) | 0.1106 | **0.1030** | −6.9 % |
| A / B / C win rate | 0.982 / 0.910 / **0.736 (FAIL)** | **0.996 / 0.941 / 0.779 (PASS)** | **C crosses 0.75** |
| scalar-dep. one / two known | 0.617 / 0.553 | **0.660 / 0.623** | up (still < 0.75) |
| AAE&2 / AAE&4 | 48.21 / 48.81 | **45.84 / 46.76** | better |
| like-for-like NN (pool 20k, n=512) | 0.0296 | 0.0296 | **NN beats us ~2.4×** |

- **Gate MET** — Scenario-A MAE/AAE improves, the hard-stratum gate holds (0.996),
  and **Scenario C moves from FAIL to PASS**; scalar dependence rises ~+4–7 pp.
- Both 10k runs are worse than the 70k baseline (MAE 0.0673) — fewer steps, expected.
- Caveats: single seed; the guidance gap dropped (185 → ~41) but that is the
  **2-sample synthetic probe** (review A5) — re-measure on real data; **NN retrieval
  still beats us** on the same items (the honest headline).
- Next: Phase 3 (representation hygiene).

---

## Phase 3 — representation hygiene  ·  DONE

- date: 2026-10-02   commit: `b811da6`
- change: **3a** removed the redundant pre-patch-embed pixel mask (proven
  behaviour-neutral by the masked-pixel-invariance test — no run needed); **3b**
  made the objective projector ablable (`loss.projector_type`).
- **3b projector ablation** — 10k steps, same schedule, Kaggle
  `…-phase-3-projector` (mlp_bn row = the Phase-2 reference run at the same 10k):

| arm | MAE | AAE | norm-L1 (A) | A / B / C win rate | scalar 1 / 2 known |
|---|---|---|---|---|---|
| none | 0.0958 | 57.66 | 0.1653 | 0.965 / 0.908 / 0.828 | 0.469 / 0.545 |
| linear | 0.0849 | 51.11 | 0.1485 | 0.982 / 0.939 / 0.863 | 0.527 / 0.572 |
| mlp | 0.0864 | 52.00 | 0.1522 | 0.980 / 0.949 / 0.811 | 0.627 / — |
| **mlp_bn (shipped)** | **0.0749** | **45.12** | **0.1298** | **0.996** / 0.941 / 0.779 | 0.660 / 0.623 |

- **Decision: keep `mlp_bn`.** The shipped BatchNorm projector wins the decisive
  metrics (MAE/AAE ≈ −12 % vs linear, −22 % vs none; best Scenario-A gate and scalar
  dependence). The review's "BatchNorm can hide collapse" concern is **not supported**
  here — BN is empirically helpful. Nuance: the non-BN arms do better on Scenario C
  (retrofit 0.863 / 0.811 vs 0.779) — revisit only if C becomes the primary gate.
- Gate (hard stratum + MAE/AAE): **met** with the incumbent unchanged.
- Next: Phase 4 (multi-target objective).

---

## Phase 4 — multi-target objective  ·  ATTEMPTED — FAILED (gate not met)

- date: 2026-10-02   commit: `72195e6`   Kaggle `…-phase-4-multitarget`
- change: added the spectrum-conditioned geometry target (`z_y_occ_spec`), the
  scalar-latent target (`z_y_scal`), their predictions, and the losses
  `L_cond` / `L_scal_t` (λ = 1 each), keeping the stable spectrum-free target.
- **Result: the model COLLAPSED.** vs the Phase-2 control (same 10k schedule):

| metric | Phase-2 control | **Phase 4** |
|---|---|---|
| MAE (test, n=17,489) | 0.0792 | **0.3214** |
| normalized-L1 (A) | 0.1378 | **0.5546** |
| occupancy IoU / F1 (A, masked) | 0.7025 / 0.8244 | **0.020 / 0.039** |
| pred occupancy fraction | 0.45 | **0.012 (near-empty)** |
| A / B / C win rate | 0.982 / 0.910 / 0.736 | **0.512 / 0.512 / 0.266 (all FAIL)** |
| scalar-dep. one / two known | 0.617 / 0.553 | **0.242 / 0.246 (below chance)** |
| guidance gap (hard) | ~41 | ~208 |

- **Root cause (measured, not guessed):** `L_cond` took **91–99 % of the gradient
  budget from step 0** (live `[grad-share]`), starving every other term (<2 %).
  `L_cond` is an **un-normalized masked MSE over 192-D latents through a
  randomly-initialized `spec_proj` head**, and at step 0 the spectrum FiLM is
  identity so `z_y_occ_spec == z_y_raw` — i.e. it is a mis-specified regression
  toward the *stable* target that hijacks training.
- **Gate: NOT MET** — worse on every axis. **Work is halted here** (no loosening).
- **Next (diagnose first):** (i) normalize `L_cond` (cosine, or scale-matched to
  the VICReg terms) and/or λ_cond ≪ 1 with a ramp; (ii) break the init degeneracy —
  the spec target must differ from the stable target at step 0 (non-identity-init
  spectrum FiLM, or a detached fixed projection); (iii) one corrective 10k arm,
  checking the gradient share lands in a sane band **before** re-examining gates.

---

## Phase 5 — scalar capacity & bounds

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 6 — encoder sizing

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
