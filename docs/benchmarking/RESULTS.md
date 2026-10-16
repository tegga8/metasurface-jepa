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

### Phase 4 — corrected run (commit `904856e`) — PARTIAL

- Fix (`docs/benchmarking/PHASE4_DIAGNOSIS.md`): cosine (scale-free) losses, ramped
  in from step 1000, and a **frozen non-identity** spectrum film. No collapse.

| metric | Phase-2 control | **Phase 4 (fixed)** |
|---|---|---|
| MAE (test, n=17,489) | 0.0792 | **0.0789** |
| normalized-L1 (A) | 0.1378 | 0.1369 |
| occupancy IoU / F1 (A) | 0.7025 / 0.8244 | 0.708 / 0.8284 |
| A / B / C win rate | 0.982 / 0.910 / **0.736** | 0.988 / 0.934 / **0.771 (PASS)** |
| scalar-dep. one / two known | 0.617 / 0.553 | **0.641 / 0.615** |
| guidance gap (hard stratum) | ~41 | **~27 (fell)** |
| `L_cond` / `L_scal_t` grad share | — | **~0.1–0.2 % (inert)** |

- **No collapse and no regression**; A/B/C all pass (C 0.736 → 0.771), scalar
  dependence +2–6 pp. **The fix removed the failure.**
- **But the gate is NOT met:** the hard-stratum guidance gap **fell** (~41 → 27)
  instead of rising, the scalar gates are still far below 0.94–0.97, and the
  multi-target terms are now **nearly inert** (~0.1–0.2 % share) — the fix
  over-corrected from 91–99 % to almost nothing, so the *forcing* mechanism barely
  acts.
- **Verdict: PARTIAL** — the collapse is fixed; the mechanism's intended effect is
  not demonstrated at this weight.
- **Next:** raise `λ_cond`/`λ_scal_t` toward a sane band (~2–5 % share, comparable to
  `L_occ`/`L_summary`) and/or strengthen the film's init, then one 10k arm — the
  `[grad-share]` probe says when the weight is right, before gates are re-examined.

### Phase 4 — target-separation probe (Task 1 of the target-information check)

- Local, no training: `scripts/diagnostics/spectrum_film_separation.py` probes the
  frozen film's init (`--film-std` overrides it). Cross-spectrum = cosine distance of
  `z_y_occ_spec` for the SAME occupancy under 4 real spectra (the batch's own + 3
  derangements via `make_shuffled_spectrum`), token-averaged like `L_cond`.

| film init std | cross-spectrum cos-dist (mean / med) | spec-vs-raw | noise floor |
|---|---|---|---|
| **0.02 (was)** | **0.037 / 0.044** | 0.020 | ~1e-8 |
| 0.1 | **0.491 / 0.484** | 0.336 | ~1e-8 |
| 0.2 | 0.805 / 0.892 | 0.673 | ~1e-8 |

- **Verdict: hypothesis CONFIRMED.** At the shipped init the target is ~96 % identical
  across spectra (cos-dist 0.037) — `L_cond` carries almost no spectrum-specific signal,
  so raising λ alone would amplify a near-redundant direction. The root cause is one
  level deeper than loss weighting.
- **Action (Task 2):** raise the frozen film's init `std` **0.02 → 0.1** (13× separation;
  0.2 rejected as "arbitrary transform dominates"). Cosine + ramp + frozen **unchanged**.
- **Sane-band check:** with std 0.1 and the term fully on (ramp=0 smoke), `[grad-share]`
  is **`L_cond` 1.1 %, `L_scal_t` 1.6 %** — in band, not inert, not dominant.
- **Next (Task 3):** one 10k arm with std 0.1 (ramp 1000→3000), re-check the hard-stratum
  guidance gap and A/B/C gates. λ retuned only as a secondary adjustment.

### Phase 4 — Task 3: film init std 0.1 (the fix) — PASS (partial on scalar)

- Change: raise the frozen `SpectrumFilm` init `std` 0.02 → 0.1 (cosine / ramp / frozen
  unchanged). `[grad-share]` stays sane: `L_cond` ~0.2–0.5 %, `L_scal_t` ~0.1–0.3 %.
- 10k arm vs the Phase-2 control and the std-0.02 attempt:

| metric | Phase-2 control | std 0.02 | **std 0.1** |
|---|---|---|---|
| MAE (test, n=17,489) | 0.0792 | 0.0789 | **0.0725** |
| AAE | 47.68 | 47.48 | **43.62** |
| normalized-L1 (A) | 0.1378 | 0.1369 | **0.1263** |
| occupancy IoU / F1 (A) | 0.7025 / 0.8244 | 0.708 / 0.8284 | **0.7099 / 0.8297** |
| A / B / C gate | 0.982 / 0.910 / 0.736 | 0.988 / 0.934 / 0.771 | **0.994 / 0.947 / 0.779** |
| scalar-dep. one / two | 0.617 / 0.553 | 0.641 / 0.615 | **0.650 / 0.596** |
| guidance gap (hard stratum) | ~41 | ~27 | **~52** |

- **Gate: PASS on the key conditions.** MAE/AAE **improve** (0.0725 / 43.62 — the best
  10k result), the hard-stratum **guidance gap rises** (41 → 52; the recorded failure of
  the std-0.02 attempt is fixed), and all A/B/C gates pass and rise. **Scalar
  dependence improves** (one-known 0.617 → 0.650) but stays **short of the 0.94–0.97
  target** → partial on that one condition.
- **Root cause resolved:** the target-information hypothesis was **correct** — the
  spectrum-conditioned target was near-redundant at the shipped init, and strengthening
  the conditioning (not the loss weight) was the fix.
- Caveats: single seed; `L_cond` share is still modest (~0.2–0.5 %); **NN retrieval
  (0.0296) untouched**; still 10k (70k baseline 0.0673).
- **Next:** the mechanism works. λ can be nudged for a stronger scalar effect, but the
  dominant open item remains the NN-retrieval gap.

### NN-scoping probe — framing decision

- Local, no training: `scripts/diagnostics/nn_scoping.py`, Phase-4 std-0.1 checkpoint,
  n=512 test items, train pool 20k, frozen surrogate, MAE units.

| regime | ours | NN | verdict |
|---|---|---|---|
| Scenario A (full S→G), overall | 0.0733 | 0.0303 | **NN wins** (ours beats on 2.9 %) |
| A, novelty q0 (closest, dist 0.0059) | 0.0318 | 0.0061 | NN 5.2× |
| A, q1 (0.0173) | 0.0497 | 0.0170 | NN 2.9× |
| A, q2 (0.0339) | 0.0819 | 0.0335 | NN 2.4× |
| A, q3 (furthest, dist 0.0657) | 0.1297 | 0.0646 | **NN 2.0×** |
| Scenario C (retrofit, 25 % mask) | 0.0741 | 0.0751 | **tie** (ours beats on 52 %) |

- **Verdict: "beat NN" is the required path; "scope beyond NN" is NOT supported.** NN
  wins in **every** novelty quartile — its edge shrinks monotonically (5.2× → 2.0×) as
  novelty rises but **never flips** — and the partial-observation regime (C) is a
  **tie**, not a win.
- **Structural finding:** NN's scored error ≈ its retrieval distance (q3: NN 0.0646 ≈
  dist 0.0657), i.e. the dataset is dense (median retrieval distance 0.0247) and the
  surrogate is near-exact — so "retrieve the spectrum-nearest training design" is a very
  strong baseline, and our model loses ground as novelty rises *faster* than NN does.
- **Decision:** treat retrieval as **the bar to beat**; Phases 5–7 become
  retrieval-competitive. Two mildly encouraging data points: the gap narrows with
  novelty, and C (design mostly given) is already a tie — i.e. the model is competitive
  in partial-observation regimes, just not better.

### Multi-seed CI (3 × 10k, current best config)

- Kaggle `metasurface-jepa-multiseed-ci`; seeds 0/1/2; benchmark (n=17,489 test) +
  scenario battery per seed.

| metric | seed0 | seed1 | seed2 | **mean ± std** |
|---|---|---|---|---|
| MAE (test, n=17,489) | 0.0800 | 0.0803 | 0.0845 | **0.0816 ± 0.0025** |
| AAE | 48.16 | 48.35 | 50.85 | 49.12 ± 1.50 |
| Scenario A gate | 0.9941 | 0.9902 | 0.9844 | **0.990 ± 0.005** |
| Scenario B gate | 0.9355 | 0.9316 | 0.9121 | **0.926 ± 0.012** |
| Scenario C gate | 0.7520 | 0.7832 | 0.7090 | **0.748 ± 0.037 (< 0.75)** |
| scalar-dep. one / two | 0.598 / 0.514 | 0.631 / 0.586 | 0.613 / 0.572 | 0.614 / 0.557 |

- **The single-seed phase gains were seed noise.** The 3-seed MAE mean (0.0816)
  sits **above** both the Phase-2 single run (0.0749) and the Phase-4 std-0.1 run
  (0.0725) — those were lucky draws. The honest 10k MAE for the current config is
  **0.0816 ± 0.0025, i.e. at the Phase-2 *control*'s level (0.0792)** →
  **no established improvement from Phases 2/3/4.**
- **Scenario C does not reliably pass:** mean 0.748 < 0.75 and one seed (0.709)
  **fails**. The earlier "C crosses the gate" was a single-seed artifact.
- Scenario A/B pass robustly (0.990 / 0.926); scalar dependence still fails
  (0.614 / 0.557).
- Seed std ≈ **2.5 %** of the mean — the same order as the effects that were being
  chased, which is exactly why single-seed comparisons were misleading.
- **Decision:** the earlier Phase-2/Phase-4 "improvements" are **retracted** as
  unestablished; the only results that survive the CIs are the robust gates (A/B),
  the NN loss, and the scalar failure.

### Scenario C — evaluation-seed precision (A3/A4) + tail diagnosis (A5)

- A1 added `--eval-seed` (measurement-only; `E=0` reproduces the historical numbers).
  Fixed checkpoint = the **seed-0 checkpoint of the 3-seed CI** (`ms/seed0.pt`).
  Protocol unchanged: val split, n=512, ratio 0.25, random placement, all scalars
  known, frozen model+surrogate.
- **A4 — 5 evaluation seeds {3,4,5,6,7} on the fixed checkpoint** (headline
  `scenario_C_rns.gap.real_beats_shuffled_fraction`): 0.75195, 0.74414, 0.74805,
  0.73828, 0.75195 → **mean 0.7469 ± 0.0058 (5 evaluation seeds)**; 95 % Student-t CI
  (df=4) **[0.7397, 0.7541] → straddles 0.75 → UNRESOLVED.**
  Evaluation-seed std (0.006) ≪ training-seed std (0.037): C's uncertainty is
  **training-seed variance, not measurement noise.**
- **A5 — per-sample tail** (pooled 2,560 samples = 5 × 512): real median 0.096 / mean
  0.128; shuffled median 0.175 / mean 0.233; paired-diff median 0.054; pooled
  `real<shuffled` fraction 0.7469. **~25 % of samples fail** (`fail_diff_mean` −0.061
  vs `win_diff_mean` +0.162). Masked IoU stable 0.742–0.759; no collapse (pred occ
  frac 0.443 vs true 0.425). → the shortfall is **broad (~25 %), not a narrow tail**.
- **Decision (per the plan's rule): Case 3 — unresolved.** A5 done; **do not add more
  seeds** (eval noise already shown small); **B/C not triggered** (C is not
  *established* below 0.75). The crossed variance design (3 ckpts × 3 eval seeds) is
  available but would formalise a conclusion the two stds already give.

### Scenario C — crossed seed study (3 training checkpoints × 3 eval seeds)

- Measurement-only (no training). Checkpoints = the 3 CI checkpoints
  (`ms/seed{0,1,2}.pt`, kernel `metasurface-jepa-multiseed-ci`); eval seeds {3,4,5}
  via the A1 control. Kaggle `metasurface-jepa-ms-crossed`, commit `3a69527`.
  Protocol unchanged (val split, n=512, ratio 0.25, random placement, all scalars
  known, frozen model+surrogate).
- **All 9 raw gates** (`scenario_C_rns.gap.real_beats_shuffled_fraction`):

| ckpt (train seed) | e3 | e4 | e5 | mean ± std (3 eval seeds) |
|---|---|---|---|---|
| seed0 | 0.75195 | 0.74414 | 0.74805 | **0.7480 ± 0.0039** |
| seed1 | 0.77930 | 0.79102 | 0.78125 | **0.7839 ± 0.0063** |
| seed2 | 0.75781 | 0.73633 | 0.77734 | **0.7572 ± 0.0205** |

- **Overall (replication unit = checkpoint, n=3 training seeds): 0.7630 ± 0.0186;
  95 % Student-t CI (df=2) = [0.7168, 0.8093] → straddles 0.75 → UNRESOLVED (Case 3).**
- Averaging 3 eval seeds per checkpoint **reduced** the training-seed std (0.0186 vs
  the 0.037 from single-eval-seed per-checkpoint values) — part of what looked like
  training-seed variance at one eval seed was evaluation noise — but the CI still
  straddles.
- **A5 carried through** (per checkpoint, pooled over its 3 eval seeds; 1,536 samples):
  real median ≈ 0.085–0.090 / mean ≈ 0.115–0.122; shuffled median ≈ 0.165–0.170 /
  mean ≈ 0.227–0.235; failing fraction 0.748 / 0.784 / 0.757; `fail_diff_mean`
  ≈ −0.048…−0.051, `win_diff_mean` ≈ +0.154…+0.167; masked IoU ≈ 0.740–0.752;
  occupancy fraction ≈ 0.443–0.449 (no collapse).
- **Sample-aligned failure consistency** (512 fixed items; per-item mean paired diff
  across the 3 eval seeds, counted across checkpoints): fail in **0 / 1 / 2 / 3**
  checkpoints = **427 / 51 / 16 / 18** items; failing in ≥1 = 0.166, ≥2 = 0.066,
  **all 3 = 0.035**. → the failure population is **mostly checkpoint-specific**: only
  **3.5 %** of items fail for all three checkpoints, so the ~16.6 % that fail at all
  are largely **training-seed-specific failure allocation**, not a common structural
  set (a small ~3.5 % structural core fails everywhere).
- **Decision:** Case 3 — **unresolved even after explicitly measuring both training-seed
  and evaluation-seed variance** → the 0.75 gate sits at the model's performance
  boundary. Per the plan: do not add seeds; do not call C a pass/fail from any single
  checkpoint. No targeted C mechanism is justified by this evidence alone.

### Full-epoch (70k) baseline — CURRENT config, 3 training seeds  ·  DONE

- Kaggle `metasurface-jepa-fe-current`, commit `c35196f`; **current config unchanged
  except `train.seed ∈ {0,1,2}`**; `--max-steps 70000` (≈1 epoch, batch 2). Existing
  benchmark protocol. Replication unit = **training seed**. All stage exit codes 0.

| metric | seed0 | seed1 | seed2 | **mean ± std (3 training seeds)** | 95 % t-CI |
|---|---|---|---|---|---|
| MAE (test, n=17,489) | 0.0500 | 0.0494 | 0.0476 | **0.0490 ± 0.0013** | [0.0459, 0.0521] |
| AAE | 30.09 | 29.76 | 28.68 | **29.51 ± 0.74** | [27.68, 31.34] |
| AAE&2 / AAE&4 | 30.83 / 31.55 | 30.42 / 31.14 | 29.17 / 29.68 | 30.14 / 30.79 | — |
| Scenario A | 0.9961 | 0.9961 | 0.9980 | **0.9967 ± 0.0011** | [0.9940, 0.9994] |
| Scenario B | 0.9805 | 0.9727 | 0.9785 | **0.9772 ± 0.0040** | [0.9673, 0.9871] |
| Scenario C | 0.9375 | 0.9355 | 0.9180 | **0.9303 ± 0.0110** | [0.9030, 0.9576] |
| scalar one / two | 0.508 / 0.564 | 0.529 / 0.535 | 0.477 / 0.486 | **0.505 / 0.529** | straddles ~0.5 |

- **Verdict: 70k MATERIALLY improves the 10k 3-seed baseline on fidelity + gates.**
  MAE 0.0816 → **0.0490** (−40 %), AAE 49.1 → 29.5, A 0.990 → 0.9967, B 0.926 → 0.9772,
  and **Scenario C 0.748 → 0.930 — now robustly ABOVE the 0.75 gate (CI [0.903, 0.958])**.
  This also beats the earlier *old-config* single-seed 70k (MAE 0.0673) and MetaDiT-S
  (0.0801). (NN retrieval, 0.0296, still leads — now ~1.65×.)
- **But the scalar path is unchanged: still ~chance (0.505 / 0.529)** — more training
  did **not** fix scalar conditioning. This is the surviving, established weakness.
- **Per the plan: STOP here** — no scalar intervention in this pass.

### Step 3 — scalar-predictor FiLM at 70k × 3 seeds  ·  NEGATIVE (stop the scalar line)

- Intervention arm only (baseline = `fe_current`). `scalar_predictor_film = true`;
  70k × seeds {0,1,2}. Training kernel `metasurface-jepa-scalar-film`; battery re-run
  by `metasurface-jepa-sf-eval` (the first battery used the flag-off config against
  film-enabled checkpoints → strict-load failure; corrected run shown).
- Replication unit = **training seed**; per `ACCEPTANCE_PROTOCOL.md`.

| metric | baseline (`fe_current`) | **intervention** |
|---|---|---|
| MAE (test, n=17,489) | 0.0490 ± 0.0013 | **0.0506 ± 0.0028** (0.0477 / 0.0533 / 0.0507) |
| AAE | 29.51 ± 0.74 | **30.44** (28.72 / 32.08 / 30.52) |
| Scenario A | 0.9967 ± 0.0011 | **0.9941** (0.9961 / 0.9883 / 0.9980) |
| Scenario B | 0.9772 ± 0.0040 | **0.9746** (0.9766 / 0.9707 / 0.9766) |
| Scenario C | 0.9303 ± 0.0110 | **0.9362** (0.9277 / 0.9375 / 0.9434) |
| **scalar one / two** | 0.505 / 0.529 | **0.525 / 0.555** |

- **Verdict: NEGATIVE.** Scalar dependence does **not** move materially off chance —
  one-known ~0.52 (CI straddles 0.5), two-known ~0.55 — and MAE/A/B/C are within noise
  of the baseline (MAE slightly worse, not better). The predictor-FiLM route does not
  fix scalar conditioning.
- **Per the plan's stop condition → STOP the scalar architecture line** and reassess
  the underlying task/objective; do not stack more losses/modules.

### Controlled ablation — JEPA vs conventional objective (70k × 3 seeds)  ·  Outcome A

- The **only** change vs the established baseline is `objective=conventional`
  (full-occupancy BCE + scalar + frozen-surrogate physics under the same ramp; no
  VICReg / EMA-target / masked-token / latent terms). All three train logs confirm
  `OBJECTIVE MODE: CONVENTIONAL`. Kernel `metasurface-jepa-objective-ablation` v2,
  commit `2218d70`. Replication unit = **training seed**.

| metric | JEPA baseline | Conventional | direction |
|---|---|---|---|
| MAE (test, n=17,489) | **0.0490 ± 0.0013** | **0.0922 ± 0.0195** | JEPA ~1.9× better |
| AAE | 29.51 ± 0.74 | 55.49 ± 11.78 | JEPA better |
| Scenario A | **0.9967 ± 0.0011** | 0.9792 | JEPA better |
| Scenario B | **0.9772 ± 0.0040** | 0.9460 | JEPA better |
| Scenario C | **0.9303 ± 0.0110** | 0.8288 | JEPA better |
| scalar one / two | 0.505 / 0.529 | 0.4955 / 0.5013 | ~chance in both |
| occupancy IoU (masked) | ~0.75 | ~0.64 | JEPA better |

- Per-seed conventional: MAE **0.0914 / 0.0730 / 0.1121** (A 0.9668 / 0.9844 / 0.9863;
  B 0.9082 / 0.9688 / 0.9609; C 0.7676 / 0.8672 / 0.8516; scalar one 0.5098 / 0.4883 /
  0.4883; two 0.5195 / 0.4512 / 0.5332).
- **Outcome A — the JEPA objective is materially better.** **Every** conventional seed
  (MAE 0.0730–0.1121) is worse than **every** JEPA seed (0.0476–0.0500): complete
  per-seed separation, ~1.9× worse on the mean; the fidelty gates and occupancy IoU are
  materially lower too. Scalar dependence is ~chance in both arms (the ablation does not
  fix it).
- Caveat (honest): the conventional MAE 95 % t-CI (df=2) is wide — **[0.0437, 0.1406]** —
  and technically overlaps the JEPA CI **[0.0459, 0.0521]** because of the conventional
  arm's high variance. The 3-vs-3 per-seed separation and the large mean gap carry the
  conclusion; do not over-read the CI overlap.
- **Claim:** the JEPA latent objective contributes materially beyond conventional
  supervised training of the same architecture on this benchmark.

---

## Phase 5 — scalar capacity & bounds

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 6 — encoder sizing

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

### Phase 6 — split-ratio re-splits (60:20:20 / 50:25:25 / 40:40:20) · DONE

- date: 2026-10-05   kernels: `anosvol/metasurface-jepa-split-{60-20-20,50-25-25,40-40-20}`
  (Kaggle T4; run account `anosvol`, read-only pulls)   pin: `fda7236`
- change: **data-allocation study** (operator override 2026-10-04, `AGENTS.md`) — the released
  pool (174,883) re-split at three ratios with **one permutation** (perm seed 42, nested
  prefixes; `scripts/data/make_split_ratio.py`). Total data unchanged; only the allocation
  moves. 1 epoch per arm (`steps = train // 2` @ batch 2), seeds {0,1,2}. **All stage exit
  codes 0 on every rung × seed** (train / scenario-A / common-slice / eval battery).
- evaluation: MetaDiT-parity scenario-A benchmark (4 candidates) on the **shared 20 % test
  slice (n=34,976)** for cross-arm comparability; A/B/C gate battery (n=512) on each arm's
  own val split. **Not MetaDiT-protocol comparable** (allocation shifted by design).
- reference row (unchanged, from `fe_current` above): 80:10:10 · 139,906 train / 70k steps ·
  MAE 0.0490 ± 0.0013 · A/B/C 0.9967 / 0.9772 / 0.9303.

| arm | train / steps | **MAE** (mean ± std, 3 seeds) | per-seed | A / B / C gate (mean) |
|---|---|---|---|---|
| **80:10:10** (reference) | 139,906 / 70,000 | **0.0490 ± 0.0013** | 0.0500 / 0.0494 / 0.0476 | 0.997 / 0.977 / 0.930 |
| **60:20:20** | 104,931 / 52,465 | **0.0544 ± 0.0013** | 0.0539 / 0.0536 / 0.0559 | 0.994 / 0.973 / 0.915 |
| **50:25:25** | 87,443 / 43,721 | **0.0552 ± 0.0018** | 0.0532 / 0.0559 / 0.0566 | 0.991 / 0.973 / 0.898 |
| **40:40:20** | 69,954 / 34,977 | **0.0619 ± 0.0006** | 0.0617 / 0.0615 / 0.0626 | 0.992 / 0.965 / 0.884 |

- 50:25:25 note: this arm's own test split is the 25 % slice (superset of the common 20 %);
  it was scored on the common slice as well — own-slice MAE **0.0551 ± 0.0018** vs common
  0.0552 → no slice artifact.

| arm | AAE | AAE&2 / AAE&4 | IoU / F1 (masked) | pred occ frac (true) | scalar MAE (unknown) | scalar 1 / 2 known | floor |
|---|---|---|---|---|---|---|---|
| 60:20:20 | 32.78 | 33.44 / 34.14 | 0.744 / 0.853 | 0.424 (0.415) | 0.089 | 0.509 / 0.538 | 0.00709 |
| 50:25:25 | 33.25 | 33.87 / 34.57 | 0.741 / 0.850 | 0.424 (0.415) | 0.091 | 0.531 / 0.542 | 0.00701 |
| 40:40:20 | 37.29 | 37.87 / 38.54 | 0.739 / 0.849 | 0.425 (0.422) | 0.096 | 0.592 / 0.574 | 0.00709 |

- **Reading:** the data axis is monotone but gentle above ~60 % train — MAE +11 % (60 %),
  +13 % (50 %), +26 % (40 %) vs the 80:10:10 reference; the 60→50 step is within noise of
  each other (0.0544 vs 0.0552, overlapping seed spread).
- **Gate C degrades smoothly with less data (0.930 → 0.915 → 0.898 → 0.884) but passes
  comfortably (≥ 0.75) at every rung and every seed.** A/B pass at every rung.
- **Scalar dependence stays ~chance on every rung (0.51–0.59)** — the known open weakness,
  unchanged by data allocation.
- No occupancy collapse on any rung (pred occ fraction ≈ true; all-empty / all-occupied both
  False). Every rung still beats MetaDiT-S (0.0801) and vanilla DiT (0.1677) on the common
  slice (reference context; not protocol-matched).
- artifacts: per-seed JSONs, logs, and checkpoints in the kernel outputs
  (`kaggle kernels output anosvol/metasurface-jepa-split-<arm>`).

---

## Phase 6b — S0 rung · length-proportional staging arm · symbol round-trip (2026-10-06)

All arms below: `akashkesav` kernels, 70k × seeds {0,1,2}, canonical battery, v1 schedule
unless stated. Trainable counts are live-audited (`model_size_audit`).

### Model-size ladder (3 seeds each)

| rung | trainable | MAE (mean ± std) | per-seed | A / B / C (mean) |
|---|---|---|---|---|
| **S0 tiny** (19.2 % of base) | 2,193,284 | **0.0564 ± 0.0003** | 0.0563 / 0.0562 / 0.0568 | 0.997 / 0.977 / 0.919 |
| **S1 small** | 4,232,836 | **0.0542 ± 0.0028** | 0.0543 / 0.0576 / 0.0507 | 0.992 / 0.979 / 0.917 |
| **S2 slim** | 8,142,980 | **0.0518 ± 0.0009** | 0.0516 / 0.0507 / 0.0530 | 0.995 / 0.976 / 0.920 |
| **base (v1)** | 11,402,628 | **0.0490 ± 0.0013** | 0.0500 / 0.0494 / 0.0476 | 0.997 / 0.977 / 0.930 |
| **L1 wide** | 19,896,708 | **0.0531 ± 0.0036** | 0.0513 / 0.0583 / 0.0498 | 0.996 / 0.975 / 0.920 |

- **Reading: the base is the accuracy peak of the ladder.** Shrinking costs MAE
  (+5.7 % S2, +10.6 % S1, +15.1 % S0); growing costs +8.4 % (L1). Occupancy IoU ≈ 0.75
  at every rung (no collapse); scalar dependence ~chance at every rung (the standing
  weakness, size-independent).

### Length-proportional (v2) staging arm — `sched_fix` — gate FAILED, not adopted

| arm | MAE (mean ± std) | per-seed | A / B / C |
|---|---|---|---|
| base v1 (absolutes 2000/3000) | **0.0490 ± 0.0013** | 0.0500 / 0.0494 / 0.0476 | 0.997 / 0.977 / 0.930 |
| base v2 (fracs 0.20 / 0.30, mask 0.30) | **0.0531 ± 0.0019** | 0.0507 / 0.0545 / 0.0530 | 0.996 / 0.970 / 0.905 |

- **Verdict: regression.** Every v2 seed is worse than every v1 seed (disjoint ranges):
  MAE +8.4 %, gate C −2.5 pp. The gate is not loosened — the v2 schedule is **not
  adopted**; the default reverts to the v1 absolutes (`resolve_staging_steps` keeps the
  mechanism for future retunes).
- Reading: MAE *is* the spectrum (physics) error. At 70k steps, holding physics off for
  the first 20 % and full only at the 50 % mark buys less total physics exposure than the
  compressed v1 window (physics full from ~7 % of training) — the proportional-staging
  argument did not survive measurement.

### Per-term gradient share at the trained state (6 val batches, each term alone)

| model | L_inv | L_var | L_cov | JEPA block | L_phys | L_occ | L_scalar | L_cond |
|---|---|---|---|---|---|---|---|---|
| S1 (v1) | 10.4 | 30.6 | 43.2 | **84.2** | 7.7 | 2.1 | 1.9 | 0.8 |
| S2 (v1) | 17.1 | 39.7 | 29.2 | **86.0** | 6.5 | 1.6 | 2.1 | 1.6 |
| L1 (v1) | 18.7 | 35.1 | 33.4 | **87.2** | 5.5 | 1.8 | 2.5 | 0.5 |
| base (v2) | 9.0 | 31.8 | 41.0 | **81.8** | 9.5 | 1.8 | 3.0 | 0.7 |

- The JEPA latent block carries ≈82–87 % of the gradient budget at the trained state on
  every rung; the surrogate-physics term holds ≈6–10 % (below the 10–25 % design band),
  consistent with the scalar-dependence weakness. The historical "base = 66 % physics"
  row was an old-code artifact (recorded in AUDIT_REPORT_192D.md).

### Symbol round-trip — 9 glyphs, Meep-fed, scenario A (full mask, all scalars unknown)

| model | mean IoU | per-symbol best / worst | control (val, CST-fed) |
|---|---|---|---|
| S2 slim | **0.304 ± 0.109** | torii 0.510 / faravahar 0.220 | 0.776 |
| S1 small | **0.306 ± 0.106** | torii 0.508 / faravahar 0.149 | 0.771 |
| L1 wide | **0.310 ± 0.105** | torii 0.488 / faravahar 0.183 | 0.782 |
| base (v2, seed0) | **0.311 ± 0.110** | torii 0.497 / faravahar 0.173 | 0.782 |

- The symbols (swastika, cross, crescent, aum, khanda, dharmachakra, yin-yang, torii,
  faravahar) sit far outside the dataset's blob morphology; recovery from their full-wave
  Meep spectra is partial at best — ~0.30 IoU vs ~0.78 in-distribution control, uniform
  across model sizes. Thin glyph-scale features are not uniquely pinned by the spectrum:
  several symbols show low surrogate spectrum error at low IoU (the model finds *a*
  design that explains the spectrum, not *the* glyph).
- **Visual verdict (5-row grid, all four models):** the decodes are dataset-style blobs
  (bowties / two-lobe masses), not glyphs — every OOD spectrum is mapped back onto the
  learned design manifold. The higher-IoU symbols (torii, swastika, chakram) are simply
  those whose gross silhouette is closest to that manifold (H / double-bar families).
  The full-size base behaves identically to the small rungs → not a capacity limitation.
- artifacts: `symbol_roundtrip_preview_with_base.png`,
  `symbol_roundtrip_preview.png`, `symbol_designs.npz` / `_preview.png`,
  `kaggle/symbol_designs` + `kaggle/symbol_meep` kernel outputs.

### In-distribution round-trip reference — MetaDiT's own released test split (n = 200, true CST spectra)

Same pipeline as the symbol round-trip (scenario A: full mask, all scalars unknown),
fed the dataset's own released spectra; occupancy-stratified 200-item slice of
`test_set.mat`.

| model | IoU (n=200) | F1 | surrogate spec err |
|---|---|---|---|
| S2 slim | **0.777 ± 0.079** | 0.872 | 0.0896 |
| S1 small | **0.776 ± 0.078** | 0.872 | 0.0891 |
| L1 wide | **0.777 ± 0.080** | 0.872 | 0.0917 |
| base (v2, seed0) | **0.781 ± 0.079** | 0.875 | 0.0868 |

- The 0.78-vs-0.31 gap isolates the OOD effect: on MetaDiT's own data the model
  recovers the true geometry from the true spectrum at ~0.78 IoU, uniform across model
  sizes; on the generated symbols the same pipeline collapses to ~0.31. The full-split
  metrics (MAE/gates in the tables above) were likewise always measured on MetaDiT's
  released splits — the generated sets are extra OOD probes, not the benchmark.

---

## Task 1 (2026-10-07) — novel-geometry round-trip, rebuilt on a corrected Meep harness

Everything below supersedes the earlier symbol-round-trip and Meep-based claims in this
file. Those rested on an instrument that violated passivity and on a convention whose
held-out error was worse than predicting zero. Both defects are described first because
they invalidate the prior numbers rather than merely qualify them.

### 0. Two defects found and fixed

**The Meep harness produced unphysical spectra.** `_meep_novel.py` ran every design for a
fixed 1100 time units in a lossless cell. The cell is PML-terminated in z and periodic in
x/y, so a lossless slab traps energy the z-PML cannot absorb; the run stopped mid-ring
and the flux at the monitor was still oscillating. Consequence: **36 of 43 designs across
the symbol-decodes and batch40 kernels violated the passivity bound |T| <= 1**, up to
**|T| = 4.7e8** for a passive dielectric. Three further defects: the mode parity channels
were meaningless (`def`/`EVEN_Z`/`ODD_Z` came out byte-identical for every design, because
`eig_parity` is not resolvable for a zero-thickness ModeRegion); there was no validity gate;
and the rewrite had dropped the complex phase, so the output could not have been used to
condition the model at all. Fixed with a lossy dielectric, field-decay termination plus a
hard ringdown cap, a single meaningful mode channel, and a per-design `passivity_ok` flag.
Runtime fell ~1500s -> ~110s per design.

**The Meep->CST convention carried no signal.** The old `_meep_convention.json` was a
2-parameter global phase rotation fitted on 4 items with residuals up to **3898**, one item
explicitly discarded, fitted on the defective harness's output. Refitted on 19 usable
validation geometries with the fixed harness, leave-one-out:

| model | held-out err | corr | |
|---|---|---|---|
| phase, conj=False | 1.2079 | 0.680 | **the old parameterisation** |
| phase, conj=True | 1.1815 | 0.680 | |
| scalar gain, conj=False | 0.7502 | 0.533 | |
| scalar gain, conj=True | 0.1278 | 0.698 | |
| **full 2x2 map** | **0.1138** | **0.785** | **selected** |

So the fixed harness reproduces MetaDiT's CST spectra to ~11% of the target standard
deviation held out. In-sample 0.0895 vs 0.1138 held out, so it is not overfitting 19
points. The old convention's 1.18 is worse than predicting zero — **every result built on
it was uninterpretable, and the prior OOD-collapse claim is withdrawn.**

### 1. Data status

| set | designs | passivity failures | ringdown-capped |
|---|---|---|---|
| calibration (validation geometries) | 20 | 1 (peak 1.7e8) | 0 |
| batch40 novel geometries | 40 | 2 (asym02 1.074, asym15 1.021) | 0 |
| decoded, full mask | 38 | 3 | 0 |
| decoded, 50% mask | 38 | 3 | 0 |

Failures are **excluded, not repaired**. Two of the calibration failures previously hit the
ringdown cap; raising `--max-ring` to 2500 removed that failure mode entirely.

### 2. Decode and geometry metrics (base model, sched-fix step 69900)

| mask | IoU vs true novel geometry | surrogate spec err | \|T\| corr |
|---|---|---|---|
| full | 0.674 | 0.280 | 0.664 |
| 50% | 0.777 | 0.245 | 0.717 |

### 3. Meep-vs-Meep physics, with a shuffled null

Both sides measured with the same instrument, so the fitted convention cancels entirely.

| subset | real err | shuffled | median err margin | win rate | sign test (corr) |
|---|---|---|---|---|---|
| full mask, all | 0.428 | 1.056 | -0.279 | 29/35 | p = 0.017 |
| 50% mask, all | 0.347 | 1.142 | -0.291 | 27/35 | p = 0.041 |
| **symmetric, full mask** | 0.444 | 1.431 | **-0.060** | — | **p = 1.00** |
| asymmetric, full mask | 0.396 | 0.804 | -0.250 | — | p = 0.035 |
| symmetric, 50% mask | 0.287 | 1.254 | -0.289 | — | p = 0.263 |

**The symmetric + full-mask stratum is a null**: median |T| correlation margin +0.0025,
sign test p = 1.0. Its mean margins (-0.99, +0.23) look favourable but are carried entirely
by shuffled-error outliers of 12-14, with CI [-2.60, -0.0015]. The pooled result is
carried by the asymmetric designs.

### 4. Is that stratum ill-posed? (degeneracy check)

Measured on MetaDiT's own validation split with no model involved: for each sampled
geometry, find the OTHER geometry whose true CST spectrum is closest, then compare the two
geometries.

| stratum | NN/random spectrum distance | median IoU with that neighbour | frac IoU < 0.8 | frac < 0.5 |
|---|---|---|---|---|
| whole corpus (n=3000) | 0.034 | 0.663 | 0.871 | 0.198 |
| occupancy band 1040-2240 (n=2820) | 0.036 | 0.671 | 0.862 | 0.161 |
| least mirror-symmetric 300 | 0.054 | 0.662 | 0.897 | 0.167 |
| **most mirror-symmetric 300** | **0.068** | **0.625** | 0.910 | 0.233 |

**Verdict: partially, not fully, degenerate.** The spectrum is highly informative — the
nearest-spectrum neighbour sits 29x closer than a random pair — so the condition is not
ill-posed. But for ~87% of geometries the closest-spectrum neighbour is a *different*
design (IoU < 0.8), and mirror symmetry makes this worse (median NN IoU 0.625 and 23.3%
below 0.5, vs 0.662 and 16.7% for the least symmetric subset).

The relevant calibration: **our model's IoU on that stratum is 0.674, at or slightly above
the corpus's own nearest-neighbour IoU of 0.625-0.671.** The model is therefore performing
at approximately the intrinsic identifiability limit of the spectrum->geometry map, not
below it. The null is a real result under a weakly-identified condition — so the stratum is
recorded as **weakly identified, not excluded by design**.

### 5. Style statistics vs the real MetaDiT corpus

| set | occ | holes | edge | mirror | components |
|---|---|---|---|---|---|
| real MetaDiT (n=402) | 0.428 | 0.55 | 0.125 | **1.000** | 1.12 |
| our novel (40) | 0.426 | 0.35 | 0.097 | 0.859 | 1.00 |
| JEPA decoded, full mask | 0.409 | **2.05** | 0.109 | 0.904 | **3.50** |
| JEPA decoded, 50% mask | 0.408 | **1.97** | 0.114 | 0.883 | 1.95 |
| MetaDiT-S (same input) | 0.398 | **0.21** | 0.121 | **1.000** | 1.47 |

**The real corpus is exactly mirror-symmetric**: 300/300 random validation geometries are
bit-identical under `fliplr`, under `flipud`, and under 180 degree rotation. This is
architectural, not physical — MetaDiT's `restore_structure` builds its 64x64 output by
mirroring a 32x32 quadrant, so every generation is mirror-symmetric by construction. Our
decoder has no equivalent constraint.

**Our novel designs are not drawn from the corpus distribution** (mirror 0.859 vs 1.000):
`make_random_batch40.py` enforces only a partial mirror, so the OOD probe is more OOD than
intended. The decodes fragment — up to 3.50 components and ~4x too many holes against the
corpus — and carry visible speckle and horizontal seam artifacts.

### 6. Same-input MetaDiT-S comparison

MetaDiT-S conditioned on the identical `[2,301]` targets, 500 steps, cfg 4.0, seed 0.
Cluster (2x A100 80GB) produced all 38 generations in 90.8s of sampling.

| model | IoU vs original novel geometry |
|---|---|
| JEPA full mask | 0.674 |
| JEPA 50% mask | **0.777** |
| **MetaDiT-S diffusion** | **0.576** |

The two models fail differently: we reconstruct the input more accurately; MetaDiT-S emits
far cleaner, more corpus-like bodies (0.21 holes, 1.47 components, mirror exactly 1.000).

**Caveats:** MetaDiT-S is stochastic and this is a single seed; a fair comparison needs
several seeds per input. And the targets are our novel geometries (mirror 0.865), which are
unlike anything in MetaDiT's corpus (mirror 1.000), which plausibly depresses its IoU.

### 7. Verdict and what is still open

- The harness and convention are now trustworthy: **0.1138** held-out against real CST.
- On novel geometries the decode **beats a shuffled null** for asymmetric designs and under
  a 50% mask, but **not** for symmetric designs under a full mask (p = 1.0).
- That stratum is **weakly identified** rather than ill-posed, and the model sits at the
  corpus's own nearest-neighbour identifiability limit there.
- Against MetaDiT-S on identical input we are **better on IoU** (0.674 / 0.777 vs 0.576)
  and **worse on geometry well-formedness** (fragmentation up to 3.50 components, ~4x too
  many holes, visible speckle and horizontal seams against MetaDiT-S's 1.47 / 0.21).
- **Superseded by this section:** the earlier symbol round-trip IoU ~0.31 and the
  "decoded spectra do not match symbol spectra" finding. Both were measured with the
  passivity-violating harness and the signal-free convention, so neither supports a claim.

### 8. What is NOT established here

- Nothing above says the JEPA beats MetaDiT. It beats MetaDiT-S on geometry IoU for a
  single diffusion seed on 38 OOD targets, and loses to it on geometry well-formedness.
  A multi-seed comparison is required before any such claim.
- The base-model headline numbers (MAE 0.0490 +- 0.0013, gates A/B/C
  0.9967/0.9772/0.9303) are unaffected by anything here: they are measured on MetaDiT's
  released test split with true CST spectra and never touched the Meep harness.
- The scalar path remains the recorded open weakness (0.505 / 0.529, at chance).
- Task 1 covers the geometry-first direction only. The spectrum-first direction (arbitrary
  target spectrum -> model -> geometry -> physics) has not been run.

---

## 2026-10-15 — v1/v2 re-scoring, the three-model comparison, imp1/imp2 (cluster: `paramsakti`)

All numbers below were produced on the HPC cluster (`paramsakti`, `mm24r002`) with the fixed
Meep harness; the Meep→CST convention is **0.1138** held out. **Supersedes**: every decode
number earlier in this file that came from the `sched_fix` (v2, failed-gate) checkpoint, and
the confounded MetaDiT scalar table. v1 = the retrained base (`checkpoints/unified/final.pt`
on the cluster, 70k, 1 seed); v2 = `sched_fix` seed0.

### 0. Scorer validation — why these numbers are comparable

- Re-scoring **v2 from raw cluster data with the current scorers reproduced this ledger
  exactly**: task 1 full `0.4284/1.0558`, half `0.3467/1.1419`; task 2 A `0.8487/0.6527`,
  B `1.9729/0.5756`. The v1 numbers come from the same code path.
- Cross-model comparisons use the **intersection** of each model's passivity-clean designs
  and **one shared shuffled permutation** (per-model subsets are not comparable).

### 1. Task 1 (geometry-first) physics — v1 / v2 / MetaDiT-S on the common 31 designs

| model | real err ↓ | shuffled | med. margin err / corr | win rate err / corr | sign p err / corr |
|---|---|---|---|---|---|
| base v1 | 0.4629 | 1.2866 | −0.2282 / +0.1981 | 23/31 · 23/31 | 0.011 · 0.011 |
| v2 | 0.4464 | 1.2928 | −0.3677 / +0.2239 | 23/31 · 21/31 | 0.011 · 0.071 |
| MetaDiT-S | 0.5432 | 1.3614 | −0.2517 / +0.1958 | 24/31 · 22/31 | 0.0033 · 0.029 |

- All three beat the shuffled null; **MetaDiT-S has the worst absolute agreement** (highest
  error, lowest correlation) and the best per-design win rate. v1 and v2 are statistically
  tied here.
- **Subset sensitivity (recorded, not smoothed):** v1's full-mask result is p=0.296 on its own
  33 designs but p=0.011 on the common 31 — two excluded designs carried large shuffled-error
  outliers. The common-subset value is the fair one; the sign test remains underpowered for
  small n (task 3 below, n=10).
- Still open: the symmetric + full-mask stratification has not been recomputed for v1.

### 2. Task 2 (spectrum-first) physics — three-way, same targets and null

| model | A real err / corr | A shuffled | A med. margin err / corr | A p | B real | B shuffled | B margin | B p |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.8607 / +0.4597 | 0.7353 / +0.5744 | +0.168 / −0.092 | 0.13 / 0.29 | 1.8362 | 0.6781 | +1.20 / −0.364 | 1.2e−7 |
| v2 | 0.8487 / +0.4609 | 0.6527 / +0.6092 | +0.149 / −0.114 | 0.13 / 0.13 | 1.9729 | 0.5756 | +1.37 / −0.420 | 0.031 |
| MetaDiT-S | 0.8368 / +0.4409 | 0.8038 / +0.5876 | +0.071 / −0.153 | 0.31 / 0.064 | 1.8393 | 0.7274 | +1.06 / −0.452 | 9.5e−7 |

- **Negative for all three models** — none beats a mismatched target on the spectrum-first
  task, and MetaDiT is marginally the worst on set A (largest negative correlation margin).
  Single-shot arbitrary-spectrum inversion is therefore **unsolved in this comparison**, not a
  weakness of our architecture relative to the reference.
- Passivity of the decodes: set A 24/24 clean for v1 and MetaDiT; set B clean for v1 (24/24),
  MetaDiT 21/24, v2 only 6/24.

### 3. Task 3 (scalar sensitivity) — matched conditions for all three models

Geometry rows (n=10, identical conditioning; IoU vs the true novel geometry):

| model | true scalars | −20 % | +20 % | occupancy (−20 / +20) |
|---|---|---|---|---|
| JEPA v1 | 0.607 (given-scalars row) | 0.153 | 0.145 | 0.071 / 0.067 |
| JEPA v2 | 0.622 | 0.280 | 0.288 | 0.125 / 0.133 |
| MetaDiT-S | 0.728 | 0.435 | 0.472 | 0.380 / 0.375 |

- MetaDiT tracks perturbed scalars through the spectrum far better; ours collapse to
  near-empty (v1) or fragment (v2). Caveat on our side: the JEPA must also *predict* the
  scalars in that regime (MetaDiT has no scalar input), so part of the gap is the
  conditioning path, not tracking ability.
- Physics (`_compare_phase3_physics.py`, same-instrument for the given-scalars rows):
  - `jepa_v1_B_true` n=10: real err 0.396 / corr **+0.892** vs shuffled 1.384 / +0.576
    (margins −0.99 err / +0.32 corr; win 6/10 · 6/10; sign p=0.75 — underpowered).
  - `jepa_v2_B_true` n=9: real 0.420 / **+0.899** vs shuffled 0.900 / +0.636
    (+0.26 corr margin; win 6/9 · 7/9; p=0.51/0.18).
  - **Three perturbed-scalar rows are excluded**: every design failed the passivity gate
    (peaks 1.02–1.10 for v2's sets, 1.17–2.11 for v1's +20 % set; all converged, none
    capped). These decodes are the near-empty ones (occ 0.07–0.13).
  - **CORRECTION (see §9): the explanation given here was wrong.** This passage originally
    attributed the excess to the sample and reference runs ending on different field-decay
    windows, and prescribed an identical window as the fix. The shared-window re-run was
    executed and changed every peak by **< 0.001**, refuting it. The measured cause is that
    the gate divides by a source spectrum spanning ~4.4 decades and mixes an eigenmode
    coefficient ratio (not an energy quantity) into a passivity statement. Corrected
    verdicts, and which rows survive, are in §9.
  - `v1_spec_m20`'s null (real == shuffled): peaks 0.998–0.999, which §9 shows is the
    *correct* value for a near-empty decode against an air reference, not an artefact.

### 4. imp1 — mirror-parameterised (quadrant) decoder · 1 seed · preliminary

Separate model (`configs/unified_imp1.yaml`, decoder emits a 32×32 quadrant and mirrors it,
plus refinement blocks; 11,565,444 trainable vs base 11,403,012; base config builds
byte-identically and every base checkpoint still strict-loads).

| metric | base v1 (3 seeds) | imp1 s0 (1 seed) |
|---|---|---|
| MAE, scenario A | 0.0490 ± 0.0013 (full split) | 0.0514 (n=512) |
| AAE | 29.51 | 30.92 |
| gate A / B / C | 0.9967 / 0.9772 / 0.9303 | 0.9961 / 0.9609 / 0.9238 |
| scalar 1 / 2 known | 0.505 / 0.529 | 0.559 / 0.527 |
| occupancy IoU (masked) | ~0.75 | 0.7544 |
| pred occ fraction (true) | 0.427 (0.425) | 0.422 (0.425) |

- **Parity within single-seed noise** on A/B/C and MAE; no improvement established.
  Guidance sweep: w=0 → 0.553, w=1 → 0.129 (best), w≥2 breaks down (7.2 / 11.8 / 14.9).
- Seeds 1-2 are queued (the cluster allows one GPU job at a time).

### 5. imp2 — leak-free (mirror-pair) masking · in flight

- **Measured leak** under the shipped random placement: **0.67** of masked tokens have a
  visible mirror partner at ratio 0.5 (the corpus is exactly mirror-symmetric by
  construction). Any symmetric decoder exploits this path automatically (imp1 does so by
  construction), which is why masked-region scores under leaky masks are not comparable.
- `mask_placement: mirror_pairs` closes the masked set under {fliplr, flipud, rot180}:
  **leak exactly 0.0 at every ratio**; calibration 0.25→0.27, 0.50→0.50, 0.75→0.75.
- Tests: 8 new (incl. masked pixels provably cannot change `z_hat`/`scalar_pred`, and the
  JEPA loss is invariant to the unmasked region) → suite **401 passed / 10 skipped**.
- First ablation datum (`scripts/diagnostics/mirror_leak.py`, base v1, n=6, ratio 0.5):
  masked-region IoU 0.5488 (leaky) → **0.6140** (leak-free) — base v1 does *not* exploit the
  leak; imp1 is the arm the instrument exists for.
- Training in flight (step ~37k/70k); its battery and the three-checkpoint leak ablation run
  in the same job.

### 6. Operational record for the next session

- `paramsakti` QOS: **one concurrent GPU job per user** (`QOSMaxJobsPerUserLimit` otherwise);
  `small` accepts several concurrent CPU jobs. GPU nodes are drained/reserved intermittently.
- The home **quota (50 GB) filled** mid-run, killing 3 Meep workers with `[Errno 122]`; a
  targeted resume job re-solved exactly those designs (`passivity_ok=True`, peaks 0.94–0.98).
  27 GB of the quota is the operator's own conda environments (untouched).
- Meep throughput on one 48-core node: ~7–40 min per 10-design set (memory-bandwidth bound;
  the first set of a run also pays the empty-cell reference solves for new scalar combos).

### 7. What is NOT established today

- **Everything on v1 is one training seed** → preliminary by `ACCEPTANCE_PROTOCOL.md`; the
  three-seed replication is queued, not done.
- imp1 vs base v1: parity, not an improvement.
- Task-3 perturbed-spectrum physics: **unmeasurable with the current harness** (reference/run
  decay-window mismatch, aggravated by near-empty decodes). No model claim is made from it.
- The mandated gate-8 (real/**null**/**shuffled**, stratified by mask ratio) still has no test.
- Phase-2's negative result is now a *comparative* statement (ours and MetaDiT's, same
  instrument) but remains single-seed per model.

---

## 2026-10-15 (later) — three seeds for imp1/imp2, the five-model physics table, corrected phase 2

Adds to the block above. All three model variants were trained 70k × seeds {0,1,2} on the
cluster, each with its own battery; all numbers here are from those runs.

### 1. imp1 — mirror-parameterised (quadrant) decoder · 3 seeds · NEGATIVE

| metric | imp1 (3 seeds) | base v1 (3 seeds) | verdict |
|---|---|---|---|
| MAE (n=512) | 0.0517 ± 0.0021 | 0.0490 ± 0.0013 (full split) | worse † |
| gate A | 0.9928 ± 0.0030 | 0.9967 ± 0.0011 | **worse, CIs do not overlap** |
| gate B | 0.9714 ± 0.0091 | 0.9772 ± 0.0040 | tied |
| gate C | 0.9297 ± 0.0070 | 0.9303 ± 0.0110 | tied |
| scalar 1 / 2 known | 0.523 ± 0.037 / 0.519 ± 0.007 | 0.505 / 0.529 | tied |
| occ IoU (masked) | 0.7565 | ~0.750 | equal |

**The mirror decoder is a clean negative on the battery** — identical on B/C and the scalars,
consistently slightly worse on A and MAE. Per-seed MAE 0.0514 / 0.0497 / 0.0539; A 0.9961 /
0.9922 / 0.9902.

### 2. imp2 — leak-free (mirror-pair) masking · 3 seeds · a TRADE

| metric | imp2 (3 seeds) | base v1 (3 seeds) | verdict |
|---|---|---|---|
| MAE (n=512) | 0.0501 ± 0.0009 | 0.0490 ± 0.0013 (full split) | inconclusive † |
| gate A | 0.9928 ± 0.0030 | 0.9967 ± 0.0011 | base better (separated) |
| **gate B** | **0.9824 ± 0.0020** | 0.9772 ± 0.0040 | **imp2 better: all 3 imp2 seeds ≥ base's best seed** |
| gate C | 0.9251 ± 0.0100 | 0.9303 ± 0.0110 | tied |
| scalar 1 / 2 known | 0.525 ± 0.011 / 0.539 ± 0.026 | 0.505 / 0.529 | higher means, overlapping |
| occ IoU (masked) | 0.7540 ± 0.0007 | ~0.750 | equal |

Per-seed imp2: MAE 0.0509 / 0.0492 / 0.0503; A 0.9922 / 0.9961 / 0.9902; B 0.9824 / 0.9805 /
0.9844; C 0.9277 / 0.9336 / 0.9141.

### 3. Mirror-leak ablation — the mechanism result

Masked-region IoU, ratio 0.5, n=38, identical inputs; only the mask placement changes:

| checkpoint | leaky masks (leak 0.431) | leak-free masks (leak 0.000) | Δ |
|---|---|---|---|
| base v1 | 0.7530 | 0.7363 | −0.0166 |
| imp1 | 0.7516 | 0.7328 | −0.0187 |
| imp2 seed 0 | 0.7138 | 0.7385 | **+0.0248** |
| imp2 seed 1 | 0.7198 | 0.7459 | **+0.0261** |

- The leak is real and inflates the leaky-mask scores of both models trained with it by
  ~0.017–0.019 IoU. imp1's mirror-by-construction decoder does **not** exploit it more than
  the baseline had already learned to.
- imp2 is the only checkpoint better under leak-free evaluation and the worst under leaky
  masks — both signs as predicted, replicated across its seeded runs.

### 4. Task-1 physics, five models on the identical 28 designs

Common passivity-clean intersection, one shared shuffled permutation:

| model | real err | shuffled | med. margin err | real corr | med. margin corr | win err/corr | sign p err/corr |
|---|---|---|---|---|---|---|---|
| v2 | 0.4482 | 1.156 | −0.341 | **+0.777** | **+0.286** | 22/28 · 23/28 | 3.7e−3 · 9.1e−4 |
| base v1 | 0.4519 | 1.201 | −0.264 | +0.766 | +0.236 | 22/28 · 21/28 | 3.7e−3 · 0.013 |
| imp2 | 0.4754 | 1.182 | −0.343 | +0.755 | +0.181 | 21/28 · 20/28 | 0.013 · 0.036 |
| imp1 | 0.5006 | 1.253 | **−0.346** | +0.657 | +0.132 | 22/28 · 19/28 | 3.7e−3 · 0.087 |
| MetaDiT-S | 0.5497 | 0.962 | −0.275 | +0.606 | +0.183 | 24/28 · 22/28 | 1.8e−4 · 3.7e−3 |

- **All five beat the shuffled null.** Ranking on absolute agreement: v2 ≈ base v1 > imp2 >
  imp1 > MetaDiT-S; imp1's correlation is only marginally significant (p=0.087).
- On the models' own sets, imp2's half-mask correlation (+0.810) is the best of the five.
- Supersedes the earlier "v1 full-mask is not significant" caveat: on matched subsets it is
  significant (p=6e−5 at n=30, p=3.7e−3 at n=28).

### 5. Phase 2 re-specified — the targets were unreachable, and fixing that does not rescue it

- **The defect:** the original set A sat **1.87** from the nearest realizable spectrum in the
  evaluator's normalized-L1 units against a corpus self-NN density of ~0.33 (27× further), and
  70 % of the way to the "unrelated spectra" scale (2.73); set B sat at 4.27, beyond unrelated.
  Every prior phase-2 decode was scored against a target no corpus geometry can realize.
  Roughness — the generator's original screen — does not detect this.
- **The fix:** `_make_feasible_spectra.py` screens on two metrics (normalized L1 and a
  scale-free correlation) and accepts A′ only near the manifold: **l1n 0.348 (corpus 0.331),
  corr 0.96**; B′ remains infeasible (l1n 4.49) as the control.
- **The result (same instrument, same null):**

| arm | real err / corr | shuffled | median margin err / corr | sign p |
|---|---|---|---|---|
| JEPA v1 (n=19) | 0.714 / +0.502 | 0.706 / +0.652 | +0.060 / −0.075 | 0.65 / 0.17 |
| JEPA v2 (n=19) | 0.691 / +0.549 | 0.702 / +0.625 | +0.093 / −0.119 | 0.36 / 0.65 |
| MetaDiT-S (n=23) | 0.722 / +0.402 | 0.689 / +0.617 | +0.033 / −0.176 | 0.40 / 0.40 |

- Absolute agreement improves substantially on feasible targets (v1 0.861 → 0.714; v2 0.849 →
  0.691), so the unreachable benchmark *was* inflating the numbers. **But every model still
  fails the fair control** — a mismatched realizable target is explained just as well — and
  MetaDiT's correlation margin is the worst of the three.
- **Corrected verdict:** the phase-2 negative is a property of the single-shot
  spectrum→geometry formulation, not of our architecture and not of the benchmark's targets.

### 6. What is NOT established (additions)

- imp1/imp2 battery numbers are 3 seeds but on the n=512 benchmark subset †; the full-split
  (17,489) re-measure is owed before their MAE is compared with base's 0.0490.
- imp1's and imp2's physics used seed-0 checkpoints only; the other seeds have no Meep pass.
- imp2's gate-B edge (all three seeds ≥ base's best) is small and offset by gate A: a trade,
  not an improvement.
- The task-3 harness fix and the gate-8 stratified test remain open, as in the block above.

### 7. Variants re-measured on the full test split (resolves †)

Scenario A, the whole 17,489-item test split — the same protocol as the baseline's
0.0490 ± 0.0013 (base seeds 0.0500 / 0.0494 / 0.0476):

| variant | MAE (3 seeds) | per-seed | vs base v1 |
|---|---|---|---|
| **imp1** | **0.0524 ± 0.0014** | 0.0531 / 0.0508 / 0.0533 | **+7.0 %; every imp1 seed worse than every base seed** |
| **imp2** | **0.0522 ± 0.0028** | 0.0519 / 0.0496 / 0.0551 | +6.5 %; 2 of 3 seeds above base's range |

- **Both variants are worse than base v1 on fidelity** once measured on the full split.
  imp1's deficit is a complete per-seed separation and lines up with its CI-separated gate-A
  deficit — the mirror decoder is a coherent negative.
- imp2's spread (0.0496–0.0551) is twice base's; its gate-B edge (0.9824 vs 0.9772, every seed
  ≥ base's best) is therefore bought with MAE: a trade, not an improvement.
- Method note: the n=512 subset read understated both variants' deficit (imp2's seed 2 reads
  0.0503 on the subset and 0.0551 on the full split) — subset comparisons flatter.
- Supersedes the "inconclusive †" rows in §§1–2 above; the gate and scalar columns there stand.

### 8. Gate 8 — stratified real / null / shuffled (the plan's critical gap)

`scripts/diagnostics/gate8_stratified.py`, n=512 per evaluation seed × seeds {3,4,5}
(1,536 pooled samples per stratum), paired per-sample win rate `real < shuffled` against the
0.75 threshold, bootstrap CI on the paired difference (5,000 resamples). Strata: *easy*
= mask ratio 0.25 with all scalars known; *hard* = mask ratio 1.00 with all scalars unknown
(the mandated gate).

| checkpoint | easy win rate | easy paired diff | **hard win rate** | hard paired diff (95 % CI) |
|---|---|---|---|---|
| **base v1** | 0.8945 | +0.1786 | **0.9974** | +0.5082 [+0.4966, +0.5206] |
| imp1 s0 | 0.8717 | +0.1674 | **0.9935** | +0.4964 [+0.4373, +0.5511] |
| imp2 s0 | 0.9023 | +0.2051 | **0.9948** | +0.5000 [+0.4244, +0.5711] |

- **The gate passes for all three checkpoints, decisively.** Gate 8 — "real must beat shuffled
  in physics-consistency, stratified by regime, gated on the hard stratum" — was the validation
  plan's *critical* gap (the suite covered real-vs-null only, unstratified). Answer: in the
  hard stratum the true spectrum beats a deranged one on 99.4–99.7 % of samples, with the CI
  nowhere near 0.5. The goal-ignoring collapse the plan was built to catch is **not present**.
- **The hard stratum is where conditioning is strongest, not weakest** (mean error real 0.129
  vs shuffled 0.62 vs null 0.53): with every geometry token and all three scalars masked, the
  spectrum is the only signal the predictor has and it clearly uses it. The easy stratum's
  lower win rate (0.87–0.90) reflects the visible context carrying the answer, not a defect.
- Real-vs-null alone would have been the weaker instrument here: null distance is +0.42 to
  +0.43 in the hard stratum while the shuffled margin is +0.50 — the two agree in direction,
  but only the shuffled control rules out the generic "predictions move when any input moves"
  explanation.
- Ranking on the hard gate: base v1 0.9974 > imp2 0.9948 > imp1 0.9935 — ordering consistent
  with the fidelity results (base best), and all three far above threshold.
- Not established here: gate 8 on the *scaling* arms (S0–L1) and on v2; and the hard-stratum
  numbers are eval-seed means of one checkpoint per arm, not 3 training seeds.

### 9. CORRECTION — the task-3 passivity gate was measuring itself, not the models

**The claim this replaces.** §3 explained the |T| > 1 rows as a measurement artefact in which
the sample and its empty-cell reference "end on different field-decay windows", and prescribed
"an identical ringout window for the run/reference pair" as the fix. That was a hypothesis
stated as a finding. It was then implemented and tested, and it is **wrong**.

**Refutation.** The measured window mismatch was real and large — the affected designs ring
for up to 470 time units while the reference adopted from `runs/t1/ref.json` stops at 105.2, a
4.5× difference — so the hypothesis was plausible. Four sets (`v1/v2 × spec_m20/spec_p20`)
were re-run with one shared explicit post-source window (`--ring-fixed 493.5`), sample and
reference integrated over the identical window. Result: **every peak moved by less than
0.001** (v2 spec_m20: 1.0218–1.0309 before, 1.0217–1.0308 after; v1 spec_m20: 0.9985–0.9994
before, 0.9984–1.0006 after). The excess is not window-related.

**The measured cause, from the stored curves only (no new FDTD).**

1. *The denominator is a source spectrum spanning ~4.4 decades.* Every stored reference — two
   files, two scalar combos — is the **identical** smooth curve: 2.77e-4 of its band median at
   ω = 0.10, 12.17× at ω = 0.1517, 1.11e-3 at ω = 0.20. It does not move with (l, h, r), which
   is what an empty air cell must look like: it is the source's own spectrum. The gate takes
   `max` over the whole band, so it includes frequencies where the denominator is 1e-4 of the
   band centre — where the ratio is noise over noise. Measured at every recorded peak, the
   denominator sat at 2.8e-4 to 4.8e-4 of its band median.
2. *The channel that exceeds 1 is not an energy quantity.* The harness scores
   `max(flux ratio, eigenmode-coefficient ratio)`. The second is `a_s/a_e` from
   `get_eigenmode_coefficients`, which MEEP normalises by the mode of the geometry at the
   monitor — different for an air cell and for a cell containing a slab — so it is not a
   transmission coefficient and |T| ≤ 1 does not apply to it. It is exactly the channel that
   exceeds 1 at *well-conditioned* frequencies for v2 (depth 2.1–2.7× median).

**Corrected verdicts** (flux channel only, i.e. the energy statement; denominator floor 1e-2
of band median, throat of the band excluded; floor swept 0 → 5e-2 and the verdicts are stable):

| set | flux peak range | verdict vs 1.02 | what changed |
|---|---|---|---|
| `v1_B_true` | 0.918–0.985 | pass | unchanged |
| `v1_spec_m20` | 0.998–0.999 | pass | was "benign artefact" — it is the correct value for a near-empty decode vs air |
| **`v1_spec_p20`** | **1.026–1.193** | **still fails** | survives every correction; 1.026 even at the 5e-2 floor |
| `v2_B_true` | 0.908–0.976 | pass | the 1.2584 outlier was a band-edge denominator |
| `v2_spec_m20` | 0.996–0.997 | **pass** | its 1.022–1.031 excess was entirely mode-channel |
| `v2_spec_p20` | 0.989–0.994 | **pass** | its 1.070–1.096 excess was entirely mode-channel |

So **two of the three excluded rows are recovered** (both v2 rows — the models are fine; the
gate was not), one row (`v1_spec_p20`) remains a genuine finding at 1.026–1.193, and the
`v1_spec_m20` "pinned null" is reclassified as a correct measurement rather than an artefact.

**What is not yet established.** Whether `v1_spec_p20`'s excess is physics or the
instrument's own offset: the surviving excess approaches a uniform ~2.6 % (1.026 at the
strictest floor, across all 10 designs), which is the size of a PML-reflection systematic, and
1.02 is a tolerance tighter than that. The control that settles it is running — the harness
applied to an **empty** cell (sample and reference physically identical, so the reading is the
statistic's own error), twice per scalar combo, job `642465`. Until it returns, no passivity
claim about `v1_spec_p20` should be quoted in either direction.

**Harness status.** `--ring-fixed` (shared window) is retained: it is correct practice for
paired comparisons and is now guarded so protocols can never be mixed, even though it is not
the fix for this. The gate itself still needs the two corrections above wired in (flux-only
scoring, conditioned-band evaluation) before any future run's `passivity_ok` is trustworthy.