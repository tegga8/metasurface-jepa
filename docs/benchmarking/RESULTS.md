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

- The symbols (swastika, cross, crescent, aum, khanda, dharmachakra, yin-yang, torii,
  faravahar) sit far outside the dataset's blob morphology; recovery from their full-wave
  Meep spectra is partial at best — ~0.30 IoU vs ~0.78 in-distribution control, uniform
  across model sizes. Thin glyph-scale features are not uniquely pinned by the spectrum:
  several symbols show low surrogate spectrum error at low IoU (the model finds *a*
  design that explains the spectrum, not *the* glyph).
- artifacts: `symbol_roundtrip_preview.png`, `symbol_designs.npz` / `_preview.png`,
  `kaggle/symbol_designs` + `kaggle/symbol_meep` kernel outputs.
