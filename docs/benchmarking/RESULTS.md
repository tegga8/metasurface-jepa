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

---

## Phase 5 — scalar capacity & bounds

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _

---

## Phase 6 — encoder sizing

- date: _TBD_   commit: _TBD_   change: _
- before / after / delta / gate: _
