# FINAL_REPORT.md — Architecture Repair / Cleanup

Date: 2026-08-20 (session). Workspace: metasurface-jepa, branch `main`.
Snapshot of decisions and state is in `ARCHITECTURE_CLEANUP_AUDIT.md`; this report is the
operator-facing summary of the repair work and what is still required before the pipeline can be
trusted.

**Overall status: NOT READY for model-selection conclusions.** All code-level repairs are done
and the local test suite is green, but the two gates that actually certify a winner — the short
GPU audit (§15/§34) and the multi-seed gate (§35) — must still be run on cloud GPU per
`CLOUD_TRAINING.md`. Nothing below claims a training result.

---

## A. What was wrong and what was repaired

The Milestone-B codebase had accreted an "adaptive ladder" architecture that violated the
operator's spec: a shared `model.proj`, multiple objective rungs (`jepa_var`, `jepa_vicreg2`),
a LOSS_LADDER progression with phase transitions, a `direct` model variant, an
`itertools.cycle` epoch loop, and per-phase checkpoint files. This was removed wholesale and
replaced by the architecture the spec describes:

- **One predictor** (`src/predictor/gclct.py`): a single physics-conditioned GCLCT. Conditioning
  is the FiLM path `cond = Sequential(SiLU(), Linear(hidden, 6*hidden))` with a **zero-initialized
  last layer**; the six output groups are `gamma1/beta1 .. gamma3/beta3`, one per attention
  block (self-attn, cross-attn, MLP). `c_physics` enters through this path only — never by
  concatenation. No Perceiver bottleneck, no base+delta prediction, no head_type parameter.
- **Exactly three objectives** (`src/losses/objectives.py`), registry `OBJECTIVES =
  {jepa_vicreg, jepa_barlow, lejepa}` (test-locked). Every objective owns its own projector
  (`src/losses/objective_modules.py`); there is **no `model.proj`** anywhere. `L_J` is computed
  in the raw latent space (no projector); the projection/regularization happens in
  objective-owned space only.
- **One engine** (`src/train/engine.py`): fixed `FixedValidation.evaluate`, `null_gap`,
  `healthy_references`, `classify_failure_mode` (five-way, §26), and the §30 checkpoint
  round-trip (model + objective + optimizer + scheduler + RNG + EMA + best-state) with strict
  objective-name and optimizer-ownership gates.
- **One training driver** (`scripts/train/train_milestone_b.py`): CLI + `--resume`, explicit
  epoch iteration (`for bi, (G, S) in enumerate(loader)`), no `itertools`, one checkpoint per
  epoch plus a best-model snapshot, checkpoints named `{exp}_{objective}_latest.pt` /
  `{exp}_{objective}_best_model.pt`.

## B. The three objectives (final)

| Name | Target encoder | Regularizer | Notes |
| --- | --- | --- | --- |
| `jepa_vicreg` | EMA copy | VICReg (variance + covariance + invariance) | Variance term prevents the absolute-spread collapse. |
| `jepa_barlow` | EMA copy | Barlow Twins (mean-form, `alpha=0.005`) | Targets **redundancy** (off-diagonal cross-correlation), not identical-branch spread — spread collapse is VICReg's job. |
| `lejepa` | shared encoder, `stop_grad_target=False` | SIGReg (sliced-ECF Gaussianity, §40 parity) | No EMA/teacher copy; both branches receive gradient. |

Ablation E (§10.2, Milestone G) decides between EMA-based (vicreg/barlow) and LeJEPA on this
same architecture; the code supports all three so that decision can be made empirically.

## C. Test suite

`python -m pytest tests -q` → **133 passed, 6 skipped** (skips are CUDA-only tests on the
CPU-only local machine). The suite covers:

- Objective math: VICReg gradients + collapse/recovery, Barlow mean-form components + redundancy
  collapse, LeJEPA SIGReg gradients + collapse penalization, sigreg metadata (per-branch
  `num_slices/num_points/t_grid/seed/mean_phi_dev`).
- Predictor conditioning: FiLM six-group shape/behavior, zero-init proof, output depends on
  `c_physics`, gradient reaches `c_physics`/`a_goal`, no `head_type` argument.
- Engine: fixed-validation metric paths, healthy-reference determinism, null-gap metric
  consistency, checkpoint full round-trip (§30), strict objective-name mismatch raises,
  optimizer ownership mismatch raises, missing `objective_state` fails loudly, EMA state
  round-trip, best-model-only loadable.
- Registry/architecture locks: exactly three rungs, no `model.proj` needed, no dead ladder
  imports, no `cycle(` in the training path.

Smoke run (CPU, toy dims): `eval_vicreg_sanity.py --smoke --short-audit` ran 6 optimizer steps
end-to-end and wrote `checkpoints/milestone_b/vicreg_sanity/eval_jepa_vicreg_short_audit.json` —
no crash in `classify_failure_mode` or `term_ratios`; the component ratios now sum to 1.0
(0.3187 + 0.6676 + 0.0137), confirming the term-ratio "total" bug fix.

## D. §32 dead-code / architecture greps

Run over `src/` and `scripts/`: `model.proj`, `.proj`, `Perceiver`, `bottleneck`, `base*delta`,
`LOSS_LADDER`, `ladder`, `adaptive`, `select_winner`, `phase_checkpoint`, `jepa_vicreg2`,
`jepa_var`.

**Result: no code paths remain.** The only hits are prose (docstrings/comments) stating that
these things were removed, plus legitimate internal projection layers inside the attention
stack / patch embedder (`spectrum_encoder.proj`, `geometry_encoder.proj` — attention output
projections, not the forbidden shared `model.proj`). `vicreg.vicreg_loss` (the historical
single-branch helper) is still defined and directly exercised by `test_vicreg_objective.py` /
`test_vicreg_collapse.py`; it is not dead code but is documented as legacy.

## E. Files added

- `scripts/eval/physics_conditioning_audit.py` — §10 audit: `c_physics`/`a_goal` embedding
  health (feature std, pairwise cos, eff-rank), predictor real-vs-null / real-vs-shuffled
  deltas, physics cos_err raw/proj, and a three-way verdict:
  - `CASE_A_EMBEDDING_COLLAPSE` — `rank_fraction < 0.02` or `mean_feature_std < 0.05`
  - `CASE_B_PREDICTOR_DEAD` — both deltas `< 1e-4`
  - `CASE_C_HEALTHY` — bars documented in the script header; **the exact numeric bars for
    Case C were not fixed by the spec and are flagged for human confirmation before this audit
    is used as a gate** (Standing Rule 3).
- `src/reference/` — deterministic reference model building for healthy-reference baselines.
- Tests: `test_predictor_conditioning.py`, `test_barlow_gradients.py`, `test_barlow_collapse.py`,
  `test_lejepa_gradients.py`, `test_lejepa_collapse.py`, `test_checkpoint_resume.py`.
- `external/lejepa/` — cloned upstream LeJEPA reference (§40 parity, see F).

## F. §40 upstream parity (LeJEPA)

`git clone --depth 1 https://github.com/galilai-group/lejepa` now succeeds (earlier attempt
timed out leaving a `.gitkeep` stub; stub removed, clone complete at
`external/lejepa/`). Parity check performed against
`external/lejepa/lejepa/multivariate/slicing.py` and `univariate/epps_pulley.py`:

- Same test family: sliced multivariate Gaussianity via random **unit-Gaussian 1D projections**
  of the sample cloud, ECF quadratic-distance statistic `T = N·∫|φ_emp(t) − φ_N(t)|² w(t) dt`,
  symmetry-weighted t-grid over `[0, t_max]`.
- Concrete choices differ and are recorded as **reportable choices, not implicit defaults**
  (per §3.3): our `src/losses/sigreg.py` uses a fixed t-grid `{0.25, 0.5, 1.0, 1.5, 2.0}` and a
  min(masked features, requested) point count; upstream uses `linspace(0, t_max, n_points)` with
  trapezoid weights and a distributed-synchronized per-step projection seed.
- **Verdict: parity is confirmed at the family level; ours is a compatible re-implementation,
  not a copy, and is not presented as the canonical upstream package.** Whether the concrete grid
  choice matters empirically is Milestone G / Ablation E work on cloud GPU.

## G. Files deleted

`configs/milestone_b_adaptive.yaml`, `src/train/adaptive.py`,
`external/lejepa/.gitkeep` (stub), and the ladder tests
`test_jepa_var_objective.py`, `test_jepa_vicreg2_objective.py`, `test_ladder_extension.py`,
`test_ladder_ratios.py`, `test_ladder_summary.py`, `test_winner_phase_ok.py`.

## H. Known remaining code debt (non-blocking)

- `physics_conditioning_audit.py` Case C bars need operator confirmation (§10, Standing Rule 3).
- The historical `vicreg.vicreg_loss` helper is kept for backward compat (tested, not dead).
- `eval_vicreg_sanity.py` term-ratio "total" and `classify_failure_mode` call were fixed; its
  Case A/B gates reuse the same bars as the audit and inherit the same confirmation need.

## I. What still requires cloud GPU (NOT done locally)

- **§15/§34 short GPU audit**: real-batch forward/backward through the full §11 model + frozen
  surrogate + in-batch negatives; the short audit abort criteria (§15) apply.
- **§35 multi-seed gate**: seed-sweep determining whether the three-objective comparison is
  decisive (the actual winner-selection run).
- The LeJEPA-vs-EMA Ablation E comparison (Milestone G scope).
- Any real training run at non-toy batch sizes.

The local machine (RTX 3050 4GB) is dev-only per AGENTS.md; these runs belong on Kaggle/Colab
per `CLOUD_TRAINING.md` and must be reviewed by the human operator (step-3 review) before any
conclusion is drawn.

## J. Verify-this-on-GPU checklist (wired into scripts, runnable as-is)

- `python scripts/eval/eval_vicreg_sanity.py --config configs/milestone_b.yaml --objective jepa_vicreg`
- `python scripts/eval/eval_vicreg_sanity.py ... --objective jepa_barlow`
- `python scripts/eval/eval_vicreg_sanity.py ... --objective lejepa`
- `python scripts/eval/physics_conditioning_audit.py --config ... --checkpoint ... --objective ...`
- `python scripts/eval/compare_milestone_b_candidates.py` (three-objective table)
- `python scripts/eval/unseen_multimask_generalization.py`
- `python scripts/eval/decisive_representation_validation.py`

Each writes its report under `checkpoints/milestone_b/` and expects checkpoints named
`{exp}_{objective}_latest.pt` / `_best_model.pt`.

## K. Honest bottom line

- Local state: architecture repaired to spec, dead code removed, three objectives test-locked,
  engine + §30 checkpointing hardened, 133 tests green, §40 upstream parity confirmed.
- NOT READY to name a winner or declare any representation-health / guidance-gap finding: those
  are GPU-gated and unrun. Until §15/§34 and §35 pass with human review, the correct status is
  "code ready, results pending," and no section of this report should be cited as evidence of a
  training outcome.