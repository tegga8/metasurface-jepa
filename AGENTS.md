# AGENTS.md — Goal-Conditioned Physics JEPA for Metasurface Inverse Design

This file is the operational playbook for any coding agent working on this repository. It does
not restate architecture, losses, or research rationale — all of that lives in the design
document at `docs/design_doc.md` (the attached v2 design doc). This file only specifies **what to
build next, in what order, with what stop conditions**.

Every phase below maps directly onto the design doc's own structure:
- Setup work not covered by the design doc → **Phase -1** (this file only)
- `§7` Training pathway Phases 0–7 → embedded inside the milestones below
- `§7.1` Milestone sequencing A–I → the phase headers below, in the same order, same letters

Do not reorder, merge, or rename phases. Do not invent a different breakdown.

---

## Standing rules (always in effect, every phase)

1. **One milestone at a time.** Never implement components from two different lettered milestones
   (§7.1 A–I) in the same working session, even if they look related. If Milestone C work exposes
   a need for something that belongs to Milestone D, stop and note it — do not pull it forward.
2. **No unmotivated mechanisms.** Do not add a mechanism, loss term, or architectural piece unless
   a specific, measurable failure observed in an earlier milestone motivates it. "This might help"
   is not sufficient justification. Cite the failure mode (§13) or ablation (§10.2) it addresses.
3. **Stop and ask on ambiguity.** If a design doc section is ambiguous, or references a number,
   threshold, or hyperparameter that isn't concretely specified, stop and ask the human operator
   for a decision rather than guessing a value. This applies especially to gate thresholds — the
   design doc frequently states *that* a gate must pass without stating the exact numeric bar.
4. **Re-verify unverified external assumptions before relying on them.** The design doc explicitly
   flags several things as hypotheses to re-check, not settled fact:
   - MetaDiT repository file paths (`model/dit.py`, `train/train_metadit.py`,
     `model/spec_encoder.py`, `model/clip_model.py`) — §1.2 says these should be re-confirmed
     against the live repo at the start of Phase 0, since AAAI-26 camera-ready repos are often
     restructured post-acceptance.
   - The citation `arXiv:2606.27014` mentioned in §18 as "not independently re-checked" — do not
     cite or rely on it without independent verification first.
   Do this re-verification at the point the design doc specifies (start of Phase 0 for the
   MetaDiT paths), not later, and not by assuming the document's hypothesis is correct.
5. **Compute budget awareness.** Before starting Milestones G (LeJEPA) or I (flow/diffusion),
   re-check §6's compute budget table and §7.1's explicit instruction: "Milestones A–F must
   succeed before spending compute on the LeJEPA ablation or flow-matching extension — both
   roughly multiply cost without changing the underlying prediction question, so they should not
   be run in parallel with the core pipeline's first pass."
6. **Every phase's done criteria are a gate, not a suggestion.** Do not start milestone N+1 work
   until milestone N's "must pass to proceed" criteria are met and recorded (see per-phase
   sections below, and the repo layout's `checkpoints/<phase>/REPORT.md` convention).
7. **Guardrail checks are mandatory, not optional cleanup.** Each phase below lists the specific
   failure mode(s) from §13 it is most exposed to. Run the associated check before declaring the
   phase done, not only in a final aggregate ablation pass.
8. **Local machine is for code, not training.** Per the Compute Environment section below, the
   local machine (RTX 3050, 4GB VRAM / 16GB RAM) is confirmed insufficient for real training runs
   at the model sizes in §11. Any milestone whose task prompt involves gradient-based training
   must be executed on the cloud GPU (Kaggle or Colab) per the workflow below — the coding agent
   should write and unit-test code locally (forward pass on tiny/toy dims, batch size 1, CPU or
   local GPU) but should not attempt full training runs locally, and should not silently shrink
   model sizes to fit local VRAM without flagging it as a deviation per Standing Rule 2/3.
9. **Dated operator overrides.** If the human operator explicitly decides to deviate from this
   file (e.g. pulling forward a mechanism from a later lettered milestone), that decision must be
   recorded here with a date and a reference, so future sessions do not re-derive or reverse it.
   - **2026-08-17 — Milestone B deliberately expanded to an adaptive `L_J -> VICReg ->
     LeJEPA/SIGReg` screening ladder.** This overrides the normal one-milestone-at-a-time
     sequencing for this specific experiment: the EMA-JEPA-only scope stated in the Milestone B
     section ("do not implement LeJEPA yet, that is Milestone G") is set aside here, and
     VICReg/LeJEPA-SIGReg objectives, phase transitions, and winner selection live inside
     Milestone B's adaptive controller by operator decision. Tracked and worked via
     `checkpoints/milestone_b/BUGLOG.md` (see the fix directive that originated this entry).
   - **2026-08-24 — Physics-Guided Masked Retrofit sequence re-scoped mid-preflight.** The
     operator replaced the previously provided retrofit "Phase 2 (Geometry Decoder)" content
     with "Phase 2 — Fix VICReg Training/Validation Plumbing Before Cloud Run" (plumbing-only:
     objective device placement + eval-mode hygiene in validation/reference paths), explicitly
     forbidding decoder/inverse-design/swapped-spectrum work. Rationale: the verified VICReg
     CPU/CUDA construction defect blocks the genuine Milestone-B cloud training run that all
     checkpoint-gated preflight items depend on (`checkpoints/milestone_b/physics_validation/
     PHYSICS_TARGET_SELECTION_TRAINED_REPORT.md`). Same session therefore closed retrofit
     Phase 1 (Gate-0 PASS; Gate-2 candidate-freezing DEFERRED; target-health/Gate-1 stay
     BLOCKED pending a genuine trained checkpoint) and then executed the new plumbing phase.
     See `checkpoints/physics_retrofit/preflight/REPORT.md` and
     `checkpoints/milestone_b/PHASE2_VICREG_PLUMBING_REPORT.md`.

---

## Compute environment (decided — do not re-litigate per phase)

**Local machine:** RTX 3050, 4GB VRAM, 16GB system RAM.

**Decision, made once here so Phase -1 does not need to re-ask it:** the §11 model sizes (context
+ predictor + decoder + frozen surrogate, Adam optimizer states, in-batch negatives for the
InfoNCE loss from Milestone E, K=32 sampling in Milestone H) do not fit in 4GB VRAM at any batch
size useful for actual training. The local machine is therefore designated **dev-only**:

- Local machine: repo/environment setup, dataset shape verification, frozen-weight inference
  checks (Phase 0, Milestone A), writing and syntax/shape-testing new modules (forward pass only,
  toy dimensions, batch size 1), debugging.
- Cloud GPU (Kaggle T4/P100 16GB, or Colab T4/A100 16GB+): all actual gradient-based training,
  starting at **Milestone B** onward, and any evaluation run large enough to need real batch
  sizes (e.g. Milestone E's InfoNCE in-batch negatives, Milestone H's K=32 sampling).

Full step-by-step mechanics (repo sync, dataset staging, checkpoint persistence, resume-on-
disconnect) are specified once in **`CLOUD_TRAINING.md`** at the repo root — every milestone's
task prompt below that involves training should be run following that document, rather than each
phase re-deriving its own cloud workflow. Do not duplicate cloud-setup instructions inline in a
milestone prompt; reference `CLOUD_TRAINING.md` instead.

**Session-continuity implication for you (the human operator):** because training happens on a
separate machine/session from the coding agent, the loop for any training milestone is:
1. Coding-agent session (local): write/update code + config for the milestone, commit, push.
2. Cloud session (Kaggle/Colab, manual): pull latest code, run training per `CLOUD_TRAINING.md`,
   let it produce `checkpoints/<milestone>/REPORT.md` and any checkpoint files, push results back.
3. New coding-agent session (local): pull, read `checkpoints/<milestone>/REPORT.md`, verify done
   criteria, decide whether to proceed to the next milestone.
Do not skip step 3's manual review — an agent should not self-certify a training milestone's done
criteria from inside a session that didn't run the training.

---

## Repo layout (fixed — reuse across every phase, do not reinvent per-phase)

```
repo/
  AGENTS.md                     # this file
  CLOUD_TRAINING.md             # Kaggle/Colab runbook (setup, sync, resume) — canonical, referenced not duplicated
  docs/
    design_doc.md               # the attached v2 design document (source of truth)
  notebooks/
    cloud_train_runner.ipynb    # generic notebook: clone/pull, install, stage data, run a milestone script, push results
  external/
    metadit/                    # cloned MetaDiT repo (Phase 0)
    lejepa/                     # cloned LeJEPA reference repo (Milestone G only)
  data/
    metadit/                    # cached MetaDiT dataset (Phase 0) — staged as a Kaggle Dataset / Drive folder for cloud runs
  src/
    data/                       # dataset loading, block-masking (§2), curriculum sampling
    encoders/
      geometry_encoder.py       # §3.1
      context_encoder.py        # §3.2 (+ Perceiver bottleneck)
      target_encoder.py         # §3.3 (EMA + LeJEPA variants)
      spectrum_encoder.py       # §3.4 (reused MetaDiT spec encoder + goal-token pooling)
    predictor/
      gclct.py                  # §3.5 Goal-Conditioned Latent Completion Transformer
      guidance.py               # §3.5.1 classifier-free goal guidance
      routing.py                # §3.5.2 sparse spectral-to-spatial routing
    decoders/
      geometry_decoder.py       # §3.6
    surrogate/
      em_surrogate.py           # frozen MetaDiT forward EM surrogate wrapper, §1.6
    losses/
      jepa_loss.py              # L_J
      sigreg.py                 # LeJEPA SIGReg, §3.3 Variant B
      geometry_loss.py          # L_G
      physics_loss.py           # L_S
      alignment_loss.py         # L_A
      goal_infonce.py           # L_goal, §4
      curriculum.py             # loss-weight schedule per §4.1 table
    diagnostics/
      routing_jacobian.py       # §20.1 externally-validated routing consistency
      goal_token_entropy.py     # §20.2 goal-token utilization entropy
      guidance_gap.py           # §20.3 normalized classifier-free guidance diagnostic
    stochastic/
      latent_seed.py            # §7.4 stochastic latent interface (Milestone H)
      flow.py                   # §7.4 optional flow-matching extension (Milestone I only)
    model.py                    # top-level assembly per §11.1 data-flow spec
  configs/
    phase0.yaml
    milestone_a.yaml
    milestone_b.yaml
    milestone_c.yaml
    milestone_d.yaml
    milestone_e.yaml
    milestone_f.yaml
    milestone_g.yaml
    milestone_h.yaml
    milestone_i.yaml
  scripts/
    train/
      train_milestone_<x>.py    # must run standalone via `python scripts/train/train_milestone_<x>.py --config ...`
                                 # so the cloud notebook can invoke it without extra glue code
    eval/
      reproduce_metadit_baseline.py     # Phase 0 / Milestone A
      eval_hypotheses.py                # H1–H6, §12
      run_ablation.py                   # §10.2 ablation table, parametrized by ablation ID
    diagnostics/
      run_routing_jacobian_check.py     # §20.1
      run_guidance_gap_sweep.py         # §20.3
  checkpoints/
    phase0/REPORT.md
    milestone_a/ + REPORT.md
    milestone_b/ + REPORT.md
    milestone_c/ + REPORT.md
    milestone_d/ + REPORT.md
    milestone_e/ + REPORT.md
    milestone_f/ + REPORT.md
    milestone_g/ + REPORT.md
    milestone_h/ + REPORT.md
    milestone_i/ + REPORT.md
  logs/
```

Every milestone's `checkpoints/<milestone>/REPORT.md` must record: what was built, which config
was used, the done-criteria metrics actually observed, which guardrail checks were run and their
results, where training was executed (local dev-only smoke test vs. cloud full run, per the
Compute Environment section), and any deviations from the design doc (with justification). Later
milestones' task prompts assume this report exists and is readable.

**Training scripts must be resumable.** Because cloud sessions (Kaggle ~9–12hr cap, Colab free
tier idle-disconnects) can end mid-run, every `scripts/train/train_milestone_<x>.py` must accept
a `--resume <checkpoint_path>` flag and checkpoint frequently (at minimum every epoch, ideally
more often for long epochs) rather than assuming one uninterrupted run. This is a hard requirement
starting at Milestone B, not an optimization to add later.

---

## Phase -1 — Environment, scaffolding, and access setup

The design doc assumes an environment and dataset access already exist; it specifies neither.
This phase must happen first and is not itself a research milestone.

**Goal.** Stand up a working repo, environment, dataset, and confirm the compute plan (local
dev-only + cloud training, per the Compute Environment section above) is ready to execute, before
any modeling work begins.

**Preconditions / gate.** None — this is the first phase. Nothing in `src/` should exist yet.

**Exact task prompt:**
> Set up the project environment and repo scaffolding for the Goal-Conditioned Physics JEPA
> project described in `docs/design_doc.md`. Do the following, in order, and stop after step 7 —
> do not begin any modeling work (that starts at Milestone A):
> 1. Create the repo directory layout exactly as specified in `AGENTS.md`'s "Repo layout" section.
> 2. Set up a local Python environment with the dependencies needed to run the MetaDiT reference
>    implementation and a standard PyTorch training stack (exact package list is not specified in
>    the design doc — infer minimally from the MetaDiT repo's own requirements once cloned in
>    step 3, and do not add packages beyond what MetaDiT plus standard training/eval tooling
>    requires). This environment is for dev/debugging only, per the Compute Environment section —
>    do not size anything in it for full training.
> 3. Clone the MetaDiT repository from the exact URL in `docs/design_doc.md` §18:
>    `https://github.com/JessePrince/metadit`, into `external/metadit/`. Do not clone from any
>    other URL or a fork.
> 4. Download the MetaDiT dataset and released weights from the exact URL in §18:
>    `https://huggingface.co/datasets/Hao-Li-131/MetaDiT-AAAI2026`, into `data/metadit/`. Also
>    note in the report how this same dataset should be staged for cloud runs (Kaggle Dataset
>    upload or Drive folder) per `CLOUD_TRAINING.md` — staging itself happens in `CLOUD_TRAINING.md`'s
>    workflow, not here, but the download source is confirmed here.
> 5. Per Standing Rule 4 and design doc §1.2: inspect the actual current structure of
>    `external/metadit/` and record in `checkpoints/phase0/REPORT.md` whether the file paths the
>    design doc references (`model/dit.py`, `train/train_metadit.py`, `model/spec_encoder.py`,
>    `model/clip_model.py`) still exist at those paths. If they have moved, record the actual
>    current paths — do not silently substitute them into later code without noting the
>    discrepancy in the report.
> 6. Ensure `scripts/train/train_milestone_<x>.py` files (created in later milestones) are
>    designed from the start to run standalone via CLI (`python scripts/train/train_milestone_b.py
>    --config configs/milestone_b.yaml [--resume <path>]`) with no notebook-specific glue, since
>    they will be invoked from `notebooks/cloud_train_runner.ipynb` on Kaggle/Colab, not written
>    interactively there. Record this convention in the report so later milestones follow it.
> 7. Confirm read access to `CLOUD_TRAINING.md` and `notebooks/cloud_train_runner.ipynb` at the
>    repo root/notebooks — these already specify the cloud training mechanics; do not re-derive a
>    different cloud workflow.
>
> Stop after step 7. Do not clone any other repository (LeJEPA, eb_jepa) yet — those are needed
> only starting at Milestone G and the goal-conditioning work in Milestone E, respectively, and
> should be fetched at that point per Standing Rule 1.

**Files/paths.** `external/metadit/`, `data/metadit/`, repo scaffold as in "Repo layout" above,
`checkpoints/phase0/REPORT.md`, `CLOUD_TRAINING.md`, `notebooks/cloud_train_runner.ipynb`.

**Done criteria.**
- *Must pass to proceed*: repo layout exists; MetaDiT repo cloned from the exact §18 URL; dataset
  and weights downloaded from the exact §18 URL; file-path re-verification recorded in
  `checkpoints/phase0/REPORT.md`; CLI-first convention for training scripts recorded.
- *Nice to have*: a `requirements.txt`/lockfile pinned to working versions.

**Guardrails specific to this phase.** No failure mode from §13 applies yet (no model exists).
The only risk is silently proceeding on an unverified repo-structure assumption (§1.2) or
inventing a different cloud workflow than the one fixed in `CLOUD_TRAINING.md` — both disallowed
by Standing Rules 3–4 and the Compute Environment section.

---

## Phase 0 — Dataset verification (design doc §7, "Phase 0")

**Goal.** Verify the MetaDiT dataset and released components load and behave as documented,
before any new code is trusted. Per §7: *"Nothing else proceeds until this reproducibility gate
passes."*

**Preconditions / gate.** Phase -1 complete: repo, MetaDiT clone, dataset, and weights present;
file-path re-verification recorded.

**Exact task prompt:**
> Following `docs/design_doc.md` §7 "Phase 0 — dataset verification," using the repo scaffolded
> in Phase -1: load the MetaDiT dataset and verify (a) geometry tensor shape is `3×64×64`, (b)
> spectral target shape is `301×2`, (c) the released spectrum encoder, geometry ViT, and forward
> EM surrogate weights load without error, using whatever the file-path re-verification in
> `checkpoints/phase0/REPORT.md` found to be the actual current paths (not the paths as originally
> written in the design doc, if they've moved). Then reproduce MetaDiT's own forward prediction
> and spectrum evaluation on a held-out split, and record baseline metrics exactly as MetaDiT
> reports them (do not redefine the metric).
>
> This is inference-only work (no training), so it can run entirely on the local machine per the
> Compute Environment section — no cloud session is needed for this phase.
>
> The design doc states this gate must pass before anything else proceeds, but does not give a
> numeric tolerance for "reproduced." Propose a tolerance (e.g. matching MetaDiT's reported
> numbers within some percentage, or matching order-of-magnitude and rank behavior) and get it
> confirmed by the human operator before treating the gate as passed — do not pick a threshold
> unilaterally per Standing Rule 3.
>
> Stop once shapes are verified, weights load, and baseline reproduction is reported alongside
> the proposed tolerance. Do not begin any new model code (that starts at Milestone A / §7 Phase
> 1) in this session.

**Files/paths.** `src/data/`, `scripts/eval/reproduce_metadit_baseline.py`,
`checkpoints/phase0/REPORT.md`.

**Done criteria.**
- *Must pass to proceed*: shapes verified; all three released weight sets load; baseline
  reproduction numbers recorded; tolerance for "reproduced" explicitly proposed and confirmed by
  the human operator.
- *Nice to have*: automated shape/load assertions wired into a reusable test rather than a
  one-off script.

**Guardrails specific to this phase.** This phase exists specifically to prevent building on top
of an unverified foundation — the risk is proceeding past this gate on assumption rather than
confirmed reproduction. If reproduction fails or is ambiguous, stop; do not patch around it by
adjusting the model architecture, since nothing modeling-related has been built yet.

---

## Milestone A — Reproduce MetaDiT representation learning (§7 Phase 1, §7.1 "A")

**Goal.** Use the released MetaDiT encoders as-is to confirm geometry↔spectrum alignment works,
establishing pretrained initialization and a common embedding reference. Encoders are **not**
modified in this milestone.

**Preconditions / gate.** Phase 0 reproducibility gate passed and confirmed in
`checkpoints/phase0/REPORT.md`.

**Exact task prompt:**
> Following `docs/design_doc.md` §7 "Phase 1" and §1.3–§1.4 (spectrum encoder and contrastive
> pretraining description): using the verified MetaDiT weights from Phase 0, confirm the
> `G → z_G`, `S → z_S` mappings work as described, and confirm geometry-spectrum retrieval (i.e.
> that matching geometry/spectrum pairs are closer in the shared 512-dim embedding space than
> mismatched pairs, per §1.4's CLIP-style contrastive setup). Do not modify, fine-tune, or
> retrain any MetaDiT component in this milestone — it is reproduction only, exactly as the
> design doc specifies ("Encoders are not modified yet"). Do not begin implementing the context
> encoder, target encoder, or predictor (those start at Milestone B) even if it would be
> convenient to do so now.
>
> This is inference-only (frozen weights, no gradients), so it can run on the local machine per
> the Compute Environment section — no cloud session needed.
>
> Report retrieval accuracy (top-1/top-5, or whatever metric MetaDiT's own repo uses for this) in
> `checkpoints/milestone_a/REPORT.md`. If MetaDiT's repo does not expose a ready-made retrieval
> eval, implement the minimal one needed to check this, using only released frozen weights.

**Files/paths.** `src/encoders/spectrum_encoder.py` (thin wrapper around MetaDiT's released
weights only — no modification), `scripts/eval/reproduce_metadit_baseline.py` (extend from Phase
0), `checkpoints/milestone_a/`.

**Done criteria.**
- *Must pass to proceed*: geometry-spectrum retrieval confirmed working with released weights;
  results recorded.
- *Nice to have*: a small qualitative dump of a few retrieved geometry/spectrum pairs for sanity
  inspection.

**Guardrails specific to this phase.** Not directly exposed to a §13 failure mode yet (no new
model exists), but a subtle risk is drifting into building new components under the guise of
"just checking something" — Standing Rule 1 applies: this milestone is reproduction only.

---

## Milestone B — Vanilla deterministic JEPA (§7 Phase 2, §7.1 "B", §7.2)

**Goal.** Answer the central question before adding anything else: *"can a physical
goal-conditioned predictor infer the latent state of a complete structure from incomplete
structure, under a masking scheme that is not locally solvable?"* (§7, Phase 2). No geometry
decoder yet.

**Preconditions / gate.** Milestone A done. First training milestone — before starting, confirm
`CLOUD_TRAINING.md` has been read and a Kaggle or Colab session is available, per the Compute
Environment section; this milestone's actual training run does not happen on the local 4GB-VRAM
machine.

**Exact task prompt:**
> Following `docs/design_doc.md` §2 (mask topology — block masking, not random-pixel, per the
> explicit warning that uniform random masking is solvable by local interpolation and would
> silently invalidate the experiment), §3.1–§3.3 (geometry encoder, context encoder with Perceiver
> bottleneck, target encoder), §3.5 (predictor, GCLCT), and §7.2 (minimal first experiment):
>
> Implement the block-masking scheme exactly as specified in §2 (1–4 axis-aligned rectangular
> blocks in a 16×16 token grid at patch size 4, minimum side length 3 tokens, half of batches
> random-placed and half placed over resonance-relevant regions identified via the frozen EM
> surrogate's sensitivity map). Implement the geometry encoder (§3.1, initialized from MetaDiT's
> ViT per §1.1/§3.1), the context encoder with the Perceiver-IO-style 64-token bottleneck (§3.2),
> the target encoder in its EMA-JEPA variant only (§3.3 Variant A — do not implement LeJEPA yet,
> that is Milestone G), and the predictor (§3.5) at the sizes given in §11. Do NOT implement
> hierarchical prediction, classifier-free guidance, or sparse routing yet — those are also part
> of §3.5/§3.5.1/§3.5.2 but the design doc's own phase table (§4.1) says goal-dropout/guidance
> training only begins in Phase 4, and routing is exercised starting Phase 5. For this milestone,
> use dense (non-sparse) attention over all 16 goal tokens.
>
> Run exactly the minimal first experiment in §7.2: block-masked 50% context (blocks placed
> uniformly at random for this specific test), loss `L = L_J + 0.1·L_G`-only is NOT applicable
> here since there is no decoder yet — use `L = L_J` only (per §4.1's phase table: "Phase 2 — JEPA
> pretrain: L_JEPA, L_reg only"). Compare: (a) a direct masked generator baseline (no JEPA latent
> objective — this is Baseline 2, §10.1), (b) this JEPA model, (c) this JEPA model with the goal
> replaced by a null/zero token as a cheap proxy for goal-ignoring collapse (Failure Mode 2, §13).
>
> Also implement full training at block-mask ratios 20/40/60/80/100% (excluding 0% from the main
> analysis, per §7 Phase 2) once the minimal experiment passes. Begin logging goal-token
> utilization entropy (§20.2) from this milestone onward, as the design doc specifies.
>
> Write `scripts/train/train_milestone_b.py` as a standalone CLI script with a `--resume` flag and
> frequent checkpointing (Repo layout section, "Training scripts must be resumable"), since it
> will be run inside `notebooks/cloud_train_runner.ipynb` on Kaggle/Colab per `CLOUD_TRAINING.md`,
> not run to completion locally. You (the agent) may run a tiny local smoke test (batch size 1,
> a handful of steps, toy dims if needed) purely to confirm the script doesn't crash — this is not
> a substitute for the real run, which happens on cloud GPU and is reviewed by the human operator
> afterward via `checkpoints/milestone_b/REPORT.md`.
>
> Stop condition: if JEPA does not beat the direct masked baseline at 50% block masking, STOP and
> report this as a negative result per §7.2 — do not proceed to add more components to try to fix
> it within this milestone. This is a decision-relevant negative result requiring human review,
> not a bug to engineer around. If it does beat the baseline, run the full 20/40/60/80% sweep,
> report results, and stop — do not begin implementing the geometry decoder (Milestone C).

**Files/paths.** `src/data/` (block masking), `src/encoders/geometry_encoder.py`,
`src/encoders/context_encoder.py`, `src/encoders/target_encoder.py` (EMA variant only),
`src/predictor/gclct.py` (dense attention only in this milestone), `src/losses/jepa_loss.py`,
`src/diagnostics/goal_token_entropy.py`, `configs/milestone_b.yaml`,
`scripts/train/train_milestone_b.py`, `checkpoints/milestone_b/`.

**Done criteria.**
- *Must pass to proceed*: minimal 50% block-mask experiment run **on cloud GPU** and compared
  against direct masked baseline and null-goal proxy, per §7.2's explicit stop condition; if
  passed, full 20/40/60/80% sweep results recorded; goal-token entropy logging active and
  producing values; results reviewed by the human operator per the Compute Environment section's
  step-3 review requirement before Milestone C starts.
- *Nice to have*: qualitative visualization of which mask blocks are hardest to predict.

**Guardrails specific to this phase.**
- **Failure Mode 2 risk (§13, §9)**: predictor ignores the spectrum entirely,
  `P(Z_x, S) ≈ P(Z_x)`. The null-goal proxy comparison in §7.2 step 4 is the cheap early check;
  do not skip it even though the full InfoNCE/guidance machinery isn't built yet.
- **Mask-topology risk (§2, Ablation K)**: if block masking is implemented incorrectly and
  collapses toward locally-solvable behavior, every downstream milestone is invalidated. Confirm
  block masking is actually being used, not random-pixel masking, before trusting any result here.
- **"JEPA only wins from more parameters" risk (§13)**: when comparing against the direct masked
  baseline, match parameter count/compute as the design doc's failure-mode list requires — do not
  let the JEPA model simply be bigger.

---

## Milestone C — Geometry decoder (§7 Phase 3, §7.1 "C")

**Goal.** Attach the geometry decoder so predicted latents become realizable structures,
establishing latent prediction → actual structure without letting the EM surrogate dominate yet.

**Preconditions / gate.** Milestone B done: JEPA beat the direct baseline at 50% block masking
(if it didn't, this milestone should not start — see Milestone B's stop condition). Cloud session
available per `CLOUD_TRAINING.md`.

**Exact task prompt:**
> Following `docs/design_doc.md` §3.6 (geometry decoder) and §4.1 (loss curriculum for "Phase
> 3 — decoder"): implement the geometry decoder exactly as specified — context-aware,
> `Ĝ = D_G(Ẑ_y, G_x)`, with known pixels retained/enforced at inference and only missing pixels
> taking the decoder's output, output channels matching MetaDiT's three continuous parameters
> with the constraint layer described (bounded activation for `r_atom`/`h_atom`/`l_lattice`,
> binary/sigmoid treatment for occupancy if decoded explicitly). Introduce `L_G` (BCE/L1/L2 per
> §4, channel-type dependent), ramped from 0 to target over the first 20% of this phase's training
> — do not activate it at full weight from step one, per §4.1's explicit rationale ("let latent
> prediction stabilize before decoder gradients touch it").
>
> Do NOT introduce `L_S` or the EM surrogate loop yet — that is Milestone D / Phase 4. Do not
> begin implementing goal-dropout or sparse routing yet — those also belong to later milestones.
> Reuse the EMA-JEPA predictor and training setup from Milestone B unchanged except for the
> decoder addition and new loss term.
>
> During weight tuning, per §4.1's specific instruction: check that removing `L_G`'s target weight
> (i.e. ablation-style, keep it at 0) leaves latent-prediction accuracy (the Milestone B metric)
> unchanged beyond a small tolerance. The design doc calls for this check but does not give the
> tolerance number — propose one and confirm with the human operator before relying on it (Standing
> Rule 3).
>
> Extend `scripts/train/train_milestone_c.py` following the same CLI/resumable convention as
> Milestone B's script (`CLOUD_TRAINING.md`), for execution on Kaggle/Colab.
>
> Stop once the decoder produces valid geometries from predicted latents and `L_G` ramp-in is
> confirmed not to have disturbed `L_J` beyond the confirmed tolerance. Do not proceed to the
> physics loop (Milestone D) in this session.

**Files/paths.** `src/decoders/geometry_decoder.py`, `src/losses/geometry_loss.py`,
`src/losses/curriculum.py` (ramp schedule), `configs/milestone_c.yaml`,
`scripts/train/train_milestone_c.py`, `checkpoints/milestone_c/`.

**Done criteria.**
- *Must pass to proceed*: decoder implemented and producing physically valid (constraint-satisfying)
  geometry tensors; `L_G` ramp schedule implemented and active; confirmed the ramp-in doesn't
  disturb `L_J` beyond the human-confirmed tolerance; cloud run reviewed per the Compute
  Environment section.
- *Nice to have*: visual comparison of decoded geometry vs. ground truth for a few validation
  examples.

**Guardrails specific to this phase.**
- **Failure Mode 1 risk, early form (§13, §4.1)**: even before the physics loss exists, a decoder
  loss introduced too aggressively can already start pulling the latent toward being "whatever
  the decoder needs" rather than a genuine predictive target. The ramp schedule and the
  tolerance-check above exist specifically to catch this early, per §4.1's "catch weight drift
  while tuning, not only in the final ablation table."
- **"Decoder does all the work" risk (§5, §13)**: if the decoder is powerful enough to produce
  plausible structures from near-arbitrary latents, the latent objective loses meaning. This is
  formally checked later via probes (§12, H5) but keep it in mind now — don't over-invest decoder
  capacity beyond what §11 specifies without a documented reason.

---

## Milestone D — Physics loop (§7 Phase 4, §7.1 "D")

**Goal.** Close the loop through the frozen EM surrogate and demonstrate latent predictions
correspond to structures that actually approach the requested electromagnetic target.

**Preconditions / gate.** Milestone C done and its done criteria met. Cloud session available.

**Exact task prompt:**
> Following `docs/design_doc.md` §1.6 (forward EM surrogate), §4 (physics response loss `L_S` and
> latent physics alignment loss `L_A`), §4.1 ("Phase 4 — physics loop": augment, never dominate,
> `λ_JEPA` fixed at 1 throughout), and §3.5.1 (classifier-free goal guidance — begins in this
> phase per the phase table):
>
> Wrap the frozen MetaDiT EM surrogate (`Ĝ → F_EM → Ŝ`) as specified in §1.6, and introduce `L_S`
> (ramped per §4.1, not full-weight immediately) and `L_A` (using the pretrained MetaDiT
> geometry-spectrum aligned projectors, §4). Do not fine-tune the surrogate — it stays frozen, as
> §1.6 and the exact data-flow spec in §11.1 both specify.
>
> Also implement classifier-free goal guidance exactly as specified in §3.5.1: during training,
> replace `A_goal` with a learned null token `A_∅` with probability ~10%; at inference combine
> `Ẑ_guided = P(Z_x, A_∅) + w · [P(Z_x, A_goal) − P(Z_x, A_∅)]`. Do not implement sparse top-k
> routing yet (§3.5.2) — that belongs to Milestone E per the design doc's own phase table, even
> though both are part of §3.5's goal-sensitivity mechanisms.
>
> Guard explicitly against Failure Mode 1 (§13, §5): physics loss dominance turning the model into
> a direct generator with a decorative JEPA latent. Do this via the exact check §4.1 specifies:
> run Ablation D (remove `L_S` entirely) and confirm generation quality changes while
> latent-prediction accuracy (`L_J`, from Milestone B/C) stays within tolerance. Reuse the
> tolerance value confirmed in Milestone C if applicable, or propose/confirm a new one specific to
> this check if the human operator indicates it should differ.
>
> Note: adding the frozen surrogate into the training loop increases memory pressure further
> (§6 relative cost ~1.6×) — size batch/gradient-accumulation for the cloud GPU actually available
> (Kaggle T4/P100 16GB or Colab equivalent), not for the local machine, per the Compute
> Environment section.
>
> Stop once the physics loop is functioning, `L_S`/`L_A` are ramped in per the curriculum, CFG is
> implemented and its single-forward-pass diagnostic (`‖P(Z_x,A_goal) − P(Z_x,A_∅)‖`) is logged,
> and Ablation D has been run and reported. Do not begin goal-sensitivity InfoNCE training or
> sparse routing (Milestone E) in this session.

**Files/paths.** `src/surrogate/em_surrogate.py`, `src/losses/physics_loss.py`,
`src/losses/alignment_loss.py`, `src/predictor/guidance.py`, `configs/milestone_d.yaml`,
`scripts/train/train_milestone_d.py`, `scripts/eval/run_ablation.py` (Ablation D),
`checkpoints/milestone_d/`.

**Done criteria.**
- *Must pass to proceed*: physics loop functioning end-to-end (`Ĝ → F_EM → Ŝ`); `L_S`/`L_A`
  ramped per §4.1; CFG implemented with the guidance-gap diagnostic logging; Ablation D run and
  reported, with `L_J` degradation within the confirmed tolerance; cloud run reviewed.
- *Nice to have*: a first pass at the §20.3 normalized guidance-gap metric (full sweep across mask
  ratios is not required until Milestone F, but computing it once here is useful early signal).

**Guardrails specific to this phase.**
- **Failure Mode 1, full form (§13, §5, §4.1)**: this is the phase where physics-loss dominance is
  most likely to actually manifest, since decoder gradients can now flow back through `Ẑ_y`.
  Ablation D is the mandated check — do not skip it or defer it to the final ablation table.
- **Failure Mode 2 (§13, §9, §3.5.1)**: the guidance-gap diagnostic
  (`‖P(Z_x,A_goal) − P(Z_x,A_∅)‖` small and roughly constant across very different targets) is a
  direct, single-forward-pass symptom of goal-ignoring collapse. Check it, don't just implement
  the guidance mechanism and assume it works.

---

## Milestone E — Goal-sensitivity (§7 Phase 5, §7.1 "E")

**Goal.** Introduce counterfactual goal pairs and the InfoNCE goal-sensitivity loss, and test
whether the predictor actually uses the physical goal — including via sparse routing consistency.

**Preconditions / gate.** Milestone D done and its done criteria met. Cloud session available;
note the InfoNCE loss needs a real batch of in-batch negatives to be meaningful, so batch size on
the cloud GPU should be sized deliberately for this (record the chosen batch size and rationale
in the report) — do not run this milestone's core loss with a batch size so small the negatives
are trivial.

**Exact task prompt:**
> Following `docs/design_doc.md` §4 (goal-sensitivity InfoNCE loss `L_goal`), §7 Phase 5, §9
> (goal-sensitivity as a first-class safeguard — read the full §9 discussion of the Pendharkar
> paper's 2×2 controllable/relevant failure-mode taxonomy before implementing, since it explains
> *why* this milestone's checks are structured this way), §3.5.2 (sparse spectral-to-spatial
> routing), and §20.1 (externally-validated routing consistency):
>
> Implement counterfactual goal-pair sampling — same or similar context `G_c`, different goals
> `S_a, S_b` — and the InfoNCE loss `L_goal` over `(context, goal, target-latent)` triples using
> in-batch negatives, exactly as specified in §4. Activate it per §4.1's phase table (added in
> Phase 5, on top of everything already active). Now implement sparse top-k routing (§3.5.2): gate
> scores per masked query against the 16 goal tokens, top-k (k=2 or 3) used in cross-attention
> instead of dense softmax over all 16. Replace Milestone D's dense-attention predictor path with
> this sparse-routed version.
>
> Implement both routing-consistency checks: (a) the internal check from H6/Ablation J — does
> region A consistently route to the same frequency band across independently sampled geometries
> with similar goals; and (b) the externally-validated check from §20.1 — compute the frozen EM
> surrogate's input-output Jacobian `J_i = ∂S(f_i)/∂G` per training geometry, aggregate per 16×16
> spatial block, and correlate the learned top-k routing assignment against this Jacobian-derived
> ground-truth map via rank correlation. Per §20.1, this is "nearly free" (a handful of backward
> passes through the already-frozen, already-differentiable surrogate) — do not skip it in favor
> of the internal-only check, since the design doc is explicit that internal consistency alone can
> be "stable and still wrong."
>
> Do not begin the zero-context/context-curriculum work (Milestone F) yet, even though goal
> sensitivity and zero-context discovery are conceptually related.

**Files/paths.** `src/losses/goal_infonce.py`, `src/predictor/routing.py`,
`src/diagnostics/routing_jacobian.py`, `configs/milestone_e.yaml`,
`scripts/train/train_milestone_e.py`, `scripts/diagnostics/run_routing_jacobian_check.py`,
`checkpoints/milestone_e/`.

**Done criteria.**
- *Must pass to proceed*: `L_goal` implemented and active with a deliberately sized batch;
  sparse top-k routing implemented and replacing dense attention; both the internal (H6) and
  externally-validated (§20.1) routing consistency checks run and reported, with rank-correlation
  numbers between routing assignment and Jacobian-derived ground truth; cloud run reviewed.
- *Nice to have*: top-1/top-5 goal-to-latent retrieval accuracy (a free byproduct of the InfoNCE
  loss, per §4).

**Guardrails specific to this phase.**
- **Failure Mode 2, precise form (§9)**: this is the phase the design doc identifies as the
  direct, targeted fix for the "exogenous but control-relevant feature discarding" failure mode
  documented in the Pendharkar paper (arXiv:2606.30068) — cite this specific 2×2-cell match if
  writing up results, per §9's instruction.
- **Self-validating-routing trap (§20.1)**: a routing table can be internally stable yet
  physically wrong (e.g., a positional shortcut). Do not report H6/Ablation J results based on the
  internal check alone — the Jacobian correlation is required for this milestone's done criteria,
  not optional.

---

## Milestone F — Zero-context curriculum (§7 Phases 6–7, §7.1 "F", §7.3)

**Goal.** Train across the full context-availability continuum (100%→0% visible geometry) via
curriculum, then run the true zero-context, physics-only discovery experiment.

**Preconditions / gate.** Milestone E done and its done criteria met. Cloud session available.

**Exact task prompt:**
> Following `docs/design_doc.md` §7 Phase 6 ("context curriculum") and Phase 7 ("zero-context
> generation"), §7.3 (minimal zero-context experiment), §8 and §8.1 (the four-case unified
> interpretation and the limitation of zero-context generation), and §20.3 (normalized guidance
> gap across the curriculum):
>
> Implement the curriculum exactly as staged in §7 Phase 6: early training on 100/75/50% visible
> context, middle on 75/50/25%, late on 50/25/0%, retaining some earlier-regime examples
> throughout to prevent catastrophic specialization (do not do an abrupt jump straight to
> `P(M=100%)` training as in §7.3's "minimal" version until the fuller curriculum is in place —
> §7.3 describes the minimal first probe, §7 Phase 6 describes the full curriculum; implement the
> minimal probe first as a fast sanity check, then the full curriculum).
>
> Run the §7.3 minimal zero-context experiment as the first checkpoint: `G_c = ∅` at inference,
> generate 32 structures for one target spectrum, evaluate all 32 with the released surrogate,
> report mean error, best error, diversity, uniqueness, validity.
>
> Then run the full Phase 7 zero-context evaluation across the curriculum-trained model. Compute
> and report the §20.3 normalized guidance-gap curve
> (`‖P(Z_x,A_goal) − P(Z_x,A_∅)‖ / σ(Z_x)`) as a single plot across 20/40/60/80/100% masking
> buckets — this is explicitly called out as a required report for this phase, not an optional
> extra.
>
> Per §8.1, do not claim "novel geometry" from zero-context generation merely because outputs look
> visually different — measure it via nearest-neighbour structural distance, topology statistics,
> latent distance from the training set, component density, and spectral-target novelty, as
> specified.
>
> Do not begin the LeJEPA ablation (Milestone G) or stochastic latent work (Milestone H) yet — per
> Standing Rule 5 and §7.1's explicit sequencing instruction, Milestones A–F must fully succeed
> first.

**Files/paths.** `src/data/` (curriculum sampler extension), `src/diagnostics/guidance_gap.py`,
`configs/milestone_f.yaml`, `scripts/train/train_milestone_f.py`,
`scripts/diagnostics/run_guidance_gap_sweep.py`, `scripts/eval/eval_hypotheses.py` (novelty
metrics from §8.1), `checkpoints/milestone_f/`.

**Done criteria.**
- *Must pass to proceed*: full curriculum trained without catastrophic forgetting of higher-context
  regimes; §7.3 minimal zero-context experiment run and reported (mean/best error, diversity,
  uniqueness, validity for 32 samples); full Phase 7 zero-context evaluation run; §20.3 normalized
  guidance-gap curve produced across the full curriculum; §8.1 novelty metrics computed (not just
  visual inspection); cloud run reviewed.
- *Nice to have*: qualitative gallery of zero-context generations across a spread of target
  spectra.

**Guardrails specific to this phase.**
- **"Zero-context generation collapses to trivial/memorized shapes" (§13)**: this is the primary
  risk this milestone is built to detect. The §8.1 novelty metrics and diversity/uniqueness
  numbers from §7.3 exist specifically to catch this — do not report zero-context success based on
  spectral error alone.
- **Failure Mode 2 at the extreme**: at 0% context, the goal is the *only* input; if the guidance
  gap collapses toward zero specifically at low-context buckets, this is a severe, not minor,
  finding — flag it prominently rather than averaging it away in a single aggregate number.

---

## Milestone G — LeJEPA ablation (§7.1 "G", §3.3 Variant B, Ablation E)

**Goal.** Replace the EMA target-encoder mechanism with LeJEPA's SIGReg regularization,
empirically (not philosophically) deciding between the two per Ablation E (§10.2).

**Preconditions / gate.** Milestone F done and its done criteria met. Per Standing Rule 5, confirm
with the human operator that spending the additional compute (§6: ~1.8× the Phase-2 baseline unit
territory, on top of everything already trained) is warranted now rather than deferred, and that
Kaggle/Colab quota (weekly GPU-hour caps per `CLOUD_TRAINING.md`) can absorb it.

**Exact task prompt:**
> Following `docs/design_doc.md` §3.3 (Variant B — LeJEPA) and Ablation E (§10.2): clone the
> LeJEPA reference implementation from the exact URL in §18,
> `https://github.com/galilai-group/lejepa`, and re-verify (per Standing Rule 4) that its current
> structure matches what §3.3 describes before integrating anything from it. Implement SIGReg-style
> distribution regularization on the embeddings directly, removing the EMA/teacher copy for this
> variant. Per §3.3's precision note, treat the specific SIGReg test choice (e.g. Epps–Pulley) and
> the `num_slices`/`num_points` hyperparameters as reportable choices, not implicit defaults — record
> them explicitly in `checkpoints/milestone_g/REPORT.md`.
>
> Train this variant using the same architecture as the current best EMA-JEPA model from
> Milestone F (same predictor, decoder, losses — only the target-encoder mechanism changes), and
> run Ablation E: compare EMA-JEPA vs. LeJEPA on the same evaluation suite used in Milestones B–F
> (latent prediction quality, geometry/physics accuracy, at minimum). Decide the outcome
> empirically per the design doc's explicit instruction — do not favor one on stated-preference
> grounds.
>
> Do not begin stochastic latent work (Milestone H) in this session, even if LeJEPA training
> finishes quickly.

**Files/paths.** `external/lejepa/` (cloned reference), `src/encoders/target_encoder.py` (extend
with LeJEPA variant, do not fork into a separate file unless the interface genuinely diverges),
`src/losses/sigreg.py`, `configs/milestone_g.yaml`, `scripts/train/train_milestone_g.py`,
`checkpoints/milestone_g/`.

**Done criteria.**
- *Must pass to proceed (for downstream milestones that might use LeJEPA)*: Ablation E comparison
  run and reported with a clear empirical recommendation; SIGReg hyperparameters documented.
- Note: Milestones H and I do not strictly require LeJEPA to "win" — they can proceed with
  whichever target-encoder variant Ablation E supports, or continue with EMA-JEPA if LeJEPA shows
  no advantage. This milestone's done criterion is that the comparison exists and is decisive
  enough to make that call, not that LeJEPA must outperform.

**Guardrails specific to this phase.** No new §13 failure mode is specific to this milestone
beyond the general "gains disappear when parameter count and compute are matched" caution — match
compute between the EMA and LeJEPA variants in Ablation E, per §13's failure-mode list, or the
comparison is not valid.

---

## Milestone H — Stochastic latent (§7.1 "H", §7.4)

**Goal.** Support one-to-many inverse design by making `P(Z_y | Z_x, S)` a distribution rather
than a deterministic map, testing whether multimodal solution coverage actually improves (H4,
§12).

**Preconditions / gate.** Milestone F done (Milestone G's outcome determines which target-encoder
variant to build this on top of, per its done criteria). Cloud session available; K=32 sampling
for evaluation adds memory/compute overhead (§6: ~K× at eval) — size the eval run for the cloud
GPU actually available.

**Exact task prompt:**
> Following `docs/design_doc.md` §7.4 (stochastic latent interface) and Ablation G (§10.2, "How
> much multimodality is actually gained?"): implement the **one-shot** stochastic form first, as
> explicitly instructed — `Ẑ_y = P(Z_x, S, ε)`, with `ε ~ N(0, I)`, `d_z = 64`, injected through a
> learned pathway `c_z = f_z(ε)` into the predictor's masked-query blocks via
> `AdaLN(x; c_physics, c_z)` — not simply concatenated to the spectrum vector, and not latent
> diffusion. §7.4 is explicit that this one-shot form must be tried and evaluated before any more
> expressive generative path is considered.
>
> Evaluate exactly as H4 (§12) specifies: for a single target `S*`, generate `G_1..G_K`; report
> average and best-of-K spectral accuracy, geometry diversity, latent diversity, and pairwise
> structural distance. A good model should not simply produce the same structure `K` times — check
> this explicitly, don't just report averages that could hide mode collapse.
>
> Stop after this one-shot stochastic model is trained and evaluated. Do NOT implement the
> flow-matching extension (Milestone I) in this session — per §7.4 and §7.1, that is only justified
> "if this fails to capture multimodality," which is a judgment this milestone's own H4 results
> must first establish.

**Files/paths.** `src/stochastic/latent_seed.py`, `configs/milestone_h.yaml`,
`scripts/train/train_milestone_h.py`, `scripts/eval/eval_hypotheses.py` (extend with H4 metrics),
`checkpoints/milestone_h/`.

**Done criteria.**
- *Must pass to proceed (to consider Milestone I at all)*: one-shot stochastic model trained; H4
  metrics (avg/best-of-K spectral accuracy, geometry diversity, latent diversity, pairwise
  structural distance) reported; an explicit determination of whether mode coverage is adequate or
  insufficient, since this determination is the actual gate for whether Milestone I is justified.
- *Nice to have*: qualitative K-sample gallery for a few representative target spectra.

**Guardrails specific to this phase.**
- **"Stochastic samples fail to improve mode coverage" (§13)**: this is the specific failure this
  milestone must check for, not assume away. If `K` samples collapse to near-identical structures,
  this is itself a valid and reportable outcome (per §13's framing that a negative result here is
  scientifically valid), not a bug to silently patch by, e.g., increasing `d_z` without
  justification (Standing Rule 2).

---

## Milestone I — Latent flow/diffusion (§7.1 "I") — conditional, may not be needed

**Goal.** Only if Milestone H's one-shot stochastic model demonstrably fails to capture
multimodality: replace the one-shot predictor with a flow-matching path in latent space,
preserving the principle that generation happens in latent structural state space, not raw
geometry space.

**Preconditions / gate.** Milestone H complete, **and** its H4 evaluation shows a clear,
documented multimodality shortfall. Per §7.4: *"Only if this fails to capture multimodality
should a continuous latent generation path be introduced."* If Milestone H's results are
ambiguous rather than a clear failure, stop and get explicit human confirmation that this
milestone should proceed at all — do not treat "could be better" as sufficient justification,
per Standing Rule 2. Also re-confirm the §6 compute-budget note that this milestone costs
roughly 3–5× (multi-step sampling) — get this explicitly signed off given Standing Rule 5, and
confirm Kaggle/Colab quota can absorb it.

**Exact task prompt:**
> Only proceed with this prompt if Milestone H's `checkpoints/milestone_h/REPORT.md` documents a
> clear multimodality shortfall and a human operator has confirmed this milestone should proceed
> despite the ~3–5× compute cost noted in `docs/design_doc.md` §6.
>
> Following §7.4: implement a flow `z_t` from noise to target latent, conditioned on `(Z_x, A)`,
> learning `v_θ(z_t, t | Z_x, A)`. This replaces the one-shot predictor's stochastic path from
> Milestone H, not the deterministic backbone from Milestones B–F — keep the underlying context
> encoder, spectrum encoder, and decoder unchanged; only the mechanism generating `Ẑ_y` from
> `(Z_x, A, noise)` changes. Preserve the principle stated in §7.4 and §16: generation happens in
> latent structural state space, not raw geometry space — do not let this become diffusion over
> geometry directly, which would collapse back toward MetaDiT's own approach and undercut this
> project's central claim (§1.7, §0).
>
> Re-run the same H4 multimodality evaluation used in Milestone H, on the same target spectra, to
> get a like-for-like comparison against the one-shot stochastic model. Report whether the
> additional compute cost is justified by the multimodality gain.

**Files/paths.** `src/stochastic/flow.py`, `configs/milestone_i.yaml`,
`scripts/train/train_milestone_i.py`, `checkpoints/milestone_i/`.

**Done criteria.**
- *Must pass to proceed (to a final writeup)*: flow-matching path implemented over the latent
  space only (not geometry space); H4 evaluation re-run and directly compared against Milestone
  H's one-shot results; explicit conclusion on whether the added complexity/compute was justified.
- *Nice to have*: none beyond the above — this is the terminal milestone in §7.1's sequence.

**Guardrails specific to this phase.**
- **Scope creep back toward MetaDiT (§1.7, §0)**: the single biggest risk here is implementing
  diffusion over raw geometry (which is what MetaDiT already does) instead of latent
  flow-matching — this would directly undercut the project's stated core claim. Confirm the flow
  operates on `Z`, not `G`, before training.
- **Unjustified compute spend (§6, Standing Rule 5)**: this milestone is explicitly conditional;
  do not let it get built "just in case" ahead of a documented Milestone H shortfall.

---

## If something fails

If any milestone's "must pass to proceed" criteria are not met:

1. **Stop. Do not proceed to the next milestone**, and do not start adding new mechanisms to the
   current milestone in an attempt to force the criteria to pass, unless a specific failure mode
   below clearly motivates a specific, scoped fix.
2. **Consult §13 ("Failure modes") of the design doc first.** It enumerates the conditions under
   which this project should be considered a failure (MetaDiT's baseline is consistently better;
   JEPA only helps due to more parameters; physics loss overwhelms latent prediction; predictor
   ignores the spectrum; zero-context collapses to memorized shapes; stochastic samples don't
   improve coverage; latent probes show no organizational advantage; gains vanish when
   compute/parameters are matched) and explicitly states: *"If that happens, the correct and
   scientifically valuable conclusion is that JEPA is unnecessary for this task."* A clean
   negative result, correctly diagnosed and reported, is a valid outcome — not something to
   engineer around by adding complexity.
3. **Check whether the failure maps to a specific guardrail already named in the relevant
   milestone section above.** Each milestone above lists which §13 failure mode(s) it's exposed
   to and which check catches it — start there rather than re-deriving the diagnosis from scratch.
4. **Record the failure and diagnosis in that milestone's `checkpoints/<milestone>/REPORT.md`**,
   including which specific check failed and the observed numbers, before deciding whether to:
   (a) diagnose and retry within the same milestone's scope, (b) escalate to the human operator
   for a scope/threshold decision, or (c) stop the project line entirely if §13's criteria are
   met.
5. **Never respond to a failed gate by silently loosening the gate's own criteria.** If a
   threshold that was proposed and confirmed per Standing Rule 3 turns out to be wrong, that is
   itself a decision requiring a new round of human confirmation — not a unilateral adjustment.
6. **If the failure looks like a compute/environment problem rather than a research result**
   (e.g. out-of-memory on Kaggle/Colab, session disconnect losing progress, dataset staging
   failure), consult `CLOUD_TRAINING.md` first — this is an operational failure, not a §13
   research failure mode, and should be diagnosed as such (resume from checkpoint, reduce batch
   size, switch Kaggle↔Colab) rather than triggering a research-conclusion discussion.
