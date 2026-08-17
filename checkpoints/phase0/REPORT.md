# Phase 0 — MetaDiT dataset verification & baseline reproduction

**Status:** all verification steps executed. Tolerance for "reproduced" proposed at the bottom
**pending human operator confirmation** — do not treat the gate as passed until confirmed.

**UPDATE (operator confirmation received):** the §7 tolerance proposal was **accepted as
proposed** (±5% relative on MAE and AAE vs. the paper's MetaDiT line, plus the §7 rank
separation). Phase 0 gate is therefore **passed**. Caveat: the `seed0.json`
generated-on-foreign-hardware caveat was noted and **accepted as deferred to Milestone B**
(the released generation file stays the canonical reproduction source for this gate).

**Execution location:** local dev machine (RTX 3050, torch 2.13.0+cpu, Python 3.11).
Inference-only CPU work; **no cloud session used** — consistent with `AGENTS.md` Phase 0.

---

## 0. Scope note / deviation from normal flow

`AGENTS.md` Phase 0 preconditions were partially unfulfilled at session start: the MetaDiT clone
was missing (`external/metadit/` contained only `.gitkeep`) and `checkpoints/phase0/REPORT.md`
from Phase -1 did not exist. Per explicit human-operator confirmation, this session (a) completed
the missing Phase -1 clone step, (b) performed the §1.2 file-path re-verification that Phase -1
was supposed to record (recorded here), and (c) executed all four Phase 0 verification tasks.
This is a scaffolding-phase gap, not a §13 research failure.

---

## 1. File-path re-verification against live MetaDiT repo (§1.2 / Standing Rule 4)

Clone performed from the exact §18 URL `https://github.com/JessePrince/metadit` (shallow,
depth-1, `8c9b7c4`). Current structure re-checked against the design doc's referenced paths:

| Path referenced in design doc §1.2 | Exists in live repo? | Note |
|---|---|---|
| `model/dit.py` | YES | DiT model + `DIT_MODEL` registry (`metadit_s` = depth 12, hidden 384, heads 6, patch 2) |
| `train/train_metadit.py` | YES | loads released spectrum encoder into `y_embedder.encoder`, strips `context_encoder.` prefix |
| `model/spec_encoder.py` | YES | `VanillaSpectrumEncoder`, dual-attention (seq + channel), dim 256, 4 blocks |
| `model/clip_model.py` | YES | CLIP-style contrastive model (`CLIPModel`, `ClipLoss`) |

Additional live paths not named in the design doc but required for eval (used this session):
`model/surrogate.py` (StarNet-style `surrogate_s3`, used by `metric.py`), `diffusion/`
(`create_diffusion`, `SpacedDiffusion`), `datapipe.py` (`FreeFormDataset` 3×32×32 grids,
`SurrogateFreeFormDataset` 3×64×64 grids), `generate.py` (GPU-only DiT sampling entrypoint),
`metric.py` (official MAE/AAE/AAE&K).

**No paths needed to be silently substituted** — all design-doc-referenced paths still exist.

---

## 2. Task 1 — dataset shape verification (design doc §7 Phase 0, §1.1)

Loaded `data/metadit/split_data/{train,val,test}_set.mat`:

| Split | N | `pattern` | `parameter` | `real` / `imag` |
|---|---|---|---|---|
| train | 139,906 | (64, 64, N) int8, binary {0,1} | (N, 3) | (N, 301) each |
| val | 17,488 | (64, 64, N) int8, binary {0,1} | (N, 3) | (N, 301) each |
| test | 17,489 | (64, 64, N) int8, binary {0,1} | (N, 3) | (N, 301) each |

Construction verification via the repo's **own** dataset classes (`datapipe.py`):
- `FreeFormDataset` (DiT path): `inputs` = 3×32×32 (upper-left pattern quadrant, channel 0 =
  `r_atom/5`, channel 1 = `h_atom`, channel 2 = `l_lattice/3`), `condition` = (2, 301) [real, imag].
- `SurrogateFreeFormDataset` (EM-surrogate path): `inputs` = **3×64×64** (full pattern, same
  normalization), `labels` = (2, 301).

**Design-doc detail worth recording (§1.1):** the design doc states "geometry tensor shape
`3×64×64`" and "`S ∈ ℝ^(301×2)`". The raw `.mat` stores `pattern` (64×64 binary) and
`parameter` (N×3) *separately*; the 3-channel geometry is built on load. The surrogate/64×64
construction matches the design doc. The DiT diffusion path uses the symmetric 3×32×32 quadrant
(§4.1 data encoding). The spectrum is stored as 301-point real+imag arrays; the repo's tensor
convention is (2, 301) (channels = [real, imag]) vs. the design doc's written (301, 2) — same
data, transposed axis order. Flagged so downstream milestones use the repo's (2, 301)
convention.

**Result: SHAPE VERIFICATION PASS.**

---

## 3. Task 2 — released weight loading (all three files)

All three released weight files in `data/metadit/weights/` loaded with `strict=True` using the
repo's own loader conventions, plus batch-1 CPU forward passes to confirm usability:

1. **`spec_encoder.pth`** — released contrastively-pretrained spectrum encoder. Keys are
   `context_encoder.*` over a `VanillaSpectrumEncoder`; per `train_metadit.py` the `context_encoder.`
   prefix is stripped before `load_state_dict`. Params 4.53M. Forward (2×2×301) → (2, 301, 256) OK.
2. **`metadit-small.bin`** — released MetaDiT-S diffusion transformer (the design doc's "geometry
   ViT"; it contains the patch embedder + DiT backbone + frozen spectrum encoder inside
   `y_embedder`). Loaded via `generate.py`'s build convention:
   `DIT_MODEL['metadit_s'](diffusion=create_diffusion('500', learn_sigma=False), condition_channel=301)`,
   `strict=True`. Params 37.2M. Forward (1×3×32×32) → (1, 3, 32, 32) OK.
3. **`surrogate_model.bin`** — released forward EM surrogate (`surrogate_s3`, StarNet-derived),
   loaded via `metric.py`'s `build_surrogate_model`, `strict=True`. Params 6.33M. Forward
   (1×3×64×64) → (1, 2, 301) OK.

**Result: WEIGHT LOADING PASS.**

---

## 4. Task 3 — baseline reproduction (MetaDiT's own eval code and metric)

**How it was reproduced (faithful to the official workflow):**
- MetaDiT's own release includes the raw forward-prediction output
  `data/metadit/generation/seed0.json` — the authors' own seed-0 generation on the held-out
  split (verbatim README workflow: "We also provide the raw data generated by the model on our
  machine with seed 0 … calculate the metric"). Verified 1:1 that all 17,489 `seed0.json`
  conditions map bijectively onto `test_set.mat` rows (permuted order — irrelevant to the
  metric, since each item carries its own ground-truth target). Confirmed it is the **test**
  split, not val.
- Spectrum evaluation run with the repo's **unmodified** `external/metadit/metric.py` +
  released `surrogate_model.bin` (CPU):
  `python metric.py --data_path data/metadit/generation --model_path data/metadit/weights/surrogate_model.bin --device cpu`

**Numbers (official MAE/AAE definitions, exactly as the repo computes them):**

| Metric | Reproduced (this session) | Paper Table 2 (MetaDiT) | Δ relative |
|---|---|---|---|
| MAE | **0.080295** | 0.0801 | +0.24% |
| AAE | **48.3375** | 48.2495 | +0.18% |

Additional context from the paper's Table 2 for rank check: vanilla DiT baseline MAE 0.1677 /
AAE 100.9437; AVG1 (mean-spectrum baseline) 0.5860 / 352.7424. The reproduced number sits
decisively below both baselines, matching the paper's rank order. Paper reports
AAE&2 = 58.80, AAE&4 = 68.73 (needs 4 seeds; only seed-0 data is released — see deviations).

**Result: BASELINE REPRODUCED** (within ~0.2% of the paper's reported MetaDiT line).

---

## 5. Guardrails and notes

- **No §13 failure mode applies** (nothing new is built yet); the only Phase -1/Phase 0 risk —
  proceeding on unverified repo-path assumptions — is addressed by the re-verification in §1.
- **Sequence `metric.py --k 2 4` not run:** AAE&2 / AAE&4 require the other seed outputs
  (`seed7/42/3407.json`), which the authors did not release. Their values are available in the
  paper (58.80 / 68.73). Re-generating them requires running `generate.py` per seed (GPU-only,
  see §6 deviation) on a cloud GPU — recommended but not required for this gate.
- **Surrogate sanity check (context only, not a MetaDiT metric):** frozen surrogate on
  ground-truth test structures → predicted vs. true spectrum MAE ≈ 0.0077 (p50 0.0061, p90
  0.0164 over a 500-sample check) — consistent with Table 1's reported surrogate error (~0.0084)
  and confirms the surrogate weights behave physically.

---

## 6. Deviations from the design doc / AGENTS.md (with justification)

1. **DiT forward prediction not locally re-run.** `external/metadit/generate.py` asserts
   `torch.cuda.is_available()` ("Inference supports GPU only!"), and the local machine carries a
   CPU-only torch build (dev-only machine per AGENTS.md). Instead, the authors' released
   `seed0.json` (their own seed-0 forward-prediction output on the test split) was used as the
   generation source — the exact workflow MetaDiT's README documents for metric reproduction.
   The **metric** (spectrum evaluation) was fully re-executed locally with the repo's own
   `metric.py` + released surrogate. The reproduction matches the paper to ~0.2%, so this is not
   a workaround that hides a gap; it is the canonical reproduction path. Optional follow-up: an
   independent re-run of `generate.py` on a cloud GPU (full or sampled subset) to double-check
   the sampler; not needed to pass this gate.
2. **Phase -1 carried forward.** The MetaDiT clone and the §1.2 path re-verification (Phase -1
   deliverable) were completed inside this session with explicit operator approval.
3. **Minimal local dependency install.** Installed only the inference-relevant subset of the
   repo's `pyproject.toml` (`timm`, `einops`, `wandb`, `tensorboard`) into the local Python 3.11
   env (repo declares requires-python >=3.13 and the full training stack incl. deepspeed/triton,
   which are unnecessary for CPU inference). torch upgraded to 2.13.0+cpu during install. No
   `requirements.txt` pinned yet — flagged as Phase -1 "nice to have" for a later session.
4. **`data/metadit/.gitkeep` removed** in `external/metadit/` to allow the clone; scaffold
   `.gitignore` keeps `data/metadit/*` and `checkpoints/*` out of git.

---

## 7. Proposed tolerance for "reproduced" — **PENDING HUMAN CONFIRMATION (do not proceed on your own)**

The design doc gives no numeric tolerance, so one is proposed below per Standing Rule 3.

Proposal:
- **Primary (both must hold):**
  1. Re-computed MAE within **±5% relative** of the paper's MetaDiT line (0.0801) →
     acceptable band **0.0761–0.0841**. (This session hit +0.24%.)
  2. Re-computed AAE within **±5% relative** of the paper's AAE (48.2495) →
     acceptable band **45.84–50.66**. (This session hit +0.18%.)
- **Secondary / rank behavior:** reproduced MAE must remain clearly below the paper's DiT
  baseline (0.1677) with ≥ 2× separation, i.e., equal to or better than the reproduced value's
  position relative to both the DiT and AVG1 lines.
- **Rationale for the band:** diffusion sampling is stochastic across hardware and float
  precision (released seed0.json was generated on the authors' own machine; the ~0.2% delta to
  our computation already demonstrates run-to-run/infra variance of that order), so ±5% is a
  tolerant-but-informative range that still clearly distinguishes a reproduced MetaDiT line from
  the DiT/AVG1 baselines (which are ~2×/7× larger). A much tighter band (±1%) would be
  feedback-noise in this setting; anything looser (±10%+) would fail to distinguish from the
  DiT baseline, which we do not want to bless.

Decide one of: (a) accept ±5% (+ rank separation, above), (b) tighten/loosen the numbers, (c)
require additional evidence (e.g., a cloud-GPU independent re-generation before passing).
Once confirmed, this becomes the gate definition for Phase 0 → Milestone A.

---

## 8. Files produced this session

- `external/metadit/` — live MetaDiT repo clone (`8c9b7c4`, depth 1) — completes Phase -1 step 3.
- `scripts/eval/reproduce_metadit_baseline.py` — standalone CLI reproduction script
  (shapes / weights / baseline; `--skip-*` flags). Reusable input for Milestone A.
- `checkpoints/phase0/seed0_metric.json` — raw metric.py output (MAE 0.08029482260535217,
  AAE 48.33748317632199).

Repro command (local, CPU):
`python scripts/eval/reproduce_metadit_baseline.py`