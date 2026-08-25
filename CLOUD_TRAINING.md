# CLOUD_TRAINING.md — Kaggle / Colab Training Runbook (Phase 2 Updated)

This is the **canonical** cloud-training workflow referenced by `AGENTS.md`. Any milestone whose
task prompt involves gradient-based training points here instead of re-deriving its own cloud
setup. Local machine (RTX 3050, 4GB VRAM / 16GB RAM) is dev-only — see `AGENTS.md`'s
"Compute environment" section for why.

Every `scripts/train/train_milestone_<x>.py` is written as a standalone CLI script
(`python scripts/train/train_milestone_<x>.py --config configs/milestone_<x>.yaml [--resume
<path>]`) specifically so it can be invoked identically from either platform below with no
notebook-specific glue code.

---

## 0. One-time setup (do this once per platform, not per milestone)

### GitHub repo
Push your local repo to GitHub (private is fine, both platforms support private repo access via
a personal access token).

```bash
git init
git remote add origin https://github.com/<you>/<repo>.git
git add .
git commit -m "initial scaffold"
git push -u origin main
```

### Dataset staging
The MetaDiT dataset/weights come from the exact URL in the design doc §18:
`https://huggingface.co/datasets/Hao-Li-131/MetaDiT-AAAI2026`. Don't re-download this every
session on either platform — stage it once:

- **Kaggle**: create a private Kaggle Dataset from the downloaded `data/metadit/` folder
  (kaggle.com → "New Dataset" → upload), then attach it as a notebook input
  (`/kaggle/input/<dataset-name>/`).
- **Colab**: upload `data/metadit/` to a fixed Google Drive folder once
  (`/content/drive/MyDrive/<project>/data/metadit/`), and mount Drive each session instead of
  re-downloading.

### Dependency contract
The repository pins an explicitly tested PyTorch/Torchvision combination in `requirements.txt`:
- **PyTorch 2.5.1**
- **Torchvision 0.20.1**

Do not change these without re-running the full preflight suite. The preflight script
`scripts/preflight/milestone_b_preflight.py` will fail if the environment does not match.

---

## 1. Kaggle workflow

**Per-session steps** (repeat each time you start a new Kaggle session for a milestone):

1. Open/create a Notebook, attach GPU accelerator (Settings → Accelerator → GPU T4 x2 or P100).
   Turn Internet **ON** if cloning from GitHub or installing packages.
2. Attach the staged MetaDiT dataset as a notebook input (Add Data → your dataset).
3. Clone the repo and install deps:
   ```python
   !git clone https://github.com/<you>/<repo>.git
   %cd repo
   !pip install -r requirements.txt
   ```
4. **Run preflight check** (mandatory before any training):
   ```python
   !python scripts/preflight/milestone_b_preflight.py --config configs/milestone_b.yaml
   ```
   This verifies: environment contract, git state, dataset, model/objective, tiny training,
   validation, physics controls, checkpoint save/load/resume, and config validation.
   **Exit code 1 = DO NOT START TRAINING.**
5. **Run preflight to discover and link dataset** (replaces manual symlink):
    ```python
    !python scripts/preflight/milestone_b_preflight.py --config configs/milestone_b.yaml
    ```
    The preflight will auto-discover the dataset location and create the `data/metadit` symlink.
6. Confirm GPU:
   ```python
   !nvidia-smi
   ```
7. Run the milestone's training script (resume if a checkpoint already exists from a prior
   session):
   ```python
   !python scripts/train/train_milestone_b.py \
       --config configs/milestone_b.yaml \
       --resume /kaggle/working/checkpoints/milestone_b/latest.pt   # omit if starting fresh
   ```
8. Checkpoint into `/kaggle/working/checkpoints/...` throughout the run (the training script
   handles this per its `--resume`-compatible checkpointing, per `AGENTS.md`'s "Training scripts
   must be resumable" requirement).
9. **Configure persistent checkpoint storage** (mandatory per Phase 2 §5):
   - Before training, ensure `checkpoints/` is symlinked to persistent storage
   - At every checkpoint: save → verify exists → verify loadable
   - After training: checkpoint provenance validation + checkpoint integrity validation
   - If persistent storage is not configured: **ABORT BEFORE TRAINING**
10. **Before the session ends** (Kaggle sessions cap at ~9–12 hours, quota ~30 GPU-hrs/week):
    - "Save Version" → "Save & Run All" to snapshot `/kaggle/working/` so it isn't lost, **or**
    - push results directly back to GitHub from within the notebook:
      ```python
      %cd repo
      !git config user.email "you@example.com"
      !git config user.name "you"
      !git add checkpoints/milestone_b/
      !git commit -m "milestone B: cloud training run, see REPORT.md"
      !git push
      ```
      (requires a GitHub personal access token set as a Kaggle Secret, referenced via
      `!git remote set-url origin https://<token>@github.com/<you>/<repo>.git`)

**Resuming after a session ends/quota resets:** start a new session, repeat steps 1–5, then pass
`--resume` pointing at the last checkpoint pulled from GitHub (or restored from a saved notebook
version).

---

## 2. Colab workflow

**Per-session steps:**

1. Open a new Colab notebook, Runtime → Change runtime type → GPU (T4 free tier; A100/more
   reliable T4 on Colab Pro).
2. Mount Drive (this is where checkpoints/data persist across sessions, since Colab's local disk
   is wiped each session):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Clone the repo (into local Colab disk — code doesn't need to persist on Drive, only
   data/checkpoints do):
   ```python
   !git clone https://github.com/<you>/<repo>.git
   %cd repo
   !pip install -r requirements.txt
   ```
4. **Run preflight check** (mandatory before any training):
   ```python
   !python scripts/preflight/milestone_b_preflight.py --config configs/milestone_b.yaml \
       --data-root /content/drive/MyDrive/<project>/data/metadit
   ```
5. Symlink data and checkpoints to the persistent Drive folder:
   ```python
   !ln -s /content/drive/MyDrive/<project>/data/metadit data/metadit
   !mkdir -p /content/drive/MyDrive/<project>/checkpoints
   !ln -s /content/drive/MyDrive/<project>/checkpoints checkpoints
   ```
6. Confirm GPU:
   ```python
   !nvidia-smi
   ```
7. Run the milestone's training script, same as Kaggle:
   ```python
   !python scripts/train/train_milestone_b.py \
       --config configs/milestone_b.yaml \
       --resume checkpoints/milestone_b/latest.pt   # omit if starting fresh
   ```
   Because `checkpoints/` is symlinked to Drive, checkpoints survive disconnects automatically —
   no separate save step needed, but do periodically confirm files are actually landing on Drive
   (Colab disconnects can occasionally drop the last few seconds of I/O).
8. Push results back to GitHub when the run reaches a stopping point (same git commands as the
   Kaggle section above), or just leave results on Drive and copy `REPORT.md` back manually.

**Watch out for:** free-tier idle timeouts and ~12hr hard session caps — this is exactly why
resumable checkpointing (step 7) matters more on Colab than almost anywhere else in this project.

---

## 3. Which platform for which milestone

No strict rule, but as a default:

- **Milestone B–D** (moderate size, need to run the §7.2 minimal experiment plus a 20/40/60/80%
  sweep, then the physics loop): either platform works; Kaggle's dataset-attachment model avoids
  re-uploading the ~170k-sample dataset repeatedly, which is convenient here.
- **Milestone E** (InfoNCE with in-batch negatives — wants a real batch size, more memory
  pressure): prefer whichever platform is currently giving you the larger/more reliable GPU
  (check `nvidia-smi` output at session start on both if unsure).
- **Milestone F** (full context curriculum, longest sequential training): Kaggle's weekly quota
  structure suits a "train a chunk, checkpoint, resume next session" pattern well.
- **Milestone G–I** (LeJEPA ablation, stochastic latent K=32 eval, optional flow-matching):
  highest compute cost per §6 — confirm quota availability on whichever platform before starting,
  per `AGENTS.md` Standing Rule 5.

---

## 4. Sync-back checklist (do this before closing every cloud session)

- [ ] Latest checkpoint file(s) saved somewhere persistent (Kaggle Dataset/Drive/GitHub) — not
      only on the ephemeral session disk.
- [ ] `checkpoints/<milestone>/REPORT.md` updated with: metrics observed, done-criteria status,
      which platform/GPU was used, any deviations from the design doc.
- [ ] Results pulled back into your local repo (via `git pull`) before opening the next
      coding-agent session — the next session's task prompt assumes this REPORT.md is current.
- [ ] If the milestone is not yet done, note in REPORT.md exactly what checkpoint to `--resume`
      from and what remains, so the next cloud session (possibly days later, possibly you've
      forgotten details) can pick up cleanly.

---

## 5. Post-training verification (Phase 2 §13)

After training completes, verify the EXACT produced checkpoint:

```bash
python scripts/diagnostics/checkpoint_provenance_audit.py
python scripts/preflight/checkpoint_integrity_check.py \
    --checkpoint <EXACT_CHECKPOINT>
python scripts/eval/eval_vicreg_sanity.py \
    --checkpoint <EXACT_CHECKPOINT> \
    --config configs/milestone_b.yaml \
    --device cuda:0
python scripts/eval/physics_conditioning_audit.py \
    --checkpoint <EXACT_CHECKPOINT> \
    --config configs/milestone_b.yaml \
    --device cuda:0
```

Then compare final evaluation against in-loop validation. They must use the same
validation/mask/metric implementation.

---

## 6. Final acceptance (Phase 2 §14)

Create `checkpoints/milestone_b/FINAL_PIPELINE_ACCEPTANCE.md` recording:

| Gate | Result |
|------|--------|
| Fresh clone | PASS/FAIL |
| Dependency install | PASS/FAIL |
| Dataset discovery | PASS/FAIL |
| CPU preflight | PASS/FAIL |
| CUDA preflight | PASS/FAIL |
| One-step CUDA train | PASS/FAIL |
| Checkpoint save/load | PASS/FAIL |
| Resume | PASS/FAIL |
| Persistent checkpoint | PASS/FAIL |
| In-loop/final metric consistency | PASS/FAIL |
| Physics-control consistency | PASS/FAIL |
| Static audit | PASS/FAIL |
| Full tests | PASS/FAIL |

Phase 2 is complete only when this exact chain succeeds:
```
fresh clone
→ fresh Kaggle session
→ dataset discovered
→ dependencies verified
→ CUDA preflight PASS
→ tiny CUDA training PASS
→ checkpoint PASS
→ resume PASS
→ full training completes
→ persistent checkpoint survives
→ reload PASS
→ final evaluation PASS
```
with **zero manual source-code edits inside Kaggle**.