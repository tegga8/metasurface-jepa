# Milestone A — Reproduce MetaDiT representation learning (§7 Phase 1)

**Status:** executed per operator-approved narrowed scope (see §1 deviation). Results below.
**Waiting for human operator confirmation before Milestone B starts.**

**Execution location:** local dev machine (RTX 3050, torch 2.13.0+cpu, Python 3.11).
Inference-only CPU work; **no cloud session used** — consistent with `AGENTS.md` Milestone A.

---

## 1. Scope deviation (Standing Rule 4) — the §1.4 CLIP checkpoint was never released

The design doc §1.4 says the CLIP-style contrastive weights (`G → E_G → z_G`, `S → E_S → z_S`,
shared 512-dim space, `model/clip_model.py`) are "released weights [to] use as initialization".
**This was re-verified against the live sources and is only half true:**

| Source | Finding |
|---|---|
| HF dataset `Hao-Li-131/MetaDiT-AAAI2026` (gated listing, API tree) | `weights/` holds exactly 4 files: `metadit-small.bin`, `surrogate_model.bin`, `spec_encoder.pth`, `README.md` (0 bytes). No CLIP checkpoint. |
| GitHub `JessePrince/metadit` (API) | 0 releases, 0 tags, 1 branch (`main`). No alternate download channel. |
| Repo README + HF README | "clip" appears only as `scripts/train_clip.sh` / `train/train_clip.py` (training script). No released CLIP weights mentioned. |
| `train/train_metadit.py` | Released DiT was initialized with the released spectrum encoder only (`y_embedder.encoder`, `requires_grad_(False)`); the CLIP geometry ViT and `img_proj`/`context_proj` projections were used only for pretraining and never exported. |

Conclusion: the **spectrum-encoder side** of §1.4 is released; the **geometry-ViT side and the
512-dim projections are genuinely unreleased** (not a missed download). The literal Milestone A
retrieval check (matching vs. mismatched pairs in the shared 512-dim space) cannot be executed
with released weights without retraining, which the milestone forbids.

**Operator-approved narrowed scope (confirmed this session):**
1. `S → z_S` through the released spectrum encoder — shape/stats/determinism.
2. Consistency of that encoder with the encoder embedded in the released DiT (i.e. the encoder
   Phase 0's reproduction actually exercised).
3. **Proxy check** for the unreleased geometry-side embedding: released-DiT geometry-only
   features cluster sensibly against (a) 2×2×2 quantile-binned physical parameters and (b)
   topology descriptors — an independent second signal on the embedding space.

Milestone A's literal done criterion ("retrieval confirmed with released weights") is replaced
by this scope; the formal geometry↔spectrum retrieval test lands with our own encoders at
Milestone B (initialized from this same released DiT ViT) and goal-to-latent retrieval at
Milestone E (§4 InfoNCE).

## 2. Check 1 — `S → z_S` with the released spectrum encoder — PASS

- `src/encoders/spectrum_encoder.py` (thin wrapper, no weight modification) loads
  `spec_encoder.pth` (prefix `context_encoder.` stripped per `train_metadit.py`) into
  `VanillaSpectrumEncoder`, strict=True. Params: **4,526,144**.
- Forward `(64, 2, 301) → (64, 301, 256)` OK.
- Determinism: two identical forwards, max|Δ| = 0.
- Token norms: mean 13.22, std 7.14, [7.15, 46.25].
- Cross-sample mean |cos(z_S_i, z_S_j)| (i≠j) = **0.184** — distinct spectra produce
  well-separated embeddings, no collapse.

## 3. Check 2 — consistency with the encoder inside the released DiT — PASS

- `metadit-small.bin` → `metadit_s` (strict) → compare `y_embedder.encoder` weights vs
  `spec_encoder.pth`: **max abs diff = 0** (bitwise identical).
- Outputs on the same 64 spectra: **max abs diff = 0**.
- Interpretation: the released "pretrained" spectrum encoder IS the frozen condition encoder the
  released DiT conditions on. Phase 0's end-to-end reproduction (MAE 0.0803) therefore already
  exercised the full §1.3/§1.4 spectrum-encoding path. §1.4's claim is substantiated for the
  S-side; the shared-space G-side is the unreleased part.

## 4. Check 3 — geometry-only ViT-feature clustering sanity — PASS (moderate signal)

Setup: N=1500 fixed-seed (seed 0) test-set samples; features = mean-pooled tokens after
DiT blocks 6 and 11, extracted with **zero spectrum condition** (y = zeros(301 tokens),
t = 0) so features depend on geometry only; L2-normalized; k-means (k=8 / k=4, n_init=10) vs.
groupings; ARI + silhouette. Runtime ~3.3 s/chunk × 47 chunks ≈ 2.6 min, local CPU.

**Binning (per operator instruction — equal-count/quantile splits, not equal-width):**
- params (`parameter` columns = `[l_lattice, h_atom, r_atom]`): 2 bins each, edges
  l_lattice **2.76**, h_atom **0.86**, r_atom **4.57**; occupancy 8/8 cells, sizes
  [168, 183, 187, 187, 188, 195, 195, 197] — well balanced. Quantile (not equal-width)
  is the right choice: sampling is dense but not uniform (ranges [2.5, 3.0], [0.5, 1.0],
  [3.5, 5.0]), and quantile binning prevents degenerate cells.
- topology: fill_fraction quartiles [0.391, 0.445, 0.479]; `n_components` quartiles
  degenerate [1, 1, 1] (≥75% of patterns are single-connected component — expected for
  fabricated meta-atoms), so its ARI≈0 is a natural floor, not a signal; `n_holes` quartiles
  [0, 1, 1].

| Feature set | Grouping | kmeans ARI | silhouette(kmeans) | silhouette(known groups) |
|---|---|---|---|---|
| block 6 | param 2×2×2 cells | **0.187** | 0.194 | 0.078 |
| block 6 | fill_fraction quartiles | **0.236** | 0.248 | 0.071 |
| block 6 | n_components quartiles | 0.001 | 0.248 | 0.637 |
| block 6 | n_holes quartiles | 0.008 | 0.248 | 0.014 |
| block 11 | param 2×2×2 cells | **0.397** | 0.181 | 0.112 |
| block 11 | fill_fraction quartiles | 0.123 | 0.198 | 0.026 |
| block 11 | n_components quartiles | 0.001 | 0.198 | 0.471 |
| block 11 | n_holes quartiles | 0.006 | 0.198 | 0.012 |

Interpretation: the released DiT's deeper features carry real geometry-sensitive structure —
block-11 mean-pooled features recover the 8 (l_lattice, h_atom, r_atom) cells with ARI 0.397
vs. ~0 chance, and fill-fraction separation peaks at block 6 (ARI 0.236). Silhouettes are low
(0.02–0.25), expected for a diffusion backbone never trained for discriminative clustering.
The embedding space is **not geometry-blind**; this is the "second independent signal" the
operator asked for, with the caveat that it is a proxy, not the §1.4 retrieval.

## 5. Done-criteria assessment

- *Must pass to proceed*: geometry-spectrum retrieval confirmed with released weights — **not
  literally executable** (CLIP weights unreleased, §1). Operator-approved replacement scope
  executed and passing (Checks 1–3). Recorded here as the Standing Rule 4 deviation.
- *Nice to have*: qualitative retrieval-pair dump — not applicable (no shared-space model).

## 6. Guardrail check (Milestone A-specific)

No §13 failure mode applies yet (no new model built). Guardrail risk was scope drift
("building new components under the guise of checking") — handled by (a) not constructing any
stand-in z_G from untrained projections, (b) using only released frozen weights, (c) flagging
the scope change to the operator before implementing.

## 7. Files produced

- `src/encoders/spectrum_encoder.py` — thin released-weight wrapper (Milestone A deliverable).
- `scripts/eval/eval_milestone_a.py` — standalone CLI (runs Checks 1–3; `--n-samples`,
  `--seed`). New script rather than extending `reproduce_metadit_baseline.py` (AGENTS.md lists
  "extend from Phase 0"): the Phase 0 script is kept untouched as the gate artifact; both share
  the same metadit loading conventions. Recorded as a minor deviation.
- `checkpoints/milestone_a/milestone_a_results.json` — raw results (cells, bin edges, ARI/
  silhouette per feature set).

Repro command (local, CPU): `python scripts/eval/eval_milestone_a.py --n-samples 1500`

## 8. Decision requested (operator)

Approve Milestone A as complete under the narrowed scope so Milestone B can start (its
geometry encoder initializes from the same released DiT ViT — now known to carry
geometry-sensitive features per Check 3 — and the first real geometry↔spectrum retrieval
check happens there / at Milestone E).
