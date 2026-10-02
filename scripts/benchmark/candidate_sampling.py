"""K-candidate generators for the AAE&K analogue.

MetaDiT's AAE&K takes the worst of K generations per condition, where the K come
from independent diffusion seeds. The unified JEPA is deterministic (one geometry
per input), so AAE&K is undefined without a candidate source. These generators
provide one, explicitly labelled as an analogue:

    latent-jitter      : Gaussian noise sigma added to the predictor's occupancy
                         latents before hard decode.
    cfg-guidance-grid  : candidates from classifier-free guidance weights w.

`latent-jitter` with `sigma == 0` reproduces the deterministic single-shot
prediction exactly (asserted in tests), so K=1/sigma=0 == plain AAE.

This measures decode-robustness to injected noise only and must never be
presented as MetaDiT-equivalent seed diversity (AGENTS.md rule 8, README.md §3).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
BENCH_DIR = os.path.join(REPO_ROOT, "scripts", "benchmark")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import torch

import metadit_metrics as mm


def _decode(model, z_hat, scalar_pred, occ, mask, scalar_known, scalar_values):
    geometry, _ = model.decode_geometry(
        z_hat, scalar_pred, occ_input=occ, mask=mask,
        scalar_known=scalar_known, scalar_values=scalar_values,
        hard_forward=True)
    return geometry


@torch.no_grad()
def latent_jitter_geometries(model, out, occ, mask, scalar_known, scalar_values,
                             sigma, k, seed=1234):
    """K hard geometries from jittered occupancy latents (candidate 0 = noiseless).

    Candidate index 0 uses the unmodified `z_hat` so K=1/sigma=0 is exactly the
    deterministic deployment. Noise is drawn on CPU with a fixed generator and
    moved to the latent device (no device-mismatch RNG).
    """
    z_hat = out["z_hat"]
    scalar_pred = out["scalar_pred"]
    gen = torch.Generator().manual_seed(int(seed))
    geoms = []
    for j in range(int(k)):
        if j == 0 or sigma <= 0.0:
            z = z_hat
        else:
            noise = torch.randn(z_hat.shape, generator=gen, dtype=torch.float32)
            z = z_hat + noise.to(z_hat.device) * float(sigma)
        geoms.append(_decode(model, z, scalar_pred, occ, mask,
                             scalar_known, scalar_values))
    return geoms


@torch.no_grad()
def candidate_aae_matrix(model, surrogate, out, occ, mask, scalar_known,
                         scalar_values, spec, sigma, k, seed=1234):
    """(B, K) per-item AAE matrix from latent-jitter candidates."""
    geoms = latent_jitter_geometries(model, out, occ, mask, scalar_known,
                                     scalar_values, sigma, k, seed)
    cols = [mm.per_item_aae(spec, surrogate(g).prediction) for g in geoms]
    return torch.stack(cols, dim=1)


@torch.no_grad()
def cfg_guidance_aae_matrix(model, surrogate, occ, sv, scalar_known, spec, mask,
                            weights):
    """(B, len(weights)) per-item AAE matrix from the CFG guidance grid.

    A second, more meaningful candidate source for a deterministic model: the
    guidance weight genuinely changes the deployed design (w=0 unconditional,
    w=1 plain real-goal). Reuses the canonical cfg_forward.
    """
    from predictor.guidance import cfg_forward

    cols = []
    for w in weights:
        z_guided, scalar_guided, _ = cfg_forward(
            model, occ, sv, scalar_known, spec, mask, float(w))
        geom = _decode(model, z_guided, scalar_guided, occ, mask,
                       scalar_known, sv)
        cols.append(mm.per_item_aae(spec, surrogate(geom).prediction))
    return torch.stack(cols, dim=1)
