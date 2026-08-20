"""Top-level model assembly (§11.1 data-flow spec; Milestone B slice).

Variant 'jepa'  — GoalConditionedJEPA: block-masked context -> Ẑ_y against EMA target
                   latent, L = L_J only (§4.1 Phase 2).

Frozen released components stay outside the trainable state: the released spectrum encoder
keys are filtered from saved checkpoints (re-loaded from disk on every build), and the EM
surrogate / released DiT are constructed by the training script on demand.

The historical direct masked generator (Baseline 2, §10.1) is not part of this module;
it lives as a self-contained reference implementation in `src/reference/` and is
unreachable from the active training path.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

import torch
import torch.nn.functional as F
from torch import nn

from data.mask import apply_mask_to_pixels
from encoders.context_encoder import ContextEncoder
from encoders.geometry_encoder import GeometryEncoder
from encoders.spectrum_encoder import ReleasedSpectrumEncoder, SpectrumPath
from encoders.target_encoder import EMAEncoder
from losses.jepa_loss import jepa_loss
from predictor.gclct import GCLCT

PIXEL_GRID = 16  # 64 / patch_size 4


class _JEPAForwardMixin:
    """Shared context/spectrum/predictor forward producing latent delta predictions."""

    def _encode(self, G, S, M, goal_mode, need_attn):
        """Full student-side encode: masked context -> z_x; spectrum -> (c_physics,
        a_goal); predictor -> z_hat. Returns them all so the model output contract
        (spec §30) exposes every representation boundary explicitly."""
        G_c = apply_mask_to_pixels(G, M)

        # Full-resolution context representation.
        z_x = self.context_encoder(G_c, M)

        assert z_x.ndim == 3, (
            f"Context encoder output must be [B,256,384], got {tuple(z_x.shape)}"
        )
        assert z_x.shape[1] == 256, (
            f"Context encoder must preserve 256 tokens, got {z_x.shape[1]}"
        )
        assert z_x.shape[2] == 384, (
            f"Context encoder embedding dim must be 384, got {z_x.shape[2]}"
        )

        c_physics, a_goal = self.spectrum_path(
            S,
            goal_mode=goal_mode,
        )

        mask = (M.view(M.shape[0], -1) == 0)

        assert mask.shape[1] == 256, (
            f"Mask must have 256 token positions, got {tuple(mask.shape)}"
        )
        assert mask.any(dim=1).all(), (
            "Every sample must contain at least one masked token"
        )

        mask_token = self.context_encoder.mask_token
        pos = self.context_encoder.geo.pos_embed

        queries = torch.where(
            mask.unsqueeze(-1),
            mask_token + pos,
            z_x,
        )

        # KEEP ALL 256 geometry tokens.
        kv = torch.cat([z_x, a_goal], dim=1)

        z_hat, weights = self.predictor(
            queries,
            kv,
            c_physics,
            need_weights=need_attn,
        )

        assert z_hat.ndim == 3, (
            f"Predictor output must be [B,256,384], got {tuple(z_hat.shape)}"
        )
        assert z_hat.shape[1] == 256, (
            f"Predictor must output 256 tokens, got {z_hat.shape[1]}"
        )
        assert z_hat.shape[2] == 384, (
            f"Predictor output dim must be 384, got {z_hat.shape[2]}"
        )

        assert z_x.shape == z_hat.shape, (
            f"Context/prediction mismatch: "
            f"z_x={tuple(z_x.shape)}, z_hat={tuple(z_hat.shape)}"
        )

        return z_hat, z_x, mask, weights, c_physics, a_goal

    def query_predictions(self, G, S, M, goal_mode="real"):
        z_hat, *_ = self._encode(
            G, S, M, goal_mode, need_attn=False
        )
        return z_hat


class GoalConditionedJEPA(_JEPAForwardMixin, nn.Module):
    def __init__(self, hidden=384, num_heads=6, geo_depth=6, predictor_depth=6,
                 goal_tokens=16, num_predictor_heads=6,
                 momentum_start=0.996, momentum_end=0.999):
        super().__init__()
        self.hidden = hidden
        geo = GeometryEncoder(hidden=hidden, num_heads=num_heads, depth=geo_depth)
        self.context_encoder = ContextEncoder(geo, hidden=hidden)
        self.spectrum_path = SpectrumPath(None, hidden=hidden, goal_tokens=goal_tokens)
        self.predictor = GCLCT(depth=predictor_depth, hidden=hidden,
                               num_heads=num_predictor_heads)
        self.ema = EMAEncoder(
            geo,
            momentum_start=momentum_start,
            momentum_end=momentum_end
        )
        for name, param in self.ema.named_parameters():
            assert not param.requires_grad, (
                f"EMA target parameter is trainable: {name}"
            )

        self.geometry_encoder = geo

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True,):
        z_hat, z_x, mask, weights, c_physics, a_goal = self._encode(
            G, S, M, goal_mode, need_attn
        )

        out = dict(
            z_hat=z_hat,
            z_x=z_x,
            mask=mask,
            attn_weights=weights,
            c_physics=c_physics,
            a_goal=a_goal,
        )

        if with_target:
            z_y_raw = self.ema(G)

            assert z_y_raw.ndim == 3, (
                f"EMA target output must be [B,256,384], got {tuple(z_y_raw.shape)}"
            )
            assert z_y_raw.shape[1] == 256, (
                f"EMA target must output 256 tokens, got {z_y_raw.shape[1]}"
            )
            assert z_y_raw.shape[2] == 384, (
                f"EMA target dim must be 384, got {z_y_raw.shape[2]}"
            )

            assert z_y_raw.shape == z_hat.shape, (
                f"Target/prediction mismatch: "
                f"z_y_raw={tuple(z_y_raw.shape)}, z_hat={tuple(z_hat.shape)}"
            )

            # Spec §10: expose the raw target AND an explicit feature-wise
            # normalization boundary (F.layer_norm over the 384-D feature axis,
            # per-sample per-token — no learnable weights, never overwrites raw).
            out["z_y"] = z_y_raw            # backward-compatible raw alias
            out["z_y_raw"] = z_y_raw
            out["z_y_normalized"] = F.layer_norm(z_y_raw, (z_y_raw.shape[-1],))

        return out

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        L, per_sample = jepa_loss(out["z_hat"], out["z_y"], out["mask"], proj=None)
        return L, out


def load_released_metadit_state_dict(weights_path):
    return torch.load(weights_path, map_location="cpu")


def init_geometry_from_metadit(model, metadit_weights, blocks_to_take=6):
    sd = load_released_metadit_state_dict(metadit_weights)
    model.geometry_encoder.init_from_metadit(sd, blocks_to_take=blocks_to_take)
    return sd


def set_spectrum_path(model, spec_weights, device):
    released = ReleasedSpectrumEncoder(spec_weights, device=device)
    model.spectrum_path.released = released
    model.spectrum_path.released.to(device)


def build_model(cfg, spec_weights, device="cpu", init_from_metadit=True,
                metadit_weights=None, blocks_to_take=6):
    variant = cfg.get("variant", "jepa")

    if variant != "jepa":
        raise RuntimeError(
            "Only the JEPA variant is enabled during the architecture refactor."
        )
    kwargs = dict(hidden=cfg.get("hidden", 384),
                  num_heads=cfg.get("num_heads", 6),
                  geo_depth=cfg.get("geo_depth", 6),
                  predictor_depth=cfg.get("predictor_depth", 8),
                  goal_tokens=cfg.get("goal_tokens", 16),
                  num_predictor_heads=cfg.get("num_predictor_heads", 6))
    kwargs.update(
        momentum_start=cfg.get("ema_momentum_start", 0.996),
        momentum_end=cfg.get("ema_momentum_end", 0.999),
    )

    model = GoalConditionedJEPA(**kwargs)

    set_spectrum_path(model, spec_weights, device)
    if init_from_metadit:
        assert metadit_weights is not None
        init_geometry_from_metadit(model, metadit_weights,
                                   blocks_to_take=cfg.get("geo_depth", 6))
    if variant == "jepa":
        model.ema.target.load_state_dict(model.geometry_encoder.state_dict())
    model.to(device)
    return model


SAVED_EXCLUDES = (".released.",)


def saveable_state_dict(model):
    """Drop frozen released components (re-loaded from disk on rebuild)."""
    return {k: v for k, v in model.state_dict().items()
            if not any(x in k for x in SAVED_EXCLUDES)}


def load_into_model(model, sd, device, strict=True):
    """Load a saved state dict into a model, refusing silent mismatches.

    strict=True (Bug #11): a checkpoint whose keys do not exactly match the model
    raises instead of silently leaving parameters at init — previously strict=False
    could load a stale/renamed checkpoint and bias every downstream result without
    any warning. Frozen released components (SAVED_EXCLUDES) are filtered on BOTH
    sides: they are excluded from checkpoints at save time (re-loaded from disk on
    every build via set_spectrum_path) and therefore excluded from the strict
    comparison here too.
    """
    keys = [k for k in sd if not any(x in k for x in SAVED_EXCLUDES)]
    filtered = {k: sd[k] for k in keys}
    if strict:
        model_keys = set(model.state_dict())
        released = {k for k in model_keys if any(x in k for x in SAVED_EXCLUDES)}
        expected = set(filtered)
        missing = sorted(expected - model_keys)      # ckpt keys the model lacks
        unexpected = sorted((model_keys - expected) - released)  # model keys the ckpt lacks
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint/model key mismatch (refusing silent non-strict load; "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
                f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''})")
    model_keys = model.state_dict()
    model.load_state_dict({k: filtered[k] for k in filtered if k in model_keys},
                          strict=False)
    model.to(device)