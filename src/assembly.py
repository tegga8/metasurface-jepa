"""Top-level model assembly (§11.1 data-flow spec; Milestone B slice).

Variant 'jepa'  — GoalConditionedJEPA: block-masked context -> Ẑ_y against EMA target
                   latent, L = L_J only (§4.1 Phase 2).
Variant 'direct' — DirectMaskedGenerator (Baseline 2, §10.1): G_c + S -> Ĝ pixels,
                   masked-pixel L1, no JEPA latent objective anywhere.

Frozen released components stay outside the trainable state: the released spectrum encoder
keys are filtered from saved checkpoints (re-loaded from disk on every build), and the EM
surrogate / released DiT are constructed by the training script on demand.
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
from torch import nn

from data.mask import apply_mask_to_pixels
from encoders.context_encoder import ContextEncoder, PerceiverBottleneck
from encoders.geometry_encoder import GeometryEncoder
from encoders.spectrum_encoder import ReleasedSpectrumEncoder, SpectrumPath
from encoders.target_encoder import EMAEncoder
from losses.jepa_loss import ProjectionMLP, jepa_loss
from predictor.gclct import GCLCT

PIXEL_GRID = 16  # 64 / patch_size 4


def _unpatchify(tokens, patch_size=4, channels=3):
    """(B, 256, channels*patch^2) -> (B, channels, 64, 64)."""
    b, n, _ = tokens.shape
    grid = int(n ** 0.5)
    x = tokens.reshape(b, grid, grid, patch_size, patch_size, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(b, channels, grid * patch_size, grid * patch_size)


class _JEPAForwardMixin:
    """Shared context/spectrum/predictor forward producing latent delta predictions."""

    def _encode(self, G, S, M, goal_mode, need_attn):
        G_c = apply_mask_to_pixels(G, M)
        z_x = self.context_encoder(G_c, M)                 # (B, 256, 384)
        z_xb = self.perceiver(z_x)                          # (B, 64, 384)
        c_physics, a_goal = self.spectrum_path(S, goal_mode=goal_mode)
        mask = (M.view(M.shape[0], -1) == 0)                # (B, 256) bool, 1 = masked
        mask_token = self.context_encoder.mask_token        # (1, 1, 384)
        pos = self.context_encoder.geo.pos_embed            # (1, 256, 384)
        queries = torch.where(mask.unsqueeze(-1), mask_token + pos, z_x)
        kv = torch.cat([z_xb, a_goal], dim=1)               # (B, 80, 384)
        delta, weights = self.predictor(queries, kv, c_physics, need_weights=need_attn)
        base = torch.where(mask.unsqueeze(-1), mask_token, z_x)
        return base, delta, z_x, mask, weights

    def query_predictions(self, G, S, M, goal_mode="real"):
        base, delta, *_ = self._encode(G, S, M, goal_mode, need_attn=False)
        return base + delta


class GoalConditionedJEPA(_JEPAForwardMixin, nn.Module):
    def __init__(self, hidden=384, num_heads=6, geo_depth=6, predictor_depth=8,
                 bottleneck_tokens=64, goal_tokens=16, num_predictor_heads=6,
                 momentum_start=0.996, momentum_end=0.999):
        super().__init__()
        self.hidden = hidden
        geo = GeometryEncoder(hidden=hidden, num_heads=num_heads, depth=geo_depth)
        self.context_encoder = ContextEncoder(geo, hidden=hidden)
        self.perceiver = PerceiverBottleneck(bottleneck_tokens, hidden, num_heads)
        self.spectrum_path = SpectrumPath(None, hidden=hidden, goal_tokens=goal_tokens)
        self.predictor = GCLCT(depth=predictor_depth, hidden=hidden,
                               num_heads=num_predictor_heads, head_type="latent")
        self.proj = ProjectionMLP(hidden)
        self.ema = EMAEncoder(geo, momentum_start=momentum_start,
                              momentum_end=momentum_end)
        self.geometry_encoder = geo  # for the training script (EMA source)

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        """Returns dict: z_hat (B,256,384), mask (B,256) bool, target z_y (B,256,384)."""
        base, delta, z_x, mask, weights = self._encode(G, S, M, goal_mode, need_attn)
        out = dict(z_hat=base + delta, mask=mask, attn_weights=weights)
        if with_target:
            out["z_y"] = self.ema(G)
        return out

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        L, per_sample = jepa_loss(out["z_hat"], out["z_y"], out["mask"], proj=self.proj)
        return L, out


class DirectMaskedGenerator(_JEPAForwardMixin, nn.Module):
    def __init__(self, hidden=384, num_heads=6, geo_depth=6, predictor_depth=8,
                 bottleneck_tokens=64, goal_tokens=16, num_predictor_heads=6):
        super().__init__()
        self.hidden = hidden
        geo = GeometryEncoder(hidden=hidden, num_heads=num_heads, depth=geo_depth)
        self.context_encoder = ContextEncoder(geo, hidden=hidden)
        self.perceiver = PerceiverBottleneck(bottleneck_tokens, hidden, num_heads)
        self.spectrum_path = SpectrumPath(None, hidden=hidden, goal_tokens=goal_tokens)
        self.predictor = GCLCT(depth=predictor_depth, hidden=hidden,
                               num_heads=num_predictor_heads, head_type="pixel")
        self.geometry_encoder = geo

    def forward(self, G, S, M, goal_mode="real", need_attn=False):
        _, delta, z_x, mask, weights = self._encode(G, S, M, goal_mode, need_attn)
        g_hat = _unpatchify(delta)
        return dict(g_hat=g_hat, z_latent=delta, mask=mask, attn_weights=weights)

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        pmask = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        ub = (pmask == 0).unsqueeze(1)
        ub = ub.expand_as(out["g_hat"])
        diff = (out["g_hat"] - G).abs()
        masked = diff[ub]
        full = diff.mean()
        return (masked.mean() if masked.numel() else full), out


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
    assert variant in ("jepa", "direct")
    kwargs = dict(hidden=cfg.get("hidden", 384),
                  num_heads=cfg.get("num_heads", 6),
                  geo_depth=cfg.get("geo_depth", 6),
                  predictor_depth=cfg.get("predictor_depth", 8),
                  bottleneck_tokens=cfg.get("bottleneck_tokens", 64),
                  goal_tokens=cfg.get("goal_tokens", 16),
                  num_predictor_heads=cfg.get("num_predictor_heads", 6))
    if variant == "jepa":
        kwargs.update(momentum_start=cfg.get("ema_momentum_start", 0.996),
                      momentum_end=cfg.get("ema_momentum_end", 0.999))
        model = GoalConditionedJEPA(**kwargs)
    else:
        model = DirectMaskedGenerator(**kwargs)

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


def load_into_model(model, sd, device):
    keys = [k for k in sd if not any(x in k for x in SAVED_EXCLUDES)]
    model.load_state_dict({k: sd[k] for k in keys}, strict=False)
    model.to(device)