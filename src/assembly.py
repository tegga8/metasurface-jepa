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
from encoders.occupancy_encoder import OccupancyEncoder
from encoders.scalar_encoder import ScalarEncoder
from fusion.fusion_encoder import FusionEncoder
from decoders.scalar_decoder import ScalarDecoder
from decoders.geometry_decoder import GeometryDecoder
from data.factorize import assemble_metadit_geometry
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

        hidden = self.hidden
        assert z_x.ndim == 3, (
            f"Context encoder output must be [B,256,{hidden}], got {tuple(z_x.shape)}"
        )
        assert z_x.shape[1] == 256, (
            f"Context encoder must preserve 256 tokens, got {z_x.shape[1]}"
        )
        assert z_x.shape[2] == hidden, (
            f"Context encoder embedding dim must be {hidden}, got {z_x.shape[2]}"
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

        hidden = self.hidden
        assert z_hat.ndim == 3, (
            f"Predictor output must be [B,256,{hidden}], got {tuple(z_hat.shape)}"
        )
        assert z_hat.shape[1] == 256, (
            f"Predictor must output 256 tokens, got {z_hat.shape[1]}"
        )
        assert z_hat.shape[2] == hidden, (
            f"Predictor output dim must be {hidden}, got {z_hat.shape[2]}"
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

    def enforce_frozen_reference_modes(self):
        """Keep frozen reference modules in eval() regardless of the student's mode.

        `model.train()` recursively flips children to training mode, but the EMA
        target and the released spectrum encoder are parameter-frozen reference
        components that must behave deterministically at inference. Called from
        `train()` and `forward()` so every mode switch / checkpoint load / resume
        path is covered. The released encoder may be absent in unit-test stubs.
        """
        self.ema.target.eval()
        released = getattr(self.spectrum_path, "released", None)
        if released is not None:
            released.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enforce_frozen_reference_modes()
        return self

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True,):
        self.enforce_frozen_reference_modes()
        z_hat, z_x, mask, weights, c_physics, a_goal = self._encode(
            G, S, M, goal_mode, need_attn
        )

        b = G.shape[0]
        hidden = self.hidden
        assert c_physics.shape == (b, hidden), (
            f"c_physics must be [B,{hidden}], got {tuple(c_physics.shape)}"
        )
        assert a_goal.shape == (b, 16, hidden), (
            f"a_goal must be [B,16,{hidden}], got {tuple(a_goal.shape)}"
        )
        assert mask.shape == (b, 256), (
            f"mask must be [B,256], got {tuple(mask.shape)}"
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
                f"EMA target output must be [B,256,{hidden}], got {tuple(z_y_raw.shape)}"
            )
            assert z_y_raw.shape[1] == 256, (
                f"EMA target must output 256 tokens, got {z_y_raw.shape[1]}"
            )
            assert z_y_raw.shape[2] == hidden, (
                f"EMA target dim must be {hidden}, got {z_y_raw.shape[2]}"
            )

            assert z_y_raw.shape == z_hat.shape, (
                f"Target/prediction mismatch: "
                f"z_y_raw={tuple(z_y_raw.shape)}, z_hat={tuple(z_hat.shape)}"
            )

            # Spec §10: expose the raw target AND an explicit feature-wise
            # normalization boundary (F.layer_norm over the 384-D feature axis,
            # per-sample per-token — no learnable weights, never overwrites raw).
            # `z_y` is kept ONLY as a backward-compatible raw alias; active code
            # must consume `z_y_raw` / `z_y_normalized` explicitly (hardening §2).
            out["z_y_raw"] = z_y_raw
            out["z_y_normalized"] = F.layer_norm(z_y_raw, (z_y_raw.shape[-1],))
            out["z_y"] = z_y_raw            # compat alias — do NOT consume

        return out

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        L, per_sample = jepa_loss(out["z_hat"], out["z_y_raw"], out["mask"], proj=None)
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
        # missing: keys the model expects but checkpoint lacks
        missing = sorted(model_keys - expected - released)
        # unexpected: keys in checkpoint that model doesn't expect
        unexpected = sorted(expected - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint/model key mismatch (refusing silent non-strict load; "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
                f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''})")
    model_keys = model.state_dict()
    model.load_state_dict({k: filtered[k] for k in filtered if k in model_keys},
                          strict=False)
    model.to(device)


# ===========================================================================
# Unified JEPA model — architecture_v5.md §3.1-§3.6, §4.1, §5
#
# New internal representation: occupancy M[64,64] + l_lattice/h_atom/r_atom as
# explicit scalars with known/unknown flags + target spectrum [2,301].
# 192-D throughout (except c_physics/a_goal at 384 from the frozen SpectrumPath,
# projected downstream to 192).
# ===========================================================================

UNIFIED_ARCHITECTURE_ID = "unified_occ_param_spectrum_jepa_v1"


class UnifiedJEPA(nn.Module):
    """Unified occupancy + scalar + spectrum JEPA (architecture_v5.md §3.1-§3.6).

    Active model accepts semantically:
        occupancy      [B,1,64,64]  single-channel binary occupancy
        scalar_values  [B,3]        (l_lattice, h_atom, r_atom)
        scalar_known   [B,3] bool   which scalars are observed
        spectrum       [B,2,301]    target electromagnetic spectrum
        mask           [B,16,16]    1=visible, 0=masked (at token-grid resolution)

    Internal flow (§11.1):
        occupancy + masked scalars + FiLM → OccupancyEncoder → z_x [B,256,192]
        spectrum → SpectrumPath(frozen) → c_physics [B,384], a_goal [B,16,384]
        z_x + proj(a_goal) + scalar_summary → FusionEncoder → fused [B,273,192]
        256 mask-token queries + 1 scalar-summary query → GCLCT(c_physics)
        → z_hat [B,257,192] → occupancy_pred + scalar_summary_pred → scalar_pred
        EMA target encoder (occupancy_ema + scalar_mlp_ema) → z_y_raw [B,256,192]

    EMA rules (§3.6):
        - occupancy EMA = JEPA target for occupancy tokens only
        - scalar_mlp_ema = target-side FiLM conditioning only
        - NO scalar EMA latent loss target
    """

    architecture_id = UNIFIED_ARCHITECTURE_ID

    def __init__(self, hidden=192, num_heads=6, geo_depth=6, predictor_depth=8,
                 goal_tokens=16, num_predictor_heads=6, scalar_hidden=128,
                 n_film_blocks=6, spec_dim=256,
                 momentum_start=0.996, momentum_end=0.999):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.goal_tokens = goal_tokens
        self.architecture_id = UNIFIED_ARCHITECTURE_ID

        # Student encoders
        self.occupancy_encoder = OccupancyEncoder(
            hidden=hidden, num_heads=num_heads, depth=geo_depth
        )
        self.scalar_encoder = ScalarEncoder(
            hidden=hidden, scalar_hidden=scalar_hidden, n_film_blocks=n_film_blocks
        )

        # Spectrum path — stays at 384-D (architecture_v5.md §3.3)
        self.spectrum_path = SpectrumPath(
            None, spec_dim=spec_dim, hidden=384, goal_tokens=goal_tokens,
            num_heads=4,
        )

        # Fusion (192-D, projects a_goal 384→192 internally)
        self.fusion_encoder = FusionEncoder(
            hidden=hidden, num_heads=num_heads, depth=2, goal_dim_in=384
        )

        # Predictor — accepts 384-D c_physics, projects to 192 internally
        self.predictor = GCLCT(
            depth=predictor_depth, hidden=hidden, num_heads=num_predictor_heads,
            c_physics_dim=384,
        )

        # Scalar decode heads
        self.scalar_decoder = ScalarDecoder(hidden=hidden)

        # Geometry decoder: latent → occupancy logits + geometry (Phase 4 MD §3)
        self.geometry_decoder = GeometryDecoder(
            hidden_dim=hidden, base_dim=hidden // 2,
            num_channels=3, occupancy_head=True,
        )

        # EMA target for occupancy encoder (z_y_raw JEPA target)
        self.ema = EMAEncoder(
            self.occupancy_encoder,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
        )

        # EMA shadow copy of the scalar MLP — target-side FiLM conditioning only
        self.scalar_mlp_ema = EMAEncoder(
            self.scalar_encoder,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
        )

        # Learned mask / query tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)
        self.scalar_query_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.scalar_query_token, std=0.02)

        # Verify EMA params are frozen
        for name, param in self.ema.named_parameters():
            assert not param.requires_grad, f"occupancy EMA trainable: {name}"
        for name, param in self.scalar_mlp_ema.named_parameters():
            assert not param.requires_grad, f"scalar_mlp_ema trainable: {name}"

    # --- EMA helpers -------------------------------------------------------

    def set_total_steps(self, n):
        self.ema.set_total_steps(n)
        self.scalar_mlp_ema.set_total_steps(n)

    def enforce_frozen_reference_modes(self):
        """Keep frozen reference modules in eval() regardless of student mode."""
        self.ema.target.eval()
        self.scalar_mlp_ema.target.eval()
        released = getattr(self.spectrum_path, "released", None)
        if released is not None:
            released.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enforce_frozen_reference_modes()
        return self

    # --- Forward -----------------------------------------------------------

    def _build_scalar_input(self, scalar_values, scalar_known):
        """Build the 6-D scalar MLP input [l_val, l_known, h_val, h_known,
        r_val, r_known] with values zeroed where unknown."""
        known_f = scalar_known.float()
        masked = torch.where(scalar_known, scalar_values,
                             torch.zeros_like(scalar_values))
        return torch.stack([
            masked[:, 0], known_f[:, 0],
            masked[:, 1], known_f[:, 1],
            masked[:, 2], known_f[:, 2],
        ], dim=-1)

    def forward(self, occupancy, scalar_values, scalar_known, spectrum, mask,
                goal_mode="real", with_target=True, need_attn=False):
        """Unified forward.

        Args:
            occupancy:      [B,1,64,64] binary float occupancy.
            scalar_values:  [B,3] (l_lattice, h_atom, r_atom)
            scalar_known:   [B,3] bool — which scalars are observed.
            spectrum:       [B,2,301] target spectrum.
            mask:           [B,16,16]  1=visible, 0=masked.
            goal_mode:      "real" | "null" | "shuffled"
            with_target:    compute EMA target latent z_y_raw.
            need_attn:      return attention weights.

        Returns dict with z_hat, z_x, mask, c_physics, a_goal, scalar_pred,
        scalar_summary_pred, and (if with_target) z_y_raw, z_y_normalized, z_y.
        """
        self.enforce_frozen_reference_modes()
        b = occupancy.shape[0]
        hidden = self.hidden

        # 1. Build scalar MLP input from values + known flags
        scalar_mlp_input = self._build_scalar_input(scalar_values, scalar_known)

        # 2. Scalar encoder (live) → FiLM params + scalar summary token
        film_params, scalar_summary = self.scalar_encoder(scalar_mlp_input)

        # 3. Occupancy encoder (student) with mask replacement + FiLM
        masked_occ = apply_mask_to_pixels(occupancy, mask)
        z_x = self.occupancy_encoder(
            masked_occ, film_params=film_params,
            mask=mask, mask_token=self.mask_token,
        )  # (B, 256, hidden)

        # 4. Spectrum path (frozen)
        c_physics, a_goal = self.spectrum_path(spectrum, goal_mode=goal_mode)
        # c_physics: (B, 384), a_goal: (B, 16, 384)

        # 5. Fusion: 256 occupancy + 16 goal (projected 384→192) + 1 scalar summary
        fused = self.fusion_encoder(z_x, a_goal, scalar_summary)  # (B, 273, hidden)
        assert fused.shape[1] == 273, (
            f"Fusion must output 273 tokens (256+16+1), got {fused.shape[1]}"
        )

        # 6. Construct predictor queries
        pos = self.occupancy_encoder.pos_embed  # (1, 256, hidden)
        vis_mask = (mask.view(b, -1) > 0.5)  # True = visible, (B, 256)
        occ_queries = torch.where(
            vis_mask.unsqueeze(-1),
            fused[:, :256, :],          # visible: fused tokens
            self.mask_token + pos,      # masked: mask_token + pos
        )  # (B, 256, hidden)
        scalar_query = self.scalar_query_token.expand(b, -1, -1)  # (B, 1, hidden)
        queries = torch.cat([occ_queries, scalar_query], dim=1)    # (B, 257, hidden)

        # 7. Predictor (c_physics 384→192 via c_phys_proj)
        z_hat_raw, _ = self.predictor(queries, fused, c_physics)  # (B, 257, hidden)

        # 8. Split predictions
        occupancy_pred = z_hat_raw[:, :256, :]         # (B, 256, hidden)
        scalar_summary_pred = z_hat_raw[:, 256, :]     # (B, hidden)

        # 9. Scalar decode
        scalar_pred = self.scalar_decoder(scalar_summary_pred)  # (B, 3)

        # 10. Loss mask: True = masked position (for JEPA loss)
        loss_mask = ~vis_mask  # (B, 256)

        out = dict(
            z_hat=occupancy_pred,
            z_x=z_x,
            mask=loss_mask,
            c_physics=c_physics,
            a_goal=a_goal,
            scalar_pred=scalar_pred,
            scalar_summary_pred=scalar_summary_pred,
        )

        if with_target:
            with torch.no_grad():
                # True scalars (all known) for target-side FiLM
                true_input = torch.stack([
                    scalar_values[:, 0], torch.ones_like(scalar_values[:, 0]),
                    scalar_values[:, 1], torch.ones_like(scalar_values[:, 1]),
                    scalar_values[:, 2], torch.ones_like(scalar_values[:, 2]),
                ], dim=-1)  # (B, 6)
                film_params_ema, _ = self.scalar_mlp_ema(true_input)
                z_y_raw = self.ema(occupancy, film_params=film_params_ema)
                out["z_y_raw"] = z_y_raw
                out["z_y_normalized"] = F.layer_norm(
                    z_y_raw, (z_y_raw.shape[-1],)
                )
                out["z_y"] = z_y_raw  # compat alias

        return out

    def decode_geometry(self, z_hat, scalar_pred, occ_input=None, mask=None,
                        use_ste=False):
        """Decode predicted latents to surrogate-ready geometry (Phase 4 MD §1-§3).

        Args:
            z_hat:       [B, 256, hidden] predicted occupancy latents.
            scalar_pred: [B, 3] (l_lattice, h_atom, r_atom) physical values.
            occ_input:   [B, 1, 64, 64] original binary occupancy (for retention).
            mask:        [B, 16, 16] 1=visible, 0=masked — retains visible pixels.
            use_ste:     If True, use hard occupancy for surrogate input but
                        soft gradient path (straight-through estimator).

        Returns:
            geometry:    [B, 3, 64, 64] — r_atom/5, h_atom, l_lattice/3.
            soft_occ:    [B, 1, 64, 64] — sigmoid occupancy logits.
        """
        geometry_logits, occ_logits = self.geometry_decoder(z_hat)
        soft_occ = torch.sigmoid(occ_logits)  # (B, 1, 64, 64)

        if use_ste and self.training:
            hard_occ = (soft_occ > 0.5).float()
            occ_for_assembly = hard_occ + soft_occ - soft_occ.detach()
        else:
            occ_for_assembly = soft_occ

        if occ_input is not None and mask is not None:
            # Retain visible pixels from the input (Phase 4 MD §6: L_preserve)
            up = mask.view(z_hat.shape[0], 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
            vis = (up > 0.5).float()
            occ_for_assembly = occ_input * vis + occ_for_assembly * (1 - vis)

        l = scalar_pred[:, 0]
        h = scalar_pred[:, 1]
        r = scalar_pred[:, 2]
        geometry = assemble_metadit_geometry(occ_for_assembly, l, h, r)
        return geometry, occ_for_assembly

    def loss(self, occupancy, scalar_values, scalar_known, spectrum, mask,
             goal_mode="real"):
        """Phase-2 loss: L_JEPA + scalar L1 (on unknown positions only)."""
        out = self.forward(
            occupancy, scalar_values, scalar_known, spectrum, mask,
            goal_mode=goal_mode,
        )
        L_jepa, _ = jepa_loss(
            out["z_hat"], out["z_y_raw"], out["mask"], proj=None,
        )
        unknown = ~scalar_known  # (B, 3)
        scalar_err = (out["scalar_pred"] - scalar_values).abs() * unknown.float()
        n_unknown = unknown.sum().clamp(min=1)
        L_scalar = scalar_err.sum() / n_unknown
        L = L_jepa + L_scalar
        out["loss_components"] = {"L_jepa": L_jepa.detach().item(), "L_scalar": L_scalar.detach().item()}
        return L, out


def build_unified_model(cfg, spec_weights, device="cpu",
                        spec_config=None):
    """Build the unified JEPA model (architecture_v5.md §3.1-§3.6).

    Uses released MetaDiT spec encoder weights only where shapes genuinely permit.
    The 192-D student components (occupancy encoder, scalar encoder, fusion,
    predictor) are initialized normally — old 384-D Milestone-B weights are NOT
    loaded into the 192-D architecture.
    """
    kwargs = dict(
        hidden=cfg.get("hidden", 192),
        num_heads=cfg.get("num_heads", 6),
        geo_depth=cfg.get("geo_depth", 6),
        predictor_depth=cfg.get("predictor_depth", 8),
        goal_tokens=cfg.get("goal_tokens", 16),
        num_predictor_heads=cfg.get("num_predictor_heads", 6),
        scalar_hidden=cfg.get("scalar_hidden", 128),
        n_film_blocks=cfg.get("n_film_blocks", 6),
        spec_dim=cfg.get("spec_dim", 256),
    )
    kwargs.update(
        momentum_start=cfg.get("ema_momentum_start", 0.996),
        momentum_end=cfg.get("ema_momentum_end", 0.999),
    )

    model = UnifiedJEPA(**kwargs)
    set_spectrum_path(model, spec_weights, device)

    # Initialize EMA targets from students (NOT from old 384-D checkpoints)
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())

    model.to(device)
    return model