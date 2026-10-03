"""Top-level model assembly for the unified architecture (architecture_v5.md §3.1-§3.6).

`UnifiedJEPA` is the active model: single-channel occupancy + explicit scalar
parameters (scalar-encoder FiLM + scalar-summary token) + conditioned spectrum,
all at 192-D — the frozen SpectrumPath keeps 384-D `c_physics`/`a_goal` and both
are projected down (GCLCT `c_phys_proj`, FusionEncoder `goal_proj`).
`build_unified_model` assembles the model and seeds both EMA targets.

Frozen released components stay outside the trainable state: the released spectrum
encoder keys are filtered from saved checkpoints (re-loaded from disk on every
build), and the EM surrogate is constructed by the training script on demand.
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

from encoders.occupancy_encoder import OccupancyEncoder
from encoders.scalar_encoder import ScalarEncoder
from fusion.fusion_encoder import FusionEncoder
from decoders.scalar_decoder import ScalarDecoder
from decoders.occupancy_decoder import OccupancyDecoder
from data.factorize import assemble_metadit_geometry
from encoders.spectrum_encoder import ReleasedSpectrumEncoder, SpectrumPath
from encoders.target_encoder import EMAEncoder
from losses.jepa_loss import jepa_loss
from predictor.gclct import GCLCT

PIXEL_GRID = 16  # 64 / patch_size 4


def set_spectrum_path(model, spec_weights, device):
    """Attach the released (frozen) spectrum encoder to the model's SpectrumPath.

    The freeze must happen here: SpectrumPath is constructed with released=None
    on the production path (both builders), so SpectrumPath's own freeze branch
    never runs — without this, the released encoder's parameters stay trainable
    and enter the optimizer's parameter set (audit finding B1; architecture_v5.md
    §6: released spectrum encoder is a frozen reference, never trained).
    """
    released = ReleasedSpectrumEncoder(spec_weights, device=device)
    for p in released.parameters():
        p.requires_grad_(False)
    released.eval()
    model.spectrum_path.released = released
    model.spectrum_path.released.to(device)


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


class SpectrumFilm(nn.Module):
    """Phase 4: map c_physics [B, c_dim] to per-block (gamma, beta) FiLM.

    Applied to the TARGET occupancy encoder only, and **frozen** with a small
    non-zero init. Two properties matter:
      - non-zero init -> the target is genuinely spectrum-dependent at step 0
        (a zero-init-to-identity film makes z_y_occ_spec == z_y_raw initially);
      - frozen       -> the optimization cannot drive the conditioning back to
        identity. A trainable conditioning is NULLIFIABLE: the loss can be
        reduced by making the target spectrum-free again, which defeats the point
        of forcing spectrum use.
    gamma is biased to ~1 so the modulation is a small perturbation at init.
    """

    def __init__(self, c_dim=384, hidden=192, n_blocks=6, frozen=True):
        super().__init__()
        self.hidden = hidden
        self.heads = nn.ModuleList(
            [nn.Linear(c_dim, 2 * hidden) for _ in range(n_blocks)])
        for head in self.heads:
            # std 0.02 gave only ~0.04 cross-spectrum cosine distance at init (the
            # target was ~96 % identical across spectra -> L_cond near-redundant;
            # measured by scripts/diagnostics/spectrum_film_separation.py). 0.1
            # raises it to ~0.49 while staying below the regime where an arbitrary
            # transform dominates the target.
            nn.init.normal_(head.weight, std=0.1)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[:hidden].fill_(1.0)   # gamma ≈ 1 at init
            if frozen:
                for p in head.parameters():
                    p.requires_grad_(False)

    def forward(self, c_physics):
        out = []
        for head in self.heads:
            gamma, beta = head(c_physics).chunk(2, dim=-1)
            out.append((gamma, beta))
        return out


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
                 momentum_start=0.996, momentum_end=0.999,
                 scalar_predictor_film=False):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.goal_tokens = goal_tokens
        self.architecture_id = UNIFIED_ARCHITECTURE_ID
        # Step 2: route the LEARNED scalar representation into an explicit predict-
        # or FiLM. Off (default) ⇒ exactly the current architecture.
        self.scalar_predictor_film = bool(scalar_predictor_film)

        # Audit B18: the scalar encoder's FiLM heads must match the occupancy
        # encoder's block count — a mismatch otherwise surfaces as an opaque
        # IndexError mid-forward (or a silently dropped block).
        assert n_film_blocks == geo_depth, (
            f"n_film_blocks={n_film_blocks} must equal geo_depth={geo_depth} "
            "(one FiLM pair per occupancy-encoder block)")

        # Student encoders
        self.occupancy_encoder = OccupancyEncoder(
            hidden=hidden, num_heads=num_heads, depth=geo_depth
        )
        self.scalar_encoder = ScalarEncoder(
            hidden=hidden, scalar_hidden=scalar_hidden, n_film_blocks=n_film_blocks
        )

        # Phase 4: a FROZEN spectrum-conditioned FiLM for the TARGET encoder (a
        # non-nullifiable conditioning), plus the head that predicts that target
        # from the predictor trunk. The student encoder is left spectrum-agnostic
        # (it receives the goal through fusion/predictor as before).
        self.spectrum_film = SpectrumFilm(
            c_dim=384, hidden=hidden, n_blocks=n_film_blocks, frozen=True)
        self.spec_proj = nn.Linear(hidden, hidden, bias=True)

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
            c_physics_dim=384, scalar_film=scalar_predictor_film,
        )

        # Scalar decode heads
        self.scalar_decoder = ScalarDecoder(hidden=hidden)

        # Occupancy decoder: latent → occupancy logits only, FiLM-conditioned
        # by effective (l,h,r) at every layer (architecture_v5.md §4.1).
        # No 3-channel geometry head: the broadcast tensor is assembled once,
        # at the physics-surrogate boundary.
        self.occupancy_decoder = OccupancyDecoder(
            hidden=hidden, base_dim=hidden // 2, scalar_hidden=scalar_hidden,
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

        # 3. Spectrum path (frozen released encoder + trainable read-out)
        c_physics, a_goal = self.spectrum_path(spectrum, goal_mode=goal_mode)
        # c_physics: (B, 384), a_goal: (B, 16, 384)

        # 4. Occupancy encoder (student) with TOKEN-level mask replacement + FiLM.
        # Phase 3a: the raw pixel mask was removed (redundant — patch_embed is a
        # non-overlapping stride-4 conv, and masked tokens are overwritten with
        # `mask_token`). The student encoder is spectrum-agnostic; Phase 4 moved the
        # spectrum conditioning onto the TARGET encoder only.
        z_x = self.occupancy_encoder(
            occupancy, film_params=film_params,
            mask=mask, mask_token=self.mask_token,
        )  # (B, 256, hidden)

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

        # 7. Predictor (c_physics 384→192 via c_phys_proj; audit B16: need_attn
        #    returns the per-block cross-attention weights instead of being
        #    silently ignored)
        # Step 2: the LEARNED scalar representation (summary token) conditions the
        # predictor via FiLM when enabled — not raw (l,h,r) values.
        scalar_cond = (scalar_summary.reshape(scalar_summary.shape[0], -1)
                       if self.scalar_predictor_film else None)
        z_hat_raw, attn_weights = self.predictor(
            queries, fused, c_physics, need_weights=need_attn,
            scalar_cond=scalar_cond)  # (B, 257, hidden)

        # 8. Split predictions
        occupancy_pred = z_hat_raw[:, :256, :]         # (B, 256, hidden)
        scalar_summary_pred = z_hat_raw[:, 256, :]     # (B, hidden)

        # 8b. Phase 4: the spectrum-conditioned target head (a projection off the
        #     predictor trunk) and the scalar-latent prediction.
        z_hat_occ_spec = self.spec_proj(occupancy_pred)   # (B, 256, hidden)
        z_hat_scal = scalar_summary_pred                  # (B, hidden)

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
            z_hat_occ_spec=z_hat_occ_spec,
            z_hat_scal=z_hat_scal,
            attn_weights=(attn_weights if need_attn else None),
            # The scalar encoder's pooled conditioning token, exposed so the
            # objective can attach its own read-out to it (door (a) of the scalar
            # investigation): the token had no objective of its own, so nothing
            # pushed it to carry the scalars.
            scalar_summary=scalar_summary,
        )

        if with_target:
            with torch.no_grad():
                # True scalars (all known) for target-side FiLM
                true_input = torch.stack([
                    scalar_values[:, 0], torch.ones_like(scalar_values[:, 0]),
                    scalar_values[:, 1], torch.ones_like(scalar_values[:, 1]),
                    scalar_values[:, 2], torch.ones_like(scalar_values[:, 2]),
                ], dim=-1)  # (B, 6)
                film_params_ema, scalar_summary_ema = self.scalar_mlp_ema(true_input)
                # Stable, spectrum-FREE geometry target (the answer key + the
                # clean real/null/shuffled control), unchanged from v5.
                z_y_raw = self.ema(occupancy, film_params=film_params_ema)
                # Phase 4 spectrum-conditioned geometry target: the teacher's
                # occupancy encoder is conditioned on the TRUE spectrum by a FROZEN
                # film, so matching it REQUIRES using the goal spectrum and the
                # conditioning cannot be optimized away.
                spec_film = self.spectrum_film(c_physics)
                z_y_occ_spec = self.ema(
                    occupancy, film_params=film_params_ema,
                    spectrum_film_params=spec_film)
                # Phase 4 scalar latent target (EMA scalar encoder summary of the
                # true scalars) — forces the scalar path to carry the conditioning.
                z_y_scal = scalar_summary_ema.reshape(b, -1)
                out["z_y_raw"] = z_y_raw
                out["z_y_occ_spec"] = z_y_occ_spec
                out["z_y_scal"] = z_y_scal
                # Explicit feature-wise normalization boundary (documented
                # contract, audit B15): the active UnifiedJEPALoss consumes
                # z_y_raw through its own objective-owned projector, so this
                # export is currently unused inside the repo — it exists as the
                # declared boundary, not as a consumed input.
                out["z_y_normalized"] = F.layer_norm(
                    z_y_raw, (z_y_raw.shape[-1],)
                )
                out["z_y"] = z_y_raw  # compat alias

        return out

    def _effective_scalars(self, scalar_pred, scalar_known=None,
                           scalar_values=None):
        """Decode-time scalar rule (architecture_v5.md §4.1): the true value where
        the scalar is known, the prediction where unknown — identical in training
        and inference."""
        if scalar_known is not None and scalar_values is not None:
            return torch.where(scalar_known, scalar_values, scalar_pred)
        return scalar_pred

    def _effective_scalar_input(self, scalar_pred, scalar_known=None,
                                scalar_values=None):
        """6-dim decoder conditioning ([value, known-flag] x 3, §3.2 convention).

        Values follow the §4.1 decode-time rule: the true value where known, the
        prediction where unknown. The flags distinguish a known value from a
        predicted value of the same magnitude (audit B10). ``scalar_known=None``
        means no scalar is observed, so all flags are 0.
        """
        values = self._effective_scalars(scalar_pred, scalar_known,
                                         scalar_values)
        if scalar_known is None:
            flags = torch.zeros_like(values)
        else:
            flags = scalar_known.to(values.dtype)
        return torch.stack([
            values[:, 0], flags[:, 0],
            values[:, 1], flags[:, 1],
            values[:, 2], flags[:, 2],
        ], dim=-1)

    def decode_occupancy_logits(self, z_hat, scalar_pred, scalar_known=None,
                                scalar_values=None):
        """Raw occupancy logits [B,1,64,64] with §3.2 conditioning (audit B10)."""
        scalars = self._effective_scalar_input(scalar_pred, scalar_known,
                                               scalar_values)
        return self.occupancy_decoder(z_hat, scalars)

    def decode_occupancy_prob(self, z_hat, scalar_pred, scalar_known=None,
                              scalar_values=None):
        """Raw sigmoid occupancy probability [B,1,64,64] — no thresholding, no
        visible-pixel retention.

        This is the occupancy-QUALITY diagnostic branch (IoU/F1/fraction in the
        evaluator); it is never the deployed geometry — deployment uses
        decode_geometry's retained, hard-thresholded occupancy (audit B7).
        """
        return torch.sigmoid(self.decode_occupancy_logits(
            z_hat, scalar_pred, scalar_known, scalar_values))

    def decode_geometry(self, z_hat, scalar_pred, occ_input=None, mask=None,
                        scalar_known=None, scalar_values=None, use_ste=False,
                        hard_forward=False):
        """Decode predicted latents to surrogate-ready geometry (Phase 4 MD §1-§3,
        architecture_v5.md §4.1).

        Args:
            z_hat:       [B, 256, hidden] predicted occupancy latents.
            scalar_pred: [B, 3] (l_lattice, h_atom, r_atom) physical values.
            occ_input:   [B, 1, 64, 64] original binary occupancy (for retention).
            mask:        [B, 16, 16] 1=visible, 0=masked — retains visible pixels.
            scalar_known: [B, 3] bool — which scalars are observed. When provided
                         together with scalar_values, known scalars are
                         substituted with their true values (the scalar analog
                         of visible-occupancy retention); unknown scalars use
                         scalar_pred.
            scalar_values: [B, 3] true scalar values (used only where
                         scalar_known is True).
            use_ste:     If True (AND self.training), use hard occupancy for the
                         surrogate input with a soft backward path
                         (straight-through estimator). Training behavior only.
            hard_forward: If True, threshold occupancy to binary for the forward
                         geometry regardless of training mode (used by the
                         soft-vs-hard diagnostic — see physics_loop).

        Returns:
            geometry:     [B, 3, 64, 64] — r_atom/5, h_atom, l_lattice/3.
            occ_delivered: [B, 1, 64, 64] — the occupancy actually handed to
                          the assembler: raw sigmoid (default), STE
                          hard-forward/soft-backward (use_ste, training only),
                          or hard-thresholded (hard_forward), with visible
                          pixels retained from occ_input where supplied. This
                          is NOT the raw probability — use
                          decode_occupancy_prob for occupancy diagnostics
                          (audit B7).
        """
        # Effective scalar rule (architecture_v5.md §4.1): decode-time FiLM and
        # assembly use the true value where known, the prediction where unknown —
        # identical in training and inference.
        scalar_for_assembly = self._effective_scalars(
            scalar_pred, scalar_known, scalar_values)

        # Decoder is FiLM-conditioned by the effective (l,h,r); soft_occ is the
        # raw sigmoid probability, thresholded/retained below for deployment.
        soft_occ = self.decode_occupancy_prob(
            z_hat, scalar_pred, scalar_known, scalar_values)  # (B, 1, 64, 64)

        if use_ste and self.training:
            hard_occ = (soft_occ > 0.5).float()
            occ_for_assembly = hard_occ + soft_occ - soft_occ.detach()
        elif hard_forward:
            # Audit B18: hard_forward during training without STE would deliver
            # a zero-gradient occupancy to the surrogate (no backward path).
            # The hazard is losing a gradient, so the predicate is gradient
            # tracking, not the model's mode: under torch.no_grad() there is no
            # backward to starve and the diagnostic this message advertises is
            # legal in train mode (audit B21 — testing self.training instead
            # refused preflight's hard-assembly diagnostics and broke the gate).
            assert not self.training or not torch.is_grad_enabled(), (
                "decode_geometry: hard_forward=True on a gradient-enabled "
                "training forward requires use_ste=True — otherwise the "
                "deployed occupancy carries no gradient; run diagnostics under "
                "torch.no_grad() or eval()")
            occ_for_assembly = (soft_occ > 0.5).float()
        else:
            occ_for_assembly = soft_occ

        if occ_input is not None and mask is not None:
            # Retain visible pixels from the input (Phase 4 MD §6: L_preserve)
            up = mask.view(z_hat.shape[0], 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
            vis = (up > 0.5).float()
            occ_for_assembly = occ_input * vis + occ_for_assembly * (1 - vis)

        l = scalar_for_assembly[:, 0]
        h = scalar_for_assembly[:, 1]
        r = scalar_for_assembly[:, 2]
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
        scalar_predictor_film=cfg.get("scalar_predictor_film", False),
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