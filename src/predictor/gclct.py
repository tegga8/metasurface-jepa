"""Goal-Conditioned Latent Completion Transformer (§3.5) — Milestone B dense-only slice.

The predictor consumes exactly:
    256 context geometry tokens (z_x)            — from the masked context encoder
  + 16 structured physics/goal tokens (a_goal)   — retained inside kv (Route A)
  + a global physics condition (c_physics)       — FiLM-modulates every block (Route B)
and produces:
    256x384 latent prediction z_hat

Per block (dense attention; sparse top-k routing and classifier-free guidance are
Milestones E/D respectively):

    affine-less LayerNorm -> FiLM(c_physics) -> self-attention among latent queries
    affine-less LayerNorm -> FiLM(c_physics) -> cross-attention (q = queries,
                                                 kv = [z_x (256), a_goal (16)])
    affine-less LayerNorm -> FiLM(c_physics) -> MLP

with a residual sum around each sub-layer.

Route A preserves detailed spectral structure as attention tokens. Route B makes
physics an explicit computational dependency of every block: the FiLM conditioner
is zero-initialized so the block starts as an identity modulation and must learn
to use c_physics. There is no Perceiver bottleneck, no base+delta prediction, and
no pixel head in this path.
"""

import torch
from torch import nn

from encoders.blocks import CrossAttention, Attention


class GCLCTBlock(nn.Module):
    def __init__(self, hidden=384, num_heads=6, mlp_ratio=4.0, scalar_film=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.self_attn = Attention(hidden, num_heads)

        self.norm2 = nn.LayerNorm(
            hidden,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.cross_attn = CrossAttention(hidden, num_heads)

        self.norm3 = nn.LayerNorm(
            hidden,
            elementwise_affine=False,
            eps=1e-6,
        )

        hidden_mlp = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden_mlp),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_mlp, hidden),
        )

        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden),
        )

        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

        # Step 2 scalar-predictor FiLM: an ADDITIONAL zero-init modulation driven by
        # the LEARNED scalar representation (the scalar-summary token), never raw
        # (l,h,r). Constructed only when enabled so the flag-off model is exactly the
        # current architecture (no extra params, byte-identical state_dict).
        if scalar_film:
            self.cond_scalar = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden, 6 * hidden),
            )
            nn.init.zeros_(self.cond_scalar[-1].weight)
            nn.init.zeros_(self.cond_scalar[-1].bias)
        else:
            self.cond_scalar = None

    def forward(self, x, kv, c_physics, need_weights=False, scalar_cond=None):
        """c_physics: (B, hidden) global physics condition — REQUIRED, no dead arg.
        scalar_cond: (B, hidden) LEARNED scalar representation — optional extra FiLM
        (Step 2); None ⇒ exactly the current architecture.

        FiLM groups gamma1/beta1..gamma3/beta3 applied after each affine-less
        LayerNorm; zero-initialized cond makes the modulation an identity at init.
        """
        gamma1, beta1, gamma2, beta2, gamma3, beta3 = \
            self.cond(c_physics).chunk(6, dim=-1)
        if scalar_cond is not None and self.cond_scalar is not None:
            sg1, sb1, sg2, sb2, sg3, sb3 = \
                self.cond_scalar(scalar_cond).chunk(6, dim=-1)
            gamma1, beta1 = gamma1 + sg1, beta1 + sb1
            gamma2, beta2 = gamma2 + sg2, beta2 + sb2
            gamma3, beta3 = gamma3 + sg3, beta3 + sb3

        h = self.norm1(x)
        h = h * (1.0 + gamma1[:, None, :]) + beta1[:, None, :]
        x = x + self.self_attn(h)

        h = self.norm2(x)
        h = h * (1.0 + gamma2[:, None, :]) + beta2[:, None, :]

        cross_out, weights = self.cross_attn(
            h,
            kv,
            need_weights=need_weights,
        )
        x = x + cross_out

        h = self.norm3(x)
        h = h * (1.0 + gamma3[:, None, :]) + beta3[:, None, :]
        x = x + self.mlp(h)

        return x, weights


class GCLCT(nn.Module):
    def __init__(self, depth=8, hidden=384, num_heads=6, c_physics_dim=None,
                 scalar_film=False):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        c_physics_dim = c_physics_dim if c_physics_dim is not None else hidden
        self.c_physics_dim = c_physics_dim
        if c_physics_dim != hidden:
            self.c_phys_proj = nn.Linear(c_physics_dim, hidden)
        else:
            self.c_phys_proj = nn.Identity()
        self.blocks = nn.ModuleList(
            [GCLCTBlock(hidden, num_heads, scalar_film=scalar_film)
             for _ in range(depth)])
        self.scalar_film = scalar_film
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(hidden, hidden, bias=True)

    def forward(self, queries, kv, c_physics, need_weights=False, scalar_cond=None):
        """queries: (B, T_q, hidden); kv: (B, T_kv, hidden);
        c_physics: (B, c_physics_dim) — projected to hidden via c_phys_proj
        when c_physics_dim != hidden (architecture_v5.md §3.5).
        scalar_cond: (B, hidden) LEARNED scalar representation for the Step-2 FiLM.

        Returns z_hat (B, T_q, hidden) and per-block cross-attention weights (list of
        (B, H, T_q, T_kv) tensors) when need_weights=True.
        """
        c_physics = self.c_phys_proj(c_physics)
        x = queries
        weights = []
        for block in self.blocks:
            x, w = block(x, kv, c_physics, need_weights=need_weights,
                         scalar_cond=scalar_cond)
            if need_weights:
                weights.append(w)
        return self.head(self.final_norm(x)), weights