"""Thin wrapper around MetaDiT's released spectrum encoder (design doc §1.3 / §3.4).

Loads the released `spec_encoder.pth` (prefix `context_encoder.` stripped, matching
`external/metadit/train/train_metadit.py`) into the repo's own `VanillaSpectrumEncoder`.
No MetaDiT weights are modified. Forward: (B, 2, 301) -> (B, 301, 256).
"""

import os
import sys

import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

from model.spec_encoder import VanillaSpectrumEncoder  # noqa: E402

CKPT_PREFIX = "context_encoder."


class ReleasedSpectrumEncoder(nn.Module):
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        self.encoder = VanillaSpectrumEncoder()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        stripped = {
            k[len(CKPT_PREFIX):] if k.startswith(CKPT_PREFIX) else k: v
            for k, v in ckpt.items()
        }
        self.encoder.load_state_dict(stripped, strict=True)
        self.encoder.eval().to(device)

    def forward(self, spec):
        return self.encoder(spec)


class SpectrumPath(nn.Module):
    """Design doc §3.4: frozen released spectrum encoder + goal-token pooling.

    S -> A_local (B, 301, 256); A_g = MeanPool(A_local) projected to 384 dims (global
    physics condition for AdaLN-Zero); A_goal = 16 learned queries cross-attending over
    A_local, projected to 384 dims (fine spectral structure). With `goal_mode='null'`
    both are replaced by zeros — the §7.2 cheap proxy for goal-ignoring collapse
    (Failure Mode 2), available without the full goal-dropout/CFG machinery (§3.5.1).
    """

    def __init__(self, released_encoder, spec_dim=256, hidden=384, goal_tokens=16,
                 num_heads=4):
        super().__init__()
        self.released = released_encoder
        if self.released is not None:
            for p in self.released.parameters():
                p.requires_grad_(False)
            self.released.eval()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = spec_dim // num_heads
        self.goal_queries = nn.Parameter(torch.zeros(1, goal_tokens, spec_dim))
        nn.init.normal_(self.goal_queries, std=0.02)
        self.q = nn.Linear(spec_dim, spec_dim, bias=True)
        self.kv = nn.Linear(spec_dim, spec_dim * 2, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(spec_dim, spec_dim)
        self.proj_g = nn.Linear(spec_dim, hidden)
        self.proj_goal = nn.Linear(spec_dim, hidden)

    def _pool_goal(self, a_local):
        """16 learned queries cross-attend over A_local (B, 301, 256) -> (B, 16, 256)."""
        b, nq, _ = self.goal_queries.shape
        nk = a_local.shape[1]
        q = self.q(self.goal_queries.expand(b, -1, -1))
        k, v = torch.chunk(self.kv(a_local), 2, dim=-1)
        q = q.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, nq, self.num_heads * self.head_dim)
        return self.proj(out)

    def forward(self, S, goal_mode="real"):
        """S: (B, 2, 301) -> (c_physics (B, 384), A_goal (B, 16, 384))."""
        with torch.no_grad():
            a_local = self.released(S)                       # (B, 301, 256)
        if goal_mode == "null":
            b = S.shape[0]
            zeros = a_local.new_zeros(b, self.hidden)
            return zeros, zeros.unsqueeze(1).expand(b, self.goal_queries.shape[1], -1)
        a_g = self.proj_g(a_local.mean(dim=1))               # (B, 384)
        a_goal = self.proj_goal(self._pool_goal(a_local))    # (B, 16, 384)
        return a_g, a_goal
