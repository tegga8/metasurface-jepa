"""Scalar encoder (architecture_v5.md §3.2, Phase 1 MD §5).

Processes the three global scalar parameters (l_lattice, h_atom, r_atom) as a
single joint input, preserving their true interaction natively — no attention-based
reconstruction of joint structure from marginal tokens.

Input:  [B, 6] = [l_value, l_known, h_value, h_known, r_value, r_known]
        (value is 0.0 when unknown; an explicit known-flag per scalar prevents
         the missingness signal from degrading if the sentinel drifts).

Trunk: Linear(6, 128) -> GELU -> Linear(128, 128).

Outputs:
  - per-block FiLM parameters (gamma, beta) for the OccupancyEncoder's n_blocks,
    each (B, 192). FiLM heads are zero-initialized to identity: gamma -> 1,
    beta -> 0 (AdaLN-zero convention, matching GCLCT).
  - a pooled scalar-summary token (B, 1, 192) for the fusion transformer.

Two copies exist at integration time (Phase 2): a live scalar MLP (used by the
predictor and scalar-decode heads) and `scalar_mlp_ema` (a deepcopy used ONLY to
condition the EMA target encoder's FiLM — never a loss target).
"""

import torch
from torch import nn

SCALAR_INPUT_DIM = 6  # l_val, l_known, h_val, h_known, r_val, r_known
SCALAR_TRUNK_DIM = 128


class ScalarEncoder(nn.Module):
    """Joint scalar MLP producing FiLM parameters and a summary token.

    Args:
        hidden:         token embedding dimension (192, must match OccupancyEncoder).
        scalar_hidden:  MLP trunk width (128).
        n_film_blocks:  number of occupancy-encoder blocks to produce FiLM for.
    """

    def __init__(self, hidden=192, scalar_hidden=SCALAR_TRUNK_DIM, n_film_blocks=6):
        super().__init__()
        self.hidden = hidden
        self.scalar_hidden = scalar_hidden

        self.trunk = nn.Sequential(
            nn.Linear(SCALAR_INPUT_DIM, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, scalar_hidden),
        )

        # Per-block FiLM heads. Zero-init to identity:
        #   weight -> 0, bias split as [gamma_bias=1, beta_bias=0]
        # so head output is (1, ..., 1, 0, ..., 0) -> gamma=1, beta=0 -> identity.
        self.film_heads = nn.ModuleList(
            [nn.Linear(scalar_hidden, 2 * hidden) for _ in range(n_film_blocks)]
        )
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[:hidden].fill_(1.0)  # gamma portion

        # Pooled scalar-summary token: projection from trunk dim to hidden.
        self.summary_head = nn.Linear(scalar_hidden, hidden)

    def forward(self, scalar_input):
        """Args:
            scalar_input: (B, 6) — [l_val, l_known, h_val, h_known, r_val, r_known].

        Returns:
            film_params: list of (gamma, beta) tuples, n_film_blocks entries,
                         each (B, hidden).
            scalar_summary: (B, 1, hidden) pooled token for the fusion encoder.
        """
        trunk_out = self.trunk(scalar_input)  # (B, scalar_hidden)

        film_params = []
        for head in self.film_heads:
            gb = head(trunk_out)  # (B, 2*hidden)
            gamma, beta = gb.chunk(2, dim=-1)  # each (B, hidden)
            film_params.append((gamma, beta))

        scalar_summary = self.summary_head(trunk_out).unsqueeze(1)  # (B, 1, hidden)

        return film_params, scalar_summary
