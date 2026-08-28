"""Independent scalar known/unknown masking (architecture_v5.md §3.2, §6.2, Phase 1 MD §3).

Scalar masking is independent of occupancy spatial masking: a scalar can be fully
unknown even when its geometry region is visible, and vice versa. Each scalar carries
an explicit binary known/unknown flag so missingness is never inferred from a numeric
sentinel that could drift into the range of plausible real values.

Supported regimes (configurable per-batch sample):
  - "all_known"       : every scalar is known (retrofit/completion scenario)
  - "all_unknown"     : every scalar is unknown (pure inverse-design scenario)
  - "independent"     : each scalar masked independently with p_independent
  - "correlated"      : per-sample, all three known or all three unknown
                        (reflects realistic deployment: either you have a spec
                        sheet with all parameters, or none)
"""

import torch


class ScalarMasker:
    """Samples per-scalar known/unknown flags and builds masked scalar inputs.

    Args:
        regime: One of "all_known", "all_unknown", "independent", "correlated".
        p_independent: P(scalar is known) for the "independent" regime.
        p_correlated:  P(all scalars known) for the "correlated" regime.
        seed: RNG seed for reproducibility.
    """

    REGIMES = ("all_known", "all_unknown", "independent", "correlated")

    def __init__(self, regime="correlated", p_independent=0.5, p_correlated=0.5,
                 seed=0):
        if regime not in self.REGIMES:
            raise ValueError(
                f"regime must be one of {self.REGIMES}, got {regime!r}")
        if regime in ("independent", "correlated"):
            assert 0.0 <= p_independent <= 1.0, "p_independent must be in [0, 1]"
            assert 0.0 <= p_correlated <= 1.0, "p_correlated must be in [0, 1]"
        self.regime = regime
        self.p_independent = p_independent
        self.p_correlated = p_correlated
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def sample(self, scalar_values):
        """Sample known/unknown flags for a batch.

        Args:
            scalar_values: (B, 3) tensor of [l, h, r] physical values.

        Returns:
            masked_values: (B, 3) — value where known, 0.0 where unknown.
            known_flags:   (B, 3) bool — True where the scalar is known.
        """
        b = scalar_values.shape[0]

        if self.regime == "all_known":
            known = torch.ones(b, 3, dtype=torch.bool, device=scalar_values.device)
        elif self.regime == "all_unknown":
            known = torch.zeros(b, 3, dtype=torch.bool, device=scalar_values.device)
        elif self.regime == "independent":
            known = torch.rand(b, 3, generator=self.rng,
                               device=scalar_values.device) < self.p_independent
        elif self.regime == "correlated":
            all_known = torch.rand(b, generator=self.rng,
                                   device=scalar_values.device) < self.p_correlated
            known = all_known.unsqueeze(1).expand(b, 3)
        else:
            raise ValueError(f"unknown regime: {self.regime}")

        masked_values = torch.where(
            known.to(scalar_values.device),
            scalar_values,
            torch.zeros_like(scalar_values),
        )
        return masked_values, known

    def build_mlp_input(self, masked_values, known_flags):
        """Build the [B, 6] MLP input: [l_val, l_known, h_val, h_known, r_val, r_known].

        Args:
            masked_values: (B, 3) from sample().
            known_flags:   (B, 3) bool from sample().

        Returns:
            (B, 6) float tensor.
        """
        vals = masked_values.float()
        flags = known_flags.float()
        return torch.stack(
            [vals[:, 0], flags[:, 0],
             vals[:, 1], flags[:, 1],
             vals[:, 2], flags[:, 2]],
            dim=-1,
        )

    def get_rng_state(self) -> bytes:
        return self.rng.get_state()

    def set_rng_state(self, state: bytes) -> None:
        self.rng.set_state(state)
