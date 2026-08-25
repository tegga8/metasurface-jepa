"""Deterministic epoch sampler for exact resume (hardening spec §3).

Ensures that the same epoch reconstructs the same permutation after process restart,
enabling exact resume of DataLoader with shuffle.
"""

import torch
from torch.utils.data import Sampler


class DeterministicEpochSampler(Sampler[int]):
    """Sampler that generates a deterministic permutation per epoch.

    The permutation is derived from a base seed + epoch number, ensuring that:
    - The same epoch always produces the same permutation
    - Different epochs produce different permutations
    - Resuming from a checkpoint restores the exact epoch state
    """

    def __init__(self, data_len: int, seed: int, epoch: int = 0):
        self.data_len = int(data_len)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int):
        """Set the epoch for the next iteration."""
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.data_len, generator=g)
        return iter(perm.tolist())

    def __len__(self):
        return self.data_len