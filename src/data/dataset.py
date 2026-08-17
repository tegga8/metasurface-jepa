"""MetaDiT dataset loader for JEPA training (design doc §1.1, repo datapipe convention).

Each structure is a 3x64x64 geometry tensor (channel 0 = r_atom/5, channel 1 = h_atom,
channel 2 = l_lattice/3, zero where the meta-atom is absent) plus the 2x301 spectrum
([real, imag]) — exactly the SurrogateFreeFormDataset convention from
`external/metadit/datapipe.py` (the repo's tensor order is (2, 301), per Phase 0 REPORT §2).
"""

import os

import numpy as np
import torch
from scipy import io
from torch.utils.data import Dataset


class MetaDiTDataset(Dataset):
    def __init__(self, mat_path, max_samples=0, seed=0):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(mat_path)
        self.data = io.loadmat(mat_path)
        n = self.data["pattern"].shape[-1]
        if max_samples and max_samples < n:
            rng = np.random.RandomState(seed)
            self.indices = rng.choice(n, size=max_samples, replace=False)
        else:
            self.indices = np.arange(n)
        self.pattern = self.data["pattern"]          # (64, 64, N) int8
        self.parameter = self.data["parameter"]      # (N, 3)  [l_lattice, h_atom, r_atom]
        self.real = self.data["real"]                # (N, 301)
        self.imag = self.data["imag"]                # (N, 301)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        pattern = torch.from_numpy(self.pattern[:, :, idx]).float()      # (64, 64)
        params = self.parameter[idx]                                     # (3,)
        grid = torch.zeros(3, 64, 64, dtype=torch.float32)
        occ = pattern == 1.0
        grid[0][occ] = params[2] / 5.0
        grid[1][occ] = params[1]
        grid[2] = params[0] / 3.0
        spec = torch.stack([torch.from_numpy(self.real[idx].astype(np.float32)),
                            torch.from_numpy(self.imag[idx].astype(np.float32))], dim=0)
        return grid, spec


def collate_batch(batch):
    g, s = zip(*batch)
    return torch.stack(g, 0), torch.stack(s, 0)
