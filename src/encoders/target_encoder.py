"""Target encoder (§3.3, Variant A — EMA-JEPA): exponential moving average copy of the
geometry encoder weights; stop-gradient through Z_y.

ξ <- m·ξ + (1-m)·θ with m = 0.996 ramped linearly toward 0.999 over training (design doc
§3.3: "initial momentum m = 0.996 with a schedule toward 0.999 if stable").
"""

import copy

import torch
from torch import nn


class EMAEncoder(nn.Module):
    def __init__(self, source_encoder, momentum_start=0.996, momentum_end=0.999):
        super().__init__()
        self.target = copy.deepcopy(source_encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.total_steps = 1

    def set_total_steps(self, n):
        self.total_steps = max(1, int(n))

    def current_momentum(self, step):
        frac = min(1.0, max(0.0, step / self.total_steps))
        return self.momentum_start + (self.momentum_end - self.momentum_start) * frac

    @torch.no_grad()
    def update(self, student, step):
        m = self.current_momentum(step)
        for p_t, p_s in zip(self.target.parameters(), student.parameters()):
            p_t.lerp_(p_s, 1.0 - m)

    def forward(self, G):
        return self.target(G)
