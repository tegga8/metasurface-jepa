"""Block masking per design doc §2.

Mask topology matters: uniform random-pixel masking is locally solvable and would silently
invalidate the completion experiment. We therefore mask 1-4 axis-aligned rectangular blocks
on the 16x16 token grid (patch size 4 on 3x64x64 geometry), each with a minimum side length
of 3 tokens, sized to cover roughly `mask_ratio / num_blocks` of the grid.

Placement modes (per §2):
- "random": blocks placed uniformly at random (the §7.2 minimal experiment's placement).
- "half_sensitivity": each batch is random with p=0.5, otherwise blocks are placed over
  resonance-relevant regions, identified via the frozen EM surrogate's sensitivity map
  (a random-projection estimate of the input-output Jacobian per 4x4-pixel block).

M convention: 1 = visible, 0 = masked, shape (B, 16, 16), float32.
"""

import torch

DEFAULT_GRID = 16
DEFAULT_MIN_SIDE = 3
DEFAULT_K_RANGE = (1, 4)


def _block_shapes(k, ratio, grid, min_side, rng):
    """Per-batch block sizes: k axis-aligned rects each covering ~ratio*grid^2/k tokens."""
    shapes = []
    area_per_block = ratio * grid * grid / k
    for _ in range(k):
        h = int(torch.randint(min_side, grid + 1, (), generator=rng).item())
        w = int(round(area_per_block / h))
        w = max(min_side, min(grid, w))
        shapes.append((h, w))
    return shapes


def _window_scores(sens, h, w):
    """Sum sensitivity over every (h, w) window. sens: (B, grid, grid) -> (B, H', W')."""
    b, g, _ = sens.shape
    windows = sens.unfold(1, h, 1).unfold(2, w, 1)  # (B, H', W', h, w)
    return windows.sum(dim=(-1, -2))


def random_masks(rng, batch_size, ratio, grid=DEFAULT_GRID, min_side=DEFAULT_MIN_SIDE,
                 k_range=DEFAULT_K_RANGE):
    """Random-placement block masks. Returns M (B, grid, grid), 1 = visible."""
    if ratio >= 0.999:
        return torch.zeros(batch_size, grid, grid, dtype=torch.float32)
    k = int(torch.randint(k_range[0], k_range[1] + 1, (), generator=rng).item())
    shapes = _block_shapes(k, ratio, grid, min_side, rng)
    m = torch.ones(batch_size, grid, grid, dtype=torch.float32)
    for (h, w) in shapes:
        top = torch.randint(0, grid - h + 1, size=(batch_size,), generator=rng)
        left = torch.randint(0, grid - w + 1, size=(batch_size,), generator=rng)
        for b in range(batch_size):
            m[b, top[b]:top[b] + h, left[b]:left[b] + w] = 0.0
    return m


def sensitivity_masks(rng, geometry, ratio, surrogate, grid=DEFAULT_GRID,
                      min_side=DEFAULT_MIN_SIDE, k_range=DEFAULT_K_RANGE):
    """Place mask blocks over the most resonance-relevant regions per structure.

    geometry: (B, 3, 64, 64) complete structures (surrogate input convention).
    surrogate: frozen, differentiable forward EM surrogate (eval mode, params frozen).
    Returns M (B, grid, grid), 1 = visible.

    Sensitivity map: one backward pass through the frozen surrogate with a random output
    projection r ~ N(0,1) over the 602 output channels; the squared gradient per input
    pixel is an unbiased estimate of the per-pixel squared Jacobian norm
    (Hutchinson-style). Aggregated per 4x4-pixel block (i.e. per 16x16 token).
    """
    if ratio >= 0.999:
        return torch.zeros(geometry.shape[0], grid, grid, dtype=torch.float32)
    k = int(torch.randint(k_range[0], k_range[1] + 1, (), generator=rng).item())
    shapes = _block_shapes(k, ratio, grid, min_side, rng)

    geometry = geometry.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        pred = surrogate(geometry).prediction  # (B, 2, 301)
        r = torch.randn_like(pred)
        grad = torch.autograd.grad((pred * r).sum(), geometry, create_graph=False)[0]
    sq = grad.detach() ** 2                      # (B, 3, 64, 64)
    sq = sq.mean(dim=1)                          # mean over channels
    sens = sq.reshape(geometry.shape[0], grid, 4, grid, 4).mean(dim=(2, 4))  # (B, 16, 16)
    sens = sens / (sens.amax(dim=(1, 2), keepdim=True) + 1e-8)

    m = torch.ones(geometry.shape[0], grid, grid, dtype=torch.float32)
    for (h, w) in shapes:
        scores = _window_scores(sens, h, w)      # (B, H', W')
        idx = scores.view(geometry.shape[0], -1).argmax(dim=1)  # per-structure position
        top = idx // scores.shape[2]
        left = idx % scores.shape[2]
        for b in range(geometry.shape[0]):
            m[b, top[b]:top[b] + h, left[b]:left[b] + w] = 0.0
            sens[b, top[b]:top[b] + h, left[b]:left[b] + w] = 0.0  # greedy exclusion
    return m


class BlockMasker:
    """Samples block masks per §2. Placement: 'random' | 'half_sensitivity'."""

    def __init__(self, placement="random", grid=DEFAULT_GRID, min_side=DEFAULT_MIN_SIDE,
                 k_range=DEFAULT_K_RANGE, seed=0):
        assert placement in ("random", "half_sensitivity")
        self.placement = placement
        self.grid = grid
        self.min_side = min_side
        self.k_range = k_range
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.sensitivity_count = 0

    def sample(self, geometry, ratio, surrogate=None):
        """geometry only used for sensitivity placement. Returns M (B, grid, grid)."""
        b = geometry.shape[0]
        if self.placement == "random":
            return random_masks(self.rng, b, ratio, self.grid, self.min_side, self.k_range)
        use_sens = int(torch.randint(0, 2, (), generator=self.rng).item()) == 1
        if use_sens:
            assert surrogate is not None, "sensitivity placement needs the frozen surrogate"
            self.sensitivity_count += 1
            return sensitivity_masks(self.rng, geometry, ratio, surrogate, self.grid,
                                     self.min_side, self.k_range)
        return random_masks(self.rng, b, ratio, self.grid, self.min_side, self.k_range)


def apply_mask_to_pixels(G, M, grid=DEFAULT_GRID):
    """G_c = M ⊙ G in pixel space. M: (B, grid, grid), 1 = visible; output (B, 3, 64, 64)."""
    up = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2).unsqueeze(1)  # (B,1,64,64)
    return G * up
