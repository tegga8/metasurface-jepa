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


def _draw_random_mask(rng, batch_size, ratio, grid, min_side, k_range):
    """One uncorrected random-placement draw (block masks, 1 = visible)."""
    k = int(torch.randint(k_range[0], k_range[1] + 1, (), generator=rng).item())
    shapes = _block_shapes(k, ratio, grid, min_side, rng)
    m = torch.ones(batch_size, grid, grid, dtype=torch.float32)
    for (h, w) in shapes:
        top = torch.randint(0, grid - h + 1, size=(batch_size,), generator=rng)
        left = torch.randint(0, grid - w + 1, size=(batch_size,), generator=rng)
        for b in range(batch_size):
            m[b, top[b]:top[b] + h, left[b]:left[b] + w] = 0.0
    return m


def random_masks(rng, batch_size, ratio, grid=DEFAULT_GRID, min_side=DEFAULT_MIN_SIDE,
                 k_range=DEFAULT_K_RANGE, tolerance=0.02, max_attempts=50):
    """Random-placement block masks. Returns M (B, grid, grid), 1 = visible.

    Operator decision 2026-09-13 (audit B20): the achieved masked fraction is
    CALIBRATED toward the requested ratio. A single uncorrected draw drifts —
    the min-side clamp inflates small blocks and independently placed blocks
    overlap, so the achieved coverage is neither the requested ratio nor
    predictable from it. The first draw within `tolerance` (fraction of the
    grid) is accepted; if none qualifies after `max_attempts`, the closest draw
    is returned. Every draw consumes the same generator, so a fixed seed stays
    reproducible, and the trainer logs the achieved fraction per bucket, so a
    miss is visible rather than silent.
    """
    if ratio <= 0.0:
        # 0% mask: every position visible (1 = visible, 0 = masked).
        return torch.ones(batch_size, grid, grid, dtype=torch.float32)
    if ratio >= 0.999:
        return torch.zeros(batch_size, grid, grid, dtype=torch.float32)

    best, best_err = None, None
    for _ in range(max_attempts):
        m = _draw_random_mask(rng, batch_size, ratio, grid, min_side, k_range)
        achieved = float((m < 0.5).float().mean().item())
        err = abs(achieved - ratio)
        if best is None or err < best_err:
            best, best_err = m, err
        if err <= tolerance:
            return m
    return best


def _calibrated_single(rng, ratio, grid, min_side, k_range, tolerance, max_attempts):
    """One calibrated block mask for a single sample -> (grid, grid), 1 = visible."""
    if ratio <= 0.0:
        return torch.ones(grid, grid, dtype=torch.float32)
    if ratio >= 0.999:
        return torch.zeros(grid, grid, dtype=torch.float32)
    best, best_err = None, None
    for _ in range(max_attempts):
        m = _draw_random_mask(rng, 1, ratio, grid, min_side, k_range)[0]
        achieved = float((m < 0.5).float().mean().item())
        err = abs(achieved - ratio)
        if best is None or err < best_err:
            best, best_err = m, err
        if err <= tolerance:
            return m
    return best


def random_masks_per_sample(rng, ratios, grid=DEFAULT_GRID, min_side=DEFAULT_MIN_SIDE,
                            k_range=DEFAULT_K_RANGE, tolerance=0.02, max_attempts=50):
    """Per-sample block masks with per-sample ratios (Phase 2b).

    Same calibration discipline as `random_masks`, applied per sample so a batch
    can mix easy and hard masking within one step. `ratios`: iterable of B floats.
    Returns M (B, grid, grid), 1 = visible.
    """
    masks = [_calibrated_single(rng, float(r), grid, min_side, k_range,
                                tolerance, max_attempts)
             for r in ratios]
    return torch.stack(masks, dim=0)


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
    if ratio <= 0.0:
        return torch.ones(geometry.shape[0], grid, grid, dtype=torch.float32)
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


# --- mirror-pair (leak-free) masking ----------------------------------------
#
# Why this exists: the released corpus is EXACTLY mirror-symmetric by
# construction (MetaDiT's restore_structure builds its 64x64 output by mirroring
# a 32x32 quadrant), so every design satisfies occ == fliplr(occ) == flipud(occ)
# == rot180(occ). Under independent random/block masking a masked token therefore
# very often has a VISIBLE mirror partner, and its content is then fully
# determined by the visible one -- no prediction required. That is a leakage
# path, not an inference problem, and any decoder constrained to be symmetric
# exploits it automatically (imp1's quadrant decoder does so by construction).
#
# Closing the masked set under the group {id, fliplr, flipud, rot180} removes the
# path: no masked token ever has a visible mirror partner, so masked content
# cannot be read off the visible geometry. Same calibration discipline as random
# masking -- the achieved ratio is measured and the closest draw kept, because
# orbits have size 1, 2 or 4 and can make an exact ratio unattainable (e.g. 0.75
# can jump to 1.0).

MIRROR_GROUP = ("fliplr", "flipud", "rot180")

# Scales applied to the requested ratio before drawing, because closing a draw
# under the group multiplies its coverage (up to 4x). The ladder lets the search
# land near the request even though the min-side clamp makes a block at least
# 3x3 tokens (14% closed), i.e. sub-14% steps are impossible.
_MIRROR_SCALES = (0.15, 0.25, 0.40, 0.60, 0.85)


def close_under_mirror_group(m):
    """Union of the masked set with its fliplr, flipud and rot180 images.

    m: (B, grid, grid), 1 = visible, 0 = masked. Masked is 0, so the union of the
    masked sets is the elementwise MINIMUM.
    """
    return torch.minimum(
        torch.minimum(m, torch.flip(m, dims=[2])),
        torch.minimum(torch.flip(m, dims=[1]), torch.flip(m, dims=[1, 2])))


def mirror_leak_fraction(m):
    """Fraction of masked tokens whose mirror partner is VISIBLE.

    The quantity this whole section exists to drive to zero: a masked token with
    a visible mirror partner is determined by symmetry alone.
    """
    masked = m < 0.5
    leaky = masked & torch.flip(m, dims=[1]).gt(0.5)       # flipud partner visible
    leaky = leaky | (masked & torch.flip(m, dims=[2]).gt(0.5))   # fliplr
    leaky = leaky | (masked & torch.flip(m, dims=[1, 2]).gt(0.5))  # rot180
    n = int(masked.sum().item())
    return float(leaky.sum().item()) / n if n else 0.0


def mirror_pair_masks(rng, batch_size, ratio, grid=DEFAULT_GRID,
                      min_side=DEFAULT_MIN_SIDE, k_range=DEFAULT_K_RANGE,
                      tolerance=0.02, max_attempts=40):
    """Block masks closed under the mirror group (leak fraction exactly 0).

    Selection rule: the ALL-VISIBLE candidate is never a valid answer — the
    masked-token objective is undefined without masked tokens (VICReg needs >= 2
    valid rows and the trainer refuses to substitute zero). Closure only grows
    the masked set, so the search stops at the first candidate that reaches the
    requested ratio; when orbit granularity overshoots, the closest CLOSED
    candidate is kept and the achieved fraction is reported by the caller's
    usual per-bucket logging.
    """
    if ratio <= 0.0:
        return torch.ones(batch_size, grid, grid, dtype=torch.float32)
    if ratio >= 0.999:
        return torch.zeros(batch_size, grid, grid, dtype=torch.float32)
    # Redraw-and-keep-closest, the same calibration discipline as random_masks,
    # but over the scale ladder: closure multiplies coverage, so a draw at the
    # full requested ratio systematically overshoots.
    per_scale = max(1, max_attempts // len(_MIRROR_SCALES))
    best, best_err = None, None
    for scale in _MIRROR_SCALES:
        for _ in range(per_scale):
            m = close_under_mirror_group(
                _draw_random_mask(rng, batch_size, ratio * scale, grid,
                                  min_side, k_range))
            achieved = float((m < 0.5).float().mean().item())
            if achieved <= 0.0:
                continue
            err = abs(achieved - ratio)
            if best is None or err < best_err:
                best, best_err = m, err
            if err <= tolerance:
                return best
    if best is None:                       # nothing masked: take one closed draw
        best = close_under_mirror_group(
            _draw_random_mask(rng, batch_size, ratio, grid, min_side, k_range))
    return best


def mirror_pair_masks_per_sample(rng, ratios, grid=DEFAULT_GRID,
                                 min_side=DEFAULT_MIN_SIDE,
                                 k_range=DEFAULT_K_RANGE, tolerance=0.02,
                                 max_attempts=40):
    """Per-sample mirror-closed masks for a batch with per-sample ratios."""
    masks = [mirror_pair_masks(rng, 1, float(r), grid, min_side, k_range,
                               tolerance, max_attempts)[0] for r in ratios]
    return torch.stack(masks, dim=0)


class BlockMasker:
    """Samples block masks per §2. Placement: 'random' | 'half_sensitivity' | 'mirror_pairs'."""

    def __init__(self, placement="random", grid=DEFAULT_GRID, min_side=DEFAULT_MIN_SIDE,
                 k_range=DEFAULT_K_RANGE, seed=0):
        assert placement in ("random", "half_sensitivity", "mirror_pairs")
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
        if self.placement == "mirror_pairs":
            return mirror_pair_masks(self.rng, b, ratio, self.grid, self.min_side,
                                     self.k_range)
        use_sens = int(torch.randint(0, 2, (), generator=self.rng).item()) == 1
        if use_sens:
            assert surrogate is not None, "sensitivity placement needs the frozen surrogate"
            self.sensitivity_count += 1
            return sensitivity_masks(self.rng, geometry, ratio, surrogate, self.grid,
                                     self.min_side, self.k_range)
        return random_masks(self.rng, b, ratio, self.grid, self.min_side, self.k_range)

    def sample_per_sample(self, geometry, ratios, surrogate=None):
        """Block masks with a PER-SAMPLE ratio (Phase 2b).

        random placement: one calibrated mask per sample. half_sensitivity:
        per-sample sensitivity placement is not implemented, so it falls back to
        the batch-MEAN ratio (keeps a mixed batch well-defined); the shipped
        config uses random.
        """
        if self.placement == "mirror_pairs":
            return mirror_pair_masks_per_sample(self.rng, ratios, self.grid,
                                                self.min_side, self.k_range)
        if self.placement != "random":
            mean_ratio = sum(float(r) for r in ratios) / max(1, len(list(ratios)))
            return self.sample(geometry, mean_ratio, surrogate)
        return random_masks_per_sample(self.rng, ratios, self.grid,
                                       self.min_side, self.k_range)

    def get_rng_state(self) -> bytes:
        """Get the internal torch.Generator state for checkpointing."""
        return self.rng.get_state()

    def set_rng_state(self, state: bytes) -> None:
        """Restore the internal torch.Generator state from checkpoint."""
        self.rng.set_state(state)


def apply_mask_to_pixels(G, M, grid=DEFAULT_GRID):
    """G_c = M ⊙ G in pixel space. M: (B, grid, grid), 1 = visible; output (B, 3, 64, 64)."""
    up = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2).unsqueeze(1)  # (B,1,64,64)
    return G * up
