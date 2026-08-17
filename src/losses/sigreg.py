"""SIGReg-style sliced Gaussianity regularization (adaptive-ladder Phase 2 / LeJEPA).

Design doc §3.3 Variant B: "Remove the EMA/teacher copy; impose SIGReg-style
distribution regularization on the embeddings directly", with the precision note that
SIGReg names a FAMILY of tests and that the specific test choice plus num_slices and
num_points are reportable hyperparameters (§3.3, Ablation E requirement).

The canonical LeJEPA reference repo (github.com/galilai-group/lejepa) is fetched only
at Milestone G per AGENTS.md sequencing; this ladder phase therefore implements the
smallest specification-consistent member of the family and records the exact math so
the choice can be compared against the canonical implementation later:

    Test  : sliced empirical-characteristic-function (ECF) quadratic-distance
            Gaussianity test — EP-style family (Epps & Pulley's test is the
            canonical ECF-based normality test; a quadratic distance over a fixed
            t-grid is the ECF test's differentiable form).
    Slices: num_slices random unit directions U_s ~ normalized N(0, I), drawn with a
            fixed seed and REPORTED (they are part of the method, not an implicit
            default).
    Points: num_points samples per slice, subsampled uniformly at random from the
            masked-token features when N > num_points.
    Slice : z_s = z·U_s, standardized to zero mean / unit std per slice.
    Loss  : L_s = (1/|T|)·sum_{t in T} |phi_hat(t) - exp(-t^2/2)|^2
            where phi_hat(t) = (1/n)·sum_j exp(i·t·z_sj) is the empirical
            characteristic function of the slice and exp(-t^2/2) is the ECF of
            N(0,1); T = fixed grid {0.25, 0.5, 1.0, 1.5, 2.0}.

Every hyperparameter above is returned in `info` for the per-phase report.
"""

import torch
import torch.nn.functional as F

DEFAULT_T_GRID = (0.25, 0.5, 1.0, 1.5, 2.0)


def sigreg_loss(z, num_slices=8, num_points=256, t_grid=DEFAULT_T_GRID, seed=0):
    """z: (N, D) projected features (masked tokens across the batch) -> (loss, info).

    The slice directions are drawn once per call from a fixed-seed generator and are
    identical across calls with the same seed — deterministic given the seed, which
    is recorded in `info` for the report.
    """
    n, d = z.shape
    if n == 0:
        return z.new_zeros(()), {"num_slices": num_slices, "num_points": 0,
                                 "t_grid": list(t_grid), "seed": seed}
    gen = torch.Generator().manual_seed(seed)
    U = F.normalize(torch.randn(num_slices, d, generator=gen), dim=-1)  # (S, D)
    proj = z @ U.T                                                    # (N, S)
    if n > num_points:
        idx = torch.randperm(n, generator=gen)[:num_points]
        proj = proj[idx]
    proj = (proj - proj.mean(dim=0)) / proj.std(dim=0, unbiased=True).clamp_min(1e-6)
    t = torch.tensor(list(t_grid), dtype=z.dtype, device=z.device)    # (|T|,)
    phi = torch.exp(1j * t[:, None] * proj.T).mean(dim=-1)            # (|T|, S)
    target = torch.exp(-(t ** 2) / 2)[:, None]                        # (|T|, 1)
    loss = ((phi - target).abs() ** 2).mean()
    info = {"test": "sliced ECF quadratic-distance Gaussianity (EP-style family)",
            "num_slices": num_slices, "num_points": min(n, num_points),
            "t_grid": list(t_grid), "seed": seed,
            "mean_phi_dev": (phi - target).abs().mean().item()}
    return loss, info
