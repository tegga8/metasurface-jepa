"""Goal-token utilization entropy (§20.2).

Logged from Phase 2 (Milestone B) onward at every checkpoint on a validation batch:
    H(softmax(mean attention weight into each of the 16 goal tokens))
computed from the predictor's cross-attention over [Z_x' (64), A_goal (16)] — the
attention mass each masked structural query assigns to each goal token, averaged over
masked queries and heads, softmaxed over the 16 goal tokens, then entropy.

Low entropy (a few tokens absorbing nearly all attention) is the cheap early warning for
query-pooling collapse flagged in §3.4 — observable from epoch 1, long before the
Ablation J routing analysis in Phase 5. Averaged over all predictor blocks.
"""

import torch


def goal_token_entropy(cross_weights, mask, n_goal=16, kv_context=64):
    """cross_weights: list over blocks of (B, H, nq, kv_context+n_goal) attention weights.
    mask: (B, 256) bool, 1 = masked. Returns (H, log H) averaged over blocks."""
    hs, log_hs = [], []
    for w in cross_weights:
        w = w[..., kv_context:]                       # (B, H, nq, 16), attention into goals
        m = mask.bool()                               # (B, 256)
        per_sample = []
        for b in range(w.shape[0]):
            wb = w[b][:, m[b], :]                     # (H, n_masked, 16)
            if wb.shape[1] == 0:
                continue
            mean_w = wb.mean(dim=(0, 1))              # (16,)
            p = torch.softmax(mean_w, dim=0)
            h = -(p * torch.log(p.clamp_min(1e-9))).sum()
            per_sample.append(h)
        if not per_sample:
            return torch.tensor(0.0), torch.tensor(float("nan"))
        h = torch.stack(per_sample).mean()
        hs.append(h)
        log_hs.append(torch.log(h.clamp_min(1e-9)))
    return torch.stack(hs).mean(), torch.stack(log_hs).mean()