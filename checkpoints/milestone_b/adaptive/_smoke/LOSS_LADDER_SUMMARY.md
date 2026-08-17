# LOSS_LADDER_SUMMARY — Milestone B adaptive screening

- mask statistics: {"requested_mask_ratio": 0.5, "actual_mask_ratio_mean": 0.4326171875, "actual_mask_ratio_std": 0.038981035351753235, "actual_mask_ratio_min": 0.390625, "actual_mask_ratio_max": 0.48046875, "n_batches": 4, "n_samples": 4}
- controller: {"max_total_steps": 6, "objectives": ["jepa"], "transitions": [], "final_phase": {"objective": "jepa", "idx": 0, "stop_reason": "global_budget", "best_metric": 0.20829828083515167, "best_step": 4}}

| objective | steps | best_cos_err | repr status | proj status | goal_pairwise | stability |
|---|---|---|---|---|---|---|
| jepa | 6 | 0.208298 | WARNING | n/a | 0.9001 | stable |

**Winner (priority: no collapse > stable > meaningful improvement > goal conditioning > lower error):** {"objective": "jepa", "phase": 0, "best_cos_err": 0.20829828083515167, "representation_status": "WARNING", "selection_priority": "no-collapse>stable>improvement>goal-conditioning>lower-error"}
