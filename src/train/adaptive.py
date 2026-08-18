"""Adaptive loss-switching controller for the Milestone B exploratory ladder (§1–§24).

Pure decision logic, no torch-model knowledge: given the primary validation metric
(cos_err_r0.5) and the per-validation representation-health status, the controller
decides between

    continue     — healthy AND still improving (or still in warmup)
    switch       — plateau (post-warmup, sustained non-improvement), collapse
                   (sustained representation deterioration), or instability
                   (NaN/Inf), each with the configured patience
    stop         — global step budget exhausted, or no further objective in the
                   ladder (the run then reports the best method per §22)

Key rules implemented here (details in the module docstring of the ladder task):
- Patience is counted in VALIDATION evaluations, never training steps; no fixed
  phase-switch step exists — max_total_steps is only a safety ceiling.
- Improvement means new_metric < best_metric - min_delta (best starts +inf).
- A phase switches only on evidence, never on schedule.
- Transitions: collapse/unstable -> restart from the immutable base initialization
  (weights NOT reused); healthy-plateau -> continue from the phase's best checkpoint
  with optimizer/scheduler reset (EMA preserved).
- The global budget spans the whole ladder: a phase cannot reset it.

The controller is deliberately small and testable; the training engine
(src/train/engine.py + scripts/train/train_milestone_b.py) drives it.
"""


class PhaseState:
    def __init__(self, objective, idx, start_global_step):
        self.objective = objective
        self.idx = idx
        self.start_global_step = start_global_step
        self.phase_step = 0
        self.best_metric = float("inf")
        self.best_step = None
        self.best_health = None
        self.plateau_bad = 0
        self.collapse_bad = 0
        self.representation_status = None
        self.metric_history = []
        self.health_history = []
        self.stop_reason = None


class AdaptiveController:
    def __init__(self, cfg, objectives=("jepa", "jepa_vicreg", "lejepa")):
        self.max_total_steps = int(cfg.get("max_total_steps", 1800))
        self.warmup_steps = int(cfg.get("warmup_steps", 200))       # per phase
        self.plateau_patience = int(cfg.get("plateau_patience", 2))
        self.min_delta = float(cfg.get("min_delta", 1e-5))
        self.collapse_patience = int(cfg.get("collapse_patience", 2))
        self.objectives = list(objectives)
        self.phase = None
        self.transitions = []          # log of every phase transition

    # ------------------------------------------------------------------ lifecycle

    def start_phase(self, objective, global_step):
        assert objective in self.objectives
        self.phase = PhaseState(objective, self.objectives.index(objective),
                                global_step)
        return self.phase

    def enough_budget(self, global_step):
        """True if at least one more optimizer step is allowed by the global ceiling."""
        return global_step < self.max_total_steps

    def remaining_budget(self, global_step):
        return max(0, self.max_total_steps - global_step)

    # ------------------------------------------------------------------ decisions

    def on_validation(self, primary_metric, health_status, global_step):
        """Called once per validation evaluation. Returns a decision dict:
        {"action": "continue"|"switch"|"stop", "reason": ...|None, "phase": PhaseState}
        """
        ph = self.phase
        ph.metric_history.append(float(primary_metric))
        ph.health_history.append(health_status)
        ph.representation_status = health_status
        ph.phase_step = global_step - ph.start_global_step

        metric = float(primary_metric)
        if metric < ph.best_metric - self.min_delta:
            ph.best_metric = metric
            ph.best_step = global_step
            ph.plateau_bad = 0
        else:
            ph.plateau_bad += 1

        if health_status == "COLLAPSED":
            ph.collapse_bad += 1
        else:
            ph.collapse_bad = 0

        reason = None
        if ph.collapse_bad >= self.collapse_patience:
            reason = "collapse"
        elif (ph.phase_step >= self.warmup_steps
              and ph.plateau_bad >= self.plateau_patience):
            reason = "plateau"
        elif global_step >= self.max_total_steps:
            reason = "global_budget"

        if reason is None:
            return {"action": "continue", "reason": None, "phase": ph}

        next_idx = ph.idx + 1
        if next_idx >= len(self.objectives):
            ph.stop_reason = reason
            return {"action": "stop", "reason": reason, "phase": ph}

        transition = {"from": ph.objective, "next": self.objectives[next_idx],
                      "reason": reason, "global_step": global_step,
                      "restart": "base_init" if reason != "plateau" else "best_healthy"}
        self.transitions.append(transition)
        return {"action": "switch", "reason": reason, "transition": transition,
                "phase": ph}

    def on_unstable(self, global_step):
        """NaN/Inf or persistent exploding gradients -> immediate switch/stop."""
        ph = self.phase
        next_idx = ph.idx + 1
        if next_idx >= len(self.objectives):
            ph.stop_reason = "unstable"
            return {"action": "stop", "reason": "unstable", "phase": ph}
        transition = {"from": ph.objective, "next": self.objectives[next_idx],
                      "reason": "unstable", "global_step": global_step,
                      "restart": "base_init"}
        self.transitions.append(transition)
        return {"action": "switch", "reason": "unstable", "transition": transition,
                "phase": ph}

    def summary(self):
        return {"max_total_steps": self.max_total_steps,
                "objectives": self.objectives,
                "transitions": self.transitions,
                "final_phase": None if self.phase is None
                else {"objective": self.phase.objective, "idx": self.phase.idx,
                      "stop_reason": self.phase.stop_reason,
                      "best_metric": self.phase.best_metric,
                      "best_step": self.phase.best_step}}

    # ------------------------------------------------------------------ resume

    def state_dict(self):
        """Full controller + running-phase state for exact resume (Bug #12):
        plateau/collapse patience counters, metric/health histories, transitions,
        and phase bookkeeping — everything a fresh controller loses."""
        ph = self.phase
        phase_sd = None if ph is None else {
            "objective": ph.objective, "idx": ph.idx,
            "start_global_step": ph.start_global_step,
            "phase_step": ph.phase_step,
            "best_metric": ph.best_metric, "best_step": ph.best_step,
            "best_health": ph.best_health,
            "plateau_bad": ph.plateau_bad, "collapse_bad": ph.collapse_bad,
            "representation_status": ph.representation_status,
            "metric_history": list(ph.metric_history),
            "health_history": list(ph.health_history),
            "stop_reason": ph.stop_reason,
        }
        return {
            "max_total_steps": self.max_total_steps,
            "warmup_steps": self.warmup_steps,
            "plateau_patience": self.plateau_patience,
            "min_delta": self.min_delta,
            "collapse_patience": self.collapse_patience,
            "objectives": list(self.objectives),
            "transitions": list(self.transitions),
            "phase": phase_sd,
        }

    def load_state_dict(self, sd):
        self.max_total_steps = int(sd.get("max_total_steps", self.max_total_steps))
        self.warmup_steps = int(sd.get("warmup_steps", self.warmup_steps))
        self.plateau_patience = int(sd.get("plateau_patience", self.plateau_patience))
        self.min_delta = float(sd.get("min_delta", self.min_delta))
        self.collapse_patience = int(sd.get("collapse_patience", self.collapse_patience))
        self.objectives = list(sd.get("objectives", self.objectives))
        self.transitions = list(sd.get("transitions", self.transitions))
        ph = sd.get("phase")
        if ph is not None:
            self.phase = PhaseState(ph["objective"], ph["idx"], ph["start_global_step"])
            ps = self.phase
            ps.phase_step = int(ph.get("phase_step", 0))
            ps.best_metric = float(ph.get("best_metric", float("inf")))
            ps.best_step = ph.get("best_step")
            ps.best_health = ph.get("best_health")
            ps.plateau_bad = int(ph.get("plateau_bad", 0))
            ps.collapse_bad = int(ph.get("collapse_bad", 0))
            ps.representation_status = ph.get("representation_status")
            ps.metric_history = list(ph.get("metric_history", []))
            ps.health_history = list(ph.get("health_history", []))
            ps.stop_reason = ph.get("stop_reason")
        else:
            self.phase = None
        return self
