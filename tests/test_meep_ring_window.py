"""Task-3 window protocol: a comparison set must be able to share one window.

Why this test exists. The Meep flux monitor integrates over the whole run, so a
reference/variant pair that stops at different times is compared over different
integration windows. The task-3 scalar sweep is nothing but such pairs (same
geometry, one scalar perturbed), and nothing about a perturbed scalar changes
the source -- so any window difference is an artefact that can swamp the effect
being measured, and it explains rows that converged yet still read |T| > 1.

The properties that must hold:
  - the DEFAULT stays adaptive (one design's convergence criterion is not
    optional, and the pre-2026-10-07 fixed-1100-unit run is a known defect);
  - `--ring-fixed` is opt-in and yields a plain number for `sim.run`, so the
    window cannot depend on any per-variant convergence parameter;
  - every curve records which protocol produced it, because a fixed-window
    curve must never be compared against an adaptive one.

No Meep install is needed: the harness references meep only at call time, so a
stub module suffices to exercise the parser and the protocol selection. NOTE:
the FDTD solve itself is NOT covered here -- this test cannot show that a fixed
window produces comparable spectra, only that the harness offers one and
records it.
"""

import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Stub meep BEFORE importing the harness: meep lives in the WSL conda env and in
# the cluster kernels, not in the Windows system Python this suite runs on.
if "meep" not in sys.modules:
    _mp = types.ModuleType("meep")
    _mp.stop_when_fields_decayed = lambda **kw: (lambda sim: False)
    _mp.Ex = "Ex"
    _mp.Vector3 = lambda *a, **k: None
    sys.modules["meep"] = _mp

import _meep_novel as h  # noqa: E402


def test_default_protocol_is_adaptive():
    """No flag: the field-decay criterion stays the default (regression guard
    against silently reverting to a fixed-duration run)."""
    args = h._parse_args([])
    assert args["ring_fixed"] is None

    stop_spec, describe = h.window_plan(args)
    assert callable(stop_spec), "adaptive stop must be a callable for Meep"
    meta = describe()
    assert meta["window"] == "adaptive"
    assert meta["ringdown"] is None and meta["hit_cap"] is False


def test_ring_fixed_is_opt_in_and_numeric():
    args = h._parse_args(["--ring-fixed", "250"])
    assert args["ring_fixed"] == 250.0

    stop_spec, describe = h.window_plan(args)
    assert isinstance(stop_spec, float) and stop_spec == 250.0
    assert describe() == {"window": "fixed", "ringdown": 250.0,
                          "hit_cap": False}


def test_fixed_window_ignores_per_variant_convergence_parameters():
    """The whole point: two variants that would stop at different adaptive times
    get the SAME window when --ring-fixed is set."""
    a = h.window_plan(h._parse_args(["--ring-fixed", "200"]))
    b = h.window_plan(h._parse_args(["--ring-fixed", "200",
                                     "--decay-by", "1e-9",
                                     "--max-ring", "5000"]))
    assert a[0] == b[0] == 200.0


def test_window_name_is_the_single_source_of_truth():
    """The harness record, the resume guard and the --ref-from guard all key
    off this one field, so it must be set by the parser, not at each site."""
    assert h._parse_args([])["window_name"] == "adaptive"
    assert h._parse_args(["--ring-fixed", "100"])["window_name"] == "fixed"


def test_protocol_guard_refuses_cross_protocol_merges():
    """Resuming a fixed-window run into an adaptive file (or adopting its
    references) would divide a fixed-window flux by an adaptive one -- the exact
    artefact this protocol exists to remove. The guard must refuse it."""
    fixed = {"loss_tangent": 1e-3, "window": "fixed"}
    adaptive = {"loss_tangent": 1e-3, "window": "adaptive"}
    legacy = {"loss_tangent": 1e-3}          # written before the protocol existed

    assert h.protocol_matches(fixed, "fixed", 1e-3) is True
    assert h.protocol_matches(adaptive, "adaptive", 1e-3) is True
    assert h.protocol_matches(legacy, "adaptive", 1e-3) is True

    assert h.protocol_matches(adaptive, "fixed", 1e-3) is False
    assert h.protocol_matches(fixed, "adaptive", 1e-3) is False
    assert h.protocol_matches(legacy, "fixed", 1e-3) is False

    # the loss-tangent arm of the same guard
    assert h.protocol_matches({"loss_tangent": 2e-3, "window": "fixed"},
                              "fixed", 1e-3) is False
    # a record that constrains neither is acceptable
    assert h.protocol_matches({}, "fixed", 1e-3) is False
    assert h.protocol_matches({}, "adaptive", 1e-3) is True


def test_adaptive_record_surfaces_the_measured_ringdown():
    """The adaptive record must report when the decay test actually stopped --
    that number is what the two-pass recipe feeds back into --ring-fixed."""
    _, describe = h.window_plan(h._parse_args([]))
    # reach the closure's state the same way the stop condition would
    plan_state = describe.__closure__[0].cell_contents
    plan_state["t_end"] = 123.4
    plan_state["capped"] = True
    meta = describe()
    assert meta["ringdown"] == 123.4 and meta["hit_cap"] is True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
