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

import numpy as np

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


def test_passivity_gate_ignores_the_mode_channel():
    """|T| <= 1 is an energy statement. The eigenmode coefficient is normalised
    by the mode of the geometry at the monitor, so the air reference and a
    slab-containing sample are not on a common scale and its ratio is not a
    transmission coefficient. v2's rows exceeded 1 ONLY through this channel."""
    n = 11
    flux_s = np.full(n, 1.0)
    flux_e = np.full(n, 1.0)          # ratio exactly 1, well conditioned
    a_e = np.full(n, 1.0)
    a_s = np.full(n, 1.10)            # mode channel 10 % over
    pv = h.passivity(flux_s, flux_e, a_s, a_e)
    assert pv["mode_peak"] > 1.02          # the raw number does flag it
    assert pv["peak_ratio"] > 1.02
    assert pv["passivity_ok_raw"] is False
    assert pv["passivity_ok"] is True      # the corrected verdict does not
    assert abs(pv["peak_ratio_flux"] - 1.0) < 1e-9


def test_passivity_gate_conditions_on_the_denominator():
    """The reference is an empty air cell, so its spectrum is the source's and
    spans ~4.4 decades. A max over the whole band divides by ~1e-4 of the band
    centre at the edges, where the ratio is noise over noise -- measured, that is
    what produced the 1.17-2.11 peaks. The corrected verdict must ignore those
    frequencies and report how much of the band it dropped."""
    n = 101
    freq = np.linspace(0.1, 0.2, n)
    # source-shaped denominator: tiny at the edges, large in the middle
    flux_e = 1e-4 + np.exp(-((freq - 0.15) / 0.02) ** 2)
    flux_s = 1.0 * flux_e                     # physically identical sample
    flux_s[0] = 9.0 * flux_e[0]               # amplitude ratio 3.0, at a dead edge
    a_e = np.ones(n)
    a_s = np.ones(n)
    pv = h.passivity(flux_s, flux_e, a_s, a_e)
    assert pv["peak_ratio"] > 2.0, "the raw max is dominated by the dead edge"
    assert pv["passivity_ok_raw"] is False
    assert pv["peak_ratio_flux"] < 1.02, "conditioned verdict ignores the edge"
    assert pv["passivity_ok"] is True
    assert 0.0 < pv["null_excluded_frac"] < 0.5
    assert pv["null_floor"] == h.NULL_FLOOR


def test_passivity_gate_still_fails_a_real_violation():
    """The correction must not become a way to pass: a violation inside the
    well-conditioned part of the band is still a failure."""
    n = 101
    freq = np.linspace(0.1, 0.2, n)
    flux_e = 1e-4 + np.exp(-((freq - 0.15) / 0.02) ** 2)
    flux_s = 1.30 * flux_e                    # 30 % over, mid-band
    pv = h.passivity(flux_s, flux_e, np.ones(n), np.ones(n))
    assert pv["peak_ratio_flux"] > 1.02
    assert pv["passivity_ok"] is False
    assert pv["passivity_ok_raw"] is False


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
