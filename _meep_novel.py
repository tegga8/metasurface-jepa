"""Meep spectra for a design set (eigenmode complex transmission).

Usage (WSL): ~/meep_env/bin/python _meep_novel.py <designs.npz> <out.json>
                [--start K] [--end K] [--loss-tangent T] [--res R]
                [--max-ring T] [--decay-by D]

The npz holds `patterns` (N x 64 x 64) and `scalars` (shared (3,) or per-item
(N, 3)). An empty-cell reference is computed once per distinct scalar combo and
stored in the output, so the ratios can be recomputed offline without re-running
FDTD. For each design the script records the flux-transmission magnitude and
the fundamental-mode transmission ratio, writes the JSON after EVERY design
(resumable view), and reports whether the fields actually decayed.

FIXED 2026-10-07. The previous version ran every design for a fixed 1100 time
units in a lossless cell and produced spectra that violate passivity
(|T| <= 1) in 36/43 designs across the symbol-decodes and batch40 kernels --
values up to |T| = 4.7e8 for a passive dielectric. Three defects caused it:

  1. NO CONVERGENCE and NO LOSS. The cell is PML-terminated in z and periodic
     in x/y, so a lossless slab traps energy in modes the z-PML cannot absorb.
     The run was simply stopped at an arbitrary time, i.e. mid-ring, and the
     flux at the monitor was still oscillating. The material is now lossy
     (--loss-tangent, default 1e-3) which makes the problem dissipative, and
     the run ends on a field-decay criterion (--decay-by) rather than a fixed
     duration, with a hard cap (--max-ring) so a kernel can never wedge.
  2. THE MODE PARITY CHANNELS WERE MEANINGLESS. The ModeRegion has zero
     thickness, so `eig_parity` is not resolvable and def / EVEN_Z / ODD_Z came
     out byte-identical for every design. Only the default channel is kept;
     analyses must not treat "oz" as a distinct symmetry channel.
  3. NO VALIDITY GATE. Each design now records its peak transmission ratio and
     a `passivity_ok` flag, and the run prints a violation count. A design that
     violates |T| <= 1 in a lossy cell is a solver failure, not physics.

The loss tangent is a deviation from the lossless CST reference the dataset was
generated with. It is bounded and configurable, and the harness must be
validated against MetaDiT's own CST spectra on real dataset geometries before
its output is used for any claim (see _meep_validate_vs_cst.py).
"""

import json
import os
import sys
import time

# meep is not installed in every environment this file is analysed from: it
# lives in the WSL conda env (~/envs/meep) and in the Kaggle/cluster kernels,
# while the editor's interpreter is the Windows system Python. The import is
# resolved at runtime, not at analysis time.
import meep as mp  # type: ignore[import-not-found]
import numpy as np

RES = 32
PML = 0.5
PAD = 0.8
FCEN, DF = 0.15, 0.1
FREQS = np.linspace(0.1, 0.2, 301)

# Convergence / loss defaults (see module docstring).
LOSS_TANGENT = 1e-3      # Im(eps)/Re(eps) of the meta-atom dielectric
DECAY_BY = 1e-5          # stop when |E|^2 falls this far below its peak
DECAY_WINDOW = 7.0       # averaging window, ~1 optical period at f = 0.15
MAX_RING = 600.0         # hard cap on post-source ringdown
PASSIVITY_TOL = 1.02     # peak ratio above this is a solver failure


def _parse_args(argv):
    a = {"start": 0, "end": None, "loss": LOSS_TANGENT, "res": RES,
         "max_ring": MAX_RING, "decay_by": DECAY_BY, "ref_from": None}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k in ("--start", "--end", "--loss-tangent", "--res", "--max-ring",
                 "--decay-by", "--ref-from"):
            if i + 1 >= len(argv):
                sys.exit(f"{k} needs a value")
            v = argv[i + 1]
            key = {"--start": "start", "--end": "end",
                   "--loss-tangent": "loss", "--res": "res",
                   "--max-ring": "max_ring", "--decay-by": "decay_by",
                   "--ref-from": "ref_from"}[k]
            if key in ("start", "end", "res"):
                a[key] = int(v)
            elif key == "ref_from":
                a[key] = v
            else:
                a[key] = float(v)
            i += 2
        else:
            sys.exit(f"unknown argument: {k}")
    return a


def make_stop_condition(decay_by, window, cap):
    """Field-decay termination with a hard cap, so a run cannot wedge.

    Meep conditions are callables taking the Simulation and returning True to
    stop. mp.stop_when_fields_decayed implements the decay test; we wrap it to
    add the cap and to record whether the decay or the cap ended the run.
    """
    inner = mp.stop_when_fields_decayed(
        dt=window, c=mp.Ex, pt=mp.Vector3(0, 0, 0), decay_by=decay_by)
    state = {"t0": None, "capped": False, "t_end": None}

    def _stop(sim):
        if state["t0"] is None:          # first call: sources have just ended
            state["t0"] = sim.round_time()
            return False
        if sim.round_time() - state["t0"] > cap:
            state["capped"] = True
            state["t_end"] = sim.round_time() - state["t0"]
            return True
        stop = inner(sim)
        if stop:
            state["t_end"] = sim.round_time() - state["t0"]
        return stop

    return _stop, state


def run_case(pat, l, h, n, args):
    """One FDTD solve. Returns (flux, mode_alpha, meta)."""
    sx = sy = l
    sz = h + 2 * PAD
    src_z = -sz / 2 + PML + 0.1
    mon_z = sz / 2 - PML - 0.1

    # Lossy dielectric. Modelled as an Ohmic-equivalent conductivity: the
    # effective permittivity is eps_eff(w) = eps + i*sigma/w, so choosing
    # sigma = w0*eps*tan(delta) reproduces the requested loss tangent at the
    # band centre. A real-valued epsilon is used deliberately: Meep 1.34
    # compares epsilon_diag against 1 while emitting a stability warning, and
    # that comparison raises TypeError for a complex epsilon (defect 1).
    eps = n ** 2
    sigma = 2 * np.pi * FCEN * eps * args["loss"]
    mat = mp.Medium(epsilon=eps, D_conductivity=sigma)

    geom = []
    px = l / 64.0
    for iy in range(64):
        row = pat[iy]
        ix = 0
        while ix < 64:
            if row[ix] != 1:
                ix += 1
                continue
            jx = ix
            while jx + 1 < 64 and row[jx + 1] == 1:
                jx += 1
            x0 = -l / 2 + ix * px
            x1 = -l / 2 + (jx + 1) * px
            y0 = -l / 2 + iy * px
            y1 = -l / 2 + (iy + 1) * px
            geom.append(mp.Block(
                size=mp.Vector3(x1 - x0, y1 - y0, h),
                center=mp.Vector3((x0 + x1) / 2, (y0 + y1) / 2, 0),
                material=mat))
            ix = jx + 1

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        resolution=args["res"],
        boundary_layers=[mp.PML(PML, direction=mp.Z)],
        k_point=mp.Vector3(0, 0, 0),
        geometry=geom,
        sources=[mp.Source(
            mp.GaussianSource(FCEN, fwidth=DF), component=mp.Ex,
            center=mp.Vector3(0, 0, src_z), size=mp.Vector3(sx, sy, 0))],
    )
    trans = sim.add_flux(FREQS, mp.FluxRegion(
        center=mp.Vector3(0, 0, mon_z), size=mp.Vector3(sx, sy, 0)))
    region = mp.ModeRegion(center=mp.Vector3(0, 0, mon_z),
                           size=mp.Vector3(sx, sy, 0))
    mon = sim.add_mode_monitor(FREQS, region)

    cond, cstate = make_stop_condition(args["decay_by"], DECAY_WINDOW,
                                       args["max_ring"])
    t0 = time.time()
    sim.run(until_after_sources=cond)
    runtime = time.time() - t0

    flux = np.array(mp.get_fluxes(trans))
    # defect 2: only the default channel is meaningful for a zero-thickness
    # ModeRegion; the parity variants were byte-identical to this one.
    alpha = sim.get_eigenmode_coefficients(mon, [1]).alpha[0, :, 0]
    meta = {"runtime_s": round(runtime, 1),
            "ringdown": None if cstate["t_end"] is None
            else round(cstate["t_end"], 1),
            "hit_cap": bool(cstate["capped"]),
            "blocks": len(geom),
            "d_conductivity": round(float(sigma), 6)}
    return flux, alpha, meta


def write_json(out, out_path):
    """Atomic write: a crash mid-write must not leave a truncated resume file."""
    tmp = out_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_path)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"could not write {out_path}: {exc}") from exc


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    npz_path, out_path = sys.argv[1], sys.argv[2]
    args = _parse_args(sys.argv[3:])
    if not os.path.exists(npz_path):
        sys.exit(f"designs npz not found: {npz_path}")
    try:
        z = np.load(npz_path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read designs npz {npz_path}: {exc}")
    missing = [k for k in ("patterns", "scalars") if k not in z]
    if missing:
        sys.exit(f"{npz_path} is missing required array(s): {missing}")
    patterns = z["patterns"]
    if patterns.ndim != 3 or patterns.shape[1:] != (64, 64):
        sys.exit(f"patterns must be (N, 64, 64), got {patterns.shape}")
    sc = z["scalars"]
    if sc.ndim not in (1, 2) or sc.shape[-1] != 3:
        sys.exit(f"scalars must be (3,) or (N, 3), got {sc.shape}")
    per_item = sc.ndim > 1
    src_idx = z["source_idx"].tolist() if "source_idx" in z else None
    names = z["names"].tolist() if "names" in z else None

    lo = args["start"]
    hi = args["end"] if args["end"] is not None else len(patterns)
    lo, hi = max(0, lo), min(len(patterns), hi)

    print(f"loaded {len(patterns)} designs; scalars "
          f"{'per-item' if per_item else 'shared'}; chunk [{lo}:{hi}] "
          f"loss={args['loss']} res={args['res']} "
          f"decay_by={args['decay_by']} max_ring={args['max_ring']}",
          flush=True)

    # Resume: keep any entries already present for this chunk.
    out = {"npz": os.path.basename(npz_path),
           "per_item_scalars": bool(per_item),
           "harness": {"res": args["res"], "loss_tangent": args["loss"],
                       "decay_by": args["decay_by"],
                       "max_ring": args["max_ring"],
                       "mode_channels": ["def"]},
           "refs": {}, "designs": []}
    prev = None
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
        except (ValueError, OSError) as exc:
            # A truncated/partial file just means we redo this chunk.
            print(f"could not resume from {out_path} ({exc}); starting fresh",
                  flush=True)
    if prev is not None:
        if prev.get("harness", {}).get("loss_tangent") == args["loss"]:
            out["refs"] = prev.get("refs", {})
            out["designs"] = [d for d in prev.get("designs", [])
                              if lo <= d["idx"] < hi]
            print(f"resuming: {len(out['designs'])} designs already done",
                  flush=True)
        else:
            print("existing output used a different loss tangent; "
                  "starting fresh", flush=True)

    # Adopt empty-cell references computed by another run. When many workers
    # share one scalar combo (the batch40 set uses the dataset-mean scalars for
    # all 40 designs) each would otherwise recompute an identical reference,
    # which is duplicated FDTD time for nothing. Mixing loss tangents would
    # silently corrupt the ratios, so refuse it.
    if args["ref_from"]:
        try:
            donor = json.load(open(args["ref_from"]))
        except (OSError, ValueError) as exc:
            sys.exit(f"--ref-from {args['ref_from']}: {exc}")
        dt = donor.get("harness", {}).get("loss_tangent")
        if dt is not None and abs(dt - args["loss"]) > 1e-12:
            sys.exit(f"--ref-from was produced with loss_tangent={dt}, this "
                     f"run uses {args['loss']}")
        for k, v in donor.get("refs", {}).items():
            out["refs"].setdefault(k, v)
        print(f"adopted {len(donor.get('refs', {}))} empty-cell reference(s) "
              f"from {args['ref_from']}; {len(out['refs'])} now available",
              flush=True)
    done = {d["idx"] for d in out["designs"]}

    n_bad = 0
    for k in range(lo, hi):
        if k in done:
            continue
        pat = patterns[k].astype(np.uint8)
        l, h, r = (float(v) for v in (sc[k] if per_item else sc))
        key = f"{round(l, 4)}_{round(h, 4)}_{round(r, 4)}"
        if key not in out["refs"]:
            fe, ae, me = run_case(np.zeros((64, 64), dtype=np.uint8),
                                  l, h, r, args)
            out["refs"][key] = {"flux": fe.tolist(),
                                "abs_alpha": np.abs(ae).tolist(), "meta": me}
            write_json(out, out_path)
            print(f"[empty-{key}] {me}", flush=True)
        ref = out["refs"][key]
        flux_e = np.array(ref["flux"])
        a_e = np.array(ref["abs_alpha"])

        flux_s, a_s, meta = run_case(pat, l, h, r, args)
        flux_mag = np.sqrt(np.abs(flux_s / np.where(
            flux_e > 0, flux_e, np.inf)))
        # Complex transmission ratio, not just its magnitude: the model target
        # is a complex spectrum and the Meep->CST convention needs the phase.
        ratio = a_s / np.where(np.abs(a_e) > 0, a_e, np.nan)
        mode_ratio = np.abs(ratio)
        peak = float(max(flux_mag.max(initial=0.0),
                         mode_ratio.max(initial=0.0)))
        ok = bool(np.isfinite(peak) and peak <= PASSIVITY_TOL)
        n_bad += (not ok)

        entry = {"idx": k, "occ": int(pat.sum()), "l": l, "h": h, "r": r,
                 "flux_mag": flux_mag.tolist(),
                 "flux_re": (flux_s / np.where(flux_e > 0, flux_e, np.inf)
                             ).tolist(),
                 "flux_im": np.zeros_like(flux_s).tolist(),
                 "modes": {"def": {"mag": mode_ratio.tolist(),
                                   "re": np.nan_to_num(
                                       ratio.real).tolist(),
                                   "im": np.nan_to_num(
                                       ratio.imag).tolist()}},
                 "peak_ratio": peak, "passivity_ok": ok,
                 "hit_cap": meta["hit_cap"], "runtime_s": meta["runtime_s"],
                 "ringdown": meta["ringdown"], "blocks": meta["blocks"]}
        if names is not None:
            entry["name"] = names[k]
        if src_idx is not None:
            entry["source_idx"] = int(src_idx[k])
        out["designs"].append(entry)
        write_json(out, out_path)
        print(f"[{k}{'/' + names[k] if names else ''}] occ={entry['occ']} "
              f"peak={peak:.4f} {'OK' if ok else 'PASSIVITY-FAIL'} "
              f"ring={entry['ringdown']}s cap={meta['hit_cap']} "
              f"t={meta['runtime_s']}s", flush=True)

    n = len(out["designs"])
    print(f"MEEP SET DONE designs={n} passivity_failures={n_bad}", flush=True)
    if n_bad:
        print(f"WARNING: {n_bad}/{n} designs violate |T| <= {PASSIVITY_TOL}; "
              "these spectra are NOT usable", flush=True)


if __name__ == "__main__":
    main()