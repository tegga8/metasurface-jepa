"""Meep spectra for a design set (eigenmode complex transmission).

Usage (WSL): ~/meep_env/bin/python _meep_novel.py <designs.npz> <out.json>

The npz holds `patterns` (N x 64 x 64) and `scalars` (shared (3,) or per-item
(N, 3)). One empty-cell reference is cached per distinct scalar combo. For each
design: fixed-duration FDTD (1100 after sources), flux monitor + mode monitor;
saves flux magnitude and mode amplitudes under default / EVEN_Z / ODD_Z
parities. Writes the output JSON after EVERY design (resumable view).
"""

import json
import sys
import time

import meep as mp
import numpy as np

RES = 32
PML = 0.5
PAD = 0.8
FCEN, DF = 0.15, 0.1
FREQS = np.linspace(0.1, 0.2, 301)
STRUCT_AFTER = 1100


def eigenmodes(sim, mon):
    out = {}
    for tag, kw in (("def", {}), ("ez", {"eig_parity": mp.EVEN_Z}),
                    ("oz", {"eig_parity": mp.ODD_Z})):
        try:
            out[tag] = sim.get_eigenmode_coefficients(mon, [1], **kw).alpha[0, :, 0]
        except Exception:  # noqa: BLE001
            out[tag] = None
    return out


def run_case(pat, l, h, n, label):
    sx = sy = l
    sz = h + 2 * PAD
    src_z = -sz / 2 + PML + 0.1
    mon_z = sz / 2 - PML - 0.1

    geom = []
    mat = mp.Medium(epsilon=n ** 2)
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
        resolution=RES,
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

    t0 = time.time()
    sim.run(until_after_sources=STRUCT_AFTER)
    runtime = time.time() - t0

    flux = np.array(mp.get_fluxes(trans))
    modes = eigenmodes(sim, mon)
    print(f"[{label}] blocks={len(geom)} runtime={runtime:.0f}s", flush=True)
    return flux, modes


def main():
    npz_path, out_path = sys.argv[1], sys.argv[2]
    z = np.load(npz_path, allow_pickle=True)
    patterns = z["patterns"]
    sc = z["scalars"]
    per_item = sc.ndim > 1
    src_idx = z["source_idx"].tolist() if "source_idx" in z else None
    print(f"loaded {len(patterns)} designs; scalars "
          f"{'per-item' if per_item else 'shared'}", flush=True)

    empty_cache = {}
    out = {"npz": npz_path, "per_item_scalars": bool(per_item), "designs": []}
    for k, pat in enumerate(patterns):
        l, h, r = (float(v) for v in (sc[k] if per_item else sc))
        key = (round(l, 4), round(h, 4), round(r, 4))
        if key not in empty_cache:
            flux_e, modes_e = run_case(np.zeros((64, 64), dtype=np.uint8),
                                       l, h, r, f"empty-{key}")
            empty_cache[key] = (flux_e, modes_e)
        flux_e, modes_e = empty_cache[key]
        flux_s, modes_s = run_case(pat, l, h, r, f"design{k}")
        flux_mag = np.sqrt(np.abs(flux_s / np.maximum(flux_e, 1e-30)))
        entry = {"idx": k, "occ": int(pat.sum()), "l": l, "h": h, "r": r,
                 "flux_mag": flux_mag.tolist(), "modes": {}}
        if src_idx is not None:
            entry["source_idx"] = int(src_idx[k])
        for tag in ("def", "ez", "oz"):
            a_s, a_e = modes_s.get(tag), modes_e.get(tag)
            if a_s is None or a_e is None:
                continue
            t = a_s / np.where(np.abs(a_e) < 1e-30, 1e-30, a_e)
            entry["modes"][tag] = dict(mag=np.abs(t).tolist(),
                                       re=t.real.tolist(), im=t.imag.tolist())
        out["designs"].append(entry)
        with open(out_path, "w") as f:
            json.dump(out, f)

    print("MEEP SET DONE", flush=True)


if __name__ == "__main__":
    main()
