"""Meep calibration v3b — eigenmode extraction of the complex transmission.

add_mode_monitor(freqs, region) has no TE/TM arg in this build; the mode
coefficients are extracted with get_eigenmode_coefficients(mon, [1]) under the
default, EVEN_Z and ODD_Z parities (whichever tracks the flux ratio is the
right one). Struct runs: fixed 1100 after sources (v2 showed ringdown
truncation for high-Q items).

Run in WSL:  ~/meep_env/bin/python /mnt/d/projects/metamaterials_rewrite/_meep_calib.py
Outputs:     /tmp/meep_calib_out.json (after each item)
"""

import json
import sys
import time

import meep as mp
import numpy as np
from scipy import io

SPLIT = "/mnt/d/projects/metamaterials_rewrite/data/metadit/split_data/val_set.mat"
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
        except Exception as e:  # noqa: BLE001
            print(f"    [eig {tag} failed: {type(e).__name__}]", flush=True)
            out[tag] = None
    return out


def run_case(pat, l, h, n, label, fixed_after=None):
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
    if fixed_after is not None:
        sim.run(until_after_sources=fixed_after)
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, mon_z), 1e-3))
    runtime = time.time() - t0

    flux = np.array(mp.get_fluxes(trans))
    modes = eigenmodes(sim, mon)
    print(f"[{label}] blocks={len(geom)} runtime={runtime:.0f}s", flush=True)
    return flux, modes


def norm_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.dot(a, b)
                 / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    d = io.loadmat(SPLIT)
    pat, par, real, imag = d["pattern"], d["parameter"], d["real"], d["imag"]
    n_items = pat.shape[-1]

    occ = pat.reshape(-1, n_items).sum(axis=0)
    cand = np.where((occ > 1024) & (occ < 2253))[0]
    order = np.argsort(par[cand, 1] + 0.3 * par[cand, 2])
    fracs = [float(x) for x in sys.argv[1:]] or [0.1, 0.5, 0.9]
    picks = [int(cand[order[int(f * (len(order) - 1))]]) for f in fracs]
    out_path = "/tmp/meep_calib_extra.json" if len(sys.argv) > 1 else "/tmp/meep_calib_out.json"
    print("picked val indices:", picks, "| out:", out_path, flush=True)

    results = []
    for k, idx in enumerate(picks):
        l, h, r = (float(par[idx, 0]), float(par[idx, 1]), float(par[idx, 2]))
        m = np.zeros((64, 64), dtype=np.uint8)
        m[...] = (pat[:, :, idx] == 1)
        print(f"item {k}: idx={idx} l={l:.3f} h={h:.3f} r={r:.3f} "
              f"occ={int(m.sum())}", flush=True)

        flux_s, modes_s = run_case(m, l, h, r, f"item{k}-struct", STRUCT_AFTER)
        flux_e, modes_e = run_case(np.zeros((64, 64), dtype=np.uint8),
                                   l, h, r, f"item{k}-empty")

        flux_mag = np.sqrt(np.abs(flux_s / np.maximum(flux_e, 1e-30)))
        data_cplx = real[idx] + 1j * imag[idx]
        data_mag = np.abs(data_cplx)
        c_mag_flux = norm_corr(flux_mag, data_mag)
        print(f"  flux-mag vs data corr={c_mag_flux:.4f}", flush=True)

        entry = dict(idx=int(idx), l=l, h=h, r=r, corr_fluxmag=c_mag_flux,
                     flux_mag=flux_mag.tolist(), data_mag=data_mag.tolist(),
                     d_re=real[idx].tolist(), d_im=imag[idx].tolist(),
                     modes={})
        for tag in ("def", "ez", "oz"):
            a_s, a_e = modes_s.get(tag), modes_e.get(tag)
            if a_s is None or a_e is None:
                continue
            t = a_s / np.where(np.abs(a_e) < 1e-30, 1e-30, a_e)
            c_self = norm_corr(np.abs(t), flux_mag)
            c_dat = norm_corr(np.abs(t), data_mag)
            print(f"    mode[{tag}]: |t| vs flux-mag={c_self:.4f} "
                  f"vs data-mag={c_dat:.4f}", flush=True)
            entry["modes"][tag] = dict(
                mag=np.abs(t).tolist(), re=t.real.tolist(), im=t.imag.tolist(),
                corr_flux=float(c_self), corr_data=float(c_dat))
        results.append(entry)
        with open(out_path, "w") as f:
            json.dump(results, f)

    print("MEEP CALIB v3b DONE", flush=True)


if __name__ == "__main__":
    main()
