#!/bin/bash
#SBATCH -p small -N 1 -n 4 -t 01:00:00
#SBATCH --output=/home/mm24r002/logs/imp2score2_%j.out
# Phase 2 for imp2 with the scorer that matches this layout, plus a passivity
# census per set (phase 3's spec_p20 row vanished from the comparator, which only
# keeps passivity-clean designs -- so the census is what explains it).
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/imp2" || exit 1
W="$HOME/runs/p23imp2"

echo "=== PHASE 2 (screened A'/B', imp2) ==="
python _score_p2f.py "$W/meep" "$HOME/runs/phase3f/random_spectra.npz" \
    both jepa_imp2_A jepa_imp2_B
echo "p2 exit=$?"

echo
echo "=== passivity census (why a row can disappear) ==="
python - <<'PY'
import glob, json, os
W = os.path.expanduser("~/runs/p23imp2/meep")
for d in sorted(os.listdir(W)):
    p = os.path.join(W, d)
    if not os.path.isdir(p):
        continue
    n = ok = capped = low = 0
    peaks = []
    for f in sorted(glob.glob(os.path.join(p, "e_0_*.json"))):
        j = json.load(open(f))
        for e in j.get("designs", []):
            n += 1
            peaks.append(float(e.get("peak_ratio") or -1))
            ok += bool(e.get("passivity_ok"))
            capped += bool(e.get("hit_cap"))
            low += int(e.get("occ", 0) < 200)   # near-empty decodes
    if n:
        print(f"  {d:22s} n={n:3d} passivity_ok={ok:3d} capped={capped} "
              f"near-empty(<200 px)={low:3d} peaks {min(peaks):.3f}-{max(peaks):.3f}")
PY
echo IMP2SCORE2 DONE
