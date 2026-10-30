#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/t3win_%j.out
# Task-3 window fix, pass 2: re-run the four affected perturbed-scalar sets with
# ONE explicit post-source window (--ring-fixed), so every design and the
# empty-cell reference it is divided by integrate over the identical window.
#
# Pass 1 needed no re-run: the adaptive runs already recorded when each decay
# test stopped. Measured from those (locally): design ringdowns reach 470.0
# (jepa_v2_spec_p20) while every set's reference stops at 105.2 -- a 4.5x window
# mismatch between the two halves of the ratio, which is the artefact.
# Window = max(ringdown) * 1.05, and it is shared by all four sets so the
# m20/p20 rows are also comparable with each other.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
T=493.5
W="$HOME/runs/t3win"
mkdir -p "$W"

run_set () {
  local TAG="$1"
  local NPZ="$HOME/runs/phase3/out/$TAG.npz"
  local OUT="$W/$TAG"
  mkdir -p "$OUT/logs"
  if [ ! -f "$NPZ" ]; then echo "MISSING npz $NPZ"; return 1; fi
  local N
  N=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$NPZ', allow_pickle=True)['patterns']))")
  local REF=""
  for k in $(seq 0 $((N - 1))); do
    [ -f "$OUT/e_0_${k}.json" ] && continue
    local extra=""
    [ -n "$REF" ] && extra="--ref-from $REF"
    "$MEEP_PY" _meep_novel.py "$NPZ" "$OUT/e_0_${k}.json" \
        --start "$k" --end $((k + 1)) --max-ring 2500 --ring-fixed "$T" \
        $extra > "$OUT/logs/e_0_${k}.log" 2>&1
    if [ -z "$REF" ] && [ -f "$OUT/e_0_0.json" ]; then
      REF="$OUT/e_0_0.json"      # its refs were computed under the same window
    fi
  done
  echo "$TAG done: $(ls "$OUT"/e_0_*.json 2>/dev/null | wc -l)/$N files"
}

for TAG in jepa_v1_spec_m20 jepa_v1_spec_p20 jepa_v2_spec_m20 jepa_v2_spec_p20; do
  run_set "$TAG" &
done
wait

echo "=== window + passivity summary ==="
"$MEEP_PY" - <<'PY'
import glob, json, os
W = os.path.expanduser("~/runs/t3win")
for d in sorted(os.listdir(W)):
    p = os.path.join(W, d)
    if not os.path.isdir(p):
        continue
    peaks, wins, ok, capped, refw = [], set(), 0, 0, set()
    for f in sorted(glob.glob(os.path.join(p, "e_0_*.json"))):
        j = json.load(open(f))
        wins.add(j.get("harness", {}).get("window"))
        for r in j.get("refs", {}).values():
            refw.add((r.get("meta") or {}).get("window"))
        for e in j.get("designs", []):
            peaks.append(float(e.get("peak_ratio") or -1))
            ok += bool(e.get("passivity_ok"))
            capped += bool(e.get("hit_cap"))
    if peaks:
        print(f"{d}: n={len(peaks)} ok={ok} peaks {min(peaks):.3f}-{max(peaks):.3f} "
              f"capped={capped} windows={sorted(wins)} ref_windows={sorted(refw)}")
PY
echo "T3WIN DONE"
