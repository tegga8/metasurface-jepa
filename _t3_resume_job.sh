#!/bin/bash
#SBATCH -p small -N 1 -n 4 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/t3_resume_%j.out
# Re-run ONLY the designs lost to the disk-quota kill (jepa_v2_B_true). The
# harness is resumable per design, so a set is patched by re-running the missing
# index files rather than the whole set. Runs the three solves in parallel.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
TAG=jepa_v2_B_true
NPZ="$HOME/runs/phase3/out/$TAG.npz"
OUT="$HOME/runs/t3/$TAG"
REF="$HOME/runs/t1/ref.json"
echo "=== $(date) on $(hostname) ==="
ls -l "$NPZ" "$REF" || exit 1
N=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$NPZ', allow_pickle=True)['patterns']))")
missing=""
for k in $(seq 0 $((N - 1))); do
  [ -f "$OUT/e_0_${k}.json" ] || missing="$missing $k"
done
echo "set $TAG: $N designs, missing indices:$missing"
if [ -z "$missing" ]; then echo "nothing to do"; exit 0; fi
pids=""
for k in $missing; do
  "$MEEP_PY" _meep_novel.py "$NPZ" "$OUT/e_0_${k}.json" \
      --start "$k" --end $((k + 1)) --max-ring 2500 --ref-from "$REF" \
      > "$OUT/logs/e_0_${k}.log" 2>&1 &
  pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=1; done
echo "parallel re-run rc=$rc"
for k in $missing; do
  f="$OUT/e_0_${k}.json"
  if [ -f "$f" ]; then
    echo -n "  idx $k: "
    "$MEEP_PY" -c "
import json,sys
d=json.load(open('$f'))
e=(d.get('designs') or [{}])[0]
print('passivity_ok=', e.get('passivity_ok'), 'peak=', round(float(e.get('peak_ratio') or -1),4), 'cap=', e.get('hit_cap'))"
  else
    echo "  idx $k: STILL MISSING"
  fi
done
echo "files now: $(ls "$OUT"/e_0_*.json 2>/dev/null | wc -l)/$N"
echo "T3_RESUME DONE"
