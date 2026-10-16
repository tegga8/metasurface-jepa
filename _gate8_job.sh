#!/bin/bash
#SBATCH -p sgpu -N 1 --gres=gpu:1 -n 8 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/gate8_%j.out
# GATE 8 (architecture_v5.md §8.3, the validation plan's critical gap):
# real / null / SHUFFLED, stratified by masking regime, gated on the hard
# stratum (ratio 1.0 + all scalars unknown). Run for the baseline and both
# variants so the gate is answered for every checkpoint we quote.
#
# Runs from the ~/imp2 checkout (a full repo copy at the newer commit, so it has
# both imp configs and the shared config), with absolute checkpoint paths.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
W="$HOME/imp2"
cd "$W" || exit 1
mkdir -p "$HOME/runs/gate8"
echo "=== $(date) on $(hostname) cwd=$(pwd) ==="
ls -l scripts/diagnostics/gate8_stratified.py || exit 1

run () {  # $1 tag  $2 config  $3 checkpoint
  local tag="$1" cfg="$2" ck="$3"
  if [ ! -f "$ck" ]; then echo "SKIP $tag (no checkpoint at $ck)"; return; fi
  echo "=== $tag start $(date +%T) ==="
  python scripts/diagnostics/gate8_stratified.py "$cfg" "$ck" \
      --n 512 --eval-seeds 3 4 5 --out "$HOME/runs/gate8/$tag.json"
  echo "  $tag exit=$? end $(date +%T)"
}

run base_v1 configs/unified.yaml "$HOME/metasurface-jepa/checkpoints/unified/final.pt"
run imp1_s0 configs/unified_imp1.yaml "$HOME/imp1_s0/runs_out/final.pt"
run imp2_s0 configs/unified_imp2.yaml "$HOME/imp2/runs_out/final.pt"
echo "GATE8_JOB DONE"
