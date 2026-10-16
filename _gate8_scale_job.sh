#!/bin/bash
#SBATCH -p sgpu -N 1 --gres=gpu:1 -n 8 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/gate8scale_%j.out
# Gate 8 on the Phase-6 scaling arms (S1 small / S2 slim / L1 wide).
#
# The ledger records gate 8 as answered for base v1, imp1 and imp2 but NOT for the
# scaling arms, which is the remaining hole in the validation plan's critical
# gate. Same instrument, same strata, same n and eval seeds as the base run, so
# the numbers are directly comparable to base v1's hard-stratum 0.9974.
#
# Runs from the ~/imp2 checkout (a full repo copy with configs/scaling + the data
# symlink); checkpoints are uploaded to ~/scale_ckpts/.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
W="$HOME/imp2"
cd "$W" || exit 1
mkdir -p "$HOME/runs/gate8"
echo "=== $(date) on $(hostname) cwd=$(pwd) ==="
ls -l "$HOME"/scale_ckpts/*.pt || exit 1

run () {  # $1 tag  $2 config  $3 checkpoint
  local tag="$1" cfg="$2" ck="$3"
  if [ ! -f "$ck" ]; then echo "SKIP $tag (no checkpoint at $ck)"; return; fi
  echo "=== $tag start $(date +%T) ==="
  python scripts/diagnostics/gate8_stratified.py "$cfg" "$ck" \
      --n 512 --eval-seeds 3 4 5 --out "$HOME/runs/gate8/$tag.json"
  echo "  $tag exit=$? end $(date +%T)"
}

run scale_s1 configs/scaling/unified_s1_small.yaml "$HOME/scale_ckpts/s1.pt"
run scale_s2 configs/scaling/unified_s2_slim.yaml "$HOME/scale_ckpts/s2.pt"
run scale_l1 configs/scaling/unified_l1_wide.yaml "$HOME/scale_ckpts/l1.pt"
echo "GATE8_SCALE DONE"
