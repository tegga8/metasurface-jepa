#!/bin/bash
#SBATCH -p small -N 1 -n 16 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/gate8scaleCPU_%j.out
# Gate 8 on the Phase-6 scaling arms (S1 small / S2 slim / L1 wide) -- CPU variant.
#
# The GPU attempts died inside load_checkpoint, not in this script: the scaling
# checkpoints store a CUDA RNG state from their training host, and
# torch.cuda.set_rng_state_all() throws IndexError when the saved state does not
# match the current device count (reproducibility.restore_rng_state guards on
# torch.cuda.is_available() only). On a CPU node that guard skips the CUDA restore
# by design, so the same instrument runs unchanged -- slower, but it is the same
# code path that produced base v1's hard-stratum 0.9974.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
W="$HOME/imp2"
cd "$W" || exit 1
mkdir -p "$HOME/runs/gate8"
echo "=== $(date) on $(hostname) cwd=$(pwd) ==="

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
echo "GATE8_SCALE_CPU DONE"
