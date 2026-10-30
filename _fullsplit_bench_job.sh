#!/bin/bash
#SBATCH -p small -N 1 -n 16 -t 24:00:00
#SBATCH --output=/home/mm24r002/logs/fs_%j.out
# Full-split (17,489 test items) scenario-A benchmark for one variant's three
# seeds. `--samples 0` means "use the whole split".
#
# NOTE: each seed runs from ITS OWN checkout (the shared ~/metasurface-jepa
# checkout predates the imp configs and does not contain them -- that is what
# killed jobs 642370/642371). The per-seed checkouts carry the config, the data
# symlink, and their own runs_out/.
# Usage: sbatch --job-name=fs_imp1 _fullsplit_bench_job.sh imp1
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
V="${1:?usage: sbatch _fullsplit_bench_job.sh <imp1|imp2>}"
echo "=== $(date) on $(hostname) nproc=$(nproc) variant=$V ==="
for s in 0 1 2; do
  case "$V" in
    imp1) W="$HOME/imp1_s$s"; CFG="configs/unified_imp1.yaml" ;;
    imp2) W="$HOME/imp2_s$s"; CFG="configs/unified_imp2.yaml" ;;
    *) echo "unknown variant $V"; exit 1 ;;
  esac
  if [ ! -f "$W/$CFG" ]; then echo "MISSING $W/$CFG - skipping seed $s"; continue; fi
  if [ ! -f "$W/runs_out/final.pt" ]; then echo "MISSING ckpt in $W - skipping"; continue; fi
  cd "$W" || continue
  echo "=== $V seed $s full-split benchmark start $(date +%T) cwd=$(pwd) ==="
  python scripts/benchmark/benchmark_metadit.py --config "$CFG" \
      --checkpoint runs_out/final.pt --split test --scenario A --device cpu \
      --samples 0 --batch-size 8 --out runs_out/benchmark_full.json
  echo "  exit=$? end $(date +%T)"
done
echo "FULLSPLIT_${V}_DONE"
