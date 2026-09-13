#!/bin/bash
# Higher-rep confirmation of the most promising retuned cells from
# run_thread5_retuning_sweep.sh (POST_DISSERTATION_PLAN.md thread 5):
# LSPIN gate_sigma in {4,8}, Concrete temperature=4, all at 15 reps instead
# of 5, to check whether the win rates found (3/5, 3/5, 4/5) solidify with
# more power or regress toward chance. Same base seed as the 5-rep sweep,
# so reps 0-4 reproduce deterministically and reps 5-14 are new information.
set -e
cd "$(dirname "$0")"
PY=/banach2/wes/.conda/envs/musevo/bin/python
RUN=../data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
DEVICE=${1:-cuda:7}
COMMON="--dataset kipan --candidate-gene-pool-size 300 --true-features-per-subgroup 15 --effect-size 2.5 --subgroup-assignment kmeans --n-reps 15"

run_one() {
  tag=$1; shift
  outdir="$RUN/pd_synthetic_ground_truth_confirm_${tag}"
  if [ -f "$outdir/synthetic_recovery_overall.csv" ]; then
    echo "=== $tag (already done, skipping) ==="; return
  fi
  echo "=== $tag ==="
  $PY pd_synthetic_ground_truth_recovery.py $COMMON --device "$DEVICE" "$@" --outdir "$outdir"
}

run_one lspin_sigma4  --families L-LSPIN --gate-sigma-multiplier 4.0
run_one lspin_sigma8  --families L-LSPIN --gate-sigma-multiplier 8.0
run_one concrete_temp4 --families L-Concrete --temperature-multiplier 4.0

echo "ALL_CONFIRMATION_RUNS_DONE"
