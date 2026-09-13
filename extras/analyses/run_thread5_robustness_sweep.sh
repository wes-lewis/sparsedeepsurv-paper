#!/bin/bash
# Robustness sweep for the kmeans-subgroup synthetic ground-truth recovery
# result (POST_DISSERTATION_PLAN.md thread 5): the positive result so far
# is one calibration point (pool_size=300, true_features=15, effect_size=2.5).
# This sweeps effect_size and true_features_per_subgroup to check the
# direction (personalized > global under kmeans subgroups) holds broadly,
# not just at that one point. Runs sequentially on one GPU to be
# considerate of shared cluster usage.
set -e
cd "$(dirname "$0")"
PY=/banach2/wes/.conda/envs/musevo/bin/python
RUN=../data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
DEVICE=${1:-cuda:7}

for effect_size in 1.5 2.5 4.0; do
  for true_k in 10 15 25; do
    tag="es${effect_size}_tk${true_k}"
    echo "=== $tag ==="
    $PY pd_synthetic_ground_truth_recovery.py \
      --dataset kipan --device "$DEVICE" \
      --candidate-gene-pool-size 300 --true-features-per-subgroup "$true_k" --effect-size "$effect_size" \
      --subgroup-assignment kmeans \
      --outdir "$RUN/pd_synthetic_ground_truth_robustness_${tag}"
  done
done
echo "ALL_ROBUSTNESS_RUNS_DONE"
