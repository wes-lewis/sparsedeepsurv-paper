#!/bin/bash
# Per-family hyperparameter retuning for the synthetic ground-truth recovery
# task (POST_DISSERTATION_PLAN.md thread 5), at the validated calibration
# point (kmeans subgroups, pool_size=300, true_features_per_subgroup=15,
# effect_size=2.5). The L-LSPIN/L-Concrete configs were tuned for predictive
# C-index on the real KIPAN task, not for recovery quality here.
#
# LSPIN's lever is gate_sigma (thread 2's rescue investigation); Concrete's
# is temperature (gate_sigma is a confirmed no-op for Concrete -- see the
# 2026-09-13 commit). Swept separately per family since they don't share a
# lever. n_reps=5 throughout, matching the paired-comparison design already
# established as necessary (a single seed is not enough to trust a result
# here).
set -e
cd "$(dirname "$0")"
PY=/banach2/wes/.conda/envs/musevo/bin/python
RUN=../data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
DEVICE=${1:-cuda:7}
COMMON="--dataset kipan --candidate-gene-pool-size 300 --true-features-per-subgroup 15 --effect-size 2.5 --subgroup-assignment kmeans --n-reps 5"

for sigma_mult in 1.0 2.0 4.0 8.0; do
  tag="lspin_sigma${sigma_mult}"
  outdir="$RUN/pd_synthetic_ground_truth_retune_${tag}"
  if [ -f "$outdir/synthetic_recovery_overall.csv" ]; then
    echo "=== $tag (already done, skipping) ==="; continue
  fi
  echo "=== $tag ==="
  $PY pd_synthetic_ground_truth_recovery.py $COMMON --device "$DEVICE" \
    --families L-LSPIN --gate-sigma-multiplier "$sigma_mult" \
    --outdir "$outdir"
done

for temp_mult in 0.25 0.5 1.0 2.0 4.0; do
  tag="concrete_temp${temp_mult}"
  outdir="$RUN/pd_synthetic_ground_truth_retune_${tag}"
  if [ -f "$outdir/synthetic_recovery_overall.csv" ]; then
    echo "=== $tag (already done, skipping) ==="; continue
  fi
  echo "=== $tag ==="
  $PY pd_synthetic_ground_truth_recovery.py $COMMON --device "$DEVICE" \
    --families L-Concrete --temperature-multiplier "$temp_mult" \
    --outdir "$outdir"
done

echo "ALL_RETUNING_RUNS_DONE"
