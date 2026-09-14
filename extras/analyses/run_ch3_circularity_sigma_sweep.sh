#!/bin/bash
# Tests whether retuning LSPIN's gate_sigma (its known lever, per thread 2)
# rescues the negative-control failure found in the base circularity check
# (ch3_smoothing_circularity_check.py): does the real graph beat the null
# graph on reproducibility/histology-ARI at a better-tuned gate_sigma?
set -e
cd "$(dirname "$0")"
PY=/banach2/wes/.conda/envs/musevo/bin/python
RUN=../data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
DEVICE=${1:-cuda:3}

for mult in 2.0 4.0 8.0; do
  outdir="$RUN/ch3_smoothing_circularity_lspin_sigma${mult}"
  if [ -f "$outdir/circularity_check_summary.csv" ]; then
    echo "=== sigma x${mult} (already done, skipping) ==="; continue
  fi
  echo "=== sigma x${mult} ==="
  $PY ch3_smoothing_circularity_check.py \
    --device "$DEVICE" --families LSPIN --n-reps 6 \
    --gate-sigma-multiplier "$mult" \
    --outdir "$outdir"
done
echo "ALL_SIGMA_RESCUE_RUNS_DONE"
