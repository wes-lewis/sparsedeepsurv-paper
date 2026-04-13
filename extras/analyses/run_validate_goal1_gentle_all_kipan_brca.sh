#!/usr/bin/env bash
# Combined KIPAN + BRCA goal-1 validation under the gentler gated parameterization.
#
# This launcher:
#   1. Runs validate_models.py for both datasets with the gentler LSPIN defaults
#      already encoded in validate_models.py.
#   2. Uses baseline-matched MLP+STG architecture.
#   3. Sweeps sparsity multipliers in a narrower, post-recovery range.
#   4. Renders both:
#        - fig_validation_init_consistency.png
#        - fig_cindex_all_models_pointest.png
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_gentle_all_kipan_brca_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca/selected_showcase_configs.csv"

echo "[run_validate_goal1_gentle_all_kipan_brca] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_gentle_all_kipan_brca] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --lspin-init-bias 0.0 \
    --goal1-lambda-multipliers 1.0 1.25 1.5 2.0 2.5 3.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.5 0.75 1.0 1.25 1.5 2.0 3.0 \
    --stg-hidden-dims 64 32 \
    --linear-gated-lambda-multipliers 0.75 1.0 1.25 1.5 2.0 3.0 \
    --gating-hidden-dim 32 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_gentle_all_kipan_brca] validate_models done at $(date)"

echo "[run_validate_goal1_gentle_all_kipan_brca] Rendering fig_validation_init_consistency.png"
"$PYTHON" "${SCRIPT_DIR}/plot_validation_init_consistency.py" \
    --run-dir "${RESULTS_DIR}"

echo "[run_validate_goal1_gentle_all_kipan_brca] Rendering fig_cindex_all_models_pointest.png"
"$PYTHON" "${SCRIPT_DIR}/plot_cindex_all_models.py" \
    --run-dir "${RESULTS_DIR}" \
    --out-dir "${RESULTS_DIR}"

echo "[run_validate_goal1_gentle_all_kipan_brca] All done at $(date)"
echo "[run_validate_goal1_gentle_all_kipan_brca] Results in ${RESULTS_DIR}"
