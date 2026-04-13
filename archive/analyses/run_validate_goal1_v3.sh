#!/usr/bin/env bash
# Goal-1 validation run v3:
#   - Adds RSF and Linear Cox to CI plot
#   - Adds L-LSPIN smooth, L-Concrete smooth to fixed-init eval
#   - STG uses much smaller lambda (--stg-lambda-base 0.001 with fine multipliers)
#     to prevent Khard collapse
#   - Two-figure output: all-models C-index CI + gated-only Khard/Affinity
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_models_goal1_v3_kipan_brca_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_v3] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_v3] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.1 0.25 0.5 1.0 2.0 5.0 \
    --goal1-lambda-multipliers 0.5 0.75 1.0 1.5 2.0 3.0 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_v3] validate_models done at $(date)"

echo "[run_validate_goal1_v3] Running plot_validation_init_consistency.py"
"$PYTHON" "${SCRIPT_DIR}/plot_validation_init_consistency.py" \
    --run-dir "${RESULTS_DIR}"

echo "[run_validate_goal1_v3] Running plot_validation_supp_boxplots.py"
"$PYTHON" "${SCRIPT_DIR}/plot_validation_supp_boxplots.py" \
    --results-dir "${RESULTS_DIR}"

echo "[run_validate_goal1_v3] All done at $(date)"
echo "[run_validate_goal1_v3] Figures written to ${RESULTS_DIR}"
