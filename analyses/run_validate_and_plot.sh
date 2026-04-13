#!/usr/bin/env bash
# Run validate_models.py (resuming screen phase from existing partial dirs),
# then render both validation figure types.
# Usage: bash analyses/run_validate_and_plot.sh
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_models_all_kipan_brca_selfcontained_ste_stg_20260406_2300"

echo "[run_validate_and_plot] Results dir: ${RESULTS_DIR}"
echo "[run_validate_and_plot] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode all \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

echo "[run_validate_and_plot] validate_models done at $(date)"

echo "[run_validate_and_plot] Running plot_validation_init_consistency.py"
"$PYTHON" "${SCRIPT_DIR}/plot_validation_init_consistency.py" \
    --run-dir "${RESULTS_DIR}"

echo "[run_validate_and_plot] Running plot_validation_supp_boxplots.py"
"$PYTHON" "${SCRIPT_DIR}/plot_validation_supp_boxplots.py" \
    --results-dir "${RESULTS_DIR}"

echo "[run_validate_and_plot] All done at $(date)"
echo "[run_validate_and_plot] Figures written to ${RESULTS_DIR}"
