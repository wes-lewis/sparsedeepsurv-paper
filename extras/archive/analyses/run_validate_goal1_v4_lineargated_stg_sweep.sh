#!/usr/bin/env bash
# Goal-1 validation v4: Lambda sweep for L-LSPIN, L-Concrete, and MLP+STG.
#
# Context: in v3, L-LSPIN and L-Concrete used a single lambda (showcase smooth lambda ×1.0)
# with no sweep, and MLP+STG was Khard-matched against LSPIN which selected x0.25 (C=0.563)
# over x0.5 (C=0.591). This run:
#   - Sweeps L-LSPIN and L-Concrete across a wide lambda range per dataset
#   - Sweeps MLP+STG across a wider range including smaller lambdas
#   - Selects all three by CI (not Khard-matching)
#   - Limits LSPIN/Concrete to reference config only (λ×1.0) to save GPU budget
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_v4_lineargated_stg_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_v4] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_v4] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --goal1-lambda-multipliers 1.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.05 0.1 0.25 0.5 1.0 2.0 5.0 10.0 \
    --linear-gated-lambda-multipliers 0.05 0.1 0.25 0.5 1.0 2.0 5.0 10.0 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_v4] validate_models done at $(date)"
echo "[run_validate_goal1_v4] Results in ${RESULTS_DIR}"
