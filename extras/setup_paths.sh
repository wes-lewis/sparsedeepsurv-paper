#!/usr/bin/env bash
# Create short-name symlinks for canonical run and processed directories.
# Run once from the repository root after cloning:
#   bash extras/setup_paths.sh
set -euo pipefail

EXTRAS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$EXTRAS/data/runs"
PROCESSED="$EXTRAS/data/processed"

# Canonical run symlinks
ln -sfn validate_goal1_gentle_all_kipan_brca_20260408_175906 "$RUNS/validation"
ln -sfn adaptive_gentle_all_kipan_brca_pancan_20260408_193020 "$RUNS/adaptive"

# Broad sweep symlinks (v2 = primary, gentle = supplementary)
mkdir -p "$RUNS/broad"
ln -sfn ../ch3_kipan_broad_v2_selfcontained_ste_20260407_012336 "$RUNS/broad/kipan_v2"
ln -sfn ../ch3_brca_broad_v2_recovered_20260326                "$RUNS/broad/brca_v2"
ln -sfn ../ch3_kipan_broad_gentle_20260409_161803               "$RUNS/broad/kipan_gentle"
ln -sfn ../ch3_brca_broad_gentle_20260409_161803                "$RUNS/broad/brca_gentle"

# Processed dataset symlinks
ln -sfn kipan_20260209_213604               "$PROCESSED/kipan_processed"
ln -sfn tcga_brca20260214_001423            "$PROCESSED/brca_processed"
ln -sfn tcga_pancan_xena_20260330_top5000   "$PROCESSED/pancan_processed"

# Probe subdirectory symlinks inside adaptive run
ADAPTIVE="$RUNS/adaptive_gentle_all_kipan_brca_pancan_20260408_193020"
ln -sfn linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep \
        "$ADAPTIVE/kipan/linear_gated_probe_kipan"
ln -sfn linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep \
        "$ADAPTIVE/pancan/linear_gated_probe_pancan"

echo "Path aliases created."
