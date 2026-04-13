# Targeted Comparison

This folder covers the targeted smoothing comparisons: matched heatmaps,
selected comparison summaries, and the corresponding supplementary signal and
recurrence outputs.

The underlying run directories and script filenames still use some older
`adaptive` naming, but in the paper text these runs correspond to the targeted
comparison layer.

## Figure correspondence

Main paper figures:

- Fig. 3: KIPAN targeted-comparison heatmaps
- Fig. 4: targeted-comparison quantitative summary

Supplementary figures:

- Fig. S1: BRCA nonlinear heatmaps
- Fig. S2: BRCA linear heatmaps
- Fig. S3: PANCAN heatmaps

Nested selected-gene signal figures:

- Fig. S4: KIPAN gated-vs-random selected-gene signal
- Fig. S5: PANCAN gated-vs-random selected-gene signal

## Canonical preserved run root

```text
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020
```

Per-dataset preserved subdirectories:

```text
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

## Primary preserved outputs

In each relevant dataset subdirectory, focus on:

- `selected_showcase_configs.csv`
- `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`
- `selected_best_heatmap_configs.csv`
- `heatmap_models_summary.csv`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`

For the selected-gene signal and recurrence supplements, focus on:

- `kipan/linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep`
- `pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

Representative related CSV and supplementary-data outputs include:

- `selected_showcase_configs.csv`
- `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`
- `heatmap_models_summary.csv`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_summary.csv`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_aggregate_significance.csv`
- `*_recurrent_gene_supplement_table.csv`

## Primary preserved directories

Inspection should begin in:

```text
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca
extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

## Regeneration entry points

### KIPAN heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset kipan \
  --results-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir extras/data/processed/kipan_processed
```

### BRCA heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset brca \
  --results-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca \
  --outdir extras/data/processed/brca_processed \
  --knn-k 5
```

### PANCAN heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset pancan \
  --results-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir extras/data/processed/pancan_processed \
  --knn-k 5
```

### KIPAN targeted comparison boxplots

```bash
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --validation-dir extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir extras/figures/targeted_comparison
```

### KIPAN selected-gene signal outputs

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset kipan \
  --results-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir extras/data/runs/targeted_comparison_kipan_probe
```

### PANCAN selected-gene signal outputs

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset pancan \
  --results-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir extras/data/runs/targeted_comparison_pancan_probe
```

### PANCAN recurrent-gene summaries

```bash
python analyses/plot_pancan_linear_probe_gene_recurrence.py \
  --dataset pancan \
  --probe-dir extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep \
  --outdir extras/figures/targeted_comparison
```

## Rerun entry point

Primary launcher:

```bash
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

This creates a new run root under `extras/data/runs/` with fresh `kipan/`, `brca/`,
and `pancan/` subdirectories.

## Primary scripts

- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/ch3_pancan_adaptive_v2.py`
- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/plot_kipan_boxplots.py`
- `analyses/quick_linear_gated_probe.py`
- `analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `analyses/plot_pancan_gated_univariate_bias.py`
- `analyses/plot_kipan_gated_univariate_dotplot.py`
- `analyses/plot_kipan_feature_set_cv_transport.py`
- `analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh`
