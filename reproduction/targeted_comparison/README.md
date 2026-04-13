# Targeted Comparison

This folder covers the targeted smoothing comparisons: the matched heatmaps,
selected comparison summaries, and the visible targeted-comparison figures.

The underlying run directories and script filenames still use some older
`adaptive` naming, but in the paper text these runs correspond to the targeted
comparison layer.

## Visible outputs

Main visible targeted-comparison figures:

- `figures/diss_chap3_fig2_2.png`
- `figures/disschap3_fig3_2.png`

Visible supplementary targeted-comparison figures:

- `figures/diss_chap3_suppfig1.png`
- `figures/diss_chap3_suppfig_linear_brca_heatmap.png`
- `figures/diss_chap3_suppfig2_2.png`
- `figures/kipan_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`
- `figures/PANCAN_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`

## Canonical preserved run root

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020
```

Per-dataset preserved subdirectories:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

## The small set of preserved files that matter most

In each relevant dataset subdirectory, focus on:

- `selected_showcase_configs.csv`
- `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`
- `selected_best_heatmap_configs.csv`
- `heatmap_models_summary.csv`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`

For the selected-gene signal/recurrence supplements, focus on:

- `kipan/linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep`
- `pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

## If you only want to inspect the preserved results

Browse these directories directly:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

## If you want to rerender the visible targeted-comparison outputs

### KIPAN heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset kipan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir data/processed/kipan_20260209_213604
```

### BRCA heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset brca \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca \
  --outdir data/processed/tcga_brca20260214_001423 \
  --knn-k 5
```

### PANCAN heatmaps and selected summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset pancan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir data/processed/tcga_pancan_xena_20260330_top5000 \
  --knn-k 5
```

### KIPAN targeted comparison boxplots

```bash
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --validation-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/repro_kipan_boxplots
```

### KIPAN selected-gene signal outputs

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset kipan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan/linear_gated_probe_kipan_repro
```

### PANCAN selected-gene signal outputs

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset pancan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_repro
```

### PANCAN recurrent-gene summaries

```bash
python analyses/plot_pancan_linear_probe_gene_recurrence.py \
  --dataset pancan \
  --probe-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep \
  --outdir figures/repro_pancan_recurrence
```

## If you want to launch a fresh targeted-comparison run with the same workflow defaults

Use the multi-dataset launcher:

```bash
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

This creates a new timestamped run root under `data/runs/` with fresh
`kipan/`, `brca/`, and `pancan/` subdirectories.

## Driver and plotting scripts used here

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
