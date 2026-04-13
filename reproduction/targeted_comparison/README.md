# Targeted Comparison — Figs. 3–4, S1–S5

Run directories and script names use legacy `adaptive` naming corresponding to
the targeted comparison layer in the paper.

**Canonical run root**: `extras/data/runs/adaptive`

(Timestamped original: `adaptive_gentle_all_kipan_brca_pancan_20260408_193020`)

Per-dataset subdirectories:

```text
extras/data/runs/adaptive/kipan
extras/data/runs/adaptive/brca
extras/data/runs/adaptive/pancan
```

## Figures

Main figures:

- Fig. 3: KIPAN targeted-comparison heatmaps
- Fig. 4: targeted-comparison quantitative summary

Supplementary figures:

- Fig. S1: BRCA nonlinear heatmaps
- Fig. S2: BRCA linear heatmaps
- Fig. S3: PANCAN heatmaps
- Fig. S4: KIPAN gated-vs-random selected-gene signal
- Fig. S5: PANCAN gated-vs-random selected-gene signal

## Key outputs

Per dataset subdirectory:

- `selected_showcase_configs.csv`, `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`, `selected_best_heatmap_configs.csv`
- `heatmap_models_summary.csv`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_summary.csv`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_aggregate_significance.csv`
- `*_recurrent_gene_supplement_table.csv`

Probe subdirectories (selected-gene signal and recurrence):

- `extras/data/runs/adaptive/kipan/linear_gated_probe_kipan`
- `extras/data/runs/adaptive/pancan/linear_gated_probe_pancan`

## Regeneration

### Heatmaps and selected summaries

```bash
cd extras

python analyses/render_adaptive_manuscript_figures.py \
  --dataset kipan \
  --results-dir data/runs/adaptive/kipan \
  --outdir data/processed/kipan_processed

python analyses/render_adaptive_manuscript_figures.py \
  --dataset brca \
  --results-dir data/runs/adaptive/brca \
  --outdir data/processed/brca_processed \
  --knn-k 5

python analyses/render_adaptive_manuscript_figures.py \
  --dataset pancan \
  --results-dir data/runs/adaptive/pancan \
  --outdir data/processed/pancan_processed \
  --knn-k 5
```

### KIPAN targeted comparison boxplots

```bash
cd extras
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive/kipan \
  --validation-dir data/runs/validation \
  --out-dir figures/targeted_comparison
```

### Selected-gene signal and recurrence

```bash
cd extras

python analyses/quick_linear_gated_probe.py \
  --dataset kipan \
  --results-dir data/runs/adaptive/kipan \
  --outdir data/runs/adaptive/kipan/linear_gated_probe_kipan

python analyses/quick_linear_gated_probe.py \
  --dataset pancan \
  --results-dir data/runs/adaptive/pancan \
  --outdir data/runs/adaptive/pancan/linear_gated_probe_pancan

python analyses/plot_pancan_linear_probe_gene_recurrence.py \
  --dataset pancan \
  --probe-dir data/runs/adaptive/pancan/linear_gated_probe_pancan \
  --outdir figures/targeted_comparison
```

## Rerun

```bash
cd extras
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

## Scripts

- `extras/analyses/ch3_kipan_adaptive_v2.py`
- `extras/analyses/ch3_brca_adaptive_v2.py`
- `extras/analyses/ch3_pancan_adaptive_v2.py`
- `extras/analyses/render_adaptive_manuscript_figures.py`
- `extras/analyses/plot_kipan_boxplots.py`
- `extras/analyses/quick_linear_gated_probe.py`
- `extras/analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `extras/analyses/plot_pancan_gated_univariate_bias.py`
- `extras/analyses/plot_kipan_gated_univariate_dotplot.py`
- `extras/analyses/plot_kipan_feature_set_cv_transport.py`
- `extras/analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh`
