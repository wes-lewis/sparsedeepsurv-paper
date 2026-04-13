# Reproduction Guide

This guide is the concrete companion to the preserved run archive. It is meant
to answer a practical question:

"If I want to inspect, rerender, or rerun a specific result family, what do I
actually run?"

The commands below assume you are in the repository root:

```bash
cd sparsedeepsurv-paper
```

They also assume your Python environment can import the sibling
`sparsedeepsurv` package as described in the top-level
[`README.md`](README.md).

## What the preserved runs let you do

For the main workflows in this repository, the preserved artifacts are enough
to do all three of the following:

- inspect the exact frozen run outputs kept in the archive
- regenerate the paper-facing figures and summary files from those runs
- start a fresh run with the same workflow defaults and parameterization

That means a user can instantiate another identically parameterized run for the
main validation, adaptive, and broad workflows. The expectation should still be
"approximately similar results" rather than byte-identical outputs across
machines.

## 1. Validation and all-model comparison

Canonical preserved run:

```text
data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

### Inspect the preserved outputs

Look in this directory for:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`
- `screen_summary.csv`
- `affinity_summary.csv`

### Regenerate the preserved validation figures from the frozen run

```bash
python analyses/plot_cindex_all_models.py \
  --run-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/repro_validation

python analyses/plot_validation_init_consistency.py \
  --run-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/repro_validation

python analyses/plot_validation_supp_boxplots.py \
  --results-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

### Launch a fresh validation run with matching workflow defaults

```bash
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

This launcher creates a new timestamped run under `data/runs/` and then renders
the two main validation figures from that fresh run.

### Direct driver entry point

If you want the validation driver without the shell wrapper:

```bash
python analyses/validate_models.py --help
```

## 2. Adaptive gated analyses

Canonical preserved run root:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020
```

Per-dataset preserved subdirectories:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

### Inspect the preserved adaptive outputs

Each dataset subdirectory contains the core frozen outputs used by the paper,
including:

- `selected_showcase_configs.csv`
- `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`
- `selected_best_heatmap_configs.csv`
- `heatmap_models_summary.csv`
- `notebook_style_models_*`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`

### Regenerate KIPAN heatmaps and selected adaptive summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset kipan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir data/processed/kipan_20260209_213604
```

### Regenerate BRCA heatmaps and selected adaptive summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset brca \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca \
  --outdir data/processed/tcga_brca20260214_001423 \
  --knn-k 5
```

### Regenerate PANCAN heatmaps and selected adaptive summaries

```bash
python analyses/render_adaptive_manuscript_figures.py \
  --dataset pancan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir data/processed/tcga_pancan_xena_20260330_top5000 \
  --knn-k 5
```

### Regenerate adaptive comparison boxplots from frozen summaries

For KIPAN/validation-style comparison boxplots:

```bash
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --validation-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/repro_kipan_boxplots
```

### Launch fresh adaptive runs with matching workflow defaults

```bash
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

This creates a new timestamped adaptive run root under `data/runs/` with fresh
`kipan/`, `brca/`, and `pancan/` subdirectories.

### Direct driver entry points

```bash
python analyses/ch3_kipan_adaptive_v2.py --help
python analyses/ch3_brca_adaptive_v2.py --help
python analyses/ch3_pancan_adaptive_v2.py --help
```

## 3. Probe-style appendix analyses derived from adaptive runs

These analyses are built from the adaptive preserved runs rather than being
standalone training workflows.

### KIPAN gated-vs-random signal and dotplot-style analyses

Use the preserved adaptive KIPAN run:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan
```

Generate the linear probe outputs:

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset kipan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan/linear_gated_probe_kipan_repro
```

Generate the KIPAN dotplot-style analysis:

```bash
python analyses/plot_kipan_gated_univariate_dotplot.py \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir figures/repro_kipan_dotplot
```

Generate the KIPAN feature-set transport analysis:

```bash
python analyses/plot_kipan_feature_set_cv_transport.py \
  --dataset kipan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --outdir figures/repro_kipan_transport
```

### PANCAN gated-vs-random signal and recurrent-gene analyses

Use the preserved adaptive PANCAN run:

```text
data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan
```

Generate the linear probe outputs:

```bash
python analyses/quick_linear_gated_probe.py \
  --dataset pancan \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_repro
```

Generate the recurrent-gene summaries:

```bash
python analyses/plot_pancan_linear_probe_gene_recurrence.py \
  --dataset pancan \
  --probe-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep \
  --outdir figures/repro_pancan_recurrence
```

Generate the gated-univariate bias analysis:

```bash
python analyses/plot_pancan_gated_univariate_bias.py \
  --results-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan \
  --outdir figures/repro_pancan_bias
```

## 4. Broad sweeps

Canonical preserved broad runs:

```text
data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_kipan_broad_gentle_20260409_161803
data/runs/ch3_brca_broad_gentle_20260409_161803
```

The `v2` KIPAN and BRCA runs are the strongest preserved broad references. The
BRCA gentle broad run appears less complete at the top level and should be
treated more cautiously.

### Inspect the preserved broad outputs

Look for summary files such as:

- `notebook_style_models_accuracy_summary.csv`
- `notebook_style_models_affinity_summary.csv`
- `notebook_style_models_cluster_summary.csv`
- merged or postprocessed broad summary CSVs
- broad figure PNG/PDF outputs

### Regenerate broad-derived plots from preserved runs

KIPAN transport-size sweep:

```bash
python analyses/plot_feature_set_transport_size_sweep.py \
  --run-dir data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336 \
  --out-prefix figures/repro_kipan_broad_transport
```

KIPAN broad comparison boxplots:

```bash
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --validation-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/repro_kipan_boxplots
```

### Launch fresh broad runs with matching workflow defaults

Gentle broad launcher:

```bash
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```

Direct drivers:

```bash
python analyses/ch3_kipan_broad.py --help
python analyses/ch3_brca_broad_v2.py --help
python analyses/ch3_broad_gentle.py --help
```

## 5. A good mental model for this archive

If you are trying to reproduce a result, use this order of operations:

1. Start from the preserved run directory named in this guide.
2. Inspect the frozen CSVs and figures already stored there.
3. Use the matching rerender script to regenerate the manuscript-facing outputs.
4. Only if you want a fresh run, use the listed driver or shell launcher for
   that workflow family.

That is the intended user path through this repository.
