# Reproduction Workflows

This guide provides detailed instructions for reproducing the paper results. Three main workflows represent the primary analysis conducted for the manuscript.

## Validation and Model Comparison

**Purpose**: Validation of the sparse deep survival models across initialization conditions and comparison to standard baseline models.

**Canonical run**: `extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`

**Key outputs**:
- `fig_validation_cindex_ci_all_models.png` — Harrell C-index with confidence intervals
- `fig_cindex_all_models_pointest.png` — Point estimates across all models
- `validation_init_consistency_summary.csv` — Initialization stability analysis
- `summary.csv`, `screen_summary.csv`, `affinity_summary.csv` — Supplementary metrics

**Regenerate figures**:
```bash
python analyses/plot_cindex_all_models.py
python analyses/plot_validation_init_consistency.py
python analyses/plot_validation_supp_boxplots.py
```

**Rerun analysis**:
```bash
python analyses/validate_models.py
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

**Paper figures**: Overall validation comparison figures in main manuscript

---

## Targeted Comparison

**Purpose**: Detailed per-dataset hyperparameter configuration analysis, stability metrics, and probe-based feature importance evaluations.

**Canonical runs**:
- `extras/data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020` (main)
- Per-dataset subdirectories: `kipan/`, `brca/`, `pancan/`

**Key outputs per dataset**:
- Configuration summaries: `selected_showcase_configs.csv`, `selected_comparison_configs.csv`
- Metrics: `selected_comparison_metrics_summary.csv`, `notebook_style_models_*_summary.csv`
- Stability/C-index comparisons: `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- Gate heatmaps: `fig_gate_heatmap_*.png`
- Probe outputs: Gated vs. random signal summaries, patient-subset predictivity, recurrent-gene tables

**Regenerate figures**:
```bash
python analyses/render_adaptive_manuscript_figures.py
python analyses/plot_kipan_boxplots.py
python analyses/rerender_boxplots.py
python analyses/quick_linear_gated_probe.py
python analyses/plot_pancan_linear_probe_gene_recurrence.py
python analyses/plot_pancan_gated_univariate_bias.py
python analyses/plot_kipan_gated_univariate_dotplot.py
python analyses/plot_kipan_feature_set_cv_transport.py
```

**Rerun analysis**:
```bash
python analyses/ch3_kipan_adaptive_v2.py
python analyses/ch3_brca_adaptive_v2.py
python analyses/ch3_pancan_adaptive_v2.py
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

**Paper figures**: Configuration heatmaps, stability boxplots, per-dataset summaries in main manuscript and appendix

---

## Broad Sparsity Sweeps

**Purpose**: Comprehensive evaluation of model performance across the full sparsity landscape for multiple datasets and model architectures.

**Canonical runs**:
- `extras/data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336` (primary KIPAN)
- `extras/data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336` (primary BRCA)
- `extras/data/runs/ch3_kipan_broad_gentle_20260409_161803` (supplementary KIPAN)
- `extras/data/runs/ch3_brca_broad_gentle_20260409_161803` (supplementary BRCA)

**Key outputs**:
- Sparsity-versus-performance curves across feature set sizes
- Model family comparisons under varying regularization
- Curve summaries and statistical evaluations

**Regenerate figures**:
```bash
python analyses/plot_feature_set_transport_size_sweep.py
python analyses/plot_kipan_boxplots.py
```

**Rerun analysis**:
```bash
python analyses/ch3_kipan_broad.py
python analyses/ch3_brca_broad_v2.py
python analyses/ch3_broad_gentle.py
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```

**Paper figures**: Sparsity sweep curves and comparative performance landscape in main manuscript

---

## All Workflows: Quick Verification

To verify that the environment is properly configured, test each workflow with a help command:

```bash
python analyses/validate_models.py --help
python analyses/render_adaptive_manuscript_figures.py --help
python analyses/plot_feature_set_transport_size_sweep.py --help
```

Successful execution indicates that dependencies are satisfied and paths are correctly resolved.

---

## Experimental Data and Supplementary Materials

Each workflow references processed data artifacts and supplementary CSV files that support the paper results. These are organized under `extras/data/`:

- `extras/data/processed/`: Processed dataset splits used by analyses
- `extras/data/runs/`: Canonical run outputs (predictions, model states, derived metrics)

Regeneration scripts automatically reference these archives and handle path resolution through `analyses/_paths.py`.

For detailed figure-by-figure provenance and run documentation, see [PUBLICATION_MANIFEST.md](../PUBLICATION_MANIFEST.md).
