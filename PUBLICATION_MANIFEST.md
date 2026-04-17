# Publication Manifest

This document records the canonical preserved runs in this repository and the
main entry points that regenerate or rerun the corresponding result families.

The intent is simple:

- preserve the frozen run outputs that support the paper results
- preserve the scripts that regenerate figures and summaries from those runs
- preserve the workflow drivers needed to launch another run with the same
  parameterization

This is an archival provenance document, not a manuscript-assembly log.

## Branch context

The local archival snapshot branch for the paper repo is:

- `archive-local-20260413` at commit `d732c27`

The publication-facing cleanup branch is:

- `polish`

## Canonical workflows

The repository centers on three workflow families:

- validation and all-model comparison
- targeted comparisons
- broad sweeps

For each workflow family, the preserved materials are intended to support three
user actions:

- inspect the exact frozen outputs retained in the archive
- regenerate the paper-facing figures and summaries from those outputs
- launch another run with the same workflow defaults

Users should expect approximately similar rerun results rather than byte-for-
byte identical outputs across environments.

## 1. Validation and all-model comparison

Canonical preserved run:

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`

This run is the canonical source for the validation comparison outputs,
including the all-model C-index summaries and initialization-consistency
results.

Representative preserved outputs include:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`
- `screen_summary.csv`
- `affinity_summary.csv`

Main regeneration entry points:

- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`

Main rerun entry points:

- `analyses/validate_models.py`
- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`

## 2. Targeted comparisons

Canonical preserved run root:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020`

Canonical per-dataset subdirectories:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan`

These runs are the canonical source for:

- selected gate heatmaps
- selected configuration summaries
- selected stability/c-index comparison boxplots
- per-dataset notebook-style summary CSVs
- derived probe-style analyses for KIPAN and PANCAN

Representative preserved outputs in each dataset subdirectory include:

- `selected_showcase_configs.csv`
- `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`
- `selected_best_heatmap_configs.csv`
- `heatmap_models_summary.csv`
- `notebook_style_models_runs.csv`
- `notebook_style_models_accuracy_summary.csv`
- `notebook_style_models_affinity_summary.csv`
- `notebook_style_models_risk_summary.csv`
- `notebook_style_models_cluster_summary.csv`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`

Main regeneration entry points:

- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/plot_kipan_boxplots.py`
- `analyses/rerender_boxplots.py`

Main rerun entry points:

- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/ch3_pancan_adaptive_v2.py`
- `analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh`

### Derived probe-style outputs preserved under the targeted-comparison runs

Nested canonical probe-analysis directories include:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan/linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

These directories are the canonical source for:

- gated-vs-random signal summaries
- patient-subset predictivity summaries
- recurrent-gene tables and related recurrence summaries

Main regeneration entry points:

- `analyses/quick_linear_gated_probe.py`
- `analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `analyses/plot_pancan_gated_univariate_bias.py`
- `analyses/plot_kipan_gated_univariate_dotplot.py`
- `analyses/plot_kipan_feature_set_cv_transport.py`

## 3. Broad sweeps

Canonical preserved broad runs:

- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_recovered_20260326`
- `data/runs/ch3_kipan_broad_gentle_20260409_161803`
- `data/runs/ch3_brca_broad_gentle_20260413_190053`

These runs are the canonical source for the sparsity-versus-performance sweep
outputs retained in the archive.

Main regeneration entry points:

- `analyses/plot_feature_set_transport_size_sweep.py`
- `analyses/plot_kipan_boxplots.py`

Main rerun entry points:

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`

## Secondary helpers kept in the publication branch

These are retained because they may still be useful for provenance or
convenience, but they are not the primary workflow entry points:

- `analyses/ensure_project_data_artifacts.py`
- `analyses/relabel_gate_families.py`
- `analyses/mlp_baseline.py`
- `analyses/ch3_kipan_mlp_reference.py`
- `analyses/ch3_kipan_mlp_overlay_figure.py`

## Known unresolved item

The schematic source corresponding to `figures/lspin_fig1.png` has not yet been
traced to a generating script or source asset in this repository. This does not
affect the main run-provenance or rerender workflows described above.

## Cleanup rule

When pruning or simplifying the publication-facing branch, default to:

- keep the canonical run directories named here
- keep the regeneration and rerun entry points named here
- keep nested probe-analysis outputs under the targeted-comparison run tree
- archive uncertain extras rather than deleting them immediately

Anything outside that set should be treated as a candidate for archival
relocation rather than part of the core publication surface.
