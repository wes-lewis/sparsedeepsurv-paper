# Publication Manifest

This document records the current publication-oriented keep set for the paper
companion repo. The governing principle is:

- Preserve the exact frozen artifacts that support the main figures,
  appendix figures, appendix tables, and related supplementary data in the
  manuscript source tree.
- Avoid rerunning analyses or regenerating results.
- Treat this repo as a curated archival companion rather than a full
  from-scratch recomputation pipeline.

The local archival snapshot branch for the paper repo is:

- `archive-local-20260413` at commit `d732c27`

The working cleanup branch is:

- `polish`


## Canonical Run Directories

These are the run directories currently considered canonical for the paper.

### Validation / all-model comparison

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`

Primary use:

- Main text Figure 3.2 (`fig:ch3-cindex-all-models`)
- Validation summary tables and supporting CSVs

Key files observed in this run:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `summary.csv`
- `screen_summary.csv`
- `goal0_fixed_init_summary.csv`
- `goal1_fixed_init_summary.csv`
- `goal0_matched_summary.csv`
- `goal1_matched_summary.csv`
- `runs.csv`
- `affinity_summary.csv`
- `affinity_pairs.csv`
- `histology.csv`
- `split_block_summary.csv`
- `within_between_split_summary.csv`
- `validation_init_consistency_summary.csv`

### Adaptive gentle run for KIPAN, BRCA, and PANCAN

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020`

Primary use:

- Main text Figure 3.3 (`fig:ch3-kipan-heatmaps`)
- Main text Figure 3.4 (`fig:ch3-paper-style-quant`)
- Appendix B BRCA heatmaps
- Appendix B BRCA linear heatmaps
- Appendix B PANCAN heatmaps
- Appendix B KIPAN and PANCAN gated-vs-random signal figures
- Appendix B recurrent-gene supplementary data/table support
- Appendix B selected-config tables and targeted-comparison support

Per-dataset subdirectories:

- `kipan/`
- `brca/`
- `pancan/`

Important frozen files in each dataset subdirectory:

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
- `fig_selected_stability_metrics_with_cindex_boxplot.png`
- `fig_selected_stability_metrics_with_cindex_boxplot_mlp_predictor.png`
- `fig_selected_stability_metrics_with_cindex_boxplot_linear_predictor.png`
- heatmap PNGs for LSPIN / Concrete / L-LSPIN / L-Concrete smooth vs no smooth

Nested canonical probe-analysis outputs:

- `kipan/linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep`
- `pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

These nested outputs contain the appendix signal and recurrence artifacts:

- `fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_summary.csv`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_aggregate_significance.csv`
- `*_recurrent_gene_supplement_table.csv`
- recurrent-gene candidate and recurrence summary CSVs

### Broad sweeps

Keep all four broad runs for now:

- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_kipan_broad_gentle_20260409_161803`
- `data/runs/ch3_brca_broad_gentle_20260409_161803`

Important note:

- The current manuscript draft is believed to use the `v2` broad runs for
  both KIPAN and BRCA.
- The KIPAN gentle broad run appears fully post-processed.
- The BRCA gentle broad directory currently looks incomplete at the top level
  and may only preserve partial worker outputs plus `run.log`.
- Because of that, keep both `v2` and `gentle` broad runs for now rather than
  trying to collapse them prematurely.


## Figure Mapping

This section maps manuscript figure usage to canonical paper-repo sources.

### Main text figures

#### F3.1

Overleaf target:

- `fig:ch3-framework`
- `figures/lspin_fig1.png`

Status:

- Source code or source asset not yet traced.
- If located later, preserve it, but this is not currently blocking the paper
  repo cleanup.

#### F3.2

Overleaf target:

- `fig:ch3-cindex-all-models`
- `figures/fig_validation_cindex_all_models.png`

Canonical source run:

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`

Likely relevant renderer/helper scripts:

- `analyses/validate_models.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`

#### F3.3

Overleaf target:

- `fig:ch3-kipan-heatmaps`
- `figures/diss_chap3_fig2_2.png`

Canonical source run:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan`

Likely relevant renderer/helper scripts:

- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/debug_adaptive_heatmaps.py`

#### F3.4

Overleaf target:

- `fig:ch3-paper-style-quant`
- `figures/disschap3_fig3_2.png`

Canonical source runs:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca`

Likely relevant renderer/helper scripts:

- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/rerender_boxplots.py`

#### F3.5

Overleaf target:

- `fig:ch3-broad-generalization`
- `figures/disschap3_fig4.png`

Keep-source runs:

- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_kipan_broad_gentle_20260409_161803`
- `data/runs/ch3_brca_broad_gentle_20260409_161803`

Current working interpretation:

- The manuscript likely uses the `v2` broad runs for both datasets.

Likely relevant renderer/helper scripts:

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/plot_feature_set_transport_size_sweep.py`

### Appendix B figures

#### BRCA nonlinear heatmaps

Overleaf target:

- `fig:ch3-supp-brca-heatmaps`
- `figures/diss_chap3_suppfig1.png`

Canonical source run:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca`

#### BRCA linear heatmaps

Overleaf target:

- `fig:ch3-supp-brca-heatmaps-linear`
- `figures/diss_chap3_suppfig_linear_brca_heatmap.png`

Canonical source run:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/brca`

#### PANCAN heatmaps

Overleaf target:

- `fig:ch3-supp-PANCAN-heatmaps`
- `figures/diss_chap3_suppfig2_2.png`

Canonical source run:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan`

#### PANCAN gated-vs-random signal boxplots

Overleaf target:

- `fig:ch3-gating-vs-random-in-linear-predictors`
- `figures/PANCAN_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`

Canonical source nested output:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

#### KIPAN gated-vs-random signal boxplots

Overleaf target:

- `fig:ch3-gating-vs-random-in-linear-predictors-kipan`
- `figures/kipan_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`

Canonical source nested output:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan/linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep`


## Table and Supplementary Data Mapping

The appendix tables should be treated as authoritative with respect to which
frozen runs matter. Unless a later contrary trace is found, use the following
working mapping.

### Targeted-comparison / validation tables

Likely canonical sources:

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020`

Most likely affected appendix tables:

- `tab:ch3-dataset-summary`
- `tab:ch3-architecture-defaults`
- `tab:ch3-graph-and-protocol`
- `tab:ch3-selected-configs`
- `tab:ch3-metric-definitions`

Working interpretation:

- Dataset/protocol/default tables are manuscript-authored summaries, but they
  should remain consistent with the targeted/validation runs and package
  defaults that produced the reported figures.
- `tab:ch3-selected-configs` should be treated as depending primarily on the
  adaptive-gentle run and its `selected_showcase_configs.csv` /
  `selected_comparison_configs.csv` outputs.

### Broad-sweep table

Canonical sources:

- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336`

Appendix table:

- `tab:ch3-broad-runs`

Working interpretation:

- Broad-related appendix tables should currently be treated as `v2`-based unless
  a later explicit trace proves otherwise.

### Recurrent-gene table and supplementary data

Canonical source:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/pancan/linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep`

Appendix table / supplementary data target:

- `tab:ch3-PANCAN-recurrent-gene-examples`
- supplementary data file referenced in Appendix B for recurrent PANCAN
  selected-gene examples

Canonical frozen CSV:

- `pancan_linear_probe_recurrent_gene_supplement_table.csv`


## Likely Scripts To Preserve

These scripts are the best current candidates to keep on the publication-facing
branch because they appear directly relevant to the canonical paper figures,
tables, or frozen output structure.

### Validation / comparison scripts

- `analyses/validate_models.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`
- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`

### Adaptive / heatmap / manuscript rendering scripts

- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/ch3_pancan_adaptive_v2.py`
- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/debug_adaptive_heatmaps.py`
- `analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh`

### Broad sweep scripts

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`

### Probe / recurrence / signal analysis scripts

- `analyses/quick_linear_gated_probe.py`
- `analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `analyses/plot_kipan_feature_set_cv_transport.py`
- `analyses/plot_pancan_gated_univariate_bias.py`
- `analyses/plot_kipan_gated_univariate_dotplot.py`


## Keep-But-Not-Yet-Classified

These items may still matter, but are not yet tied tightly enough to a specific
paper output to classify as canonical or removable.

- `analyses/ensure_project_data_artifacts.py`
- `analyses/relabel_gate_families.py`
- `analyses/rerender_boxplots.py`
- `analyses/mlp_baseline.py`


## Known Unresolved Items

### F3.1 schematic source

Overleaf uses:

- `figures/lspin_fig1.png`

The generating script or source asset has not yet been traced.

### Exact Overleaf assembly scripts

Some figure files in the manuscript source tree are clearly derived composites or
renamed exports, for example:

- `figures/diss_chap3_fig2_2.png`
- `figures/disschap3_fig3_2.png`
- `figures/disschap3_fig4.png`
- `figures/diss_chap3_suppfig1.png`
- `figures/diss_chap3_suppfig2_2.png`

The manifest above captures the canonical frozen source runs, but the exact last
assembly step into the Overleaf filenames may still need to be traced if the
goal becomes preserving every final rendering step.

### BRCA gentle broad completeness

The directory

- `data/runs/ch3_brca_broad_gentle_20260409_161803`

currently appears to preserve:

- `run.log`
- `_partial_worker*` directories

but not the top-level merged summaries/figures seen in the other broad runs.
Keep it for now, but do not assume it alone is sufficient to reproduce the
manuscript broad figure.


## Cleanup Rule

When pruning the paper repo for publication, default to:

- keep all files explicitly named in this manifest
- keep scripts directly supporting those files
- archive uncertain extras rather than deleting them immediately
- do not remove broad `v2` artifacts or the adaptive-gentle run
- do not remove nested KIPAN/PANCAN probe-analysis outputs inside the adaptive
  gentle run

Anything outside that set should be treated as a candidate for archival
relocation or exclusion from the publication-facing branch.
