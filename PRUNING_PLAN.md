# First-Pass Pruning Plan

This document is a non-destructive pruning plan for the `polish` branch of
`sparsedeepsurv-paper`.

It is intentionally a planning document only:

- No files are deleted or moved yet.
- The archival safety branch is already in place:
  `archive-local-20260413` at commit `d732c27`.
- The goal is to define what should remain in the publication-facing branch,
  what should later be moved into an archive area, and what still needs a
  decision.

This plan follows the canonical keep-set described in
`PUBLICATION_MANIFEST.md`.


## Pruning Principles

For the publication-facing companion repo, prioritize:

- frozen run outputs that directly support the paper and appendix material
- scripts that generated or rerendered those kept outputs
- small frozen supplementary CSVs and tables actually referenced by the
  manuscript
- minimal documentation explaining the frozen-artifact structure

Deprioritize:

- superseded exploratory analyses
- recovery notes and local workflow documents
- scripts tied to old migrations or obsolete comparisons
- tracked assets that are not used in the paper pipeline


## Bucket A: Keep In Publication Branch

These are the files that should remain on the publication-facing
branch unless a later contradiction is discovered.

### Documentation to keep

- `.gitignore`
- `PUBLICATION_MANIFEST.md`
- `PRUNING_PLAN.md`

Future addition recommended:

- add a real top-level `README.md` that explains this repo is a curated archival
  companion for the paper figures rather than a full from-scratch
  recomputation repo.

### Core analysis drivers to keep

- `analyses/validate_models.py`
- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/ch3_pancan_adaptive_v2.py`
- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/ch3_broad_gentle.py`

### Figure / table rendering helpers to keep

- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`
- `analyses/rerender_boxplots.py`
- `analyses/plot_feature_set_transport_size_sweep.py`
- `analyses/quick_linear_gated_probe.py`
- `analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `analyses/plot_kipan_feature_set_cv_transport.py`
- `analyses/plot_pancan_gated_univariate_bias.py`
- `analyses/plot_kipan_gated_univariate_dotplot.py`
- `analyses/debug_adaptive_heatmaps.py`

### Launch scripts worth keeping

- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`
- `analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`

Even if these are not meant for public rerunning, they are useful provenance for
how the canonical frozen runs were produced.

### Configs to keep for now

- `configs/brca_broad.yaml`
- `configs/brca_targeted.yaml`
- `configs/kipan_broad.yaml`
- `configs/kipan_targeted.yaml`

These are small, low-risk provenance files and may still be helpful even if the
repo is curated around frozen outputs.

### Tracked data / figure assets to keep for now

- `data/gene_sets/enrichr/MSigDB_Hallmark_2020.gmt`
- `figures/ch3_kipan_mlp_overlay/fig_concrete_cindex_vs_khard_mlp_overlay.pdf`
- `figures/ch3_kipan_mlp_overlay/fig_lspin_cindex_vs_khard_mlp_overlay.pdf`
- `figures/ch3_kipan_mlp_overlay/mlp_reference_summary.csv`

Rationale:

- These are already tracked.
- They are small.
- They may still support discussion/provenance around the KIPAN MLP reference
  comparison.

### Test to keep

- `tests/smoke/test_rerender_boxplots.py`

It is narrow, but better than removing the only test entirely.


## Bucket B: Move To Archive Area Later

These are the strongest candidates to remove from the publication-facing branch
or relocate into an `archive/` or `internal/` folder later.

### Recovery / local working notes

- `DELIVERABLES.txt`
- `PERFORMANCE_RECOVERY_ANALYSIS.md`
- `README_RECOVERY.md`
- `RECOVERY_EXECUTION_GUIDE.md`
- `SUMMARY_KEY_FINDINGS.md`
- `VISUAL_ANALYSIS.md`

Rationale:

- These read like local working notes and recovery documents, not publication
  companion docs.
- They contain a lot of local-path and workflow-specific material.

### Older validation / tuning sweep launchers

- `analyses/run_validate_and_plot.sh`
- `analyses/run_validate_goal1_v3.sh`
- `analyses/run_validate_goal1_v4_lineargated_stg_sweep.sh`
- `analyses/run_validate_goal1_v5_arch_fix.sh`
- `analyses/run_validate_goal1_v5_baseline_best_configs.sh`
- `analyses/run_validate_goal1_v5_recovery_sweep.sh`
- `analyses/run_validate_goal1_brca_gentle_sparse_sweep.sh`

Rationale:

- These appear to be intermediate workflow scripts rather than canonical
  publication drivers.

### Intermediate tuning scripts

- `analyses/validate_goal1_tuning_sweep.py`
- `analyses/validate_lspin_patience_sweep.py`
- `analyses/validate_lspin_reg_sweep.py`

Rationale:

- Useful historically, but they look like tuning-stage exploration rather than
  required frozen-publication provenance.

### Older generation scripts likely superseded by v2 / gentle workflows

- `analyses/ch3_brca_adaptive_v1.py`
- `analyses/ch3_kipan_adaptive_v1.py`
- `analyses/ch3_pancan_adaptive_v1.py`
- `analyses/ch3_kipan_targeted.py`
- `analyses/ch3_kipan_targeted_v3.py`
- `analyses/ch3_brca_broad.py` is currently kept, but could later move to
  archive if `ch3_brca_broad_v2.py` is confirmed to fully supersede it.
- `analyses/ch3_kipan_broad.py` is currently kept, but could later move to
  archive if broad provenance is fully captured elsewhere.

Rationale:

- These appear to be older stages in the evolution of the paper workflows.


## Bucket C: Hold For Now / Needs Decision

These files should not be pruned yet because they are ambiguous, incomplete, or
still entangled with known manuscript outputs.

### Potentially legacy but still informative

- `analyses/ensure_project_data_artifacts.py`
- `analyses/relabel_gate_families.py`
- `analyses/mlp_baseline.py`
- `analyses/plot_kipan_boxplots.py`

### KIPAN MLP reference files

- `analyses/ch3_kipan_mlp_overlay_figure.py`
- `analyses/ch3_kipan_mlp_reference.py`
- `analyses/run_brca_lspin_targeted.py`

Rationale:

- These include legacy dependencies and may not be part of the final paper
  core story, but they also connect to tracked figure assets already in the repo.
- Do not prune until it is clear they are truly irrelevant to the paper
  artifacts.

### Incomplete or stub-like files

- `analyses/ch3_brca_targeted.py`

Rationale:

- This file is not complete, but pruning it should wait until we decide whether
  the publication-facing branch should retain unfinished migration remnants for
  provenance or remove them for cleanliness.


## Proposed Later File Moves

This is the first-pass proposed restructuring, without executing it yet.

### Future top-level structure

- `README.md`
- `CHAPTER3_PUBLICATION_MANIFEST.md`
- `CHAPTER3_PRUNING_PLAN.md`
- `analyses/`
  only canonical scripts
- `archive/`
  older scripts and recovery docs
- `tests/`
  smoke tests
- `configs/`
  only if still useful for provenance

### Candidate future archive destinations

- `archive/docs/`
  recovery notes, deliverables, visual-analysis notes
- `archive/analyses/`
  v1 scripts, tuning sweeps, recovery launchers, superseded validation scripts


## Non-Destructive Next Steps

When we start acting on this plan, do it in this order:

1. Add a proper `README.md` that explains the repo scope.
2. Create an `archive/` directory on `polish`.
3. Move Bucket B files into `archive/` rather than deleting them.
4. Keep Bucket C untouched until individually reviewed.
5. Only after that consider removing clearly obsolete archive contents from the
   publication-facing branch, if desired.


## Current Intent

This plan is deliberately conservative.

- It preserves everything needed for figure/table provenance.
- It avoids deleting anything before the branch structure and manifest are in
  place.
- It gives us a clean next move: reorganize first, prune later.
