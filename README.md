# `sparsedeepsurv-paper`

This repository is the publication-oriented archive for the paper analyses.
It is structured to preserve the canonical runs, figures, tables, and analysis
entry points used in the manuscript-facing workflow while reducing dependence on
workstation-specific paths.

The archive is not intended to be a from-scratch raw-data reproduction package.
Instead, it is a curated companion to the [`sparsedeepsurv`](../sparsedeepsurv)
codebase and the preserved processed artifacts under [`data/processed`](data/processed)
and [`data/runs`](data/runs). If you run the canonical scripts in this repo with
the packaged processed artifacts, you should obtain approximately similar
results and regenerated figures/tables from the preserved analysis outputs.

## Minimal setup

This repository assumes a Python environment that can import the dependencies
used by the analysis scripts, including at minimum:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `sksurv`

It also assumes access to the sibling [`sparsedeepsurv`](../sparsedeepsurv)
repository, because the paper analyses import the package directly from its
source tree. By default the scripts look for that checkout in the expected
sibling location. If your layout differs, set `SPARSEDEEPSURV_SRC` to point to
`sparsedeepsurv/src`.

For environment setup, use the sibling `sparsedeepsurv` repository's package
metadata and dependency lists, especially
[`pyproject.toml`](../sparsedeepsurv/pyproject.toml) and
[`requirements.txt`](../sparsedeepsurv/requirements.txt). Any paper-specific
extras or local helper dependencies should be treated as additions on top of
that base environment.

The goal of the `polish` branch is to preserve the canonical processed
artifacts, run outputs, and figure-generation entry points used for the paper,
not to serve as a raw-data end-to-end reproduction package.

## What a user should take away

This archive is meant to support three concrete use cases:

- inspect the exact preserved runs that underlie the paper figures and tables
- rerender the paper figures and summary outputs from those preserved runs
- launch a new run with the same model families, grids, and dataset-specific
  defaults as the preserved workflows

In other words, a user should come away thinking: "I can see which frozen run
directories matter, I can regenerate the manuscript-facing outputs from them,
and I can start a fresh run with the same parameterization if I want to check
stability or reproduce the workflow approximately."

The preserved materials are strongest for provenance and rerendering. They are
also sufficient to instantiate another identically parameterized run for the
main validation, adaptive, and broad workflows, because the corresponding
analysis drivers and processed split artifacts are included here. What they do
not guarantee is byte-for-byte reproduction of every saved output across
machines or software stacks; users should expect approximately similar results
rather than identical files.

## Layout

- `analyses/`: analysis scripts, figure renderers, and validation utilities
- `configs/`: lightweight config files for a few helper wrappers
- `data/processed/`: project-local processed split artifacts used by the paper analyses
- `data/runs/`: canonical saved run outputs used for the paper figures/tables
- `figures/`: generated manuscript and appendix figures
- `tables/`: generated summary tables
- `archive/`: noncanonical recovery notes, older scripts, and local project history retained for reference but not part of the main publication surface

## Canonical preserved runs

The main preserved run directories are:

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`
- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020`
- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_kipan_broad_gentle_20260409_161803`
- `data/runs/ch3_brca_broad_gentle_20260409_161803`

For the figure-by-figure mapping used to curate this archive, see
[`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md).

## Reproduction workflows

Most users should think in terms of workflows rather than individual scripts.

### 1. Validation and all-model comparison

Use this when you want to reproduce the validation comparison across model
families or regenerate the corresponding summary figures and tables.

Canonical preserved run:

- `data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906`

Main entry points:

- `analyses/validate_models.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`
- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`

User takeaway:

- the preserved run is enough to inspect the exact reported outputs
- the scripts are enough to rerender the validation figures from that run
- the driver is enough to launch a fresh run with the same validation setup

### 2. Adaptive gated analyses for KIPAN, BRCA, and PANCAN

Use this when you want the main gated heatmaps, boxplots, selected-config
summaries, and the probe-style appendix analyses built from the adaptive runs.

Canonical preserved run:

- `data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020`

Main entry points:

- `analyses/ch3_kipan_adaptive_v2.py`
- `analyses/ch3_brca_adaptive_v2.py`
- `analyses/ch3_pancan_adaptive_v2.py`
- `analyses/render_adaptive_manuscript_figures.py`
- `analyses/rerender_boxplots.py`
- `analyses/quick_linear_gated_probe.py`
- `analyses/plot_pancan_linear_probe_gene_recurrence.py`
- `analyses/plot_pancan_gated_univariate_bias.py`
- `analyses/plot_kipan_gated_univariate_dotplot.py`
- `analyses/plot_kipan_feature_set_cv_transport.py`

User takeaway:

- the preserved adaptive run tree contains the exact selected configs and
  derived summaries used for the paper-facing outputs
- the helper scripts rerender the figures and probe analyses from those frozen
  outputs
- the adaptive drivers define the same model grids and training defaults needed
  to launch a new run with matching parameterization

### 3. Broad sweeps

Use this when you want the sparsity-versus-performance sweep outputs and the
corresponding broad comparison figures.

Canonical preserved runs:

- `data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336`
- `data/runs/ch3_kipan_broad_gentle_20260409_161803`
- `data/runs/ch3_brca_broad_gentle_20260409_161803`

Main entry points:

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/plot_feature_set_transport_size_sweep.py`
- `analyses/plot_kipan_boxplots.py`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`

User takeaway:

- the preserved broad runs show the exact sweep outputs retained in the archive
- the broad drivers are the parameterized rerun entry points
- the gentle broad BRCA run appears less complete than the `v2` broad runs, so
  users should treat the `v2` runs as the strongest canonical broad reference

## Optional and legacy helpers

Some scripts remain useful as references or convenience wrappers, but they are
not part of the minimal canonical publication surface:

- `analyses/ch3_kipan_mlp_reference.py`
- `analyses/ch3_kipan_mlp_overlay_figure.py`
- `analyses/mlp_baseline.py`
- `analyses/ensure_project_data_artifacts.py`

In addition, placeholder wrappers such as `analyses/ch3_brca_targeted.py` and
`analyses/ch3_brca_broad.py` are retained to document intended migration points,
but they are not fully implemented in this archive. Older exploratory material,
recovery notes, and superseded scripts have been moved under [`archive/`](archive/).

## Path conventions

Publication-facing scripts in `analyses/` now resolve project paths relative to
the repository root through [`analyses/_paths.py`](analyses/_paths.py). Running
from inside the repo should therefore avoid hardcoded local paths.

Some optional helper scripts still depend on sibling development checkouts:

- `SPARSEDEEPSURV_SRC`: optional override for the `sparsedeepsurv/src` location
- `LSPIN_PYTORCH_ROOT`: optional path to a local `lspin-pytorch` checkout for a
  small number of legacy comparison/reference scripts
- `KIPAN_SOURCE_DIR`, `BRCA_SOURCE_DIR`, `PANCAN_SOURCE_DIR`: optional source
  directories for `analyses/ensure_project_data_artifacts.py`

If these variables are not set, the scripts first try the expected sibling
layout next to this repository and then fail with a clear message if the
dependency is genuinely missing.

## Typical usage

From the repository root:

```bash
python analyses/validate_models.py --help
python analyses/render_adaptive_manuscript_figures.py --help
python analyses/plot_validation_supp_boxplots.py --help
```

The shell helpers in `analyses/` also assume they are launched from within this
repository and derive paths relative to their own location.

## Scope and current status

The `polish` branch is intended to be the public-facing version of this paper
archive. The canonical outputs used in the paper are preserved here, and
the active scripts are being normalized to use relative paths and project-local
artifacts. Older exploratory material is retained under `archive/` so the main
publication surface stays smaller and easier to understand.
