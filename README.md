# `sparsedeepsurv-paper`

Analysis code and preserved run archive for the sparse deep survival models
paper (Chapter 3). Accompanies the
[`sparsedeepsurv`](https://github.com/wes-lewis/sparsedeepsurv) package.

## Reproduction

[`reproduction/README.md`](reproduction/README.md) is the primary guide.
Per-workflow detail:

- [`reproduction/validation/README.md`](reproduction/validation/README.md) — Fig. 2
- [`reproduction/targeted_comparison/README.md`](reproduction/targeted_comparison/README.md) — Figs. 3–4, S1–S5
- [`reproduction/broad_sweep/README.md`](reproduction/broad_sweep/README.md) — Fig. 5

For the figure-by-figure provenance mapping, see
[`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md).

## Canonical preserved runs

All run outputs and analysis scripts are under [`extras/`](extras/).

| Workflow | Path |
|---|---|
| Validation | `extras/data/runs/validation` |
| Targeted comparison | `extras/data/runs/adaptive` |
| Broad sweep (primary) | `extras/data/runs/broad/kipan_v2`, `broad/brca_v2` |
| Broad sweep (supplementary) | `extras/data/runs/broad/kipan_gentle`, `broad/brca_gentle` |

Short names are symlinks to the timestamped originals in `extras/data/runs/`.
Run `bash extras/setup_paths.sh` once after cloning to create them (data
directories are gitignored and symlinks are not tracked).

## Layout

- `extras/analyses/` — analysis drivers, figure renderers, validation utilities
- `extras/data/runs/` — canonical run outputs
- `extras/data/processed/` — processed split artifacts and derived outputs
- `extras/figures/` — generated manuscript and appendix figures
- `extras/tables/` — generated summary tables
- `extras/archive/` — noncanonical recovery notes, older scripts, project history
- `reproduction/` — workflow guides for reproducing paper results
- `overall/` — figure and workflow index

## Setup

See [`SETUP.md`](SETUP.md) for environment requirements, path conventions, and
optional override variables.
