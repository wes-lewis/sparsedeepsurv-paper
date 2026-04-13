# `sparsedeepsurv-paper`

Publication archive for the paper analyses. This repository preserves the canonical runs, processed artifacts, figures, and analysis entry points used in the manuscript.

## Quick Start

This repository supports three primary workflows for reproducing paper results:

1. **Validation and Model Comparison** — All-model C-index validation and initialization consistency
2. **Targeted Comparison** — Per-dataset configuration and metric summaries
3. **Broad Sparsity Sweeps** — Sparsity-versus-performance analysis across model families

For detailed instructions on each workflow, see the [reproduction guide](reproduction/README.md).

## Setup

This repository requires:

- The sibling [`sparsedeepsurv`](../sparsedeepsurv) repository (for source imports)
- Python environment with: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `sksurv`

Environment setup and path configuration are documented in [SETUP.md](SETUP.md).

## Reproduction Workflows

Each workflow includes:

- **Preserved run outputs**: Canonical models and predictions from the paper
- **Data artifacts**: Processed splits and supplementary CSVs referenced in the manuscript
- **Figure regeneration**: Scripts to rerender publication figures from preserved outputs
- **Rerun capability**: Ability to launch new runs with identical parameterization

For comprehensive details on each workflow, canonical run locations, and regeneration commands:

- [Validation workflow](reproduction/README.md#validation-and-model-comparison)
- [Targeted comparison workflow](reproduction/README.md#targeted-comparison)
- [Broad sweep workflow](reproduction/README.md#broad-sparsity-sweeps)

## Repository Organization

- `analyses/`: Analysis scripts, figure renderers, and validation utilities
- `reproduction/`: Workflow guides and primary documentation for running paper analyses
- `overall/`, `validation/`, `targeted_comparison/`, `broad_sweep/`: Detailed workflow documentation
- `extras/`: Reference data, figures, tables, and archived materials

For the complete figure-by-figure mapping, see [PUBLICATION_MANIFEST.md](PUBLICATION_MANIFEST.md).

## Repository Scope

This archive preserves the canonical outputs and scripts used for the paper. Results across different environments will be approximately similar rather than byte-for-byte identical. This is a curated reproduction package, not a raw-data end-to-end pipeline.

For additional context, see [PUBLICATION_MANIFEST.md](PUBLICATION_MANIFEST.md) for detailed run provenance and scope documentation.
