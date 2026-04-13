# Setup

## Environment

Requires a Python environment with the dependencies listed in the sibling
[`sparsedeepsurv`](../sparsedeepsurv) repository:
[`pyproject.toml`](../sparsedeepsurv/pyproject.toml) and
[`requirements.txt`](../sparsedeepsurv/requirements.txt).

Core dependencies:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `sksurv`

## Path aliases

Canonical short-name symlinks for the preserved run directories are included in
the publication branch. Recreate them with:

```bash
bash extras/setup_paths.sh
```

This creates aliases such as `extras/data/runs/validation`,
`extras/data/runs/adaptive`, and `extras/data/runs/broad/{kipan_v2,brca_v2,...}`
that are used in the reproduction commands when the shortcuts are missing or
need to be refreshed locally.

## Path resolution

Analysis scripts resolve project paths relative to `extras/` via
[`extras/analyses/_paths.py`](extras/analyses/_paths.py). The natural working
directory for running scripts is `extras/` (the repo's analysis root).

The sibling `sparsedeepsurv` checkout is required; by default scripts resolve
it from the expected sibling layout. Set `SPARSEDEEPSURV_SRC` to point to
`sparsedeepsurv/src` to override.

## Optional environment variables

- `SPARSEDEEPSURV_SRC` — path to `sparsedeepsurv/src`
- `LSPIN_PYTORCH_ROOT` — path to a local `lspin-pytorch` checkout (legacy reference scripts only)
- `KIPAN_SOURCE_DIR`, `BRCA_SOURCE_DIR`, `PANCAN_SOURCE_DIR` — source directories for `extras/analyses/ensure_project_data_artifacts.py`

If unset, scripts attempt the expected sibling layout and fail with a clear
message if the dependency is missing.

## Non-canonical scripts

Not part of the canonical publication surface:

- `extras/analyses/ch3_kipan_mlp_reference.py`
- `extras/analyses/ch3_kipan_mlp_overlay_figure.py`
- `extras/analyses/mlp_baseline.py`
- `extras/analyses/ensure_project_data_artifacts.py`

Placeholder wrappers (`extras/analyses/ch3_brca_targeted.py`,
`extras/analyses/ch3_brca_broad.py`) document intended migration points but are
not fully implemented. Older exploratory material is under
[`extras/archive/`](extras/archive/).
