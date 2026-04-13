# Setup

## Environment

Requires a Python environment with the dependencies listed in the sibling
[`sparsedeepsurv`](../sparsedeepsurv) repository:
[`pyproject.toml`](../sparsedeepsurv/pyproject.toml) and
[`requirements.txt`](../sparsedeepsurv/requirements.txt).

Core dependencies used by the analysis scripts:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `sksurv`

## Path resolution

Analysis scripts resolve project paths relative to the repository root via
[`analyses/_paths.py`](analyses/_paths.py). The sibling `sparsedeepsurv`
checkout is required; by default scripts expect it in the sibling location.
Set `SPARSEDEEPSURV_SRC` to point to `sparsedeepsurv/src` to override.

## Optional environment variables

- `SPARSEDEEPSURV_SRC` — path to `sparsedeepsurv/src`
- `LSPIN_PYTORCH_ROOT` — path to a local `lspin-pytorch` checkout, for legacy reference scripts
- `KIPAN_SOURCE_DIR`, `BRCA_SOURCE_DIR`, `PANCAN_SOURCE_DIR` — source directories for `analyses/ensure_project_data_artifacts.py`

If unset, scripts attempt the expected sibling layout and fail with a clear
message if the dependency is missing.

## Non-canonical scripts

Not part of the canonical publication surface:

- `analyses/ch3_kipan_mlp_reference.py`
- `analyses/ch3_kipan_mlp_overlay_figure.py`
- `analyses/mlp_baseline.py`
- `analyses/ensure_project_data_artifacts.py`

Placeholder wrappers (`analyses/ch3_brca_targeted.py`,
`analyses/ch3_brca_broad.py`) document intended migration points but are not
fully implemented. Older exploratory material is under [`archive/`](archive/).
