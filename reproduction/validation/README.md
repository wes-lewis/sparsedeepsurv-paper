# Validation

This folder covers the visible validation figure and its supporting summaries.

## Visible outputs

The visible validation figure in the paper is:

- Fig. 2: all-model validation comparison

The canonical preserved run also contains closely related validation outputs
used for the same result family:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`
- `screen_summary.csv`

## Canonical preserved run

```text
data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## If you only want to inspect the preserved result

Look directly in:

```text
data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## If you want to rerender the visible validation outputs

From the repository root:

```bash
python analyses/plot_cindex_all_models.py \
  --run-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/validation

python analyses/plot_validation_init_consistency.py \
  --run-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/validation
```

Optional supplementary validation plots:

```bash
python analyses/plot_validation_supp_boxplots.py \
  --results-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## If you want to launch a fresh run with the same workflow defaults

Use the shell launcher:

```bash
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

This is the clearest end-to-end entry point for validation. It creates a new
run under `data/runs/` and then renders the main validation figures
from that new run.

## Driver and plotting scripts used here

- `analyses/validate_models.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`
- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`
