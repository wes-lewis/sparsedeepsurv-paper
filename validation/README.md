# Validation

This folder covers the validation comparison and its supporting summary outputs.

## Figure correspondence

The paper figure associated with this workflow is:

- Fig. 2: all-model validation comparison

Related preserved outputs from the same workflow include:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`
- `screen_summary.csv`
- `affinity_summary.csv`
- `runs.csv`

## Canonical preserved run

```text
extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## Primary preserved outputs

Inspection should begin in:

```text
extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## Regeneration entry points

From the repository root:

```bash
python analyses/plot_cindex_all_models.py \
  --run-dir extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir extras/figures/validation

python analyses/plot_validation_init_consistency.py \
  --run-dir extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir extras/figures/validation
```

Related supplementary validation plots:

```bash
python analyses/plot_validation_supp_boxplots.py \
  --results-dir extras/data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906
```

## Rerun entry point

Primary launcher:

```bash
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

This launcher creates a new run under `extras/data/runs/` and then renders the main
validation figures from that run.

## Primary scripts

- `analyses/validate_models.py`
- `analyses/plot_cindex_all_models.py`
- `analyses/plot_validation_init_consistency.py`
- `analyses/plot_validation_supp_boxplots.py`
- `analyses/run_validate_goal1_gentle_all_kipan_brca.sh`
