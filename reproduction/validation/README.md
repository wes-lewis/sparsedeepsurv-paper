# Validation — Fig. 2

**Canonical run**: `extras/data/runs/validation`

(Timestamped original: `validate_goal1_gentle_all_kipan_brca_20260408_175906`)

Key outputs:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`, `screen_summary.csv`, `affinity_summary.csv`, `runs.csv`

## Regeneration

BRCA-dependent reruns use the canonical processed input artifact at
`extras/data/processed/tcga_brca20260214_001423`, which can be rebuilt with:

```bash
cd extras
python analyses/prepare_tcga_brca_firehose.py
```

```bash
cd extras

python analyses/plot_cindex_all_models.py \
  --run-dir data/runs/validation \
  --out-dir figures/validation

python analyses/plot_validation_init_consistency.py \
  --run-dir data/runs/validation \
  --out-dir figures/validation
```

Supplementary validation plots:

```bash
cd extras
python analyses/plot_validation_supp_boxplots.py \
  --results-dir data/runs/validation
```

## Rerun

```bash
cd extras
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

## Scripts

- `extras/analyses/prepare_tcga_brca_firehose.py`
- `extras/analyses/validate_models.py`
- `extras/analyses/plot_cindex_all_models.py`
- `extras/analyses/plot_validation_init_consistency.py`
- `extras/analyses/plot_validation_supp_boxplots.py`
- `extras/analyses/run_validate_goal1_gentle_all_kipan_brca.sh`
