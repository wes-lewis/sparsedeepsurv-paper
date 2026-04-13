# Reproduction

Three workflows cover the primary paper analyses. All commands run from
`extras/` (the analysis root of this repository). Short-name paths for
canonical run directories are symlinks in `extras/data/runs/`.

## Processed Inputs

The BRCA analyses use the processed dataset artifact
`extras/data/processed/tcga_brca20260214_001423`. That artifact can be rebuilt
from the raw TCGA Firehose BRCA RNA, CNV, and clinical files with:

```bash
cd extras
python analyses/prepare_tcga_brca_firehose.py
```

This writes the canonical `splits_and_core.npz`, `X_scaled.npz`, and
`models/scaler.joblib` files expected by the BRCA validation, targeted, and
broad-sweep drivers, together with an `artifact_manifest.json` documenting the
preprocessing path.

---

## Validation — Fig. 2

**Canonical run**: `extras/data/runs/validation`

Key outputs:

- `fig_validation_cindex_ci_all_models.png`
- `fig_cindex_all_models_pointest.png`
- `validation_init_consistency_summary.csv`
- `summary.csv`, `screen_summary.csv`, `affinity_summary.csv`, `runs.csv`

**Regenerate figures**:

```bash
cd extras
python analyses/plot_cindex_all_models.py \
  --run-dir data/runs/validation \
  --out-dir figures/validation

python analyses/plot_validation_init_consistency.py \
  --run-dir data/runs/validation \
  --out-dir figures/validation
```

**Rerun**:

```bash
cd extras
bash analyses/run_validate_goal1_gentle_all_kipan_brca.sh
```

---

## Targeted Comparison — Figs. 3–4, S1–S5

**Canonical runs**: `extras/data/runs/adaptive/{kipan,brca,pancan}`

Key outputs per dataset subdirectory:

- `selected_showcase_configs.csv`, `selected_comparison_configs.csv`
- `selected_comparison_metrics_summary.csv`, `heatmap_models_summary.csv`
- `fig_selected_stability_metrics_with_cindex_boxplot*.png`
- `fig_gate_heatmap_*.png`
- `gated_patient_subset_signal_probe_patient_subset_predictivity_summary.csv`
- `*_recurrent_gene_supplement_table.csv`

Probe subdirectories (selected-gene signal and recurrence):

- `data/runs/adaptive/kipan/linear_gated_probe_kipan`
- `data/runs/adaptive/pancan/linear_gated_probe_pancan`

**Regenerate figures**:

```bash
cd extras

python analyses/render_adaptive_manuscript_figures.py \
  --dataset kipan \
  --results-dir data/runs/adaptive/kipan \
  --outdir data/processed/kipan_processed

python analyses/render_adaptive_manuscript_figures.py \
  --dataset brca \
  --results-dir data/runs/adaptive/brca \
  --outdir data/processed/tcga_brca20260214_001423 \
  --knn-k 5

python analyses/render_adaptive_manuscript_figures.py \
  --dataset pancan \
  --results-dir data/runs/adaptive/pancan \
  --outdir data/processed/pancan_processed \
  --knn-k 5

python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive/kipan \
  --validation-dir data/runs/validation \
  --out-dir figures/targeted_comparison

python analyses/quick_linear_gated_probe.py \
  --dataset kipan \
  --results-dir data/runs/adaptive/kipan \
  --outdir data/runs/adaptive/kipan/linear_gated_probe_kipan

python analyses/quick_linear_gated_probe.py \
  --dataset pancan \
  --results-dir data/runs/adaptive/pancan \
  --outdir data/runs/adaptive/pancan/linear_gated_probe_pancan

python analyses/plot_pancan_linear_probe_gene_recurrence.py \
  --dataset pancan \
  --probe-dir data/runs/adaptive/pancan/linear_gated_probe_pancan \
  --outdir figures/targeted_comparison
```

**Rerun**:

```bash
cd extras
bash analyses/run_adaptive_gentle_all_kipan_brca_pancan.sh
```

---

## Broad Sweep — Fig. 5

**Canonical runs**:

- Primary: `extras/data/runs/broad/kipan_v2`, `extras/data/runs/broad/brca_v2`
- Supplementary: `extras/data/runs/broad/kipan_gentle`, `extras/data/runs/broad/brca_gentle`

Key outputs: broad summary CSVs, merged sweep summaries, sweep-level comparison CSVs.

**Regenerate figures**:

```bash
cd extras

python analyses/plot_feature_set_transport_size_sweep.py \
  --run-dir data/runs/broad/kipan_v2 \
  --out-prefix figures/broad_sweep/kipan_broad_transport

python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive/kipan \
  --validation-dir data/runs/validation \
  --out-dir figures/broad_sweep
```

**Rerun**:

```bash
cd extras
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```
