# Broad Sweep — Fig. 5

**Canonical runs**:

- Primary: `extras/data/runs/broad/kipan_v2`, `extras/data/runs/broad/brca_v2`
- Supplementary: `extras/data/runs/broad/kipan_gentle`, `extras/data/runs/broad/brca_gentle`

Timestamped originals:

- `ch3_kipan_broad_v2_selfcontained_ste_20260407_012336`
- `ch3_brca_broad_v2_selfcontained_ste_20260407_012336`
- `ch3_kipan_broad_gentle_20260409_161803`
- `ch3_brca_broad_gentle_20260409_161803`

## Key outputs

From the `v2` primary runs: broad summary CSVs, merged sweep summaries,
sweep-level comparison CSVs, and broad multiplot figures. The `gentle` runs
are supplementary provenance.

## Regeneration

BRCA broad reruns use the canonical processed input artifact at
`extras/data/processed/tcga_brca20260214_001423`, which can be rebuilt with:

```bash
cd extras
python analyses/prepare_tcga_brca_firehose.py
```

### KIPAN transport-size sweep

```bash
cd extras
python analyses/plot_feature_set_transport_size_sweep.py \
  --run-dir data/runs/broad/kipan_v2 \
  --out-prefix figures/broad_sweep/kipan_broad_transport
```

### KIPAN comparison boxplots

```bash
cd extras
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive/kipan \
  --validation-dir data/runs/validation \
  --out-dir figures/broad_sweep
```

## Rerun

```bash
cd extras
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```

Direct drivers:

```bash
cd extras
python analyses/ch3_kipan_broad.py --help
python analyses/ch3_brca_broad_v2.py --help
python analyses/ch3_broad_gentle.py --help
```

## Scripts

- `extras/analyses/prepare_tcga_brca_firehose.py`
- `extras/analyses/ch3_kipan_broad.py`
- `extras/analyses/ch3_brca_broad_v2.py`
- `extras/analyses/ch3_broad_gentle.py`
- `extras/analyses/plot_feature_set_transport_size_sweep.py`
- `extras/analyses/plot_kipan_boxplots.py`
- `extras/analyses/run_ch3_broad_gentle_kipan_brca.sh`
