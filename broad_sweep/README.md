# Broad Sweep

This folder covers the broad sparsity/smoothing sweeps and the associated broad
comparison outputs.

## Figure correspondence

The paper figure associated with this workflow is:

- Fig. 5: broad sweep summary

## Canonical preserved runs

```text
data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_kipan_broad_gentle_20260409_161803
data/runs/ch3_brca_broad_gentle_20260409_161803
```

The strongest preserved references are the `v2` KIPAN and BRCA runs. The BRCA
gentle broad run appears less complete at the top level, so it should be
treated as supplementary provenance rather than the only broad reference.

## Primary preserved outputs

Inspection should begin with:

```text
data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336
```

Focus on the postprocessed broad summaries and broad figure outputs there,
especially the merged broad summaries and any sweep-level comparison CSVs.

## Regeneration entry points

### KIPAN transport-size sweep plot

```bash
python analyses/plot_feature_set_transport_size_sweep.py \
  --run-dir data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336 \
  --out-prefix figures/broad_sweep/kipan_broad_transport
```

### KIPAN comparison boxplots reused in broad interpretation

```bash
python analyses/plot_kipan_boxplots.py \
  --adaptive-dir data/runs/adaptive_gentle_all_kipan_brca_pancan_20260408_193020/kipan \
  --validation-dir data/runs/validate_goal1_gentle_all_kipan_brca_20260408_175906 \
  --out-dir figures/broad_sweep
```

## Rerun entry points

Primary launcher:

```bash
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```

Direct drivers:

```bash
python analyses/ch3_kipan_broad.py --help
python analyses/ch3_brca_broad_v2.py --help
python analyses/ch3_broad_gentle.py --help
```

## Primary scripts

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/plot_feature_set_transport_size_sweep.py`
- `analyses/plot_kipan_boxplots.py`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`
