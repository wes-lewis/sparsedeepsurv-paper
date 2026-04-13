# Broad Sweep

This folder covers the broad sparsity/smoothing sweeps and the visible broad
comparison figure.

## Visible output

The visible broad-sweep figure in the paper is:

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

## If you only want to inspect the preserved results

Start with:

```text
data/runs/ch3_kipan_broad_v2_selfcontained_ste_20260407_012336
data/runs/ch3_brca_broad_v2_selfcontained_ste_20260407_012336
```

Focus on the postprocessed broad summaries and broad figure outputs there.

## If you want to rerender the visible broad-sweep outputs

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

## If you want to launch a fresh broad sweep with the same workflow defaults

Use the launcher:

```bash
bash analyses/run_ch3_broad_gentle_kipan_brca.sh
```

Or inspect the direct drivers:

```bash
python analyses/ch3_kipan_broad.py --help
python analyses/ch3_brca_broad_v2.py --help
python analyses/ch3_broad_gentle.py --help
```

## Driver and plotting scripts used here

- `analyses/ch3_kipan_broad.py`
- `analyses/ch3_brca_broad_v2.py`
- `analyses/ch3_broad_gentle.py`
- `analyses/plot_feature_set_transport_size_sweep.py`
- `analyses/plot_kipan_boxplots.py`
- `analyses/run_ch3_broad_gentle_kipan_brca.sh`
