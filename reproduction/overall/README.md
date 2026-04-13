# Overall

This page is the shortest possible map of the visible paper outputs.

## Visible figure outputs in the paper

The main figure files referenced from the paper source are:

- `figures/lspin_fig1.png`
- `figures/fig_validation_cindex_all_models.png`
- `figures/diss_chap3_fig2_2.png`
- `figures/disschap3_fig3_2.png`
- `figures/disschap3_fig4.png`

Supplementary visible figure files referenced from the paper source are:

- `figures/diss_chap3_suppfig1.png`
- `figures/diss_chap3_suppfig_linear_brca_heatmap.png`
- `figures/diss_chap3_suppfig2_2.png`
- `figures/kipan_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`
- `figures/PANCAN_fig_gated_patient_subset_signal_probe_gated_vs_random_signal_boxplots.png`

## Workflow split

These visible outputs are supported by three workflow families:

- validation:
  `figures/fig_validation_cindex_all_models.png`
- targeted comparison:
  `figures/diss_chap3_fig2_2.png`
  `figures/disschap3_fig3_2.png`
  supplementary heatmaps and gated-vs-random figures
- broad sweep:
  `figures/disschap3_fig4.png`

## Where to go next

- For validation outputs and commands: [`../validation/README.md`](../validation/README.md)
- For targeted-comparison outputs and commands: [`../targeted_comparison/README.md`](../targeted_comparison/README.md)
- For broad-sweep outputs and commands: [`../broad_sweep/README.md`](../broad_sweep/README.md)

## Important scope note

This archive is meant to preserve:

- the canonical frozen runs behind the visible outputs
- the scripts that regenerate those output families
- the drivers that launch another run with the same parameterization

It is not meant to fully track manuscript-side filename assembly or every
intermediate CSV emitted during exploratory analysis.
