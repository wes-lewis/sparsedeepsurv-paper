# Overall

This page provides a compact map of the paper result families represented in
this archive.

## Figure correspondence

Main figures:

- Fig. 1: framework overview
- Fig. 2: all-model validation comparison
- Fig. 3: KIPAN targeted-comparison heatmaps
- Fig. 4: targeted-comparison quantitative summary
- Fig. 5: broad sweep summary

Supplementary figures:

- Fig. S1: BRCA nonlinear heatmaps
- Fig. S2: BRCA linear heatmaps
- Fig. S3: PANCAN heatmaps
- Fig. S4: KIPAN gated-vs-random selected-gene signal
- Fig. S5: PANCAN gated-vs-random selected-gene signal

## Workflow correspondence

These figure outputs are supported by three workflow families:

- validation:
  Fig. 2
- targeted comparison:
  Fig. 3
  Fig. 4
  supplementary heatmaps:
  Fig. S1
  Fig. S2
  Fig. S3
  nested selected-gene signal results:
  Fig. S4
  Fig. S5
- broad sweep:
  Fig. 5

Related supplementary data and tabular outputs are organized with the same
workflow split:

- validation:
  validation summaries and initialization-consistency CSVs
- targeted comparison:
  selected configuration CSVs, heatmap summaries, and selected-gene signal or
  recurrence CSVs
- broad sweep:
  broad summary CSVs and sweep-level comparison summaries

## Where to go next

- Validation outputs and commands: [`../validation/README.md`](../validation/README.md)
- Targeted-comparison outputs and commands: [`../targeted_comparison/README.md`](../targeted_comparison/README.md)
- Broad-sweep outputs and commands: [`../broad_sweep/README.md`](../broad_sweep/README.md)

## Important scope note

This archive is meant to preserve:

- the canonical frozen runs behind the paper outputs
- the scripts that regenerate those output families
- the drivers that launch another run with the same parameterization

It is not meant to fully track manuscript-side filename assembly or every
intermediate CSV emitted during exploratory analysis.
