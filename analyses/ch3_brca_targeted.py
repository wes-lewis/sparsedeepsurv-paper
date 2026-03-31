#!/usr/bin/env python3
"""
Chapter 3 BRCA targeted analysis.

Reproduces outputs from:
    /banach2/wes/lspin-pytorch/runs/ch3_rerun_brca_20260325_notebook_style_paper

Usage:
    python ch3_brca_targeted.py --config configs/brca_targeted.yaml --out tables/ figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

# All model/training/eval logic comes from the package.
# TODO: import and wire up sparsedeepsurv calls here during migration.
import sparsedeepsurv  # noqa: F401


def main(config_path: Path, out_tables: Path, out_figures: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Port from lspin-pytorch/run_brca_patient_smoothing_notebook_style_paper.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-tables", type=Path, default=Path("tables"))
    parser.add_argument("--out-figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.config, args.out_tables, args.out_figures)
