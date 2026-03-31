#!/usr/bin/env python3
"""
Chapter 3 BRCA broad-balanced sweep.

Reproduces outputs from:
    /banach2/wes/lspin-pytorch/runs/ch3_rerun_brca_20260326_broad_balanced

Usage:
    python ch3_brca_broad.py --config configs/brca_broad.yaml --out tables/ figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sparsedeepsurv  # noqa: F401


def main(config_path: Path, out_tables: Path, out_figures: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Port from lspin-pytorch/run_brca_patient_smoothing_broad_balanced.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-tables", type=Path, default=Path("tables"))
    parser.add_argument("--out-figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.config, args.out_tables, args.out_figures)
