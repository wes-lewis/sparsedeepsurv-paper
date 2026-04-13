#!/usr/bin/env python3
"""
Re-render comparison boxplots from frozen CSV results.

Reads:
    data/selected_comparison_metrics_summary.csv
    data/ch3_broad_sweep_configs.csv

Writes:
    figures/  (boxplot PNGs)

Usage:
    python rerender_boxplots.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sparsedeepsurv.plot import plot_cindex_boxplot


def main(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    targeted_csv = data_dir / "selected_comparison_metrics_summary.csv"
    if not targeted_csv.exists():
        raise FileNotFoundError(f"Expected frozen CSV at {targeted_csv}")

    df = pd.read_csv(targeted_csv)
    fig = plot_cindex_boxplot(df, title="Targeted model comparison")
    fig.savefig(out_dir / "targeted_cindex_boxplot.pdf", bbox_inches="tight")
    print(f"Saved targeted boxplot to {out_dir / 'targeted_cindex_boxplot.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
