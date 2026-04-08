#!/usr/bin/env python3
"""Combine feature-set transport runs across exported feature-set sizes."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KIPAN_RUN_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot transport performance vs exported feature-set size.")
    p.add_argument("--run-dir", type=Path, default=KIPAN_RUN_DEFAULT)
    p.add_argument("--pattern", default="cv_transport_mlp_top*_random5/*_summary.csv")
    p.add_argument("--out-prefix", default="kipan_feature_set_transport_size_sweep")
    return p.parse_args()


def _read_summaries(run_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob(pattern)):
        m = re.search(r"top(\d+)", str(path.parent.name))
        if not m:
            continue
        n_exported = int(m.group(1))
        df = pd.read_csv(path)
        df["n_exported"] = n_exported
        df["source_summary"] = str(path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No summary CSVs matched {run_dir / pattern}")
    return pd.concat(rows, ignore_index=True)


def _plot(df: pd.DataFrame, out_png: Path) -> None:
    df = df.sort_values(["feature_set_label", "feature_set_type", "n_exported"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharex=True)
    metrics = [
        ("cindex_mean", "Mean held-out C-index"),
        ("dynamic_auc_mean", "Mean dynamic AUC"),
        ("integrated_brier_score_mean", "Mean integrated Brier score"),
    ]
    color_map = {
        "Concrete smooth": "#1b9e77",
        "Concrete smooth random matched": "#a6dba0",
        "LSPIN smooth": "#d95f02",
        "LSPIN smooth random matched": "#fdb863",
    }
    style_map = {"gated": "-", "random_matched": "--"}

    for ax, (col, ylabel) in zip(axes, metrics):
        for label, sub in df.groupby("feature_set_label", sort=False):
            sub = sub.sort_values("n_exported")
            ftype = str(sub["feature_set_type"].iloc[0])
            ax.plot(
                sub["n_exported"],
                sub[col],
                marker="o",
                linewidth=2,
                linestyle=style_map.get(ftype, "-"),
                color=color_map.get(label, "0.4"),
                label=label,
            )
        ax.set_xlabel("Exported genes")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["n_exported"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(color="0.9")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("KIPAN MLP transport of gate-derived feature sets vs random matched genes", y=1.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    df = _read_summaries(args.run_dir, args.pattern)
    out_csv = args.run_dir / f"{args.out_prefix}.csv"
    out_png = args.run_dir / f"fig_{args.out_prefix}.png"
    df.to_csv(out_csv, index=False)
    _plot(df, out_png)
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_png}")


if __name__ == "__main__":
    main()
