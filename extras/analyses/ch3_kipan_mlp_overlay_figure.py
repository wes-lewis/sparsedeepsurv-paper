#!/usr/bin/env python3
"""
KIPAN MLP+L1 reference overlay on broad sweep C-index vs Khard plots.

Loads the broad sweep summary CSV and the MLP reference C-index CSV, then
re-renders the C-index vs Khard LOESS figures for LSPIN and Concrete with an
added horizontal band showing MLP+L1 mean ± 1 SD across seeds.

Usage:
  conda activate musevo
  cd sparsedeepsurv-paper
  python analyses/ch3_kipan_mlp_overlay_figure.py \
    [--broad-dir data/runs/ch3_kipan_broad_knn8] \
    [--mlp-csv data/runs/ch3_kipan_mlp_reference/mlp_reference_cindex.csv] \
    [--out-dir figures/ch3_kipan_mlp_overlay]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


PAPER_ROOT = Path(__file__).resolve().parents[1]
BROAD_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_broad_knn8"
MLP_CSV_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_mlp_reference" / "mlp_reference_cindex.csv"
OUT_DEFAULT = PAPER_ROOT / "figures" / "ch3_kipan_mlp_overlay"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP+L1 overlay on KIPAN broad C-index figures")
    p.add_argument("--broad-dir", type=Path, default=BROAD_DEFAULT)
    p.add_argument("--mlp-csv", type=Path, default=MLP_CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--lowess-frac", type=float, default=0.55)
    p.add_argument("--x-min", type=float, default=50.0)
    p.add_argument("--x-max", type=float, default=1400.0)
    return p.parse_args()


def _kernel_smooth(
    x_log: np.ndarray, y: np.ndarray, frac: float
) -> tuple:
    if len(x_log) == 1:
        grid = np.array([x_log[0], x_log[0]])
        return grid, np.array([y[0], y[0]]), np.array([0.0, 0.0])
    grid = np.linspace(x_log.min(), x_log.max(), max(120, min(220, len(x_log) * 28)))
    x_span = max(x_log.max() - x_log.min(), 1e-6)
    bandwidth = max(x_span * max(frac, 0.18) * 0.38, 0.055)
    means, spreads = [], []
    for gx in grid:
        w = np.exp(-0.5 * ((x_log - gx) / bandwidth) ** 2)
        w_sum = np.clip(w.sum(), 1e-12, None)
        mu = float(np.sum(w * y) / w_sum)
        var = float(np.sum(w * (y - mu) ** 2) / w_sum)
        n_eff = float((w_sum ** 2) / np.clip(np.sum(w ** 2), 1e-12, None))
        spread = np.sqrt(max(var, 0.0)) * (0.65 + 0.35 / np.sqrt(max(n_eff, 1.0)))
        means.append(mu)
        spreads.append(spread)
    means = np.asarray(means, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    mean_sm = lowess(means, grid, frac=max(frac * 0.75, 0.2), return_sorted=False)
    spread_sm = lowess(spreads, grid, frac=max(frac * 0.9, 0.25), return_sorted=False)
    return grid, mean_sm, np.clip(spread_sm, 0.0, None)


def make_overlay_figure(
    df: pd.DataFrame,
    *,
    family: str,
    mlp_mean: float,
    mlp_sd: float,
    x_min: float,
    x_max: float,
    frac: float,
    outpath: Path,
) -> None:
    subdf = df[df["model_family"] == family].copy()
    subdf["mean_Khard"] = pd.to_numeric(subdf["mean_Khard"], errors="coerce")
    subdf["mean_test_c"] = pd.to_numeric(subdf["mean_test_c"], errors="coerce")
    subdf = subdf[np.isfinite(subdf["mean_Khard"]) & np.isfinite(subdf["mean_test_c"])].copy()
    subdf = subdf[(subdf["mean_Khard"] >= x_min) & (subdf["mean_Khard"] <= x_max)].copy()

    fig, ax = plt.subplots(figsize=(9.8, 6.0))

    if not subdf.empty:
        palette = sns.color_palette("tab10", n_colors=max(3, subdf["lambda_sample_smooth"].nunique()))
        smooth_vals = sorted(subdf["lambda_sample_smooth"].unique())
        x_all = subdf["mean_Khard"].to_numpy(float)
        x_lo = max(float(np.nanmin(x_all)) * 0.88, x_min)
        x_hi = min(float(np.nanmax(x_all)) * 1.08, x_max)

        for idx, smooth in enumerate(smooth_vals):
            part = subdf[subdf["lambda_sample_smooth"] == smooth].sort_values("mean_Khard")
            color = palette[idx]
            x = part["mean_Khard"].to_numpy(float)
            y = part["mean_test_c"].to_numpy(float)
            ax.scatter(x, y, s=28, alpha=0.45, color=color, edgecolors="none", zorder=3)
            if len(part) == 1:
                ax.plot(x, y, color=color, linewidth=2.6, label=f"smooth={smooth:g}")
                continue
            x_log = np.log10(x)
            grid, mean_sm, spread_sm = _kernel_smooth(x_log, y, frac)
            x_grid = 10 ** grid
            ax.fill_between(x_grid, mean_sm - spread_sm, mean_sm + spread_sm,
                            color=color, alpha=0.16, linewidth=0, zorder=1)
            ax.plot(x_grid, mean_sm, color=color, linewidth=2.8, label=f"smooth={smooth:g}", zorder=2)

        # ── MLP reference band ────────────────────────────────────────────────
        mlp_color = "#4e79a7"
        ax.axhline(mlp_mean, color=mlp_color, linewidth=2.0, linestyle="--", zorder=4,
                   label=f"MLP + L1 mean ({mlp_mean:.3f})")
        ax.axhspan(mlp_mean - mlp_sd, mlp_mean + mlp_sd,
                   color=mlp_color, alpha=0.12, linewidth=0, zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Mean K-hard (active features)")
    ax.set_ylabel("Mean test C-index")
    ax.set_title(f"{family} C-index vs sparsity — with MLP+L1 reference")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    broad_csv = args.broad_dir / "kipan_broad_tuning_merged_summary.csv"
    if not broad_csv.exists():
        broad_csv = args.broad_dir / "notebook_style_models_accuracy_summary.csv"
    df = pd.read_csv(broad_csv)
    print(f"Loaded broad summary: {len(df)} rows from {broad_csv.name}")

    mlp_df = pd.read_csv(args.mlp_csv)
    mlp_mean = float(mlp_df["cindex"].mean())
    mlp_sd = float(mlp_df["cindex"].std())
    print(f"MLP+L1: mean={mlp_mean:.4f}  sd={mlp_sd:.4f}  n={len(mlp_df)}")

    for family in ["LSPIN", "Concrete"]:
        out = args.out_dir / f"fig_{family.lower()}_cindex_vs_khard_mlp_overlay.pdf"
        make_overlay_figure(
            df,
            family=family,
            mlp_mean=mlp_mean,
            mlp_sd=mlp_sd,
            x_min=args.x_min,
            x_max=args.x_max,
            frac=args.lowess_frac,
            outpath=out,
        )

    # Also save combined summary CSV for reference
    summary = pd.DataFrame([{
        "model": "MLP + L1",
        "mean_cindex": mlp_mean,
        "std_cindex": mlp_sd,
        "n_seeds": len(mlp_df),
    }])
    summary.to_csv(args.out_dir / "mlp_reference_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
