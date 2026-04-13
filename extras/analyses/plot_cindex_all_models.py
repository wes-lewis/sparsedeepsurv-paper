#!/usr/bin/env python3
"""
Plot C-index point estimates with 95% CI error bars for all model families,
for KIPAN and BRCA, in a forest-plot style (horizontal dot plot).

Uses:
  - goal1_fixed_init_summary.csv for all selected goal-1 model families except RSF
  - screen_summary.csv for RSF only

Output: fig_cindex_all_models_pointest.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS

DEFAULT_RUN_DIR = CANONICAL_RUNS["validate_goal1_gentle_all_kipan_brca"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot C-index point estimates for all models.")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _norm_family(x: str) -> str:
    if x in {"HardSigmoid", "LSPIN"}:
        return "LSPIN"
    return x


# Display label, family (for color), sort order
_MODEL_META: list[tuple[str, str, str, int]] = [
    # (variant_key_match, display_label, color_group, sort_order)
]

# Order of rows in the plot (bottom to top in forest plot = first to last in list = top to bottom)
_PLOT_ORDER = [
    "LSPIN\nNo Smooth",
    "LSPIN\nSmooth",
    "Concrete\nNo Smooth",
    "Concrete\nSmooth",
    "L-LSPIN\nSmooth",
    "L-Concrete\nSmooth",
    "MLP+STG",
    "MLP",
    "RSF",
    "Linear Cox",
]

_COLORS = {
    "LSPIN\nNo Smooth":   "#1f77b4",   # blue
    "LSPIN\nSmooth":      "#17becf",   # cyan-blue
    "Concrete\nNo Smooth":"#e07b00",   # dark orange
    "Concrete\nSmooth":   "#ff7f0e",   # orange
    "L-LSPIN\nSmooth":    "#aec7e8",   # light blue
    "L-Concrete\nSmooth": "#ffbb78",   # light orange
    "MLP+STG":            "#9467bd",   # purple
    "MLP":                "#2ca02c",   # green
    "RSF":                "#8c564b",   # brown
    "Linear Cox":         "#7f7f7f",   # gray
}

# kept for legend grouping only
_FAMILY_MAP = {
    "LSPIN\nNo Smooth":   "LSPIN\nNo Smooth",
    "LSPIN\nSmooth":      "LSPIN\nSmooth",
    "Concrete\nNo Smooth":"Concrete\nNo Smooth",
    "Concrete\nSmooth":   "Concrete\nSmooth",
    "L-LSPIN\nSmooth":    "L-LSPIN\nSmooth",
    "L-Concrete\nSmooth": "L-Concrete\nSmooth",
    "MLP+STG":            "MLP+STG",
    "MLP":                "MLP",
    "RSF":                "RSF",
    "Linear Cox":         "Linear Cox",
}


def _load_data(run_dir: Path) -> pd.DataFrame:
    fixed = pd.read_csv(run_dir / "goal1_fixed_init_summary.csv")
    screen = pd.read_csv(run_dir / "screen_summary.csv")

    fixed["model_family"] = fixed["model_family"].map(_norm_family)
    screen["model_family"] = screen["model_family"].map(_norm_family)

    rows = []

    # From fixed-init summary: all selected model families except RSF.
    for _, r in fixed.iterrows():
        fam = r["model_family"]
        grp = str(r.get("goal1_group", ""))
        if fam == "MLP":
            label = "MLP"
        elif fam == "LinearCox":
            label = "Linear Cox"
        elif fam == "MLP+STG":
            label = "MLP+STG"
        elif grp == "nosmooth" or grp == "alt_sigma":
            label = f"{fam}\nNo Smooth"
        elif grp == "smooth":
            label = f"{fam}\nSmooth"
        elif grp == "linear_gated":
            if fam == "L-LSPIN":
                label = "L-LSPIN\nSmooth"
            elif fam == "L-Concrete":
                label = "L-Concrete\nSmooth"
            else:
                continue
        else:
            continue

        # Skip alt_sigma duplicate (same performance as nosmooth; deduplicate by keeping first)
        if grp == "alt_sigma":
            continue

        rows.append({
            "dataset": r["dataset"],
            "plot_label": label,
            "mean_test_cindex": r["mean_test_cindex"],
            "ci95_test_cindex": r["ci95_test_cindex"],
            "n_runs": r["n_runs"],
            "source": "fixed_init",
        })

    # From screen summary: RSF only.
    screen_g1 = screen[screen["experiment"] == "goal1"].copy()

    # RSF: single variant, no sweep
    for _, r in screen_g1[screen_g1["model_family"] == "RSF"].iterrows():
        rows.append({
            "dataset": r["dataset"],
            "plot_label": "RSF",
            "mean_test_cindex": r["mean_test_cindex"],
            "ci95_test_cindex": r["ci95_test_cindex"],
            "n_runs": r["n_runs"],
            "source": "screen",
        })

    df = pd.DataFrame(rows)
    # Deduplicate: if same dataset+label appears multiple times (shouldn't happen), keep highest n_runs
    df = df.sort_values("n_runs", ascending=False).drop_duplicates(subset=["dataset", "plot_label"])
    return df


def _plot(df: pd.DataFrame, outpath: Path) -> None:
    datasets = ["kipan", "brca"]
    titles = {"kipan": "KIPAN", "brca": "BRCA"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    y_positions = {label: i for i, label in enumerate(reversed(_PLOT_ORDER))}

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset].copy()
        sub = sub[sub["plot_label"].isin(_PLOT_ORDER)]

        for _, row in sub.iterrows():
            label = row["plot_label"]
            y = y_positions[label]
            x = row["mean_test_cindex"]
            xerr = row["ci95_test_cindex"]
            fam = _FAMILY_MAP.get(label, "MLP")
            color = _COLORS.get(fam, "#333333")
            marker = "D" if fam in {"MLP", "RSF", "Linear Cox", "MLP+STG"} else "o"

            ax.errorbar(
                x, y,
                xerr=xerr,
                fmt=marker,
                color=color,
                markersize=8,
                capsize=4,
                capthick=1.5,
                linewidth=1.5,
                ecolor=color,
            )

        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(reversed(_PLOT_ORDER)), fontsize=10)
        ax.set_xlabel("Test C-index", fontsize=11)
        ax.set_title(titles[dataset], fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.axvline(0.5, color="#d62728", linestyle="-", linewidth=1.2, alpha=0.5, zorder=0)

        # Add vertical reference lines for MLP and RSF if present
        for ref_label, ls in [("MLP", "--"), ("RSF", ":")]:
            ref_row = sub[sub["plot_label"] == ref_label]
            if not ref_row.empty:
                xref = float(ref_row["mean_test_cindex"].iloc[0])
                ax.axvline(xref, color=_COLORS.get(ref_label, "#333"), linestyle=ls, linewidth=1, alpha=0.4)

    # Legend: family colors
    legend_entries = [
        mlines.Line2D([], [], color="#d62728", linestyle="-", linewidth=1.5, label="Random (C=0.5)"),
        mlines.Line2D([], [], color=_COLORS["LSPIN\nNo Smooth"], marker="o", linestyle="None",
                      markersize=8, label="LSPIN no smooth (patient-specific)"),
        mlines.Line2D([], [], color=_COLORS["LSPIN\nSmooth"], marker="o", linestyle="None",
                      markersize=8, label="LSPIN smooth (patient-specific)"),
        mlines.Line2D([], [], color=_COLORS["Concrete\nNo Smooth"], marker="o", linestyle="None",
                      markersize=8, label="Concrete no smooth (patient-specific)"),
        mlines.Line2D([], [], color=_COLORS["Concrete\nSmooth"], marker="o", linestyle="None",
                      markersize=8, label="Concrete smooth (patient-specific)"),
        mlines.Line2D([], [], color=_COLORS["L-LSPIN\nSmooth"], marker="o", linestyle="None",
                      markersize=8, label="L-LSPIN smooth (linear-gated)"),
        mlines.Line2D([], [], color=_COLORS["L-Concrete\nSmooth"], marker="o", linestyle="None",
                      markersize=8, label="L-Concrete smooth (linear-gated)"),
        mlines.Line2D([], [], color=_COLORS["MLP+STG"], marker="D", linestyle="None",
                      markersize=8, label="MLP+STG (global gates)"),
        mlines.Line2D([], [], color=_COLORS["MLP"], marker="D", linestyle="None",
                      markersize=8, label="MLP (ungated)"),
        mlines.Line2D([], [], color=_COLORS["RSF"], marker="D", linestyle="None",
                      markersize=8, label="RSF (nonparametric)"),
        mlines.Line2D([], [], color=_COLORS["Linear Cox"], marker="D", linestyle="None",
                      markersize=8, label="Linear Cox (ungated)"),
    ]
    fig.legend(
        handles=legend_entries,
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.12),
    )

    fig.suptitle(
        "C-index: All Model Families — KIPAN and BRCA\n"
        "Point estimate ± 95% CI (fixed-init eval, n=12 runs; screen-only models n=5)",
        fontsize=12,
        y=1.01,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(args.run_dir)
    print(df[["dataset", "plot_label", "mean_test_cindex", "ci95_test_cindex", "n_runs"]].sort_values(
        ["dataset", "plot_label"]).to_string(index=False))

    outpath = out_dir / "fig_cindex_all_models_pointest.png"
    _plot(df, outpath)


if __name__ == "__main__":
    main()
