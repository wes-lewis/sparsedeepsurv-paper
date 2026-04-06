#!/usr/bin/env python3
"""
Render supplement boxplots for the finished validation experiment.

Figure 1: Goal 0 matched-sparsity gate comparison
Figure 2: Goal 1 dense MLP vs matched-sparsity sparse models

Each figure uses the fixed-init evaluation outputs that underlie the
summary tables reported in the supplement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "validate_models_full_20260403_163907"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot supplement boxplots from validation outputs.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return p.parse_args()


def _norm_family(x: str) -> str:
    x = str(x)
    if x == "LSPIN":
        return "HardSigmoid"
    return x


def _goal0_label(row: pd.Series) -> str:
    return _norm_family(row["model_family"])


def _goal1_label(row: pd.Series) -> str:
    fam = _norm_family(row["model_family"])
    group = str(row["goal1_group"])
    if fam == "MLP":
        return "MLP"
    if group == "nosmooth":
        return f"{fam}\nNo Smooth"
    if group == "smooth":
        return f"{fam}\nSmooth"
    if group == "alt_sigma":
        return f"{fam}\nAlt Sigma"
    return fam


def _label_order_for_goal0() -> list[str]:
    return ["HardSigmoid", "Concrete"]


def _label_order_for_goal1(dataset: str) -> list[str]:
    order = [
        "MLP",
        "HardSigmoid\nNo Smooth",
        "Concrete\nNo Smooth",
        "HardSigmoid\nSmooth",
        "Concrete\nSmooth",
    ]
    return order


def _prepare_goal0(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "runs.csv")
    aff = pd.read_csv(results_dir / "affinity_pairs.csv")
    summary = pd.read_csv(results_dir / "goal0_fixed_init_summary.csv")

    summary = summary.copy()
    summary["plot_label"] = summary.apply(_goal0_label, axis=1)

    keep = summary[["dataset", "variant_key", "plot_label"]].drop_duplicates()

    run_df = runs.merge(keep, on=["dataset", "variant_key"], how="inner")
    run_df = run_df[(run_df["status"] == "ok") & (run_df["task_stage"] == "fixed_init_eval")].copy()

    aff_df = aff.merge(keep, on=["dataset", "variant_key"], how="inner")
    aff_df = aff_df[aff_df["task_stage"] == "fixed_init_eval"].copy()
    return run_df, aff_df


def _prepare_goal1(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "runs.csv")
    aff = pd.read_csv(results_dir / "affinity_pairs.csv")
    summary = pd.read_csv(results_dir / "goal1_fixed_init_summary.csv")

    summary = summary.copy()
    summary["plot_label"] = summary.apply(_goal1_label, axis=1)

    keep = summary[["dataset", "variant_key", "plot_label"]].drop_duplicates()

    run_df = runs.merge(keep, on=["dataset", "variant_key"], how="inner")
    run_df = run_df[(run_df["status"] == "ok") & (run_df["task_stage"] == "fixed_init_eval")].copy()
    run_df = run_df[run_df["goal1_group"] != "alt_sigma"].copy()
    run_df["khard_union_ratio"] = run_df["mean_Khard"] / run_df["gene_union_count"].replace(0, pd.NA)

    aff_df = aff.merge(keep, on=["dataset", "variant_key"], how="inner")
    aff_df = aff_df[aff_df["task_stage"] == "fixed_init_eval"].copy()
    aff_df = aff_df[aff_df["variant_key"] != "goal1_hardsigmoid_alt_nosmooth"].copy()
    return run_df, aff_df


def _make_figure(
    run_df: pd.DataFrame,
    aff_df: pd.DataFrame,
    *,
    title: str,
    outpath: Path,
    order_fn,
    include_mlp_for_non_gated: bool = False,
    include_khard_union_ratio: bool = False,
    figsize: tuple[float, float] = (22, 12),
    xrotation: float = 25,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ncols = 4 if include_khard_union_ratio else 3
    fig, axes = plt.subplots(2, ncols, figsize=figsize, sharex=False)
    datasets = ["kipan", "brca"]
    nice_dataset = {"kipan": "KIPAN", "brca": "BRCA"}

    for row_idx, dataset in enumerate(datasets):
        sub_runs = run_df[run_df["dataset"] == dataset].copy()
        sub_aff = aff_df[aff_df["dataset"] == dataset].copy()
        order = [x for x in order_fn(dataset) if x in set(sub_runs["plot_label"]) | set(sub_aff["plot_label"])]

        sns.boxplot(
            data=sub_runs,
            x="plot_label",
            y="test_cindex",
            order=order,
            ax=axes[row_idx, 0],
            color="#9ecae1",
            width=0.72,
            fliersize=2,
        )
        axes[row_idx, 0].set_title(f"{nice_dataset[dataset]}: Test C-index")
        axes[row_idx, 0].set_xlabel("")
        axes[row_idx, 0].set_ylabel("C-index")

        khard_runs = sub_runs if include_mlp_for_non_gated else sub_runs[sub_runs["model_family"] != "MLP"].copy()
        khard_order = [x for x in order if x in set(khard_runs["plot_label"])]
        sns.boxplot(
            data=khard_runs,
            x="plot_label",
            y="mean_Khard",
            order=khard_order,
            ax=axes[row_idx, 1],
            color="#bcbddc",
            width=0.72,
            fliersize=2,
        )
        axes[row_idx, 1].set_title(f"{nice_dataset[dataset]}: Mean Khard")
        axes[row_idx, 1].set_xlabel("")
        axes[row_idx, 1].set_ylabel("Khard")

        aff_order = [x for x in order if x in set(sub_aff["plot_label"])]
        sns.boxplot(
            data=sub_aff,
            x="plot_label",
            y="affinity_corr",
            order=aff_order,
            ax=axes[row_idx, 2],
            color="#a1d99b",
            width=0.72,
            fliersize=2,
        )
        axes[row_idx, 2].set_title(f"{nice_dataset[dataset]}: Affinity Corr")
        axes[row_idx, 2].set_xlabel("")
        axes[row_idx, 2].set_ylabel("Affinity corr")

        if include_khard_union_ratio:
            ratio_runs = khard_runs[khard_runs["khard_union_ratio"].notna()].copy()
            ratio_order = [x for x in khard_order if x in set(ratio_runs["plot_label"])]
            sns.boxplot(
                data=ratio_runs,
                x="plot_label",
                y="khard_union_ratio",
                order=ratio_order,
                ax=axes[row_idx, 3],
                color="#fdd0a2",
                width=0.72,
                fliersize=2,
            )
            axes[row_idx, 3].set_title(f"{nice_dataset[dataset]}: Khard / Union")
            axes[row_idx, 3].set_xlabel("")
            axes[row_idx, 3].set_ylabel("Khard / union")

        for col in range(ncols):
            axes[row_idx, col].tick_params(axis="x", rotation=xrotation)
            for label in axes[row_idx, col].get_xticklabels():
                label.set_horizontalalignment("right")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir

    goal0_runs, goal0_aff = _prepare_goal0(results_dir)
    goal1_runs, goal1_aff = _prepare_goal1(results_dir)

    goal0_out = results_dir / "fig_supp_validation_goal0_boxplots.png"
    goal1_out = results_dir / "fig_supp_validation_goal1_boxplots.png"

    _make_figure(
        goal0_runs,
        goal0_aff,
        title="Validation Goal 0: Matched-Sparsity Gate Comparison",
        outpath=goal0_out,
        order_fn=lambda _dataset: _label_order_for_goal0(),
        figsize=(22, 12),
        xrotation=20,
    )
    _make_figure(
        goal1_runs,
        goal1_aff,
        title="Validation Goal 1: Dense MLP vs Matched-Sparsity Sparse Models",
        outpath=goal1_out,
        order_fn=_label_order_for_goal1,
        include_khard_union_ratio=True,
        figsize=(44, 12),
        xrotation=30,
    )

    print(f"Saved {goal0_out}")
    print(f"Saved {goal1_out}")


if __name__ == "__main__":
    main()
