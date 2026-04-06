#!/usr/bin/env python3
"""
Plot validation-only consistency summaries to assess whether fixing initialization
improves replicate stability, and by how much.

Uses Goal 1 selected variants from the finished validation run and compares:
  1. screen                : split + init + training randomness varying
  2. fixed_init_pooled     : pooled fixed-init eval runs across split blocks
  3. fixed_init_within_blk : average within-split-block variability where split
                             and init are both fixed and only training randomness varies

Note:
  Validation does not save clustering Rand index / ARI. The third panel uses
  pairwise gate-affinity correlation as the saved proxy for gate-consistency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RUN_DIR = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "validate_models_full_20260403_163907"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot validation-only initialization consistency summaries.")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _norm_family(x: str) -> str:
    x = str(x)
    if x == "LSPIN":
        return "HardSigmoid"
    return x


def _stage_label(x: str) -> str:
    return {
        "screen": "Screen\nvary split+init",
        "fixed_init_pooled": "Fixed-init eval\npooled blocks",
        "fixed_init_within_block": "Fixed-init eval\nwithin block",
    }[x]


def _variant_label(row: pd.Series) -> str:
    fam = _norm_family(row["model_family"])
    if fam == "MLP":
        return "MLP"
    if row["goal1_group"] == "nosmooth":
        return f"{fam}\nNo Smooth"
    if row["goal1_group"] == "smooth":
        return f"{fam}\nSmooth"
    if row["goal1_group"] == "alt_sigma":
        return f"{fam}\nAlt Sigma"
    return fam


def _family_order_key(label: str) -> tuple[int, str]:
    fam_rank = {"HardSigmoid": 0, "Concrete": 1, "MLP": 2}
    fam = label.split("\n", 1)[0]
    return (fam_rank.get(fam, 99), label)


def _ci95_half_width(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return float("nan")
    return 1.96 * float(s.std(ddof=1)) / (len(s) ** 0.5)


def _selected_variants(run_dir: Path) -> pd.DataFrame:
    summ = pd.read_csv(run_dir / "goal1_fixed_init_summary.csv")
    keep = summ[["dataset", "variant_key", "variant_label", "model_family", "goal1_group"]].drop_duplicates()
    keep["model_family"] = keep["model_family"].map(_norm_family)
    keep["plot_label"] = keep.apply(_variant_label, axis=1)
    return keep


def _summarize_consistency(run_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(run_dir / "runs.csv")
    runs["model_family"] = runs["model_family"].map(_norm_family)
    runs = runs[(runs["experiment"] == "goal1") & (runs["status"] == "ok")].copy()
    affinity_pairs = pd.read_csv(run_dir / "affinity_pairs.csv")
    affinity_pairs["model_family"] = affinity_pairs["model_family"].map(_norm_family)
    affinity_pairs = affinity_pairs[affinity_pairs["experiment"] == "goal1"].copy()

    keep = _selected_variants(run_dir)
    runs = runs.merge(
        keep[["dataset", "variant_key", "plot_label"]],
        on=["dataset", "variant_key"],
        how="inner",
    )
    affinity_pairs = affinity_pairs.merge(
        keep[["dataset", "variant_key", "plot_label"]],
        on=["dataset", "variant_key"],
        how="inner",
    )

    seed_map = runs[
        ["dataset", "variant_key", "task_stage", "run_seed", "split_block_id"]
    ].drop_duplicates()
    left = seed_map.rename(columns={"run_seed": "seed_i", "split_block_id": "split_block_i"})
    right = seed_map.rename(columns={"run_seed": "seed_j", "split_block_id": "split_block_j"})
    affinity_pairs = affinity_pairs.merge(
        left[["dataset", "variant_key", "task_stage", "seed_i", "split_block_i"]],
        on=["dataset", "variant_key", "task_stage", "seed_i"],
        how="left",
    )
    affinity_pairs = affinity_pairs.merge(
        right[["dataset", "variant_key", "task_stage", "seed_j", "split_block_j"]],
        on=["dataset", "variant_key", "task_stage", "seed_j"],
        how="left",
    )

    rows = []
    metrics = [
        ("test_cindex", "Test C-index"),
        ("mean_Khard", "Mean Khard"),
    ]

    for (dataset, variant_key, plot_label), g in runs.groupby(["dataset", "variant_key", "plot_label"], dropna=False):
        screen = g[g["task_stage"] == "screen"].copy()
        fixed_eval = g[g["task_stage"] == "fixed_init_eval"].copy()

        for metric_col, metric_label in metrics:
            if not screen.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "variant_key": variant_key,
                        "plot_label": plot_label,
                        "metric": metric_label,
                        "stage_key": "screen",
                        "stage_label": _stage_label("screen"),
                        "n_runs": len(screen),
                        "n_blocks": 0,
                        "mean_value": float(screen[metric_col].mean()),
                        "sd": float(screen[metric_col].std(ddof=1)),
                        "ci95_half_width": _ci95_half_width(screen[metric_col]),
                        "relative_ci95_half_width": _ci95_half_width(screen[metric_col]) / float(screen[metric_col].mean()),
                    }
                )

            if not fixed_eval.empty:
                pooled_ci = _ci95_half_width(fixed_eval[metric_col])
                rows.append(
                    {
                        "dataset": dataset,
                        "variant_key": variant_key,
                        "plot_label": plot_label,
                        "metric": metric_label,
                        "stage_key": "fixed_init_pooled",
                        "stage_label": _stage_label("fixed_init_pooled"),
                        "n_runs": len(fixed_eval),
                        "n_blocks": int(fixed_eval["split_block_id"].nunique()),
                        "mean_value": float(fixed_eval[metric_col].mean()),
                        "sd": float(fixed_eval[metric_col].std(ddof=1)),
                        "ci95_half_width": pooled_ci,
                        "relative_ci95_half_width": pooled_ci / float(fixed_eval[metric_col].mean()),
                    }
                )

                block_rows = []
                for split_block_id, gb in fixed_eval.groupby("split_block_id", dropna=False):
                    if len(gb) < 2:
                        continue
                    block_ci = _ci95_half_width(gb[metric_col])
                    block_rows.append(
                        {
                            "split_block_id": split_block_id,
                            "block_mean": float(gb[metric_col].mean()),
                            "block_ci95_half_width": block_ci,
                            "block_relative_ci95_half_width": block_ci / float(gb[metric_col].mean()),
                        }
                    )

                if block_rows:
                    br = pd.DataFrame(block_rows)
                    rows.append(
                        {
                            "dataset": dataset,
                            "variant_key": variant_key,
                            "plot_label": plot_label,
                            "metric": metric_label,
                            "stage_key": "fixed_init_within_block",
                            "stage_label": _stage_label("fixed_init_within_block"),
                            "n_runs": int(fixed_eval.groupby("split_block_id").size().iloc[0]),
                            "n_blocks": len(br),
                            "mean_value": float(fixed_eval[metric_col].mean()),
                            "sd": float("nan"),
                            "ci95_half_width": float(br["block_ci95_half_width"].mean()),
                            "relative_ci95_half_width": float(br["block_relative_ci95_half_width"].mean()),
                        }
                    )

        aff = affinity_pairs[
            (affinity_pairs["dataset"] == dataset) & (affinity_pairs["variant_key"] == variant_key)
        ].copy()
        if not aff.empty:
            for stage_key, task_stage in [
                ("screen", "screen"),
                ("fixed_init_pooled", "fixed_init_eval"),
            ]:
                stage_aff = aff[aff["task_stage"] == task_stage]
                if not stage_aff.empty:
                    rows.append(
                        {
                            "dataset": dataset,
                            "variant_key": variant_key,
                            "plot_label": plot_label,
                            "metric": "Affinity Corr",
                            "stage_key": stage_key,
                            "stage_label": _stage_label(stage_key),
                            "n_runs": len(stage_aff),
                            "n_blocks": int(stage_aff["split_block_i"].nunique()) if "split_block_i" in stage_aff else 0,
                            "mean_value": float(stage_aff["affinity_corr"].mean()),
                            "sd": float(stage_aff["affinity_corr"].std(ddof=1)) if len(stage_aff) > 1 else float("nan"),
                            "ci95_half_width": float("nan"),
                            "relative_ci95_half_width": float("nan"),
                        }
                    )

            within_block = aff[
                (aff["task_stage"] == "fixed_init_eval")
                & aff["split_block_i"].notna()
                & (aff["split_block_i"] == aff["split_block_j"])
            ].copy()
            if not within_block.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "variant_key": variant_key,
                        "plot_label": plot_label,
                        "metric": "Affinity Corr",
                        "stage_key": "fixed_init_within_block",
                        "stage_label": _stage_label("fixed_init_within_block"),
                        "n_runs": len(within_block),
                        "n_blocks": int(within_block["split_block_i"].nunique()),
                        "mean_value": float(within_block["affinity_corr"].mean()),
                        "sd": float(within_block["affinity_corr"].std(ddof=1)) if len(within_block) > 1 else float("nan"),
                        "ci95_half_width": float("nan"),
                        "relative_ci95_half_width": float("nan"),
                    }
                )

    out = pd.DataFrame(rows)
    stage_order = ["screen", "fixed_init_pooled", "fixed_init_within_block"]
    out["stage_order"] = out["stage_key"].map({k: i for i, k in enumerate(stage_order)})
    return out.sort_values(["dataset", "metric", "plot_label", "stage_order"]).reset_index(drop=True)


def _plot(summary: pd.DataFrame, outpath: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(32, 14), sharex=False)

    datasets = ["kipan", "brca"]
    metrics = ["Test C-index", "Mean Khard", "Affinity Corr"]
    stage_order = [
        _stage_label("screen"),
        _stage_label("fixed_init_pooled"),
        _stage_label("fixed_init_within_block"),
    ]

    palette = {
        _stage_label("screen"): "#9ecae1",
        _stage_label("fixed_init_pooled"): "#74c476",
        _stage_label("fixed_init_within_block"): "#fd8d3c",
    }

    for r, dataset in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sub = summary[(summary["dataset"] == dataset) & (summary["metric"] == metric)].copy()
            order = sorted(sub["plot_label"].unique(), key=_family_order_key)
            y_col = "mean_value" if metric == "Affinity Corr" else "ci95_half_width"
            sns.barplot(
                data=sub,
                x="plot_label",
                y=y_col,
                hue="stage_label",
                order=order,
                hue_order=stage_order,
                palette=palette,
                ax=ax,
            )
            ax.set_title(f"{dataset.upper()} | {metric}")
            ax.set_xlabel("")
            if metric == "Affinity Corr":
                ax.set_ylabel("Mean pairwise affinity corr")
            else:
                ax.set_ylabel("CI half-width")
            ax.tick_params(axis="x", rotation=25)
            if r == 0 and c == 0:
                ax.legend(title="")
            else:
                ax.legend_.remove()

    fig.suptitle(
        "Validation Goal 1 Consistency Under Initialization Controls\n"
        "Screen varies split + initialization; fixed-init eval pools split blocks; "
        "within-block estimates isolate runs where split and initialization are both fixed.\n"
        "Validation does not save clustering Rand index, so the rightmost panels use affinity correlation instead.",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize_consistency(args.run_dir)
    fig_path = out_dir / "fig_validation_init_consistency.png"
    summary_path = out_dir / "validation_init_consistency_summary.csv"

    _plot(summary, fig_path)
    summary.to_csv(summary_path, index=False)

    print(f"Saved {fig_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
