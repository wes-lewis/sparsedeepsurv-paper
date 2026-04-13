#!/usr/bin/env python3
"""
Render two KIPAN summary figures from the finished adaptive and validation runs.

Figure 1:
  Phase-1 vs phase-2 variance control for selected adaptive showcase configs.
  Metrics: test C-index, Khard/union ratio, pairwise cluster ARI.

Figure 2:
  Selected KIPAN model comparison.
  Metrics: test C-index, Khard/union ratio, pairwise cluster ARI.
  MLP is shown on the C-index panel only, sourced from the validation run.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS

DEFAULT_ADAPTIVE_DIR = CANONICAL_RUNS["kipan_adaptive_gentle"]
DEFAULT_VALIDATION_DIR = CANONICAL_RUNS["validate_goal1_gentle_all_kipan_brca"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render KIPAN boxplot figures from finished runs.")
    p.add_argument("--adaptive-dir", type=Path, default=DEFAULT_ADAPTIVE_DIR)
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--min-showcase-runs",
        type=int,
        default=12,
        help="Only include adaptive showcase configs with at least this many reps.",
    )
    return p.parse_args()


def _norm_family(x: str) -> str:
    x = str(x)
    if x in {"HardSigmoid", "LSPIN"}:
        return "LSPIN"
    return x


def _selection_label(selection: str) -> str:
    return "No Smooth" if str(selection) == "nosmooth" else "Smooth"


def _showcase_label(row: pd.Series) -> str:
    fam = _norm_family(row["family"])
    return f"{fam}\n" + f"s={row['gate_sigma']:.2f}\n" + _selection_label(row["selection"])


def _family_order_key(label: str) -> tuple[int, str]:
    fam_rank = {"LSPIN": 0, "Concrete": 1, "MLP": 2}
    fam = label.split("\n", 1)[0]
    return (fam_rank.get(fam, 99), label)


def _phase_name(phase: int) -> str:
    return "Phase 1" if int(phase) == 1 else "Phase 2 Warm-Start"


def _read_partial_metric_rows(adaptive_dir: Path, filename: str) -> pd.DataFrame:
    parts = []
    for phase, pat in [(1, "_partial_p1_worker*/" + filename), (2, "_partial_p2_worker*/" + filename)]:
        for p in sorted(adaptive_dir.glob(pat)):
            df = pd.read_csv(p)
            df["phase"] = phase
            parts.append(df)
    if not parts:
        raise FileNotFoundError(f"Could not find any partial files matching {filename} under {adaptive_dir}")
    out = pd.concat(parts, ignore_index=True)
    out["model_family"] = out["model_family"].map(_norm_family)
    return out


def _load_showcase(adaptive_dir: Path, min_showcase_runs: int) -> pd.DataFrame:
    df = pd.read_csv(adaptive_dir / "selected_showcase_configs.csv")
    df["family"] = df["family"].map(_norm_family)
    df["model_family"] = df["model_family"].map(_norm_family)
    df = df[df["n_runs"] >= int(min_showcase_runs)].copy()
    df["showcase_label"] = df.apply(_showcase_label, axis=1)
    df = df.sort_values("showcase_label", key=lambda s: s.map(_family_order_key)).reset_index(drop=True)
    return df


def _merge_showcase_runs(adaptive_dir: Path, showcase: pd.DataFrame) -> pd.DataFrame:
    runs = pd.read_csv(adaptive_dir / "all_runs_raw.csv")
    runs["model_family"] = runs["model_family"].map(_norm_family)
    keep = showcase[
        ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth", "selection", "showcase_label"]
    ].drop_duplicates()
    out = runs.merge(
        keep,
        on=["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"],
        how="inner",
    )
    out["phase_label"] = out["phase"].map(_phase_name)
    out["khard_over_union"] = out["mean_Khard"] / out["gene_union_count"]
    return out


def _merge_showcase_cluster_pairs(adaptive_dir: Path, showcase: pd.DataFrame) -> pd.DataFrame:
    pairs = _read_partial_metric_rows(adaptive_dir, "cluster_rows.csv")
    keep = showcase[
        ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth", "selection", "showcase_label"]
    ].drop_duplicates()
    out = pairs.merge(
        keep,
        on=["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"],
        how="inner",
    )
    out["phase_label"] = out["phase"].map(_phase_name)
    return out


def _load_validation_kipan_runs(validation_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(validation_dir / "runs.csv")
    runs["model_family"] = runs["model_family"].map(_norm_family)
    sub = runs[
        (runs["dataset"] == "kipan")
        & (runs["experiment"] == "goal1")
        & (runs["task_stage"] == "fixed_init_eval")
        & (runs["status"] == "ok")
    ].copy()
    sub["khard_over_union"] = sub["mean_Khard"] / sub["gene_union_count"]
    return sub


def _validation_label(row: pd.Series) -> str:
    fam = _norm_family(row["model_family"])
    if fam == "MLP":
        return "MLP"
    if row["goal1_group"] == "alt_sigma":
        return f"{fam}\nAlt Sigma"
    return f"{fam}\n" + _selection_label(row["goal1_group"])


def _make_variance_figure(
    showcase_runs: pd.DataFrame,
    showcase_cluster_pairs: pd.DataFrame,
    outpath: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    phase_order = ["Phase 1", "Phase 2 Warm-Start"]
    cfg_order = sorted(showcase_runs["showcase_label"].unique(), key=_family_order_key)

    sns.boxplot(
        data=showcase_runs,
        x="showcase_label",
        y="test_cindex",
        hue="phase_label",
        order=cfg_order,
        hue_order=phase_order,
        ax=axes[0],
        width=0.7,
        fliersize=2,
    )
    axes[0].set_title("Test C-index")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("C-index")

    sns.boxplot(
        data=showcase_runs,
        x="showcase_label",
        y="khard_over_union",
        hue="phase_label",
        order=cfg_order,
        hue_order=phase_order,
        ax=axes[1],
        width=0.7,
        fliersize=2,
    )
    axes[1].set_title("Khard / Union")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Khard / gene union")

    sns.boxplot(
        data=showcase_cluster_pairs,
        x="showcase_label",
        y="cluster_ari",
        hue="phase_label",
        order=cfg_order,
        hue_order=phase_order,
        ax=axes[2],
        width=0.7,
        fliersize=2,
    )
    axes[2].set_title("Pairwise Cluster ARI")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("ARI")

    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "KIPAN Adaptive v2 Variance Control\n"
        "Phase 1: fixed split, varying init + training randomness | "
        "Phase 2: same split, warm-start anchor init, varying training randomness",
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_comparison_figure(
    showcase_runs: pd.DataFrame,
    showcase_cluster_pairs: pd.DataFrame,
    validation_runs: pd.DataFrame,
    outpath: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    gated_final = showcase_runs[showcase_runs["phase"] == 2].copy()
    gated_final["comparison_label"] = gated_final["showcase_label"]

    mlp = validation_runs[validation_runs["variant_key"] == "mlp"].copy()
    mlp["comparison_label"] = "MLP"

    cindex_df = pd.concat(
        [
            mlp[["comparison_label", "test_cindex"]],
            gated_final[["comparison_label", "test_cindex"]],
        ],
        ignore_index=True,
    )

    ratio_df = gated_final[["comparison_label", "khard_over_union"]].copy()
    gated_labels = sorted(gated_final["comparison_label"].unique(), key=_family_order_key)

    keep = gated_final[
        ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth", "comparison_label"]
    ].drop_duplicates()
    ari_df = showcase_cluster_pairs[showcase_cluster_pairs["phase"] == 2].merge(
        keep,
        on=["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"],
        how="inner",
    )

    cindex_order = ["MLP"] + gated_labels

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    sns.boxplot(
        data=cindex_df,
        x="comparison_label",
        y="test_cindex",
        order=cindex_order,
        ax=axes[0],
        width=0.7,
        fliersize=2,
        color="#9ecae1",
    )
    axes[0].set_title("Test C-index")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("C-index")

    sns.boxplot(
        data=ratio_df,
        x="comparison_label",
        y="khard_over_union",
        order=gated_labels,
        ax=axes[1],
        width=0.7,
        fliersize=2,
        color="#bcbddc",
    )
    axes[1].set_title("Khard / Union")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Khard / gene union")

    sns.boxplot(
        data=ari_df,
        x="comparison_label",
        y="cluster_ari",
        order=gated_labels,
        ax=axes[2],
        width=0.7,
        fliersize=2,
        color="#a1d99b",
    )
    axes[2].set_title("Pairwise Cluster ARI")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("ARI")

    for ax in axes:
        ax.tick_params(axis="x", rotation=28)

    fig.suptitle(
        "KIPAN Selected Model Comparison\n"
        "Gated boxes from adaptive v2 phase-2 showcase runs; MLP C-index from fixed-init validation",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ci_half_width(values: Iterable[float]) -> float:
    s = pd.Series(list(values)).dropna()
    if len(s) < 2:
        return float("nan")
    return 1.96 * float(s.std(ddof=1)) / (len(s) ** 0.5)


def _write_summary_tables(
    showcase_runs: pd.DataFrame,
    showcase_cluster_pairs: pd.DataFrame,
    validation_runs: pd.DataFrame,
    out_dir: Path,
) -> None:
    var_rows = []
    for (label, phase_label), g in showcase_runs.groupby(["showcase_label", "phase_label"], dropna=False):
        cluster_sub = showcase_cluster_pairs[
            (showcase_cluster_pairs["showcase_label"] == label)
            & (showcase_cluster_pairs["phase_label"] == phase_label)
        ]
        var_rows.append(
            {
                "label": label,
                "phase_label": phase_label,
                "n_runs": len(g),
                "mean_test_cindex": float(g["test_cindex"].mean()),
                "sd_test_cindex": float(g["test_cindex"].std(ddof=1)),
                "ci95_test_cindex": _ci_half_width(g["test_cindex"]),
                "mean_khard_over_union": float(g["khard_over_union"].mean()),
                "sd_khard_over_union": float(g["khard_over_union"].std(ddof=1)),
                "ci95_khard_over_union": _ci_half_width(g["khard_over_union"]),
                "n_cluster_pairs": len(cluster_sub),
                "mean_cluster_ari": float(cluster_sub["cluster_ari"].mean()) if len(cluster_sub) else float("nan"),
                "sd_cluster_ari": float(cluster_sub["cluster_ari"].std(ddof=1)) if len(cluster_sub) > 1 else float("nan"),
                "ci95_cluster_ari": _ci_half_width(cluster_sub["cluster_ari"]) if len(cluster_sub) else float("nan"),
            }
        )

    comp_rows = []
    mlp = validation_runs[validation_runs["variant_key"] == "mlp"].copy()
    comp_rows.append(
        {
            "label": "MLP",
            "source": "validation_fixed_init",
            "n_runs": len(mlp),
            "mean_test_cindex": float(mlp["test_cindex"].mean()),
            "sd_test_cindex": float(mlp["test_cindex"].std(ddof=1)),
            "ci95_test_cindex": _ci_half_width(mlp["test_cindex"]),
            "mean_khard_over_union": float("nan"),
            "ci95_khard_over_union": float("nan"),
            "mean_cluster_ari": float("nan"),
            "ci95_cluster_ari": float("nan"),
        }
    )
    gated_final = showcase_runs[showcase_runs["phase"] == 2].copy()
    for label, g in gated_final.groupby("showcase_label", dropna=False):
        cluster_sub = showcase_cluster_pairs[
            (showcase_cluster_pairs["showcase_label"] == label)
            & (showcase_cluster_pairs["phase"] == 2)
        ]
        comp_rows.append(
            {
                "label": label,
                "source": "adaptive_phase2",
                "n_runs": len(g),
                "mean_test_cindex": float(g["test_cindex"].mean()),
                "sd_test_cindex": float(g["test_cindex"].std(ddof=1)),
                "ci95_test_cindex": _ci_half_width(g["test_cindex"]),
                "mean_khard_over_union": float(g["khard_over_union"].mean()),
                "ci95_khard_over_union": _ci_half_width(g["khard_over_union"]),
                "mean_cluster_ari": float(cluster_sub["cluster_ari"].mean()) if len(cluster_sub) else float("nan"),
                "ci95_cluster_ari": _ci_half_width(cluster_sub["cluster_ari"]) if len(cluster_sub) else float("nan"),
            }
        )

    pd.DataFrame(var_rows).sort_values(["label", "phase_label"]).to_csv(
        out_dir / "kipan_variance_boxplot_summary.csv", index=False
    )
    pd.DataFrame(comp_rows).sort_values("label", key=lambda s: s.map(_family_order_key)).to_csv(
        out_dir / "kipan_comparison_boxplot_summary.csv", index=False
    )


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir or args.adaptive_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    showcase = _load_showcase(args.adaptive_dir, args.min_showcase_runs)
    showcase_runs = _merge_showcase_runs(args.adaptive_dir, showcase)
    showcase_cluster_pairs = _merge_showcase_cluster_pairs(args.adaptive_dir, showcase)
    validation_runs = _load_validation_kipan_runs(args.validation_dir)

    variance_path = out_dir / "fig_kipan_variance_controls_boxplots.png"
    comparison_path = out_dir / "fig_kipan_selected_comparison_boxplots.png"
    _make_variance_figure(showcase_runs, showcase_cluster_pairs, variance_path)
    _make_comparison_figure(showcase_runs, showcase_cluster_pairs, validation_runs, comparison_path)
    _write_summary_tables(showcase_runs, showcase_cluster_pairs, validation_runs, out_dir)

    print(f"Saved {variance_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {out_dir / 'kipan_variance_boxplot_summary.csv'}")
    print(f"Saved {out_dir / 'kipan_comparison_boxplot_summary.csv'}")


if __name__ == "__main__":
    main()
