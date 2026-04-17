#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

FAMILY_ORDER = ["LSPIN", "Concrete"]
SMOOTH_ORDER = ["nosmooth", "smooth"]
SMOOTH_LABELS = {"nosmooth": "No smooth", "smooth": "Patient smooth"}
COLORS = {
    ("LSPIN", "nosmooth"): "#9ec1e6",
    ("LSPIN", "smooth"): "#1f5a99",
    ("Concrete", "nosmooth"): "#a7d99b",
    ("Concrete", "smooth"): "#2f7d32",
}
BRIDGE_MODEL_ORDER = [
    "MLP + L1",
    "MLP + L1 (|w|<cutoff zeroed)",
    "LSPIN (no smooth)",
    "LSPIN + patient smooth",
    "Concrete (no smooth)",
    "Concrete + patient smooth",
]
BRIDGE_MODEL_COLORS = {
    "MLP + L1": "#4e79a7",
    "MLP + L1 (|w|<cutoff zeroed)": "#59a14f",
    "LSPIN (no smooth)": "#e15759",
    "LSPIN + patient smooth": "#f28e2b",
    "Concrete (no smooth)": "#b07aa1",
    "Concrete + patient smooth": "#76b7b2",
}
STABILITY_METRICS = [
    ("affinity_corr", "Affinity reproducibility"),
    ("cluster_ari", "Cluster stability (ARI)"),
    ("risk_corr", "Risk reproducibility"),
    ("khard_over_union", "Khard / Union"),
]
STABILITY_WITH_CINDEX_METRICS = [
    ("affinity_corr", "Affinity reproducibility"),
    ("test_cindex", "Test C-index"),
    ("cluster_ari", "Cluster stability (ARI)"),
    ("khard_over_union", "Khard / Union"),
]
BROAD_METRICS = [
    ("test_cindex", "Test C-index"),
    ("affinity_corr", "Affinity reproducibility"),
    ("risk_corr", "Risk reproducibility"),
    ("cluster_ari", "Cluster stability (ARI)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-render patient-smoothing figures as cleaner boxplots")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--dataset-label", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=["broad", "selected_stability", "selected_stability_cindex", "selected_parsimony", "bridge_cindex_mlp", "bridge_cindex_mlp_single", "selected_gated_plus_mlp_single", "all"],
        default="all",
    )
    p.add_argument("--bins", type=int, default=8)
    return p.parse_args()


def load_ratio_df(results_dir: Path) -> pd.DataFrame:
    acc = pd.read_csv(results_dir / "notebook_style_models_accuracy_summary.csv")
    acc["ratio_x"] = acc["mean_Khard"] / acc["mean_gene_union_count"].clip(lower=1e-8)
    acc["smooth_group"] = np.where(np.isclose(acc["lambda_sample_smooth"], 0.0), "nosmooth", "smooth")
    return acc[["model_family", "lambda_sample_smooth", "lambda_sparse", "ratio_x", "smooth_group"]]


def merge_observations(results_dir: Path, ratio_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "notebook_style_models_runs.csv")
    merged_runs = runs.merge(ratio_df, on=["model_family", "lambda_sample_smooth", "lambda_sparse"], how="left")
    out = {
        "test_cindex": merged_runs[["model_family", "smooth_group", "ratio_x", "test_cindex"]].rename(
            columns={"test_cindex": "value"}
        ),
        "manifold_alignment": merged_runs[
            ["model_family", "smooth_group", "ratio_x", "manifold_alignment"]
        ].rename(columns={"manifold_alignment": "value"}),
    }
    pair_specs = {
        "affinity_corr": "notebook_style_models_affinity_pairs.csv",
        "risk_corr": "notebook_style_models_risk_pairs.csv",
        "cluster_ari": "notebook_style_models_cluster_pairs.csv",
    }
    for metric, fname in pair_specs.items():
        pairs = pd.read_csv(results_dir / fname)
        merged = pairs.merge(ratio_df, on=["model_family", "lambda_sample_smooth", "lambda_sparse"], how="left")
        out[metric] = merged[["model_family", "smooth_group", "ratio_x", metric]].rename(columns={metric: "value"})
    return out


def assign_ratio_bins(obs: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family in FAMILY_ORDER:
        fam = obs[obs["model_family"] == family].copy()
        if fam.empty:
            continue
        unique_x = np.sort(fam["ratio_x"].dropna().unique())
        if len(unique_x) < 2:
            continue
        q = min(bins, max(2, len(unique_x)))
        fam["ratio_bin"] = pd.qcut(fam["ratio_x"], q=q, duplicates="drop")
        centers = fam.groupby("ratio_bin", observed=False)["ratio_x"].mean().sort_values()
        bin_order = list(centers.index)
        center_map = {b: centers[b] for b in bin_order}
        fam["bin_index"] = fam["ratio_bin"].map({b: i for i, b in enumerate(bin_order)})
        fam["bin_center"] = fam["ratio_bin"].map(center_map)
        rows.append(fam)
    return pd.concat(rows, ignore_index=True) if rows else obs.iloc[0:0].copy()


def _ratio_ticklabels(df: pd.DataFrame) -> list[str]:
    labels = []
    centers = df[["bin_index", "bin_center"]].drop_duplicates().sort_values("bin_index")
    for _, row in centers.iterrows():
        labels.append(f"{float(row['bin_center']):.2f}")
    return labels


def _draw_colored_boxplots(ax, data_groups: list[np.ndarray], positions: list[float], colors: list[str], width: float) -> None:
    bp = ax.boxplot(
        data_groups,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)


def save_broad_boxplots(results_dir: Path, dataset_label: str, bins: int) -> None:
    ratio_df = load_ratio_df(results_dir)
    obs = merge_observations(results_dir, ratio_df)
    fig, axes = plt.subplots(2, len(BROAD_METRICS), figsize=(18.0, 8.2), sharex=False)
    for r, family in enumerate(FAMILY_ORDER):
        family_ticks: list[str] | None = None
        for c, (metric, title) in enumerate(BROAD_METRICS):
            ax = axes[r, c]
            fam = obs[metric]
            fam = fam[fam["model_family"] == family].copy()
            fam = assign_ratio_bins(fam, bins)
            if fam.empty:
                ax.set_visible(False)
                continue
            positions = []
            data_groups = []
            colors = []
            bin_centers = fam[["bin_index", "bin_center"]].drop_duplicates().sort_values("bin_index")
            for _, brow in bin_centers.iterrows():
                bin_idx = int(brow["bin_index"])
                for smooth_group, offset in [("nosmooth", -0.18), ("smooth", 0.18)]:
                    sub = fam[(fam["bin_index"] == bin_idx) & (fam["smooth_group"] == smooth_group)]["value"].dropna()
                    if sub.empty:
                        continue
                    positions.append(bin_idx + offset)
                    data_groups.append(sub.to_numpy())
                    colors.append(COLORS[(family, smooth_group)])
            _draw_colored_boxplots(ax, data_groups, positions, colors, width=0.32)
            ax.set_title(title)
            if c == 0:
                ax.text(-0.28, 0.5, family, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=11)
            ax.grid(axis="y", alpha=0.22)
            if metric == "test_cindex":
                ymin, ymax = ax.get_ylim()
                pad = (ymax - ymin) * 0.08
                ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
                ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
            family_ticks = _ratio_ticklabels(fam)
            xticks = list(range(len(family_ticks)))
            ax.set_xticks(xticks)
            tick_labels = [lab if (i % 2 == 0 or len(family_ticks) <= 6) else "" for i, lab in enumerate(family_ticks)]
            ax.set_xticklabels(tick_labels, rotation=0)
            ax.set_xlabel("mean_Khard / union bin center")
        if family_ticks:
            for c in range(len(BROAD_METRICS)):
                axes[r, c].set_xlim(-0.7, len(family_ticks) - 0.3)
    handles = [
        Patch(facecolor=COLORS[("LSPIN", "nosmooth")], edgecolor="#444444", label="LSPIN: no smooth"),
        Patch(facecolor=COLORS[("LSPIN", "smooth")], edgecolor="#444444", label="LSPIN: patient smooth"),
        Patch(facecolor=COLORS[("Concrete", "nosmooth")], edgecolor="#444444", label="Concrete: no smooth"),
        Patch(facecolor=COLORS[("Concrete", "smooth")], edgecolor="#444444", label="Concrete: patient smooth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{dataset_label}: broad-sweep distributions across feature-use efficiency bins", y=1.05, fontsize=14)
    fig.text(
        0.5,
        1.0,
        "Boxplots show seed-level runs for test C-index and seed-pair observations for affinity, risk, and ARI within each efficiency bin.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.03, 0.05, 1, 0.94])
    fig.savefig(results_dir / "fig_observation_binned_ratio_tradeoffs_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_selected_rows(results_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cfg = pd.read_csv(results_dir / "selected_comparison_configs.csv")
    runs = pd.read_csv(results_dir / "notebook_style_models_runs.csv")
    runs["khard_over_union"] = runs["mean_Khard"] / runs["gene_union_count"].clip(lower=1e-8)
    aff = pd.read_csv(results_dir / "notebook_style_models_affinity_pairs.csv")
    risk = pd.read_csv(results_dir / "notebook_style_models_risk_pairs.csv")
    cl = pd.read_csv(results_dir / "notebook_style_models_cluster_pairs.csv")
    return cfg, {
        "test_cindex": runs,
        "manifold_alignment": runs,
        "khard_over_union": runs,
        "mean_Khard": runs,
        "gene_union_count": runs,
        "gene_freq_ge_threshold_count": runs,
        "effective_gene_count": runs,
        "affinity_corr": aff,
        "risk_corr": risk,
        "cluster_ari": cl,
    }


def _extract_selected_metric_df(results_dir: Path, metric: str, *, aggregate_pairs: bool = False) -> pd.DataFrame:
    cfg, sources = _load_selected_rows(results_dir)
    source = sources[metric]
    out_rows = []
    for _, row in cfg.iterrows():
        fam = row["family"]
        sel = row["selection"]
        ls = row["lambda_sparse"]
        lsm = row["lambda_sample_smooth"]
        sub = source[
            (source["model_family"] == fam)
            & np.isclose(source["lambda_sparse"], ls)
            & np.isclose(source["lambda_sample_smooth"], lsm)
        ].copy()
        if sub.empty:
            continue
        if aggregate_pairs and {"run_i", "run_j"}.issubset(sub.columns):
            # Reduce dependent seed-pair rows to one mean similarity score per run.
            left = sub[["run_i", metric]].rename(columns={"run_i": "run_id"})
            right = sub[["run_j", metric]].rename(columns={"run_j": "run_id"})
            sub = (
                pd.concat([left, right], ignore_index=True)
                .groupby("run_id", as_index=False)[metric]
                .mean()
            )
        sub["family"] = fam
        sub["selection"] = sel
        sub["model"] = row["model"]
        sub["value"] = sub[metric]
        out_rows.append(sub[["family", "selection", "model", "value"]])
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["family", "selection", "model", "value"])


def _p_to_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _annotate_significance(ax, metric_df: pd.DataFrame) -> None:
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    extra_top = 0.18 * yr
    ax.set_ylim(y0, y1 + extra_top)
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    top = y1 - 0.06 * yr
    step = 0.10 * yr
    for fam_idx, fam in enumerate(FAMILY_ORDER):
        no = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == "nosmooth")]["value"].dropna().to_numpy()
        sm = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == "smooth")]["value"].dropna().to_numpy()
        if len(no) < 2 or len(sm) < 2:
            continue
        p = float(ttest_ind(no, sm, equal_var=False, nan_policy="omit").pvalue)
        x1, x2 = fam_idx - 0.18, fam_idx + 0.18
        y = top - fam_idx * step
        ax.plot([x1, x1, x2, x2], [y - 0.018 * yr, y, y, y - 0.018 * yr], color="#333333", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2.0, y + 0.010 * yr, _p_to_label(p), ha="center", va="bottom", fontsize=10, clip_on=False)


def _add_selected_boxplot(ax, metric_df: pd.DataFrame, title: str, *, add_significance: bool = False) -> None:
    x = np.arange(len(FAMILY_ORDER))
    positions = []
    data_groups = []
    colors = []
    width = 0.3
    for fam_idx, fam in enumerate(FAMILY_ORDER):
        for sel, offset in [("nosmooth", -0.18), ("smooth", 0.18)]:
            sub = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == sel)]["value"].dropna()
            if sub.empty:
                continue
            positions.append(fam_idx + offset)
            data_groups.append(sub.to_numpy())
            colors.append(COLORS[(fam, sel)])
    _draw_colored_boxplots(ax, data_groups, positions, colors, width)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER)
    ax.grid(axis="y", alpha=0.22)
    if add_significance and not metric_df.empty:
        _annotate_significance(ax, metric_df)


def save_selected_metric_boxplots(results_dir: Path, dataset_label: str, mode: str) -> None:
    if mode == "selected_stability":
        specs = STABILITY_METRICS
        outname = "fig_selected_stability_metrics_boxplot.png"
        subtitle = "Selected matched-config distributions across seeds or seed-pair comparisons"
    elif mode == "selected_stability_cindex":
        specs = STABILITY_WITH_CINDEX_METRICS
        outname = "fig_selected_stability_metrics_with_cindex_boxplot.png"
        subtitle = "Selected matched-config distributions with concordance replacing risk reproducibility"
    else:
        specs = [
            ("mean_Khard", "Selected genes per patient"),
            ("gene_union_count", "Global gene union"),
            ("gene_freq_ge_threshold_count", "Genes used in >=5% patients"),
            ("effective_gene_count", "Effective gene count"),
        ]
        outname = "fig_selected_global_parsimony_metrics_boxplot.png"
        subtitle = "Selected matched-config distributions across seeds"
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    for ax, (metric, title) in zip(axes.ravel(), specs):
        metric_df = _extract_selected_metric_df(
            results_dir,
            metric,
            aggregate_pairs=(mode in {"selected_stability", "selected_stability_cindex"} and metric in {"affinity_corr", "risk_corr", "cluster_ari"}),
        )
        _add_selected_boxplot(ax, metric_df, title, add_significance=True)
    handles = [
        Patch(facecolor=COLORS[("LSPIN", "nosmooth")], edgecolor="#444444", label="LSPIN: no smooth"),
        Patch(facecolor=COLORS[("LSPIN", "smooth")], edgecolor="#444444", label="LSPIN: patient smooth"),
        Patch(facecolor=COLORS[("Concrete", "nosmooth")], edgecolor="#444444", label="Concrete: no smooth"),
        Patch(facecolor=COLORS[("Concrete", "smooth")], edgecolor="#444444", label="Concrete: patient smooth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{dataset_label}: {subtitle}", y=1.05, fontsize=14)
    if mode == "selected_stability":
        fig.text(0.5, 1.0, "Boxplots show 12 seed-level observations for manifold alignment and run-aggregated seed-pair stability summaries for affinity, risk, and ARI; brackets compare smooth vs no-smooth within family.", ha="center", fontsize=10)
    elif mode == "selected_stability_cindex":
        fig.text(0.5, 1.0, "Boxplots show 12 seed-level observations for test C-index and manifold alignment, plus run-aggregated seed-pair stability summaries for affinity and ARI; brackets compare smooth vs no-smooth within family.", ha="center", fontsize=10)
    else:
        fig.text(0.5, 1.0, "Boxplots show 12 seed-level observations per selected config; brackets compare smooth vs no-smooth within family.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(results_dir / outname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bridge_cindex_boxplots_with_mlp(results_dir: Path, dataset_label: str) -> None:
    overall = pd.read_csv(results_dir / "comparison_overall_seed_cindex.csv")
    overall = overall[overall["model"].isin(BRIDGE_MODEL_ORDER)].copy()
    overall["model"] = pd.Categorical(overall["model"], categories=BRIDGE_MODEL_ORDER, ordered=True)
    overall = overall.sort_values("model")

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    data_groups = []
    colors = []
    positions = []
    labels = []
    for idx, model in enumerate(BRIDGE_MODEL_ORDER):
        sub = overall.loc[overall["model"] == model, "cindex"].dropna().to_numpy()
        if sub.size == 0:
            continue
        positions.append(float(idx))
        data_groups.append(sub)
        colors.append(BRIDGE_MODEL_COLORS[model])
        labels.append(model)
    _draw_colored_boxplots(ax, data_groups, positions, colors, width=0.62)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test C-index")
    ax.set_title(f"{dataset_label}: overall test C-index across seeds")
    ax.grid(axis="y", alpha=0.22)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
    ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_bridge_overall_cindex_boxplot_with_mlp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    by_hist = pd.read_csv(results_dir / "comparison_by_histology_seed_cindex.csv")
    by_hist = by_hist[by_hist["model"].isin(BRIDGE_MODEL_ORDER)].copy()
    histologies = sorted(by_hist["histology"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    data_groups = []
    colors = []
    positions = []
    centers = []
    width = 0.12
    offsets = np.linspace(-0.33, 0.33, num=len(BRIDGE_MODEL_ORDER))
    for h_idx, hist in enumerate(histologies):
        centers.append(float(h_idx))
        for m_idx, model in enumerate(BRIDGE_MODEL_ORDER):
            sub = by_hist.loc[(by_hist["histology"] == hist) & (by_hist["model"] == model), "cindex"].dropna().to_numpy()
            if sub.size == 0:
                continue
            positions.append(h_idx + float(offsets[m_idx]))
            data_groups.append(sub)
            colors.append(BRIDGE_MODEL_COLORS[model])
    _draw_colored_boxplots(ax, data_groups, positions, colors, width=width)
    ax.set_xticks(centers)
    ax.set_xticklabels(histologies, rotation=20, ha="right")
    ax.set_ylabel("Test C-index")
    ax.set_title(f"{dataset_label}: histology-specific test C-index across seeds")
    ax.grid(axis="y", alpha=0.22)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
    ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
    handles = [Patch(facecolor=BRIDGE_MODEL_COLORS[m], edgecolor="#444444", label=m) for m in BRIDGE_MODEL_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_bridge_histology_cindex_boxplot_with_mlp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bridge_overall_cindex_singlepanel_with_mlp(results_dir: Path, dataset_label: str) -> None:
    overall = pd.read_csv(results_dir / "comparison_overall_seed_cindex.csv")
    overall = overall[overall["model"].isin(BRIDGE_MODEL_ORDER)].copy()
    overall["model"] = pd.Categorical(overall["model"], categories=BRIDGE_MODEL_ORDER, ordered=True)
    overall = overall.sort_values("model")

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    data_groups = []
    colors = []
    positions = []
    labels = []
    for idx, model in enumerate(BRIDGE_MODEL_ORDER):
        sub = overall.loc[overall["model"] == model, "cindex"].dropna().to_numpy()
        if sub.size == 0:
            continue
        positions.append(float(idx))
        data_groups.append(sub)
        colors.append(BRIDGE_MODEL_COLORS[model])
        labels.append(model.replace(" + ", "\n+ ").replace(" (", "\n("))
    _draw_colored_boxplots(ax, data_groups, positions, colors, width=0.62)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test C-index")
    ax.set_title("Test C-index")
    ax.grid(axis="y", alpha=0.22)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
    ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
    fig.suptitle(f"{dataset_label}: baseline comparison across 12 seeds", y=1.02, fontsize=13)
    fig.text(
        0.5,
        0.98,
        "Separate from the selected smooth/no-smooth configs to avoid implying MLP was tuned to those same selections.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(results_dir / "fig_bridge_overall_cindex_singlepanel_with_mlp.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_selected_gated_plus_mlp_singlepanel(results_dir: Path, dataset_label: str) -> None:
    selected = pd.read_csv(results_dir / "selected_comparison_configs.csv")
    runs = pd.read_csv(results_dir / "notebook_style_models_runs.csv")
    overall = pd.read_csv(results_dir / "comparison_overall_seed_cindex.csv")

    rows = []
    for _, row in selected.iterrows():
        sub = runs[
            (runs["model_family"] == row["family"])
            & np.isclose(runs["lambda_sparse"], row["lambda_sparse"])
            & np.isclose(runs["lambda_sample_smooth"], row["lambda_sample_smooth"])
        ].copy()
        if sub.empty:
            continue
        sub["plot_model"] = row["model"]
        sub["value"] = sub["test_cindex"]
        rows.append(sub[["plot_model", "value"]])

    for model in ["MLP + L1", "MLP + L1 (|w|<cutoff zeroed)"]:
        sub = overall[overall["model"] == model].copy()
        if sub.empty:
            continue
        sub["plot_model"] = model
        sub["value"] = sub["cindex"]
        rows.append(sub[["plot_model", "value"]])

    plot_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["plot_model", "value"])
    order = [m for m in BRIDGE_MODEL_ORDER if m in plot_df["plot_model"].unique()]

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    data_groups = []
    colors = []
    positions = []
    labels = []
    for idx, model in enumerate(order):
        sub = plot_df.loc[plot_df["plot_model"] == model, "value"].dropna().to_numpy()
        if sub.size == 0:
            continue
        positions.append(float(idx))
        data_groups.append(sub)
        colors.append(BRIDGE_MODEL_COLORS[model])
        labels.append(model.replace(" + ", "\n+ ").replace(" (", "\n("))
    _draw_colored_boxplots(ax, data_groups, positions, colors, width=0.62)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test C-index")
    ax.set_title("Test C-index")
    ax.grid(axis="y", alpha=0.22)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
    ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
    fig.suptitle(f"{dataset_label}: selected gated configs plus MLP baseline", y=1.02, fontsize=13)
    fig.text(
        0.5,
        0.98,
        "LSPIN/Concrete use the same selected configs as the stability-with-C-index panel; MLP remains the run-level baseline.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(results_dir / "fig_selected_gated_plus_mlp_cindex_singlepanel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    dataset_label = args.dataset_label or results_dir.name
    if args.mode in {"broad", "all"}:
        save_broad_boxplots(results_dir, dataset_label, args.bins)
    if args.mode in {"selected_stability", "all"}:
        save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability")
    if args.mode in {"selected_stability_cindex", "all"}:
        save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability_cindex")
    if args.mode in {"selected_parsimony", "all"}:
        save_selected_metric_boxplots(results_dir, dataset_label, "selected_parsimony")
    if args.mode in {"bridge_cindex_mlp", "all"}:
        save_bridge_cindex_boxplots_with_mlp(results_dir, dataset_label)
    if args.mode in {"bridge_cindex_mlp_single", "all"}:
        save_bridge_overall_cindex_singlepanel_with_mlp(results_dir, dataset_label)
    if args.mode in {"selected_gated_plus_mlp_single", "all"}:
        save_selected_gated_plus_mlp_singlepanel(results_dir, dataset_label)


if __name__ == "__main__":
    main()
