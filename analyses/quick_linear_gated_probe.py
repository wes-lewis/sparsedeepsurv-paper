#!/usr/bin/env python3
"""Quick linear-risk gated-model probe for gate interpretability.

This is intentionally not a full adaptive sweep. It reuses the selected
adaptive hyperparameters, swaps the survival head to a linear predictor, trains
a small number of folds/replicates, and asks whether a gated feature is more
univariately predictive inside the patients where it is gated on than random
features evaluated on those same patients.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SDS_SRC = Path("/banach2/wes/lspin-repos/sparsedeepsurv/src")
ANALYSES_DIR = Path(__file__).resolve().parent
for p in [SDS_SRC, ANALYSES_DIR]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import KFold, ShuffleSplit

import sparsedeepsurv as sds
from plot_kipan_gated_univariate_dotplot import _cox_score_test_matrix


RUN_DEFAULTS = {
    "pancan": Path(
        "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
        "ch3_pancan_adaptive_v2_selfcontained_ste_randominit_20260406_141705"
    ),
    "kipan": Path(
        "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
        "ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219"
    ),
    "brca": Path(
        "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
        "ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115"
    ),
}
DATA_DEFAULTS = {
    "pancan": Path("/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/tcga_pancan_xena_20260330_top5000"),
    "kipan": Path("/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/kipan_20260209_213604"),
    "brca": Path("/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/tcga_brca20260214_001423"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["pancan", "kipan", "brca"], default="pancan")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"], choices=["LSPIN", "Concrete"])
    p.add_argument("--selections", nargs="+", default=["nosmooth", "smooth"], choices=["nosmooth", "smooth"])
    p.add_argument(
        "--lspin-smooth-lambda-scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Duplicate the selected LSPIN smooth config with scaled lambda_sparse values.",
    )
    p.add_argument(
        "--lspin-smooth-smooth-values",
        nargs="+",
        type=float,
        default=None,
        help="Optional lambda_sample_smooth values to try for LSPIN smooth variants.",
    )
    p.add_argument("--include-ungated-linear", action="store_true")
    p.add_argument("--n-folds", type=int, default=2)
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--l1-linear", type=float, default=1e-4)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--min-selected-n", type=int, default=25)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--top-genes", type=int, default=500)
    p.add_argument("--n-random-genes", type=int, default=100)
    return p.parse_args()


def _load_data(dataset: str, data_dir: Path) -> dict:
    if dataset == "pancan":
        return sds.load_pancan_split_artifacts(data_dir)
    if dataset == "kipan":
        return sds.load_kipan_split_artifacts(data_dir)
    if dataset == "brca":
        return sds.load_brca_split_artifacts(data_dir)
    raise ValueError(dataset)


def _combined_arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([data["X_train"], data["X_test"]]).astype(np.float32)
    time = np.concatenate([data["time_train"], data["time_test"]]).astype(np.float32)
    event = np.concatenate([data["event_train"], data["event_test"]]).astype(np.uint8)
    histo = np.concatenate([data["histo_train"], data["histo_test"]]).astype(str)
    genes = np.asarray(data["gene_names"]).astype(str)
    return X, time, event, histo, genes


def _tag_float(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _selected_configs(
    results_dir: Path,
    families: list[str],
    selections: list[str],
    *,
    lspin_smooth_lambda_scales: list[float],
    lspin_smooth_smooth_values: list[float] | None,
) -> pd.DataFrame:
    cfg = pd.read_csv(results_dir / "selected_comparison_configs.csv")
    cfg["family"] = cfg["family"].replace({"HardSigmoid": "LSPIN"}).astype(str)
    cfg["selection"] = cfg["selection"].astype(str)
    cfg = cfg[cfg["family"].isin(families) & cfg["selection"].isin(selections)].copy()
    cfg["gate_type"] = np.where(cfg["family"].eq("Concrete"), "concrete", "lspin_tf")
    cfg["temperature"] = np.where(cfg["family"].eq("Concrete"), 0.3, 0.5)
    cfg["concrete_mode"] = np.where(cfg["family"].eq("Concrete"), "ste", "relaxed")
    if cfg.empty:
        raise RuntimeError("No selected configs matched requested families/selections.")
    variants = []
    for _, row in cfg.iterrows():
        if str(row["family"]) == "LSPIN" and str(row["selection"]) == "smooth":
            smooth_values = (
                [float(row["lambda_sample_smooth"])]
                if lspin_smooth_smooth_values is None
                else [float(x) for x in lspin_smooth_smooth_values]
            )
            for scale in lspin_smooth_lambda_scales:
                for smooth_value in smooth_values:
                    v = row.copy()
                    v["lambda_sparse"] = float(row["lambda_sparse"]) * float(scale)
                    v["lambda_sample_smooth"] = float(smooth_value)
                    if float(scale) != 1.0 or abs(float(smooth_value) - float(row["lambda_sample_smooth"])) > 1e-12:
                        v["selection"] = (
                            f"smooth_lamx{_tag_float(scale)}"
                            f"_smooth{_tag_float(smooth_value)}"
                        )
                    variants.append(v)
        else:
            variants.append(row)
    cfg = pd.DataFrame(variants).reset_index(drop=True)
    return cfg


def _split_train_val(train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=int(seed))
    rel_train, rel_val = next(ss.split(train_idx))
    return train_idx[rel_train], train_idx[rel_val]


def _patient_subset_test(
    *,
    family: str,
    selection: str,
    fold: int,
    rep: int,
    hard: np.ndarray,
    X_holdout: np.ndarray,
    time_holdout: np.ndarray,
    event_holdout: np.ndarray,
    histo_holdout: np.ndarray,
    genes: np.ndarray,
    rng: np.random.Generator,
    min_selected_n: int,
    min_events: int,
    top_genes: int,
    n_random_genes: int,
) -> pd.DataFrame:
    gate_rate = hard.mean(axis=0)
    selected_n = hard.sum(axis=0).astype(int)
    candidate_idx = np.argsort(-gate_rate)
    candidate_idx = candidate_idx[selected_n[candidate_idx] >= int(min_selected_n)][: int(top_genes)]
    rows = []
    all_idx = np.arange(X_holdout.shape[1], dtype=int)
    for gene_idx in candidate_idx:
        mask = hard[:, gene_idx] > 0.5
        n = int(mask.sum())
        ev = int(event_holdout[mask].sum())
        if n < min_selected_n or ev < min_events:
            continue
        out = _cox_score_test_matrix(
            X_holdout[mask][:, [gene_idx]],
            time_holdout[mask],
            event_holdout[mask],
            min_events=min_events,
        )
        signed = float(out["signed_neglog10_p"][0])
        target = abs(signed) if np.isfinite(signed) else np.nan
        if not np.isfinite(target):
            continue
        pool = all_idx[all_idx != gene_idx]
        rand_idx = rng.choice(pool, size=min(int(n_random_genes), len(pool)), replace=False)
        rand = _cox_score_test_matrix(
            X_holdout[mask][:, rand_idx],
            time_holdout[mask],
            event_holdout[mask],
            min_events=min_events,
        )
        rand_abs = np.abs(np.asarray(rand["signed_neglog10_p"], dtype=float))
        rand_abs = rand_abs[np.isfinite(rand_abs)]
        if len(rand_abs) == 0:
            continue
        h_counts = pd.Series(histo_holdout[mask]).value_counts()
        rows.append({
            "family": family,
            "selection": selection,
            "fold": fold,
            "rep": rep,
            "gene_idx": int(gene_idx),
            "gene": str(genes[gene_idx]),
            "gate_rate_holdout": float(gate_rate[gene_idx]),
            "selected_patient_n": n,
            "selected_patient_events": ev,
            "top_selected_histology": str(h_counts.index[0]) if len(h_counts) else "",
            "top_selected_histology_n": int(h_counts.iloc[0]) if len(h_counts) else 0,
            "target_abs_neglog10_p": float(target),
            "target_signed_neglog10_p": float(signed),
            "random_median_abs_neglog10_p": float(np.median(rand_abs)),
            "random_p90_abs_neglog10_p": float(np.quantile(rand_abs, 0.90)),
            "target_minus_random_median": float(target - np.median(rand_abs)),
            "target_gt_random_median": bool(target > np.median(rand_abs)),
            "target_gt_random_p90": bool(target > np.quantile(rand_abs, 0.90)),
        })
    return pd.DataFrame(rows)


def _plot_summary(summary: pd.DataFrame, out_png: Path) -> None:
    if summary.empty:
        return
    df = summary.copy()
    df["label"] = [_pretty_label(f, s) for f, s in zip(df["family"], df["selection"])]
    order = list(dict.fromkeys(df["label"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(order))
    med = [
        df.loc[df["label"].eq(label), "fraction_gt_random_median"].mean()
        for label in order
    ]
    axes[0].bar(x, med, color="#4c78a8", edgecolor="0.2")
    axes[0].axhline(0.5, color="0.25", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction > random median")
    deltas = [
        df.loc[df["label"].eq(label), "median_target_minus_random_median"].dropna().to_numpy()
        for label in order
    ]
    parts = axes[1].violinplot(deltas, positions=x, showmedians=True, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor("#f58518")
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts:
            parts[key].set_color("0.2")
    axes[1].axhline(0, color="0.25", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Median advantage")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Linear-risk gated probe: patient-subset univariate signal", y=1.03)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pretty_label(family: str, selection: str) -> str:
    family = str(family)
    selection = str(selection)
    if selection == "nosmooth":
        return f"{family}\nno smooth"
    if selection == "smooth":
        return f"{family}\nsmooth"
    if selection.startswith("smooth_lamx0p25"):
        return f"{family}\nsmooth"
    if selection.startswith("smooth_lamx"):
        return f"{family}\n{selection.replace('_', ' ')}"
    if selection == "none":
        return family
    return f"{family}\n{selection}"


def _aggregate_significance(subset: pd.DataFrame) -> pd.DataFrame:
    if subset.empty:
        return pd.DataFrame()
    rows = []
    for (family, selection), sub in subset.groupby(["family", "selection"], sort=False):
        med = sub["target_gt_random_median"].dropna().astype(bool)
        p90 = sub["target_gt_random_p90"].dropna().astype(bool)
        delta = pd.to_numeric(sub["target_minus_random_median"], errors="coerce").dropna()
        try:
            binom_med = binomtest(int(med.sum()), n=int(len(med)), p=0.5, alternative="greater").pvalue
        except Exception:
            binom_med = np.nan
        try:
            binom_p90 = binomtest(int(p90.sum()), n=int(len(p90)), p=0.1, alternative="greater").pvalue
        except Exception:
            binom_p90 = np.nan
        try:
            wil_p = wilcoxon(delta, alternative="greater").pvalue if len(delta) >= 3 else np.nan
        except Exception:
            wil_p = np.nan
        rows.append({
            "family": family,
            "selection": selection,
            "n_genes": int(len(sub)),
            "n_fold_reps": int(sub[["fold", "rep"]].drop_duplicates().shape[0]),
            "fraction_gt_random_median": float(med.mean()) if len(med) else np.nan,
            "binomial_p_gt_random_median_vs_0p5": float(binom_med),
            "fraction_gt_random_p90": float(p90.mean()) if len(p90) else np.nan,
            "binomial_p_gt_random_p90_vs_0p1": float(binom_p90),
            "median_target_minus_random_median": float(delta.median()) if len(delta) else np.nan,
            "mean_target_minus_random_median": float(delta.mean()) if len(delta) else np.nan,
            "wilcoxon_p_delta_gt_zero": float(wil_p),
            "median_target_abs_neglog10_p": float(pd.to_numeric(sub["target_abs_neglog10_p"], errors="coerce").median()),
            "median_random_median_abs_neglog10_p": float(pd.to_numeric(sub["random_median_abs_neglog10_p"], errors="coerce").median()),
        })
    return pd.DataFrame(rows)


def _plot_cindex_khard(runs: pd.DataFrame, out_png: Path) -> None:
    if runs.empty:
        return
    df = runs[runs["family"].ne("UngatedLinear")].copy()
    if df.empty:
        return
    df["label"] = [_pretty_label(f, s) for f, s in zip(df["family"], df["selection"])]
    order = list(dict.fromkeys(df["label"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, metric, ylabel in [
        (axes[0], "test_cindex", "Holdout C-index"),
        (axes[1], "Khard_test_mean", "Mean hard-selected features"),
    ]:
        vals = [df.loc[df["label"].eq(label), metric].dropna().to_numpy() for label in order]
        parts = ax.violinplot(vals, showmedians=True, widths=0.75)
        for body in parts["bodies"]:
            body.set_facecolor("#6baed6")
            body.set_edgecolor("0.2")
            body.set_alpha(0.55)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if key in parts:
                parts[key].set_color("0.2")
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Linear-risk gated probe: prediction and sparsity", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_delta(subset: pd.DataFrame, out_png: Path) -> None:
    if subset.empty:
        return
    df = subset.copy()
    df["label"] = [_pretty_label(f, s) for f, s in zip(df["family"], df["selection"])]
    order = list(dict.fromkeys(df["label"]))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * len(order) + 2.0), 4.4))
    vals = [df.loc[df["label"].eq(label), "target_minus_random_median"].dropna().to_numpy() for label in order]
    parts = ax.violinplot(vals, showmedians=True, widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor("#f58518")
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts:
            parts[key].set_color("0.2")
    ax.axhline(0, color="0.25", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Gated gene advantage, -log10(p)")
    ax.set_title("Within gated patients: selected gene vs matched random-gene median")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_gated_vs_random_signal(subset: pd.DataFrame, out_png: Path) -> None:
    if subset.empty:
        return
    df = subset.copy()
    df["label"] = [_pretty_label(f, s) for f, s in zip(df["family"], df["selection"])]
    order = list(dict.fromkeys(df["label"]))
    x = np.arange(len(order), dtype=float)
    target = [
        pd.to_numeric(df.loc[df["label"].eq(label), "target_abs_neglog10_p"], errors="coerce").median()
        for label in order
    ]
    random = [
        pd.to_numeric(df.loc[df["label"].eq(label), "random_median_abs_neglog10_p"], errors="coerce").median()
        for label in order
    ]
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(order) + 2.0), 4.2))
    ax.bar(x - width / 2, target, width=width, label="Gated gene", color="#4c78a8", edgecolor="0.2")
    ax.bar(x + width / 2, random, width=width, label="Random-gene median", color="#bab0ac", edgecolor="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Median patient-subset Cox signal, -log10(p)")
    ax.set_title("Within gated patients, selected genes show stronger univariate signal")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _p_label(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-99:
        return "p<1e-99"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _plot_gated_vs_random_boxplots(
    subset: pd.DataFrame,
    aggregate: pd.DataFrame,
    out_png: Path,
) -> None:
    if subset.empty:
        return
    df = subset.copy()
    df["label"] = [_pretty_label(f, s) for f, s in zip(df["family"], df["selection"])]
    order_keys = list(dict.fromkeys(zip(df["family"], df["selection"])))
    order_labels = [_pretty_label(f, s) for f, s in order_keys]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.15 * len(order_keys) + 2.2), 4.8))
    rng = np.random.default_rng(20260407)
    xticks = []
    xticklabels = []
    width = 0.28
    colors = {"Gated gene": "#4c78a8", "Random median": "#bab0ac"}
    max_y = 0.0
    for i, ((family, selection), label) in enumerate(zip(order_keys, order_labels), start=1):
        sub = df[(df["family"].eq(family)) & (df["selection"].eq(selection))]
        gated = pd.to_numeric(sub["target_abs_neglog10_p"], errors="coerce").dropna().to_numpy()
        random = pd.to_numeric(sub["random_median_abs_neglog10_p"], errors="coerce").dropna().to_numpy()
        data = [gated, random]
        positions = [i - width / 1.5, i + width / 1.5]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.1", "linewidth": 1.3},
            boxprops={"linewidth": 0.8, "color": "0.25"},
            whiskerprops={"linewidth": 0.8, "color": "0.25"},
            capprops={"linewidth": 0.8, "color": "0.25"},
        )
        for patch, name in zip(bp["boxes"], ["Gated gene", "Random median"]):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.55)
        # light jittered sample to show distribution without drowning the plot
        for pos, values, name in zip(positions, data, ["Gated gene", "Random median"]):
            if len(values):
                idx = rng.choice(np.arange(len(values)), size=min(len(values), 200), replace=False)
                jitter = rng.normal(0, 0.025, size=len(idx))
                ax.scatter(
                    np.full(len(idx), pos) + jitter,
                    values[idx],
                    s=5,
                    color=colors[name],
                    alpha=0.16,
                    linewidths=0,
                )
        xticks.append(i)
        xticklabels.append(label)
        local_max = np.nanmax([np.nanpercentile(gated, 97.5), np.nanpercentile(random, 97.5)])
        max_y = max(max_y, float(local_max))
        agg = aggregate[(aggregate["family"].eq(family)) & (aggregate["selection"].eq(selection))]
        if not agg.empty:
            p = float(agg["wilcoxon_p_delta_gt_zero"].iloc[0])
            y = local_max + 0.18
            ax.plot([positions[0], positions[0], positions[1], positions[1]], [y - 0.05, y, y, y - 0.05],
                    color="0.25", linewidth=0.8)
            ax.text(i, y + 0.025, _p_label(p), ha="center", va="bottom", fontsize=7.5)
            max_y = max(max_y, y + 0.23)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_ylabel("Patient-subset Cox signal, -log10(p)")
    ax.set_title("Gated genes vs matched random-gene baselines within the same patients")
    ax.set_ylim(bottom=0, top=max_y * 1.02 if max_y > 0 else None)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["Gated gene"],
                   markeredgecolor="0.25", markersize=9, label="Gated gene"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["Random median"],
                   markeredgecolor="0.25", markersize=9, label="Random median"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"linear_gated_probe_{args.dataset}_{args.n_folds}fold_{args.n_reps}rep")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    configs = _selected_configs(
        results_dir,
        args.families,
        args.selections,
        lspin_smooth_lambda_scales=[float(x) for x in args.lspin_smooth_lambda_scales],
        lspin_smooth_smooth_values=args.lspin_smooth_smooth_values,
    )
    kf = KFold(n_splits=int(args.n_folds), shuffle=True, random_state=int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    run_rows = []
    subset_frames = []
    for fold, (train_all, holdout) in enumerate(kf.split(X)):
        for rep in range(int(args.n_reps)):
            seed = int(args.seed + 1000 * fold + rep)
            train_idx, val_idx = _split_train_val(train_all, seed)
            A = None
            Xt_tr = sds.as_torch(X[train_idx])
            if configs["lambda_sample_smooth"].gt(0).any():
                A = sds.build_knn_adjacency_csr(
                    Xt_tr,
                    k=int(args.knn_k),
                    pca_dim=50,
                    metric="cosine",
                    symmetrize=True,
                )
            common = dict(
                Xt_tr=Xt_tr,
                tt_tr=sds.as_torch(time[train_idx]),
                et_tr=sds.as_torch(event[train_idx]),
                Xt_val=sds.as_torch(X[val_idx]),
                tt_val=sds.as_torch(time[val_idx]),
                et_val=sds.as_torch(event[val_idx]),
                Xt_test=sds.as_torch(X[holdout]),
                tt_test=sds.as_torch(time[holdout]),
                et_test=sds.as_torch(event[holdout]),
                input_dim=X.shape[1],
                patience=int(args.patience),
                A_sample_train=A,
                device=args.device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                batch_size=int(args.batch_size),
                max_epochs=int(args.max_epochs),
                predictor="linear",
                seed=seed,
            )
            for _, cfg in configs.iterrows():
                label = f"{cfg['family']} {cfg['selection']}"
                print(f"[run] fold={fold} rep={rep} {label}", flush=True)
                model, info = sds.run_one_model(
                    **common,
                    gate_type=str(cfg["gate_type"]),
                    gate_sigma=float(cfg["gate_sigma"]),
                    lam=float(cfg["lambda_sparse"]),
                    lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
                    temperature=float(cfg["temperature"]),
                    concrete_mode=str(cfg["concrete_mode"]),
                )
                _, _, hard_t, _ = sds.get_gates(
                    model,
                    sds.as_torch(X[holdout]),
                    device=args.device,
                    hard_threshold=float(args.hard_threshold),
                    batch_size=512,
                )
                hard = hard_t.numpy().astype(np.float32)
                subset = _patient_subset_test(
                    family=str(cfg["family"]),
                    selection=str(cfg["selection"]),
                    fold=fold,
                    rep=rep,
                    hard=hard,
                    X_holdout=X[holdout],
                    time_holdout=time[holdout],
                    event_holdout=event[holdout],
                    histo_holdout=histo[holdout],
                    genes=genes,
                    rng=rng,
                    min_selected_n=int(args.min_selected_n),
                    min_events=int(args.min_events),
                    top_genes=int(args.top_genes),
                    n_random_genes=int(args.n_random_genes),
                )
                subset_frames.append(subset)
                run_rows.append({
                    "family": str(cfg["family"]),
                    "selection": str(cfg["selection"]),
                    "fold": fold,
                    "rep": rep,
                    "seed": seed,
                    "gate_sigma": float(cfg["gate_sigma"]),
                    "lambda_sparse": float(cfg["lambda_sparse"]),
                    "lambda_sample_smooth": float(cfg["lambda_sample_smooth"]),
                    "test_cindex": float(info.get("test_cindex", np.nan)),
                    "Khard_test_mean": float(info.get("Khard_test_mean", np.nan)),
                    "Ksoft_test_mean": float(info.get("Ksoft_test_mean", np.nan)),
                    "n_subset_genes_tested": int(len(subset)),
                })

            if args.include_ungated_linear:
                print(f"[run] fold={fold} rep={rep} ungated linear", flush=True)
                m = sds.make_seeded_mlp(
                    input_dim=X.shape[1],
                    hidden_dims=(),
                    dropout_p=0.0,
                    seed=seed,
                )
                info = sds.train_deepsurv_mlp_l1(
                    m,
                    sds.as_torch(X[train_idx]),
                    sds.as_torch(time[train_idx]),
                    sds.as_torch(event[train_idx]),
                    sds.as_torch(X[val_idx]),
                    sds.as_torch(time[val_idx]),
                    sds.as_torch(event[val_idx]),
                    config=sds.MLPTrainConfig(
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        lambda_l1_input=float(args.l1_linear),
                        batch_size=int(args.batch_size),
                        max_epochs=int(args.max_epochs),
                        patience=int(args.patience),
                    ),
                    device=args.device,
                    verbose=False,
                )
                c, _ = sds.eval_mlp_cindex(
                    m,
                    sds.as_torch(X[holdout]),
                    sds.as_torch(time[holdout]),
                    sds.as_torch(event[holdout]),
                    device=args.device,
                )
                run_rows.append({
                    "family": "UngatedLinear",
                    "selection": "none",
                    "fold": fold,
                    "rep": rep,
                    "seed": seed,
                    "test_cindex": float(c),
                    "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
                    "n_subset_genes_tested": 0,
                })

    runs = pd.DataFrame(run_rows)
    subset = pd.concat(subset_frames, ignore_index=True) if subset_frames else pd.DataFrame()
    if subset.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            subset.groupby(["family", "selection", "fold", "rep"], as_index=False)
            .agg(
                n_genes=("gene", "count"),
                fraction_gt_random_median=("target_gt_random_median", "mean"),
                fraction_gt_random_p90=("target_gt_random_p90", "mean"),
                median_target_abs_neglog10_p=("target_abs_neglog10_p", "median"),
                median_random_median_abs_neglog10_p=("random_median_abs_neglog10_p", "median"),
                median_target_minus_random_median=("target_minus_random_median", "median"),
            )
        )
    aggregate = _aggregate_significance(subset)
    runs.to_csv(outdir / "linear_gated_probe_runs.csv", index=False)
    subset.to_csv(outdir / "linear_gated_probe_patient_subset_predictivity.csv", index=False)
    summary.to_csv(outdir / "linear_gated_probe_patient_subset_predictivity_summary.csv", index=False)
    aggregate.to_csv(outdir / "linear_gated_probe_patient_subset_predictivity_aggregate_significance.csv", index=False)
    _plot_summary(summary, outdir / "fig_linear_gated_probe_patient_subset_predictivity.png")
    _plot_cindex_khard(runs, outdir / "fig_linear_gated_probe_cindex_khard.png")
    _plot_gene_delta(subset, outdir / "fig_linear_gated_probe_gene_delta.png")
    _plot_gated_vs_random_signal(subset, outdir / "fig_linear_gated_probe_gated_vs_random_signal.png")
    _plot_gated_vs_random_boxplots(
        subset,
        aggregate,
        outdir / "fig_linear_gated_probe_gated_vs_random_signal_boxplots.png",
    )
    print(f"[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
