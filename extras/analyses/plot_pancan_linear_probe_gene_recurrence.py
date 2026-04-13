#!/usr/bin/env python3
"""Summarize repeated-refit gene recurrence for a gated patient-subset signal probe.

This script uses the saved patient-subset predictivity table from
``quick_linear_gated_probe.py``. Each fold/replicate/model contributes a ranked
list of selected genes; we compare overlap among those lists with matched
random sets from the same feature universe, and write a candidate table of
recurrent tumor-type-associated selected genes.
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS, PROCESSED_DATASETS


RUN_DEFAULTS = {
    "pancan": CANONICAL_RUNS["pancan_adaptive_gentle"] / "linear_gated_probe_pancan_full_more_sparse_allfamilies_lamx3_3fold_2rep",
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"] / "linear_gated_probe_kipan_full_more_sparse_allfamilies_lamx3_3fold_2rep",
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"] / "linear_gated_probe_brca_3fold_2rep",
}
DATA_DEFAULTS = PROCESSED_DATASETS

# Lightweight, transparent highlight list for biology-facing examples. This is
# intentionally not used for any statistical test; it only flags recurrent
# candidates whose symbols have a recognizable tumor-biology/literature context.
CURATED_TUMOR_GENE_NOTES = {
    ("BRCA", "EGFR"): "Basal-like / triple-negative breast cancer marker and therapeutic target context",
    ("BRCA", "FGD3"): "Reported breast-cancer prognostic factor",
    ("BRCA", "EPHB2"): "Reported breast-cancer prognostic / receptor-tyrosine-kinase context",
    ("BRCA", "SERPINE1"): "Reported breast-cancer prognostic and invasion/ECM context",
    ("BRCA", "SPP1"): "Reported breast-cancer prognostic and tumor-microenvironment context",
    ("BRCA", "PBK"): "Reported breast-cancer proliferation/prognosis context",
    ("BRCA", "SLC7A11"): "Reported breast-cancer oxidative-stress/ferroptosis and prognosis context",
    ("BRCA", "TFRC"): "Iron-uptake marker with cancer proliferation context",
    ("BRCA", "TTK"): "Mitotic checkpoint kinase with breast-cancer proliferation context",
    ("BRCA", "CDK6"): "Cell-cycle kinase with breast-cancer relevance",
    ("KIRC", "COL11A1"): "ECM/collagen remodeling marker reported across solid tumors; candidate KIRC context",
    ("LGG", "ADAM12"): "Reported glioma-associated metalloprotease/prognosis context",
    ("UCEC", "CIRBP"): "Stress-response RNA-binding protein; candidate UCEC context",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["pancan", "kipan", "brca"], default="pancan")
    p.add_argument("--probe-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--top-n", nargs="+", type=int, default=[25, 50, 100, 200, 400])
    p.add_argument("--candidate-top-n", type=int, default=100)
    p.add_argument("--table-min-run-count", type=int, default=2)
    p.add_argument("--table-min-tumor-consistency", type=float, default=0.67)
    p.add_argument("--table-min-advantage", type=float, default=5.0)
    p.add_argument("--table-max-rows", type=int, default=40)
    p.add_argument("--n-random", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260408)
    return p.parse_args()


def _display_selection(selection: str) -> str:
    selection = str(selection)
    if selection == "nosmooth" or selection.startswith("nosmooth_lamx"):
        return "nosmooth"
    if selection == "smooth" or selection.startswith("smooth_lamx"):
        return "smooth"
    return selection


def _display_model(family: str, selection: str) -> str:
    return f"{family} {_display_selection(selection)}"


def _prefix(dataset: str) -> str:
    return f"{dataset}_gated_patient_subset_signal_probe"


def _load_gene_universe(data_dir: Path, observed_genes: pd.Series) -> np.ndarray:
    split_path = data_dir / "splits_and_core.npz"
    if split_path.exists():
        with np.load(split_path, allow_pickle=True) as z:
            if "gene_names" in z.files:
                return np.asarray(z["gene_names"]).astype(str)
    return np.asarray(sorted(pd.Series(observed_genes).dropna().astype(str).unique()))


def _sets_from_table(df: pd.DataFrame, n: int) -> dict[tuple[str, str, int, int], set[str]]:
    sets = {}
    key_cols = ["family", "selection", "fold", "rep"]
    for key, g in df.groupby(key_cols, sort=True):
        g = g.sort_values(
            ["gate_rate_holdout", "selected_patient_n", "target_minus_random_median"],
            ascending=[False, False, False],
        )
        genes = g["gene"].astype(str).drop_duplicates().head(n).tolist()
        if len(genes) == n:
            sets[key] = set(genes)
    return sets


def _pairwise_rows(sets: dict[tuple[str, str, int, int], set[str]], n: int) -> list[dict]:
    rows = []
    for key_a, key_b in itertools.combinations(sorted(sets), 2):
        # Only compare repeated refits within the same family/selection.
        if key_a[:2] != key_b[:2]:
            continue
        a, b = sets[key_a], sets[key_b]
        inter = len(a & b)
        union = len(a | b)
        rows.append(
            {
                "family": key_a[0],
                "selection": key_a[1],
                "model_label": _display_model(key_a[0], key_a[1]),
                "top_n": n,
                "fold_a": key_a[2],
                "rep_a": key_a[3],
                "fold_b": key_b[2],
                "rep_b": key_b[3],
                "overlap_n": inter,
                "jaccard": inter / union if union else np.nan,
                "source": "observed",
            }
        )
    return rows


def _random_pairwise_rows(
    *,
    universe: np.ndarray,
    model_keys: list[tuple[str, str, int, int]],
    n: int,
    n_random: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    grouped_keys: dict[tuple[str, str], list[tuple[str, str, int, int]]] = {}
    for key in model_keys:
        grouped_keys.setdefault(key[:2], []).append(key)
    for (family, selection), keys in grouped_keys.items():
        if len(keys) < 2:
            continue
        pairs = list(itertools.combinations(range(len(keys)), 2))
        for sim in range(int(n_random)):
            sampled = [
                set(rng.choice(universe, size=n, replace=False).astype(str))
                for _ in keys
            ]
            for ia, ib in pairs:
                a, b = sampled[ia], sampled[ib]
                inter = len(a & b)
                union = len(a | b)
                rows.append(
                    {
                        "family": family,
                        "selection": selection,
                        "model_label": _display_model(family, selection),
                        "top_n": n,
                        "random_iter": sim,
                        "overlap_n": inter,
                        "jaccard": inter / union if union else np.nan,
                        "source": "random",
                    }
                )
    return rows


def _summarize_overlap(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, selection, label, n), g in pair_df.groupby(
        ["family", "selection", "model_label", "top_n"], sort=True
    ):
        obs = g.loc[g["source"].eq("observed"), "jaccard"].dropna()
        rnd = g.loc[g["source"].eq("random"), "jaccard"].dropna()
        if len(obs) == 0 or len(rnd) == 0:
            continue
        mw = mannwhitneyu(obs, rnd, alternative="greater")
        rows.append(
            {
                "family": family,
                "selection": selection,
                "model_label": label,
                "top_n": n,
                "observed_pairs": len(obs),
                "random_pairs": len(rnd),
                "observed_mean_jaccard": float(obs.mean()),
                "observed_median_jaccard": float(obs.median()),
                "random_mean_jaccard": float(rnd.mean()),
                "random_median_jaccard": float(rnd.median()),
                "observed_mean_overlap_n": float(
                    g.loc[g["source"].eq("observed"), "overlap_n"].mean()
                ),
                "random_mean_overlap_n": float(
                    g.loc[g["source"].eq("random"), "overlap_n"].mean()
                ),
                "mannwhitney_p_greater": float(mw.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _candidate_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    top_rows = []
    key_cols = ["family", "selection", "fold", "rep"]
    for key, g in df.groupby(key_cols, sort=True):
        g = g.sort_values(
            ["gate_rate_holdout", "selected_patient_n", "target_minus_random_median"],
            ascending=[False, False, False],
        ).drop_duplicates("gene")
        top = g.head(n).copy()
        top["run_key"] = "|".join(map(str, key))
        top_rows.append(top)
    top_df = pd.concat(top_rows, ignore_index=True)

    rows = []
    for (family, selection, gene), g in top_df.groupby(["family", "selection", "gene"], sort=True):
        hist_counts = g["top_selected_histology"].astype(str).value_counts()
        rows.append(
            {
                "family": family,
                "selection": selection,
                "model_label": _display_model(family, selection),
                "gene": gene,
                "run_count": int(g["run_key"].nunique()),
                "primary_histology": hist_counts.index[0] if len(hist_counts) else "",
                "primary_histology_count": int(hist_counts.iloc[0]) if len(hist_counts) else 0,
                "primary_histology_fraction": float(hist_counts.iloc[0] / len(g)) if len(hist_counts) else np.nan,
                "median_gate_rate_holdout": float(g["gate_rate_holdout"].median()),
                "median_selected_patient_n": float(g["selected_patient_n"].median()),
                "median_target_abs_neglog10_p": float(g["target_abs_neglog10_p"].median()),
                "median_random_abs_neglog10_p": float(g["random_median_abs_neglog10_p"].median()),
                "median_target_minus_random_median": float(g["target_minus_random_median"].median()),
                "fraction_beating_random_median": float(g["target_gt_random_median"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "run_count",
            "primary_histology_fraction",
            "median_target_minus_random_median",
            "median_gate_rate_holdout",
        ],
        ascending=[False, False, False, False],
    )
    return out


def _load_annotation_cache(outdir: Path) -> pd.DataFrame:
    candidates = [
        outdir / "pancan_gated_patient_subset_signal_probe_recurrent_gene_candidate_ensembl_annotations.csv",
        outdir / "pancan_gated_patient_subset_signal_probe_recurrent_gene_candidates_annotated.csv",
        outdir / "pancan_linear_probe_recurrent_gene_candidate_ensembl_annotations.csv",
        outdir / "pancan_linear_probe_recurrent_gene_candidates_annotated.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        ann = pd.read_csv(path)
        if {"gene", "symbol"}.issubset(ann.columns):
            cols = ["gene", "symbol"]
            if "description" in ann.columns:
                cols.append("description")
            return ann[cols].drop_duplicates("gene")
    return pd.DataFrame(columns=["gene", "symbol", "description"])


def _fallback_symbol(gene: str) -> str:
    # The project-local PanCan data uses Ensembl IDs. If an annotation cache was
    # not generated, keep the output unambiguous rather than guessing symbols.
    return str(gene)


def _annotate_candidates(candidates: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    ann = _load_annotation_cache(outdir)
    out = candidates.merge(ann, on="gene", how="left")
    out["symbol"] = out["symbol"].fillna("").astype(str)
    out.loc[out["symbol"].str.len().eq(0), "symbol"] = out.loc[
        out["symbol"].str.len().eq(0), "gene"
    ].map(_fallback_symbol)
    if "description" not in out.columns:
        out["description"] = ""
    out["description"] = out["description"].fillna("").astype(str)
    out["curated_tumor_gene_note"] = [
        CURATED_TUMOR_GENE_NOTES.get((str(hist), str(sym)), "")
        for hist, sym in zip(out["primary_histology"], out["symbol"])
    ]
    out["curated_tumor_gene"] = out["curated_tumor_gene_note"].str.len().gt(0)
    return out


def _format_p(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if x < 1e-3:
        return f"{x:.1e}"
    return f"{x:.3f}"


def _make_supplement_candidate_table(
    candidates: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    *,
    min_run_count: int,
    min_tumor_consistency: float,
    min_advantage: float,
    max_rows: int,
) -> pd.DataFrame:
    out = candidates[
        candidates["run_count"].ge(int(min_run_count))
        & candidates["primary_histology_fraction"].ge(float(min_tumor_consistency))
        & candidates["median_target_minus_random_median"].gt(float(min_advantage))
    ].copy()
    out = out[~out["symbol"].astype(str).str.match(r"^ENSG\\d+", na=False)].copy()
    out = out.sort_values(
        [
            "curated_tumor_gene",
            "run_count",
            "primary_histology_fraction",
            "median_target_minus_random_median",
        ],
        ascending=[False, False, False, False],
    ).head(int(max_rows))
    sig = overlap_summary.set_index(["model_label", "top_n"])
    table_rows = []
    for _, row in out.iterrows():
        key = (row["model_label"], 100)
        if key in sig.index:
            sig_row = sig.loc[key]
            overlap_text = (
                f"{sig_row['observed_mean_overlap_n']:.1f} vs "
                f"{sig_row['random_mean_overlap_n']:.1f}; "
                f"p={_format_p(float(sig_row['mannwhitney_p_greater']))}"
            )
        else:
            overlap_text = ""
        table_rows.append(
            {
                "model": row["model_label"],
                "tumor_type": row["primary_histology"],
                "gene_symbol": row["symbol"],
                "ensembl_id": row["gene"],
                "refit_recurrence": f"{int(row['run_count'])}/6",
                "same_top_gated_tumor_across_refits": (
                    f"{int(row['primary_histology_count'])}/{int(row['run_count'])}"
                ),
                "median_gate_rate": row["median_gate_rate_holdout"],
                "median_selected_patients": row["median_selected_patient_n"],
                "median_cox_signal_neglog10p": row["median_target_abs_neglog10_p"],
                "median_random_signal_neglog10p": row["median_random_abs_neglog10_p"],
                "median_gated_gene_advantage_neglog10p": row[
                    "median_target_minus_random_median"
                ],
                "fraction_refits_beating_random_median": row[
                    "fraction_beating_random_median"
                ],
                "top100_refit_overlap_vs_random_for_model": overlap_text,
                "curated_tumor_gene_note": row["curated_tumor_gene_note"],
            }
        )
    return pd.DataFrame(table_rows)


def _plot_overlap(summary: pd.DataFrame, outdir: Path) -> None:
    labels = list(dict.fromkeys(summary["model_label"].tolist()))
    palette = {
        "Concrete nosmooth": "#607c8e",
        "Concrete smooth": "#0f6b63",
        "LSPIN nosmooth": "#b36b1e",
        "LSPIN smooth": "#bc3c29",
        "L-LSPIN nosmooth": "#7f3c8d",
        "L-LSPIN smooth": "#af5bbd",
        "L-Concrete nosmooth": "#1b9e77",
        "L-Concrete smooth": "#66c2a5",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for label in labels:
        g = summary[summary["model_label"].eq(label)].sort_values("top_n")
        color = palette.get(label, None)
        axes[0].plot(g["top_n"], g["observed_mean_jaccard"], marker="o", lw=2.0, label=label, color=color)
        axes[0].plot(g["top_n"], g["random_mean_jaccard"], marker="o", lw=1.4, ls="--", color=color, alpha=0.55)
        axes[1].plot(g["top_n"], g["observed_mean_overlap_n"], marker="o", lw=2.0, label=label, color=color)
        axes[1].plot(g["top_n"], g["random_mean_overlap_n"], marker="o", lw=1.4, ls="--", color=color, alpha=0.55)
    axes[0].set_title("Repeated-refit gene overlap")
    axes[0].set_xlabel("Top N selected genes per refit")
    axes[0].set_ylabel("Mean pairwise Jaccard index")
    axes[1].set_title("Exact shared genes")
    axes[1].set_xlabel("Top N selected genes per refit")
    axes[1].set_ylabel("Mean pairwise overlap count")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary["top_n"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(title="solid: observed; dashed: random", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(outdir / "fig_gated_patient_subset_gene_recurrence_overlap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_heatmap(candidates: pd.DataFrame, outdir: Path, top_per_model: int = 15) -> None:
    plot_df = (
        candidates.sort_values(
            ["model_label", "run_count", "median_target_minus_random_median"],
            ascending=[True, False, False],
        )
        .groupby("model_label", sort=False)
        .head(top_per_model)
        .copy()
    )
    if plot_df.empty:
        return
    label_gene = plot_df["symbol"] if "symbol" in plot_df.columns else plot_df["gene"]
    plot_df["row_label"] = (
        label_gene.astype(str)
        + " / "
        + plot_df["primary_histology"].astype(str)
        + " ("
        + plot_df["model_label"].astype(str)
        + ")"
    )
    plot_df = plot_df.sort_values(["model_label", "run_count", "median_target_minus_random_median"], ascending=[True, True, True])
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.23 * len(plot_df))))
    sc = ax.scatter(
        plot_df["median_target_minus_random_median"],
        y,
        s=25 + 23 * plot_df["run_count"],
        c=plot_df["primary_histology_fraction"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color="0.35", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"], fontsize=7)
    ax.set_xlabel("Median selected-gene advantage, -log10(p)")
    ax.set_title("Recurrent tumor-type-associated selected genes")
    ax.grid(axis="x", alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Fraction assigned to primary tumor type")
    for size, label in [(3, "3 runs"), (6, "6 runs")]:
        ax.scatter([], [], s=25 + 23 * size, c="0.65", edgecolor="white", label=label)
    ax.legend(title="Recurrence", loc="lower right", frameon=True, fontsize=7, title_fontsize=8)
    fig.savefig(outdir / "fig_gated_patient_subset_recurrent_gene_candidates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    probe_dir = args.probe_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or args.probe_dir
    outdir = args.outdir or probe_dir
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    patient_path = probe_dir / "gated_patient_subset_signal_probe_patient_subset_predictivity.csv"
    if not patient_path.exists():
        patient_path = probe_dir / "linear_gated_probe_patient_subset_predictivity.csv"
    df = pd.read_csv(patient_path)
    df["selection"] = df["selection"].astype(str)
    universe = _load_gene_universe(data_dir, df["gene"])

    all_pair_rows = []
    for n in args.top_n:
        sets = _sets_from_table(df, int(n))
        all_pair_rows.extend(_pairwise_rows(sets, int(n)))
        all_pair_rows.extend(
            _random_pairwise_rows(
                universe=universe,
                model_keys=list(sets.keys()),
                n=int(n),
                n_random=args.n_random,
                rng=rng,
            )
        )
    pair_df = pd.DataFrame(all_pair_rows)
    summary = _summarize_overlap(pair_df)
    candidates = _annotate_candidates(_candidate_table(df, int(args.candidate_top_n)), outdir)
    supp_table = _make_supplement_candidate_table(
        candidates,
        summary,
        min_run_count=args.table_min_run_count,
        min_tumor_consistency=args.table_min_tumor_consistency,
        min_advantage=args.table_min_advantage,
        max_rows=args.table_max_rows,
    )

    prefix = _prefix(args.dataset)
    pair_df.to_csv(outdir / f"{prefix}_gene_recurrence_pairwise.csv", index=False)
    summary.to_csv(outdir / f"{prefix}_gene_recurrence_summary.csv", index=False)
    candidates.to_csv(outdir / f"{prefix}_recurrent_gene_candidates.csv", index=False)
    candidates.to_csv(outdir / f"{prefix}_recurrent_gene_candidates_annotated.csv", index=False)
    supp_table.to_csv(outdir / f"{prefix}_recurrent_gene_supplement_table.csv", index=False)

    _plot_overlap(summary, outdir)
    _plot_candidate_heatmap(candidates, outdir)
    print(f"[done] wrote recurrence outputs to {outdir}")


if __name__ == "__main__":
    main()
