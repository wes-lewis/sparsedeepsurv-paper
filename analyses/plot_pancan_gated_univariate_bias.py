#!/usr/bin/env python3
"""Test whether PanCan gated genes are biased toward tumor-type survival signal.

This is a post-hoc diagnostic. It does not retrain models. It loads saved
adaptive checkpoints, aggregates hard gate rates by tumor type, recomputes
tumor-type-specific univariate Cox score-test signal from local split artifacts,
and asks whether genes are gated toward tumor types where their univariate Cox
signal ranks unusually highly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ANALYSES_DIR = Path(__file__).resolve().parent
SDS_SRC = Path("/banach2/wes/lspin-repos/sparsedeepsurv/src")
if str(ANALYSES_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSES_DIR))
if str(SDS_SRC) not in sys.path:
    sys.path.insert(0, str(SDS_SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sparsedeepsurv as sds
from plot_kipan_gated_univariate_dotplot import (
    _aggregate_gate_rates,
    _cox_score_test_matrix,
    _compute_univariate_table,
    _matching_saved_runs,
    _make_model,
    _read_partial_rows,
    _selected_config,
)


PANCAN_RUN_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "ch3_pancan_adaptive_v2_selfcontained_ste_randominit_20260406_141705"
)
PANCAN_DATA_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/"
    "tcga_pancan_xena_20260330_top5000"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=PANCAN_RUN_DEFAULT)
    p.add_argument("--data-dir", type=Path, default=PANCAN_DATA_DEFAULT)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"], choices=["LSPIN", "Concrete"])
    p.add_argument("--selection", choices=["smooth", "nosmooth"], default="smooth")
    p.add_argument("--model-scope", choices=["available", "representative"], default="available")
    p.add_argument("--univariate-scope", choices=["all", "train", "test"], default="all")
    p.add_argument("--min-histology-n", type=int, default=40)
    p.add_argument("--min-events", type=int, default=10)
    p.add_argument("--min-gate-rates", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    p.add_argument("--min-specificities", nargs="+", type=float, default=[0.00, 0.025, 0.05, 0.10])
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-permutations", type=int, default=2000)
    p.add_argument("--n-random-genes", type=int, default=200)
    p.add_argument("--subset-min-gate-rate", type=float, default=0.20)
    p.add_argument("--subset-min-specificity", type=float, default=0.025)
    p.add_argument("--subset-min-selected-n", type=int, default=40)
    p.add_argument("--subset-top-genes", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260406)
    return p.parse_args()


def _eligible_histologies(univariate_df: pd.DataFrame, min_histo_n: int, min_events: int) -> list[str]:
    counts = (
        univariate_df[["histology", "univariate_n", "univariate_events"]]
        .drop_duplicates()
        .copy()
    )
    counts = counts[
        (pd.to_numeric(counts["univariate_n"], errors="coerce") >= min_histo_n)
        & (pd.to_numeric(counts["univariate_events"], errors="coerce") >= min_events)
    ]
    return sorted(counts["histology"].astype(str).unique())


def _make_matrices(
    gate_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
    histologies: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gate = (
        gate_df[gate_df["histology"].astype(str).isin(histologies)]
        .pivot(index="gene_idx", columns="histology", values="gate_rate")
        .reindex(columns=histologies)
    )
    signal = (
        univariate_df[univariate_df["histology"].astype(str).isin(histologies)]
        .pivot(index="gene_idx", columns="histology", values="abs_neglog10_p")
        .reindex(index=gate.index, columns=histologies)
    )
    genes = (
        gate_df[["gene_idx", "gene"]]
        .drop_duplicates()
        .set_index("gene_idx")
        .reindex(gate.index)["gene"]
        .astype(str)
        .to_numpy()
    )
    return gate.to_numpy(float), signal.to_numpy(float), gate.index.to_numpy(int), genes


def _rank_desc(signal: np.ndarray) -> np.ndarray:
    # Rank 1 is strongest signal. NaNs get worst rank.
    filled = np.where(np.isfinite(signal), signal, -np.inf)
    order = np.argsort(-filled, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    rows = np.arange(signal.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, signal.shape[1] + 1)[None, :]
    return ranks


def _perm_p(values: np.ndarray, observed: float, *, alternative: str) -> float:
    if len(values) == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "greater":
        return float((1 + np.sum(values >= observed)) / (len(values) + 1))
    if alternative == "less":
        return float((1 + np.sum(values <= observed)) / (len(values) + 1))
    raise ValueError(alternative)


def _evaluate_threshold(
    *,
    gate: np.ndarray,
    signal: np.ndarray,
    gene_idx: np.ndarray,
    genes: np.ndarray,
    histologies: list[str],
    min_gate_rate: float,
    min_specificity: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    best = np.nanargmax(gate, axis=1)
    best_rate = gate[np.arange(gate.shape[0]), best]
    sorted_gate = np.sort(gate, axis=1)
    specificity = sorted_gate[:, -1] - sorted_gate[:, -2] if gate.shape[1] > 1 else best_rate
    ok_signal = np.isfinite(signal[np.arange(signal.shape[0]), best])
    keep = (best_rate >= min_gate_rate) & (specificity >= min_specificity) & ok_signal

    sub_signal = signal[keep]
    sub_best = best[keep]
    sub_gene_idx = gene_idx[keep]
    sub_genes = genes[keep]
    sub_rate = best_rate[keep]
    sub_spec = specificity[keep]
    if sub_signal.shape[0] == 0:
        return pd.DataFrame(), {
            "n_genes": 0,
            "top1_fraction": np.nan,
            "top3_fraction": np.nan,
            "mean_rank": np.nan,
            "median_rank": np.nan,
            "median_assigned_minus_best_other": np.nan,
            "top1_perm_p": np.nan,
            "top3_perm_p": np.nan,
            "mean_rank_perm_p": np.nan,
        }

    ranks = _rank_desc(sub_signal)
    row = np.arange(sub_signal.shape[0])
    assigned_signal = sub_signal[row, sub_best]
    assigned_rank = ranks[row, sub_best]
    other = sub_signal.copy()
    other[row, sub_best] = np.nan
    delta_best_other = assigned_signal - np.nanmax(other, axis=1)
    delta_median_other = assigned_signal - np.nanmedian(other, axis=1)

    perm_top1 = np.empty(n_permutations, dtype=float)
    perm_top3 = np.empty(n_permutations, dtype=float)
    perm_mean_rank = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        perm = rng.permutation(sub_best)
        perm_rank = ranks[row, perm]
        perm_top1[b] = np.mean(perm_rank == 1)
        perm_top3[b] = np.mean(perm_rank <= min(3, len(histologies)))
        perm_mean_rank[b] = np.mean(perm_rank)

    rows = pd.DataFrame({
        "gene_idx": sub_gene_idx,
        "gene": sub_genes,
        "assigned_histology": np.array(histologies, dtype=object)[sub_best],
        "assigned_gate_rate": sub_rate,
        "assigned_gate_specificity": sub_spec,
        "assigned_abs_neglog10_p": assigned_signal,
        "assigned_signal_rank": assigned_rank,
        "assigned_in_top1": assigned_rank == 1,
        "assigned_in_top3": assigned_rank <= min(3, len(histologies)),
        "assigned_minus_best_other": delta_best_other,
        "assigned_minus_median_other": delta_median_other,
    })
    summary = {
        "n_genes": int(len(rows)),
        "n_histologies": int(len(histologies)),
        "top1_fraction": float(np.mean(assigned_rank == 1)),
        "top1_null_mean": float(np.mean(perm_top1)),
        "top1_null_lo": float(np.quantile(perm_top1, 0.025)),
        "top1_null_hi": float(np.quantile(perm_top1, 0.975)),
        "top1_perm_p": _perm_p(perm_top1, float(np.mean(assigned_rank == 1)), alternative="greater"),
        "top3_fraction": float(np.mean(assigned_rank <= min(3, len(histologies)))),
        "top3_null_mean": float(np.mean(perm_top3)),
        "top3_null_lo": float(np.quantile(perm_top3, 0.025)),
        "top3_null_hi": float(np.quantile(perm_top3, 0.975)),
        "top3_perm_p": _perm_p(perm_top3, float(np.mean(assigned_rank <= min(3, len(histologies)))), alternative="greater"),
        "mean_rank": float(np.mean(assigned_rank)),
        "mean_rank_null_mean": float(np.mean(perm_mean_rank)),
        "mean_rank_null_lo": float(np.quantile(perm_mean_rank, 0.025)),
        "mean_rank_null_hi": float(np.quantile(perm_mean_rank, 0.975)),
        "mean_rank_perm_p": _perm_p(perm_mean_rank, float(np.mean(assigned_rank)), alternative="less"),
        "median_rank": float(np.median(assigned_rank)),
        "median_assigned_minus_best_other": float(np.nanmedian(delta_best_other)),
        "median_assigned_minus_median_other": float(np.nanmedian(delta_median_other)),
    }
    return rows, summary


def _plot_summary(summary: pd.DataFrame, out_png: Path) -> None:
    if summary.empty:
        return
    df = summary.copy()
    df["threshold"] = (
        "rate>=" + df["min_gate_rate"].map(lambda x: f"{x:g}")
        + "\nspec>=" + df["min_specificity"].map(lambda x: f"{x:g}")
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    colors = {"LSPIN": "#3b6ea8", "Concrete": "#cc7a29"}
    for ax, metric, ylabel in [
        (axes[0], "top1_fraction", "Assigned tumor type is top Cox-signal type"),
        (axes[1], "top3_fraction", "Assigned tumor type is in top 3 Cox-signal types"),
    ]:
        for family, sub in df.groupby("family", sort=False):
            sub = sub.sort_values(["min_gate_rate", "min_specificity"])
            x = np.arange(len(sub))
            ax.plot(x, sub[metric], marker="o", linewidth=1.8, label=family, color=colors.get(family))
            null = "top1_null_mean" if metric == "top1_fraction" else "top3_null_mean"
            lo = "top1_null_lo" if metric == "top1_fraction" else "top3_null_lo"
            hi = "top1_null_hi" if metric == "top1_fraction" else "top3_null_hi"
            ax.plot(x, sub[null], linestyle="--", linewidth=1.0, color=colors.get(family), alpha=0.75)
            ax.fill_between(x, sub[lo], sub[hi], color=colors.get(family), alpha=0.12, linewidth=0)
        ax.set_xticks(np.arange(len(df["threshold"].unique())))
        ax.set_xticklabels(df.drop_duplicates("threshold")["threshold"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, title="Solid=observed\nDashed=null mean")
    fig.suptitle("PanCan gate-to-tumor-type bias in univariate Cox signal", y=1.03)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _aggregate_sample_gate_matrix(
    cfg,
    saved_runs: pd.DataFrame,
    data: dict,
    *,
    device: str,
    hard_threshold: float,
) -> np.ndarray:
    X_test_t = sds.as_torch(data["X_test"])
    input_dim = X_test_t.shape[1]
    total = np.zeros((X_test_t.shape[0], input_dim), dtype=np.float32)
    n_models = 0
    for _, row in saved_runs.sort_values(["phase", "rep_id", "seed"]).iterrows():
        model = _make_model(cfg, input_dim=input_dim)
        state_path = Path(str(row["state_dict_path"]))
        try:
            sd = __import__("torch").load(state_path, map_location=device, weights_only=True)
        except TypeError:
            sd = __import__("torch").load(state_path, map_location=device)
        model.load_state_dict(sd)
        model.to(device)
        _, _, hard, _ = sds.get_gates(
            model,
            X_test_t,
            device=device,
            hard_threshold=hard_threshold,
            batch_size=512,
        )
        total += hard.numpy().astype(np.float32)
        n_models += 1
    if n_models == 0:
        raise RuntimeError("No saved checkpoints were available for sample-level gates.")
    return total / n_models


def _patient_subset_predictivity_test(
    *,
    family: str,
    sample_gate_rate: np.ndarray,
    gate_df: pd.DataFrame,
    data: dict,
    rng: np.random.Generator,
    min_gate_rate: float,
    min_specificity: float,
    min_selected_n: int,
    top_genes: int,
    n_random_genes: int,
    min_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = np.asarray(data["X_test"])
    time = np.asarray(data["time_test"])
    event = np.asarray(data["event_test"])
    genes = np.asarray(data["gene_names"]).astype(str)

    per_gene = (
        gate_df.sort_values(["gene_idx", "gate_rate"], ascending=[True, False])
        .groupby("gene_idx", as_index=False)
        .first()
    )
    cand = per_gene[
        (pd.to_numeric(per_gene["gate_rate"], errors="coerce") >= min_gate_rate)
        & (pd.to_numeric(per_gene["gate_specificity"], errors="coerce") >= min_specificity)
    ].copy()
    cand = cand.sort_values(["gate_specificity", "gate_rate"], ascending=[False, False]).head(top_genes)

    rows = []
    random_pool_all = np.arange(X.shape[1], dtype=int)
    for _, row in cand.iterrows():
        gene_idx = int(row["gene_idx"])
        selected_mask = sample_gate_rate[:, gene_idx] >= 0.5
        selected_n = int(selected_mask.sum())
        selected_events = int(event[selected_mask].sum()) if selected_n else 0
        if selected_n < min_selected_n or selected_events < min_events:
            continue

        target = _cox_score_test_matrix(
            X[selected_mask][:, [gene_idx]],
            time[selected_mask],
            event[selected_mask],
            min_events=min_events,
        )
        target_signed = float(target["signed_neglog10_p"][0])
        target_abs = abs(target_signed) if np.isfinite(target_signed) else np.nan
        if not np.isfinite(target_abs):
            continue

        random_pool = random_pool_all[random_pool_all != gene_idx]
        random_idx = rng.choice(random_pool, size=min(n_random_genes, len(random_pool)), replace=False)
        rand = _cox_score_test_matrix(
            X[selected_mask][:, random_idx],
            time[selected_mask],
            event[selected_mask],
            min_events=min_events,
        )
        rand_abs = np.abs(np.asarray(rand["signed_neglog10_p"], dtype=float))
        rand_abs = rand_abs[np.isfinite(rand_abs)]
        if len(rand_abs) == 0:
            continue
        rows.append({
            "family": family,
            "gene_idx": gene_idx,
            "gene": genes[gene_idx],
            "assigned_histology": str(row["histology"]),
            "assigned_gate_rate": float(row["gate_rate"]),
            "assigned_gate_specificity": float(row["gate_specificity"]),
            "selected_patient_n": selected_n,
            "selected_patient_events": selected_events,
            "target_abs_neglog10_p": target_abs,
            "target_signed_neglog10_p": target_signed,
            "random_mean_abs_neglog10_p": float(np.mean(rand_abs)),
            "random_median_abs_neglog10_p": float(np.median(rand_abs)),
            "random_p90_abs_neglog10_p": float(np.quantile(rand_abs, 0.90)),
            "target_minus_random_median": float(target_abs - np.median(rand_abs)),
            "target_gt_random_median": bool(target_abs > np.median(rand_abs)),
            "target_gt_random_p90": bool(target_abs > np.quantile(rand_abs, 0.90)),
            "random_gene_count": int(len(rand_abs)),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame([{
            "family": family,
            "n_genes": 0,
            "fraction_gt_random_median": np.nan,
            "fraction_gt_random_p90": np.nan,
            "median_target_minus_random_median": np.nan,
        }])
    summary = pd.DataFrame([{
        "family": family,
        "n_genes": int(len(detail)),
        "mean_selected_patient_n": float(detail["selected_patient_n"].mean()),
        "median_selected_patient_n": float(detail["selected_patient_n"].median()),
        "fraction_gt_random_median": float(detail["target_gt_random_median"].mean()),
        "fraction_gt_random_p90": float(detail["target_gt_random_p90"].mean()),
        "median_target_abs_neglog10_p": float(detail["target_abs_neglog10_p"].median()),
        "median_random_median_abs_neglog10_p": float(detail["random_median_abs_neglog10_p"].median()),
        "median_target_minus_random_median": float(detail["target_minus_random_median"].median()),
    }])
    return detail, summary


def _plot_patient_subset(summary: pd.DataFrame, detail: pd.DataFrame, out_png: Path) -> None:
    if summary.empty or detail.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    colors = {"LSPIN": "#3b6ea8", "Concrete": "#cc7a29"}
    order = [x for x in ["LSPIN", "Concrete"] if x in set(summary["family"])]
    x = np.arange(len(order))
    frac = [float(summary.loc[summary["family"] == fam, "fraction_gt_random_median"].iloc[0]) for fam in order]
    axes[0].bar(x, frac, color=[colors.get(fam, "0.4") for fam in order], edgecolor="0.2")
    axes[0].axhline(0.5, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_ylabel("Fraction of gated genes > random median")
    axes[0].set_ylim(0, 1)
    parts = axes[1].violinplot(
        [detail.loc[detail["family"] == fam, "target_minus_random_median"].dropna().to_numpy() for fam in order],
        positions=x,
        showmeans=False,
        showmedians=True,
        widths=0.75,
    )
    for body, fam in zip(parts["bodies"], order):
        body.set_facecolor(colors.get(fam, "0.4"))
        body.set_edgecolor("0.2")
        body.set_alpha(0.55)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in parts:
            parts[key].set_color("0.2")
            parts[key].set_linewidth(1.0)
    axes[1].axhline(0, color="0.25", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("Gated gene Cox signal minus random median")
    for ax in axes:
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Within selected patients, are gated genes more predictive than random genes?", y=1.03)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    outdir = args.outdir or args.results_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print("[info] loading PanCan data", flush=True)
    data = sds.load_pancan_split_artifacts(args.data_dir)
    print("[info] reading adaptive rows", flush=True)
    runs = _read_partial_rows(args.results_dir)
    print("[info] computing univariate Cox score table", flush=True)
    univariate_df = _compute_univariate_table(
        data,
        scope=args.univariate_scope,
        min_events=args.min_events,
    )
    histologies = _eligible_histologies(univariate_df, args.min_histology_n, args.min_events)
    print(f"[info] eligible tumor types={len(histologies)}: {', '.join(histologies)}", flush=True)

    summaries = []
    assignment_frames = []
    subset_detail_frames = []
    subset_summary_frames = []
    rng = np.random.default_rng(args.seed)
    for family in args.families:
        print(f"[info] aggregating gates for {family} {args.selection}", flush=True)
        cfg = _selected_config(args.results_dir, family, args.selection, runs)
        saved = _matching_saved_runs(args.results_dir, cfg, runs, model_scope=args.model_scope)
        gate_df, model_df = _aggregate_gate_rates(
            cfg,
            saved,
            data,
            device=args.device,
            hard_threshold=args.hard_threshold,
        )
        gate_df = gate_df[gate_df["histology"].astype(str).isin(histologies)].copy()
        gate, signal, gene_idx, genes = _make_matrices(gate_df, univariate_df, histologies)
        model_df.to_csv(outdir / f"pancan_{family.lower()}_{args.selection}_univariate_bias_models_used.csv", index=False)
        gate_df.to_csv(outdir / f"pancan_{family.lower()}_{args.selection}_univariate_bias_gate_rates.csv", index=False)

        print(f"[info] testing patient-subset predictivity for {family} {args.selection}", flush=True)
        sample_gate_rate = _aggregate_sample_gate_matrix(
            cfg,
            saved,
            data,
            device=args.device,
            hard_threshold=args.hard_threshold,
        )
        subset_detail, subset_summary = _patient_subset_predictivity_test(
            family=family,
            sample_gate_rate=sample_gate_rate,
            gate_df=gate_df,
            data=data,
            rng=rng,
            min_gate_rate=args.subset_min_gate_rate,
            min_specificity=args.subset_min_specificity,
            min_selected_n=args.subset_min_selected_n,
            top_genes=args.subset_top_genes,
            n_random_genes=args.n_random_genes,
            min_events=args.min_events,
        )
        subset_detail_frames.append(subset_detail)
        subset_summary_frames.append(subset_summary)

        for min_rate in args.min_gate_rates:
            for min_spec in args.min_specificities:
                assignments, summary = _evaluate_threshold(
                    gate=gate,
                    signal=signal,
                    gene_idx=gene_idx,
                    genes=genes,
                    histologies=histologies,
                    min_gate_rate=min_rate,
                    min_specificity=min_spec,
                    n_permutations=args.n_permutations,
                    rng=rng,
                )
                summary.update({
                    "family": family,
                    "selection": args.selection,
                    "min_gate_rate": min_rate,
                    "min_specificity": min_spec,
                    "n_gate_models": int(gate_df["n_gate_models"].max()) if len(gate_df) else 0,
                    "univariate_scope": args.univariate_scope,
                })
                summaries.append(summary)
                if not assignments.empty:
                    assignments["family"] = family
                    assignments["selection"] = args.selection
                    assignments["min_gate_rate"] = min_rate
                    assignments["min_specificity"] = min_spec
                    assignment_frames.append(assignments)

    summary_df = pd.DataFrame(summaries)
    assignments_df = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    subset_detail_df = pd.concat(subset_detail_frames, ignore_index=True) if subset_detail_frames else pd.DataFrame()
    subset_summary_df = pd.concat(subset_summary_frames, ignore_index=True) if subset_summary_frames else pd.DataFrame()

    prefix = f"pancan_{args.selection}_gated_univariate_bias"
    summary_csv = outdir / f"{prefix}_summary.csv"
    assignments_csv = outdir / f"{prefix}_assignments.csv"
    univariate_csv = outdir / f"{prefix}_cox_score_all_genes.csv"
    subset_detail_csv = outdir / f"{prefix}_patient_subset_predictivity.csv"
    subset_summary_csv = outdir / f"{prefix}_patient_subset_predictivity_summary.csv"
    fig_png = outdir / f"fig_{prefix}.png"
    subset_png = outdir / f"fig_{prefix}_patient_subset_predictivity.png"
    summary_df.to_csv(summary_csv, index=False)
    assignments_df.to_csv(assignments_csv, index=False)
    univariate_df.to_csv(univariate_csv, index=False)
    subset_detail_df.to_csv(subset_detail_csv, index=False)
    subset_summary_df.to_csv(subset_summary_csv, index=False)
    _plot_summary(summary_df, fig_png)
    _plot_patient_subset(subset_summary_df, subset_detail_df, subset_png)

    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {assignments_csv}", flush=True)
    print(f"[done] wrote {univariate_csv}", flush=True)
    print(f"[done] wrote {subset_detail_csv}", flush=True)
    print(f"[done] wrote {subset_summary_csv}", flush=True)
    print(f"[done] wrote {fig_png}", flush=True)
    print(f"[done] wrote {subset_png}", flush=True)


if __name__ == "__main__":
    main()
