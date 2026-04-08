#!/usr/bin/env python3
"""Plot histology-specific univariate survival signal for gated KIPAN genes.

This script is intentionally post hoc: it does not retrain models. It loads
saved KIPAN adaptive checkpoints, extracts held-out hard gates, selects genes
whose gates are enriched in a histology, and recomputes univariate Cox models
from the split artifacts for the displayed genes.
"""
from __future__ import annotations

import argparse
import re
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ANALYSES_DIR = Path(__file__).resolve().parent
SDS_SRC = Path("/banach2/wes/lspin-repos/sparsedeepsurv/src")

for path in [SDS_SRC]:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, chi2, mannwhitneyu, spearmanr, wilcoxon

import sparsedeepsurv as sds


KIPAN_RUN_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219"
)
KIPAN_DATA_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/"
    "kipan_20260209_213604"
)

HISTO_LABELS = {
    "kidneyclearcellrenalcarcinoma": "Clear cell",
    "kidneypapillaryrenalcellcarcinoma": "Papillary",
    "kidneychromophobe": "Chromophobe",
}


@dataclass(frozen=True)
class SelectedConfig:
    family: str
    selection: str
    gate_type: str
    gate_sigma: float
    lambda_sparse: float
    lambda_sample_smooth: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a KIPAN supplementary dotplot linking histology-specific "
            "gating enrichment to univariate Cox signal."
        )
    )
    p.add_argument("--results-dir", type=Path, default=KIPAN_RUN_DEFAULT)
    p.add_argument("--data-dir", type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--family", choices=["LSPIN", "Concrete"], default="LSPIN")
    p.add_argument("--selection", choices=["smooth", "nosmooth"], default="smooth")
    p.add_argument(
        "--model-scope",
        choices=["available", "representative"],
        default="available",
        help="Use all saved checkpoints for the selected config, or only the heatmap representative.",
    )
    p.add_argument(
        "--univariate-scope",
        choices=["all", "train", "test"],
        default="all",
        help="Samples used to fit the per-histology univariate Cox models.",
    )
    p.add_argument("--top-genes-per-histology", type=int, default=10)
    p.add_argument("--min-histology-n", type=int, default=20)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument(
        "--min-gate-rate",
        type=float,
        default=0.30,
        help="Minimum mean within-histology gate rate for selected genes.",
    )
    p.add_argument(
        "--min-specificity",
        type=float,
        default=0.20,
        help="Minimum excess gate rate over the next-most-gated histology.",
    )
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--fig-width", type=float, default=7.2)
    p.add_argument("--fig-height-per-gene", type=float, default=0.24)
    p.add_argument("--max-color-score", type=float, default=12.0)
    p.add_argument("--max-scatter-genes-per-histology", type=int, default=6000)
    p.add_argument("--gene-set-gmt", type=Path, default=None)
    p.add_argument(
        "--gene-set-library",
        type=str,
        default="MSigDB_Hallmark_2020",
        help=(
            "Enrichr/gseapy library to download and cache when --gene-set-gmt is not provided. "
            "Use 'none' to skip curated enrichment."
        ),
    )
    p.add_argument(
        "--gene-set-cache-dir",
        type=Path,
        default=Path("/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/gene_sets/enrichr"),
        help="Project-local cache for downloaded Enrichr/gseapy gene-set libraries.",
    )
    p.add_argument("--n-ablation-random", type=int, default=100)
    p.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip inference-time ablation and preserve existing ablation outputs if present.",
    )
    p.add_argument("--ablation-seed", type=int, default=20260406)
    return p.parse_args()


def _hlabel(x: str) -> str:
    return HISTO_LABELS.get(str(x), str(x))


def _read_partial_rows(results_dir: Path) -> pd.DataFrame:
    dfs = []
    for p in sorted(results_dir.glob("_partial_p*_worker*/run_rows.csv")):
        df = pd.read_csv(p)
        df["partial_dir"] = str(p.parent)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No partial run_rows.csv files found under {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def _close(series: pd.Series, value: float, tol: float = 1e-9) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce").astype(float), float(value), atol=tol)


def _selected_config(results_dir: Path, family: str, selection: str, runs: pd.DataFrame) -> SelectedConfig:
    cfg_path = results_dir / "selected_comparison_configs.csv"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing selected comparison config file: {cfg_path}")
    selected = pd.read_csv(cfg_path)
    row = selected[
        (selected["family"].astype(str) == family)
        & (selected["selection"].astype(str) == selection)
    ]
    if row.empty:
        raise ValueError(f"No selected row for family={family!r}, selection={selection!r}")
    r = row.iloc[0]

    matches = runs[
        (runs["model_family"].replace({"HardSigmoid": "LSPIN"}).astype(str) == family)
        & _close(runs["gate_sigma"], float(r["gate_sigma"]))
        & _close(runs["lambda_sparse"], float(r["lambda_sparse"]))
        & _close(runs["lambda_sample_smooth"], float(r["lambda_sample_smooth"]))
    ]
    if matches.empty:
        raise ValueError("Selected config was not found in partial run rows.")
    gate_type = str(matches["gate_type"].iloc[0])
    return SelectedConfig(
        family=family,
        selection=selection,
        gate_type=gate_type,
        gate_sigma=float(r["gate_sigma"]),
        lambda_sparse=float(r["lambda_sparse"]),
        lambda_sample_smooth=float(r["lambda_sample_smooth"]),
    )


def _state_path_for_row(row: pd.Series) -> Path:
    partial = Path(str(row["partial_dir"]))
    return partial / "state_dicts" / f"cfg{int(row['global_cfg_idx'])}_rep{int(row['rep_id'])}.pt"


def _matching_saved_runs(
    results_dir: Path,
    cfg: SelectedConfig,
    runs: pd.DataFrame,
    *,
    model_scope: str,
) -> pd.DataFrame:
    rows = runs[
        (runs["model_family"].replace({"HardSigmoid": "LSPIN"}).astype(str) == cfg.family)
        & _close(runs["gate_sigma"], cfg.gate_sigma)
        & _close(runs["lambda_sparse"], cfg.lambda_sparse)
        & _close(runs["lambda_sample_smooth"], cfg.lambda_sample_smooth)
    ].copy()
    if rows.empty:
        raise ValueError(f"No run rows matched selected config: {cfg}")

    if model_scope == "representative":
        heatmap_path = results_dir / "heatmap_models_summary.csv"
        heatmap = pd.read_csv(heatmap_path)
        rep = heatmap[
            (heatmap["family"].astype(str) == cfg.family)
            & (heatmap["selection"].astype(str) == cfg.selection)
        ]
        if rep.empty:
            raise ValueError(f"No representative model found in {heatmap_path}")
        seed = int(rep["representative_seed"].iloc[0])
        rows = rows[rows["seed"].astype(int) == seed].copy()
        if rows.empty:
            raise ValueError(f"Representative seed {seed} was not found in partial run rows.")

    rows["state_dict_path"] = rows.apply(_state_path_for_row, axis=1).astype(str)
    rows["state_dict_exists"] = rows["state_dict_path"].map(lambda x: Path(x).exists())
    return rows[rows["state_dict_exists"]].copy()


def _make_model(cfg: SelectedConfig, input_dim: int) -> torch.nn.Module:
    temperature = 0.3 if cfg.gate_type == "concrete" else 0.5
    concrete_mode = "ste" if cfg.gate_type == "concrete" else "relaxed"
    return sds.make_model(
        input_dim=input_dim,
        gate_type=cfg.gate_type,
        gate_sigma=cfg.gate_sigma,
        temperature=temperature,
        concrete_mode=concrete_mode,
    )


def _aggregate_gate_rates(
    cfg: SelectedConfig,
    saved_runs: pd.DataFrame,
    data: dict,
    *,
    device: str,
    hard_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    histo_test = np.asarray(data["histo_test"]).astype(str)
    histologies = sorted(pd.unique(histo_test), key=lambda x: _hlabel(x))
    gene_names = np.asarray(data["gene_names"]).astype(str)
    input_dim = X_test.shape[1]

    accum = {h: np.zeros(input_dim, dtype=np.float64) for h in histologies}
    n_models = 0
    model_rows = []

    for _, row in saved_runs.sort_values(["phase", "rep_id", "seed"]).iterrows():
        model = _make_model(cfg, input_dim=input_dim)
        state_path = Path(str(row["state_dict_path"]))
        try:
            sd = torch.load(state_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(state_path, map_location=device)
        model.load_state_dict(sd)
        model.to(device)
        _, _, hard, _ = sds.get_gates(
            model,
            X_test,
            device=device,
            hard_threshold=hard_threshold,
            batch_size=512,
        )
        G = hard.numpy().astype(np.float32)
        for h in histologies:
            mask = histo_test == h
            accum[h] += G[mask].mean(axis=0)
        n_models += 1
        model_rows.append({
            "seed": int(row["seed"]),
            "phase": int(row["phase"]),
            "rep_id": int(row["rep_id"]),
            "state_dict_path": str(state_path),
            "test_cindex": float(row.get("test_cindex", np.nan)),
            "mean_Khard": float(row.get("mean_Khard", np.nan)),
        })

    if n_models == 0:
        raise RuntimeError("No saved checkpoints were available for this selected config.")

    records = []
    rate_matrix = {h: accum[h] / n_models for h in histologies}
    for gene_idx, gene in enumerate(gene_names):
        vals = np.array([rate_matrix[h][gene_idx] for h in histologies], dtype=float)
        for j, h in enumerate(histologies):
            other_max = float(np.delete(vals, j).max()) if len(vals) > 1 else 0.0
            records.append({
                "histology": h,
                "histology_label": _hlabel(h),
                "gene": gene,
                "gene_idx": gene_idx,
                "gate_rate": float(vals[j]),
                "other_histology_max_gate_rate": other_max,
                "gate_specificity": float(vals[j] - other_max),
                "n_gate_models": n_models,
            })
    return pd.DataFrame(records), pd.DataFrame(model_rows)


def _select_genes(
    gate_df: pd.DataFrame,
    *,
    top_genes_per_histology: int,
    min_gate_rate: float,
    min_specificity: float,
    min_histo_n: int,
    histo_counts: dict[str, int],
) -> pd.DataFrame:
    picks = []
    for histology, sub in gate_df.groupby("histology", sort=False):
        if histo_counts.get(histology, 0) < min_histo_n:
            continue
        cand = sub[
            (sub["gate_rate"] >= min_gate_rate)
            & (sub["gate_specificity"] >= min_specificity)
        ].copy()
        cand = cand.sort_values(
            ["gate_specificity", "gate_rate"],
            ascending=[False, False],
        ).head(top_genes_per_histology)
        cand["assigned_histology"] = histology
        cand["assigned_histology_label"] = _hlabel(histology)
        cand["rank_within_histology"] = np.arange(1, len(cand) + 1)
        picks.append(cand)
    if not picks:
        raise RuntimeError("No histology-specific gated genes passed the selection thresholds.")
    return pd.concat(picks, ignore_index=True)


def _univariate_arrays(data: dict, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scope == "train":
        return (
            np.asarray(data["X_train"]),
            np.asarray(data["time_train"]),
            np.asarray(data["event_train"]),
            np.asarray(data["histo_train"]).astype(str),
        )
    if scope == "test":
        return (
            np.asarray(data["X_test"]),
            np.asarray(data["time_test"]),
            np.asarray(data["event_test"]),
            np.asarray(data["histo_test"]).astype(str),
        )
    return (
        np.vstack([data["X_train"], data["X_test"]]),
        np.concatenate([data["time_train"], data["time_test"]]),
        np.concatenate([data["event_train"], data["event_test"]]),
        np.concatenate([data["histo_train"], data["histo_test"]]).astype(str),
    )


def _cox_score_test_matrix(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    min_events: int,
) -> dict[str, np.ndarray | str]:
    """Vectorized one-covariate Cox score test at beta=0 for all columns."""
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    ok_rows = np.isfinite(time) & np.isfinite(event)
    X = X[ok_rows]
    time = time[ok_rows]
    event = event[ok_rows]

    n, p = X.shape
    if n < 10 or int(event.sum()) < min_events:
        nan = np.full(p, np.nan, dtype=float)
        return {"coef": nan, "p": nan, "signed_neglog10_p": nan, "status": "skipped_low_information"}

    # Risk set for a subject at time t contains samples with time >= t.
    order = np.argsort(-time, kind="mergesort")
    Xs = X[order]
    es = event[order].astype(bool)
    risk_n = np.arange(1, n + 1, dtype=np.float64)
    csum = np.cumsum(Xs, axis=0)
    csum2 = np.cumsum(Xs * Xs, axis=0)

    mean = csum / risk_n[:, None]
    var = np.maximum(csum2 / risk_n[:, None] - mean * mean, 0.0)
    U = ((Xs - mean) * es[:, None]).sum(axis=0)
    I = (var * es[:, None]).sum(axis=0)

    ok_cols = np.isfinite(U) & np.isfinite(I) & (I > 1e-12) & (np.nanstd(Xs, axis=0) > 1e-10)
    coef = np.full(p, np.nan, dtype=float)
    pvals = np.full(p, np.nan, dtype=float)
    signed = np.full(p, np.nan, dtype=float)

    z = np.zeros(p, dtype=float)
    z[ok_cols] = U[ok_cols] / np.sqrt(I[ok_cols])
    pvals[ok_cols] = chi2.sf(z[ok_cols] * z[ok_cols], df=1)
    coef[ok_cols] = U[ok_cols] / I[ok_cols]
    p_safe = np.maximum(pvals[ok_cols], np.finfo(float).tiny)
    signed[ok_cols] = np.sign(coef[ok_cols]) * -np.log10(p_safe)
    return {"coef": coef, "p": pvals, "signed_neglog10_p": signed, "status": "ok"}


def _compute_univariate_table(
    data: dict,
    *,
    scope: str,
    min_events: int,
) -> pd.DataFrame:
    X, time, event, histo = _univariate_arrays(data, scope)
    gene_names = np.asarray(data["gene_names"]).astype(str)
    rows = []
    for histology in sorted(pd.unique(histo), key=lambda x: _hlabel(str(x))):
        mask = histo == histology
        out = _cox_score_test_matrix(X[mask], time[mask], event[mask], min_events=min_events)
        for gene_idx, gene in enumerate(gene_names):
            pval = float(out["p"][gene_idx]) if np.ndim(out["p"]) else np.nan
            signed = float(out["signed_neglog10_p"][gene_idx]) if np.ndim(out["signed_neglog10_p"]) else np.nan
            rows.append({
                "histology": str(histology),
                "histology_label": _hlabel(str(histology)),
                "gene": gene,
                "gene_idx": gene_idx,
                "cox_score_coef": float(out["coef"][gene_idx]) if np.ndim(out["coef"]) else np.nan,
                "cox_score_p": pval,
                "signed_neglog10_p": signed,
                "abs_neglog10_p": abs(signed) if np.isfinite(signed) else np.nan,
                "univariate_scope": scope,
                "univariate_n": int(mask.sum()),
                "univariate_events": int(event[mask].sum()),
                "univariate_status": str(out["status"]),
            })
    return pd.DataFrame(rows)


def _attach_univariate(
    plot_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "histology", "gene_idx", "cox_score_coef", "cox_score_p",
        "signed_neglog10_p", "abs_neglog10_p", "univariate_scope",
        "univariate_n", "univariate_events", "univariate_status",
    ]
    return plot_df.merge(univariate_df[cols], on=["histology", "gene_idx"], how="left")


def _plot_dotplot(
    plot_df: pd.DataFrame,
    selected_genes: pd.DataFrame,
    *,
    out_png: Path,
    fig_width: float,
    fig_height_per_gene: float,
    max_color_score: float,
) -> None:
    histologies = (
        plot_df[["histology", "histology_label"]]
        .drop_duplicates()
        .sort_values("histology_label")
    )
    x_map = {h: i for i, h in enumerate(histologies["histology"])}

    gene_order = (
        selected_genes[["gene", "assigned_histology_label", "rank_within_histology"]]
        .drop_duplicates("gene")
        .sort_values(["assigned_histology_label", "rank_within_histology", "gene"])
    )
    genes = gene_order["gene"].tolist()
    y_map = {g: i for i, g in enumerate(reversed(genes))}

    df = plot_df[plot_df["gene"].isin(genes)].copy()
    df["x"] = df["histology"].map(x_map)
    df["y"] = df["gene"].map(y_map)
    df["positive_specificity"] = df["gate_specificity"].clip(lower=0)
    size = 28 + 760 * df["positive_specificity"].clip(upper=0.6)
    color = df["signed_neglog10_p"].clip(-max_color_score, max_color_score)

    height = max(4.5, 1.4 + fig_height_per_gene * len(genes))
    fig, ax = plt.subplots(figsize=(fig_width, height))
    sc = ax.scatter(
        df["x"],
        df["y"],
        s=size,
        c=color,
        cmap="coolwarm",
        vmin=-max_color_score,
        vmax=max_color_score,
        edgecolor="0.25",
        linewidth=0.35,
        alpha=0.88,
    )

    ax.set_xticks(range(len(histologies)))
    ax.set_xticklabels(histologies["histology_label"], rotation=25, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(list(reversed(genes)), fontsize=8)
    ax.set_xlim(-0.55, len(histologies) - 0.45)
    ax.set_ylim(-0.6, len(genes) - 0.4)
    ax.grid(axis="both", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Histology")
    ax.set_ylabel("Genes frequently and specifically gated in a histology")
    ax.set_title("KIPAN LSPIN smooth gates highlight histology-linked survival genes")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Univariate Cox signal: signed -log10(p)")

    for val in [0.05, 0.15, 0.30]:
        ax.scatter([], [], s=28 + 760 * val, color="0.75", edgecolor="0.25", label=f"{val:.2f}")
    leg = ax.legend(
        title="Gate specificity",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.add_artist(leg)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _matched_histology_test(
    selected: pd.DataFrame,
    gate_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, sel in selected.iterrows():
        gene_idx = int(sel["gene_idx"])
        gene = str(sel["gene"])
        assigned = str(sel["assigned_histology"])
        uni = univariate_df[univariate_df["gene_idx"].astype(int) == gene_idx].copy()
        gates = gate_df[gate_df["gene_idx"].astype(int) == gene_idx][
            ["histology", "gate_rate", "gate_specificity"]
        ]
        uni = uni.merge(gates, on="histology", how="left")
        assigned_row = uni[uni["histology"].astype(str) == assigned]
        other = uni[uni["histology"].astype(str) != assigned]
        if assigned_row.empty or other.empty:
            continue
        ar = assigned_row.iloc[0]
        other_abs = pd.to_numeric(other["abs_neglog10_p"], errors="coerce")
        other_signed = pd.to_numeric(other["signed_neglog10_p"], errors="coerce")
        assigned_abs = float(ar["abs_neglog10_p"])
        assigned_signed = float(ar["signed_neglog10_p"])
        max_other_abs = float(other_abs.max()) if other_abs.notna().any() else np.nan
        matched_rank = int(
            1 + (pd.to_numeric(uni["abs_neglog10_p"], errors="coerce") > assigned_abs).sum()
        ) if np.isfinite(assigned_abs) else -1
        rows.append({
            "gene": gene,
            "gene_idx": gene_idx,
            "assigned_histology": assigned,
            "assigned_histology_label": _hlabel(assigned),
            "assigned_gate_rate": float(ar["gate_rate"]),
            "assigned_gate_specificity": float(ar["gate_specificity"]),
            "assigned_abs_neglog10_p": assigned_abs,
            "assigned_signed_neglog10_p": assigned_signed,
            "max_other_abs_neglog10_p": max_other_abs,
            "mean_other_abs_neglog10_p": float(other_abs.mean()),
            "max_other_signed_neglog10_p": float(other_signed.iloc[other_abs.argmax()]) if other_abs.notna().any() else np.nan,
            "matched_minus_max_other_abs": assigned_abs - max_other_abs,
            "matched_minus_mean_other_abs": assigned_abs - float(other_abs.mean()),
            "matched_is_top_abs": bool(np.isfinite(assigned_abs) and assigned_abs >= max_other_abs),
            "matched_rank_abs": matched_rank,
        })
    matched = pd.DataFrame(rows)
    if matched.empty:
        return matched, pd.DataFrame()

    delta = pd.to_numeric(matched["matched_minus_max_other_abs"], errors="coerce").dropna()
    n_top = int(matched["matched_is_top_abs"].sum())
    n = int(matched["matched_is_top_abs"].notna().sum())
    try:
        sign_p = float(binomtest(n_top, n=n, p=1 / 3, alternative="greater").pvalue) if n else np.nan
    except Exception:
        sign_p = np.nan
    try:
        wil_p = float(wilcoxon(delta, alternative="greater").pvalue) if len(delta) >= 3 else np.nan
    except Exception:
        wil_p = np.nan
    summary = pd.DataFrame([{
        "n_selected_genes": n,
        "n_matched_histology_top_abs_signal": n_top,
        "fraction_matched_top_abs_signal": n_top / n if n else np.nan,
        "binomial_p_vs_one_third": sign_p,
        "median_matched_minus_max_other_abs": float(delta.median()) if len(delta) else np.nan,
        "mean_matched_minus_max_other_abs": float(delta.mean()) if len(delta) else np.nan,
        "wilcoxon_p_delta_gt_zero": wil_p,
    }])
    return matched, summary


def _plot_matched_histology(matched: pd.DataFrame, out_png: Path) -> None:
    if matched.empty:
        return
    df = matched.sort_values(
        ["assigned_histology_label", "matched_minus_max_other_abs", "gene"],
        ascending=[True, False, True],
    )
    y = np.arange(len(df))
    colors = {
        label: color
        for label, color in zip(sorted(df["assigned_histology_label"].unique()), plt.cm.Set2.colors)
    }
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.25 * len(df) + 1.5)))
    ax.barh(
        y,
        df["matched_minus_max_other_abs"],
        color=[colors[x] for x in df["assigned_histology_label"]],
        edgecolor="0.25",
        linewidth=0.4,
    )
    ax.axvline(0, color="0.2", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Matched histology signal minus strongest other histology (-log10 p)")
    ax.set_ylabel("Selected gene")
    ax.set_title("Do histology-specific gates point to the same histology's survival signal?")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor="0.25", label=label)
        for label, c in colors.items()
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _enrichment_summary(
    selected: pd.DataFrame,
    univariate_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for histology in sorted(selected["assigned_histology"].unique(), key=_hlabel):
        selected_genes = set(
            selected.loc[selected["assigned_histology"] == histology, "gene_idx"].astype(int)
        )
        sub = univariate_df[univariate_df["histology"].astype(str) == histology].copy()
        sub["is_selected_for_histology"] = sub["gene_idx"].astype(int).isin(selected_genes)
        sel = pd.to_numeric(sub.loc[sub["is_selected_for_histology"], "abs_neglog10_p"], errors="coerce").dropna()
        bg = pd.to_numeric(sub.loc[~sub["is_selected_for_histology"], "abs_neglog10_p"], errors="coerce").dropna()
        if len(sel) and len(bg):
            mw = mannwhitneyu(sel, bg, alternative="greater")
            rank_biserial = 2 * float(mw.statistic) / (len(sel) * len(bg)) - 1
        else:
            mw = None
            rank_biserial = np.nan
        rows.append({
            "histology": histology,
            "histology_label": _hlabel(histology),
            "n_selected": int(len(sel)),
            "n_background": int(len(bg)),
            "selected_median_abs_neglog10_p": float(sel.median()) if len(sel) else np.nan,
            "background_median_abs_neglog10_p": float(bg.median()) if len(bg) else np.nan,
            "median_difference": float(sel.median() - bg.median()) if len(sel) and len(bg) else np.nan,
            "mannwhitney_p_selected_gt_background": float(mw.pvalue) if mw is not None else np.nan,
            "rank_biserial_effect": rank_biserial,
        })
    return pd.DataFrame(rows)


def _plot_enrichment(
    selected: pd.DataFrame,
    univariate_df: pd.DataFrame,
    out_png: Path,
) -> None:
    pieces = []
    for histology in sorted(selected["assigned_histology"].unique(), key=_hlabel):
        selected_genes = set(
            selected.loc[selected["assigned_histology"] == histology, "gene_idx"].astype(int)
        )
        sub = univariate_df[univariate_df["histology"].astype(str) == histology].copy()
        sub["group"] = np.where(sub["gene_idx"].astype(int).isin(selected_genes), "Selected", "Background")
        # Keep the full selected set and a deterministic background cap for readable plotting.
        bg = sub[sub["group"] == "Background"].sort_values("gene_idx").head(1500)
        pieces.append(pd.concat([sub[sub["group"] == "Selected"], bg], ignore_index=True))
    df = pd.concat(pieces, ignore_index=True)

    histologies = sorted(df["histology"].unique(), key=_hlabel)
    fig, axes = plt.subplots(1, len(histologies), figsize=(3.2 * len(histologies), 4.0), sharey=True)
    if len(histologies) == 1:
        axes = [axes]
    for ax, histology in zip(axes, histologies):
        sub = df[df["histology"] == histology]
        vals = [
            sub.loc[sub["group"] == "Background", "abs_neglog10_p"].dropna().values,
            sub.loc[sub["group"] == "Selected", "abs_neglog10_p"].dropna().values,
        ]
        ax.boxplot(vals, tick_labels=["Background", "Selected"], showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#d9e6ee", "edgecolor": "0.25"},
                   medianprops={"color": "0.15", "linewidth": 1.4})
        ax.scatter(
            np.full(len(vals[1]), 2.0) + np.linspace(-0.06, 0.06, max(len(vals[1]), 1))[:len(vals[1])],
            vals[1],
            s=22,
            color="#d95f02",
            edgecolor="0.25",
            linewidth=0.3,
            zorder=3,
        )
        ax.set_title(_hlabel(histology))
        ax.set_xlabel("")
        ax.grid(axis="y", color="0.9")
    axes[0].set_ylabel("Univariate Cox score-test signal, -log10(p)")
    fig.suptitle("Selected histology-gated genes vs background survival signal", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _scatter_table(
    gate_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = gate_df.merge(
        univariate_df[["histology", "gene_idx", "abs_neglog10_p", "signed_neglog10_p"]],
        on=["histology", "gene_idx"],
        how="left",
    )
    selected_pairs = set(zip(selected["assigned_histology"].astype(str), selected["gene_idx"].astype(int)))
    df["selected_for_histology"] = [
        (str(h), int(g)) in selected_pairs for h, g in zip(df["histology"], df["gene_idx"])
    ]

    rows = []
    for histology, sub in df.groupby("histology", sort=False):
        ok = sub[["gate_rate", "gate_specificity", "abs_neglog10_p"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ok) >= 3:
            rho_rate, p_rate = spearmanr(ok["gate_rate"], ok["abs_neglog10_p"])
            rho_spec, p_spec = spearmanr(ok["gate_specificity"], ok["abs_neglog10_p"])
        else:
            rho_rate = p_rate = rho_spec = p_spec = np.nan
        rows.append({
            "histology": histology,
            "histology_label": _hlabel(histology),
            "n_genes": int(len(ok)),
            "spearman_gate_rate_vs_abs_signal": float(rho_rate),
            "spearman_gate_rate_p": float(p_rate),
            "spearman_specificity_vs_abs_signal": float(rho_spec),
            "spearman_specificity_p": float(p_spec),
        })
    return df, pd.DataFrame(rows)


def _plot_scatter(
    scatter_df: pd.DataFrame,
    out_png: Path,
    *,
    max_points_per_histo: int,
) -> None:
    histologies = sorted(scatter_df["histology"].unique(), key=_hlabel)
    fig, axes = plt.subplots(1, len(histologies), figsize=(4.0 * len(histologies), 3.6), sharey=True)
    if len(histologies) == 1:
        axes = [axes]
    for ax, histology in zip(axes, histologies):
        sub = scatter_df[scatter_df["histology"] == histology].replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(subset=["gate_specificity", "abs_neglog10_p"])
        bg = sub[~sub["selected_for_histology"]].sort_values("gene_idx").head(max_points_per_histo)
        sel = sub[sub["selected_for_histology"]]
        ax.scatter(bg["gate_specificity"], bg["abs_neglog10_p"], s=5, color="0.7", alpha=0.25, linewidths=0)
        ax.scatter(sel["gate_specificity"], sel["abs_neglog10_p"], s=42, color="#d95f02",
                   edgecolor="0.2", linewidth=0.4)
        for _, row in sel.iterrows():
            ax.text(row["gate_specificity"] + 0.01, row["abs_neglog10_p"], str(row["gene"]), fontsize=6)
        ax.axvline(0, color="0.85", linewidth=1)
        ax.set_title(_hlabel(histology))
        ax.set_xlabel("Gate specificity")
        ax.grid(color="0.92")
    axes[0].set_ylabel("Univariate Cox score-test signal, -log10(p)")
    fig.suptitle("Gate specificity vs histology-specific univariate survival signal", y=1.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _expression_enrichment(
    selected: pd.DataFrame,
    data: dict,
    *,
    scope: str,
) -> pd.DataFrame:
    X, _, _, histo = _univariate_arrays(data, scope)
    rows = []
    for _, row in selected.iterrows():
        gene_idx = int(row["gene_idx"])
        histology = str(row["assigned_histology"])
        in_mask = histo == histology
        out_mask = ~in_mask
        xin = X[in_mask, gene_idx].astype(float)
        xout = X[out_mask, gene_idx].astype(float)
        pooled = np.sqrt((np.nanvar(xin, ddof=1) + np.nanvar(xout, ddof=1)) / 2)
        rows.append({
            "gene": str(row["gene"]),
            "gene_idx": gene_idx,
            "assigned_histology": histology,
            "assigned_histology_label": _hlabel(histology),
            "mean_expression_assigned": float(np.nanmean(xin)),
            "mean_expression_other": float(np.nanmean(xout)),
            "expression_mean_difference": float(np.nanmean(xin) - np.nanmean(xout)),
            "expression_standardized_difference": float((np.nanmean(xin) - np.nanmean(xout)) / (pooled + 1e-12)),
            "expression_scope": scope,
        })
    return pd.DataFrame(rows)


def _plot_expression(expr: pd.DataFrame, out_png: Path) -> None:
    if expr.empty:
        return
    df = expr.sort_values(
        ["assigned_histology_label", "expression_standardized_difference", "gene"],
        ascending=[True, False, True],
    )
    colors = {
        label: color
        for label, color in zip(sorted(df["assigned_histology_label"].unique()), plt.cm.Set2.colors)
    }
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 0.25 * len(df) + 1.5)))
    ax.barh(
        y,
        df["expression_standardized_difference"],
        color=[colors[x] for x in df["assigned_histology_label"]],
        edgecolor="0.25",
        linewidth=0.4,
    )
    ax.axvline(0, color="0.2", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Expression enrichment in gated histology vs others, standardized mean difference")
    ax.set_ylabel("Selected gene")
    ax.set_title("Are histology-gated genes expression-enriched in the same histology?")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _score_vector_for_histology(
    score: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    min_events: int,
) -> dict:
    out = _cox_score_test_matrix(score.reshape(-1, 1), time, event, min_events=min_events)
    signed = float(out["signed_neglog10_p"][0]) if np.ndim(out["signed_neglog10_p"]) else np.nan
    return {
        "cox_score_coef": float(out["coef"][0]) if np.ndim(out["coef"]) else np.nan,
        "cox_score_p": float(out["p"][0]) if np.ndim(out["p"]) else np.nan,
        "signed_neglog10_p": signed,
        "abs_neglog10_p": abs(signed) if np.isfinite(signed) else np.nan,
        "status": str(out["status"]),
    }


def _module_survival(
    selected: pd.DataFrame,
    data: dict,
    *,
    scope: str,
    min_events: int,
) -> pd.DataFrame:
    X, time, event, histo = _univariate_arrays(data, scope)
    rows = []
    for histology, sub in selected.groupby("assigned_histology", sort=False):
        gene_idx = sub["gene_idx"].astype(int).to_numpy()
        mask = histo == histology
        if len(gene_idx) == 0:
            continue
        module_score = X[mask][:, gene_idx].mean(axis=1)
        out = _score_vector_for_histology(
            module_score,
            time[mask],
            event[mask],
            min_events=min_events,
        )
        out.update({
            "histology": histology,
            "histology_label": _hlabel(histology),
            "n_module_genes": int(len(gene_idx)),
            "module_genes": ";".join(sub["gene"].astype(str).tolist()),
            "module_scope": scope,
            "n": int(mask.sum()),
            "events": int(event[mask].sum()),
        })
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_module_survival(module_df: pd.DataFrame, out_png: Path) -> None:
    if module_df.empty:
        return
    df = module_df.sort_values("histology_label")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = ["#1b9e77" if x >= 0 else "#7570b3" for x in df["signed_neglog10_p"]]
    ax.bar(df["histology_label"], df["abs_neglog10_p"], color=colors, edgecolor="0.25", linewidth=0.5)
    for i, row in enumerate(df.itertuples(index=False)):
        direction = "risk" if row.signed_neglog10_p >= 0 else "protective"
        ax.text(i, row.abs_neglog10_p + 0.05, direction, ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_ylabel("Module Cox score-test signal, -log10(p)")
    ax.set_xlabel("Gated histology module")
    ax.set_title("Survival signal of histology-specific gated gene modules")
    ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _risk_scores_deterministic(model: torch.nn.Module, X: torch.Tensor, *, device: str) -> np.ndarray:
    model.eval()
    risks = []
    for i in range(0, X.shape[0], 512):
        xb = X[i: i + 512].to(device)
        out = model(xb, deterministic=True)
        risk = out[0] if isinstance(out, (tuple, list)) else out
        risks.append(risk.detach().cpu().numpy())
    return np.concatenate(risks, axis=0).astype(float)


def _safe_cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray, min_events: int) -> float:
    if len(risk) < 10 or int(np.asarray(event).sum()) < min_events:
        return np.nan
    try:
        return float(sds.concordance_index(np.asarray(risk), np.asarray(time), np.asarray(event)))
    except Exception:
        return np.nan


def _run_ablation(
    cfg: SelectedConfig,
    saved_runs: pd.DataFrame,
    selected: pd.DataFrame,
    data: dict,
    *,
    device: str,
    n_random: int,
    seed: int,
    min_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X_test_np = np.asarray(data["X_test"], dtype=np.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    time = np.asarray(data["time_test"])
    event = np.asarray(data["event_test"])
    histo = np.asarray(data["histo_test"]).astype(str)
    input_dim = X_test.shape[1]
    selected_by_histo = {
        str(h): sub[["gene", "gene_idx"]].drop_duplicates()
        for h, sub in selected.groupby("assigned_histology", sort=False)
    }
    all_gene_idx = np.arange(input_dim)
    rows = []

    for _, model_row in saved_runs.sort_values(["phase", "rep_id", "seed"]).iterrows():
        model = _make_model(cfg, input_dim=input_dim).to(device)
        state_path = Path(str(model_row["state_dict_path"]))
        try:
            sd = torch.load(state_path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(state_path, map_location=device)
        model.load_state_dict(sd)

        for histology, genes in selected_by_histo.items():
            mask = histo == histology
            if int(mask.sum()) < 10 or int(event[mask].sum()) < min_events:
                continue
            gene_idx = genes["gene_idx"].astype(int).to_numpy()
            X_sub = X_test_np[mask].copy()
            base_risk = _risk_scores_deterministic(model, torch.tensor(X_sub, dtype=torch.float32), device=device)
            base_c = _safe_cindex(base_risk, time[mask], event[mask], min_events)

            X_abl = X_sub.copy()
            X_abl[:, gene_idx] = 0.0
            abl_risk = _risk_scores_deterministic(model, torch.tensor(X_abl, dtype=torch.float32), device=device)
            abl_c = _safe_cindex(abl_risk, time[mask], event[mask], min_events)
            risk_corr = float(np.corrcoef(base_risk, abl_risk)[0, 1]) if np.std(abl_risk) > 1e-12 else np.nan

            exclude = set(gene_idx.tolist())
            pool = np.array([g for g in all_gene_idx if g not in exclude], dtype=int)
            rand_drops = []
            rand_corrs = []
            for rand_id in range(n_random):
                rand_idx = rng.choice(pool, size=len(gene_idx), replace=False)
                X_rand = X_sub.copy()
                X_rand[:, rand_idx] = 0.0
                rand_risk = _risk_scores_deterministic(model, torch.tensor(X_rand, dtype=torch.float32), device=device)
                rand_c = _safe_cindex(rand_risk, time[mask], event[mask], min_events)
                rand_drops.append(base_c - rand_c if np.isfinite(base_c) and np.isfinite(rand_c) else np.nan)
                rand_corrs.append(float(np.corrcoef(base_risk, rand_risk)[0, 1]) if np.std(rand_risk) > 1e-12 else np.nan)

            obs_drop = base_c - abl_c if np.isfinite(base_c) and np.isfinite(abl_c) else np.nan
            rand_drops_arr = np.asarray(rand_drops, dtype=float)
            rand_corrs_arr = np.asarray(rand_corrs, dtype=float)
            valid = np.isfinite(rand_drops_arr)
            empirical_p = (
                (1 + int(np.sum(rand_drops_arr[valid] >= obs_drop))) / (1 + int(valid.sum()))
                if np.isfinite(obs_drop) and int(valid.sum()) else np.nan
            )
            rows.append({
                "seed": int(model_row["seed"]),
                "phase": int(model_row["phase"]),
                "rep_id": int(model_row["rep_id"]),
                "histology": histology,
                "histology_label": _hlabel(histology),
                "n_test": int(mask.sum()),
                "events": int(event[mask].sum()),
                "n_ablated_genes": int(len(gene_idx)),
                "ablated_genes": ";".join(genes["gene"].astype(str).tolist()),
                "baseline_cindex": base_c,
                "selected_ablation_cindex": abl_c,
                "selected_cindex_drop": obs_drop,
                "selected_risk_corr": risk_corr,
                "random_cindex_drop_mean": float(np.nanmean(rand_drops_arr)),
                "random_cindex_drop_median": float(np.nanmedian(rand_drops_arr)),
                "random_cindex_drop_sd": float(np.nanstd(rand_drops_arr, ddof=1)),
                "random_risk_corr_mean": float(np.nanmean(rand_corrs_arr)),
                "random_risk_corr_median": float(np.nanmedian(rand_corrs_arr)),
                "empirical_p_random_drop_ge_selected": empirical_p,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = detail.groupby(["histology", "histology_label"], as_index=False).agg(
        n_models=("seed", "nunique"),
        selected_cindex_drop_mean=("selected_cindex_drop", "mean"),
        selected_cindex_drop_median=("selected_cindex_drop", "median"),
        random_cindex_drop_mean=("random_cindex_drop_mean", "mean"),
        selected_risk_corr_mean=("selected_risk_corr", "mean"),
        empirical_p_median=("empirical_p_random_drop_ge_selected", "median"),
    )
    return detail, summary


def _plot_ablation(summary: pd.DataFrame, out_png: Path) -> None:
    if summary.empty:
        return
    df = summary.sort_values("histology_label")
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    ax.bar(x - width / 2, df["selected_cindex_drop_mean"], width, label="Selected genes",
           color="#d95f02", edgecolor="0.25", linewidth=0.5)
    ax.bar(x + width / 2, df["random_cindex_drop_mean"], width, label="Random matched genes",
           color="0.72", edgecolor="0.25", linewidth=0.5)
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["histology_label"], rotation=20, ha="right")
    ax.set_ylabel("C-index drop after inference-time masking")
    ax.set_title("Ablating histology-specific gated genes")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_gmt(path: Path) -> dict[str, set[str]]:
    gene_sets = {}
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene_sets[parts[0]] = set(parts[2:])
    return gene_sets


def _safe_slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x).strip("_")


def _write_gmt(gene_sets: dict[str, set[str]], path: Path, *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for name, genes in sorted(gene_sets.items()):
            clean_genes = sorted({str(g).strip() for g in genes if str(g).strip()})
            if not clean_genes:
                continue
            fh.write("\t".join([str(name), source, *clean_genes]) + "\n")


def _load_gene_sets(
    *,
    gmt_path: Path | None,
    library: str | None,
    cache_dir: Path,
) -> tuple[dict[str, set[str]] | None, str, str]:
    """Load curated gene sets from a GMT or download/cache an Enrichr library."""
    if gmt_path is not None:
        if not gmt_path.exists():
            raise FileNotFoundError(f"--gene-set-gmt does not exist: {gmt_path}")
        return _read_gmt(gmt_path), "gmt", str(gmt_path)

    if library is None or str(library).strip().lower() in {"", "none", "skip"}:
        return None, "skipped", "gene-set download disabled"

    cache_path = cache_dir / f"{_safe_slug(str(library))}.gmt"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return _read_gmt(cache_path), "enrichr_cache", str(cache_path)

    try:
        import gseapy as gp
    except Exception as exc:
        return None, "skipped_missing_gseapy", f"gseapy import failed: {exc}"

    try:
        gene_sets = gp.get_library(name=str(library), organism="Human")
    except Exception as exc:
        return None, "download_failed", f"gseapy.get_library({library!r}) failed: {exc}"

    clean = {
        str(name): {str(g).strip() for g in genes if str(g).strip()}
        for name, genes in gene_sets.items()
        if genes
    }
    if not clean:
        return None, "download_failed_empty", f"Downloaded library {library!r} had no gene sets."
    _write_gmt(clean, cache_path, source=f"Enrichr:{library}")
    return clean, "enrichr_download", str(cache_path)


def _curated_gene_set_enrichment(
    selected: pd.DataFrame,
    *,
    gene_sets: dict[str, set[str]] | None,
    gene_set_source: str,
    gene_set_message: str,
    universe: set[str],
) -> pd.DataFrame:
    if not gene_sets:
        return pd.DataFrame([{
            "status": gene_set_source,
            "message": gene_set_message,
        }])
    from scipy.stats import fisher_exact
    rows = []
    for histology, sub in selected.groupby("assigned_histology", sort=False):
        selected_genes = set(sub["gene"].astype(str)) & universe
        for set_name, genes in gene_sets.items():
            genes = set(genes) & universe
            a = len(selected_genes & genes)
            b = len(selected_genes - genes)
            c = len(genes - selected_genes)
            d = len(universe - selected_genes - genes)
            if a == 0:
                continue
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({
                "status": "ok",
                "gene_set_source": gene_set_source,
                "gene_set_message": gene_set_message,
                "histology": histology,
                "histology_label": _hlabel(histology),
                "gene_set": set_name,
                "overlap": a,
                "selected_size": len(selected_genes),
                "gene_set_size": len(genes),
                "odds_ratio": odds,
                "p": p,
                "overlap_genes": ";".join(sorted(selected_genes & genes)),
            })
    if not rows:
        return pd.DataFrame([{
            "status": "ok_no_overlaps",
            "gene_set_source": gene_set_source,
            "gene_set_message": gene_set_message,
        }])
    df = pd.DataFrame(rows).sort_values("p")
    df["rank"] = np.arange(1, len(df) + 1)
    raw_q = (df["p"] * len(df) / df["rank"]).clip(upper=1.0).to_numpy()
    df["q_bh"] = np.minimum.accumulate(raw_q[::-1])[::-1]
    return df


def _plot_curated_enrichment(enrichment: pd.DataFrame, out_png: Path, *, top_n: int = 10) -> None:
    if enrichment.empty or "gene_set" not in enrichment.columns:
        return
    df = enrichment[enrichment.get("status", "") == "ok"].copy()
    if df.empty:
        return
    df["neglog10_p"] = -np.log10(np.maximum(pd.to_numeric(df["p"], errors="coerce"), np.finfo(float).tiny))
    df = df.sort_values(["p", "histology_label", "gene_set"]).head(top_n).copy()
    df["label"] = df["histology_label"].astype(str) + ": " + df["gene_set"].astype(str)
    df = df.iloc[::-1]

    colors = ["#1b9e77" if q < 0.1 else "#7570b3" for q in pd.to_numeric(df["q_bh"], errors="coerce")]
    fig, ax = plt.subplots(figsize=(7.5, max(3.6, 0.35 * len(df) + 1.2)))
    ax.barh(df["label"], df["neglog10_p"], color=colors, edgecolor="0.25", linewidth=0.5)
    ax.axvline(-np.log10(0.05), color="0.35", linestyle="--", linewidth=1.0, label="p = 0.05")
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(
            row.neglog10_p + 0.03,
            i,
            f"q={row.q_bh:.2g}; {row.overlap_genes}",
            va="center",
            fontsize=7,
        )
    ax.set_xlabel("Hallmark enrichment, -log10(Fisher p)")
    ax.set_ylabel("")
    ax.set_title("Curated Hallmark enrichment among histology-specific gated genes")
    ax.grid(axis="x", color="0.9")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    outdir = args.outdir or args.results_dir
    outdir.mkdir(parents=True, exist_ok=True)

    data = sds.load_kipan_split_artifacts(args.data_dir)
    runs = _read_partial_rows(args.results_dir)
    cfg = _selected_config(args.results_dir, args.family, args.selection, runs)
    saved_runs = _matching_saved_runs(
        args.results_dir,
        cfg,
        runs,
        model_scope=args.model_scope,
    )

    requested_runs = runs[
        (runs["model_family"].replace({"HardSigmoid": "LSPIN"}).astype(str) == cfg.family)
        & _close(runs["gate_sigma"], cfg.gate_sigma)
        & _close(runs["lambda_sparse"], cfg.lambda_sparse)
        & _close(runs["lambda_sample_smooth"], cfg.lambda_sample_smooth)
    ]
    print(
        f"[info] selected {cfg.family} {cfg.selection}: sigma={cfg.gate_sigma} "
        f"lambda={cfg.lambda_sparse} smooth={cfg.lambda_sample_smooth}",
        flush=True,
    )
    print(
        f"[info] using {len(saved_runs)} saved checkpoint(s) out of "
        f"{len(requested_runs)} matching run row(s)",
        flush=True,
    )

    gate_df, model_df = _aggregate_gate_rates(
        cfg,
        saved_runs,
        data,
        device=args.device,
        hard_threshold=args.hard_threshold,
    )
    print("[info] computing per-histology Cox score-test table for all genes", flush=True)
    univariate_df = _compute_univariate_table(
        data,
        scope=args.univariate_scope,
        min_events=args.min_events,
    )
    test_histo = np.asarray(data["histo_test"]).astype(str)
    histo_counts = {h: int((test_histo == h).sum()) for h in pd.unique(test_histo)}
    selected = _select_genes(
        gate_df,
        top_genes_per_histology=args.top_genes_per_histology,
        min_gate_rate=args.min_gate_rate,
        min_specificity=args.min_specificity,
        min_histo_n=args.min_histology_n,
        histo_counts=histo_counts,
    )

    selected_genes = selected[["gene", "gene_idx"]].drop_duplicates()
    plot_df = gate_df.merge(selected_genes, on=["gene", "gene_idx"], how="inner")
    plot_df = plot_df.merge(
        selected[["gene", "assigned_histology", "assigned_histology_label", "rank_within_histology"]],
        on="gene",
        how="left",
    )
    plot_df = _attach_univariate(
        plot_df,
        univariate_df,
    )

    matched_df, matched_summary = _matched_histology_test(selected, gate_df, univariate_df)
    enrich_summary = _enrichment_summary(selected, univariate_df)
    scatter_df, scatter_summary = _scatter_table(gate_df, univariate_df, selected)
    expr_df = _expression_enrichment(selected, data, scope=args.univariate_scope)
    module_df = _module_survival(
        selected,
        data,
        scope=args.univariate_scope,
        min_events=args.min_events,
    )
    ablation_df = pd.DataFrame()
    ablation_summary = pd.DataFrame()
    if not args.skip_ablation:
        ablation_df, ablation_summary = _run_ablation(
            cfg,
            saved_runs,
            selected,
            data,
            device=args.device,
            n_random=args.n_ablation_random,
            seed=args.ablation_seed,
            min_events=args.min_events,
        )
    gene_sets, gene_set_source, gene_set_message = _load_gene_sets(
        gmt_path=args.gene_set_gmt,
        library=args.gene_set_library,
        cache_dir=args.gene_set_cache_dir,
    )
    curated_enrichment = _curated_gene_set_enrichment(
        selected,
        gene_sets=gene_sets,
        gene_set_source=gene_set_source,
        gene_set_message=gene_set_message,
        universe=set(np.asarray(data["gene_names"]).astype(str)),
    )

    prefix = f"kipan_{args.family.lower()}_{args.selection}_gated_univariate"
    plot_csv = outdir / f"{prefix}_dotplot_data.csv"
    selected_csv = outdir / f"{prefix}_selected_genes.csv"
    models_csv = outdir / f"{prefix}_models_used.csv"
    univariate_csv = outdir / f"{prefix}_cox_score_all_genes.csv"
    matched_csv = outdir / f"{prefix}_matched_histology.csv"
    matched_summary_csv = outdir / f"{prefix}_matched_histology_summary.csv"
    enrich_csv = outdir / f"{prefix}_enrichment_summary.csv"
    scatter_csv = outdir / f"{prefix}_scatter_all_genes.csv"
    scatter_summary_csv = outdir / f"{prefix}_scatter_summary.csv"
    expr_csv = outdir / f"{prefix}_expression_enrichment.csv"
    module_csv = outdir / f"{prefix}_module_survival.csv"
    ablation_csv = outdir / f"{prefix}_ablation.csv"
    ablation_summary_csv = outdir / f"{prefix}_ablation_summary.csv"
    curated_csv = outdir / f"{prefix}_curated_gene_set_enrichment.csv"
    out_png = outdir / f"fig_{prefix}_dotplot.png"
    matched_png = outdir / f"fig_{prefix}_matched_histology.png"
    enrich_png = outdir / f"fig_{prefix}_enrichment.png"
    scatter_png = outdir / f"fig_{prefix}_scatter.png"
    expr_png = outdir / f"fig_{prefix}_expression_enrichment.png"
    module_png = outdir / f"fig_{prefix}_module_survival.png"
    ablation_png = outdir / f"fig_{prefix}_ablation.png"
    curated_png = outdir / f"fig_{prefix}_curated_gene_set_enrichment.png"

    plot_df.to_csv(plot_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    model_df.to_csv(models_csv, index=False)
    univariate_df.to_csv(univariate_csv, index=False)
    matched_df.to_csv(matched_csv, index=False)
    matched_summary.to_csv(matched_summary_csv, index=False)
    enrich_summary.to_csv(enrich_csv, index=False)
    scatter_df.to_csv(scatter_csv, index=False)
    scatter_summary.to_csv(scatter_summary_csv, index=False)
    expr_df.to_csv(expr_csv, index=False)
    module_df.to_csv(module_csv, index=False)
    if not args.skip_ablation or not ablation_csv.exists():
        ablation_df.to_csv(ablation_csv, index=False)
    if not args.skip_ablation or not ablation_summary_csv.exists():
        ablation_summary.to_csv(ablation_summary_csv, index=False)
    curated_enrichment.to_csv(curated_csv, index=False)
    _plot_dotplot(
        plot_df,
        selected,
        out_png=out_png,
        fig_width=args.fig_width,
        fig_height_per_gene=args.fig_height_per_gene,
        max_color_score=args.max_color_score,
    )
    _plot_matched_histology(matched_df, matched_png)
    _plot_enrichment(selected, univariate_df, enrich_png)
    _plot_scatter(scatter_df, scatter_png, max_points_per_histo=args.max_scatter_genes_per_histology)
    _plot_expression(expr_df, expr_png)
    _plot_module_survival(module_df, module_png)
    if not args.skip_ablation:
        _plot_ablation(ablation_summary, ablation_png)
    _plot_curated_enrichment(curated_enrichment, curated_png)

    print(f"[done] wrote {out_png}", flush=True)
    print(f"[done] wrote {matched_png}", flush=True)
    print(f"[done] wrote {enrich_png}", flush=True)
    print(f"[done] wrote {scatter_png}", flush=True)
    print(f"[done] wrote {expr_png}", flush=True)
    print(f"[done] wrote {module_png}", flush=True)
    if args.skip_ablation:
        print(f"[done] skipped ablation recomputation; preserved existing {ablation_png}", flush=True)
    else:
        print(f"[done] wrote {ablation_png}", flush=True)
    print(f"[done] wrote {curated_png}", flush=True)
    print(f"[done] wrote {plot_csv}", flush=True)
    print(f"[done] wrote {selected_csv}", flush=True)
    print(f"[done] wrote {models_csv}", flush=True)
    print(f"[done] wrote {matched_summary_csv}", flush=True)
    print(f"[done] wrote {enrich_csv}", flush=True)
    print(f"[done] wrote {scatter_summary_csv}", flush=True)
    print(f"[done] wrote {module_csv}", flush=True)
    print(f"[done] wrote {ablation_summary_csv}", flush=True)
    print(f"[done] wrote {curated_csv}", flush=True)


if __name__ == "__main__":
    main()
