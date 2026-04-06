#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from types import SimpleNamespace

ANALYSES_DIR = Path(__file__).resolve().parent
PAPER_ROOT = ANALYSES_DIR.parent
SDS_SRC = Path("/banach2/wes/lspin-repos/sparsedeepsurv/src")
LSPIN_ROOT = Path("/banach2/wes/lspin-pytorch")

for p in [str(SDS_SRC), str(LSPIN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import sparsedeepsurv as sds
from cleaned_analyses.pipelines import repro_survival_pipeline as rp
from cleaned_analyses import rerender_patient_smoothing_boxplots as boxplots


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render notebook-style selected figures for adaptive runs.")
    p.add_argument("--dataset", choices=["kipan", "brca", "pancan"], required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lspin-temperature", type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)
    p.add_argument("--heatmap-top-genes", type=int, default=180)
    p.add_argument("--heatmap-min-frac-on", type=float, default=0.05)
    p.add_argument("--heatmap-max-frac-on", type=float, default=0.95)
    p.add_argument("--skip-heatmaps", action="store_true")
    return p.parse_args()


def _norm_family(x: str) -> str:
    x = str(x)
    if x in {"HardSigmoid", "LSPIN"}:
        return "LSPIN"
    return x


def _model_label(family: str, selection: str) -> str:
    if family == "LSPIN":
        return "LSPIN (no smooth)" if selection == "nosmooth" else "LSPIN + patient smooth"
    if family == "Concrete":
        return "Concrete (no smooth)" if selection == "nosmooth" else "Concrete + patient smooth"
    raise ValueError(f"Unexpected family: {family}")


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, 0.0
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, ci


def _load_dataset(dataset: str, outdir: Path) -> Dict[str, np.ndarray]:
    if dataset == "kipan":
        return sds.load_kipan_split_artifacts(outdir)
    if dataset == "brca":
        return sds.load_brca_split_artifacts(outdir)
    if dataset == "pancan":
        return sds.load_pancan_split_artifacts(outdir)
    raise ValueError(dataset)


def _infer_global_cfg_idx(dataset: str, row: pd.Series) -> Optional[int]:
    if dataset == "kipan":
        from ch3_kipan_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.25, 0.20],
            concrete_sigmas=[0.10, 0.15],
            lspin_lambdas=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007],
            concrete_lambdas=[0.005, 0.004, 0.003, 0.0025, 0.002],
            sample_smooth_grid=[0.0, 0.05, 0.1, 0.15],
        )
    elif dataset == "brca":
        from ch3_brca_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.10, 0.15],
            concrete_sigmas=[0.10, 0.15],
            lspin_lambdas=[0.010, 0.007, 0.005, 0.0033, 0.0025, 0.0018],
            concrete_lambdas=[0.004, 0.003, 0.0022, 0.0018, 0.0015, 0.0012],
            sample_smooth_grid=[0.0, 0.1, 0.2, 0.4],
        )
    elif dataset == "pancan":
        from ch3_pancan_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.25, 0.20],
            concrete_sigmas=[0.10, 0.15],
            lspin_lambdas=[0.0012, 0.0009, 0.0007, 0.0006, 0.0005],
            concrete_lambdas=[0.002, 0.0017, 0.0014, 0.0012, 0.0010],
            sample_smooth_grid=[0.0, 0.05, 0.1, 0.15],
        )
    else:
        return None
    configs = _build_configs(args)
    family = str(row["family"])
    gate_type = "lspin_tf" if family == "LSPIN" else "concrete"
    for cfg in configs:
        if (
            cfg["family_label"] == family
            and cfg["gate_type"] == gate_type
            and np.isclose(float(cfg["gate_sigma"]), float(row["gate_sigma"]))
            and np.isclose(float(cfg["lam"]), float(row["lambda_sparse"]))
            and np.isclose(float(cfg["smooth"]), float(row["lambda_sample_smooth"]))
        ):
            return int(cfg["global_cfg_idx"])
    return None


def _prepare_split(data: Dict[str, np.ndarray], *, seed: int, knn_k: int):
    X_train = data["X_train"]
    X_test = data["X_test"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])])
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    Xt_tr = sds.as_torch(X_train[i_tr])
    Xt_val = sds.as_torch(X_train[i_val])
    Xt_test = sds.as_torch(X_test)
    tt_tr = sds.as_torch(data["time_train"][i_tr])
    tt_val = sds.as_torch(data["time_train"][i_val])
    tt_test = sds.as_torch(data["time_test"])
    et_tr = sds.as_torch(data["event_train"][i_tr])
    et_val = sds.as_torch(data["event_train"][i_val])
    et_test = sds.as_torch(data["event_test"])
    A_train = sds.build_knn_adjacency_csr(
        Xt_tr, k=knn_k, pca_dim=50, metric="cosine", symmetrize=True
    )
    return {
        "Xt_tr": Xt_tr,
        "Xt_val": Xt_val,
        "Xt_test": Xt_test,
        "tt_tr": tt_tr,
        "tt_val": tt_val,
        "tt_test": tt_test,
        "et_tr": et_tr,
        "et_val": et_val,
        "et_test": et_test,
        "A_train": A_train,
        "input_dim": X_train.shape[1],
    }


def _choose_representative_seed(df_runs: pd.DataFrame, *, family: str, gate_sigma: float, lam: float, smooth: float) -> int:
    sub = df_runs[
        (df_runs["model_family"] == family)
        & np.isclose(df_runs["gate_sigma"], gate_sigma)
        & np.isclose(df_runs["lambda_sparse"], lam)
        & np.isclose(df_runs["lambda_sample_smooth"], smooth)
    ].copy()
    if sub.empty:
        raise ValueError(f"No runs found for family={family}, sigma={gate_sigma}, lambda={lam}, smooth={smooth}")
    cols = ["mean_Khard", "gene_union_count", "test_cindex"]
    means = sub[cols].mean()
    stds = sub[cols].std(ddof=0).replace(0, 1.0)
    z = ((sub[cols] - means) / stds).pow(2).sum(axis=1)
    return int(sub.loc[z.idxmin(), "seed"])


def _build_selected_metric_table(selected_cfg: pd.DataFrame, df_runs: pd.DataFrame, df_aff: pd.DataFrame, df_risk: pd.DataFrame, df_cluster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sel in selected_cfg.iterrows():
        family = str(sel["family"])
        sigma = float(sel["gate_sigma"])
        lam = float(sel["lambda_sparse"])
        smooth = float(sel["lambda_sample_smooth"])
        model = str(sel["model"])
        run_sub = df_runs[
            (df_runs["model_family"] == family)
            & np.isclose(df_runs["gate_sigma"], sigma)
            & np.isclose(df_runs["lambda_sparse"], lam)
            & np.isclose(df_runs["lambda_sample_smooth"], smooth)
        ]
        aff_sub = df_aff[
            (df_aff["model_family"] == family)
            & np.isclose(df_aff["gate_sigma"], sigma)
            & np.isclose(df_aff["lambda_sparse"], lam)
            & np.isclose(df_aff["lambda_sample_smooth"], smooth)
        ]
        risk_sub = df_risk[
            (df_risk["model_family"] == family)
            & np.isclose(df_risk["gate_sigma"], sigma)
            & np.isclose(df_risk["lambda_sparse"], lam)
            & np.isclose(df_risk["lambda_sample_smooth"], smooth)
        ]
        cluster_sub = df_cluster[
            (df_cluster["model_family"] == family)
            & np.isclose(df_cluster["gate_sigma"], sigma)
            & np.isclose(df_cluster["lambda_sparse"], lam)
            & np.isclose(df_cluster["lambda_sample_smooth"], smooth)
        ]
        rows.append(
            {
                "family": family,
                "selection": str(sel["selection"]),
                "model": model,
                "mean_Khard": _mean_ci(run_sub["mean_Khard"])[0],
                "ci_Khard": _mean_ci(run_sub["mean_Khard"])[1],
                "mean_khard_over_union": _mean_ci(run_sub["mean_Khard"] / run_sub["gene_union_count"].clip(lower=1e-8))[0],
                "ci_khard_over_union": _mean_ci(run_sub["mean_Khard"] / run_sub["gene_union_count"].clip(lower=1e-8))[1],
                "mean_test_c": _mean_ci(run_sub["test_cindex"])[0],
                "ci_test_c": _mean_ci(run_sub["test_cindex"])[1],
                "mean_manifold_alignment": _mean_ci(run_sub["manifold_alignment"])[0],
                "ci_manifold_alignment": _mean_ci(run_sub["manifold_alignment"])[1],
                "mean_gene_union_count": _mean_ci(run_sub["gene_union_count"])[0],
                "ci_gene_union_count": _mean_ci(run_sub["gene_union_count"])[1],
                "mean_gene_freq_ge_threshold_count": _mean_ci(run_sub["gene_freq_ge_threshold_count"])[0],
                "ci_gene_freq_ge_threshold_count": _mean_ci(run_sub["gene_freq_ge_threshold_count"])[1],
                "mean_effective_gene_count": _mean_ci(run_sub["effective_gene_count"])[0],
                "ci_effective_gene_count": _mean_ci(run_sub["effective_gene_count"])[1],
                "mean_affinity_corr": _mean_ci(aff_sub["affinity_corr"])[0],
                "ci_affinity_corr": _mean_ci(aff_sub["affinity_corr"])[1],
                "mean_risk_corr": _mean_ci(risk_sub["risk_corr"])[0],
                "ci_risk_corr": _mean_ci(risk_sub["risk_corr"])[1],
                "mean_top_risk_overlap_ratio": _mean_ci(risk_sub["top_risk_overlap_ratio"])[0],
                "ci_top_risk_overlap_ratio": _mean_ci(risk_sub["top_risk_overlap_ratio"])[1],
                "mean_cluster_ari": _mean_ci(cluster_sub["cluster_ari"])[0],
                "ci_cluster_ari": _mean_ci(cluster_sub["cluster_ari"])[1],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["model"] = pd.Categorical(out["model"], categories=boxplots.BRIDGE_MODEL_ORDER, ordered=True)
        out = out.sort_values("model").reset_index(drop=True)
    return out


def _pick_matched_comparison_configs(results_dir: Path, *, required_n_runs: int = 12) -> pd.DataFrame:
    acc = pd.read_csv(results_dir / "accuracy_summary.csv")
    acc["model_family"] = acc["model_family"].map(_norm_family)
    acc = acc[acc["n_runs"].astype(int) >= int(required_n_runs)].copy()

    rows = []
    for family in sorted(acc["model_family"].dropna().unique()):
        fam = acc[acc["model_family"] == family].copy()
        no_df = fam[np.isclose(fam["lambda_sample_smooth"], 0.0)].copy()
        sm_df = fam[fam["lambda_sample_smooth"] > 0.0].copy()
        if no_df.empty or sm_df.empty:
            continue

        pair_candidates = []
        for _, sm in sm_df.iterrows():
            cand = no_df.copy()
            cand["khard_gap_abs"] = (cand["mean_Khard"] - float(sm["mean_Khard"])).abs()
            cand["khard_gap_rel"] = cand["khard_gap_abs"] / cand["mean_Khard"].clip(lower=1e-8)
            cand["sigma_gap"] = (cand["gate_sigma"] - float(sm["gate_sigma"])).abs()
            no = cand.sort_values(
                ["khard_gap_rel", "khard_gap_abs", "sigma_gap", "mean_test_c"],
                ascending=[True, True, True, False],
            ).iloc[0]
            c_drop = float(no["mean_test_c"] - sm["mean_test_c"])
            pair_candidates.append(
                {
                    "no": no,
                    "sm": sm,
                    "khard_gap_rel": float(no["khard_gap_rel"]),
                    "khard_gap_abs": float(no["khard_gap_abs"]),
                    "sigma_gap": float(no["sigma_gap"]),
                    "c_drop": c_drop,
                }
            )

        if not pair_candidates:
            continue

        best = min(
            pair_candidates,
            key=lambda p: (
                p["khard_gap_rel"],
                p["khard_gap_abs"],
                p["sigma_gap"],
                -float((float(p["no"]["mean_test_c"]) + float(p["sm"]["mean_test_c"])) / 2.0),
                -float(p["no"]["mean_test_c"]),
            ),
        )
        no = best["no"]
        sm = best["sm"]
        for selection, row in [("nosmooth", no), ("smooth", sm)]:
            rr = dict(row)
            rows.append(
                {
                    "family": family,
                    "gate_sigma": float(rr["gate_sigma"]),
                    "selection": selection,
                    **rr,
                    "model": _model_label(family, selection),
                }
            )
    return pd.DataFrame(rows)


def _write_notebook_style_files(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "all_runs_raw.csv")
    aff = pd.read_csv(results_dir / "all_affinity_pairs.csv")
    risk = pd.read_csv(results_dir / "all_risk_pairs.csv")
    cluster = pd.read_csv(results_dir / "all_cluster_pairs.csv")
    selected = pd.read_csv(results_dir / "selected_showcase_configs.csv")

    for df in [runs, aff, risk, cluster, selected]:
        df["model_family"] = df["model_family"].map(_norm_family)
    selected["family"] = selected["family"].map(_norm_family)

    runs.to_csv(results_dir / "notebook_style_models_runs.csv", index=False)
    aff.to_csv(results_dir / "notebook_style_models_affinity_pairs.csv", index=False)
    risk.to_csv(results_dir / "notebook_style_models_risk_pairs.csv", index=False)
    cluster.to_csv(results_dir / "notebook_style_models_cluster_pairs.csv", index=False)

    acc = pd.read_csv(results_dir / "accuracy_summary.csv")
    aff_sum = pd.read_csv(results_dir / "affinity_summary.csv")
    risk_sum = pd.read_csv(results_dir / "risk_summary.csv")
    cluster_sum = pd.read_csv(results_dir / "cluster_summary.csv")
    for df in [acc, aff_sum, risk_sum, cluster_sum]:
        df["model_family"] = df["model_family"].map(_norm_family)
    acc.to_csv(results_dir / "notebook_style_models_accuracy_summary.csv", index=False)
    aff_sum.to_csv(results_dir / "notebook_style_models_affinity_summary.csv", index=False)
    risk_sum.to_csv(results_dir / "notebook_style_models_risk_summary.csv", index=False)
    cluster_sum.to_csv(results_dir / "notebook_style_models_cluster_summary.csv", index=False)

    selected_cfg = _pick_matched_comparison_configs(results_dir)
    if selected_cfg.empty:
        selected_cfg = selected.copy()
        selected_cfg["model"] = [
            _model_label(str(fam), str(sel))
            for fam, sel in zip(selected_cfg["family"], selected_cfg["selection"])
        ]
    selected_cfg.to_csv(results_dir / "selected_comparison_configs.csv", index=False)

    selected_metric_table = _build_selected_metric_table(selected_cfg, runs, aff, risk, cluster)
    selected_metric_table.to_csv(results_dir / "selected_comparison_metrics_summary.csv", index=False)
    return runs, aff, risk, cluster, selected_cfg


def _retrain_selected_model(
    *,
    row: pd.Series,
    split: Dict[str, object],
    seed: int,
    device: str,
    args: argparse.Namespace,
):
    family = str(row["family"])
    gate_type = "lspin_tf" if family == "LSPIN" else "concrete"
    temperature = float(args.lspin_temperature if family == "LSPIN" else args.concrete_temperature)
    concrete_mode = str(getattr(args, "concrete_mode", "ste"))

    sds.set_all_seeds(seed)
    model, info = sds.run_one_model(
        split["Xt_tr"],
        split["tt_tr"],
        split["et_tr"],
        split["Xt_val"],
        split["tt_val"],
        split["et_val"],
        split["Xt_test"],
        split["tt_test"],
        split["et_test"],
        input_dim=int(split["input_dim"]),
        gate_type=gate_type,
        gate_sigma=float(row["gate_sigma"]),
        lam=float(row["lambda_sparse"]),
        lambda_sample_smooth=float(row["lambda_sample_smooth"]),
        patience=int(args.patience),
        A_sample_train=split["A_train"],
        device=device,
        lr=float(args.lr),
        temperature=temperature,
        concrete_mode=(concrete_mode if family != "LSPIN" else "relaxed"),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
    )
    return model, info


def _load_phase1_model_if_available(
    *,
    dataset: str,
    results_dir: Path,
    row: pd.Series,
    runs: pd.DataFrame,
    input_dim: int,
    device: str,
    args: argparse.Namespace,
):
    family = str(row["family"])
    gate_type = "lspin_tf" if family == "LSPIN" else "concrete"
    temperature = float(args.lspin_temperature if family == "LSPIN" else args.concrete_temperature)
    concrete_mode = str(getattr(args, "concrete_mode", "ste"))

    sub = runs[
        (runs["model_family"] == family)
        & np.isclose(runs["gate_sigma"], float(row["gate_sigma"]))
        & np.isclose(runs["lambda_sparse"], float(row["lambda_sparse"]))
        & np.isclose(runs["lambda_sample_smooth"], float(row["lambda_sample_smooth"]))
        & (runs["phase"] == 1)
    ].copy()
    if sub.empty:
        return None, None

    global_cfg_idx = None
    if "global_cfg_idx" in sub.columns:
        global_cfg_idx = int(sub.iloc[0]["global_cfg_idx"])
    else:
        global_cfg_idx = _infer_global_cfg_idx(dataset, row)
    if global_cfg_idx is None:
        return None, None

    cols = ["mean_Khard", "gene_union_count", "test_cindex"]
    means = sub[cols].mean()
    stds = sub[cols].std(ddof=0).replace(0, 1.0)
    z = ((sub[cols] - means) / stds).pow(2).sum(axis=1)
    best = sub.loc[z.idxmin()]
    sd_path = None
    for part_dir in sorted(results_dir.glob("_partial_p1_worker*")):
        cand = part_dir / "state_dicts" / f"cfg{int(global_cfg_idx)}_rep{int(best['rep_id'])}.pt"
        if cand.exists():
            sd_path = cand
            break
    if sd_path is None:
        return None, None

    model = sds.make_model(
        input_dim=int(input_dim),
        gate_type=gate_type,
        gate_sigma=float(row["gate_sigma"]),
        temperature=temperature,
        concrete_mode=(concrete_mode if family != "LSPIN" else "relaxed"),
    ).to(device)
    sd = __import__("torch").load(sd_path, map_location=device)
    model.load_state_dict(sd)
    info = {
        "seed": int(best["seed"]),
        "state_dict_path": str(sd_path),
        "test_cindex": float(best["test_cindex"]),
        "best_val_cindex": float(best["best_val_cindex"]),
    }
    return model, info


def _save_heatmaps(
    *,
    dataset: str,
    results_dir: Path,
    outdir: Path,
    selected_cfg: pd.DataFrame,
    runs: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    data = _load_dataset(dataset, outdir)
    split = _prepare_split(data, seed=int(args.seed), knn_k=int(args.knn_k))
    device = sds.resolve_device(str(getattr(args, "device", "auto")))

    heatmap_rows = []
    stem_family = {"LSPIN": "lspin", "Concrete": "concrete"}
    for _, row in selected_cfg.iterrows():
        family = str(row["family"])
        selection = str(row["selection"])
        model, info = _load_phase1_model_if_available(
            dataset=dataset,
            results_dir=results_dir,
            row=row,
            runs=runs,
            input_dim=int(split["input_dim"]),
            device=device,
            args=args,
        )
        if model is None:
            seed = _choose_representative_seed(
                runs,
                family=family,
                gate_sigma=float(row["gate_sigma"]),
                lam=float(row["lambda_sparse"]),
                smooth=float(row["lambda_sample_smooth"]),
            )
            model, info = _retrain_selected_model(
                row=row,
                split=split,
                seed=seed,
                device=device,
                args=args,
            )
        else:
            seed = int(info["seed"])
        stem = f"{stem_family[family]}_{selection}"
        cluster = rp.save_fig_gate_heatmap_with_histo(
            model,
            Xt=split["Xt_test"],
            histo=data["histo_test"],
            gene_names=data["gene_names"],
            device=device,
            outpath=results_dir / f"fig_gate_heatmap_{stem}.png",
            title=(
                f"{family} {'no smooth' if selection == 'nosmooth' else 'patient smooth'} heatmap\n"
                f"meanK={float(row['mean_Khard']):.0f}, union={float(row['mean_gene_union_count']):.0f}, C={float(row['mean_test_c']):.3f}"
            ),
            view="hard",
            min_frac_on=float(args.heatmap_min_frac_on),
            max_frac_on=float(args.heatmap_max_frac_on),
            top_genes=int(args.heatmap_top_genes),
        )
        if cluster is not None:
            cluster.to_csv(results_dir / f"heatmap_sample_order_{stem}.csv", index=False)
        heatmap_rows.append(
            {
                "family": family,
                "selection": selection,
                "model": _model_label(family, selection),
                "lambda_sparse": float(row["lambda_sparse"]),
                "lambda_sample_smooth": float(row["lambda_sample_smooth"]),
                "gate_sigma": float(row["gate_sigma"]),
                "representative_seed": int(seed),
                "test_cindex": float(info.get("test_cindex", np.nan)),
                "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
            }
        )
    pd.DataFrame(heatmap_rows).to_csv(results_dir / "heatmap_models_summary.csv", index=False)
    selected_cfg[
        [
            "family",
            "selection",
            "gate_sigma",
            "lambda_sparse",
            "lambda_sample_smooth",
            "mean_Khard",
            "mean_gene_union_count",
            "mean_test_c",
        ]
    ].to_csv(results_dir / "selected_best_heatmap_configs.csv", index=False)


def render(results_dir: Path, outdir: Path, dataset: str, args: argparse.Namespace) -> None:
    runs, aff, risk, cluster, selected_cfg = _write_notebook_style_files(results_dir)
    dataset_label = dataset.upper() if dataset != "pancan" else "PanCan"
    boxplots.save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability")
    boxplots.save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability_cindex")
    if not bool(getattr(args, "skip_heatmaps", False)):
        _save_heatmaps(
            dataset=dataset,
            results_dir=results_dir,
            outdir=outdir,
            selected_cfg=selected_cfg,
            runs=runs,
            args=args,
        )


def main() -> None:
    args = _parse_args()
    render(args.results_dir.resolve(), args.outdir.resolve(), args.dataset, args)


if __name__ == "__main__":
    main()
