#!/usr/bin/env python3
"""
Validation runner for SparseDeepSurv chapter-3 experiments.

Implements two experiment families:
  - Goal 0: gate-type comparison (`lspin_tf` vs `concrete`)
  - Goal 1: sparse-vs-MLP comparison on KIPAN and BRCA

The script intentionally uses the source trees directly instead of relying on
editable installs, so each worker repeats the required `sys.path.insert(...)`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
for _path in [SDS_SRC]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import sparsedeepsurv as sds
from mlp_baseline import MLPTrainConfig, eval_cindex as eval_mlp_cindex, make_seeded_mlp, train_deepsurv_mlp_l1

PAPER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DEFAULT = PAPER_ROOT / "data" / "runs" / "validate_models"
KIPAN_DATA_DEFAULT = Path("/banach2/wes/lspin-pytorch/runs/kipan_20260209_213604")
BRCA_DATA_DEFAULT = Path("/banach2/wes/lspin-pytorch/runs/tcga_brca20260214_001423")
KIPAN_SHOWCASE_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_adaptive_v1" / "selected_showcase_configs.csv"
BRCA_SHOWCASE_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_brca_adaptive_v2" / "selected_showcase_configs.csv"

DATASET_DEFAULTS = {
    "kipan": {
        "data_path": KIPAN_DATA_DEFAULT,
        "showcase_path": KIPAN_SHOWCASE_DEFAULT,
        "knn_k": 8,
        "lspin_main_sigma": 0.20,
        "lspin_alt_sigma": 0.15,
        "concrete_sigma": 0.10,
    },
    "brca": {
        "data_path": BRCA_DATA_DEFAULT,
        "showcase_path": BRCA_SHOWCASE_DEFAULT,
        "knn_k": 5,
        "lspin_main_sigma": 0.10,
        "lspin_alt_sigma": 0.15,
        "concrete_sigma": 0.10,
    },
}

SHOWCASE_FAMILY_ALIASES = {
    "hardsigmoid": {"hardsigmoid", "lspin"},
    "concrete": {"concrete"},
}


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self._log = open(log_path, "a", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def fileno(self):
        return self._stdout.fileno()

def _load_dataset(dataset_name: str, data_path: Path) -> Dict[str, np.ndarray]:
    if dataset_name == "kipan":
        return sds.load_kipan_split_artifacts(data_path)
    if dataset_name == "brca":
        return sds.load_brca_split_artifacts(data_path)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _pick_showcase_row(
    showcase_csv: Path,
    *,
    family: str,
    selection: str,
    sigma_target: float,
) -> Dict[str, object]:
    df = pd.read_csv(showcase_csv)
    family_key = family.lower()
    aliases = SHOWCASE_FAMILY_ALIASES.get(family_key, {family_key})
    sub = df[
        (df["family"].astype(str).str.lower().isin(aliases))
        & (df["selection"].astype(str).str.lower() == selection.lower())
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No showcase rows found in {showcase_csv} for family={family}, selection={selection}"
        )
    sub["sigma_dist"] = (sub["gate_sigma"].astype(float) - float(sigma_target)).abs()
    row = sub.sort_values(["sigma_dist", "mean_test_c"], ascending=[True, False]).iloc[0].to_dict()
    row["sigma_target"] = float(sigma_target)
    row["sigma_match"] = "exact" if abs(float(row["gate_sigma"]) - float(sigma_target)) < 1e-12 else "nearest"
    return row


def _goal0_sweep_variants(
    lspin_no: Dict[str, object],
    concrete_no: Dict[str, object],
    lambda_multipliers: Iterable[float],
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    families = [
        ("HardSigmoid", "lspin_tf", float(lspin_no["gate_sigma"]), float(lspin_no["lambda_sparse"])),
        ("Concrete", "concrete", float(concrete_no["gate_sigma"]), float(concrete_no["lambda_sparse"])),
    ]
    for family_label, gate_type, gate_sigma, base_lambda in families:
        for mult in lambda_multipliers:
            lam = float(base_lambda) * float(mult)
            mult_label = str(mult).replace(".", "p")
            variants.append({
                "experiment": "goal0",
                "variant_key": f"goal0_{family_label.lower()}_lamx{mult_label}",
                "variant_label": f"{family_label} x{float(mult):.2g}",
                "model_family": family_label,
                "model_kind": "gated",
                "gate_type": gate_type,
                "gate_sigma": gate_sigma,
                "lambda_sparse": lam,
                "lambda_sample_smooth": 0.0,
                "selection": "goal0_sweep",
                "sigma_match": "exact",
                "lambda_base": float(base_lambda),
                "lambda_multiplier": float(mult),
                "goal0_target_reference": bool(family_label == "HardSigmoid" and abs(float(mult) - 1.0) < 1e-12),
            })
    return variants


def _goal1_sweep_variants(
    lspin_no: Dict[str, object],
    lspin_sm: Dict[str, object],
    lspin_alt: Dict[str, object],
    concrete_no: Dict[str, object],
    concrete_sm: Dict[str, object],
    lambda_multipliers: Iterable[float],
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = [
        {
            "experiment": "goal1",
            "variant_key": "mlp",
            "variant_label": "MLP",
            "model_family": "MLP",
            "model_kind": "mlp",
            "gate_type": None,
            "gate_sigma": np.nan,
            "lambda_sparse": np.nan,
            "lambda_sample_smooth": np.nan,
            "selection": "reference",
            "sigma_match": "na",
            "lambda_base": np.nan,
            "lambda_multiplier": np.nan,
            "goal0_target_reference": False,
            "goal1_group": "dense",
            "goal1_target_reference": False,
        }
    ]

    grouped_families = [
        ("nosmooth", "HardSigmoid", "lspin_tf", float(lspin_no["gate_sigma"]), float(lspin_no["lambda_sparse"]), 0.0, True),
        ("nosmooth", "Concrete", "concrete", float(concrete_no["gate_sigma"]), float(concrete_no["lambda_sparse"]), 0.0, False),
        ("smooth", "HardSigmoid", "lspin_tf", float(lspin_sm["gate_sigma"]), float(lspin_sm["lambda_sparse"]), float(lspin_sm["lambda_sample_smooth"]), True),
        ("smooth", "Concrete", "concrete", float(concrete_sm["gate_sigma"]), float(concrete_sm["lambda_sparse"]), float(concrete_sm["lambda_sample_smooth"]), False),
    ]

    for group_label, family_label, gate_type, gate_sigma, base_lambda, smooth, is_ref in grouped_families:
        for mult in lambda_multipliers:
            lam = float(base_lambda) * float(mult)
            mult_label = str(mult).replace(".", "p")
            variants.append({
                "experiment": "goal1",
                "variant_key": f"goal1_{group_label}_{family_label.lower()}_lamx{mult_label}",
                "variant_label": f"{family_label} {group_label} x{float(mult):.2g}",
                "model_family": family_label,
                "model_kind": "gated",
                "gate_type": gate_type,
                "gate_sigma": gate_sigma,
                "lambda_sparse": lam,
                "lambda_sample_smooth": float(smooth),
                "selection": f"goal1_{group_label}_sweep",
                "sigma_match": "exact",
                "lambda_base": float(base_lambda),
                "lambda_multiplier": float(mult),
                "goal0_target_reference": False,
                "goal1_group": group_label,
                "goal1_target_reference": bool(is_ref and abs(float(mult) - 1.0) < 1e-12),
            })

    variants.append({
        "experiment": "goal1",
        "variant_key": "goal1_lspin_alt_nosmooth",
        "variant_label": f"HardSigmoid sigma={float(lspin_alt['gate_sigma']):.2f}",
        "model_family": "HardSigmoid",
        "model_kind": "gated",
        "gate_type": "lspin_tf",
        "gate_sigma": float(lspin_alt["gate_sigma"]),
        "lambda_sparse": float(lspin_alt["lambda_sparse"]),
        "lambda_sample_smooth": float(lspin_alt["lambda_sample_smooth"]),
        "selection": str(lspin_alt["selection"]),
        "sigma_match": str(lspin_alt["sigma_match"]),
        "lambda_base": float(lspin_alt["lambda_sparse"]),
        "lambda_multiplier": 1.0,
        "goal0_target_reference": False,
        "goal1_group": "alt_sigma",
        "goal1_target_reference": False,
    })
    return variants


def _build_variants(
    dataset_name: str,
    showcase_csv: Path,
    mode: str,
    goal0_lambda_multipliers: Iterable[float],
    goal1_lambda_multipliers: Iterable[float],
) -> List[Dict[str, object]]:
    defaults = DATASET_DEFAULTS[dataset_name]
    lspin_no = _pick_showcase_row(
        showcase_csv, family="HardSigmoid", selection="nosmooth", sigma_target=defaults["lspin_main_sigma"]
    )
    lspin_sm = _pick_showcase_row(
        showcase_csv, family="HardSigmoid", selection="smooth", sigma_target=defaults["lspin_main_sigma"]
    )
    lspin_alt = _pick_showcase_row(
        showcase_csv, family="HardSigmoid", selection="nosmooth", sigma_target=defaults["lspin_alt_sigma"]
    )
    concrete_no = _pick_showcase_row(
        showcase_csv, family="Concrete", selection="nosmooth", sigma_target=defaults["concrete_sigma"]
    )
    concrete_sm = _pick_showcase_row(
        showcase_csv, family="Concrete", selection="smooth", sigma_target=defaults["concrete_sigma"]
    )

    goal0 = _goal0_sweep_variants(lspin_no, concrete_no, goal0_lambda_multipliers)

    goal1 = _goal1_sweep_variants(
        lspin_no,
        lspin_sm,
        lspin_alt,
        concrete_no,
        concrete_sm,
        goal1_lambda_multipliers,
    )

    if mode == "goal0":
        return goal0
    if mode == "goal1":
        return goal1
    if mode == "all":
        return goal0 + goal1
    raise ValueError(f"Unknown mode: {mode}")


def _make_tasks(args: argparse.Namespace, variants_by_dataset: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for dataset_name in args.datasets:
        for seed_offset in range(args.n_seeds):
            run_seed = int(args.seed_base + seed_offset)
            for variant in variants_by_dataset[dataset_name]:
                task = {
                    "dataset": dataset_name,
                    "data_path": str(DATASET_DEFAULTS[dataset_name]["data_path"]),
                    "knn_k": int(DATASET_DEFAULTS[dataset_name]["knn_k"]),
                    "run_seed": run_seed,
                    "split_seed": run_seed,
                    "train_seed": run_seed,
                    "seed_offset": seed_offset,
                    "task_stage": "screen",
                    "save_init_artifact": True,
                    "load_init_artifact": "",
                    **variant,
                }
                tasks.append(task)
    tasks.sort(key=lambda row: (row["dataset"], row["experiment"], row["variant_key"], row["run_seed"]))
    return tasks


def _split_dataset(data: Dict[str, np.ndarray], split_seed: int, knn_k: int) -> Dict[str, object]:
    from sklearn.model_selection import ShuffleSplit

    X_train = data["X_train"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])], dtype=object)
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    return {
        "Xt_tr": sds.as_torch(X_train[i_tr]),
        "Xt_val": sds.as_torch(X_train[i_val]),
        "Xt_test": sds.as_torch(data["X_test"]),
        "tt_tr": sds.as_torch(data["time_train"][i_tr]),
        "tt_val": sds.as_torch(data["time_train"][i_val]),
        "tt_test": sds.as_torch(data["time_test"]),
        "et_tr": sds.as_torch(data["event_train"][i_tr]),
        "et_val": sds.as_torch(data["event_train"][i_val]),
        "et_test": sds.as_torch(data["event_test"]),
        "histo_test": data["histo_test"],
        "input_dim": int(X_train.shape[1]),
        "A_train": sds.build_knn_adjacency_csr(
            sds.as_torch(X_train[i_tr]),
            k=int(knn_k),
            pca_dim=50,
            metric="cosine",
            symmetrize=True,
        ),
    }


def _affinity_artifact_path(artifacts_dir: Path, task: Dict[str, object]) -> Path:
    stem = (
        f"{task['dataset']}__{task['experiment']}__{task['task_stage']}"
        f"__{task['variant_key']}__seed{task['run_seed']}"
    )
    return artifacts_dir / f"{stem}_aff.npy"


def _init_artifact_path(artifacts_dir: Path, task: Dict[str, object]) -> Path:
    stem = (
        f"{task['dataset']}__{task['experiment']}__{task['task_stage']}"
        f"__{task['variant_key']}__seed{task['run_seed']}"
    )
    return artifacts_dir / f"{stem}_init.pt"


def _run_mlp_task(task: Dict[str, object], split: Dict[str, object], device: str, args) -> Dict[str, object]:
    import torch

    train_seed = int(task.get("train_seed", task["run_seed"]))
    model = make_seeded_mlp(
        input_dim=int(split["input_dim"]),
        hidden_dims=tuple(args.mlp_hidden),
        dropout_p=float(args.mlp_dropout),
        seed=train_seed,
    ).to(device)

    init_artifact = ""
    if str(task.get("load_init_artifact", "")):
        try:
            state = torch.load(str(task["load_init_artifact"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(task["load_init_artifact"]), map_location=device)
        model.load_state_dict(state)
        init_artifact = str(task["load_init_artifact"])
    elif bool(task.get("save_init_artifact", False)):
        init_path = _init_artifact_path(Path(task["artifacts_dir"]), task)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
        init_artifact = str(init_path)

    info = train_deepsurv_mlp_l1(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        config=MLPTrainConfig(
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_weight_decay),
            lambda_l1_input=float(args.mlp_l1),
            batch_size=int(args.batch_size),
            max_epochs=int(args.mlp_max_epochs),
            patience=int(args.mlp_patience),
        ),
        device=device,
        verbose=False,
    )
    test_cindex, _ = eval_mlp_cindex(
        model,
        split["Xt_test"], split["tt_test"], split["et_test"],
        device=device,
    )
    return {
        "test_cindex": float(test_cindex),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", info.get("epochs_ran", -1))),
        "mean_Ksoft": np.nan,
        "mean_Khard": np.nan,
        "gene_union_count": np.nan,
        "gene_freq_ge_threshold_count": np.nan,
        "effective_gene_count": np.nan,
        "affinity_artifact": "",
        "init_artifact": init_artifact,
    }


def _run_gated_task(
    task: Dict[str, object],
    split: Dict[str, object],
    device: str,
    args,
    artifacts_dir: Path,
) -> Dict[str, object]:
    import torch

    temperature = float(args.lspin_temperature)
    if task["gate_type"] == "concrete":
        temperature = float(args.concrete_temperature)

    model = sds.make_model(
        input_dim=int(split["input_dim"]),
        gate_type=str(task["gate_type"]),
        gate_sigma=float(task["gate_sigma"]),
        temperature=temperature,
        concrete_mode=(str(args.concrete_mode) if str(task["gate_type"]) == "concrete" else "relaxed"),
    ).to(device)

    init_artifact = ""
    if str(task.get("load_init_artifact", "")):
        try:
            state = torch.load(str(task["load_init_artifact"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(task["load_init_artifact"]), map_location=device)
        model.load_state_dict(state)
        init_artifact = str(task["load_init_artifact"])
    elif bool(task.get("save_init_artifact", False)):
        init_path = _init_artifact_path(Path(task["artifacts_dir"]), task)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
        init_artifact = str(init_path)

    info = sds.train_gated_deepsurv(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        split["Xt_test"], split["tt_test"], split["et_test"],
        A_sample_train=split["A_train"],
        config=sds.GatedTrainConfig(
            lr=float(args.gated_lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lambda_sparse=float(task["lambda_sparse"]),
            lambda_sample_smooth=float(task["lambda_sample_smooth"]),
            lambda_gene_smooth=0.0,
        ),
        device=device,
        verbose=False,
    )
    _, g_det, hard_g, _ = sds.get_gates(model, split["Xt_test"], device=device, hard_threshold=0.5, batch_size=512)
    G_det = g_det.numpy()
    G_hard = hard_g.numpy()
    gpars = sds.global_parsimony_metrics(G_hard, freq_threshold=float(args.global_freq_threshold))
    aff_vec = sds.affinity_upper_vec(sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True))
    aff_path = _affinity_artifact_path(artifacts_dir, task)
    np.save(aff_path, aff_vec)

    row = {
        "test_cindex": float(info.get("test_cindex", np.nan)),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", -1)),
        "mean_Ksoft": float(G_det.sum(axis=1).mean()),
        "mean_Khard": float(G_hard.sum(axis=1).mean()),
        "affinity_artifact": str(aff_path),
        "init_artifact": init_artifact,
        **gpars,
    }

    if task["dataset"] == "kipan":
        d_hist = sds.cindex_by_histology(
            sds.RiskOnlyWrapper(model).to(device),
            split["Xt_test"], split["tt_test"], split["et_test"], split["histo_test"],
            device=device,
        )
        row["_histology_rows"] = d_hist.to_dict("records")

    return row


def _worker(worker_id: int, tasks: List[Dict[str, object]], device: str, args_dict: Dict[str, object], results_dir: str) -> str:
    import sys as _sys

    for _path in [SDS_SRC]:
        if _path not in _sys.path:
            _sys.path.insert(0, _path)

    import sparsedeepsurv as _sds

    global sds
    sds = _sds

    args = argparse.Namespace(**args_dict)
    results_path = Path(results_dir)
    artifacts_dir = results_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stage_label = str(tasks[0].get("task_stage", "screen")) if tasks else "screen"
    partial_dir = results_path / f"_partial_{stage_label}_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    worker_log_path = partial_dir / "worker.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with worker_log_path.open("a", buffering=1) as fh:
            fh.write(msg + "\n")

    dataset_cache: Dict[str, Dict[str, np.ndarray]] = {}
    split_cache: Dict[tuple[str, int], Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    log(f"[worker {worker_id}|{stage_label}] start device={device} n_tasks={len(tasks)}")

    for idx, task in enumerate(tasks, start=1):
        task = {**task, "artifacts_dir": str(artifacts_dir)}
        dataset_name = str(task["dataset"])
        if dataset_name not in dataset_cache:
            dataset_cache[dataset_name] = _load_dataset(dataset_name, Path(str(task["data_path"])))

        split_key = (dataset_name, int(task.get("split_seed", task["run_seed"])))
        if split_key not in split_cache:
            split_cache[split_key] = _split_dataset(
                dataset_cache[dataset_name],
                int(task.get("split_seed", task["run_seed"])),
                int(task["knn_k"]),
            )

        split = split_cache[split_key]
        sds.set_all_seeds(int(task.get("train_seed", task["run_seed"])))
        t0 = time.time()
        log(
            f"[worker {worker_id}] [{idx}/{len(tasks)}] "
            f"{dataset_name} {task['experiment']} {task['variant_key']} "
            f"split={task.get('split_seed', task['run_seed'])} "
            f"train={task.get('train_seed', task['run_seed'])} "
            f"device={device}"
        )

        try:
            if task["model_kind"] == "mlp":
                metrics = _run_mlp_task(task, split, device, args)
            else:
                metrics = _run_gated_task(task, split, device, args, artifacts_dir)
        except Exception as exc:
            rows.append({
                **task,
                "test_cindex": np.nan,
                "best_val_cindex": np.nan,
                "best_epoch": np.nan,
                "mean_Ksoft": np.nan,
                "mean_Khard": np.nan,
                "gene_union_count": np.nan,
                "gene_freq_ge_threshold_count": np.nan,
                "effective_gene_count": np.nan,
                "affinity_artifact": "",
                "init_artifact": str(task.get("load_init_artifact", "")),
                "status": "failed",
                "error": str(exc),
                "elapsed_sec": float(time.time() - t0),
            })
            log(
                f"[worker {worker_id}|{stage_label}]   FAILED "
                f"{task['dataset']} {task['variant_key']} "
                f"split={task.get('split_seed', task['run_seed'])} "
                f"train={task.get('train_seed', task['run_seed'])} "
                f"elapsed={time.time() - t0:.1f}s error={exc}"
            )
            continue

        row = {
            **task,
            **metrics,
            "status": "ok",
            "error": "",
            "elapsed_sec": float(time.time() - t0),
        }
        hist_recs = row.pop("_histology_rows", [])
        rows.append(row)
        log(
            f"[worker {worker_id}|{stage_label}]   ok "
            f"{task['dataset']} {task['variant_key']} "
            f"test_c={float(row['test_cindex']):.4f} "
            f"val_c={float(row['best_val_cindex']):.4f} "
            f"epoch={int(row['best_epoch'])} "
            f"Ksoft={float(row['mean_Ksoft']):.1f} "
            f"Khard={float(row['mean_Khard']):.1f} "
            f"elapsed={float(row['elapsed_sec']):.1f}s"
        )
        for rec in hist_recs:
            hist_rows.append({
                **task,
                "seed": int(task["run_seed"]),
                **rec,
            })

    pd.DataFrame(rows).to_csv(partial_dir / "runs.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(partial_dir / "histology.csv", index=False)
    log(f"[worker {worker_id}|{stage_label}] wrote → {partial_dir} runs={len(rows)} hist={len(hist_rows)}")
    return str(partial_dir)


def _summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type",
        "lambda_sparse", "lambda_base", "lambda_multiplier", "gate_sigma", "goal1_group",
    ]
    df_runs = df_runs.copy()
    for col in group_cols:
        if col not in df_runs.columns:
            df_runs[col] = np.nan
    out = (
        df_runs[df_runs["status"] == "ok"]
        .groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_seed", "size"),
            mean_test_cindex=("test_cindex", "mean"),
            std_test_cindex=("test_cindex", "std"),
            mean_best_val_cindex=("best_val_cindex", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            std_best_epoch=("best_epoch", "std"),
            mean_Ksoft=("mean_Ksoft", "mean"),
            std_Ksoft=("mean_Ksoft", "std"),
            mean_Khard=("mean_Khard", "mean"),
            std_Khard=("mean_Khard", "std"),
            mean_gene_union_count=("gene_union_count", "mean"),
            std_gene_union_count=("gene_union_count", "std"),
            mean_effective_gene_count=("effective_gene_count", "mean"),
            std_effective_gene_count=("effective_gene_count", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "experiment", "variant_key"])
    )
    for mean_col, std_col, ci_col in [
        ("mean_test_cindex", "std_test_cindex", "ci95_test_cindex"),
        ("mean_Ksoft", "std_Ksoft", "ci95_Ksoft"),
        ("mean_Khard", "std_Khard", "ci95_Khard"),
        ("mean_gene_union_count", "std_gene_union_count", "ci95_gene_union_count"),
        ("mean_effective_gene_count", "std_effective_gene_count", "ci95_effective_gene_count"),
    ]:
        out[ci_col] = 1.96 * out[std_col].astype(float) / np.sqrt(out["n_runs"].clip(lower=1).astype(float))
    return out


def _summarize_split_blocks(df_runs: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty:
        return pd.DataFrame()
    cols = [
        "dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family",
        "goal1_group", "split_seed",
    ]
    df_runs = df_runs.copy()
    for col in cols:
        if col not in df_runs.columns:
            df_runs[col] = np.nan
    block = (
        df_runs[df_runs["status"] == "ok"]
        .groupby(cols, dropna=False)
        .agg(
            n_reps=("train_seed", "size"),
            split_mean_test_cindex=("test_cindex", "mean"),
            split_sd_test_cindex=("test_cindex", "std"),
            split_mean_Ksoft=("mean_Ksoft", "mean"),
            split_sd_Ksoft=("mean_Ksoft", "std"),
            split_mean_Khard=("mean_Khard", "mean"),
            split_sd_Khard=("mean_Khard", "std"),
        )
        .reset_index()
    )
    return block


def _summarize_within_between(df_block: pd.DataFrame) -> pd.DataFrame:
    if df_block.empty:
        return pd.DataFrame()
    cols = ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "goal1_group"]
    out = (
        df_block.groupby(cols, dropna=False)
        .agg(
            n_splits=("split_seed", "size"),
            mean_of_split_means_test_cindex=("split_mean_test_cindex", "mean"),
            between_split_sd_test_cindex=("split_mean_test_cindex", "std"),
            mean_within_split_sd_test_cindex=("split_sd_test_cindex", "mean"),
            mean_of_split_means_Ksoft=("split_mean_Ksoft", "mean"),
            between_split_sd_Ksoft=("split_mean_Ksoft", "std"),
            mean_within_split_sd_Ksoft=("split_sd_Ksoft", "mean"),
            mean_of_split_means_Khard=("split_mean_Khard", "mean"),
            between_split_sd_Khard=("split_mean_Khard", "std"),
            mean_within_split_sd_Khard=("split_sd_Khard", "mean"),
        )
        .reset_index()
    )
    return out


def _summarize_affinity(df_aff: pd.DataFrame) -> pd.DataFrame:
    if df_aff.empty:
        return pd.DataFrame()
    return (
        df_aff.groupby(
            ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type"],
            dropna=False,
        )
        .agg(
            mean_affinity_corr=("affinity_corr", "mean"),
            std_affinity_corr=("affinity_corr", "std"),
            n_pairs=("affinity_corr", "size"),
        )
        .reset_index()
    )


def _select_goal0_matched(summary_runs: pd.DataFrame, summary_aff: pd.DataFrame) -> pd.DataFrame:
    if summary_runs.empty:
        return pd.DataFrame()

    goal0 = summary_runs[
        (summary_runs["experiment"] == "goal0")
        & (summary_runs["task_stage"] == "screen")
    ].copy()
    if goal0.empty:
        return pd.DataFrame()

    aff_lookup = (
        summary_aff[
            (summary_aff["experiment"] == "goal0")
            & (summary_aff["task_stage"] == "screen")
        ].copy()
        if not summary_aff.empty else pd.DataFrame()
    )
    rows: List[Dict[str, object]] = []
    family_order = ["HardSigmoid", "Concrete"]

    for dataset_name, sub in goal0.groupby("dataset", dropna=False):
        ref = sub[
                (sub["model_family"] == "HardSigmoid")
                & np.isclose(sub["lambda_multiplier"].astype(float), 1.0)
            ].copy()
        if ref.empty:
            ref = sub[sub["model_family"] == "HardSigmoid"].copy()
        if ref.empty:
            continue
        ref_row = ref.sort_values(["mean_test_cindex"], ascending=[False]).iloc[0]
        target_khard = float(ref_row["mean_Khard"])

        for family in family_order:
            fam = sub[sub["model_family"] == family].copy()
            if fam.empty:
                continue
            fam["khard_delta"] = (fam["mean_Khard"].astype(float) - target_khard).abs()
            fam = fam.sort_values(["khard_delta", "mean_test_cindex"], ascending=[True, False])
            pick = fam.iloc[0].to_dict()
            pick["target_mean_Khard"] = target_khard
            pick["khard_delta"] = float(abs(float(pick["mean_Khard"]) - target_khard))
            pick["goal0_reference_variant_key"] = str(ref_row["variant_key"])
            rows.append(pick)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected

    if not aff_lookup.empty:
        selected = selected.merge(
            aff_lookup[
                ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
            ],
            on=["dataset", "task_stage", "variant_key"],
            how="left",
        )

    return selected.sort_values(["dataset", "model_family"]).reset_index(drop=True)


def _select_goal1_matched(summary_runs: pd.DataFrame, summary_aff: pd.DataFrame) -> pd.DataFrame:
    if summary_runs.empty:
        return pd.DataFrame()

    goal1 = summary_runs[
        (summary_runs["experiment"] == "goal1")
        & (summary_runs["task_stage"] == "screen")
    ].copy()
    if goal1.empty:
        return pd.DataFrame()

    aff_lookup = (
        summary_aff[
            (summary_aff["experiment"] == "goal1")
            & (summary_aff["task_stage"] == "screen")
        ].copy()
        if not summary_aff.empty else pd.DataFrame()
    )
    rows: List[Dict[str, object]] = []
    family_order = ["HardSigmoid", "Concrete"]

    dense = goal1[goal1["model_family"] == "MLP"].copy()
    for _, row in dense.iterrows():
        picked = row.to_dict()
        picked["target_mean_Khard"] = np.nan
        picked["khard_delta"] = np.nan
        picked["goal1_reference_variant_key"] = ""
        rows.append(picked)

    for dataset_name, sub_dataset in goal1.groupby("dataset", dropna=False):
        for group_label in ["nosmooth", "smooth"]:
            sub = sub_dataset[sub_dataset["variant_key"].astype(str).str.contains(f"goal1_{group_label}_")].copy()
            if sub.empty:
                continue
            ref = sub[
                (sub["model_family"] == "HardSigmoid")
                & np.isclose(sub["lambda_multiplier"].astype(float), 1.0)
            ].copy()
            if ref.empty:
                ref = sub[sub["model_family"] == "HardSigmoid"].copy()
            if ref.empty:
                continue
            ref_row = ref.sort_values(["mean_test_cindex"], ascending=[False]).iloc[0]
            target_khard = float(ref_row["mean_Khard"])

            for family in family_order:
                fam = sub[sub["model_family"] == family].copy()
                if fam.empty:
                    continue
                fam["khard_delta"] = (fam["mean_Khard"].astype(float) - target_khard).abs()
                fam = fam.sort_values(["khard_delta", "mean_test_cindex"], ascending=[True, False])
                pick = fam.iloc[0].to_dict()
                pick["goal1_group"] = group_label
                pick["target_mean_Khard"] = target_khard
                pick["khard_delta"] = float(abs(float(pick["mean_Khard"]) - target_khard))
                pick["goal1_reference_variant_key"] = str(ref_row["variant_key"])
                rows.append(pick)

    alt_rows = goal1[goal1["variant_key"] == "goal1_lspin_alt_nosmooth"].copy()
    for _, row in alt_rows.iterrows():
        picked = row.to_dict()
        picked["target_mean_Khard"] = np.nan
        picked["khard_delta"] = np.nan
        picked["goal1_reference_variant_key"] = ""
        rows.append(picked)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected

    if not aff_lookup.empty:
        selected = selected.merge(
            aff_lookup[
                ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
            ],
            on=["dataset", "task_stage", "variant_key"],
            how="left",
        )

    return selected.sort_values(["dataset", "goal1_group", "model_family"]).reset_index(drop=True)


def _compute_affinity_pairs(df_runs: pd.DataFrame) -> pd.DataFrame:
    gated = df_runs[(df_runs["status"] == "ok")].copy()
    if gated.empty:
        return pd.DataFrame()
    gated["affinity_artifact"] = gated["affinity_artifact"].fillna("")
    gated = gated[gated["affinity_artifact"].astype(str).ne("")].copy()
    if gated.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    group_cols = ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type"]
    for keys, sub in gated.groupby(group_cols, dropna=False):
        sub = sub.sort_values("run_seed")
        vecs = []
        seeds = []
        for _, row in sub.iterrows():
            aff_path = Path(str(row["affinity_artifact"]))
            if not aff_path.exists():
                continue
            vecs.append(np.load(aff_path))
            seeds.append(int(row["run_seed"]))
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                rows.append({
                    "dataset": keys[0],
                    "experiment": keys[1],
                    "task_stage": keys[2],
                    "variant_key": keys[3],
                    "variant_label": keys[4],
                    "model_family": keys[5],
                    "gate_type": keys[6],
                    "seed_i": seeds[i],
                    "seed_j": seeds[j],
                    "affinity_corr": sds.affinity_corr_from_vec(vecs[i], vecs[j]),
                })
    return pd.DataFrame(rows)


def _choose_init_run(screen_runs: pd.DataFrame, selected_row: pd.Series) -> pd.Series | None:
    sub = screen_runs[
        (screen_runs["dataset"] == selected_row["dataset"])
        & (screen_runs["experiment"] == selected_row["experiment"])
        & (screen_runs["task_stage"] == "screen")
        & (screen_runs["variant_key"] == selected_row["variant_key"])
        & (screen_runs["status"] == "ok")
        & screen_runs["init_artifact"].astype(str).ne("")
    ].copy()
    if sub.empty:
        return None
    if np.isfinite(selected_row.get("mean_Khard", np.nan)):
        sub["khard_delta"] = (sub["mean_Khard"].astype(float) - float(selected_row["mean_Khard"])).abs()
    else:
        sub["khard_delta"] = 0.0
    sub = sub.sort_values(["khard_delta", "test_cindex"], ascending=[True, False])
    return sub.iloc[0]


def _choose_eval_split_runs(
    screen_runs: pd.DataFrame,
    selected_row: pd.Series,
    n_eval_splits: int,
) -> pd.DataFrame:
    sub = screen_runs[
        (screen_runs["dataset"] == selected_row["dataset"])
        & (screen_runs["experiment"] == selected_row["experiment"])
        & (screen_runs["task_stage"] == "screen")
        & (screen_runs["variant_key"] == selected_row["variant_key"])
        & (screen_runs["status"] == "ok")
        & screen_runs["init_artifact"].astype(str).ne("")
    ].copy()
    if sub.empty:
        return sub
    if np.isfinite(selected_row.get("mean_Khard", np.nan)):
        sub["khard_delta"] = (sub["mean_Khard"].astype(float) - float(selected_row["mean_Khard"])).abs()
    else:
        sub["khard_delta"] = 0.0
    sub = sub.sort_values(["khard_delta", "test_cindex"], ascending=[True, False])
    sub = sub.drop_duplicates(subset=["split_seed"], keep="first")
    return sub.head(int(n_eval_splits)).reset_index(drop=True)


def _build_fixed_init_tasks(
    args: argparse.Namespace,
    selected_df: pd.DataFrame,
    screen_runs: pd.DataFrame,
) -> List[Dict[str, object]]:
    if selected_df.empty or args.n_reps_per_eval_split <= 0 or args.n_eval_splits <= 0:
        return []
    tasks: List[Dict[str, object]] = []
    seen_keys = set()
    for _, row in selected_df.iterrows():
        key = (row["dataset"], row["experiment"], row["variant_key"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        eval_split_rows = _choose_eval_split_runs(screen_runs, row, args.n_eval_splits)
        if eval_split_rows.empty:
            continue
        dataset_name = str(row["dataset"])
        for split_block_id, (_, init_row) in enumerate(eval_split_rows.iterrows()):
            split_seed = int(init_row["split_seed"])
            for rep_offset in range(args.n_reps_per_eval_split):
                train_seed = int(args.eval_train_seed_base + split_block_id * 1000 + rep_offset)
                run_seed = train_seed
                tasks.append({
                    "dataset": dataset_name,
                    "data_path": str(DATASET_DEFAULTS[dataset_name]["data_path"]),
                    "knn_k": int(DATASET_DEFAULTS[dataset_name]["knn_k"]),
                    "run_seed": run_seed,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "seed_offset": rep_offset,
                    "split_block_id": split_block_id,
                    "task_stage": "fixed_init_eval",
                    "save_init_artifact": False,
                    "load_init_artifact": str(init_row["init_artifact"]),
                    "fixed_init_source_seed": int(init_row["run_seed"]),
                    "fixed_init_source_variant_key": str(init_row["variant_key"]),
                    "dataset_selected_mean_Khard": float(row.get("mean_Khard", np.nan)),
                    "dataset_target_mean_Khard": float(row.get("target_mean_Khard", np.nan)),
                    "dataset_khard_delta": float(row.get("khard_delta", np.nan)),
                    "dataset_selected_summary_stage": str(row.get("task_stage", "screen")),
                    "dataset_selected_summary_file": f"{row['experiment']}_matched_summary.csv",
                    "goal1_group": row.get("goal1_group", np.nan),
                    "experiment": str(row["experiment"]),
                    "variant_key": str(row["variant_key"]),
                    "variant_label": str(row["variant_label"]),
                    "model_family": str(row["model_family"]),
                    "model_kind": str(init_row["model_kind"]),
                    "gate_type": init_row["gate_type"],
                    "gate_sigma": init_row["gate_sigma"],
                    "lambda_sparse": init_row["lambda_sparse"],
                    "lambda_sample_smooth": init_row["lambda_sample_smooth"],
                    "selection": init_row["selection"],
                    "sigma_match": init_row.get("sigma_match", "na"),
                    "lambda_base": init_row.get("lambda_base", np.nan),
                    "lambda_multiplier": init_row.get("lambda_multiplier", np.nan),
                    "goal0_target_reference": init_row.get("goal0_target_reference", False),
                })
    tasks.sort(key=lambda row: (row["dataset"], row["experiment"], row["variant_key"], row["run_seed"]))
    return tasks


def _run_task_batch(
    tasks: List[Dict[str, object]],
    args: argparse.Namespace,
    results_dir: Path,
) -> List[Path]:
    if not tasks:
        return []
    shards: List[List[Dict[str, object]]] = [[] for _ in args.devices]
    for idx, task in enumerate(tasks):
        shards[idx % len(args.devices)].append(task)
    stage_label = str(tasks[0].get("task_stage", "screen"))
    print(
        f"[main] dispatching {len(tasks)} {stage_label} tasks across {len(args.devices)} devices: "
        + ", ".join(f"{device}={len(shard)}" for device, shard in zip(args.devices, shards)),
        flush=True,
    )

    partial_dirs: List[Path] = []
    with ProcessPoolExecutor(max_workers=len(args.devices)) as ex:
        futures = {}
        for worker_id, (device, shard) in enumerate(zip(args.devices, shards)):
            if not shard:
                continue
            fut = ex.submit(
                _worker,
                worker_id,
                shard,
                device,
                vars(args),
                str(results_dir),
            )
            futures[fut] = (worker_id, device, len(shard))
        for fut in as_completed(futures):
            worker_id, device, n_tasks = futures[fut]
            try:
                partial_dir = Path(fut.result())
                partial_dirs.append(partial_dir)
                print(
                    f"[main] worker {worker_id} device={device} done "
                    f"tasks={n_tasks} → {partial_dir}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[main] worker {worker_id} device={device} FAILED "
                    f"tasks={n_tasks}: {exc}",
                    flush=True,
                )
    print(f"[main] completed {stage_label} partial_dirs={len(partial_dirs)}", flush=True)
    return partial_dirs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SparseDeepSurv gate types and sparse-vs-MLP baselines")
    p.add_argument("--mode", choices=["goal0", "goal1", "all"], default="all")
    p.add_argument("--datasets", nargs="+", choices=["kipan", "brca"], default=["kipan", "brca"])
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=123)
    p.add_argument("--n-eval-splits", type=int, default=4)
    p.add_argument("--n-reps-per-eval-split", type=int, default=3)
    p.add_argument("--eval-train-seed-base", type=int, default=100123)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--gated-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lspin-temperature", type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)
    p.add_argument("--concrete-mode", choices=["relaxed", "ste"], default="ste")
    p.add_argument("--global-freq-threshold", type=float, default=0.05)

    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-l1", type=float, default=3e-3)
    p.add_argument("--mlp-max-epochs", type=int, default=300)
    p.add_argument("--mlp-patience", type=int, default=25)

    p.add_argument("--kipan-showcase", type=Path, default=KIPAN_SHOWCASE_DEFAULT)
    p.add_argument("--brca-showcase", type=Path, default=BRCA_SHOWCASE_DEFAULT)
    p.add_argument("--kipan-data", type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--brca-data", type=Path, default=BRCA_DATA_DEFAULT)
    p.add_argument(
        "--goal0-lambda-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Lambda sweep multipliers for Goal 0 matched-sparsity gate comparison",
    )
    p.add_argument(
        "--goal1-lambda-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Lambda sweep multipliers for Goal 1 matched-sparsity sparse-model comparison",
    )
    p.add_argument("--smoke-test", action="store_true", help="Run only one task per dataset for import/CLI validation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(args.results_dir / "run.log")

    DATASET_DEFAULTS["kipan"]["showcase_path"] = args.kipan_showcase.resolve()
    DATASET_DEFAULTS["brca"]["showcase_path"] = args.brca_showcase.resolve()
    DATASET_DEFAULTS["kipan"]["data_path"] = args.kipan_data.resolve()
    DATASET_DEFAULTS["brca"]["data_path"] = args.brca_data.resolve()

    variants_by_dataset = {
        dataset_name: _build_variants(
            dataset_name,
            Path(DATASET_DEFAULTS[dataset_name]["showcase_path"]),
            args.mode,
            args.goal0_lambda_multipliers,
            args.goal1_lambda_multipliers,
        )
        for dataset_name in args.datasets
    }
    manifest_rows = []
    for dataset_name, variants in variants_by_dataset.items():
        for variant in variants:
            manifest_rows.append({"dataset": dataset_name, **variant})
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.results_dir / "variant_manifest.csv", index=False)
    print(f"[main] wrote variant_manifest.csv rows={len(manifest_df)}", flush=True)

    tasks = _make_tasks(args, variants_by_dataset)
    if args.smoke_test:
        kept = []
        seen = set()
        for task in tasks:
            key = (task["dataset"], task["experiment"])
            if key in seen:
                continue
            kept.append(task)
            seen.add(key)
        tasks = kept

    print(f"[main] mode={args.mode} datasets={args.datasets} screen_tasks={len(tasks)}")
    print(f"[main] devices={args.devices}")
    print(f"[main] results={args.results_dir}")

    screen_partial_dirs = sorted(
        partial_dir
        for partial_dir in args.results_dir.glob("_partial_screen_worker*")
        if (partial_dir / "runs.csv").exists()
    )
    if screen_partial_dirs:
        print(
            f"[main] found {len(screen_partial_dirs)} reusable screen partial dirs; reusing screen outputs",
            flush=True,
        )
    else:
        screen_partial_dirs = _run_task_batch(tasks, args, args.results_dir)

    run_frames = []
    hist_frames = []
    for partial_dir in sorted(screen_partial_dirs):
        run_csv = partial_dir / "runs.csv"
        hist_csv = partial_dir / "histology.csv"
        if run_csv.exists():
            run_frames.append(pd.read_csv(run_csv))
        if hist_csv.exists() and hist_csv.stat().st_size > 1:
            try:
                hist_frames.append(pd.read_csv(hist_csv))
            except pd.errors.EmptyDataError:
                pass

    df_screen_runs = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    df_screen_hist = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    df_screen_aff = _compute_affinity_pairs(df_screen_runs) if len(df_screen_runs) else pd.DataFrame()
    df_screen_aff_summary = _summarize_affinity(df_screen_aff) if len(df_screen_aff) else pd.DataFrame()
    df_screen_summary = _summarize_runs(df_screen_runs) if len(df_screen_runs) else pd.DataFrame()
    df_goal0_matched = _select_goal0_matched(df_screen_summary, df_screen_aff_summary) if len(df_screen_summary) else pd.DataFrame()
    df_goal1_matched = _select_goal1_matched(df_screen_summary, df_screen_aff_summary) if len(df_screen_summary) else pd.DataFrame()
    print(
        f"[main] screen complete ok_runs={int((df_screen_runs['status'] == 'ok').sum()) if len(df_screen_runs) else 0} "
        f"goal0_matched={len(df_goal0_matched)} goal1_matched={len(df_goal1_matched)}",
        flush=True,
    )

    fixed_init_tasks = _build_fixed_init_tasks(
        args,
        pd.concat([df_goal0_matched, df_goal1_matched], ignore_index=True) if len(df_goal0_matched) or len(df_goal1_matched) else pd.DataFrame(),
        df_screen_runs,
    )
    if args.smoke_test and fixed_init_tasks:
        kept = []
        seen = set()
        for task in fixed_init_tasks:
            key = (task["dataset"], task["experiment"], task["split_seed"])
            if key in seen:
                continue
            kept.append(task)
            seen.add(key)
        fixed_init_tasks = kept

    print(f"[main] fixed_init_eval_tasks={len(fixed_init_tasks)}")
    fixed_partial_dirs = sorted(
        partial_dir
        for partial_dir in args.results_dir.glob("_partial_fixed_init_eval_worker*")
        if (partial_dir / "runs.csv").exists()
    )
    if fixed_partial_dirs:
        print(
            f"[main] found {len(fixed_partial_dirs)} reusable fixed-init partial dirs; reusing eval outputs",
            flush=True,
        )
    else:
        fixed_partial_dirs = _run_task_batch(fixed_init_tasks, args, args.results_dir)

    all_run_frames = [df_screen_runs] if len(df_screen_runs) else []
    all_hist_frames = [df_screen_hist] if len(df_screen_hist) else []
    for partial_dir in sorted(fixed_partial_dirs):
        run_csv = partial_dir / "runs.csv"
        hist_csv = partial_dir / "histology.csv"
        if run_csv.exists():
            all_run_frames.append(pd.read_csv(run_csv))
        if hist_csv.exists() and hist_csv.stat().st_size > 1:
            try:
                all_hist_frames.append(pd.read_csv(hist_csv))
            except pd.errors.EmptyDataError:
                pass

    df_runs = pd.concat(all_run_frames, ignore_index=True) if all_run_frames else pd.DataFrame()
    df_hist = pd.concat(all_hist_frames, ignore_index=True) if all_hist_frames else pd.DataFrame()
    df_aff = _compute_affinity_pairs(df_runs) if len(df_runs) else pd.DataFrame()
    df_aff_summary = _summarize_affinity(df_aff) if len(df_aff) else pd.DataFrame()
    df_summary = _summarize_runs(df_runs) if len(df_runs) else pd.DataFrame()
    df_split_blocks = _summarize_split_blocks(df_runs) if len(df_runs) else pd.DataFrame()
    df_within_between = _summarize_within_between(df_split_blocks) if len(df_split_blocks) else pd.DataFrame()

    def _stage_summary(experiment: str) -> pd.DataFrame:
        sub = df_summary[
            (df_summary["experiment"] == experiment)
            & (df_summary["task_stage"] == "fixed_init_eval")
        ].copy()
        if sub.empty:
            return sub
        if not df_aff_summary.empty and "experiment" in df_aff_summary.columns:
            aff_sub = df_aff_summary[
                (df_aff_summary["experiment"] == experiment)
                & (df_aff_summary["task_stage"] == "fixed_init_eval")
            ].copy()
        else:
            aff_sub = pd.DataFrame()
        if not aff_sub.empty:
            sub = sub.merge(
                aff_sub[
                    ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
                ],
                on=["dataset", "task_stage", "variant_key"],
                how="left",
            )
        return sub

    df_goal0_fixed = _stage_summary("goal0")
    df_goal1_fixed = _stage_summary("goal1")
    print(
        f"[main] fixed-init complete ok_runs={int(((df_runs['status'] == 'ok') & (df_runs['task_stage'] == 'fixed_init_eval')).sum()) if len(df_runs) else 0} "
        f"split_blocks={len(df_split_blocks)} within_between={len(df_within_between)}",
        flush=True,
    )

    df_runs.to_csv(args.results_dir / "runs.csv", index=False)
    df_hist.to_csv(args.results_dir / "histology.csv", index=False)
    df_aff.to_csv(args.results_dir / "affinity_pairs.csv", index=False)
    df_aff_summary.to_csv(args.results_dir / "affinity_summary.csv", index=False)
    df_summary.to_csv(args.results_dir / "summary.csv", index=False)
    df_split_blocks.to_csv(args.results_dir / "split_block_summary.csv", index=False)
    df_within_between.to_csv(args.results_dir / "within_between_split_summary.csv", index=False)
    df_screen_summary.to_csv(args.results_dir / "screen_summary.csv", index=False)
    df_goal0_matched.to_csv(args.results_dir / "goal0_matched_summary.csv", index=False)
    df_goal1_matched.to_csv(args.results_dir / "goal1_matched_summary.csv", index=False)
    df_goal0_fixed.to_csv(args.results_dir / "goal0_fixed_init_summary.csv", index=False)
    df_goal1_fixed.to_csv(args.results_dir / "goal1_fixed_init_summary.csv", index=False)

    print(
        f"[main] wrote variant_manifest.csv, runs.csv, screen_summary.csv, summary.csv, "
        f"split_block_summary.csv, within_between_split_summary.csv, "
        f"goal0_matched_summary.csv, goal1_matched_summary.csv, "
        f"goal0_fixed_init_summary.csv, goal1_fixed_init_summary.csv to {args.results_dir}"
    )


if __name__ == "__main__":
    main()
