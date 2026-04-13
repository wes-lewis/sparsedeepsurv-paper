#!/usr/bin/env python3
"""
Hyperparameter tuning sweep for gated models in Goal 1 (KIPAN + BRCA).

Sweeps over:
  - gating_hidden_dim: capacity of the gating network
  - gated_lr: learning rate for gated models
  - lr_schedule: cosine annealing vs none
  - temp_schedule: temperature annealing for Concrete (linear anneal from temp_init to temp_final)

For each combination, runs the same goal1 variants as validate_models.py (LSPIN no/smooth,
Concrete no/smooth, L-LSPIN, L-Concrete) on both datasets with n_seeds seeds per split.

Results are written to a timestamped output directory under:
  data/runs/validate_goal1_tuning_<timestamp>/

Usage:
  conda run -n musevo python analyses/validate_goal1_tuning_sweep.py \
      --datasets kipan brca \
      --n-seeds 3 \
      --n-workers 4
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
if SDS_SRC not in sys.path:
    sys.path.insert(0, SDS_SRC)

import sparsedeepsurv as sds

PAPER_ROOT = Path(__file__).resolve().parents[1]
KIPAN_DATA = PAPER_ROOT / "data" / "processed" / "kipan_20260209_213604"
BRCA_DATA = PAPER_ROOT / "data" / "processed" / "tcga_brca20260214_001423"
KIPAN_SHOWCASE = (
    PAPER_ROOT / "data" / "runs"
    / "ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219"
    / "selected_showcase_configs.csv"
)
BRCA_SHOWCASE = (
    PAPER_ROOT / "data" / "runs"
    / "ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115"
    / "selected_showcase_configs.csv"
)

# Dataset-specific hyperparameters matching validate_models.py defaults
DATASET_DEFAULTS = {
    "kipan": {
        "data_path": KIPAN_DATA,
        "showcase_path": KIPAN_SHOWCASE,
        "knn_k": 8,
        "lspin_sigma": 0.22,
        "concrete_sigma": 0.15,
    },
    "brca": {
        "data_path": BRCA_DATA,
        "showcase_path": BRCA_SHOWCASE,
        "knn_k": 5,
        "lspin_sigma": 0.10,
        "concrete_sigma": 0.10,
    },
}

# Sweep grid
GATING_HIDDEN_DIMS = [64, 128, 256, 512]
GATED_LRS = [5e-4, 1e-3, 2e-3]
LR_SCHEDULES = [None, "cosine"]
# Temperature annealing only for Concrete; None = no annealing
TEMP_SCHEDULES = [None, "linear"]
TEMP_INIT = 0.8     # starting temperature when annealing
TEMP_FINAL = 0.05   # final temperature when annealing

MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-5
LR_WARMUP_EPOCHS = 10
HIDDEN_DIM = 128    # predictor hidden dim (fixed)


def _pick_showcase_row(showcase_csv: Path, family: str, selection: str, sigma_target: float) -> Dict:
    df = pd.read_csv(showcase_csv)
    aliases = {"lspin": {"hardsigmoid", "lspin"}, "concrete": {"concrete"}}.get(family.lower(), {family.lower()})
    sub = df[
        (df["family"].astype(str).str.lower().isin(aliases))
        & (df["selection"].astype(str).str.lower() == selection.lower())
    ].copy()
    if sub.empty:
        raise ValueError(f"No showcase rows for family={family}, selection={selection} in {showcase_csv}")
    sub["sigma_dist"] = (sub["gate_sigma"].astype(float) - float(sigma_target)).abs()
    return sub.sort_values(["sigma_dist", "mean_test_c"], ascending=[True, False]).iloc[0].to_dict()


def _load_split(data: Dict[str, np.ndarray], split_seed: int, knn_k: int) -> Dict:
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
        "input_dim": int(X_train.shape[1]),
        "A_train": sds.build_knn_adjacency_csr(
            sds.as_torch(X_train[i_tr]), k=int(knn_k), pca_dim=50, metric="cosine", symmetrize=True
        ),
    }


def _run_one(task: Dict, device: str) -> Dict:
    for _p in [SDS_SRC]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import sparsedeepsurv as sds_w

    dataset_name = task["dataset"]
    defaults = DATASET_DEFAULTS[dataset_name]

    if dataset_name == "kipan":
        data = sds_w.load_kipan_split_artifacts(defaults["data_path"])
    else:
        data = sds_w.load_brca_split_artifacts(defaults["data_path"])

    split = _load_split(data, task["split_seed"], defaults["knn_k"])
    sds_w.set_all_seeds(task["train_seed"])

    gate_type = task["gate_type"]
    temperature = task["temperature"]

    model = sds_w.make_model(
        input_dim=int(split["input_dim"]),
        gate_type=gate_type,
        gate_sigma=task["gate_sigma"],
        temperature=temperature,
        concrete_mode="ste",
        predictor=task["predictor"],
        gating_hidden_dim=task["gating_hidden_dim"],
    ).to(device)

    temp_schedule = task.get("temp_schedule")
    temp_init = float(TEMP_INIT) if (temp_schedule is not None and gate_type == "concrete") else None
    temp_final = float(TEMP_FINAL) if (temp_schedule is not None and gate_type == "concrete") else None

    config = sds_w.GatedTrainConfig(
        lr=task["lr"],
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lambda_sparse=task["lambda_sparse"],
        lambda_sample_smooth=task["lambda_sample_smooth"],
        lr_schedule=task.get("lr_schedule"),
        lr_warmup_epochs=LR_WARMUP_EPOCHS if task.get("lr_schedule") == "cosine" else 0,
        temp_schedule=temp_schedule,
        temp_init=temp_init,
        temp_final=temp_final,
    )

    info = sds_w.train_gated_deepsurv(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        split["Xt_test"], split["tt_test"], split["et_test"],
        A_sample_train=split["A_train"],
        config=config,
        device=device,
        verbose=False,
    )

    _, g_det, hard_g, _ = sds_w.get_gates(model, split["Xt_test"], device=device, hard_threshold=0.5, batch_size=512)
    G_hard = hard_g.numpy()

    return {
        **{k: task[k] for k in [
            "dataset", "variant_key", "model_family", "gate_type", "predictor",
            "gate_sigma", "lambda_sparse", "lambda_sample_smooth",
            "gating_hidden_dim", "lr", "lr_schedule", "temp_schedule",
            "split_seed", "train_seed",
        ]},
        "test_cindex": float(info.get("test_cindex", np.nan)),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", -1)),
        "mean_Khard": float(G_hard.sum(axis=1).mean()),
        "status": "ok",
    }


def _worker(tasks: List[Dict], device: str, worker_id: int, out_dir: Path) -> List[Dict]:
    rows = []
    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            row = _run_one(task, device)
            row["elapsed_sec"] = float(time.time() - t0)
            rows.append(row)
            if i % 5 == 0 or i == len(tasks):
                print(f"[w{worker_id}] {i}/{len(tasks)} {task['dataset']} {task['variant_key']} "
                      f"ghdim={task['gating_hidden_dim']} lr={task['lr']} "
                      f"lrsched={task['lr_schedule']} tsched={task['temp_schedule']} "
                      f"C={row['test_cindex']:.4f} elapsed={row['elapsed_sec']:.1f}s", flush=True)
        except Exception as exc:
            rows.append({**{k: task.get(k) for k in task}, "status": "failed", "error": str(exc),
                         "elapsed_sec": float(time.time() - t0), "test_cindex": np.nan,
                         "best_val_cindex": np.nan, "best_epoch": -1, "mean_Khard": np.nan})
            print(f"[w{worker_id}] FAILED {task['dataset']} {task['variant_key']}: {exc}", flush=True)
    # Write partial results
    partial_path = out_dir / f"_partial_w{worker_id}.csv"
    pd.DataFrame(rows).to_csv(partial_path, index=False)
    return rows


def _build_tasks(datasets: List[str], n_seeds: int, seed_base: int) -> List[Dict]:
    tasks = []
    for dataset_name in datasets:
        defaults = DATASET_DEFAULTS[dataset_name]
        showcase = defaults["showcase_path"]

        lspin_no = _pick_showcase_row(showcase, "LSPIN", "nosmooth", defaults["lspin_sigma"])
        lspin_sm = _pick_showcase_row(showcase, "LSPIN", "smooth", defaults["lspin_sigma"])
        concrete_no = _pick_showcase_row(showcase, "Concrete", "nosmooth", defaults["concrete_sigma"])
        concrete_sm = _pick_showcase_row(showcase, "Concrete", "smooth", defaults["concrete_sigma"])

        variants = [
            ("lspin_nosmooth",  "LSPIN",     "lspin_tf", "mlp",    float(lspin_no["gate_sigma"]),    float(lspin_no["lambda_sparse"]),    0.0,                                    0.22),
            ("lspin_smooth",    "LSPIN",     "lspin_tf", "mlp",    float(lspin_sm["gate_sigma"]),    float(lspin_sm["lambda_sparse"]),    float(lspin_sm["lambda_sample_smooth"]), 0.22),
            ("concrete_nosmooth","Concrete", "concrete", "mlp",    float(concrete_no["gate_sigma"]), float(concrete_no["lambda_sparse"]), 0.0,                                    0.5),
            ("concrete_smooth", "Concrete", "concrete", "mlp",    float(concrete_sm["gate_sigma"]), float(concrete_sm["lambda_sparse"]), float(concrete_sm["lambda_sample_smooth"]), 0.5),
            ("llspin_smooth",   "L-LSPIN",  "lspin_tf", "linear", float(lspin_sm["gate_sigma"]),    float(lspin_sm["lambda_sparse"]),    float(lspin_sm["lambda_sample_smooth"]), 0.22),
            ("lconcrete_smooth","L-Concrete","concrete", "linear", float(concrete_sm["gate_sigma"]), float(concrete_sm["lambda_sparse"]), float(concrete_sm["lambda_sample_smooth"]), 0.5),
        ]

        for (vkey, family, gate_type, predictor, gate_sigma, lambda_sparse, lambda_smooth, base_temp) in variants:
            for ghdim, lr, lr_sched in itertools.product(GATING_HIDDEN_DIMS, GATED_LRS, LR_SCHEDULES):
                # Temperature sweep: only Concrete benefits from annealing
                temp_schedules_here = TEMP_SCHEDULES if gate_type == "concrete" else [None]
                for temp_sched in temp_schedules_here:
                    # temperature: use base_temp unless annealing (in which case trainer sets temp_init)
                    temperature = float(TEMP_FINAL) if temp_sched is not None else float(base_temp)
                    sweep_key = (
                        f"{vkey}_ghdim{ghdim}_lr{str(lr).replace('.','p')}"
                        f"_lrsched{lr_sched or 'none'}_tsched{temp_sched or 'none'}"
                    )
                    for seed_offset in range(n_seeds):
                        seed = seed_base + seed_offset
                        tasks.append({
                            "dataset": dataset_name,
                            "variant_key": sweep_key,
                            "model_family": family,
                            "gate_type": gate_type,
                            "predictor": predictor,
                            "gate_sigma": gate_sigma,
                            "temperature": temperature,
                            "lambda_sparse": lambda_sparse,
                            "lambda_sample_smooth": lambda_smooth,
                            "gating_hidden_dim": ghdim,
                            "lr": lr,
                            "lr_schedule": lr_sched,
                            "temp_schedule": temp_sched,
                            "split_seed": seed,
                            "train_seed": seed,
                        })

    tasks.sort(key=lambda t: (t["dataset"], t["variant_key"], t["split_seed"]))
    return tasks


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "model_family", "gate_type", "predictor",
        "gating_hidden_dim", "lr", "lr_schedule", "temp_schedule",
        "lambda_sparse", "lambda_sample_smooth",
    ]
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    agg = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("test_cindex", "size"),
            mean_test_cindex=("test_cindex", "mean"),
            std_test_cindex=("test_cindex", "std"),
            mean_Khard=("mean_Khard", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )
    agg["ci95"] = 1.96 * agg["std_test_cindex"] / np.sqrt(agg["n_runs"].clip(lower=1))
    return agg.sort_values(["dataset", "model_family", "mean_test_cindex"], ascending=[True, True, False])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter tuning sweep for gated survival models.")
    p.add_argument("--datasets", nargs="+", default=["kipan", "brca"], choices=["kipan", "brca"])
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--devices", nargs="+", default=["cpu"],
                   help="List of devices to use, e.g. --devices cuda:0 cuda:2 cuda:4")
    p.add_argument("--workers-per-device", type=int, default=2,
                   help="Number of parallel workers per device")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PAPER_ROOT / "data" / "runs" / f"validate_goal1_tuning_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    tasks = _build_tasks(args.datasets, args.n_seeds, args.seed_base)
    print(f"Total tasks: {len(tasks)}", flush=True)

    # Assign a device to each worker, round-robin across devices
    device_list = args.devices
    n_workers = len(device_list) * args.workers_per_device
    n_workers = min(n_workers, len(tasks))
    worker_devices = [(device_list[wid % len(device_list)]) for wid in range(n_workers)]

    print(f"Workers: {n_workers} across devices {device_list} "
          f"({args.workers_per_device} per device)", flush=True)

    chunks = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        chunks[i % n_workers].append(task)

    all_rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {
            pool.submit(_worker, chunk, worker_devices[wid], wid, out_dir): wid
            for wid, chunk in enumerate(chunks) if chunk
        }
        for fut in as_completed(futs):
            wid = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"Worker {wid} ({worker_devices[wid]}) done ({len(rows)} results)", flush=True)
            except Exception as exc:
                print(f"Worker {wid} ({worker_devices[wid]}) CRASHED: {exc}", flush=True)

    df = pd.DataFrame(all_rows)
    runs_path = out_dir / "tuning_runs.csv"
    df.to_csv(runs_path, index=False)
    print(f"Saved runs: {runs_path}")

    summary = _summarize(df)
    summary_path = out_dir / "tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print top-5 per dataset per model family
    if not summary.empty:
        for dataset_name in args.datasets:
            dsub = summary[summary["dataset"] == dataset_name]
            print(f"\n=== {dataset_name.upper()} top configs by model family ===")
            for family, fsub in dsub.groupby("model_family"):
                top = fsub.nlargest(3, "mean_test_cindex")[
                    ["gating_hidden_dim", "lr", "lr_schedule", "temp_schedule",
                     "mean_test_cindex", "ci95", "mean_Khard", "n_runs"]
                ]
                print(f"\n  {family}:")
                print(top.to_string(index=False))


if __name__ == "__main__":
    main()
