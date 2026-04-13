#!/usr/bin/env python3
"""
LSPIN regularization sweep for Goal 1 (KIPAN + BRCA).

Motivated by the finding that LSPIN on BRCA suffers from gating-mediated overfitting:
val-test gap of ~0.17 (vs ~0.07 for Concrete), with corr(val_C, gap)=0.80 and
corr(val_C, test_C)=0.20. The gating network memorizes validation-fold feature structure
rather than generalizing.

Sweeps over (LSPIN variants only):
  - lambda_sparse:          L1 penalty weight on expected gate count
  - gate_sigma:             noise std in the stochastic LSPIN gate
  - gate_hidden_dropout_p:  dropout in the gating network hidden layer

Fixed at best values from prior tuning sweep:
  - gating_hidden_dim = 64
  - lr = 2e-3
  - lr_schedule = None
  - temp_schedule = None (LSPIN does not use temperature annealing)

Usage:
  conda run -n musevo python analyses/validate_lspin_reg_sweep.py \\
      --datasets kipan brca \\
      --n-seeds 3 \\
      --devices cuda:0 cuda:2 cuda:4 cuda:6 \\
      --workers-per-device 3
"""
from __future__ import annotations

import argparse
import itertools
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

DATASET_DEFAULTS = {
    "kipan": {
        "data_path": KIPAN_DATA,
        "showcase_path": KIPAN_SHOWCASE,
        "knn_k": 8,
        "lspin_sigma_ref": 0.22,
    },
    "brca": {
        "data_path": BRCA_DATA,
        "showcase_path": BRCA_SHOWCASE,
        "knn_k": 5,
        "lspin_sigma_ref": 0.10,
    },
}

# Sweep grid — LSPIN-focused regularization axes
LSPIN_LAMBDAS = [0.007, 0.02, 0.05, 0.1]
LSPIN_SIGMAS = [0.1, 0.22, 0.5]
LSPIN_DROPOUTS = [0.0, 0.2, 0.4]

# Fixed at best values from prior tuning sweep
GATING_HIDDEN_DIM = 64
LR = 2e-3
LR_SCHEDULE = None

MAX_EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-5
SEED_BASE = 42


def _pick_showcase_row(showcase_csv: Path, family: str, selection: str) -> Dict:
    df = pd.read_csv(showcase_csv)
    aliases = {"lspin": {"hardsigmoid", "lspin"}}.get(family.lower(), {family.lower()})
    sub = df[
        (df["family"].astype(str).str.lower().isin(aliases))
        & (df["selection"].astype(str).str.lower() == selection.lower())
    ].copy()
    if sub.empty:
        raise ValueError(f"No showcase rows for family={family}, selection={selection} in {showcase_csv}")
    return sub.sort_values("mean_test_c", ascending=False).iloc[0].to_dict()


def _load_split(data: Dict, split_seed: int, knn_k: int) -> Dict:
    from sklearn.model_selection import ShuffleSplit

    X_train = data["X_train"]
    strat_key = np.array(
        [f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])], dtype=object
    )
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

    model = sds_w.make_model(
        input_dim=int(split["input_dim"]),
        gate_type="lspin_tf",
        gate_sigma=task["gate_sigma"],
        temperature=0.22,  # unused by lspin_tf but required arg
        concrete_mode="ste",
        predictor=task["predictor"],
        gating_hidden_dim=GATING_HIDDEN_DIM,
        gate_hidden_dropout_p=task["gate_hidden_dropout_p"],
    ).to(device)

    config = sds_w.GatedTrainConfig(
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lambda_sparse=task["lambda_sparse"],
        lambda_sample_smooth=task["lambda_sample_smooth"],
        lr_schedule=LR_SCHEDULE,
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
            "dataset", "variant_key", "model_family", "predictor",
            "gate_sigma", "lambda_sparse", "lambda_sample_smooth",
            "gate_hidden_dropout_p", "split_seed", "train_seed",
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
                print(
                    f"[w{worker_id}] {i}/{len(tasks)} {task['dataset']} {task['variant_key']} "
                    f"lam={task['lambda_sparse']:.3f} sigma={task['gate_sigma']:.2f} "
                    f"drop={task['gate_hidden_dropout_p']:.1f} "
                    f"C={row['test_cindex']:.4f} elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            rows.append({
                **{k: task.get(k) for k in task},
                "status": "failed", "error": str(exc),
                "elapsed_sec": float(time.time() - t0),
                "test_cindex": np.nan,
                "best_val_cindex": np.nan,
                "best_epoch": -1,
                "mean_Khard": np.nan,
            })
            print(f"[w{worker_id}] FAILED {task['dataset']} {task['variant_key']}: {exc}", flush=True)
    partial_path = out_dir / f"_partial_w{worker_id}.csv"
    pd.DataFrame(rows).to_csv(partial_path, index=False)
    return rows


def _build_tasks(datasets: List[str], n_seeds: int, seed_base: int) -> List[Dict]:
    tasks = []
    for dataset_name in datasets:
        defaults = DATASET_DEFAULTS[dataset_name]
        showcase = defaults["showcase_path"]

        lspin_no = _pick_showcase_row(showcase, "LSPIN", "nosmooth")
        lspin_sm = _pick_showcase_row(showcase, "LSPIN", "smooth")

        variants = [
            ("lspin_nosmooth", "LSPIN",  "mlp",    float(lspin_no["lambda_sparse"]), 0.0),
            ("lspin_smooth",   "LSPIN",  "mlp",    float(lspin_sm["lambda_sparse"]), float(lspin_sm["lambda_sample_smooth"])),
            ("llspin_smooth",  "L-LSPIN","linear", float(lspin_sm["lambda_sparse"]), float(lspin_sm["lambda_sample_smooth"])),
        ]

        for (vkey, family, predictor, _base_lam, lambda_smooth) in variants:
            for lam, sigma, dropout in itertools.product(LSPIN_LAMBDAS, LSPIN_SIGMAS, LSPIN_DROPOUTS):
                sweep_key = (
                    f"{vkey}_lam{lam:.3f}_sig{sigma:.2f}_drop{dropout:.1f}"
                )
                for seed_offset in range(n_seeds):
                    seed = seed_base + seed_offset
                    tasks.append({
                        "dataset": dataset_name,
                        "variant_key": sweep_key,
                        "model_family": family,
                        "predictor": predictor,
                        "gate_sigma": sigma,
                        "lambda_sparse": lam,
                        "lambda_sample_smooth": lambda_smooth,
                        "gate_hidden_dropout_p": dropout,
                        "split_seed": seed,
                        "train_seed": seed,
                    })

    tasks.sort(key=lambda t: (t["dataset"], t["variant_key"], t["split_seed"]))
    return tasks


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "model_family", "predictor",
        "lambda_sparse", "gate_sigma", "gate_hidden_dropout_p", "lambda_sample_smooth",
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
            mean_val_cindex=("best_val_cindex", "mean"),
            mean_Khard=("mean_Khard", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )
    agg["ci95"] = 1.96 * agg["std_test_cindex"] / np.sqrt(agg["n_runs"].clip(lower=1))
    agg["val_test_gap"] = agg["mean_val_cindex"] - agg["mean_test_cindex"]
    return agg.sort_values(["dataset", "model_family", "mean_test_cindex"], ascending=[True, True, False])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["kipan", "brca"])
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--workers-per-device", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PAPER_ROOT / "data" / "runs" / f"validate_lspin_reg_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    tasks = _build_tasks(args.datasets, args.n_seeds, args.seed_base)
    print(f"Total tasks: {len(tasks)}", flush=True)

    n_workers = len(args.devices) * args.workers_per_device
    worker_devices = [
        args.devices[wid % len(args.devices)] for wid in range(n_workers)
    ]

    # Round-robin chunk tasks across workers
    chunks: List[List[Dict]] = [[] for _ in range(n_workers)]
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
                print(f"Worker {wid} done ({len(rows)} tasks)", flush=True)
            except Exception as exc:
                print(f"Worker {wid} raised: {exc}", flush=True)

    df = pd.DataFrame(all_rows)
    runs_path = out_dir / "lspin_reg_runs.csv"
    df.to_csv(runs_path, index=False)
    print(f"Saved runs: {runs_path}")

    summary = _summarize(df)
    summary_path = out_dir / "lspin_reg_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Print top 5 per dataset per family, ranked by test C-index
    for ds in args.datasets:
        print(f"\n=== {ds.upper()} — Top 5 per family ===")
        sub = summary[summary["dataset"] == ds]
        for fam in sub["model_family"].unique():
            top = sub[sub["model_family"] == fam].nlargest(5, "mean_test_cindex")
            print(f"  {fam}:")
            for _, r in top.iterrows():
                print(
                    f"    lam={r.lambda_sparse:.3f}  sig={r.gate_sigma:.2f}  "
                    f"drop={r.gate_hidden_dropout_p:.1f}  "
                    f"C={r.mean_test_cindex:.4f}±{r.ci95:.4f}  "
                    f"gap={r.val_test_gap:.4f}  Khard={r.mean_Khard:.0f}"
                )


if __name__ == "__main__":
    main()
