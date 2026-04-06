#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from render_adaptive_manuscript_figures import (
    _load_dataset,
    _prepare_split,
    _retrain_selected_model,
)
from cleaned_analyses.pipelines import repro_survival_pipeline as rp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug adaptive selected heatmap rendering.")
    p.add_argument("--dataset", choices=["kipan", "brca", "pancan"], required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--selection", choices=["nosmooth", "smooth"], required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--knn-k", type=int, default=8)
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    selected = pd.read_csv(args.results_dir / "selected_comparison_configs.csv")
    row = selected[
        (selected["family"] == args.family)
        & (selected["selection"] == args.selection)
    ].iloc[0]
    print(
        "selected row:",
        row[
            [
                "family",
                "selection",
                "gate_sigma",
                "lambda_sparse",
                "lambda_sample_smooth",
                "mean_test_c",
                "mean_Khard",
            ]
        ].to_dict(),
        flush=True,
    )

    print("loading dataset...", flush=True)
    data = _load_dataset(args.dataset, args.outdir)
    print("preparing split...", flush=True)
    split = _prepare_split(data, seed=int(args.seed), knn_k=int(args.knn_k))

    print("retraining selected model...", flush=True)
    t0 = time.time()
    model, info = _retrain_selected_model(
        row=row,
        split=split,
        seed=int(args.seed),
        device=str(args.device),
        args=args,
    )
    print(f"retraining finished in {time.time() - t0:.2f}s", flush=True)
    print(f"info: {info}", flush=True)

    stem = f"{args.family.lower()}_{args.selection}"
    outpath = args.results_dir / f"_debug_heatmap_{stem}.png"
    print(f"saving heatmap -> {outpath}", flush=True)
    t1 = time.time()
    cluster = rp.save_fig_gate_heatmap_with_histo(
        model,
        Xt=split["Xt_test"],
        histo=data["histo_test"],
        gene_names=data["gene_names"],
        device=str(args.device),
        outpath=outpath,
        title=f"DEBUG {args.family} {args.selection}",
        view="hard",
        min_frac_on=float(args.heatmap_min_frac_on),
        max_frac_on=float(args.heatmap_max_frac_on),
        top_genes=int(args.heatmap_top_genes),
    )
    print(f"heatmap saved in {time.time() - t1:.2f}s", flush=True)
    if cluster is not None:
        order_csv = args.results_dir / f"_debug_heatmap_{stem}_order.csv"
        cluster.to_csv(order_csv, index=False)
        print(f"sample order -> {order_csv}", flush=True)


if __name__ == "__main__":
    main()
