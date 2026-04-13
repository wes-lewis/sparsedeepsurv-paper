#!/usr/bin/env python3
"""Post hoc paired C-index comparisons from saved validation runs.

This script performs matched run-level comparisons using the saved
`fixed_init_eval` rows in a validation `runs.csv`. It is intended for the
common case where aggregate validation summaries were preserved but per-patient
test risk scores were not. As a result, the analysis is explicitly *not* a
patient-level paired bootstrap or a Noether-style concordance test.

Instead, for each requested model pair, the script:

1. Selects runs from the same dataset and `fixed_init_eval` stage.
2. Matches the two models on the intersection of `train_seed`.
3. Compares the paired saved `test_cindex` values across those matched runs.

Reported statistics include the mean paired difference, a t-based 95% interval
for that paired difference, a paired t test, and a Wilcoxon signed-rank test.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS


DEFAULT_RUN_DIR = CANONICAL_RUNS["validate_goal1_gentle_all_kipan_brca"]
DEFAULT_OUTDIR_NAME = "posthoc_paired_cindex"

DEFAULT_COMPARISONS = {
    "brca": [
        ("goal1_linear_cox", "goal1_smooth_llspin_lamx0p75"),
        ("goal1_linear_cox", "goal1_smooth_lconcrete_lamx0p75"),
        ("goal1_linear_cox", "mlp"),
        ("mlp", "goal1_smooth_llspin_lamx0p75"),
        ("mlp", "goal1_smooth_lconcrete_lamx0p75"),
    ],
    "kipan": [
        ("goal1_linear_cox", "mlp"),
        ("goal1_nosmooth_lspin_lamx1p0", "goal1_smooth_lspin_lamx1p0"),
        ("goal1_nosmooth_concrete_lamx1p0", "goal1_smooth_concrete_lamx1p0"),
    ],
}


@dataclass(frozen=True)
class ComparisonSpec:
    left: str
    right: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--dataset", choices=["brca", "kipan"], default="brca")
    p.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Variant-key pair formatted as LEFT:RIGHT. May be passed multiple times.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/posthoc_paired_cindex.",
    )
    return p.parse_args()


def _parse_comparison(text: str) -> ComparisonSpec:
    if ":" not in text:
        raise ValueError(f"Invalid comparison '{text}'; expected LEFT:RIGHT.")
    left, right = [part.strip() for part in text.split(":", 1)]
    if not left or not right:
        raise ValueError(f"Invalid comparison '{text}'; both sides must be non-empty.")
    return ComparisonSpec(left=left, right=right)


def _default_comparisons(dataset: str) -> list[ComparisonSpec]:
    return [ComparisonSpec(left=a, right=b) for a, b in DEFAULT_COMPARISONS[dataset]]


def _comparison_label(left_label: str, right_label: str) -> str:
    return f"{right_label} minus {left_label}"


def _paired_t_interval(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return mean - tcrit * se, mean + tcrit * se


def _safe_wilcoxon(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return float("nan")
    try:
        return float(stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _match_pair(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    key_cols = ["train_seed"]
    value_cols = [
        "variant_key",
        "variant_label",
        "model_family",
        "run_seed",
        "split_seed",
        "test_cindex",
        "best_val_cindex",
        "best_epoch",
        "split_block_id",
        "fixed_init_source_seed",
    ]

    left_df = (
        df[df["variant_key"] == left][key_cols + value_cols]
        .drop_duplicates(subset=key_cols)
        .rename(columns={col: f"left_{col}" for col in value_cols})
    )
    right_df = (
        df[df["variant_key"] == right][key_cols + value_cols]
        .drop_duplicates(subset=key_cols)
        .rename(columns={col: f"right_{col}" for col in value_cols})
    )
    merged = left_df.merge(right_df, on=key_cols, how="inner").sort_values(key_cols).reset_index(drop=True)
    if not merged.empty:
        merged["delta_test_cindex"] = merged["right_test_cindex"] - merged["left_test_cindex"]
    return merged


def _summarize_pair(dataset: str, matched: pd.DataFrame) -> dict[str, object]:
    diffs = matched["delta_test_cindex"].to_numpy(dtype=float)
    left_label = str(matched["left_variant_label"].iloc[0])
    right_label = str(matched["right_variant_label"].iloc[0])
    left_key = str(matched["left_variant_key"].iloc[0])
    right_key = str(matched["right_variant_key"].iloc[0])

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else float("nan")
    t_ci_lo, t_ci_hi = _paired_t_interval(diffs)
    t_res = stats.ttest_rel(
        matched["right_test_cindex"].to_numpy(dtype=float),
        matched["left_test_cindex"].to_numpy(dtype=float),
        nan_policy="omit",
    )
    return {
        "dataset": dataset,
        "left_variant_key": left_key,
        "left_variant_label": left_label,
        "right_variant_key": right_key,
        "right_variant_label": right_label,
        "contrast": _comparison_label(left_label, right_label),
        "n_matched_runs": int(len(matched)),
        "mean_left_test_cindex": float(matched["left_test_cindex"].mean()),
        "mean_right_test_cindex": float(matched["right_test_cindex"].mean()),
        "mean_paired_difference": mean_diff,
        "sd_paired_difference": sd_diff,
        "paired_t_pvalue": float(t_res.pvalue) if np.isfinite(t_res.pvalue) else float("nan"),
        "wilcoxon_pvalue": _safe_wilcoxon(diffs),
        "paired_t_ci95_lo": t_ci_lo,
        "paired_t_ci95_hi": t_ci_hi,
        "matched_train_seeds": ",".join(str(int(x)) for x in sorted(matched["train_seed"].unique())),
        "left_split_seeds": ",".join(str(int(x)) for x in sorted(matched["left_split_seed"].unique())),
        "right_split_seeds": ",".join(str(int(x)) for x in sorted(matched["right_split_seed"].unique())),
        "analysis_note": (
            "Matched run-level comparison on intersected saved validation replicates; "
            "not a patient-level paired bootstrap."
        ),
    }


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    runs_path = run_dir / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs.csv under {run_dir}")

    outdir = (args.outdir or (run_dir / DEFAULT_OUTDIR_NAME)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(runs_path)
    df = df[(df["dataset"] == args.dataset) & (df["task_stage"] == "fixed_init_eval")].copy()
    if df.empty:
        raise ValueError(f"No fixed_init_eval rows found for dataset '{args.dataset}' in {runs_path}")

    comparisons = [_parse_comparison(text) for text in args.comparison] if args.comparison else _default_comparisons(args.dataset)

    summary_rows: list[dict[str, object]] = []
    details_frames: list[pd.DataFrame] = []

    for comp in comparisons:
        matched = _match_pair(df, comp.left, comp.right)
        if matched.empty:
            raise ValueError(
                f"No matched runs found for comparison {comp.left}:{comp.right} "
                f"on dataset '{args.dataset}'."
            )
        if len(matched) < 2:
            raise ValueError(
                f"Only {len(matched)} matched run for comparison {comp.left}:{comp.right}; "
                "need at least 2 matched runs."
            )
        summary_rows.append(_summarize_pair(args.dataset, matched))
        matched = matched.copy()
        matched.insert(0, "dataset", args.dataset)
        matched.insert(1, "comparison", f"{comp.left}:{comp.right}")
        details_frames.append(matched)

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(details_frames, ignore_index=True)

    summary_path = outdir / f"{args.dataset}_paired_validation_cindex_summary.csv"
    details_path = outdir / f"{args.dataset}_paired_validation_cindex_pairs.csv"
    readme_path = outdir / "README.txt"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    readme_path.write_text(
        "\n".join(
            [
                "Post hoc paired validation comparison",
                "",
                "This directory contains matched run-level comparisons formed from the",
                "intersection of saved fixed_init_eval validation replicates.",
                "",
                "These summaries compare paired saved test C-index values across matched",
                "train_seed combinations. The within-training validation split seed may",
                "still differ between models and is reported in the pair-details table.",
                "These analyses are not patient-level paired bootstrap intervals and not",
                "Noether-style concordance tests.",
                "",
                f"Run directory: {run_dir}",
                f"Dataset: {args.dataset}",
                f"Comparisons: {', '.join(f'{c.left}:{c.right}' for c in comparisons)}",
                "",
                f"Summary CSV: {summary_path.name}",
                f"Pair details CSV: {details_path.name}",
            ]
        )
        + "\n"
    )

    print(f"[ok] wrote summary: {summary_path}")
    print(f"[ok] wrote pair details: {details_path}")
    print(f"[ok] wrote note: {readme_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
