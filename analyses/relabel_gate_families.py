#!/usr/bin/env python3
"""
Rewrite finished CSV outputs to use the clearer gate-family names:

  HardSigmoid -> LSPIN

This script is intentionally post-hoc and file-based so we can update already
finished run directories without retraining anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXACT_VALUE_MAP = {
    "HardSigmoid": "LSPIN",
}

STRING_REPLACEMENTS = [
    ("goal0_hardsigmoid_", "goal0_lspin_"),
    ("goal1_hardsigmoid_", "goal1_lspin_"),
    ("_hardsigmoid_", "_lspin_"),
    ("HardSigmoid ", "LSPIN "),
    ("HardSigmoid-", "LSPIN-"),
    ("HardSigmoid", "LSPIN"),
]


def _relabel_value(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        out = EXACT_VALUE_MAP.get(value, value)
        for old, new in STRING_REPLACEMENTS:
            out = out.replace(old, new)
        return out
    return value


def _rewrite_csv(path: Path, *, backup_suffix: str, dry_run: bool) -> bool:
    df = pd.read_csv(path)
    changed = False
    for col in df.columns:
        if df[col].dtype == object:
            new_col = df[col].map(_relabel_value)
            if not new_col.equals(df[col]):
                df[col] = new_col
                changed = True
    if not changed:
        return False
    if dry_run:
        print(f"[dry-run] would relabel {path}")
        return True
    backup = path.with_name(path.name + backup_suffix)
    if not backup.exists():
        path.rename(backup)
    df.to_csv(path, index=False)
    print(f"[rewrite] {path} (backup: {backup.name})")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Relabel gate-family names in finished run CSV outputs")
    p.add_argument("results_dirs", nargs="+", type=Path)
    p.add_argument("--backup-suffix", default=".bak")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    total = 0
    for results_dir in args.results_dirs:
        if not results_dir.exists():
            raise FileNotFoundError(results_dir)
        print(f"[dir] {results_dir}")
        for csv_path in sorted(results_dir.glob("*.csv")):
            total += int(_rewrite_csv(csv_path, backup_suffix=args.backup_suffix, dry_run=args.dry_run))

    print(f"[done] files_rewritten={total}")


if __name__ == "__main__":
    main()
