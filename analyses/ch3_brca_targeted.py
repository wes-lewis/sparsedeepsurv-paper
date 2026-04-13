#!/usr/bin/env python3
"""
Chapter 3 BRCA targeted analysis.

This is a placeholder entry point for a BRCA targeted analysis that was not yet
fully migrated into the paper archive. The canonical BRCA targeted outputs used
in the dissertation are preserved under `data/runs/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# All model/training/eval logic comes from the package.
# TODO: import and wire up sparsedeepsurv calls here during migration.
import sparsedeepsurv  # noqa: F401


def main(config_path: Path, out_tables: Path, out_figures: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "This wrapper has not been migrated into the self-contained paper archive. "
        "Use the preserved canonical outputs under data/runs/ instead."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-tables", type=Path, default=Path("tables"))
    parser.add_argument("--out-figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.config, args.out_tables, args.out_figures)
