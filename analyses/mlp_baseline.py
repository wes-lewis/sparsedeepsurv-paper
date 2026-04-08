#!/usr/bin/env python3
"""Compatibility shim for the package-local dense MLP baseline.

New analysis code should import these objects from ``sparsedeepsurv`` directly.
This module remains so older paper scripts do not break while the validation
pipeline moves to the self-contained package API.
"""
from __future__ import annotations

import sys

SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
if SDS_SRC not in sys.path:
    sys.path.insert(0, SDS_SRC)

from sparsedeepsurv import (  # noqa: E402,F401
    DeepSurvMLP,
    MLPTrainConfig,
    eval_mlp_cindex as eval_cindex,
    make_seeded_mlp,
    train_deepsurv_mlp_l1,
)

__all__ = [
    "DeepSurvMLP",
    "MLPTrainConfig",
    "eval_cindex",
    "make_seeded_mlp",
    "train_deepsurv_mlp_l1",
]
