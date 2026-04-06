#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
import sys
if SDS_SRC not in sys.path:
    sys.path.insert(0, SDS_SRC)

from sparsedeepsurv.eval.concordance import eval_cindex_and_risk
from sparsedeepsurv.train.trainer import negative_partial_log_likelihood
from sparsedeepsurv.train.seed import set_seed


class DeepSurvMLP(nn.Module):
    """
    Paper-local DeepSurv MLP baseline matching the old lspin_pytorch behavior:
    two hidden layers, optional dropout, explicit L1 on the input layer, and
    Cox partial likelihood training with early stopping on validation C-index.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout_p: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(h)))
            layers.append(nn.ReLU())
            if float(dropout_p) > 0:
                layers.append(nn.Dropout(float(dropout_p)))
            prev_dim = int(h)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.input_layer = self.net[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        risk = self.net(x).squeeze(-1)
        return torch.clamp(risk, min=-20.0, max=20.0)


@dataclass
class MLPTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_l1_input: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    grad_clip: float = 5.0


def train_deepsurv_mlp_l1(
    model: DeepSurvMLP,
    X_train: torch.Tensor,
    time_train: torch.Tensor,
    event_train: torch.Tensor,
    X_val: torch.Tensor,
    time_val: torch.Tensor,
    event_val: torch.Tensor,
    *,
    config: MLPTrainConfig,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, float]:
    model = model.to(device)
    X_train = X_train.to(device)
    time_train = time_train.to(device)
    event_train = event_train.to(device)
    X_val = X_val.to(device)
    time_val = time_val.to(device)
    event_val = event_val.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    train_ds = TensorDataset(X_train, time_train, event_train)
    train_dl = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True)

    best_state = None
    best_val_c = -np.inf
    best_epoch = -1
    epochs_ran = 0

    for epoch in range(int(config.max_epochs)):
        model.train()
        for xb, tb, eb in train_dl:
            xb = xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            if eb.sum().item() == 0:
                continue

            opt.zero_grad()
            risk = model(xb)
            npll = negative_partial_log_likelihood(risk, tb, eb)
            l1 = float(config.lambda_l1_input) * model.input_layer.weight.abs().sum()
            loss = npll + l1
            loss.backward()
            if float(config.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            opt.step()

        val_c, _ = eval_cindex_and_risk(model, X_val, time_val, event_val, device=device)
        epochs_ran = epoch + 1
        if verbose:
            print(f"[mlp] epoch={epoch} val_c={val_c:.4f}", flush=True)

        if val_c > best_val_c + 1e-4:
            best_val_c = float(val_c)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= int(config.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_cindex": float(best_val_c),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
    }


@torch.no_grad()
def eval_cindex(
    model: nn.Module,
    X: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    device: str = "cpu",
) -> tuple[float, np.ndarray]:
    return eval_cindex_and_risk(model, X, time, event, device=device)


def make_seeded_mlp(
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout_p: float,
    seed: int,
) -> DeepSurvMLP:
    set_seed(int(seed))
    return DeepSurvMLP(input_dim=input_dim, hidden_dims=tuple(hidden_dims), dropout_p=float(dropout_p))
