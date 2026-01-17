from __future__ import annotations

"""Minimal PyTorch Federated Learning (FedAvg) for NSL-KDD (binary).

This is intentionally lightweight:
- Uses the existing preprocessing pipeline to get a fixed feature representation.
- Trains a small MLP on each client locally.
- Aggregates model parameters via FedAvg.

It’s designed for Phase 4 (local IDS model) + Phase 5 (baseline FL).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for nsl_kdd.torch_fl. Install torch or ensure it's available in your environment."
    ) from e

from preprocessing.preprocess import fit_preprocess, transform_with_preprocessor


class MLPIDS(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class FLConfig:
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 1337


@dataclass(frozen=True)
class FLResult:
    metrics: Dict[str, float]
    history: List[Dict[str, float]]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_loader(X: pd.DataFrame, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)
    yt = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _set_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No client states provided")
    if len(states) != len(weights):
        raise ValueError("states and weights must have same length")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Sum of weights must be > 0")

    keys = list(states[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for st, w in zip(states, weights):
            term = st[k] * (w / total)
            acc = term if acc is None else acc + term
        out[k] = acc  # type: ignore[assignment]
    return out


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> None:
    model.train()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate(model: nn.Module, X: pd.DataFrame, y: np.ndarray, *, device: str) -> Dict[str, float]:
    model.eval()
    model.to(device)

    Xt = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    logits = model(Xt).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }


def train_fedavg_binary(
    client_dfs: Iterable[pd.DataFrame],
    test_df: pd.DataFrame,
    *,
    config: FLConfig = FLConfig(),
) -> FLResult:
    """Train a binary FedAvg model.

    Preprocessing is fit once on the *union* of client data (equivalent to fitting on
    global train), then applied consistently to each client and to the test set.
    """

    client_dfs = list(client_dfs)
    if not client_dfs:
        raise ValueError("Expected at least one client dataframe")

    _set_seed(config.seed)

    # Fit preprocessing on combined train.
    combined_train = pd.concat(client_dfs, ignore_index=True)
    X_train_all, y_train_all, prep = fit_preprocess(combined_train)

    # Transform each client using the shared preprocessor.
    clients_xy: List[Tuple[pd.DataFrame, np.ndarray]] = []
    start = 0
    for cdf in client_dfs:
        end = start + len(cdf)
        # Slice from the already-preprocessed combined to avoid refit/transform cost variations.
        Xc = X_train_all.iloc[start:end].reset_index(drop=True)
        yc = y_train_all[start:end]
        clients_xy.append((Xc, yc))
        start = end

    X_test, y_test = transform_with_preprocessor(test_df, prep)

    n_features = X_train_all.shape[1]
    global_model = MLPIDS(n_features)

    history: List[Dict[str, float]] = []

    for rnd in range(config.rounds):
        client_states: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []

        for Xc, yc in clients_xy:
            # Initialize local model from global weights each round
            local_model = MLPIDS(n_features)
            _set_state(local_model, _get_state(global_model))

            loader = _to_loader(Xc, yc, batch_size=config.batch_size, shuffle=True)
            local_train(
                local_model,
                loader,
                epochs=config.local_epochs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                device=config.device,
            )

            client_states.append(_get_state(local_model))
            weights.append(float(len(Xc)))

        # Aggregate
        new_state = fedavg(client_states, weights)
        _set_state(global_model, new_state)

        metrics = evaluate(global_model, X_test, y_test, device=config.device)
        metrics["round"] = float(rnd + 1)
        history.append(metrics)

    final_metrics = history[-1].copy() if history else evaluate(global_model, X_test, y_test, device=config.device)
    return FLResult(metrics=final_metrics, history=history)
