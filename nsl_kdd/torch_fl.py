from __future__ import annotations

"""Minimal PyTorch Federated Learning (FedAvg) for NSL-KDD (binary).

This is intentionally lightweight:
- Uses the existing preprocessing pipeline to get a fixed feature representation.
- Trains a small MLP on each client locally.
- Aggregates model parameters via FedAvg.

Phase 6/7 extensions implemented here:
- Label-flip poisoning (one or more malicious clients, configurable)
- Robust aggregation defenses:
  - Option A: cosine-similarity trust weighting + optional outlier drop
  - Option B: update clipping + trimmed-mean / coordinate-median aggregation

It’s designed for Phase 4 (local IDS model) + Phase 5 (baseline FL) and to be
extended into a poisoning-resistant FL-IDS demo.
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

# Cross-layer feature groups (logical mapping used for Phase 7 cross-layer consistency).
# These refer to the *raw* NSL-KDD columns (before preprocessing).
CROSS_LAYER_GROUPS: Dict[str, List[str]] = {
    "network": ["duration", "src_bytes", "dst_bytes", "count", "srv_count"],
    "transport": ["protocol_type", "flag", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"],
    "application": ["service", "logged_in", "num_failed_logins", "hot", "num_compromised"],
}


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

    # ---------------- Phase 6: poisoning ----------------
    # 1-based indices of malicious clients (as shown in data/clients/client_1.csv ...). Example: [1] or [2,5]
    malicious_clients: Tuple[int, ...] = ()
    # Fraction of labels to flip in each malicious client (0..1). For binary labels: y := 1 - y
    label_flip_rate: float = 0.0

    # ---------------- Phase 7: defenses ----------------
    # Aggregation strategy
    # - "fedavg": classic FedAvg
    # - "cosine": trust-weighted FedAvg using cosine similarity (Option A)
    # - "trimmed_mean": coordinate-wise trimmed mean of client updates (Option B)
    # - "median": coordinate-wise median of client updates (Option B)
    aggregation: str = "fedavg"

    # Option A: cosine
    # If > 0, drop this many lowest-similarity clients each round (after computing similarity to mean update)
    cosine_drop_k: int = 0

    # Option B: clipping + robust reducer
    # If set, clip each client update by L2 norm before aggregation (recommended for robustness)
    clip_norm: float | None = None
    # Trim fraction for trimmed mean (0..0.49). Example: 0.2 trims 20% low + 20% high.
    trim_ratio: float = 0.2

    # Option A: trust score = a*cosine + b*loss_stability + c*cross_layer
    trust_alpha: float = 1.0
    trust_beta: float = 0.0
    trust_gamma: float = 0.5


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


def _sub_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute a - b (parameter-wise)."""
    return {k: (a[k] - b[k]) for k in a.keys()}


def _add_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (a[k] + b[k]) for k in a.keys()}


def _scale_state(a: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: (a[k] * s) for k in a.keys()}


def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for v in state.values():
        parts.append(v.reshape(-1).detach().float().cpu())
    return torch.cat(parts) if parts else torch.zeros(0)


def _l2_norm(state: Dict[str, torch.Tensor]) -> float:
    vec = _flatten_state(state)
    return float(torch.linalg.vector_norm(vec, ord=2).item())


def _clip_update(update: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
    n = _l2_norm(update)
    if n <= 0 or n <= max_norm:
        return update
    return _scale_state(update, max_norm / n)


def _cosine_similarity(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    va = _flatten_state(a)
    vb = _flatten_state(b)
    denom = float(torch.linalg.vector_norm(va, ord=2).item() * torch.linalg.vector_norm(vb, ord=2).item())
    if denom <= 0:
        return 0.0
    return float(torch.dot(va, vb).item() / denom)


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


def _coordinate_median(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No updates")
    out: Dict[str, torch.Tensor] = {}
    keys = list(updates[0].keys())
    for k in keys:
        stacked = torch.stack([u[k].detach() for u in updates], dim=0)
        out[k] = torch.median(stacked, dim=0).values
    return out


def _coordinate_trimmed_mean(updates: List[Dict[str, torch.Tensor]], *, trim_ratio: float) -> Dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("No updates")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5)")

    n = len(updates)
    trim_k = int(np.floor(trim_ratio * n))

    out: Dict[str, torch.Tensor] = {}
    keys = list(updates[0].keys())
    for k in keys:
        stacked = torch.stack([u[k].detach() for u in updates], dim=0)
        # Sort along client dimension.
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if trim_k > 0 and (2 * trim_k) < n:
            trimmed = sorted_vals[trim_k : n - trim_k]
        else:
            trimmed = sorted_vals
        out[k] = trimmed.mean(dim=0)

    return out


def _flip_labels(y: np.ndarray, *, rate: float, rng: np.random.Generator) -> np.ndarray:
    if rate <= 0:
        return y
    if rate >= 1:
        return 1 - y
    mask = rng.random(size=len(y)) < rate
    y2 = y.copy()
    y2[mask] = 1 - y2[mask]
    return y2


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

    # Binary confusion-matrix derived metric.
    # FPR = FP / (FP + TN). If there are no negatives in y, define FPR=0.
    y_true = y.astype(int)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    denom = fp + tn
    fpr = float(fp / denom) if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
    }


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _cross_layer_consistency(raw_client_df: pd.DataFrame, *, groups: Dict[str, List[str]] = CROSS_LAYER_GROUPS) -> float:
    """Heuristic cross-layer consistency score in [0, 1].

    Idea: in real attacks, multiple layers tend to shift together.
    We compute an anomaly intensity per layer using a robust z-score
    (distance from median), then reward agreement between layers.

    This is lightweight and works on NSL-KDD tabular features.
    """

    if raw_client_df.empty:
        return 0.0

    layer_intensities: Dict[str, float] = {}
    for layer, cols in groups.items():
        cols_present = _safe_cols(raw_client_df, cols)
        if not cols_present:
            continue

        block = raw_client_df[cols_present].copy()
        # Keep numeric columns only.
        num_block = block.select_dtypes(include=["number", "bool"]).astype(float)
        if num_block.shape[1] == 0:
            continue

        med = num_block.median(axis=0)
        mad = (num_block - med).abs().median(axis=0).replace(0.0, np.nan)
        z = ((num_block - med).abs() / mad).to_numpy(dtype=float)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # Aggregate into a single layer intensity (robust: median-of-abs-z).
        layer_intensities[layer] = float(np.median(z))

    if len(layer_intensities) < 2:
        return 0.0

    vals = np.array(list(layer_intensities.values()), dtype=float)
    # Higher agreement when layers have similar intensity.
    # Normalize with a soft function: score = 1 / (1 + std)
    std = float(np.std(vals))
    score = 1.0 / (1.0 + std)
    return float(np.clip(score, 0.0, 1.0))


def _bce_loss_logits(model: nn.Module, X: pd.DataFrame, y: np.ndarray, *, device: str) -> float:
    model.eval()
    model.to(device)

    Xt = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    yt = torch.tensor(y.astype(np.float32), dtype=torch.float32).to(device)

    logits = model(Xt)
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, yt)
    return float(loss.detach().cpu().item())


def train_fedavg_binary(
    client_dfs: Iterable[pd.DataFrame],
    test_df: pd.DataFrame,
    *,
    config: FLConfig = FLConfig(),
) -> FLResult:
    """Train a binary FL model.

    Preprocessing is fit once on the *union* of client data (equivalent to fitting on
    global train), then applied consistently to each client and to the test set.

    Logging:
    - Always logs locally under `runs/<run_id>/` (run.json + rounds.csv/json)
    - Optionally logs to MongoDB Atlas if `MONGODB_URI` is set
    """

    client_dfs = list(client_dfs)
    if not client_dfs:
        raise ValueError("Expected at least one client dataframe")

    _set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Keep raw client frames in the same order to compute cross-layer consistency.
    raw_clients = list(client_dfs)

    # Fit preprocessing on combined train.
    combined_train = pd.concat(raw_clients, ignore_index=True)
    X_train_all, y_train_all, prep = fit_preprocess(combined_train)

    # Transform each client using the shared preprocessor.
    clients_xy: List[Tuple[pd.DataFrame, np.ndarray]] = []
    start = 0
    for cdf in raw_clients:
        end = start + len(cdf)
        Xc = X_train_all.iloc[start:end].reset_index(drop=True)
        yc = y_train_all[start:end]
        clients_xy.append((Xc, yc))
        start = end

    X_test, y_test = transform_with_preprocessor(test_df, prep)

    n_features = X_train_all.shape[1]
    global_model = MLPIDS(n_features)

    history: List[Dict[str, float]] = []

    # ---------- Local logging (always on) ----------
    local_run = None
    local_logger = None
    try:
        from pathlib import Path

        from nsl_kdd.local_logger import LocalLogger

        local_logger = LocalLogger(Path.cwd() / "runs")
        local_run = local_logger.start_run(
            config={
                "rounds": config.rounds,
                "local_epochs": config.local_epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "device": config.device,
                "seed": config.seed,
                "malicious_clients": list(config.malicious_clients),
                "label_flip_rate": config.label_flip_rate,
                "aggregation": config.aggregation,
                "cosine_drop_k": config.cosine_drop_k,
                "clip_norm": config.clip_norm,
                "trim_ratio": config.trim_ratio,
                "trust_alpha": config.trust_alpha,
                "trust_beta": config.trust_beta,
                "trust_gamma": config.trust_gamma,
            },
            meta={
                "n_clients": len(raw_clients),
            },
        )
    except Exception:
        local_logger = None
        local_run = None

    # ---------- Mongo logging (optional) ----------
    mongo = None
    run_id = None
    try:
        from nsl_kdd.mongo_logger import build_mongo_logger_from_env

        mongo = build_mongo_logger_from_env()
    except Exception:
        mongo = None

    if mongo is not None:
        try:
            mongo.ping()
            run = mongo.create_run(
                config={
                    "rounds": config.rounds,
                    "local_epochs": config.local_epochs,
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "device": config.device,
                    "seed": config.seed,
                    "malicious_clients": list(config.malicious_clients),
                    "label_flip_rate": config.label_flip_rate,
                    "aggregation": config.aggregation,
                    "cosine_drop_k": config.cosine_drop_k,
                    "clip_norm": config.clip_norm,
                    "trim_ratio": config.trim_ratio,
                    "trust_alpha": config.trust_alpha,
                    "trust_beta": config.trust_beta,
                    "trust_gamma": config.trust_gamma,
                },
                meta={
                    "n_clients": len(raw_clients),
                },
            )
            run_id = run.run_id
        except Exception:
            mongo = None
            run_id = None

    for rnd in range(config.rounds):
        global_state = _get_state(global_model)

        client_states: List[Dict[str, torch.Tensor]] = []
        client_updates: List[Dict[str, torch.Tensor]] = []
        base_weights: List[float] = []
        similarities: List[float] = []
        trusts: List[float] = []
        loss_stabilities: List[float] = []
        cross_layer_scores: List[float] = []

        # Compute baseline global loss per client (for loss stability scoring).
        global_losses: List[float] = []
        for Xc, yc in clients_xy:
            global_losses.append(_bce_loss_logits(global_model, Xc, yc, device=config.device))

        # ---- local training ----
        for idx, ((Xc, yc), raw_df) in enumerate(zip(clients_xy, raw_clients), start=1):
            # Phase 6: label flipping on malicious clients
            yc_train = yc
            if config.label_flip_rate > 0 and idx in set(config.malicious_clients):
                sub_rng = np.random.default_rng(config.seed + 1000 * rnd + idx)
                yc_train = _flip_labels(yc, rate=float(config.label_flip_rate), rng=sub_rng)

            local_model = MLPIDS(n_features)
            _set_state(local_model, global_state)

            loader = _to_loader(Xc, yc_train, batch_size=config.batch_size, shuffle=True)
            local_train(
                local_model,
                loader,
                epochs=config.local_epochs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                device=config.device,
            )

            st = _get_state(local_model)
            upd = _sub_state(st, global_state)

            # Phase 7 Option B: clip updates
            if config.clip_norm is not None:
                upd = _clip_update(upd, float(config.clip_norm))
                st = _add_state(global_state, upd)

            # Phase 7: compute diagnostics
            # Loss stability: how much client local update changes their own loss relative to global.
            local_loss = _bce_loss_logits(local_model, Xc, yc, device=config.device)
            g_loss = global_losses[idx - 1]
            # If local update improves client loss (delta negative), that's more "stable".
            delta = local_loss - g_loss
            # Map to [0,1], reward non-increasing loss.
            loss_stability = 1.0 / (1.0 + max(0.0, delta))
            loss_stabilities.append(float(np.clip(loss_stability, 0.0, 1.0)))

            # Cross-layer consistency (raw feature groups)
            cross_layer_scores.append(_cross_layer_consistency(raw_df))

            client_states.append(st)
            client_updates.append(upd)
            base_weights.append(float(len(Xc)))

        # ---- aggregation ----
        agg = str(config.aggregation).lower()
        dropped: List[int] = []

        if agg == "fedavg":
            new_state = fedavg(client_states, base_weights)

        elif agg == "cosine":
            mean_update = _coordinate_trimmed_mean(client_updates, trim_ratio=0.0)
            similarities = [_cosine_similarity(u, mean_update) for u in client_updates]
            # cosine similarity [-1,1] -> [0,1]
            cos_trust = [max(0.0, (s + 1.0) / 2.0) for s in similarities]

            # Trust score composition
            a = float(config.trust_alpha)
            b = float(config.trust_beta)
            c = float(config.trust_gamma)
            denom = max(1e-12, a + b + c)

            trusts = [
                float(
                    np.clip(
                        (a * cos_trust[i] + b * loss_stabilities[i] + c * cross_layer_scores[i]) / denom,
                        0.0,
                        1.0,
                    )
                )
                for i in range(len(client_updates))
            ]

            keep = list(range(len(client_updates)))
            if config.cosine_drop_k > 0 and config.cosine_drop_k < len(keep):
                order = sorted(range(len(similarities)), key=lambda i: similarities[i])
                drop_idx = order[: int(config.cosine_drop_k)]
                dropped = [i + 1 for i in drop_idx]
                keep = [i for i in keep if i not in set(drop_idx)]

            kept_states = [client_states[i] for i in keep]
            kept_weights = [base_weights[i] for i in keep]
            # Trust-weighted FedAvg
            if trusts:
                kept_trusts = [trusts[i] for i in keep]
                norm = float(sum(kept_trusts))
                if norm <= 1e-12:
                    kept_weights = kept_weights
                else:
                    kept_weights = [w * (t / norm) for w, t in zip(kept_weights, kept_trusts)]
            new_state = fedavg(kept_states, kept_weights)

        elif agg == "trimmed_mean":
            updates = list(client_updates)
            robust_upd = _coordinate_trimmed_mean(updates, trim_ratio=float(config.trim_ratio))
            new_state = _add_state(global_state, robust_upd)

        elif agg == "median":
            updates = list(client_updates)
            robust_upd = _coordinate_median(updates)
            new_state = _add_state(global_state, robust_upd)

        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        # ------------------- client feedback (server → client) -------------------
        # Persist per-client diagnostics each round so future work can simulate
        # "sending back" trust/quarantine signals to clients.
        if local_logger is not None and local_run is not None:
            try:
                from nsl_kdd.client_feedback import ClientFeedback, round_feedback_payload

                dropped_set = set(int(x) for x in dropped)
                msgs = []
                for cid in range(1, len(client_updates) + 1):
                    used = cid not in dropped_set
                    sim = similarities[cid - 1] if similarities else None
                    trust = trusts[cid - 1] if trusts else None
                    loss_stab = loss_stabilities[cid - 1] if loss_stabilities else None
                    xlayer = cross_layer_scores[cid - 1] if cross_layer_scores else None
                    notes = None
                    if cid in dropped_set:
                        notes = "dropped_by_server"
                    msgs.append(
                        ClientFeedback(
                            client_id=cid,
                            used=used,
                            trust=trust,
                            cosine_similarity=sim,
                            loss_stability=loss_stab,
                            cross_layer=xlayer,
                            notes=notes,
                        )
                    )

                local_logger.log_client_feedback(
                    run=local_run,
                    round_num=int(rnd + 1),
                    payload=round_feedback_payload(round_num=int(rnd + 1), feedback=msgs),
                )
            except Exception:
                pass

        metrics = evaluate(global_model, X_test, y_test, device=config.device)
        metrics["round"] = float(rnd + 1)

        # Diagnostics
        if similarities:
            metrics["cosine_sim_min"] = float(np.min(similarities))
            metrics["cosine_sim_mean"] = float(np.mean(similarities))
        if trusts:
            metrics["trust_min"] = float(np.min(trusts))
            metrics["trust_mean"] = float(np.mean(trusts))
        if loss_stabilities:
            metrics["loss_stability_mean"] = float(np.mean(loss_stabilities))
        if cross_layer_scores:
            metrics["cross_layer_mean"] = float(np.mean(cross_layer_scores))
        if dropped:
            metrics["dropped_clients"] = float(len(dropped))

        history.append(metrics)

        # Local persistence
        if local_logger is not None and local_run is not None:
            try:
                local_logger.log_round(run=local_run, round_num=int(rnd + 1), metrics=metrics)
            except Exception:
                pass

        # Optional MongoDB persistence
        if mongo is not None and run_id is not None:
            try:
                mongo.log_round(run_id=run_id, round_num=int(rnd + 1), metrics=metrics)
            except Exception:
                pass

    final_metrics = history[-1].copy() if history else evaluate(global_model, X_test, y_test, device=config.device)
    return FLResult(metrics=final_metrics, history=history)
