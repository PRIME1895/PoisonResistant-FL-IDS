from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from nsl_kdd.data import load_nsl_kdd  # noqa: E402
from nsl_kdd.torch_fl import FLConfig, train_fedavg_binary  # noqa: E402


def test_torch_fedavg_smoke_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"

    # Small & fast.
    train_df = load_nsl_kdd(train_path).head(5000)
    test_df = load_nsl_kdd(test_path).head(2000)

    # Two tiny clients.
    client_dfs = [train_df.iloc[:2500].reset_index(drop=True), train_df.iloc[2500:5000].reset_index(drop=True)]

    cfg = FLConfig(rounds=2, local_epochs=1, batch_size=256, lr=1e-3, device="cpu", seed=1)
    res = train_fedavg_binary(client_dfs, test_df, config=cfg)

    assert 0.0 <= res.metrics["accuracy"] <= 1.0
    assert 0.0 <= res.metrics["f1"] <= 1.0
