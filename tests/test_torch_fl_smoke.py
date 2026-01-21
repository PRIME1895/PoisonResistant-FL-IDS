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


def test_torch_fl_poisoning_and_defenses_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"

    train_df = load_nsl_kdd(train_path).head(4000)
    test_df = load_nsl_kdd(test_path).head(1500)

    # 4 clients.
    client_dfs = [
        train_df.iloc[0:1000].reset_index(drop=True),
        train_df.iloc[1000:2000].reset_index(drop=True),
        train_df.iloc[2000:3000].reset_index(drop=True),
        train_df.iloc[3000:4000].reset_index(drop=True),
    ]

    # Phase 6: poison client 2 by flipping 50% labels.
    poisoned = FLConfig(
        rounds=1,
        local_epochs=1,
        batch_size=256,
        lr=1e-3,
        device="cpu",
        seed=7,
        malicious_clients=(2,),
        label_flip_rate=0.5,
        aggregation="fedavg",
    )
    res_poisoned = train_fedavg_binary(client_dfs, test_df, config=poisoned)
    assert 0.0 <= res_poisoned.metrics["accuracy"] <= 1.0

    # Phase 7 Option A: cosine trust weighting (drop 1 suspected outlier)
    defended_a = FLConfig(
        rounds=1,
        local_epochs=1,
        batch_size=256,
        lr=1e-3,
        device="cpu",
        seed=7,
        malicious_clients=(2,),
        label_flip_rate=0.5,
        aggregation="cosine",
        cosine_drop_k=1,
        trust_alpha=1.0,
        trust_beta=0.5,
        trust_gamma=0.5,
    )
    res_a = train_fedavg_binary(client_dfs, test_df, config=defended_a)
    assert 0.0 <= res_a.metrics["f1"] <= 1.0
    # Diagnostics exposed
    assert "trust_mean" in res_a.metrics
    assert "cross_layer_mean" in res_a.metrics
    assert "loss_stability_mean" in res_a.metrics

    # Phase 7 Option B: clipping + trimmed mean
    defended_b = FLConfig(
        rounds=1,
        local_epochs=1,
        batch_size=256,
        lr=1e-3,
        device="cpu",
        seed=7,
        malicious_clients=(2,),
        label_flip_rate=0.5,
        aggregation="trimmed_mean",
        clip_norm=5.0,
        trim_ratio=0.25,
    )
    res_b = train_fedavg_binary(client_dfs, test_df, config=defended_b)
    assert 0.0 <= res_b.metrics["f1"] <= 1.0

