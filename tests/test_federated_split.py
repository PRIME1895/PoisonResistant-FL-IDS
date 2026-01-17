from __future__ import annotations

from pathlib import Path

from nsl_kdd.data import load_nsl_kdd
from nsl_kdd.federated import split_non_iid


def test_split_non_iid_produces_5_clients_with_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"

    # Sample for test speed.
    df = load_nsl_kdd(train_path).head(20000)

    clients, manifest = split_non_iid(df, n_clients=5, client_size=2000, seed=42)

    assert manifest["n_clients"] == 5
    assert manifest["client_size"] == 2000
    assert len(clients) == 5
    assert all(len(c) == 2000 for c in clients)

    # Each client keeps label column.
    assert all("label" in c.columns for c in clients)
