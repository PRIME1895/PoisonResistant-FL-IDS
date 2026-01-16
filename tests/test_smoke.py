from __future__ import annotations

from pathlib import Path

from nsl_kdd.data import load_nsl_kdd
from nsl_kdd.pipeline import train_and_eval


def test_smoke_train_binary_sample() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"

    # Keep the test fast: use a small sample.
    train_df = load_nsl_kdd(train_path).head(2000)
    test_df = load_nsl_kdd(test_path).head(1000)

    result = train_and_eval(train_df, test_df, binary=True)

    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["f1"] <= 1.0
