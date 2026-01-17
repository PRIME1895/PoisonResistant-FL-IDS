from __future__ import annotations

from pathlib import Path

from nsl_kdd.data import load_nsl_kdd
from preprocessing.preprocess import fit_preprocess, transform_with_preprocessor


def test_preprocess_fit_then_transform_shapes_match() -> None:
    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"

    # Small sample keeps CI/dev runs fast.
    train_df = load_nsl_kdd(train_path).head(2000)
    test_df = load_nsl_kdd(test_path).head(1000)

    X_train, y_train, prep = fit_preprocess(train_df)
    X_test, y_test = transform_with_preprocessor(test_df, prep)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Same feature space between splits (no leakage / no refit on test).
    assert list(X_train.columns) == list(X_test.columns)

    # Binary labels.
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})
