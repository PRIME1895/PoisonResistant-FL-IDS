from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TrainResult:
    metrics: Dict[str, float]
    n_train: int
    n_test: int


def make_xy(df: pd.DataFrame, *, binary: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the NSL-KDD data.")

    y_raw = df["label"].astype(str)
    X = df.drop(columns=["label"]).copy()

    if binary:
        y = (y_raw != "normal").astype(int).to_numpy()
    else:
        # Multiclass placeholder: keep raw label strings.
        y = y_raw.to_numpy()

    return X, y


def build_baseline(binary: bool = True) -> Pipeline:
    # Explicitly choose feature groups.
    cat_cols = [c for c in CATEGORICAL_FEATURES if c]

    # Numeric selector: everything else (ColumnTransformer requires explicit lists).
    num_cols = [
        c
        for c in [
            "duration",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ]
        if c not in cat_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    # Logistic Regression is a strong, fast baseline for the binary demo.
    if binary:
        clf = LogisticRegression(max_iter=300)
    else:
        clf = LogisticRegression(max_iter=300, multi_class="auto")

    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    binary: bool = True,
) -> TrainResult:
    X_train, y_train = make_xy(train_df, binary=binary)
    X_test, y_test = make_xy(test_df, binary=binary)

    model = build_baseline(binary=binary)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if binary:
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
        }

    return TrainResult(metrics=metrics, n_train=len(train_df), n_test=len(test_df))
