import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# NSL-KDD feature groups
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERICAL_COLUMNS = [
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

def _binary_labels(df: pd.DataFrame) -> np.ndarray:
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the data.")

    y = df["label"]
    if pd.api.types.is_bool_dtype(y):
        return y.astype(int).to_numpy()

    if pd.api.types.is_numeric_dtype(y):
        # Numeric labels: treat 0 as normal/benign, everything else as attack.
        return (y != 0).astype(int).to_numpy()

    # String labels: normalize common IDS label names.
    y_raw = y.astype(str).str.strip().str.rstrip(".").str.lower()
    normal_mask = y_raw.isin({"normal", "benign", "background", "normal."})
    return (~normal_mask).astype(int).to_numpy()

def build_preprocessor(
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build a reusable preprocessing transformer (fit on train, transform on test)."""
    # Use explicit checks instead of "or" to avoid falling back on empty lists
    if categorical_columns is None:
        cat_cols = CATEGORICAL_COLUMNS
    else:
        cat_cols = list(categorical_columns)
    
    if numerical_columns is None:
        num_cols = NUMERICAL_COLUMNS
    else:
        num_cols = list(numerical_columns)

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
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
        verbose_feature_names_out=False,
    )

def fit_preprocess(train_df: pd.DataFrame, cat_cols: list[str] = None, num_cols: list[str] = None) -> tuple[pd.DataFrame, np.ndarray, ColumnTransformer]:
    """Fit preprocessing on train and return (X_train_processed_df, y_train, fitted_preprocessor).
    
    Args:
        train_df: Training dataframe with 'label' column
        cat_cols: List of categorical column names. If None, attempts to auto-detect.
        num_cols: List of numerical column names. If None, attempts to auto-detect.
    """
    df = train_df.copy()

    # Drop difficulty column if present (repo's loader drops `difficulty`, but keep it robust).
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    if "difficulty_level" in df.columns:
        df = df.drop(columns=["difficulty_level"])

    y = _binary_labels(df)
    X = df.drop(columns=["label"]).copy()
    
    # Handle inf and NaN values early
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Auto-detect columns if not provided
    if cat_cols is None:
        # Try to use NSL-KDD defaults, fallback to auto-detection
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    else:
        cat_cols = [c for c in cat_cols if c in X.columns]
    
    if num_cols is None:
        # Try to use NSL-KDD defaults, fallback to auto-detection
        num_cols = [c for c in NUMERICAL_COLUMNS if c in X.columns]
        
        # If we didn't find NSL-KDD columns, auto-detect based on dtype
        if len(num_cols) == 0:
            num_cols = []
            for c in X.columns:
                # Check if column is actually numeric (not just numeric dtype)
                if X[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16', 'uint8', 'uint16', 'uint32', 'uint64']:
                    num_cols.append(c)
    else:
        num_cols = [c for c in num_cols if c in X.columns]
    
    # If still no categorical columns found, try to detect based on dtype
    if len(cat_cols) == 0:
        cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category']
    
    # Ensure we have at least some columns
    all_cols = set(cat_cols + num_cols)
    missing_cols = set(X.columns) - all_cols
    
    if len(missing_cols) > 0:
        # Try to infer type for missing columns
        for c in missing_cols:
            try:
                # Try to convert to numeric
                pd.to_numeric(X[c], errors='raise')
                num_cols.append(c)
            except (ValueError, TypeError):
                # If conversion fails, treat as categorical
                cat_cols.append(c)

    preprocessor = build_preprocessor(cat_cols, num_cols)
    Xp = preprocessor.fit_transform(X)

    feature_names = list(preprocessor.get_feature_names_out())
    Xp_df = pd.DataFrame(Xp, columns=feature_names)

    return Xp_df, y, preprocessor

def transform_with_preprocessor(
    df: pd.DataFrame, preprocessor: ColumnTransformer
) -> tuple[pd.DataFrame, np.ndarray]:
    """Transform any split with an already-fitted preprocessor."""
    dfx = df.copy()

    if "difficulty" in dfx.columns:
        dfx = dfx.drop(columns=["difficulty"])
    if "difficulty_level" in dfx.columns:
        dfx = dfx.drop(columns=["difficulty_level"])

    y = _binary_labels(dfx)
    X = dfx.drop(columns=["label"]).copy()
    
    # Handle inf and NaN values (important for CICIDS)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    Xp = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    Xp_df = pd.DataFrame(Xp, columns=feature_names)

    return Xp_df, y

# Backwards-compatible helper name (used by your earlier draft).
# NOTE: This fits a new transformer each call; prefer fit_preprocess + transform_with_preprocessor.
def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    Xp_df, y, _ = fit_preprocess(dataset)
    out = Xp_df.copy()
    out["label"] = y
    return out

# Example usage
if __name__ == "__main__":
    # Smoke-run preprocessing using the repo's robust loader.
    from pathlib import Path

    from nsl_kdd.data import load_nsl_kdd, train_test_paths

    root = Path(__file__).resolve().parents[1]
    train_path, test_path = train_test_paths(root)

    train_df = load_nsl_kdd(train_path)
    test_df = load_nsl_kdd(test_path)

    X_train, y_train, prep = fit_preprocess(train_df)
    X_test, y_test = transform_with_preprocessor(test_df, prep)

    print("Preprocessing complete")
    print(f"Train: X={X_train.shape}, y={y_train.shape}, positives={int(y_train.sum())}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}, positives={int(y_test.sum())}")
    print("Feature sample:")
    print(X_train.head())
