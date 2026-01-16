"""NSL-KDD schema helpers.

NSL-KDD (derived from KDD'99) typically has 41 features + label (+ optional difficulty).
The `KDDTrain+.txt` / `KDDTest+.txt` files commonly include the difficulty column.

We keep the schema tolerant: if a file has 43 columns, we treat the last two as
(label, difficulty) and drop difficulty.
"""

from __future__ import annotations

KDD99_FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
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

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]


def make_column_names(num_columns: int) -> list[str]:
    """Return column names that match the detected column count.

    Supported:
    - 42 columns: 41 features + label
    - 43 columns: 41 features + label + difficulty
    """

    if num_columns == 42:
        return [*KDD99_FEATURE_NAMES, "label"]

    if num_columns == 43:
        return [*KDD99_FEATURE_NAMES, "label", "difficulty"]

    # Fallback: generate generic names so we can still load and inspect.
    cols = [f"c{i}" for i in range(num_columns)]
    if num_columns >= 2:
        cols[-2] = "label"
        cols[-1] = "difficulty"
    return cols
