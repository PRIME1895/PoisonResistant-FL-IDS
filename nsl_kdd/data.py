from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd

from .schema import make_column_names

Delimiter = Literal[",", "whitespace"]


def detect_delimiter(path: Path, sample_lines: int = 5) -> Delimiter:
    """Detect whether a NSL-KDD file is comma-separated or whitespace-separated."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip())

    if not lines:
        # Empty file fallback
        return "whitespace"

    comma_votes = sum(1 for ln in lines if "," in ln)
    return "," if comma_votes >= max(1, len(lines) // 2) else "whitespace"


def peek_num_columns(path: Path, delimiter: Delimiter) -> int:
    """Read the first row and return the number of columns."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()

    if not first:
        return 0

    if delimiter == ",":
        return len(next(csv.reader([first], delimiter=",")))
    return len(first.split())


def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    """Load a NSL-KDD txt file into a DataFrame.

    - Auto-detects delimiter (comma vs whitespace)
    - Assigns best-effort column names based on detected column count
    - Drops `difficulty` column when present
    - Normalizes label strings (strips whitespace and trailing '.')
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    delim = detect_delimiter(p)
    n_cols = peek_num_columns(p, delim)
    names = make_column_names(n_cols)

    if delim == ",":
        df = pd.read_csv(p, header=None, names=names)
    else:
        df = pd.read_csv(p, header=None, names=names, sep=r"\s+", engine="python")

    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")

    return df


def train_test_paths(project_root: str | Path) -> Tuple[Path, Path]:
    root = Path(project_root)
    train_path = root / "KDDTrain+.txt"
    test_path = root / "KDDTest+.txt"
    return train_path, test_path
