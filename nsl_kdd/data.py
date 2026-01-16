from __future__ import annotations
    return train_path, test_path
    test_path = root / "KDDTest+.txt"
    train_path = root / "KDDTrain+.txt"
    root = Path(project_root)
def train_test_paths(project_root: str | Path) -> Tuple[Path, Path]:


    return df

        df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    if "label" in df.columns:
    # Normalize label strings (some datasets include trailing '.')

        df = df.drop(columns=["difficulty"])
    if "difficulty" in df.columns:

        df = pd.read_csv(p, header=None, names=names, delim_whitespace=True)
    else:
        df = pd.read_csv(p, header=None, names=names)
    if delim == ",":

    names = make_column_names(n_cols)
    n_cols = peek_num_columns(p, delim)
    delim = detect_delimiter(p)

        raise FileNotFoundError(f"File not found: {p}")
    if not p.exists():
    p = Path(path)

    """
    - Drops `difficulty` column when present.
    - Assigns standard column names if column count matches common NSL-KDD formats.
    - Auto-detects delimiter (comma vs whitespace).

    """Load a NSL-KDD txt file into a DataFrame.
def load_nsl_kdd(path: str | Path) -> pd.DataFrame:


    return len(first.split())
        return len(next(csv.reader([first], delimiter=",")))
    if delimiter == ",":

        first = f.readline().strip()
    with path.open("r", encoding="utf-8", errors="replace") as f:
def peek_num_columns(path: Path, delimiter: Delimiter) -> int:


    return "," if comma_votes >= max(1, sample_lines // 2) else "whitespace"
    comma_votes = sum(1 for ln in lines if "," in ln)
    # If most lines contain commas, assume CSV.

        lines = [next(f).strip() for _ in range(sample_lines)]
    with path.open("r", encoding="utf-8", errors="replace") as f:
    """Detect whether a NSL-KDD file is comma-separated or whitespace-separated."""
def detect_delimiter(path: Path, sample_lines: int = 5) -> Delimiter:


Delimiter = Literal[",", "whitespace"]


from .schema import make_column_names

import pandas as pd

from typing import Literal, Tuple
from pathlib import Path
import csv

