from __future__ import annotations

"""Federated (non-IID) client splitting utilities.

Goal (Phase 3): simulate 5 clients with heterogeneous data distributions.
We keep it deterministic via a random seed.

Outputs are DataFrames that keep the original NSL-KDD columns (including `label`).
Downstream code can then choose binary labels or multiclass.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .attacks import label_to_family


@dataclass(frozen=True)
class SplitSpec:
    """Target mixture for a client.

    We do best-effort allocation by drawing up to the requested proportion for each
    family, then filling remaining rows from what's left.
    """

    name: str
    proportions: Dict[str, float]  # family -> fraction, should sum to <= 1


DEFAULT_5_CLIENT_SPECS: List[SplitSpec] = [
    SplitSpec("client_1_mostly_normal", {"normal": 0.80, "dos": 0.10, "probe": 0.05, "r2l": 0.03, "u2r": 0.02}),
    SplitSpec("client_2_mostly_dos", {"dos": 0.75, "normal": 0.15, "probe": 0.05, "r2l": 0.04, "u2r": 0.01}),
    SplitSpec("client_3_mixed", {"normal": 0.40, "dos": 0.30, "probe": 0.15, "r2l": 0.10, "u2r": 0.05}),
    SplitSpec("client_4_mostly_probe", {"probe": 0.70, "normal": 0.15, "dos": 0.10, "r2l": 0.04, "u2r": 0.01}),
    # "rare attacks" client: emphasize R2L/U2R while still including some normal.
    SplitSpec("client_5_rare_attacks", {"r2l": 0.60, "u2r": 0.20, "normal": 0.15, "probe": 0.03, "dos": 0.02}),
]


def add_family_column(df: pd.DataFrame, *, col: str = "family") -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column to derive attack families.")
    out = df.copy()
    out[col] = out["label"].map(label_to_family)
    return out


def split_non_iid(
    df: pd.DataFrame,
    *,
    n_clients: int = 5,
    client_size: int | None = None,
    seed: int = 1337,
    specs: List[SplitSpec] | None = None,
    keep_family_column: bool = False,
) -> Tuple[List[pd.DataFrame], Dict[str, object]]:
    """Split a labeled NSL-KDD dataframe into heterogeneous client datasets.

    Args:
        df: DataFrame with `label`.
        n_clients: number of clients (default 5).
        client_size: rows per client; if None, uses floor(len(df)/n_clients).
        seed: RNG seed for deterministic splits.
        specs: optional list of SplitSpec; defaults to DEFAULT_5_CLIENT_SPECS.
        keep_family_column: if True, keep added `family` column in outputs.

    Returns:
        (clients, manifest)
    """
    if n_clients <= 0:
        raise ValueError("n_clients must be > 0")

    specs = specs or DEFAULT_5_CLIENT_SPECS
    if len(specs) != n_clients:
        raise ValueError(f"Expected {n_clients} specs, got {len(specs)}")

    rng = np.random.default_rng(seed)
    df_f = add_family_column(df, col="family")

    if client_size is None:
        client_size = len(df_f) // n_clients
    if client_size <= 0:
        raise ValueError("client_size must be > 0")

    # Build pools per family.
    pools: Dict[str, np.ndarray] = {}
    for fam, fam_df in df_f.groupby("family"):
        idx = fam_df.index.to_numpy()
        rng.shuffle(idx)
        pools[str(fam)] = idx

    # Cursor per family pool.
    cursor: Dict[str, int] = {fam: 0 for fam in pools}

    def take(fam: str, k: int) -> np.ndarray:
        if k <= 0:
            return np.array([], dtype=int)
        if fam not in pools:
            return np.array([], dtype=int)
        start = cursor[fam]
        end = min(start + k, len(pools[fam]))
        cursor[fam] = end
        return pools[fam][start:end]

    rare_families = ["u2r", "r2l"]

    def remaining_indices_prioritized(prefer: list[str] | None = None) -> np.ndarray:
        """Return remaining indices, optionally prioritizing some families."""
        prefer = prefer or []
        chunks = []

        # First, preferred families.
        for fam in prefer:
            if fam in pools:
                start = cursor[fam]
                if start < len(pools[fam]):
                    chunks.append(pools[fam][start:])

        # Then, everything else.
        for fam, idx in pools.items():
            if fam in prefer:
                continue
            start = cursor[fam]
            if start < len(idx):
                chunks.append(idx[start:])

        if not chunks:
            return np.array([], dtype=int)

        r = np.concatenate(chunks)
        rng.shuffle(r)
        return r

    client_indices: List[np.ndarray] = []
    for spec in specs:
        chosen: List[np.ndarray] = []
        for fam, frac in spec.proportions.items():
            k = int(round(frac * client_size))
            chosen.append(take(fam, k))

        picked = np.concatenate(chosen) if chosen else np.array([], dtype=int)

        # Fill or trim to exact client_size.
        if len(picked) < client_size:
            prefer = rare_families if "u2r" in spec.proportions or "r2l" in spec.proportions else []
            fill = remaining_indices_prioritized(prefer)[: (client_size - len(picked))]
            picked = np.concatenate([picked, fill]) if len(fill) else picked
        if len(picked) > client_size:
            rng.shuffle(picked)
            picked = picked[:client_size]

        client_indices.append(picked)

    clients = []
    for idx in client_indices:
        cdf = df_f.loc[idx].sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
        if not keep_family_column:
            cdf = cdf.drop(columns=["family"])
        clients.append(cdf)

    manifest: Dict[str, object] = {
        "n_clients": n_clients,
        "client_size": client_size,
        "seed": seed,
        "specs": [{"name": s.name, "proportions": s.proportions} for s in specs],
    }
    return clients, manifest


def family_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Convenience: count rows per family in a split."""
    df_f = add_family_column(df, col="family")
    vc = df_f["family"].value_counts()
    return {str(k): int(v) for k, v in vc.items()}
