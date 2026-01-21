from __future__ import annotations

"""Compare multiple run folders on a single plot and export merged metrics.

This supports your 6 scenario folders under runs/: e.g.
- Centralized_baseline
- baseline_FedAvg
- Poisoning_attack(label_flipping)
- Cross-layer_trust_weighting(cosine)+outlier_drop
- trimmed_mean(clipping+robust aggregation)
- coordinate_median(trimming+robust aggregation)

It will:
- load each folder's rounds.json/rounds.csv
- write a merged JSON + CSV with a `run_name` column
- generate a single comparison plot (multiple curves)

Note: "Centralized_baseline" may not have rounds in a true sense. We still plot it
as a flat line if it contains a single round row.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def _load_rounds(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    json_path = run_dir / "rounds.json"
    csv_path = run_dir / "rounds.csv"

    if json_path.exists():
        df = pd.read_json(json_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing rounds.json/csv in {run_dir}")

    df = df.copy()
    if "round" not in df.columns:
        # fallback: treat whole file as a single point
        df["round"] = 1

    # normalize round to int
    df["round"] = df["round"].astype(int)

    # deduplicate if needed (keep last by logged_at when present)
    if "logged_at" in df.columns:
        df = df.sort_values(["round", "logged_at"]).drop_duplicates(subset=["round"], keep="last")
    else:
        df = df.drop_duplicates(subset=["round"], keep="last")

    return df.sort_values("round").reset_index(drop=True)


def merge_runs(runs_root: Path, run_names: Iterable[str]) -> pd.DataFrame:
    runs_root = Path(runs_root)
    frames: list[pd.DataFrame] = []
    for name in run_names:
        df = _load_rounds(runs_root / name)
        df.insert(0, "run_name", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save_merged(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV and JSON side-by-side
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    df.to_json(out_path.with_suffix(".json"), orient="records", indent=2)


def plot_comparison(df: pd.DataFrame, *, metric: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    for name, g in df.groupby("run_name"):
        if metric not in g.columns:
            continue
        g = g.sort_values("round")
        plt.plot(g["round"], g[metric], marker="o", linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


DEFAULT_RUNS = [
    "Centralized_baseline",
    "baseline_FedAvg",
    "Poisoning_attack(label_flipping)",
    "Cross-layer_trust_weighting(cosine)+outlier_drop",
    "trimmed_mean(clipping+robust aggregation)",
    "coordinate_median(trimming+robust aggregation)",
]


def main() -> int:
    root = Path.cwd()
    runs_root = root / "runs"
    out_root = root / "figures" / "comparison"
    out_root.mkdir(parents=True, exist_ok=True)

    merged = merge_runs(runs_root, DEFAULT_RUNS)
    save_merged(merged, out_root / "all_round_metrics")

    # single comparison plots (accuracy and recall are most useful)
    plot_comparison(
        merged,
        metric="accuracy",
        out_path=out_root / "comparison_accuracy.png",
        title="All Scenarios — Accuracy vs Rounds",
    )
    plot_comparison(
        merged,
        metric="recall",
        out_path=out_root / "comparison_recall.png",
        title="All Scenarios — Recall vs Rounds",
    )

    print(f"Wrote merged metrics: {out_root / 'all_round_metrics.csv'}")
    print(f"Wrote merged metrics: {out_root / 'all_round_metrics.json'}")
    print(f"Wrote plot: {out_root / 'comparison_accuracy.png'}")
    print(f"Wrote plot: {out_root / 'comparison_recall.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
