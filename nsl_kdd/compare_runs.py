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

New (quality-of-life):
- If the named scenario folders don't exist, we can auto-discover timestamped
  runs (e.g., runs/20260308_022055) that contain rounds logs.
"""

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


def _final_round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-run summary using the last logged round."""

    rows: list[pd.Series] = []
    for name, g in df.groupby("run_name"):
        g = g.sort_values("round")
        last = g.iloc[-1].copy()
        last["run_name"] = name
        rows.append(last)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Keep a stable subset when present.
    preferred = [
        "run_name",
        "round",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols].reset_index(drop=True)


DEFAULT_RUNS = [
    "Centralized_baseline",
    "baseline_FedAvg",
    "Poisoning_attack(label_flipping)",
    "Cross-layer_trust_weighting(cosine)+outlier_drop",
    "trimmed_mean(clipping+robust aggregation)",
    "coordinate_median(trimming+robust aggregation)",
]


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    """Return run subdirectories that contain rounds logs."""
    runs_root = Path(runs_root)
    out: list[Path] = []
    if not runs_root.exists():
        return out
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "rounds.json").exists() or (p / "rounds.csv").exists():
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def _existing_named_runs(runs_root: Path, run_names: Iterable[str]) -> list[str]:
    """Filter to the subset of run_names that actually exist and have rounds logs."""
    out: list[str] = []
    for name in run_names:
        p = Path(runs_root) / name
        if not p.is_dir():
            continue
        if (p / "rounds.json").exists() or (p / "rounds.csv").exists():
            out.append(name)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple FL runs and plot common metrics.")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Folder containing individual run subfolders (default: runs)",
    )
    parser.add_argument(
        "--out-root",
        default=str(Path("figures") / "comparison"),
        help="Output folder for merged metrics and plots (default: figures/comparison)",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help=(
            "Comma-separated run folder names to compare (e.g. runA,runB). "
            "If omitted, uses DEFAULT_RUNS when present; otherwise auto-discovers timestamped runs."
        ),
    )

    args = parser.parse_args(argv)

    root = Path.cwd()
    runs_root = root / args.runs_root
    out_root = root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.runs:
        run_names = [x.strip() for x in args.runs.split(",") if x.strip()]
    else:
        # Prefer the curated scenario folders only when the whole set exists;
        # otherwise fallback to discovery of timestamped runs.
        existing_defaults = _existing_named_runs(runs_root, DEFAULT_RUNS)
        if len(existing_defaults) == len(DEFAULT_RUNS):
            run_names = existing_defaults
        else:
            discovered = _discover_run_dirs(runs_root)
            if not discovered:
                raise SystemExit(f"No run directories with rounds.json/csv found under: {runs_root}")
            run_names = [p.name for p in discovered]

    merged = merge_runs(runs_root, run_names)
    save_merged(merged, out_root / "all_round_metrics")

    plot_comparison(
        merged,
        metric="accuracy",
        out_path=out_root / "comparison_accuracy.png",
        title="All Runs — Accuracy vs Rounds",
    )
    plot_comparison(
        merged,
        metric="recall",
        out_path=out_root / "comparison_recall.png",
        title="All Runs — Recall vs Rounds",
    )
    plot_comparison(
        merged,
        metric="false_positive_rate",
        out_path=out_root / "comparison_false_positive_rate.png",
        title="All Runs — False Positive Rate vs Rounds",
    )

    final = _final_round_summary(merged)
    if not final.empty:
        final.to_csv(out_root / "final_round_summary.csv", index=False)
        final.to_json(out_root / "final_round_summary.json", orient="records", indent=2)

    print(f"Wrote merged metrics: {out_root / 'all_round_metrics.csv'}")
    print(f"Wrote merged metrics: {out_root / 'all_round_metrics.json'}")
    print(f"Wrote plot: {out_root / 'comparison_accuracy.png'}")
    print(f"Wrote plot: {out_root / 'comparison_recall.png'}")
    print(f"Wrote plot: {out_root / 'comparison_false_positive_rate.png'}")
    if not final.empty:
        print(f"Wrote final summary: {out_root / 'final_round_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
