from __future__ import annotations

"""Plotting utilities for research-grade figures (3 core plots).

Reads local run logs from `runs/<run_id>/rounds.json` or `rounds.csv`.

Creates:
1) Accuracy vs FL rounds (clean vs poisoned vs defended)
2) Recall vs FL rounds (clean vs poisoned vs defended)
3) Defense signal vs rounds (Phase 7 only) e.g. trust_mean, cosine_sim_mean, dropped_clients

Outputs PNGs into a chosen directory.

Usage (via main.py command):
- python main.py plot --clean runs/<id1> --poisoned runs/<id2> --defended runs/<id3>
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RunSeries:
    label: str
    df: pd.DataFrame


def _load_rounds(run_path: Path) -> pd.DataFrame:
    """Load per-round metrics for a run.

    Accepts:
    - a run directory: runs/<run_id>/
    - a direct file path: runs/<run_id>/rounds.json or rounds.csv
    """

    p = Path(run_path)
    if p.is_dir():
        json_path = p / "rounds.json"
        csv_path = p / "rounds.csv"
    else:
        # If a file is passed directly.
        if p.suffix.lower() == ".json":
            json_path = p
            csv_path = p.with_suffix(".csv")
        elif p.suffix.lower() == ".csv":
            csv_path = p
            json_path = p.with_suffix(".json")
        else:
            # Unknown suffix: try interpreting as a run_id or basename under its parent.
            json_path = p / "rounds.json"
            csv_path = p / "rounds.csv"

    if json_path.exists():
        df = pd.read_json(json_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No rounds.json or rounds.csv found for: {run_path}")

    if "round" not in df.columns:
        raise ValueError("Expected 'round' column in run logs")

    df = df.copy()
    df["round"] = df["round"].astype(int)
    df = df.sort_values("round").reset_index(drop=True)
    return df


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_lines(
    series: Iterable[RunSeries],
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    for s in series:
        if y not in s.df.columns:
            continue
        plt.plot(s.df[x], s.df[y], marker="o", linewidth=2, label=s.label)

    plt.title(title)
    plt.xlabel("FL Round")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_core_plots(
    *,
    clean_run: Path,
    poisoned_run: Path,
    defended_run: Path,
    out_dir: Path = Path("figures"),
    defended_label: str = "Phase 7 — Defended (Trust-Aware)",
) -> list[Path]:
    """Generate the 3 key plots and return written filepaths."""

    out_dir = _ensure_out_dir(Path(out_dir))

    clean_df = _load_rounds(clean_run)
    poison_df = _load_rounds(poisoned_run)
    def_df = _load_rounds(defended_run)

    s_clean = RunSeries("Phase 5 — Clean FedAvg", clean_df)
    s_poison = RunSeries("Phase 6 — FedAvg + Poisoning", poison_df)
    s_def = RunSeries(defended_label, def_df)

    written: list[Path] = []

    # Plot 1: Accuracy vs rounds
    p1 = out_dir / "plot1_accuracy_vs_rounds.png"
    _plot_lines(
        [s_clean, s_poison, s_def],
        x="round",
        y="accuracy",
        title="Accuracy vs FL Rounds",
        ylabel="Accuracy",
        out_path=p1,
    )
    written.append(p1)

    # Plot 2: Recall vs rounds (IDS-friendly)
    p2 = out_dir / "plot2_recall_vs_rounds.png"
    _plot_lines(
        [s_clean, s_poison, s_def],
        x="round",
        y="recall",
        title="Recall vs FL Rounds (Detection Rate)",
        ylabel="Recall",
        out_path=p2,
    )
    written.append(p2)

    # Plot 3: Defense signal vs rounds (defended run only)
    # Prefer trust_mean; fall back to loss_stability_mean/cross_layer_mean if trust_mean isn't present.
    candidates = [
        ("trust_mean", "Average Trust Score"),
        ("cosine_sim_mean", "Cosine Similarity (Mean)"),
        ("dropped_clients", "Dropped Clients"),
        ("loss_stability_mean", "Loss Stability (Mean)"),
        ("cross_layer_mean", "Cross-Layer Consistency (Mean)"),
    ]

    chosen = None
    for col, label in candidates:
        if col in def_df.columns:
            chosen = (col, label)
            break

    if chosen is None:
        # Still create an empty placeholder plot to keep deliverables consistent.
        col, ylab = "accuracy", "(no defense signal logged)"
    else:
        col, ylab = chosen

    p3 = out_dir / "plot3_defense_signal_vs_rounds.png"
    _plot_lines(
        [s_def],
        x="round",
        y=col,
        title=f"Defense Signal vs FL Rounds ({ylab})",
        ylabel=ylab,
        out_path=p3,
    )
    written.append(p3)

    return written
