from __future__ import annotations

# Ensure headless-safe backend (important for CI/pytest environments)
import matplotlib

matplotlib.use("Agg", force=True)

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int


def plot_confusion_matrix(
    counts: ConfusionCounts,
    *,
    out_path: Path,
    title: str = "Confusion Matrix (Binary)",
    normalize: bool = False,
) -> Path:
    """Save a confusion matrix plot.

    Args:
        counts: ConfusionCounts(tp, fp, tn, fn)
        out_path: Path to write PNG
        title: Plot title
        normalize: If True, normalize rows (true class) to proportions.

    Returns:
        out_path
    """

    import numpy as np
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array([[counts.tn, counts.fp], [counts.fn, counts.tp]], dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        fmt = ".2f"
    else:
        cm_show = cm
        fmt = ".0f"

    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(cm_show, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal (0)", "Attack (1)"], rotation=20, ha="right")
    plt.yticks(tick_marks, ["Normal (0)", "Attack (1)"])

    thresh = cm_show.max() / 2.0 if cm_show.size else 0.5
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                format(cm_show[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_show[i, j] > thresh else "black",
                fontsize=11,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Small footer with raw counts.
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return out_path


def _to_int(x: object) -> int:
    try:
        return int(round(float(x)))  # type: ignore[arg-type]
    except Exception:
        return 0


def extract_counts_from_metrics(metrics: dict) -> ConfusionCounts:
    """Extract confusion counts from a metrics dict (supports float storage)."""

    return ConfusionCounts(
        tp=_to_int(metrics.get("tp", 0)),
        fp=_to_int(metrics.get("fp", 0)),
        tn=_to_int(metrics.get("tn", 0)),
        fn=_to_int(metrics.get("fn", 0)),
    )
