from __future__ import annotations

"""Batch plotting for all scenario folders under runs/.

You said you want plots for all the run folders (except specific timestamp runs).
This script:
- finds subfolders under runs/
- for each folder that contains rounds.json/rounds.csv, generates 3 plots using that
  folder as clean/poisoned/defended source (useful for a quick per-scenario plot set).

If you want the 3-curve comparison plot (clean vs poisoned vs defended) you should
use `main.py plot` and provide three folders.

This command is for generating plots for each scenario folder automatically.
"""

from pathlib import Path

from nsl_kdd.plots import generate_core_plots


def iter_run_dirs(runs_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "rounds.json").exists() or (p / "rounds.csv").exists():
            out.append(p)
    return sorted(out)


def main() -> int:
    root = Path.cwd()
    runs_root = root / "runs"
    if not runs_root.exists():
        raise SystemExit("runs/ folder not found")

    # Exclude raw timestamp runs if desired; keep only named scenario folders.
    run_dirs = [p for p in iter_run_dirs(runs_root) if not p.name.startswith("202")]

    out_root = root / "figures" / "all_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    for p in run_dirs:
        out_dir = out_root / p.name
        generate_core_plots(clean_run=p, poisoned_run=p, defended_run=p, out_dir=out_dir, defended_label=p.name)
        print(f"Wrote: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
