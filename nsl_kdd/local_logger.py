from __future__ import annotations

"""Local experiment logging.

Stores one JSON file and one CSV file per run.

Folder layout:
- runs/<run_id>/run.json      (config + summary)
- runs/<run_id>/rounds.csv    (per-round metrics)
- runs/<run_id>/rounds.json   (per-round metrics)

This is intentionally simple and requires no external services.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import json

import pandas as pd


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    dir: Path


class LocalLogger:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, *, config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> LocalRun:
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        _safe_write_json(
            run_dir / "run.json",
            {
                "run_id": run_id,
                "created_at": _utcnow_iso(),
                "config": config,
                "meta": meta or {},
            },
        )
        return LocalRun(run_id=run_id, dir=run_dir)

    def log_round(self, *, run: LocalRun, round_num: int, metrics: Dict[str, Any]) -> None:
        # Append to rounds.json
        rounds_path = run.dir / "rounds.json"
        if rounds_path.exists():
            data = json.loads(rounds_path.read_text(encoding="utf-8"))
        else:
            data = []

        row = {"round": int(round_num), "logged_at": _utcnow_iso(), **metrics}
        data.append(row)
        _safe_write_json(rounds_path, data)

        # Also mirror to CSV for quick viewing
        df = pd.DataFrame(data)
        df.to_csv(run.dir / "rounds.csv", index=False)

    def log_client_feedback(self, *, run: LocalRun, round_num: int, payload: Dict[str, Any]) -> None:
        """Append per-round server→client feedback payload.

        Stored at: runs/<run_id>/client_feedback.json
        Format: a list of {round: int, clients: [...]}
        """

        path = run.dir / "client_feedback.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = []

        row = {"round": int(round_num), "logged_at": _utcnow_iso(), **payload}
        data.append(row)
        _safe_write_json(path, data)
