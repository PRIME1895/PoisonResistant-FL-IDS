from __future__ import annotations

"""MongoDB Atlas experiment logging (Phase 8).

Security rule: never hardcode credentials.

Expected environment variables:
- MONGODB_URI: MongoDB Atlas connection string (mongodb+srv://...)
- MONGODB_DB: database name (default: fl_ids)

Collections:
- runs: one document per experiment run (config + timestamps)
- rounds: one document per round per run (metrics)

This module is intentionally lightweight so it can be reused in Heroku later.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    created_at: datetime


class MongoLogger:
    def __init__(
        self,
        *,
        uri: str,
        db_name: str = "fl_ids",
        app_name: str = "NSL-KDD-FL-IDS",
        connect_timeout_ms: int = 5000,
    ) -> None:
        from pymongo import MongoClient

        self._client = MongoClient(
            uri,
            appname=app_name,
            serverSelectionTimeoutMS=connect_timeout_ms,
        )
        self._db = self._client[db_name]

        # Ensure indexes (best-effort; safe if already exists)
        self._db.runs.create_index("created_at")
        self._db.rounds.create_index([("run_id", 1), ("round", 1)], unique=True)

    def ping(self) -> None:
        # Forces a connection + throws if URI is bad/unreachable.
        self._client.admin.command("ping")

    def create_run(self, *, config: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> RunInfo:
        doc = {
            "created_at": _utcnow(),
            "config": config,
            "meta": meta or {},
        }
        res = self._db.runs.insert_one(doc)
        return RunInfo(run_id=str(res.inserted_id), created_at=doc["created_at"])

    def log_round(self, *, run_id: str, round_num: int, metrics: Dict[str, Any]) -> None:
        doc = {
            "run_id": run_id,
            "round": int(round_num),
            "logged_at": _utcnow(),
            "metrics": metrics,
        }
        self._db.rounds.replace_one({"run_id": run_id, "round": int(round_num)}, doc, upsert=True)


def build_mongo_logger_from_env() -> Optional[MongoLogger]:
    """Create MongoLogger from env vars.

    Returns None if MONGODB_URI is not set.
    """

    import os

    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None

    db_name = os.getenv("MONGODB_DB", "fl_ids")
    return MongoLogger(uri=uri, db_name=db_name)
