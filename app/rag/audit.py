"""Audit logging for ingest operations. Append-only JSONL."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from settings import settings

logger = logging.getLogger(__name__)


def log_ingest_event(
    action: str,
    datasource: str,
    *,
    filename: str = "",
    chunks: int = 0,
    status: str = "ok",
    error: str = "",
) -> None:
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "datasource": datasource,
        "filename": filename,
        "chunks": chunks,
        "status": status,
        "error": error,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_retrieve_event(
    *,
    user: str,
    model_name: str = "",
    datasources_queried: list[str],
    query: str,
    hits: int,
    status: str = "ok",
    error: str = "",
) -> None:
    path = Path(settings.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "retrieve",
        "user": user,
        "model_name": model_name,
        "datasources_queried": datasources_queried,
        "query_length": len(query),
        "hits": hits,
        "status": status,
        "error": error,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_recent_events(n: int = 20) -> list[dict]:
    path = Path(settings.audit_log_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-n:] if len(lines) > n else lines
    return [json.loads(line) for line in recent]
