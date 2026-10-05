"""Persist extension analyze requests for admin inspection."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.data_paths import data_dir

logger = logging.getLogger(__name__)

LOG_PATH = data_dir() / "request_log.json"

MAX_ENTRIES = 5000
DEFAULT_PAGE_SIZE = 10

_lock = threading.Lock()
_memory: list[dict[str, Any]] = []
_last_persist_error: str | None = None


def _ensure_data_dir() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)


def _read_file_entries() -> list[dict[str, Any]]:
    _ensure_data_dir()
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read request log file: %s", exc)
    return []


def _write_file_entries(entries: list[dict[str, Any]]) -> None:
    global _last_persist_error
    _ensure_data_dir()
    tmp_path = LOG_PATH.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)
        tmp_path.replace(LOG_PATH)
        _last_persist_error = None
    except OSError as exc:
        _last_persist_error = str(exc)
        logger.warning("Could not persist request log to %s: %s", LOG_PATH, exc)


def init_from_disk() -> None:
    """Load any on-disk history into memory (startup / after deploy with volume)."""
    global _memory
    file_entries = _read_file_entries()
    with _lock:
        if len(file_entries) > len(_memory):
            _memory = file_entries[:MAX_ENTRIES]


def format_response_for_log(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    answer = result.get("answer")
    if answer is not None:
        kind = result.get("type") or "unknown"
        return f"[{kind}] {answer}"
    return json.dumps(result, ensure_ascii=False)


def append_entry(
    *,
    kind: str,
    provider: str,
    model: str,
    mode: str,
    request_text: str,
    response_text: str = "",
    ok: bool = True,
    error: str | None = None,
) -> None:
    entry = {
        "id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "kind": kind,
        "provider": provider,
        "model": model,
        "mode": mode,
        "request_text": request_text,
        "response_text": response_text,
        "ok": ok,
        "error": error,
    }
    snapshot: list[dict[str, Any]]
    with _lock:
        _memory.insert(0, entry)
        if len(_memory) > MAX_ENTRIES:
            del _memory[MAX_ENTRIES:]
        snapshot = list(_memory)
    _write_file_entries(snapshot)


def entry_count() -> int:
    with _lock:
        return len(_memory)


def latest_timestamp() -> str | None:
    with _lock:
        if not _memory:
            return None
        return _memory[0].get("ts")


def get_page(page: int, per_page: int = DEFAULT_PAGE_SIZE) -> tuple[list[dict[str, Any]], int, int, int]:
    per_page = max(1, min(per_page, 100))
    with _lock:
        entries = list(_memory)
    total = len(entries)
    total_pages = max(1, (total + per_page - 1) // per_page)
    current_page = max(1, min(page, total_pages))
    start = (current_page - 1) * per_page
    return entries[start : start + per_page], total, total_pages, current_page


def debug_info() -> dict[str, Any]:
    with _lock:
        count = len(_memory)
    return {
        "log_path": str(LOG_PATH),
        "memory_entries": count,
        "last_persist_error": _last_persist_error,
    }
