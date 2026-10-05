"""Persist extension analyze requests for admin inspection."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_PATH = DATA_DIR / "request_log.json"

MAX_ENTRIES = 5000
DEFAULT_PAGE_SIZE = 10

_lock = threading.Lock()


def _ensure_log_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", encoding="utf-8") as handle:
            json.dump([], handle)


def _load_entries() -> list[dict[str, Any]]:
    _ensure_log_file()
    with LOG_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return data


def _save_entries(entries: list[dict[str, Any]]) -> None:
    _ensure_log_file()
    with LOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


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
    with _lock:
        entries = _load_entries()
        entries.insert(0, entry)
        if len(entries) > MAX_ENTRIES:
            entries = entries[:MAX_ENTRIES]
        _save_entries(entries)


def get_page(page: int, per_page: int = DEFAULT_PAGE_SIZE) -> tuple[list[dict[str, Any]], int, int, int]:
    per_page = max(1, min(per_page, 100))
    with _lock:
        entries = _load_entries()
    total = len(entries)
    total_pages = max(1, (total + per_page - 1) // per_page)
    current_page = max(1, min(page, total_pages))
    start = (current_page - 1) * per_page
    return entries[start : start + per_page], total, total_pages, current_page
