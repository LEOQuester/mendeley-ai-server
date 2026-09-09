import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app import ref_doc

BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_PATH = BASE_DIR / "data" / "chat_sessions.json"

SESSION_IDLE_HOURS = 2
MAX_HISTORY_TURNS = 4

_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_all() -> dict[str, Any]:
    if not SESSIONS_PATH.exists():
        return {"sessions": {}}
    with SESSIONS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_all(data: dict[str, Any]) -> None:
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SESSIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def clear_all_sessions() -> None:
    with _lock:
        _save_all({"sessions": {}})


def delete_session(session_id: str) -> None:
    with _lock:
        data = _load_all()
        data.get("sessions", {}).pop(session_id, None)
        _save_all(data)


def active_session_count() -> int:
    data = _load_all()
    return len(data.get("sessions", {}))


def _current_doc_version() -> str | None:
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return None
    return meta.get("doc_version")


def _session_is_stale(session: dict[str, Any]) -> bool:
    doc_version = _current_doc_version()
    if not doc_version or session.get("doc_version") != doc_version:
        return True
    last_used = session.get("last_used_at")
    if not last_used:
        return True
    try:
        idle_limit = _now() - timedelta(hours=SESSION_IDLE_HOURS)
        return _parse_iso(last_used) < idle_limit
    except ValueError:
        return True


def _new_session(model: str) -> tuple[str, dict[str, Any]]:
    doc_version = _current_doc_version() or "none"
    session_id = uuid.uuid4().hex
    now = _now().isoformat()
    session = {
        "id": session_id,
        "doc_version": doc_version,
        "model": model,
        "created_at": now,
        "last_used_at": now,
        "turns": [],
    }
    return session_id, session


def get_or_create_session(session_id: str | None, model: str) -> tuple[str, dict[str, Any]]:
    with _lock:
        data = _load_all()
        sessions: dict[str, Any] = data.setdefault("sessions", {})

        if session_id and session_id in sessions:
            session = sessions[session_id]
            if not _session_is_stale(session) and session.get("model") == model:
                session["last_used_at"] = _now().isoformat()
                _save_all(data)
                return session_id, session
            sessions.pop(session_id, None)

        new_id, session = _new_session(model)
        sessions[new_id] = session
        _save_all(data)
        return new_id, session


def append_turn(session_id: str, question: str, answer: str) -> None:
    with _lock:
        data = _load_all()
        session = data.get("sessions", {}).get(session_id)
        if not session:
            return
        session.setdefault("turns", []).append(
            {
                "question": question[:2000],
                "answer": answer[:2000],
                "at": _now().isoformat(),
            }
        )
        session["turns"] = session["turns"][-MAX_HISTORY_TURNS:]
        session["last_used_at"] = _now().isoformat()
        _save_all(data)


def gemini_history_contents(session: dict[str, Any]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for turn in session.get("turns", []):
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if not question:
            continue
        contents.append({"role": "user", "parts": [{"text": question}]})
        if answer:
            contents.append({"role": "model", "parts": [{"text": answer}]})
    return contents


def groq_history_messages(session: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in session.get("turns", []):
        question = str(turn.get("question") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
    return messages
