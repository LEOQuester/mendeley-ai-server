"""Persist API keys and config metadata in MySQL (survives redeploys)."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)

# Default MySQL (Railway/local — env vars override if set)
MYSQL_HOST_DEFAULT = "primeict.lk"
MYSQL_PORT_DEFAULT = 3306
MYSQL_USER_DEFAULT = "primeic1_mcq_tool_user"
MYSQL_PASSWORD_DEFAULT = "Mcqtooluser123!@"
MYSQL_DATABASE_DEFAULT = "primeict1_mcq_tool"

CONFIG_META_KEY = "config_meta"
_lock = threading.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
  id INT AUTO_INCREMENT PRIMARY KEY,
  provider VARCHAR(16) NOT NULL,
  key_value TEXT NOT NULL,
  sort_order INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_provider_sort (provider, sort_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS app_settings (
  setting_key VARCHAR(64) PRIMARY KEY,
  setting_value LONGTEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _db_setting(name: str, default: str | int) -> str:
    raw = os.getenv(name, "").strip()
    if raw:
        return raw
    return str(default)


def mysql_enabled() -> bool:
    return True


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=_db_setting("MYSQL_HOST", MYSQL_HOST_DEFAULT),
        user=_db_setting("MYSQL_USER", MYSQL_USER_DEFAULT),
        password=os.getenv("MYSQL_PASSWORD", "").strip() or MYSQL_PASSWORD_DEFAULT,
        database=_db_setting("MYSQL_DATABASE", MYSQL_DATABASE_DEFAULT),
        port=int(_db_setting("MYSQL_PORT", MYSQL_PORT_DEFAULT)),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=15,
        read_timeout=30,
        write_timeout=30,
    )


@contextmanager
def _connection() -> Iterator[pymysql.connections.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    if not mysql_enabled():
        return
    with _connection() as conn:
        with conn.cursor() as cur:
            for statement in _SCHEMA_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    cur.execute(stmt)
    logger.info("MySQL schema ready (%s)", _db_setting("MYSQL_DATABASE", MYSQL_DATABASE_DEFAULT))


def _load_keys(provider: str) -> list[str]:
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT key_value FROM api_keys
                WHERE provider = %s
                ORDER BY sort_order ASC, id ASC
                """,
                (provider,),
            )
            rows = cur.fetchall()
    return [str(row["key_value"]).strip() for row in rows if str(row["key_value"]).strip()]


def _save_keys_for_provider(provider: str, keys: list[str]) -> None:
    cleaned = [k.strip() for k in keys if k and k.strip()]
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE provider = %s", (provider,))
            for index, key in enumerate(cleaned):
                cur.execute(
                    """
                    INSERT INTO api_keys (provider, key_value, sort_order)
                    VALUES (%s, %s, %s)
                    """,
                    (provider, key, index),
                )


def save_keys(gemini_keys: list[str], groq_keys: list[str]) -> None:
    with _lock:
        _save_keys_for_provider("gemini", gemini_keys)
        _save_keys_for_provider("groq", groq_keys)


def load_keys() -> tuple[list[str], list[str]]:
    with _lock:
        return _load_keys("gemini"), _load_keys("groq")


def load_config_meta(default_meta: dict[str, Any]) -> dict[str, Any]:
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT setting_value FROM app_settings WHERE setting_key = %s",
                (CONFIG_META_KEY,),
            )
            row = cur.fetchone()
    if not row:
        return default_meta
    try:
        parsed = json.loads(row["setting_value"])
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return default_meta


def save_config_meta(meta: dict[str, Any]) -> None:
    payload = json.dumps(meta, ensure_ascii=False)
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_settings (setting_key, setting_value)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                """,
                (CONFIG_META_KEY, payload),
            )


def migrate_from_file_config(file_config: dict[str, Any]) -> None:
    """One-time import when DB is empty but local config.json has data."""
    if not mysql_enabled():
        return
    gemini, groq = load_keys()
    if gemini or groq:
        return

    file_gemini = file_config.get("gemini_keys") or []
    file_groq = file_config.get("groq_keys") or []
    if not file_gemini and not file_groq:
        return

    save_keys(list(file_gemini), list(file_groq))
    logger.info(
        "Imported API keys from file into MySQL (gemini=%s, groq=%s)",
        len(file_gemini),
        len(file_groq),
    )

    meta = {k: v for k, v in file_config.items() if k not in ("gemini_keys", "groq_keys")}
    if meta:
        save_config_meta(meta)


def ping() -> bool:
    if not mysql_enabled():
        return False
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
    return bool(row and row.get("ok") == 1)
