import json
import logging
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from app import mysql_store
from app.data_paths import data_dir

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = data_dir()
CONFIG_PATH = DATA_DIR / "config.json"
EXAMPLE_PATH = BASE_DIR / "data" / "config.example.json"

_lock = threading.Lock()


def _parse_env_keys(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def _default_config() -> dict[str, Any]:
    if EXAMPLE_PATH.exists():
        with EXAMPLE_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "gemini_keys": [],
        "groq_keys": [],
        "gemini_text_models": [],
        "gemini_vision_models": [],
        "groq_text_models": [],
        "defaults": {
            "gemini_text": "gemini-3.1-flash-lite",
            "gemini_vision": "gemini-3.7-flash",
            "groq_text": "openai/gpt-oss-120b",
        },
        "ref_doc_settings": {
            "max_inject_tokens_gemini": 5000,
            "max_inject_tokens_groq": 1800,
            "max_inject_tokens_vision": 1500,
            "implicit_stable_prefix_tokens": 4500,
        },
    }


def _config_meta_only(config: dict[str, Any]) -> dict[str, Any]:
    meta = deepcopy(config)
    meta.pop("gemini_keys", None)
    meta.pop("groq_keys", None)
    return meta


def _merge_config(meta: dict[str, Any], gemini_keys: list[str], groq_keys: list[str]) -> dict[str, Any]:
    merged = deepcopy(meta)
    merged["gemini_keys"] = list(gemini_keys)
    merged["groq_keys"] = list(groq_keys)
    return merged


def _load_file_config() -> dict[str, Any]:
    ensure_config()
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def init_db() -> None:
    if not mysql_store.mysql_enabled():
        return
    mysql_store.init_schema()
    if not mysql_store.ping():
        raise RuntimeError("MySQL is configured but connection failed. Check MYSQL_* env vars.")
    try:
        file_config = _load_file_config() if CONFIG_PATH.exists() else _default_config()
        mysql_store.migrate_from_file_config(file_config)
    except Exception as exc:
        logger.warning("MySQL key migration from file skipped: %s", exc)


def ensure_config() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if mysql_store.mysql_enabled():
        return
    if CONFIG_PATH.exists():
        return

    config = _default_config()
    gemini_env = _parse_env_keys("GEMINI_API_KEYS")
    groq_env = _parse_env_keys("GROQ_API_KEYS")
    if gemini_env:
        config["gemini_keys"] = gemini_env
    if groq_env:
        config["groq_keys"] = groq_env

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_config() -> dict[str, Any]:
    with _lock:
        if mysql_store.mysql_enabled():
            default_meta = _config_meta_only(_default_config())
            meta = mysql_store.load_config_meta(default_meta)
            gemini_keys, groq_keys = mysql_store.load_keys()
            return _merge_config(meta, gemini_keys, groq_keys)

        ensure_config()
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)


def save_config(config: dict[str, Any]) -> None:
    with _lock:
        if mysql_store.mysql_enabled():
            mysql_store.save_keys(
                list(config.get("gemini_keys") or []),
                list(config.get("groq_keys") or []),
            )
            mysql_store.save_config_meta(_config_meta_only(config))
            return

        ensure_config()
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)


def public_config() -> dict[str, Any]:
    config = load_config()
    return {
        "gemini_text_models": config.get("gemini_text_models", []),
        "gemini_vision_models": config.get("gemini_vision_models", []),
        "groq_text_models": config.get("groq_text_models", []),
        "defaults": config.get("defaults", {}),
        "has_gemini_keys": bool(config.get("gemini_keys")),
        "has_groq_keys": bool(config.get("groq_keys")),
        "ref_doc": _public_ref_doc_summary(),
    }


def _public_ref_doc_summary() -> dict[str, Any]:
    from app import ref_doc

    summary = ref_doc.admin_summary()
    if not summary:
        return {"enabled": False, "loaded": False}
    return {
        "enabled": summary.get("enabled", False),
        "loaded": True,
        "original_name": summary.get("original_name"),
        "word_count": summary.get("word_count", 0),
        "token_estimate": summary.get("token_estimate", 0),
        "token_estimate_upper": summary.get("token_estimate_upper", 0),
        "chunk_count": summary.get("chunk_count", 0),
        "doc_version": summary.get("doc_version"),
        "gemini_cache_mode": summary.get("gemini_cache_mode", "none"),
    }


def admin_config() -> dict[str, Any]:
    from app import ref_doc

    config = deepcopy(load_config())
    config["gemini_keys"] = _mask_keys(config.get("gemini_keys", []))
    config["groq_keys"] = _mask_keys(config.get("groq_keys", []))
    config["ref_doc"] = ref_doc.admin_summary()
    config.setdefault(
        "ref_doc_settings",
        {
            "max_inject_tokens_gemini": 5000,
            "max_inject_tokens_groq": 1800,
            "max_inject_tokens_vision": 1500,
            "implicit_stable_prefix_tokens": 4500,
        },
    )
    config["storage_backend"] = "mysql" if mysql_store.mysql_enabled() else "file"
    return config


def _mask_keys(keys: list[str]) -> list[dict[str, str]]:
    masked = []
    for index, key in enumerate(keys):
        suffix = key[-4:] if len(key) >= 4 else "****"
        masked.append({"index": index, "preview": f"****{suffix}"})
    return masked


def add_key(provider: str, key: str) -> None:
    key = key.strip()
    if not key:
        raise ValueError("API key cannot be empty.")

    field = "gemini_keys" if provider == "gemini" else "groq_keys"
    config = load_config()
    keys = config.setdefault(field, [])
    if key in keys:
        raise ValueError("This key already exists.")
    keys.append(key)
    save_config(config)


def remove_key(provider: str, index: int) -> None:
    field = "gemini_keys" if provider == "gemini" else "groq_keys"
    config = load_config()
    keys = config.setdefault(field, [])
    if index < 0 or index >= len(keys):
        raise ValueError("Invalid key index.")
    keys.pop(index)
    save_config(config)


def add_model(provider: str, model_type: str, model_id: str, label: str) -> None:
    model_id = model_id.strip()
    label = label.strip()
    if not model_id or not label:
        raise ValueError("Model id and label are required.")

    field_map = {
        ("gemini", "text"): "gemini_text_models",
        ("gemini", "vision"): "gemini_vision_models",
        ("groq", "text"): "groq_text_models",
    }
    field = field_map.get((provider, model_type))
    if not field:
        raise ValueError("Invalid provider or model type.")

    config = load_config()
    models = config.setdefault(field, [])
    if any(model["id"] == model_id for model in models):
        raise ValueError("Model id already exists.")
    models.append({"id": model_id, "label": label})
    save_config(config)


def remove_model(provider: str, model_type: str, model_id: str) -> None:
    field_map = {
        ("gemini", "text"): "gemini_text_models",
        ("gemini", "vision"): "gemini_vision_models",
        ("groq", "text"): "groq_text_models",
    }
    field = field_map.get((provider, model_type))
    if not field:
        raise ValueError("Invalid provider or model type.")

    config = load_config()
    models = config.setdefault(field, [])
    config[field] = [model for model in models if model["id"] != model_id]
    save_config(config)


def update_defaults(defaults: dict[str, str]) -> None:
    config = load_config()
    current = config.setdefault("defaults", {})
    for key in ("gemini_text", "gemini_vision", "groq_text"):
        if key in defaults and defaults[key]:
            current[key] = defaults[key]
    save_config(config)


def update_ref_doc_settings(settings: dict[str, int]) -> None:
    config = load_config()
    current = config.setdefault("ref_doc_settings", {})
    for key in ("max_inject_tokens_gemini", "max_inject_tokens_groq", "max_inject_tokens_vision"):
        if key in settings and settings[key] > 0:
            current[key] = int(settings[key])
    save_config(config)
