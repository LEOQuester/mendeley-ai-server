import re
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

KEY_COOLDOWN_MS = 60_000
KEY_DEAD_MS = 10 * 60_000

_key_cooldowns: dict[str, float] = {}


def _is_rate_limit_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {429, 503}:
        return True
    if status == 403 and any(token in msg for token in ("quota", "exhausted", "resource_exhausted")):
        return True
    return any(
        token in msg
        for token in (
            "429",
            "503",
            "rate limit",
            "rate_limit",
            "too many requests",
            "resource_exhausted",
            "quota exceeded",
            "quota",
            "exhausted",
        )
    )


def _is_gemini_overload_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {500, 502, 503, 504}:
        return True
    return bool(re.search(r"\bunavailable\b|\boverloaded\b|high demand|try again later", msg))


def _is_rotatable_key_error(message: str, status: int | None) -> bool:
    if _is_rate_limit_error(message, status) or _is_gemini_overload_error(message, status):
        return True
    msg = message.lower()
    if status in {401, 403}:
        return True
    return bool(
        re.search(
            r"api[_ ]key|not valid|expired|permission_denied|unauthorized|consumer_invalid|api_key_invalid",
            msg,
        )
    )


def _cooldown_ms_for_error(message: str, status: int | None) -> int:
    msg = message.lower()
    if re.search(r"api[_ ]key|not valid|expired|unauthorized|consumer_invalid", msg) or status == 401:
        return KEY_DEAD_MS
    if _is_gemini_overload_error(message, status) or _is_rate_limit_error(message, status):
        return 0
    return KEY_COOLDOWN_MS


def _mark_key_cooldown(key: str, message: str, status: int | None) -> None:
    ms = _cooldown_ms_for_error(message, status)
    if not key or ms <= 0:
        return
    _key_cooldowns[key] = time.time() * 1000 + ms


def _is_key_on_cooldown(key: str) -> bool:
    until = _key_cooldowns.get(key)
    if not until:
        return False
    if until <= time.time() * 1000:
        _key_cooldowns.pop(key, None)
        return False
    return True


async def retry_with_rotation_async(
    keys: list[str],
    api_fn: Callable[[str], Awaitable[T]],
    max_keys: int | None = None,
) -> T:
    if not keys:
        raise RuntimeError("No API keys configured for this provider.")

    limit = max_keys or len(keys)
    tried: set[str] = set()
    last_error: Exception | None = None

    fresh = [key for key in keys if not _is_key_on_cooldown(key)]
    cooling = [key for key in keys if _is_key_on_cooldown(key)]
    ordered = fresh + cooling

    for key in ordered:
        if len(tried) >= limit or key in tried:
            continue
        tried.add(key)
        try:
            return await api_fn(key)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            status = getattr(exc, "status", None)
            if not _is_rotatable_key_error(message, status):
                raise
            _mark_key_cooldown(key, message, status)

    raise last_error or RuntimeError("All API keys failed.")
