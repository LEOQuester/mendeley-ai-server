import logging
import re
import time
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

KEY_COOLDOWN_MS = 60_000
KEY_DEAD_MS = 10 * 60_000

_key_cooldowns: dict[str, float] = {}


def _is_rate_limit_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {429, 503}:
        return True
    if status == 403 and any(token in msg for token in ("quota", "exhausted", "resource_exhausted", "rate")):
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
            "resource exhausted",
            "quota exceeded",
            "quota",
            "exhausted",
            "tokens per minute",
            "tpm",
            "rpm",
        )
    )


def _is_gemini_overload_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {500, 502, 503, 504}:
        return True
    return bool(re.search(r"\bunavailable\b|\boverloaded\b|high demand|try again later", msg))


def _is_invalid_key_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status == 401:
        return True
    if status == 403 and any(
        token in msg for token in ("api key", "api_key", "permission", "denied", "invalid", "unauthorized")
    ):
        return True
    if status == 400 and re.search(
        r"api[_ ]?key|invalid[_ ]?key|key[_ ]?invalid|bad[_ ]?key|permission[_ ]?denied|unauthorized",
        msg,
    ):
        return True
    return bool(
        re.search(
            r"api[_ ]key.*(invalid|expired|disabled|not valid)|"
            r"invalid[_ ]api[_ ]key|"
            r"incorrect api key|"
            r"api_key_invalid|"
            r"consumer_invalid|"
            r"permission_denied|"
            r"unauthenticated",
            msg,
        )
    )


def _is_rotatable_key_error(message: str, status: int | None) -> bool:
    if _is_rate_limit_error(message, status):
        return True
    if _is_gemini_overload_error(message, status):
        return True
    if _is_invalid_key_error(message, status):
        return True
    return False


def _cooldown_ms_for_error(message: str, status: int | None) -> int:
    if _is_invalid_key_error(message, status):
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


def _key_preview(api_key: str) -> str:
    if len(api_key) >= 4:
        return f"****{api_key[-4:]}"
    return "****"


async def retry_with_rotation_async(
    keys: list[str],
    api_fn: Callable[[str], Awaitable[T]],
    max_keys: int | None = None,
    provider_label: str = "Provider",
) -> T:
    """Try each API key in the pool when errors look rate-limit, overload, or invalid-key."""
    if not keys:
        raise RuntimeError(f"No {provider_label} API keys configured.")

    limit = max_keys or len(keys)
    tried: set[str] = set()
    last_error: Exception | None = None
    failures: list[str] = []

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
            preview = _key_preview(key)
            if not _is_rotatable_key_error(message, status):
                logger.warning("%s key %s failed (non-rotatable): %s", provider_label, preview, message[:160])
                raise
            failures.append(f"{preview}: {message[:120]}")
            logger.info(
                "%s key %s rotatable error (%s), trying next key",
                provider_label,
                preview,
                status or "n/a",
            )
            _mark_key_cooldown(key, message, status)

    summary = "; ".join(failures[-3:]) if failures else str(last_error)
    msg = f"{provider_label}: all {len(tried)} API key(s) failed. {summary[:400]}"
    if last_error is not None:
        status = getattr(last_error, "status", None)
        try:
            from app.providers.groq import ProviderError as GroqProviderError
        except ImportError:
            GroqProviderError = None
        try:
            from app.providers.gemini import ProviderError as GeminiProviderError
        except ImportError:
            GeminiProviderError = None
        for cls in (GroqProviderError, GeminiProviderError):
            if cls and isinstance(last_error, cls):
                raise cls(msg, status) from last_error
        raise type(last_error)(msg) from last_error
    raise RuntimeError(msg)


# Public aliases for provider modules
is_rotatable_provider_error = _is_rotatable_key_error
is_invalid_key_error = _is_invalid_key_error
