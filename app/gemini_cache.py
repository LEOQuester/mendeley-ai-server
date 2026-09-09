import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import httpx

from app import ref_doc
from app.prompts import REFERENCE_DOC_PREAMBLE, SYSTEM_PROMPT_MCQ

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600
IMPLICIT_CACHE_MIN_TOKENS = 4096


class CacheCreateResult(NamedTuple):
    cache: dict[str, Any] | None
    mode: str
    note: str


class CacheError(Exception):
    pass


def explicit_cache_min_tokens(model: str) -> int:
    """Model-aware minimum for explicit cachedContents (Google docs, Mar 2026)."""
    model_id = str(model or "").lower()
    if model_id.startswith("gemini-3"):
        return 4096
    if model_id.startswith("gemini-2"):
        return 2048
    return IMPLICIT_CACHE_MIN_TOKENS


def build_cache_payload(system_prompt: str, ref_text: str) -> tuple[str, int, int]:
    combined = f"{REFERENCE_DOC_PREAMBLE}\n{ref_text.strip()}"
    typical = ref_doc.estimate_tokens(combined) + ref_doc.estimate_tokens(system_prompt)
    upper = ref_doc.estimate_tokens_upper_bound(combined) + ref_doc.estimate_tokens_upper_bound(system_prompt)
    return combined, typical, upper


def _is_free_tier_cache_block(message: str) -> bool:
    return bool(re.search(r"TotalCachedContentStorageTokensPerModelFreeTier|limit=0", message, re.I))


def _is_cache_size_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "minimum token",
            "min token",
            "too few tokens",
            "cached content is of",
            "invalid_argument",
            "invalid argument",
        )
    )


def _is_retryable_cache_error(status: int | None, message: str) -> bool:
    if status in {400, 403, 429, 500, 502, 503, 504}:
        return True
    return _is_free_tier_cache_block(message) or _is_cache_size_error(message)


async def try_create_explicit_cache(
    api_key: str,
    model: str,
    system_prompt: str,
    ref_text: str,
) -> CacheCreateResult:
    """Try explicit cache; always fall back to implicit on any expected failure."""
    combined, typical_tokens, upper_tokens = build_cache_payload(system_prompt, ref_text)
    min_tokens = explicit_cache_min_tokens(model)

    if typical_tokens < min_tokens:
        note = (
            f"Document ~{typical_tokens:,} typical tokens "
            f"(upper bound ~{upper_tokens:,}); explicit cache needs ≥{min_tokens:,} for {model}."
        )
        logger.info("Skipping explicit Gemini cache: %s", note)
        return CacheCreateResult(None, "implicit", note)

    url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={api_key}"
    body = {
        "model": f"models/{model}",
        "displayName": "mendeley-ref-doc",
        "contents": [{"role": "user", "parts": [{"text": combined}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "ttl": f"{CACHE_TTL_SECONDS}s",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=body, headers={"Content-Type": "application/json"})

    if response.is_success:
        data = response.json()
        expires = datetime.now(UTC) + timedelta(seconds=CACHE_TTL_SECONDS)
        note = (
            f"Explicit cache created (~{typical_tokens:,} typical tokens, "
            f"upper bound ~{upper_tokens:,})."
        )
        return CacheCreateResult(
            {
                "name": data.get("name"),
                "expires_at": expires.isoformat(),
                "mode": "explicit",
            },
            "explicit",
            note,
        )

    message = response.text
    status = response.status_code

    if _is_free_tier_cache_block(message):
        note = "Free tier blocked explicit cache (limit=0). Using implicit stable-prefix mode."
    elif _is_cache_size_error(message):
        note = f"Explicit cache rejected size/min tokens ({status}). Using implicit stable-prefix mode."
    elif status == 429:
        note = "Explicit cache rate-limited (429). Using implicit stable-prefix mode."
    elif status in {400, 403}:
        note = f"Explicit cache unavailable ({status}). Using implicit stable-prefix mode."
    else:
        note = f"Explicit cache failed ({status}). Using implicit stable-prefix mode."

    logger.warning("Gemini explicit cache fallback: %s Detail: %s", note, message[:240])
    return CacheCreateResult(None, "implicit", note)


async def refresh_gemini_cache_for_doc(api_key: str, model: str, ref_text: str) -> CacheCreateResult:
    """Delete old cache if present, then attempt explicit create with implicit fallback."""
    meta = ref_doc.load_meta()
    old_cache = meta.get("gemini_cache_name") if meta else None
    if old_cache:
        await delete_explicit_cache(api_key, old_cache)

    try:
        return await try_create_explicit_cache(api_key, model, SYSTEM_PROMPT_MCQ, ref_text)
    except Exception as exc:
        logger.warning("Gemini cache refresh exception: %s", exc)
        _, typical, upper = build_cache_payload(SYSTEM_PROMPT_MCQ, ref_text)
        return CacheCreateResult(
            None,
            "implicit",
            f"Explicit cache error ({exc}). Using implicit stable-prefix mode "
            f"(~{typical:,} typical / ~{upper:,} upper-bound tokens).",
        )


async def delete_explicit_cache(api_key: str, cache_name: str | None) -> None:
    if not cache_name:
        return
    url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.delete(url)
    except Exception:
        return


def cache_is_valid(meta: dict[str, Any] | None, model: str) -> bool:
    if not meta:
        return False
    if meta.get("gemini_cache_mode") != "explicit":
        return False
    if meta.get("gemini_cache_model") != model:
        return False
    if not meta.get("gemini_cache_name"):
        return False
    expires = meta.get("gemini_cache_expires_at")
    if not expires:
        return False
    try:
        return datetime.fromisoformat(expires.replace("Z", "+00:00")) > datetime.now(UTC)
    except ValueError:
        return False


def uses_implicit_stable_prefix(meta: dict[str, Any] | None) -> bool:
    return bool(meta and meta.get("enabled") and meta.get("gemini_cache_mode") == "implicit")
