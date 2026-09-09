import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app import ref_doc
from app.prompts import REFERENCE_DOC_PREAMBLE

CACHE_TTL_SECONDS = 3600
IMPLICIT_CACHE_MIN_TOKENS = 4096


class CacheError(Exception):
    pass


def _is_free_tier_cache_block(message: str) -> bool:
    return bool(re.search(r"TotalCachedContentStorageTokensPerModelFreeTier|limit=0", message, re.I))


async def try_create_explicit_cache(
    api_key: str,
    model: str,
    system_prompt: str,
    ref_text: str,
) -> dict[str, Any] | None:
    """Create Gemini explicit cache. Returns None if free tier blocks it."""
    combined = f"{REFERENCE_DOC_PREAMBLE}\n{ref_text.strip()}"
    if ref_doc.estimate_tokens(combined) < IMPLICIT_CACHE_MIN_TOKENS:
        return None

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
        return {
            "name": data.get("name"),
            "expires_at": expires.isoformat(),
            "mode": "explicit",
        }

    message = response.text
    if _is_free_tier_cache_block(message):
        return None
    if response.status_code in {400, 403, 429}:
        return None
    raise CacheError(f"Gemini cache create failed ({response.status_code}): {message[:220]}")


async def delete_explicit_cache(api_key: str, cache_name: str | None) -> None:
    if not cache_name:
        return
    url = f"https://generativelanguage.googleapis.com/v1beta/{cache_name}?key={api_key}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.delete(url)


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
