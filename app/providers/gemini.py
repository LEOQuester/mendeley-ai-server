import asyncio
import json
from typing import Any

import httpx

from app import chat_sessions, ref_doc
from app.gemini_cache import cache_is_valid
from app.prompts import (
    API_TEST_PROMPT,
    PING_TEST_PROMPT,
    SYSTEM_PROMPT_DESCRIPTIVE,
    SYSTEM_PROMPT_MCQ,
    TEXT_ONLY_PREAMBLE,
    TEXT_RESPONSE_SCHEMA,
    VISION_SYSTEM_PROMPT,
)
from app.providers.key_rotation import is_invalid_key_error, retry_with_rotation_async
from app.providers.response_parser import parse_ai_response

TEXT_FETCH_TIMEOUT = 60.0
VISION_FETCH_TIMEOUT = 60.0

_gemini_http_client: httpx.AsyncClient | None = None


def _gemini_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(max(TEXT_FETCH_TIMEOUT, VISION_FETCH_TIMEOUT))


async def _get_gemini_http_client() -> httpx.AsyncClient:
    global _gemini_http_client
    if _gemini_http_client is None or _gemini_http_client.is_closed:
        _gemini_http_client = httpx.AsyncClient(
            timeout=_gemini_http_timeout(),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _gemini_http_client


async def close_gemini_http_client() -> None:
    global _gemini_http_client
    if _gemini_http_client is not None and not _gemini_http_client.is_closed:
        await _gemini_http_client.aclose()
    _gemini_http_client = None

GEMINI_TEXT_OVERLOAD_FALLBACKS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

VISION_REQUEST_VARIANTS = [
    {"snake": False, "schema": False, "system_in_user": True},
    {"snake": True, "schema": False, "system_in_user": True},
    {"snake": False, "schema": True, "system_in_user": False},
]

VISION_TEST_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _is_gemini3_model(model_id: str) -> bool:
    return str(model_id or "").startswith("gemini-3")


def _gemini_vision_thinking_level(model_id: str) -> str:
    return "MINIMAL" if "lite" in str(model_id or "").lower() else "LOW"


def _build_gemini_text_generation_config(model_id: str, text_mode: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "responseSchema": TEXT_RESPONSE_SCHEMA,
        "maxOutputTokens": 4096 if text_mode == "mcq" else 6144,
    }
    if _is_gemini3_model(model_id):
        config["thinkingConfig"] = {"thinkingLevel": "HIGH"}
    else:
        config["temperature"] = 0
    return config


def _get_gemini_response_text(data: dict[str, Any]) -> str | None:
    block = data.get("promptFeedback", {}).get("blockReason")
    if block:
        raise ProviderError(f"Gemini blocked the request ({block}).")

    candidate = (data.get("candidates") or [None])[0]
    if not candidate:
        raise ProviderError("Gemini returned no answer.")

    reason = str(candidate.get("finishReason") or candidate.get("finish_reason") or "").upper()
    if reason in {"SAFETY", "BLOCKED"}:
        raise ProviderError("Gemini blocked this request.")

    parts = candidate.get("content", {}).get("parts") or []
    visible = [
        part for part in parts
        if part.get("thought") is not True and str(part.get("text") or "").strip()
    ]
    if visible:
        return str(visible[-1]["text"]).strip()

    any_text = [str(part.get("text") or "").strip() for part in parts if str(part.get("text") or "").strip()]
    return any_text[-1] if any_text else None


def _extract_provider_error_message(err_text: str) -> str:
    try:
        import json

        parsed = json.loads(err_text)
        return parsed.get("error", {}).get("message") or parsed.get("message") or err_text
    except Exception:
        return err_text


def _raise_http_error(label: str, status: int, err_text: str) -> None:
    detail = _extract_provider_error_message(err_text)
    raise ProviderError(f"{label} ({status}): {detail[:220]}", status=status)


def _is_gemini_overload_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {500, 502, 503, 504}:
        return True
    import re

    return bool(re.search(r"\bunavailable\b|\boverloaded\b|high demand|try again later", msg))


def _is_rate_limit_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if status in {429, 503}:
        return True
    return any(token in msg for token in ("429", "503", "rate limit", "quota", "exhausted"))


def _is_vision_retryable_error(message: str, status: int | None) -> bool:
    msg = message.lower()
    if is_invalid_key_error(message, status):
        return False
    if "all api key(s) failed" in msg or "all " in msg and "api key(s) failed" in msg:
        return _is_gemini_overload_error(message, status) or _is_rate_limit_error(message, status)
    return (
        _is_gemini_overload_error(message, status)
        or _is_rate_limit_error(message, status)
        or "empty response from gemini vision" in msg
    )


def _should_try_next_gemini_text_model(exc: Exception) -> bool:
    message = str(exc)
    status = getattr(exc, "status", None)
    if is_invalid_key_error(message, status):
        return False
    return _is_gemini_overload_error(message, status) or _is_rate_limit_error(message, status)


def _build_gemini_vision_generation_config(model_id: str, use_schema: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "maxOutputTokens": 512,
        "thinkingConfig": {"thinkingLevel": _gemini_vision_thinking_level(model_id)},
    }
    if use_schema:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = TEXT_RESPONSE_SCHEMA
    return config


def _build_gemini_vision_contents(base64_image: str, mime_type: str, variant: dict[str, Any]) -> list[dict[str, Any]]:
    if variant["snake"]:
        image_part = {"inline_data": {"mime_type": mime_type, "data": base64_image}}
    else:
        image_part = {"inlineData": {"mimeType": mime_type, "data": base64_image}}

    prompt_prefix = f"{VISION_SYSTEM_PROMPT}\n\n" if variant["system_in_user"] else ""
    user_prompt = (
        f"{prompt_prefix}Read the exam question in this screenshot. "
        'If MCQ with options: {"type":"mcq","answer":"exact option text(s)","source":"document"|"general"}. '
        'If not MCQ: {"type":"descriptive","answer":"one short sentence","source":"document"|"general"}. '
        "JSON only."
    )
    return [{"parts": [image_part, {"text": user_prompt}]}]


def _build_gemini_vision_request_body(
    model: str,
    base64_image: str,
    mime_type: str,
    variant: dict[str, Any],
    ref_excerpt: str = "",
) -> dict[str, Any]:
    system_text = VISION_SYSTEM_PROMPT
    if ref_excerpt and not variant["system_in_user"]:
        system_text = f"{VISION_SYSTEM_PROMPT}\n\n{ref_excerpt}"

    body: dict[str, Any] = {
        "contents": _build_gemini_vision_contents(base64_image, mime_type, variant),
        "generationConfig": _build_gemini_vision_generation_config(model, variant["schema"]),
    }
    if variant["system_in_user"] and ref_excerpt:
        body["contents"][0]["parts"].insert(0, {"text": ref_excerpt})
    elif not variant["system_in_user"]:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    return body


async def _post_gemini(model: str, api_key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    client = await _get_gemini_http_client()
    response = await client.post(
        url,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if not response.is_success:
        _raise_http_error("Gemini API error", response.status_code, response.text)
    return response.json()


def _compose_user_text(question_text: str, ref_excerpt: str) -> str:
    if ref_excerpt:
        return f"{ref_excerpt}\n\n{TEXT_ONLY_PREAMBLE}{question_text}"
    return f"{TEXT_ONLY_PREAMBLE}{question_text}"


async def request_gemini_text(
    api_key: str,
    model: str,
    question_text: str,
    text_mode: str,
    ref_excerpt: str = "",
    session: dict[str, Any] | None = None,
) -> dict[str, str]:
    system_prompt = SYSTEM_PROMPT_MCQ if text_mode == "mcq" else SYSTEM_PROMPT_DESCRIPTIVE
    meta = ref_doc.load_meta()
    history = chat_sessions.gemini_history_contents(session) if session else []
    user_text = f"{TEXT_ONLY_PREAMBLE}{question_text}"
    body: dict[str, Any] = {
        "generationConfig": _build_gemini_text_generation_config(model, text_mode),
    }

    if cache_is_valid(meta, model):
        body["cachedContent"] = meta["gemini_cache_name"]
        body["contents"] = history + [{"role": "user", "parts": [{"text": user_text}]}]
    else:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        composed = _compose_user_text(question_text, ref_excerpt)
        body["contents"] = history + [{"role": "user", "parts": [{"text": composed}]}]

    data = await _post_gemini(model, api_key, body, TEXT_FETCH_TIMEOUT)
    text = _get_gemini_response_text(data)
    if not text:
        reason = (data.get("candidates") or [{}])[0].get("finishReason")
        if str(reason).upper() == "MAX_TOKENS":
            raise ProviderError("Gemini ran out of output tokens while answering.")
        raise ProviderError("Empty response from Gemini")
    return parse_ai_response(text, default_type=text_mode)


async def call_gemini_text(
    keys: list[str],
    model: str,
    question_text: str,
    text_mode: str,
    ref_excerpt: str = "",
    session_id: str | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, str]:
    models = [model] + [item for item in GEMINI_TEXT_OVERLOAD_FALLBACKS if item != model]
    last_error: Exception | None = None

    for current_model in models:
        for attempt in range(2):
            try:
                result = await retry_with_rotation_async(
                    keys,
                    lambda key, current=current_model: request_gemini_text(
                        key, current, question_text, text_mode, ref_excerpt, session
                    ),
                    provider_label="Gemini",
                )
                if session_id:
                    chat_sessions.append_turn(session_id, question_text, json.dumps(result))
                return result
            except Exception as exc:
                last_error = exc
                message = str(exc)
                status = getattr(exc, "status", None)
                if not _should_try_next_gemini_text_model(exc):
                    raise
                if attempt == 0 and _is_gemini_overload_error(message, status):
                    await asyncio.sleep(0.4)
                    continue
                break

    raise last_error or ProviderError("Gemini text request failed.")


async def request_gemini_vision_once(
    api_key: str,
    model: str,
    base64_image: str,
    mime_type: str,
    variant: dict[str, Any],
    ref_excerpt: str = "",
) -> dict[str, str]:
    body = _build_gemini_vision_request_body(model, base64_image, mime_type, variant, ref_excerpt)
    data = await _post_gemini(model, api_key, body, VISION_FETCH_TIMEOUT)
    text = _get_gemini_response_text(data)
    if not text:
        reason = (data.get("candidates") or [{}])[0].get("finishReason")
        raise ProviderError(f"Empty response from Gemini Vision ({reason or 'no text'})")
    try:
        return parse_ai_response(text)
    except ValueError:
        return {"type": "mcq", "answer": text[:500]}


async def request_gemini_vision(
    api_key: str,
    model: str,
    base64_image: str,
    mime_type: str,
    ref_excerpt: str = "",
) -> dict[str, str]:
    last_error: Exception | None = None
    for variant in VISION_REQUEST_VARIANTS:
        try:
            return await request_gemini_vision_once(
                api_key, model, base64_image, mime_type, variant, ref_excerpt
            )
        except Exception as exc:
            last_error = exc
            message = str(exc)
            status = getattr(exc, "status", None)
            if not _is_vision_retryable_error(message, status):
                raise
    raise last_error or ProviderError("Gemini Vision failed.")


def _should_try_next_gemini_vision_model(exc: Exception) -> bool:
    message = str(exc)
    status = getattr(exc, "status", None)
    if is_invalid_key_error(message, status):
        return False
    return _is_gemini_overload_error(message, status) or _is_rate_limit_error(message, status)


async def call_gemini_vision(
    keys: list[str],
    model: str,
    base64_image: str,
    mime_type: str,
    ref_excerpt: str = "",
) -> dict[str, str]:
    preferred = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"]
    models = [model] + [item for item in preferred if item != model][:2]
    last_error: Exception | None = None

    for current_model in models:
        try:
            return await retry_with_rotation_async(
                keys,
                lambda key, current=current_model: request_gemini_vision(
                    key, current, base64_image, mime_type, ref_excerpt
                ),
                provider_label="Gemini",
            )
        except Exception as exc:
            last_error = exc
            if not _should_try_next_gemini_vision_model(exc):
                raise

    raise last_error or ProviderError("Gemini Vision failed.")


async def test_gemini_text_key(api_key: str, model: str) -> str:
    data = await _post_gemini(
        model,
        api_key,
        {
            "contents": [{"parts": [{"text": API_TEST_PROMPT}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": TEXT_RESPONSE_SCHEMA,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        },
        30.0,
    )
    text = _get_gemini_response_text(data)
    if not text:
        raise ProviderError("Gemini text returned an empty response.")
    parse_ai_response(text)
    return f"Text: {model} OK."


async def test_gemini_vision_key(api_key: str, model: str) -> str:
    await request_gemini_vision(api_key, model, VISION_TEST_PNG, "image/png")
    return f"Vision: {model} OK."


async def ping_gemini_text_key(api_key: str, model: str) -> str:
    data = await _post_gemini(
        model,
        api_key,
        {
            "contents": [{"parts": [{"text": PING_TEST_PROMPT}]}],
            "generationConfig": {
                "maxOutputTokens": 16,
                "temperature": 0,
            },
        },
        30.0,
    )
    text = _get_gemini_response_text(data)
    if not text:
        raise ProviderError("Gemini text returned an empty response.")
    return text.strip()
