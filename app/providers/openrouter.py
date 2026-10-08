import json
from typing import Any

import httpx

from app import chat_sessions
from app.prompts import (
    API_TEST_PROMPT,
    PING_TEST_PROMPT,
    SYSTEM_PROMPT_DESCRIPTIVE,
    SYSTEM_PROMPT_MCQ,
    TEXT_ONLY_PREAMBLE,
    VISION_SYSTEM_PROMPT,
)
from app.providers.groq import ProviderError, _extract_provider_error_message, _raise_http_error
from app.providers.key_rotation import retry_with_rotation_async
from app.providers.response_parser import parse_ai_response

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_APP_TITLE = "Mendeley AI Server"
OPENROUTER_HTTP_REFERER = "https://github.com/LEOQuester/mendeley-ai-server"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
    }


def _compose_user_text(question_text: str, ref_excerpt: str) -> str:
    if ref_excerpt:
        return f"{ref_excerpt}\n\n{TEXT_ONLY_PREAMBLE}{question_text}"
    return f"{TEXT_ONLY_PREAMBLE}{question_text}"


async def request_openrouter_text(
    api_key: str,
    model: str,
    question_text: str,
    text_mode: str,
    ref_excerpt: str = "",
    session: dict[str, Any] | None = None,
) -> dict[str, str]:
    system_prompt = SYSTEM_PROMPT_MCQ if text_mode == "mcq" else SYSTEM_PROMPT_DESCRIPTIVE
    max_tokens = 4096 if text_mode == "mcq" else 6144
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if session:
        messages.extend(chat_sessions.groq_history_messages(session))
    messages.append({"role": "user", "content": _compose_user_text(question_text, ref_excerpt)})
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(OPENROUTER_CHAT_URL, json=payload, headers=_headers(api_key))

    if not response.is_success:
        _raise_http_error("OpenRouter text error", response.status_code, response.text)

    data = response.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Empty response from OpenRouter")
    return parse_ai_response(text)


async def call_openrouter_text(
    keys: list[str],
    model: str,
    question_text: str,
    text_mode: str,
    ref_excerpt: str = "",
    session_id: str | None = None,
    session: dict[str, Any] | None = None,
) -> dict[str, str]:
    result = await retry_with_rotation_async(
        keys,
        lambda key: request_openrouter_text(
            key, model, question_text, text_mode, ref_excerpt=ref_excerpt, session=session
        ),
        provider_label="OpenRouter",
    )
    if session_id:
        chat_sessions.append_turn(session_id, question_text, json.dumps(result))
    return result


async def request_openrouter_vision(
    api_key: str,
    model: str,
    base64_image: str,
    mime_type: str,
    ref_excerpt: str = "",
) -> dict[str, str]:
    data_url = f"data:{mime_type};base64,{base64_image}"
    user_parts: list[dict[str, Any]] = []
    if ref_excerpt:
        user_parts.append({"type": "text", "text": ref_excerpt})
    user_parts.append(
        {
            "type": "text",
            "text": "Analyze this exam screenshot. Return JSON only per the system instructions.",
        }
    )
    user_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_parts},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(OPENROUTER_CHAT_URL, json=payload, headers=_headers(api_key))

    if not response.is_success:
        _raise_http_error("OpenRouter vision error", response.status_code, response.text)

    data = response.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Empty response from OpenRouter Vision")
    try:
        return parse_ai_response(text)
    except ValueError:
        return {"type": "mcq", "answer": text[:500]}


async def call_openrouter_vision(
    keys: list[str],
    model: str,
    base64_image: str,
    mime_type: str,
    ref_excerpt: str = "",
) -> dict[str, str]:
    return await retry_with_rotation_async(
        keys,
        lambda key: request_openrouter_vision(key, model, base64_image, mime_type, ref_excerpt),
        provider_label="OpenRouter",
    )


async def test_openrouter_key(api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            OPENROUTER_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a test endpoint. Reply with valid JSON only, no markdown.",
                    },
                    {"role": "user", "content": API_TEST_PROMPT},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 64,
            },
            headers=_headers(api_key),
        )

    if not response.is_success:
        _raise_http_error("OpenRouter test failed", response.status_code, response.text)

    text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("OpenRouter test returned empty content.")
    parse_ai_response(text)
    return f"OpenRouter: {model} OK."


async def ping_openrouter_text_key(api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            OPENROUTER_CHAT_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": PING_TEST_PROMPT}],
                "temperature": 0,
                "max_tokens": 16,
            },
            headers=_headers(api_key),
        )

    if not response.is_success:
        detail = _extract_provider_error_message(response.text)
        raise ProviderError(detail[:220], response.status_code)

    return (response.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
