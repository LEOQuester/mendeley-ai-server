import json
import re
from typing import Any

import httpx

from app import chat_sessions
from app.prompts import API_TEST_PROMPT, PING_TEST_PROMPT, SYSTEM_PROMPT_DESCRIPTIVE, SYSTEM_PROMPT_MCQ, TEXT_ONLY_PREAMBLE
from app.providers.key_rotation import is_rotatable_provider_error, retry_with_rotation_async
from app.providers.response_parser import parse_ai_response


class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


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


def _is_image_refusal_text(text: str) -> bool:
    msg = text.lower()
    return bool(
        re.search(
            r"cannot process image|can't process image|does not support image|unable to (see|view|process|analyze)",
            msg,
        )
    )


def _compose_groq_user_content(question_text: str, ref_excerpt: str) -> str:
    if ref_excerpt:
        return f"{ref_excerpt}\n\n{TEXT_ONLY_PREAMBLE}{question_text}"
    return f"{TEXT_ONLY_PREAMBLE}{question_text}"


async def request_groq_text(
    api_key: str,
    model: str,
    question_text: str,
    text_mode: str,
    allow_retry: bool = True,
    ref_excerpt: str = "",
    session: dict[str, Any] | None = None,
) -> dict[str, str]:
    system_prompt = SYSTEM_PROMPT_MCQ if text_mode == "mcq" else SYSTEM_PROMPT_DESCRIPTIVE
    max_tokens = 4096 if text_mode == "mcq" else 6144
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if session:
        messages.extend(chat_sessions.groq_history_messages(session))
    messages.append({"role": "user", "content": _compose_groq_user_content(question_text, ref_excerpt)})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    if allow_retry and model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "high"
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )

    if not response.is_success:
        detail = _extract_provider_error_message(response.text)
        status = response.status_code
        if is_rotatable_provider_error(detail, status):
            _raise_http_error("Groq text error", status, response.text)
        if allow_retry and (status == 400 or _is_image_refusal_text(detail)):
            return await request_groq_text(
                api_key, model, question_text, text_mode, allow_retry=False, ref_excerpt=ref_excerpt, session=session
            )
        _raise_http_error("Groq text error", status, response.text)

    data = response.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Empty response from Groq")
    parsed = parse_ai_response(text, default_type=text_mode)
    if allow_retry and _is_image_refusal_text(parsed.get("answer", "")):
        return await request_groq_text(
            api_key, model, question_text, text_mode, allow_retry=False, ref_excerpt=ref_excerpt, session=session
        )
    return parsed


async def call_groq_text(
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
        lambda key: request_groq_text(key, model, question_text, text_mode, ref_excerpt=ref_excerpt, session=session),
        provider_label="Groq",
    )
    if session_id:
        chat_sessions.append_turn(session_id, question_text, json.dumps(result))
    return result


async def test_groq_key(api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a test endpoint. Reply with valid JSON only, no markdown."},
                    {"role": "user", "content": API_TEST_PROMPT},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": 64,
            },
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )

    if not response.is_success:
        if response.status_code == 429:
            raise ProviderError("Groq rate limit hit. Wait a minute or try another model.")
        _raise_http_error("Groq test failed", response.status_code, response.text)

    text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Groq returned an empty response.")
    parse_ai_response(text)
    return f"Groq key works with {model}."


async def ping_groq_text_key(api_key: str, model: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": PING_TEST_PROMPT,
                    }
                ],
                "temperature": 0,
                "max_tokens": 8,
            },
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )

    if not response.is_success:
        if response.status_code == 429:
            raise ProviderError("Groq rate limit hit. Wait a minute or try another model.")
        _raise_http_error("Groq ping failed", response.status_code, response.text)

    text = (response.json().get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Groq returned an empty response.")
    return text.strip()
