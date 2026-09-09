import re
from typing import Any

import httpx

from app.prompts import API_TEST_PROMPT, SYSTEM_PROMPT_DESCRIPTIVE, SYSTEM_PROMPT_MCQ, TEXT_ONLY_PREAMBLE
from app.providers.key_rotation import retry_with_rotation_async
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


async def request_groq_text(
    api_key: str,
    model: str,
    question_text: str,
    text_mode: str,
    allow_retry: bool = True,
) -> dict[str, str]:
    system_prompt = SYSTEM_PROMPT_MCQ if text_mode == "mcq" else SYSTEM_PROMPT_DESCRIPTIVE
    max_tokens = 4096 if text_mode == "mcq" else 6144
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{TEXT_ONLY_PREAMBLE}{question_text}"},
        ],
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
        if allow_retry and (response.status_code == 400 or _is_image_refusal_text(detail)):
            return await request_groq_text(api_key, model, question_text, text_mode, allow_retry=False)
        _raise_http_error("Groq text error", response.status_code, response.text)

    data = response.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    if not text:
        raise ProviderError("Empty response from Groq")
    parsed = parse_ai_response(text)
    if allow_retry and _is_image_refusal_text(parsed.get("answer", "")):
        return await request_groq_text(api_key, model, question_text, text_mode, allow_retry=False)
    return parsed


async def call_groq_text(keys: list[str], model: str, question_text: str, text_mode: str) -> dict[str, str]:
    return await retry_with_rotation_async(
        keys,
        lambda key: request_groq_text(key, model, question_text, text_mode),
    )


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
