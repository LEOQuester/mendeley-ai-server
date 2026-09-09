import json
import re
from typing import Any

from app.prompts import MCQ_ANSWER_JOIN


def extract_json_text(raw: str) -> str:
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r'([{,]\s*)([a-zA-Z0-9_$]+)\s*:', r'\1"\2":', cleaned)
    cleaned = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', cleaned)
    return cleaned


def clean_answer_text(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    text = re.sub(
        r'^\{\s*"type"\s*:\s*"(?:mcq|descriptive)"\s*,\s*"answer"\s*:\s*"',
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^type\s*:\s*(?:mcq|descriptive)?,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'[{}"\\]', "", text)
    text = re.sub(r"^(?:mcq|descriptive)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    if text.lower() in {"mcq", "descriptive"}:
        return ""
    return text


def unwrap_answer_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [unwrap_answer_value(item) for item in value]
        return MCQ_ANSWER_JOIN.join(part for part in parts if part)
    if not isinstance(value, str):
        return str(value).strip()

    current = value.strip()
    for _ in range(3):
        if not current.startswith("{") and not current.startswith("["):
            break
        try:
            inner = json.loads(extract_json_text(current))
        except json.JSONDecodeError:
            break
        if isinstance(inner, list):
            current = unwrap_answer_value(inner)
            continue
        if isinstance(inner, dict):
            answers = inner.get("answers") or inner.get("Answers") or inner.get("choices")
            if isinstance(answers, list) and answers:
                current = unwrap_answer_value(answers)
                continue
            if inner.get("answer") is not None:
                current = unwrap_answer_value(inner["answer"])
                continue
        break

    cleaned = clean_answer_text(current)
    return cleaned or current


def answer_from_parsed_object(parsed: dict[str, Any]) -> str:
    answers = parsed.get("answers") or parsed.get("Answers") or parsed.get("choices") or parsed.get("correct_options")
    if isinstance(answers, list) and answers:
        return unwrap_answer_value(answers)
    raw = (
        parsed.get("answer")
        or parsed.get("Answer")
        or parsed.get("solution")
        or parsed.get("choice")
        or parsed.get("text")
        or parsed.get("result")
    )
    if raw in (None, ""):
        return ""
    return unwrap_answer_value(raw)


def parse_ai_response(raw: str) -> dict[str, str]:
    if not raw:
        return {"type": "mcq", "answer": "No response text received."}

    try:
        parsed = json.loads(extract_json_text(raw))
        if isinstance(parsed, dict):
            answer = answer_from_parsed_object(parsed)
            if answer:
                type_val = str(parsed.get("type") or parsed.get("Type") or "mcq").lower()
                return {
                    "type": "descriptive" if "descript" in type_val else "mcq",
                    "answer": answer,
                }
    except json.JSONDecodeError:
        pass

    answers_match = re.search(r'"answers"\s*:\s*\[([\s\S]*?)\]', raw, re.IGNORECASE)
    if answers_match:
        items = [
            match.group(1).replace('\\"', '"').strip()
            for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', answers_match.group(1))
        ]
        items = [item for item in items if item]
        if items:
            return {"type": "mcq", "answer": unwrap_answer_value(items)}

    answer_match = re.search(
        r'"(?:answer|Answer|choice|solution|text|result)"\s*:\s*"((?:[^"\\]|\\.)*)"',
        raw,
        re.IGNORECASE,
    )
    if answer_match:
        cleaned = unwrap_answer_value(answer_match.group(1))
        if cleaned:
            return {
                "type": "descriptive" if "descript" in raw.lower() else "mcq",
                "answer": cleaned,
            }

    clean_text = unwrap_answer_value(re.sub(r"```[\s\S]*?```", "", raw))
    if clean_text and not re.match(r'^[\s{["]*type\b', clean_text, re.IGNORECASE):
        return {"type": "mcq", "answer": clean_text[:500]}

    raise ValueError("Could not parse answer from response.")
