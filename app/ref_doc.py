import json
import re
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = BASE_DIR / "data" / "ref_docs"
CHUNKS_PATH = REF_DIR / "chunks.json"
SOURCE_PATH = REF_DIR / "source.bin"
META_FILENAME = "meta.json"

CHUNK_SIZE_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

_lock = threading.Lock()

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "that", "this", "these", "those", "it", "its", "as", "if", "then", "than", "when",
    "where", "which", "who", "whom", "what", "how", "why", "not", "no", "yes", "all",
    "any", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "too", "very", "can", "just", "don", "now", "q",
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def _split_chunks(text: str) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + CHUNK_SIZE_CHARS)
        if end < len(cleaned):
            split_at = cleaned.rfind("\n\n", start, end)
            if split_at <= start + 400:
                split_at = cleaned.rfind(". ", start, end)
            if split_at > start + 400:
                end = split_at + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _extract_text(filename: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith((".txt", ".md", ".markdown", ".csv")):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages).strip()
            if text:
                return text
            raise ValueError("Could not extract text from PDF. Try a .txt or .md export.")
        except ImportError as exc:
            raise ValueError("PDF support requires pypdf. Upload .txt or .md instead.") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"PDF could not be read: {exc}") from exc

    raise ValueError("Unsupported file type. Upload .txt, .md, or .pdf.")


def _ensure_ref_dir() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)


def _load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_chunks(chunks: list[dict[str, Any]]) -> None:
    _ensure_ref_dir()
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2)


def _save_meta(meta: dict[str, Any]) -> None:
    _ensure_ref_dir()
    with (REF_DIR / META_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def load_meta() -> dict[str, Any] | None:
    path = REF_DIR / META_FILENAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clear_ref_doc() -> None:
    with _lock:
        for path in (CHUNKS_PATH, SOURCE_PATH, REF_DIR / META_FILENAME):
            if path.exists():
                path.unlink()


def save_ref_doc(filename: str, raw: bytes, enabled: bool = True) -> dict[str, Any]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    text = _extract_text(filename, raw)
    if len(text.strip()) < 80:
        raise ValueError("Reference document is too short or empty after extraction.")

    chunks = _split_chunks(text)
    if not chunks:
        raise ValueError("Could not split reference document into chunks.")

    chunk_records = [
        {
            "id": index,
            "text": chunk,
            "token_estimate": estimate_tokens(chunk),
        }
        for index, chunk in enumerate(chunks)
    ]

    with _lock:
        _ensure_ref_dir()
        SOURCE_PATH.write_bytes(raw)
        _save_chunks(chunk_records)
        meta = {
            "enabled": enabled,
            "original_name": filename,
            "stored_name": SOURCE_PATH.name,
            "char_count": len(text),
            "token_estimate": estimate_tokens(text),
            "chunk_count": len(chunk_records),
            "uploaded_at": datetime.now(UTC).isoformat(),
            "gemini_cache_name": None,
            "gemini_cache_model": None,
            "gemini_cache_expires_at": None,
            "gemini_cache_mode": "none",
        }
        _save_meta(meta)
        return meta


def set_enabled(enabled: bool) -> dict[str, Any]:
    meta = load_meta()
    if not meta:
        raise ValueError("No reference document uploaded.")
    meta["enabled"] = enabled
    _save_meta(meta)
    return meta


def update_cache_meta(cache_name: str | None, model: str | None, expires_at: str | None, mode: str) -> None:
    meta = load_meta()
    if not meta:
        return
    meta["gemini_cache_name"] = cache_name
    meta["gemini_cache_model"] = model
    meta["gemini_cache_expires_at"] = expires_at
    meta["gemini_cache_mode"] = mode
    _save_meta(meta)


def retrieve_excerpt(question_text: str, max_tokens: int, top_k: int = 3) -> str:
    meta = load_meta()
    if not meta or not meta.get("enabled"):
        return ""

    chunks = _load_chunks()
    if not chunks:
        return ""

    query_terms = _tokenize(question_text)
    if not query_terms:
        selected = chunks[: min(top_k, len(chunks))]
    else:
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in chunks:
            chunk_terms = _tokenize(chunk["text"])
            if not chunk_terms:
                continue
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            score = overlap / (len(query_terms) ** 0.5)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in scored[:top_k]] if scored else chunks[: min(top_k, len(chunks))]

    parts: list[str] = []
    used_tokens = 0
    header_tokens = 40
    budget = max(200, max_tokens - header_tokens)

    for chunk in selected:
        chunk_tokens = chunk.get("token_estimate") or estimate_tokens(chunk["text"])
        if used_tokens + chunk_tokens > budget and parts:
            break
        parts.append(chunk["text"])
        used_tokens += chunk_tokens
        if used_tokens >= budget:
            break

    if not parts and chunks:
        first = chunks[0]["text"]
        parts = [first[: budget * 4]]

    body = "\n\n---\n\n".join(parts)
    return (
        "REFERENCE DOCUMENT (read before answering; prefer facts from here when relevant):\n"
        f"{body}\n\n---\n"
    )


def admin_summary() -> dict[str, Any] | None:
    meta = load_meta()
    if not meta:
        return None
    return {
        "enabled": meta.get("enabled", False),
        "original_name": meta.get("original_name"),
        "char_count": meta.get("char_count", 0),
        "token_estimate": meta.get("token_estimate", 0),
        "chunk_count": meta.get("chunk_count", 0),
        "uploaded_at": meta.get("uploaded_at"),
        "gemini_cache_mode": meta.get("gemini_cache_mode", "none"),
        "gemini_cache_expires_at": meta.get("gemini_cache_expires_at"),
    }


def full_ref_text_for_cache() -> str:
    chunks = _load_chunks()
    if not chunks:
        return ""
    return "\n\n".join(chunk["text"] for chunk in chunks)
