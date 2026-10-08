from app import ref_doc
from app.gemini_cache import cache_is_valid, uses_implicit_stable_prefix


def build_ref_excerpt(question_text: str, max_tokens: int, top_k: int = 4) -> str:
    """Inject only the most relevant sections — not the full document."""
    return ref_doc.retrieve_excerpt(question_text, max_tokens=max_tokens, top_k=top_k)


def gemini_uses_explicit_cache(model: str | None) -> bool:
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return False
    return bool(model and cache_is_valid(meta, model))


def resolve_ref_inject(
    provider: str,
    question_text: str,
    config: dict,
    *,
    vision: bool = False,
    gemini_model: str | None = None,
) -> str:
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return ""

    # Screenshot vision has no question text — skip ref inject (faster, fewer tokens).
    if vision and not str(question_text or "").strip():
        return ""

    settings = config.get("ref_doc_settings") or {}
    if provider == "gemini" and not vision and gemini_model and gemini_uses_explicit_cache(gemini_model):
        return ""

    if vision:
        max_tokens = settings.get("max_inject_tokens_vision", 1500)
        top_k = settings.get("excerpt_top_k_vision", 3)
    elif provider == "groq":
        max_tokens = settings.get("max_inject_tokens_groq", 1800)
        top_k = settings.get("excerpt_top_k_groq", 3)
    else:
        max_tokens = settings.get("max_inject_tokens_gemini", 5000)
        top_k = settings.get("excerpt_top_k_gemini", 4)

    if provider == "gemini" and not vision and uses_implicit_stable_prefix(meta):
        stable_tokens = int(settings.get("implicit_stable_prefix_tokens", 4500))
        excerpt_tokens = max(800, max_tokens - stable_tokens)
        stable = ref_doc.stable_implicit_prefix(stable_tokens)
        excerpt = build_ref_excerpt(question_text, excerpt_tokens, top_k)
        if stable and excerpt:
            return f"{stable}\n\n{excerpt}"
        return stable or excerpt

    return build_ref_excerpt(question_text, max_tokens=max_tokens, top_k=top_k)
