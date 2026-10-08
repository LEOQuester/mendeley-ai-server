import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.prompts import PING_EXPECTED_ANSWER
from app import chat_sessions, ref_doc, request_log, store
from app.auth import SESSION_COOKIE, create_session_token, require_admin, verify_password
from app.gemini_cache import delete_explicit_cache, refresh_gemini_cache_for_doc
from app.providers import gemini, groq
from app.ref_strategy import gemini_uses_explicit_cache, resolve_ref_inject

BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

app = FastAPI(title="Mendeley AI Server", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

cors_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if cors_origins.strip() == "*" else [item.strip() for item in cors_origins.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class AnalyzeTextLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = (request.url.path or "").rstrip("/") or "/"
        if request.method.upper() != "POST" or path.lower() != "/api/analyze/text":
            return await call_next(request)

        body_bytes = await request.body()

        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        replay_request = Request(request.scope, receive)
        response = await call_next(replay_request)

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            _log_analyze_text_http(body_bytes, response.status_code, response_body)
        except Exception:
            logger.exception("Failed to write analyze/text entry to request log")

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


app.add_middleware(AnalyzeTextLogMiddleware)


class TextAnalyzeRequest(BaseModel):
    provider: Literal["gemini", "groq"] = "gemini"
    question_text: str = Field(min_length=1)
    mode: Literal["auto", "mcq", "descriptive"] = "auto"
    model: str | None = None
    session_id: str | None = None


class VisionAnalyzeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    mime_type: str = "image/jpeg"
    model: str | None = None
    session_id: str | None = None


class SessionResetRequest(BaseModel):
    session_id: str | None = None


class TestKeyRequest(BaseModel):
    provider: Literal["gemini", "groq"]
    key: str = Field(min_length=1)
    model: str | None = None


class AdminTextPingRequest(BaseModel):
    provider: Literal["gemini", "groq"]
    model: str = Field(min_length=1)
    key_index: int = Field(default=0, ge=0)


def _normalize_ping_token(text: str) -> str:
    cleaned = (text or "").strip().strip("\"'`")
    if not cleaned:
        return ""
    return cleaned.split()[0].strip(".,;:!?")


def _resolve_text_mode(question_text: str, mode: str) -> str:
    if mode in {"mcq", "descriptive"}:
        return mode
    if re.search(r"\n\d+\.\s", question_text) or re.search(r"^Q:\s", question_text, re.MULTILINE):
        return "mcq"
    return "descriptive"


def _normalize_model(model_id: str | None, allowed: list[dict], fallback: str) -> str:
    ids = {model["id"] for model in allowed}
    if model_id and model_id in ids:
        return model_id
    return fallback


def _ref_mode(provider: str, model: str, ref_context: str) -> str:
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return "none"
    if provider == "gemini" and gemini_uses_explicit_cache(model):
        return "cache"
    if ref_context:
        return "excerpt"
    return "none"


async def _refresh_gemini_cache(config: dict, model: str) -> None:
    keys = store.ordered_provider_keys("gemini", config)
    if not keys:
        ref_doc.update_cache_meta(None, None, None, "none", "No Gemini API keys configured.")
        return
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return

    ref_text = ref_doc.full_ref_text_for_cache()
    if not ref_text:
        return

    try:
        result = await refresh_gemini_cache_for_doc(keys[0], model, ref_text)
        if result.cache:
            ref_doc.update_cache_meta(
                result.cache["name"],
                model,
                result.cache["expires_at"],
                result.mode,
                result.note,
            )
        else:
            ref_doc.update_cache_meta(None, None, None, result.mode, result.note)
    except Exception as exc:
        logger.warning("Gemini cache refresh skipped after ref-doc upload: %s", exc)
        ref_doc.update_cache_meta(
            None,
            None,
            None,
            "implicit",
            f"Cache refresh failed ({exc}). Using implicit stable-prefix mode.",
        )


@app.on_event("startup")
def startup() -> None:
    store.ensure_config()
    store.init_db()
    request_log.init_from_disk()


@app.on_event("shutdown")
async def shutdown() -> None:
    await gemini.close_gemini_http_client()


def _parse_analyze_request_body(body_bytes: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body_bytes.decode("utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return {}


def _log_analyze_text_http(body_bytes: bytes, status_code: int, response_body: bytes) -> None:
    req = _parse_analyze_request_body(body_bytes)
    question_text = str(req.get("question_text") or "").strip()
    if not question_text and body_bytes:
        question_text = body_bytes[:2000].decode("utf-8", errors="replace")
    provider = str(req.get("provider") or "—")
    model = str(req.get("model") or "—")
    mode = str(req.get("mode") or "auto")

    ok = 200 <= status_code < 300
    response_text = ""
    error: str | None = None

    if ok:
        try:
            result = json.loads(response_body.decode("utf-8"))
            if isinstance(result, dict):
                response_text = request_log.format_response_for_log(result)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_text = response_body[:2000].decode("utf-8", errors="replace")
    else:
        try:
            payload = json.loads(response_body.decode("utf-8"))
            if isinstance(payload, dict):
                detail = payload.get("detail")
                if isinstance(detail, list):
                    error = " ".join(
                        str(item.get("msg") or item.get("message") or item) for item in detail
                    )
                else:
                    error = str(detail or payload.get("message") or payload)
            else:
                error = str(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error = response_body[:500].decode("utf-8", errors="replace") or f"HTTP {status_code}"

    request_log.append_entry(
        kind="text",
        provider=provider,
        model=model,
        mode=mode,
        request_text=question_text or "(empty or invalid JSON body)",
        response_text=response_text,
        ok=ok,
        error=error if not ok else None,
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/admin", status_code=302)


@app.get("/health")
async def health() -> dict[str, Any]:
    from app import mysql_store

    payload: dict[str, Any] = {
        "status": "ok",
        "storage_backend": "mysql" if mysql_store.mysql_enabled() else "file",
        "mysql_configured": mysql_store.mysql_configured(),
        "analyze_log_entries": request_log.entry_count(),
        "analyze_log_latest_utc": request_log.latest_timestamp(),
        **request_log.debug_info(),
    }
    if mysql_store.mysql_enabled():
        try:
            payload["mysql_ok"] = mysql_store.ping()
        except Exception as exc:
            payload["mysql_ok"] = False
            payload["mysql_error"] = str(exc)[:200]
    return payload


@app.get("/api/config")
async def get_public_config() -> JSONResponse:
    return JSONResponse(store.public_config())


def _admin_session_valid(session: str | None) -> bool:
    if not session:
        return False
    try:
        from app.auth import _serializer

        _serializer().loads(session, max_age=60 * 60 * 12)
        return True
    except Exception:
        return False


@app.post("/api/analyze/text")
async def analyze_text(payload: TextAnalyzeRequest) -> JSONResponse:
    config = store.load_config()
    text_mode = _resolve_text_mode(payload.question_text, payload.mode)
    model = ""
    ref_context = ""

    try:
        if payload.provider == "groq":
            keys = store.ordered_provider_keys("groq", config)
            model = _normalize_model(
                payload.model,
                config.get("groq_text_models", []),
                config.get("defaults", {}).get("groq_text", "openai/gpt-oss-120b"),
            )
            if not keys:
                raise HTTPException(status_code=503, detail="No Groq API keys configured on the server.")
            session_id, session = chat_sessions.get_or_create_session(payload.session_id, model)
            ref_context = resolve_ref_inject("groq", payload.question_text, config)
            result = await groq.call_groq_text(
                keys, model, payload.question_text, text_mode, ref_context,
                session_id=session_id, session=session,
            )
        else:
            keys = store.ordered_provider_keys("gemini", config)
            model = _normalize_model(
                payload.model,
                config.get("gemini_text_models", []),
                config.get("defaults", {}).get("gemini_text", "gemini-3.1-flash-lite"),
            )
            if not keys:
                raise HTTPException(status_code=503, detail="No Gemini API keys configured on the server.")
            session_id, session = chat_sessions.get_or_create_session(payload.session_id, model)
            ref_context = resolve_ref_inject(
                "gemini", payload.question_text, config, gemini_model=model
            )
            result = await gemini.call_gemini_text(
                keys, model, payload.question_text, text_mode, ref_context,
                session_id=session_id, session=session,
            )

        return JSONResponse(
            {
                **result,
                "session_id": session_id,
                "ref_mode": _ref_mode(payload.provider, model, ref_context),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/analyze/vision")
async def analyze_vision(payload: VisionAnalyzeRequest) -> JSONResponse:
    config = store.load_config()
    keys = store.ordered_provider_keys("gemini", config)
    if not keys:
        raise HTTPException(status_code=503, detail="No Gemini API keys configured on the server.")

    model = _normalize_model(
        payload.model,
        config.get("gemini_vision_models", []),
        config.get("defaults", {}).get("gemini_vision", "gemini-3.7-flash"),
    )

    try:
        ref_context = resolve_ref_inject("gemini", "", config, vision=True)
        result = await gemini.call_gemini_vision(
            keys, model, payload.image_base64, payload.mime_type, ref_context,
        )
        return JSONResponse(
            {
                **result,
                "session_id": payload.session_id,
                "ref_mode": _ref_mode("gemini", model, ref_context),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/session/reset")
async def reset_chat_session(payload: SessionResetRequest) -> JSONResponse:
    if payload.session_id:
        chat_sessions.delete_session(payload.session_id)
    else:
        chat_sessions.clear_all_sessions()
    return JSONResponse({"ok": True})


@app.post("/api/test-key")
async def test_key(payload: TestKeyRequest) -> JSONResponse:
    config = store.load_config()
    try:
        if payload.provider == "gemini":
            text_model = _normalize_model(
                payload.model,
                config.get("gemini_text_models", []),
                config.get("defaults", {}).get("gemini_text", "gemini-3.1-flash-lite"),
            )
            vision_model = _normalize_model(
                payload.model,
                config.get("gemini_vision_models", []),
                config.get("defaults", {}).get("gemini_vision", "gemini-3.7-flash"),
            )
            text_result = await gemini.test_gemini_text_key(payload.key, text_model)
            vision_result = await gemini.test_gemini_vision_key(payload.key, vision_model)
            return JSONResponse({"ok": True, "message": f"{text_result} {vision_result}"})
        model = _normalize_model(
            payload.model,
            config.get("groq_text_models", []),
            config.get("defaults", {}).get("groq_text", "openai/gpt-oss-120b"),
        )
        message = await groq.test_groq_key(payload.key, model)
        return JSONResponse({"ok": True, "message": message})
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)


@app.post("/admin/api/text-ping")
async def admin_text_ping(
    payload: AdminTextPingRequest,
    _: Annotated[None, Depends(require_admin)],
) -> JSONResponse:
    """Text-only ping (same API path as extension text selection — not vision/screenshot)."""
    config = store.load_config()
    keys_field = "gemini_keys" if payload.provider == "gemini" else "groq_keys"
    model_field = "gemini_text_models" if payload.provider == "gemini" else "groq_text_models"
    default_model = config.get("defaults", {}).get(
        "gemini_text" if payload.provider == "gemini" else "groq_text",
        "gemini-3.1-flash-lite" if payload.provider == "gemini" else "openai/gpt-oss-120b",
    )

    keys = config.get(keys_field, [])
    if not keys:
        return JSONResponse(
            {"ok": False, "message": f"No {payload.provider} API keys configured."},
            status_code=400,
        )
    if payload.key_index >= len(keys):
        return JSONResponse({"ok": False, "message": "Invalid key index."}, status_code=400)

    model = _normalize_model(payload.model, config.get(model_field, []), default_model)
    api_key = keys[payload.key_index]
    key_preview = f"****{api_key[-4:]}" if len(api_key) >= 4 else "****"

    try:
        if payload.provider == "gemini":
            raw = await gemini.ping_gemini_text_key(api_key, model)
        else:
            raw = await groq.ping_groq_text_key(api_key, model)
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "provider": payload.provider,
                "model": model,
                "key_preview": key_preview,
                "expected": PING_EXPECTED_ANSWER,
                "message": str(exc),
            },
            status_code=400,
        )

    actual = _normalize_ping_token(raw)
    ok = actual == PING_EXPECTED_ANSWER
    return JSONResponse(
        {
            "ok": ok,
            "provider": payload.provider,
            "model": model,
            "key_preview": key_preview,
            "expected": PING_EXPECTED_ANSWER,
            "actual": raw,
            "actual_normalized": actual,
            "message": "Ping OK — model replied 200." if ok else f'Expected "{PING_EXPECTED_ANSWER}", got "{actual or raw}".',
        },
        status_code=200 if ok else 400,
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None):
    logged_in = _admin_session_valid(session)

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "logged_in": logged_in,
            "config": store.admin_config() if logged_in else None,
            "ping_models_json": json.dumps(
                {
                    "gemini": store.load_config().get("gemini_text_models", []) if logged_in else [],
                    "groq": store.load_config().get("groq_text_models", []) if logged_in else [],
                    "gemini_keys": store.admin_config().get("gemini_keys", []) if logged_in else [],
                    "groq_keys": store.admin_config().get("groq_keys", []) if logged_in else [],
                    "defaults": store.load_config().get("defaults", {}) if logged_in else {},
                }
            )
            if logged_in
            else "{}",
        },
    )


@app.get("/admin/request-log", response_class=HTMLResponse)
async def admin_request_log(
    request: Request,
    page: int = 1,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
):
    if not _admin_session_valid(session):
        return RedirectResponse(url="/admin", status_code=302)

    entries, total, total_pages, current_page = request_log.get_page(page, request_log.DEFAULT_PAGE_SIZE)
    return templates.TemplateResponse(
        "request_log.html",
        {
            "request": request,
            "entries": entries,
            "total": total,
            "page": current_page,
            "total_pages": total_pages,
            "page_size": request_log.DEFAULT_PAGE_SIZE,
            "log_debug": request_log.debug_info(),
        },
    )


@app.post("/admin/login")
async def admin_login(password: Annotated[str, Form()]):
    if not verify_password(password):
        raise HTTPException(status_code=401, detail="Invalid password.")
    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 12,
    )
    return response


@app.post("/admin/logout")
async def admin_logout():
    response = RedirectResponse(url="/admin", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.post("/admin/keys/add")
async def admin_add_key(
    provider: Annotated[str, Form()],
    key: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.add_key(provider, key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/keys/remove")
async def admin_remove_key(
    provider: Annotated[str, Form()],
    index: Annotated[int, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.remove_key(provider, index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/keys/premium")
async def admin_set_premium_key(
    provider: Annotated[str, Form()],
    key: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.set_premium_key(provider, key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/keys/premium/clear")
async def admin_clear_premium_key(
    provider: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.clear_premium_key(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/models/add")
async def admin_add_model(
    provider: Annotated[str, Form()],
    model_type: Annotated[str, Form()],
    model_id: Annotated[str, Form()],
    label: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.add_model(provider, model_type, model_id, label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/models/remove")
async def admin_remove_model(
    provider: Annotated[str, Form()],
    model_type: Annotated[str, Form()],
    model_id: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    try:
        store.remove_model(provider, model_type, model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ref-doc/upload")
async def admin_upload_ref_doc(
    file: UploadFile = File(...),
    _: Annotated[None, Depends(require_admin)] = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        raw = await file.read()
        meta = ref_doc.save_ref_doc(file.filename, raw, enabled=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Failed to write reference document to disk")
        raise HTTPException(
            status_code=500,
            detail="Server could not save the uploaded file. Check Railway disk/volume settings.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected ref-doc upload failure")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    try:
        chat_sessions.clear_all_sessions()
        config = store.load_config()
        model = config.get("defaults", {}).get("gemini_text", "gemini-3.1-flash-lite")
        await _refresh_gemini_cache(config, model)
    except Exception as exc:
        logger.warning("Ref doc saved but cache refresh failed: %s", exc)

    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ref-doc/toggle")
async def admin_toggle_ref_doc(
    enabled: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)] = None,
):
    try:
        ref_doc.set_enabled(enabled.lower() in {"1", "true", "on", "yes"})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ref-doc/remove")
async def admin_remove_ref_doc(_: Annotated[None, Depends(require_admin)] = None):
    config = store.load_config()
    meta = ref_doc.load_meta()
    keys = store.ordered_provider_keys("gemini", config)
    if meta and keys:
        await delete_explicit_cache(keys[0], meta.get("gemini_cache_name"))
    ref_doc.clear_ref_doc()
    chat_sessions.clear_all_sessions()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ref-doc/settings")
async def admin_ref_doc_settings(
    max_inject_tokens_gemini: Annotated[int, Form()],
    max_inject_tokens_groq: Annotated[int, Form()],
    max_inject_tokens_vision: Annotated[int, Form()],
    _: Annotated[None, Depends(require_admin)] = None,
):
    store.update_ref_doc_settings(
        {
            "max_inject_tokens_gemini": max_inject_tokens_gemini,
            "max_inject_tokens_groq": max_inject_tokens_groq,
            "max_inject_tokens_vision": max_inject_tokens_vision,
        }
    )
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/defaults")
async def admin_update_defaults(
    gemini_text: Annotated[str, Form()],
    gemini_vision: Annotated[str, Form()],
    groq_text: Annotated[str, Form()],
    _: Annotated[None, Depends(require_admin)],
):
    store.update_defaults(
        {
            "gemini_text": gemini_text,
            "gemini_vision": gemini_vision,
            "groq_text": groq_text,
        }
    )
    return RedirectResponse(url="/admin", status_code=303)
