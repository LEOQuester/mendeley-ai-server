import os
import re
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app import ref_doc, store
from app.auth import SESSION_COOKIE, create_session_token, require_admin, verify_password
from app.gemini_cache import delete_explicit_cache, try_create_explicit_cache
from app.prompts import SYSTEM_PROMPT_MCQ
from app.providers import gemini, groq

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Mendeley AI Server", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

cors_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if cors_origins.strip() == "*" else [item.strip() for item in cors_origins.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class TextAnalyzeRequest(BaseModel):
    provider: Literal["gemini", "groq"] = "gemini"
    question_text: str = Field(min_length=1)
    mode: Literal["auto", "mcq", "descriptive"] = "auto"
    model: str | None = None


class VisionAnalyzeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    mime_type: str = "image/jpeg"
    model: str | None = None


class TestKeyRequest(BaseModel):
    provider: Literal["gemini", "groq"]
    key: str = Field(min_length=1)
    model: str | None = None


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


def _ref_settings(config: dict) -> dict:
    return config.get("ref_doc_settings") or {
        "max_inject_tokens_gemini": 3500,
        "max_inject_tokens_groq": 1500,
        "max_inject_tokens_vision": 1200,
    }


def _build_ref_excerpt(provider: str, question_text: str, config: dict, *, vision: bool = False) -> str:
    settings = _ref_settings(config)
    if vision:
        max_tokens = settings.get("max_inject_tokens_vision", 1200)
    elif provider == "groq":
        max_tokens = settings.get("max_inject_tokens_groq", 1500)
    else:
        max_tokens = settings.get("max_inject_tokens_gemini", 3500)
    return ref_doc.retrieve_excerpt(question_text, max_tokens=max_tokens)


async def _refresh_gemini_cache(config: dict, model: str) -> None:
    keys = config.get("gemini_keys", [])
    if not keys:
        return
    meta = ref_doc.load_meta()
    if not meta or not meta.get("enabled"):
        return

    ref_text = ref_doc.full_ref_text_for_cache()
    if not ref_text:
        return

    old_cache = meta.get("gemini_cache_name")
    if old_cache:
        await delete_explicit_cache(keys[0], old_cache)

    created = await try_create_explicit_cache(keys[0], model, SYSTEM_PROMPT_MCQ, ref_text)
    if created:
        ref_doc.update_cache_meta(created["name"], model, created["expires_at"], "explicit")
    else:
        ref_doc.update_cache_meta(None, None, None, "implicit")


@app.on_event("startup")
def startup() -> None:
    store.ensure_config()


@app.get("/")
async def root():
    return RedirectResponse(url="/admin", status_code=302)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def get_public_config() -> JSONResponse:
    return JSONResponse(store.public_config())


@app.post("/api/analyze/text")
async def analyze_text(payload: TextAnalyzeRequest) -> JSONResponse:
    config = store.load_config()
    text_mode = _resolve_text_mode(payload.question_text, payload.mode)

    try:
        if payload.provider == "groq":
            keys = config.get("groq_keys", [])
            if not keys:
                raise HTTPException(status_code=503, detail="No Groq API keys configured on the server.")
            model = _normalize_model(
                payload.model,
                config.get("groq_text_models", []),
                config.get("defaults", {}).get("groq_text", "openai/gpt-oss-120b"),
            )
            ref_excerpt = _build_ref_excerpt("groq", payload.question_text, config)
            result = await groq.call_groq_text(keys, model, payload.question_text, text_mode, ref_excerpt)
        else:
            keys = config.get("gemini_keys", [])
            if not keys:
                raise HTTPException(status_code=503, detail="No Gemini API keys configured on the server.")
            model = _normalize_model(
                payload.model,
                config.get("gemini_text_models", []),
                config.get("defaults", {}).get("gemini_text", "gemini-3.1-pro-preview"),
            )
            ref_excerpt = _build_ref_excerpt("gemini", payload.question_text, config)
            result = await gemini.call_gemini_text(keys, model, payload.question_text, text_mode, ref_excerpt)

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/analyze/vision")
async def analyze_vision(payload: VisionAnalyzeRequest) -> JSONResponse:
    config = store.load_config()
    keys = config.get("gemini_keys", [])
    if not keys:
        raise HTTPException(status_code=503, detail="No Gemini API keys configured on the server.")

    model = _normalize_model(
        payload.model,
        config.get("gemini_vision_models", []),
        config.get("defaults", {}).get("gemini_vision", "gemini-3.7-flash"),
    )

    try:
        ref_excerpt = _build_ref_excerpt("gemini", "", config, vision=True)
        result = await gemini.call_gemini_vision(
            keys, model, payload.image_base64, payload.mime_type, ref_excerpt
        )
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/test-key")
async def test_key(payload: TestKeyRequest) -> JSONResponse:
    config = store.load_config()
    try:
        if payload.provider == "gemini":
            text_model = _normalize_model(
                payload.model,
                config.get("gemini_text_models", []),
                config.get("defaults", {}).get("gemini_text", "gemini-3.1-pro-preview"),
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


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None):
    logged_in = False
    if session:
        try:
            from app.auth import _serializer

            _serializer().loads(session, max_age=60 * 60 * 12)
            logged_in = True
        except Exception:
            logged_in = False

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "logged_in": logged_in,
            "config": store.admin_config() if logged_in else None,
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
    raw = await file.read()
    try:
        meta = ref_doc.save_ref_doc(file.filename, raw, enabled=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config = store.load_config()
    model = config.get("defaults", {}).get("gemini_text", "gemini-3.1-pro-preview")
    await _refresh_gemini_cache(config, model)
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
    keys = config.get("gemini_keys", [])
    if meta and keys:
        await delete_explicit_cache(keys[0], meta.get("gemini_cache_name"))
    ref_doc.clear_ref_doc()
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
