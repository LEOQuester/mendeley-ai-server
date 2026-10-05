# Mendeley AI Server

FastAPI backend for the Web Importer for Mendeley extension. Hosts API keys, model lists, and AI proxy endpoints so the browser extension never stores provider secrets.

## Features

- Text and vision analysis endpoints for the extension
- API key pool with automatic rotation on rate limits
- **Reference document upload** with smart chunk retrieval (free-tier safe)
- Gemini implicit/explicit context caching when available
- Admin web UI at `/admin` for keys, models, ref docs, and defaults
- Railway-ready deployment

## Reference document architecture (free tier)

Free APIs have large *context windows* but small *tokens-per-minute* budgets. Sending a full PDF on every question will hit Groq/Gemini rate limits quickly.

This server uses a **retrieve-then-inject** pattern:

1. Upload `.txt`, `.md`, or `.pdf` in `/admin`
2. Server chunks the document locally (no embedding API cost)
3. For each question, keyword overlap picks the top 2–3 relevant chunks
4. Token estimates use **~1.35 tokens/word** (typical Gemini count). A char/4 upper bound is stored for comparison.
5. Per-question injection depends on cache mode:
   - **Explicit cache**: question only (full doc cached on upload)
   - **Implicit cache**: stable ~4.5K-token doc prefix + relevant excerpt (~5K total budget)
   - **Groq / vision**: excerpt only (~1.8K / ~1.5K tokens)

**Gemini caching strategy**

- **Explicit cache** (tried on upload): needs ≥4,096 tokens on Gemini 3 models. Free tier often returns `limit=0` → automatic fallback.
- **Implicit cache** (fallback): on 400/429/`INVALID_ARGUMENT`/free-tier block, server sends the same stable doc prefix (≥4,096 tokens) on every Gemini request so Google can cache it automatically ([docs](https://ai.google.dev/gemini-api/docs/caching)).
- Never fails the upload — always downgrades gracefully with a note shown in admin.

**Groq limits** (official): `openai/gpt-oss-120b` ≈ 30 RPM, 8K TPM, 200K TPD — keep injected context small ([Groq rate limits](https://console.groq.com/docs/rate-limits))

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `ADMIN_PASSWORD` in `.env`, then optionally seed keys:

```powershell
$env:GEMINI_API_KEYS="key1,key2"
$env:GROQ_API_KEYS="key1,key2"
python scripts/seed_config.py
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

- Health: `http://localhost:8000/health`
- Admin: `http://localhost:8000/admin`
- Public config: `http://localhost:8000/api/config`

In **Admin → Text API ping test**, pick provider, **text model**, and API key, then **Run text ping test**. This uses the same **text** API as the extension (**select text + H**, not screenshot). Success = model replies exactly `200`.

**Extension request log** (`/admin/request-log`, button on the admin home page): timestamped table of text sent to `POST /api/analyze/text` and the model response. Newest first, 10 entries per page.

## Railway deployment

1. Create a new Railway service from this folder/repo.
2. Set environment variables:
   - `ADMIN_PASSWORD` (required)
   - `GEMINI_API_KEYS` (comma-separated, optional for first boot)
   - `GROQ_API_KEYS` (comma-separated, optional for first boot)
   - `CORS_ORIGINS=*` or your extension origin
3. Deploy. Railway uses the `Procfile` start command.
4. Open `https://your-app.up.railway.app/admin` and add keys/models if needed.
5. Put the Railway URL into the extension settings as the API server URL.

## API

### `GET /api/config`

Returns model lists and defaults for the extension.

### `POST /api/analyze/text`

```json
{
  "provider": "gemini",
  "question_text": "Q: ...",
  "mode": "auto",
  "model": "gemini-3.1-pro-preview"
}
```

### `POST /api/analyze/vision`

```json
{
  "image_base64": "...",
  "mime_type": "image/jpeg",
  "model": "gemini-3.7-flash"
}
```
