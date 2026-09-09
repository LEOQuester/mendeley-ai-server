# Mendeley AI Server

FastAPI backend for the Mendeley Web Importer extension. Hosts API keys, model lists, and AI proxy endpoints so the browser extension never stores provider secrets.

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
4. Only those chunks are injected, within per-provider token budgets:
   - **Gemini**: ~3500 tokens (tuned for implicit cache prefix ≥4096 when stable)
   - **Groq**: ~1500 tokens (fits inside ~8K TPM on `openai/gpt-oss-120b`)
   - **Vision**: ~1200 tokens in the system prompt

**Gemini caching strategy**

- **Implicit cache** (default): same reference prefix at the start of prompts; automatic on Gemini 2.5+/3.x when prefix ≥4096 tokens and requests are close together ([docs](https://ai.google.dev/gemini-api/docs/caching))
- **Explicit cache** (best-effort on upload): tries `cachedContents.create`; often blocked on free tier (`limit=0`) and falls back to implicit/stable-prefix mode

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
