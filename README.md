# Mendeley AI Server

FastAPI backend for the Mendeley Web Importer extension. Hosts API keys, model lists, and AI proxy endpoints so the browser extension never stores provider secrets.

## Features

- Text and vision analysis endpoints for the extension
- API key pool with automatic rotation on rate limits
- Admin web UI at `/admin` for keys, models, and defaults
- Railway-ready deployment

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
