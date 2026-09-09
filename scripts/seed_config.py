"""One-time helper to seed local config.json with API keys from environment."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import store  # noqa: E402


def main() -> None:
    store.ensure_config()
    config = store.load_config()

    gemini_keys = [key.strip() for key in os.getenv("GEMINI_API_KEYS", "").split(",") if key.strip()]
    groq_keys = [key.strip() for key in os.getenv("GROQ_API_KEYS", "").split(",") if key.strip()]

    if gemini_keys:
        config["gemini_keys"] = gemini_keys
    if groq_keys:
        config["groq_keys"] = groq_keys

    store.save_config(config)
    print(f"Saved config to {store.CONFIG_PATH}")
    print(f"Gemini keys: {len(config.get('gemini_keys', []))}")
    print(f"Groq keys: {len(config.get('groq_keys', []))}")


if __name__ == "__main__":
    main()
