import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    override = os.getenv("MENDELEY_DATA_DIR", "").strip()
    if override:
        return Path(override)
    return _PKG_ROOT / "data"
