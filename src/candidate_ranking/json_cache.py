from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def load_json_cache(cache_path: Path, cache_label: str) -> dict:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable %s %s: %s", cache_label, cache_path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_json_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(f"{cache_path.suffix}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(cache_path)
