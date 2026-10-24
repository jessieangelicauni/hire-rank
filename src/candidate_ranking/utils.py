from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class OllamaLatencyCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._t0 = time.perf_counter()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        latency_ms = (time.perf_counter() - self._t0) * 1000
        logger.debug("Ollama latency: %.0f ms", latency_ms)

T = TypeVar("T")


class GenerationError(Exception):
    pass


def invoke_and_validate(
    chain: Runnable,
    payload: dict[str, Any],
    expected_type: type[T],
    validate: Callable[[T], str | None],
    log_context: str,
) -> T:
    try:
        result = chain.invoke(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s failed: %s", log_context, exc)
        raise GenerationError(f"{log_context} failed: {exc}") from exc

    if not isinstance(result, expected_type):
        error = ValueError(f"chain returned {type(result).__name__}, expected {expected_type.__name__}")
        logger.warning("%s returned unexpected type %s", log_context, type(result).__name__)
        raise GenerationError(f"{log_context} failed: {error}") from error

    validation_error = validate(result)
    if validation_error is not None:
        logger.warning("%s failed validation: %s", log_context, validation_error)
        raise GenerationError(f"{log_context} failed: {validation_error}")

    return result


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
