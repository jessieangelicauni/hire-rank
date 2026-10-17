from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonFileHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._write()

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )
        self._write()

    def _write(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        tmp.replace(self._path)


def configure_logging(warnings_path: Path, console_level: int = logging.ERROR) -> JsonFileHandler:
    # Scoped to this package's own logger (not root) so warnings.json captures
    # only candidate_ranking's diagnostics, not third-party library chatter
    # (httpx, sentence-transformers, etc.) that would otherwise propagate up.
    app_logger = logging.getLogger("candidate_ranking")
    app_logger.setLevel(logging.WARNING)
    app_logger.propagate = False
    app_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    app_logger.addHandler(console_handler)

    json_handler = JsonFileHandler(warnings_path)
    json_handler.setLevel(logging.WARNING)
    app_logger.addHandler(json_handler)

    return json_handler
