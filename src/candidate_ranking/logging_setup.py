from __future__ import annotations

import logging
import sys


def configure_logging(console_level: int = logging.ERROR) -> None:
    app_logger = logging.getLogger("candidate_ranking")
    app_logger.setLevel(logging.WARNING)
    app_logger.propagate = False
    app_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    app_logger.addHandler(console_handler)
