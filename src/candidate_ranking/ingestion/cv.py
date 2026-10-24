from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from candidate_ranking.utils import load_json_cache, save_json_cache
from candidate_ranking.models import Candidate, slugify

logger = logging.getLogger(__name__)


def _candidate_from_cache(cached_entry: dict) -> Candidate | None:
    try:
        return Candidate.model_validate(cached_entry["candidate"])
    except (KeyError, ValidationError) as exc:
        logger.warning("Discarding invalid cache entry: %s", exc)
        return None


def _parse_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages_text = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    raw_text = re.sub(r"[ \t]{2,}", " ", "\n".join(pages_text)).strip()
    return raw_text, len(reader.pages)


def load_candidates(cv_dir: Path, cache_path: Path) -> list[Candidate]:
    cache = load_json_cache(cache_path, "CV cache")
    candidates: list[Candidate] = []
    cache_dirty = False

    for path in sorted(cv_dir.glob("*.pdf")):
        stat = path.stat()
        cache_key = path.name
        cached_entry = cache.get(cache_key)

        if (
            cached_entry is not None
            and cached_entry.get("mtime") == stat.st_mtime
            and cached_entry.get("size") == stat.st_size
        ):
            candidate_from_cache = _candidate_from_cache(cached_entry)
            if candidate_from_cache is not None:
                candidates.append(candidate_from_cache)
                continue

        try:
            raw_text, _num_pages = _parse_pdf(path)
            candidate = Candidate(
                id=slugify(path.stem),
                raw_text=raw_text,
                parse_status="ok",
            )
        except Exception as exc:
            logger.warning("Failed to parse CV %s: %s", path, exc)
            candidate = Candidate(
                id=slugify(path.stem),
                raw_text="",
                parse_status="failed",
            )

        candidates.append(candidate)
        if candidate.parse_status == "ok":
            cache[cache_key] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "candidate": candidate.model_dump(),
            }
            cache_dirty = True

    if cache_dirty:
        save_json_cache(cache_path, cache)

    return candidates
