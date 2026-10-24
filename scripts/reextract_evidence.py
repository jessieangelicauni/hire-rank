"""Re-extract the `evidence` text snippet for already-generated assessments,
without calling Jev again.

`evidence_strength` and `probabilities` are Jev's judgment and are left
untouched; only the locally-derived evidence snippet is recomputed with the
current `_find_evidence_sentence` logic (literal/alias term match, BM25-
ranked among multiple hits, else cosine-similarity fallback against the
candidate's own CV sentences -- the same matching the shortlist gate already
uses, never an LLM's own attribution of it). Useful after a fix to evidence
extraction, so already-cached assessments don't carry a stale or
never-filled snippet.

Usage:
    .venv/bin/python scripts/reextract_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.scoring.assessment import _find_evidence_sentence


def _reextract(evaluations: list[dict], candidate) -> int:
    changed = 0
    for ev in evaluations:
        new_evidence = (
            _find_evidence_sentence(candidate, ev["requirement"]) or "" if ev["evidence_strength"] != "none" else ""
        )
        if new_evidence != ev.get("evidence", ""):
            ev["evidence"] = new_evidence
            changed += 1
    return changed


def main() -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}

    total_files = 0
    total_changed = 0

    cache_assessments_dir = cfg.cache_dir / "assessments"
    if cache_assessments_dir.exists():
        for path in cache_assessments_dir.glob("*/*.json"):
            cv_id = path.stem
            candidate = candidates_by_id.get(cv_id)
            if candidate is None:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            changed = _reextract(data["assessment"]["evaluations"], candidate)
            if changed:
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                total_changed += changed
            total_files += 1

    for path in cfg.runs_dir.glob("*/*/assessments.json"):
        if path.is_relative_to(cfg.cache_dir):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        file_changed = 0
        for cv_id, assessment in data.items():
            candidate = candidates_by_id.get(cv_id)
            if candidate is None:
                continue
            file_changed += _reextract(assessment["evaluations"], candidate)
        if file_changed:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            total_changed += file_changed
        total_files += 1

    print(f"Checked {total_files} assessment file(s); re-extracted {total_changed} evidence field(s).")


if __name__ == "__main__":
    main()
