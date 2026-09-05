"""Re-verifies the weakness self-correction retry loop's real numbers
against the live model, for run 20260831-010721.

Regenerates assessments for all 348 shortlisted (job, candidate) pairs by
calling generate_assessment directly (bypassing the production assessment
cache), using an on_attempt hook to observe what the original run's logging
never captured: how many pairs had a weakness contradiction on the first
attempt, how many were fixed by the single retry, and how many still
contradicted afterward (and were dropped).

Because every ChatOllama call in this codebase uses temperature=0, this is
expected to be a deterministic replay of what already happened during the
original run -- it should reproduce the 26-pair drop count already visible
in runs/20260831-010721/warnings.json, while also surfacing the
previously-unlogged first-attempt and fixed-by-retry counts.

Does not touch runs/_cache/assessments/ or any existing run output; writes
only a new runs/20260831-010721/weakness_retry_audit_report.json.

Run with: uv run python scripts/audit_weakness_contradictions.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptEvent:
    jd_id: str
    candidate_id: str
    attempt: int
    contradicted_skills: list[str]
    weaknesses: list[str]


def classify_pair(events: list[AttemptEvent]) -> str:
    first = events[0]
    if not first.contradicted_skills:
        return "clean"
    if len(events) == 1:
        return "dropped"
    return "dropped" if events[1].contradicted_skills else "fixed_by_retry"


def build_report(
    classifications: dict[tuple[str, str], str],
    total_weaknesses_checked: int,
    items_dropped: int,
    offline_residual_contradictions: int,
    offline_denominator: int,
) -> dict:
    total_pairs = len(classifications)
    pairs_with_initial_contradiction = sum(1 for c in classifications.values() if c != "clean")
    pairs_fixed_by_retry = sum(1 for c in classifications.values() if c == "fixed_by_retry")
    pairs_dropped = sum(1 for c in classifications.values() if c == "dropped")
    return {
        "total_pairs": total_pairs,
        "total_weaknesses_checked": total_weaknesses_checked,
        "pairs_with_initial_contradiction": pairs_with_initial_contradiction,
        "pairs_fixed_by_retry": pairs_fixed_by_retry,
        "pairs_dropped": pairs_dropped,
        "items_dropped": items_dropped,
        "offline_audit_residual_contradictions": offline_residual_contradictions,
        "offline_audit_denominator": offline_denominator,
        "offline_audit_rate": offline_residual_contradictions / offline_denominator,
    }
