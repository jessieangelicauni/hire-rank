from __future__ import annotations

from typing import Callable

import numpy as np

from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.skills import build_candidate_skill_index, shortlist_candidates

_JD_SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"]
_MUST_HAVE = ["Python", "SQL"]


def _one_hot_embedder(vocab: list[str]) -> Callable[[list[str]], np.ndarray]:
    """Deterministic stand-in for a real sentence-transformer embedder: an exact
    (case-insensitive) string match gets cosine similarity 1.0, anything else 0.0."""
    index = {v.strip().lower(): i for i, v in enumerate(vocab)}
    dim = len(index)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            key = text.strip().lower()
            if key in index:
                vectors[row, index[key]] = 1.0
        return vectors

    return embed


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _jd_skills(must_have: list[str]) -> JDSkills:
    return JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=_JD_SKILLS,
        must_have_skills=must_have,
    )


def _candidate(candidate_id: str, skills: list[str]) -> Candidate:
    return Candidate(
        id=candidate_id, source_path="cv.pdf", raw_text="...", num_pages=1, char_count=10,
        parse_status="ok", skills=skills,
    )


def test_shortlist_excludes_candidate_who_meets_total_but_not_must_have():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_excludes_candidate_who_meets_must_have_but_not_total():
    core_only = _candidate("core-only", ["Python", "SQL", "AWS"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([core_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_includes_candidate_who_meets_both():
    both = _candidate("both", ["Python", "SQL", "AWS", "Docker", "Kubernetes"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([both], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["both"]


def test_shortlist_empty_must_have_skills_preserves_prior_behavior():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills([]), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["peripheral-only"]
