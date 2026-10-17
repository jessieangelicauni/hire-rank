from __future__ import annotations

import logging
import re
from typing import Callable

import numpy as np

from candidate_ranking.models import Candidate, JobProfile

logger = logging.getLogger(__name__)


_NEGATION_CUES = [
    "no explicit", "no specific", "no direct", "no mention", "no experience", "no evidence",
    "not specifically", "not explicitly", "not mentioned",
    "lacks", "lack of", "lacking", "insufficient", "limited", "without", "missing", "absent",
    "does not mention", "doesn't mention",
]
_NEGATION_CUE_WINDOW = 60
_CONTRAST_MARKERS = ["though", "but", "while", "despite", "however", "although", "only", "just"]
_CATEGORY_PRECEDED_BY = ["for "]
_CATEGORY_FOLLOWED_BY = ["-based", " tools", " platforms", " delivery", " systems", " solutions", " architecture"]


def find_negated_skill_mentions(text: str, skill_names: list[str]) -> list[str]:
    lowered = text.lower()
    mentioned: list[str] = []
    for skill in skill_names:
        match = re.search(rf"\b{re.escape(skill.lower())}\b", lowered)
        if match is None:
            continue
        index = match.start()
        skill_end = match.end()
        immediately_before = lowered[max(0, index - 10) : index]
        immediately_after = lowered[skill_end : skill_end + 15]
        if any(immediately_before.endswith(marker) for marker in _CATEGORY_PRECEDED_BY):
            continue
        if any(immediately_after.startswith(marker) for marker in _CATEGORY_FOLLOWED_BY):
            continue

        preceding_text = lowered[max(0, index - _NEGATION_CUE_WINDOW) : index]
        cue_end = max(
            (preceding_text.rfind(cue) + len(cue) for cue in _NEGATION_CUES if cue in preceding_text),
            default=-1,
        )
        if cue_end == -1:
            continue
        text_since_cue = preceding_text[cue_end:]
        if any(marker in text_since_cue for marker in _CONTRAST_MARKERS):
            continue
        mentioned.append(skill)
    return mentioned


def bridge_negated_jd_skills_to_candidate(
    weaknesses: list[str],
    candidate_skills: list[str],
    jd_technical_skills: list[str],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
) -> dict[str, str]:
    if not candidate_skills or not jd_technical_skills:
        return {}
    negated_jd_skills: set[str] = set()
    for weakness in weaknesses:
        negated_jd_skills.update(find_negated_skill_mentions(weakness, jd_technical_skills))
    if not negated_jd_skills:
        return {}

    negated_list = sorted(negated_jd_skills)
    combined_vectors = embedder(negated_list + candidate_skills)
    jd_vectors = combined_vectors[: len(negated_list)]
    candidate_vectors = combined_vectors[len(negated_list) :]
    similarities = jd_vectors @ candidate_vectors.T

    bridged: dict[str, str] = {}
    for i, jd_skill in enumerate(negated_list):
        best_index = int(similarities[i].argmax())
        if similarities[i][best_index] >= threshold:
            bridged[jd_skill] = candidate_skills[best_index]
    return bridged


def build_skill_embedder(model_name: str, device: str | None = None) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype("float32")

    return embed


def build_candidate_skill_index(
    candidates: list[Candidate], embedder: Callable[[list[str]], np.ndarray]
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    row_map: list[tuple[str, str]] = [
        (candidate.id, skill) for candidate in candidates for skill in candidate.skills
    ]
    if not row_map:
        raise ValueError("cannot build a skill index: no candidate has any extracted skills")

    vectors = embedder([skill for _candidate_id, skill in row_map])
    return vectors, row_map


def match_candidate_skills(
    profile: JobProfile,
    index: np.ndarray,
    row_map: list[tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
) -> dict[str, set[str]]:
    all_skills = profile.must_have_skills + profile.nice_to_have_skills
    if not all_skills:
        return {}

    deduplicated_skills: list[str] = []
    seen: set[str] = set()
    for skill in all_skills:
        normalized = skill.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated_skills.append(skill)

    query_vectors = embedder(deduplicated_skills)
    similarities = query_vectors @ index.T

    matched: dict[str, set[str]] = {}
    for row in range(len(deduplicated_skills)):
        for row_index in np.flatnonzero(similarities[row] >= threshold):
            candidate_id, _skill = row_map[row_index]
            matched.setdefault(candidate_id, set()).add(deduplicated_skills[row])
    return matched


def shortlist_candidates(
    jp: JobProfile,
    index: np.ndarray,
    row_map: list[tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
    min_matches: int = 2,
) -> list[str]:
    all_skills = jp.must_have_skills + jp.nice_to_have_skills
    if not all_skills:
        logger.warning("JP %s has no extracted skills; shortlist is empty", jp.id)
        return []

    matched = match_candidate_skills(jp, index, row_map, embedder, threshold)

    deduplicated_count = len({s.strip().lower() for s in all_skills})
    required = min(min_matches, deduplicated_count)

    must_have_normalized = {s.strip().lower() for s in jp.must_have_skills}

    def has_all_must_haves(skills: set[str]) -> bool:
        matched_lower = {skill.strip().lower() for skill in skills}
        return must_have_normalized <= matched_lower

    shortlisted = sorted(
        candidate_id
        for candidate_id, skills in matched.items()
        if len(skills) >= required and has_all_must_haves(skills)
    )
    if not shortlisted:
        logger.warning(
            "JP %s skill-match shortlist is empty at threshold=%s, required matches=%s, "
            "requiring all %d must-have skill(s)", jp.id, threshold, required, len(must_have_normalized),
        )
    return shortlisted
