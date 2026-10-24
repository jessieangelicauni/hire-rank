from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from candidate_ranking.models import Candidate, JobProfile

# Cosine-similarity floor for the embedding fallback below. Calibrate against
# real resume/job-skill pairs, not by assumption: a false negative here means
# a candidate never reaches Jev at all, which is worse than some over-inclusion.
DEFAULT_SIMILARITY_THRESHOLD = 0.55
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Skills the embedding match alone misses too often (e.g. "Python" vs "Built
# web apps with Django" scores well under threshold despite Django implying
# Python) -- kept as a cheap literal-match supplement, mirroring the same
# aliases used in assessment.py's evidence-strength rubric.
IMPLIED_SKILLS: dict[str, str] = {
    "django": "python",
    "spring boot": "java",
    "react": "javascript",
    "kubernetes": "docker",
}

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}
_jp_skill_vectors: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
_candidate_sentence_vectors: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def _word_pattern(term: str) -> re.Pattern[str]:
    pattern = _PATTERN_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        _PATTERN_CACHE[term] = pattern
    return pattern


_SENTENCE_SPLIT = re.compile(r"(?<=[.!])\s+|[\n|]+")


def _split_sentences(text: str) -> list[str]:
    """Split on '.'/'!' only when followed by whitespace, so a dotted
    identifier with no internal space (Node.js, ASP.NET, a version number
    like 3.5) stays in one piece instead of being torn in two. Each piece's
    own trailing '.'/'!' is then stripped, same as when it was the delimiter."""
    return [
        s for s in (p.strip().rstrip(".!") for p in _SENTENCE_SPLIT.split(text)) if len(s) > 15
    ]


def _literal_or_implied_match(text: str, skill: str) -> bool:
    if _word_pattern(skill).search(text):
        return True
    implying_terms = [alias for alias, implied in IMPLIED_SKILLS.items() if implied == skill.lower()]
    return any(_word_pattern(term).search(text) for term in implying_terms)


def _skill_vectors(jp: JobProfile) -> np.ndarray:
    key = (jp.id, tuple(jp.technical_skills))
    cached = _jp_skill_vectors.get(key)
    if cached is not None:
        return cached
    vectors = (
        _model().encode(jp.technical_skills, normalize_embeddings=True)
        if jp.technical_skills else np.zeros((0, 1), dtype="float32")
    )
    _jp_skill_vectors[key] = vectors
    return vectors


def _candidate_vectors(candidate: Candidate) -> tuple[list[str], np.ndarray]:
    key = (candidate.id, candidate.raw_text)
    cached = _candidate_sentence_vectors.get(key)
    if cached is not None:
        return cached
    sentences = _split_sentences(candidate.raw_text)
    vectors = (
        _model().encode(sentences, normalize_embeddings=True)
        if sentences else np.zeros((0, 1), dtype="float32")
    )
    _candidate_sentence_vectors[key] = (sentences, vectors)
    return sentences, vectors


def matching_skill_count(
    candidate: Candidate, jp: JobProfile, threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> int:
    """How many of the job's `technical_skills` this candidate's CV demonstrates:
    a literal mention, a known implied-skill alias, or a CV sentence whose
    embedding clears `threshold` cosine similarity against the skill's own
    embedding. Pure local computation, no API call -- this only decides
    whether a candidate reaches Jev at all."""
    if not jp.technical_skills:
        return 0
    text = candidate.raw_text
    sentences, candidate_vecs = _candidate_vectors(candidate)
    similarities = _skill_vectors(jp) @ candidate_vecs.T if sentences else None

    count = 0
    for i, skill in enumerate(jp.technical_skills):
        if _literal_or_implied_match(text, skill):
            count += 1
        elif similarities is not None and similarities[i].max() >= threshold:
            count += 1
    return count


def passes_shortlist_gate(
    candidate: Candidate, jp: JobProfile, minimum: int, threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """True if `candidate` should be assessed at all for `jp`. `minimum` is
    capped at the job's own technical_skills count, so a job with fewer
    skills than `minimum` can't silently reject every candidate."""
    if minimum <= 0:
        return True
    effective_minimum = min(minimum, len(jp.technical_skills))
    return matching_skill_count(candidate, jp, threshold) >= effective_minimum
