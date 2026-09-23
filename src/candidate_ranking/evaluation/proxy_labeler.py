from __future__ import annotations

from pathlib import Path

from candidate_ranking.evaluation.criteria_optimization import (
    REQUIREMENT_LEVEL_COUNT,
    RequirementTriple,
    triple_key,
)
from candidate_ranking.json_cache import load_json_cache, save_json_cache
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA


class ProxyLabelClient:
    def __init__(self, anthropic_client, cache_path: Path, model: str) -> None:
        self._client = anthropic_client
        self._cache_path = cache_path
        self._model = model
        self._cache = load_json_cache(cache_path, "proxy label cache")

    def label(self, triple: RequirementTriple, jd: JobDescription, candidate: Candidate) -> int:
        key = triple_key(triple)
        if key in self._cache:
            return self._cache[key]
        level = self._call_model(triple, jd, candidate)
        self._cache[key] = level
        save_json_cache(self._cache_path, self._cache)
        return level

    def _call_model(self, triple: RequirementTriple, jd: JobDescription, candidate: Candidate) -> int:
        rubric = "\n".join(f"{i}: {desc}" for i, desc in enumerate(_REQUIREMENT_FIT_CRITERIA))
        prompt = (
            f"Job Title: {jd.title}\n\nJob Description:\n{jd.raw_text}\n\n"
            f"Candidate CV:\n{candidate.raw_text}\n\n"
            f"Requirement: '{triple.requirement}'\n\n"
            "Rate how well the candidate's CV supports this requirement, using exactly one of these "
            f"{REQUIREMENT_LEVEL_COUNT} levels (reply with only the number, nothing else):\n{rubric}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        try:
            level = int(text[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"proxy label response not parseable as 0-4: {text!r}") from exc
        if not (0 <= level <= REQUIREMENT_LEVEL_COUNT - 1):
            raise ValueError(f"proxy label out of range: {level}")
        return level
