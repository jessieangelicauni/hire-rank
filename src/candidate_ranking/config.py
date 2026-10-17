from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RunConfig:
    preset: str
    jp_dir: Path
    cv_dir: Path
    cache_dir: Path
    runs_dir: Path
    max_jps: int | None
    max_candidates: int | None
    ollama_model: str
    ollama_base_url: str
    ollama_num_parallel: int
    skill_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_num_ctx: int = 16384
    ranking_pl_prior_variance: float = 1.0
    ranking_min_delta_u: float = 0.01
    jev_api_key: str | None = None
    ranking_pairwise_max_iterations: int = 80

    @classmethod
    def full(cls, project_root: Path) -> "RunConfig":
        return cls(
            preset="full",
            jp_dir=project_root / "job-profile",
            cv_dir=project_root / "cv",
            cache_dir=project_root / "runs" / "_cache",
            runs_dir=project_root / "runs",
            max_jps=None,
            max_candidates=None,
            ollama_model="qwen3:14b",
            ollama_base_url="http://localhost:11434",
            ollama_num_parallel=4,
            skill_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )


_ENV_OVERRIDES: dict[str, tuple[str, Callable[[str], object]]] = {
    "CANDIDATE_RANKING_JP_DIR": ("jp_dir", Path),
    "CANDIDATE_RANKING_CV_DIR": ("cv_dir", Path),
    "CANDIDATE_RANKING_CACHE_DIR": ("cache_dir", Path),
    "CANDIDATE_RANKING_RUNS_DIR": ("runs_dir", Path),
    "CANDIDATE_RANKING_MAX_JDS": ("max_jds", int),
    "CANDIDATE_RANKING_MAX_CANDIDATES": ("max_candidates", int),
    "CANDIDATE_RANKING_MODEL": ("ollama_model", str),
    "CANDIDATE_RANKING_OLLAMA_BASE_URL": ("ollama_base_url", str),
    "CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL": ("ollama_num_parallel", int),
    "CANDIDATE_RANKING_OLLAMA_NUM_CTX": ("ollama_num_ctx", int),
    "CANDIDATE_RANKING_SKILL_EMBEDDING_MODEL": ("skill_embedding_model", str),
    "CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE": ("ranking_pl_prior_variance", float),
    "CANDIDATE_RANKING_MIN_DELTA_U": ("ranking_min_delta_u", float),
    "CANDIDATE_RANKING_JEV_API_KEY": ("jev_api_key", str),
    "CANDIDATE_RANKING_PAIRWISE_MAX_ITERATIONS": ("ranking_pairwise_max_iterations", int),
}

ENV_OVERRIDE_VARS: tuple[str, ...] = tuple(_ENV_OVERRIDES)


def apply_env_overrides(cfg: RunConfig) -> RunConfig:
    overrides: dict[str, object] = {}
    for env_var, (field_name, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if not raw:
            continue
        try:
            overrides[field_name] = caster(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {env_var}={raw!r}: {exc}") from exc
    return replace(cfg, **overrides)
