from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.active_listwise_ranking import rank_active_listwise

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    per_jd: dict[str, float]
    mean_tau: float
    min_tau: float
    min_tau_jd: str


def kendall_tau(
    utilities_a: dict[str, float],
    utilities_b: dict[str, float],
) -> float:
    from scipy.stats import kendalltau as _kendalltau

    cids = sorted(utilities_a)
    a = [utilities_a[c] for c in cids]
    b = [utilities_b[c] for c in cids]
    tau, _ = _kendalltau(a, b)
    return float(tau)


def evaluate_jd_stability(
    jd: JobDescription,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> float:
    run_a = rank_active_listwise(jd, assessments, llm, subset_size, max_iterations, min_delta_u, pl_prior_variance)
    run_b = rank_active_listwise(jd, assessments, llm, subset_size, max_iterations, min_delta_u, pl_prior_variance)
    return kendall_tau(run_a, run_b)


def run_stability_report(
    jd_assessments: dict[str, tuple[JobDescription, dict[str, Assessment]]],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> StabilityReport:
    per_jd: dict[str, float] = {}
    for jd_id, (jd, assessments) in jd_assessments.items():
        if len(assessments) < 2:
            logger.info("Skipping JD %s: fewer than 2 assessed candidates", jd_id)
            continue
        per_jd[jd_id] = evaluate_jd_stability(
            jd, assessments, llm, subset_size, max_iterations, min_delta_u, pl_prior_variance
        )

    if not per_jd:
        raise ValueError("No JDs with ≥ 2 assessed candidates found")

    min_jd = min(per_jd, key=per_jd.__getitem__)
    return StabilityReport(
        per_jd=per_jd,
        mean_tau=sum(per_jd.values()) / len(per_jd),
        min_tau=per_jd[min_jd],
        min_tau_jd=min_jd,
    )


def load_cached_assessments(cache_dir: Path) -> dict[str, dict[str, Assessment]]:
    assessments_dir = cache_dir / "assessments"
    if not assessments_dir.exists():
        return {}
    result: dict[str, dict[str, Assessment]] = {}
    for jd_dir in assessments_dir.iterdir():
        if not jd_dir.is_dir():
            continue
        for candidate_file in jd_dir.glob("*.json"):
            try:
                assessment = Assessment.model_validate_json(
                    candidate_file.read_text(encoding="utf-8")
                )
                result.setdefault(jd_dir.name, {})[assessment.candidate_id] = assessment
            except Exception as exc:
                logger.warning("Skipping %s: %s", candidate_file, exc)
    return result
