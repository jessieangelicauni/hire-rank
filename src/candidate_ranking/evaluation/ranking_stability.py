from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from candidate_ranking.models import Assessment, JobProfile
from candidate_ranking.scoring.active_pairwise_ranking import rank_active_pairwise
from candidate_ranking.utils import load_json_cache

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    per_jd: dict[str, float]
    mean_tau: float
    min_tau: float
    min_tau_jd: str
    rbo_per_jd: dict[str, float]
    mean_rbo: float
    min_rbo: float
    min_rbo_jd: str


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


def rank_biased_overlap(
    utilities_a: dict[str, float],
    utilities_b: dict[str, float],
    p: float = 0.9,
) -> float:
    """Top-weighted rank agreement between two runs over the same item set
    (Webber, Moffat & Zobel 2010). Kendall-tau weighs every pair of ranks
    equally, so two runs that agree on who's in the top 5 but shuffle the
    bottom half of a 26-candidate pool score about the same as two runs that
    disagree on the top 5 but agree on the tail — not what a recruiter cares
    about. RBO fixes that: weight decays geometrically with rank depth, so
    agreement near the top dominates the score. `p` sets how fast weight
    decays (p=0.9 concentrates most of the weight in roughly the top 10 ranks
    regardless of total list length). Returns 1.0 for identical orderings.

    This is the "RBO" base measure evaluated to the full list depth — since
    both rankings here always cover the exact same closed candidate set,
    there is no unseen tail to extrapolate over (unlike the open/web-search
    setting RBO was originally designed for).
    """
    order_a = sorted(utilities_a, key=utilities_a.__getitem__, reverse=True)
    order_b = sorted(utilities_b, key=utilities_b.__getitem__, reverse=True)
    k = len(order_a)
    if k <= 1:
        return 1.0

    weighted_sum = 0.0
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    for depth in range(1, k + 1):
        seen_a.add(order_a[depth - 1])
        seen_b.add(order_b[depth - 1])
        agreement_at_depth = len(seen_a & seen_b) / depth
        weighted_sum += (p ** (depth - 1)) * agreement_at_depth
    # Normalize by the weight actually available over this finite, closed list
    # (both rankings cover the exact same item set up to depth k) rather than
    # the (1-p) factor for an infinite/open tail — otherwise identical lists
    # would score below 1.0 simply for being short relative to 1/(1-p).
    total_weight = sum(p ** d for d in range(k))
    return weighted_sum / total_weight


def evaluate_jd_stability(
    jp: JobProfile,
    assessments: dict[str, Assessment],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> tuple[float, float]:
    kwargs = dict(llm=llm, subset_size=subset_size, max_iterations=max_iterations,
                  min_delta_u=min_delta_u, pl_prior_variance=pl_prior_variance)
    run_a = rank_active_pairwise(jp, assessments, **kwargs)
    run_b = rank_active_pairwise(jp, assessments, **kwargs)
    return kendall_tau(run_a, run_b), rank_biased_overlap(run_a, run_b)


def run_stability_report(
    jd_assessments: dict[str, tuple[JobDescription, dict[str, Assessment]]],
    llm: "BaseChatModel",
    subset_size: int = 8,
    max_iterations: int = 40,
    min_delta_u: float = 0.01,
    pl_prior_variance: float = 1.0,
) -> StabilityReport:
    per_jd: dict[str, float] = {}
    rbo_per_jd: dict[str, float] = {}
    for jp_id, (jp, assessments) in jd_assessments.items():
        if len(assessments) < 2:
            logger.info("Skipping JD %s: fewer than 2 assessed candidates", jp_id)
            continue
        tau, rbo = evaluate_jd_stability(
            jp, assessments, llm, subset_size, max_iterations, min_delta_u, pl_prior_variance
        )
        per_jd[jp_id] = tau
        rbo_per_jd[jp_id] = rbo

    if not per_jd:
        raise ValueError("No JDs with ≥ 2 assessed candidates found")

    min_tau_jd = min(per_jd, key=per_jd.__getitem__)
    min_rbo_jd = min(rbo_per_jd, key=rbo_per_jd.__getitem__)
    return StabilityReport(
        per_jd=per_jd,
        mean_tau=sum(per_jd.values()) / len(per_jd),
        min_tau=per_jd[min_tau_jd],
        min_tau_jd=min_tau_jd,
        rbo_per_jd=rbo_per_jd,
        mean_rbo=sum(rbo_per_jd.values()) / len(rbo_per_jd),
        min_rbo=rbo_per_jd[min_rbo_jd],
        min_rbo_jd=min_rbo_jd,
    )


def load_job_profiles(cache_dir: Path) -> dict[str, JobProfile]:
    cache = load_json_cache(cache_dir / "job_profile.json", "job profile cache")
    profiles: dict[str, JobProfile] = {}
    for jp_id, entry in cache.items():
        try:
            profiles[jp_id] = JobProfile.model_validate(entry["job_profile"])
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid cached job profile for %s: %s", jp_id, exc)
    return profiles


def _assessment_matches_profile(assessment: Assessment, profile: JobProfile | None) -> bool:
    """Reject assessments scored against a must-have list that no longer matches the
    JD's current job profile — stale leftovers from before the JD was edited. Mixing
    these into a ranking subset confuses the LLM: the candidate's evidence doesn't even
    speak to the same requirements as the rest of the subset."""
    if profile is None:
        return True
    evaluated_requirements = {ev.requirement.strip().lower() for ev in assessment.must_have_evaluations}
    current_must_haves = {skill.strip().lower() for skill in profile.must_have_skills}
    return evaluated_requirements == current_must_haves


def load_cached_assessments(
    cache_dir: Path, job_profiles: dict[str, JobProfile] | None = None
) -> dict[str, dict[str, Assessment]]:
    assessments_dir = cache_dir / "assessments"
    if not assessments_dir.exists():
        return {}
    if job_profiles is None:
        job_profiles = load_job_profiles(cache_dir)
    result: dict[str, dict[str, Assessment]] = {}
    for jd_dir in assessments_dir.iterdir():
        if not jd_dir.is_dir():
            continue
        profile = job_profiles.get(jd_dir.name)
        for candidate_file in jd_dir.glob("*.json"):
            try:
                data = json.loads(candidate_file.read_text(encoding="utf-8"))
                assessment = Assessment.model_validate(data["assessment"])
                if not _assessment_matches_profile(assessment, profile):
                    logger.warning(
                        "Skipping stale assessment %s: evaluated against an old job profile", candidate_file
                    )
                    continue
                result.setdefault(jd_dir.name, {})[assessment.candidate_id] = assessment
            except Exception as exc:
                logger.warning("Skipping %s: %s", candidate_file, exc)
    return result
