from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window

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
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, int] = (42, 137),
) -> float:
    run_a = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seeds[0])
    run_b = rank_sliding_window(jd, assessments, build_chain, window_size, step, rng_seed=seeds[1])
    return kendall_tau(run_a, run_b)


def run_stability_report(
    jd_assessments: dict[str, tuple[JobDescription, dict[str, Assessment]]],
    build_chain: Callable[[type[BaseModel]], Runnable],
    window_size: int = 15,
    step: int = 7,
    seeds: tuple[int, int] = (42, 137),
) -> StabilityReport:
    per_jd: dict[str, float] = {}
    for jd_id, (jd, assessments) in jd_assessments.items():
        qualified = {cid: a for cid, a in assessments.items() if a.qualified}
        if len(qualified) < 2:
            logger.info("Skipping JD %s: fewer than 2 qualified candidates", jd_id)
            continue
        per_jd[jd_id] = evaluate_jd_stability(jd, qualified, build_chain, window_size, step, seeds)

    if not per_jd:
        raise ValueError("No JDs with ≥ 2 qualified candidates found")

    min_jd = min(per_jd, key=per_jd.__getitem__)
    return StabilityReport(
        per_jd=per_jd,
        mean_tau=sum(per_jd.values()) / len(per_jd),
        min_tau=per_jd[min_jd],
        min_tau_jd=min_jd,
    )
