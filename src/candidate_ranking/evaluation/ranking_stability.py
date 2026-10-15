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
