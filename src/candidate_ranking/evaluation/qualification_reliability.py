from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from candidate_ranking.models import Assessment, Candidate, JobProfile
from candidate_ranking.scoring.assessment import generate_assessment

if TYPE_CHECKING:
    from candidate_ranking.scoring.jev_client import JevClient

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityReport:
    """Test-retest reliability of the qualification gate, with no ground truth.

    There is no labeled "correct" answer for whether a candidate is qualified —
    only a recruiter's judgment could provide that, and this evaluation has none.
    What CAN be measured without any ground truth is whether the same judge (the
    LLM, at its configured temperature) reaches the same conclusion run after run
    on the exact same (JD, candidate) input. Low agreement means the conclusion is
    mostly noise regardless of how plausible any single run looks; it does not by
    itself say which run (if any) is "right".
    """

    n_pairs: int
    n_repeats: int
    qualified_agreement_rate: float
    qualified_cohens_kappa: float
    evidence_strength_agreement_rate: float
    per_pair_qualified: dict[str, list[bool]]


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Chance-corrected agreement between two paired binary label sequences.

    Raw agreement rate alone overstates reliability whenever one label is much
    more common than the other (e.g. if 90% of candidates get `qualified=True`,
    two independent coin-flips would already "agree" 82% of the time). Kappa
    subtracts out the agreement expected from the labels' own base rates, so a
    kappa near 0 means the judge is no more self-consistent than chance given
    its own qualify/disqualify tendency, and 1.0 means perfect agreement beyond
    that baseline.
    """
    n = len(labels_a)
    if n == 0:
        return 1.0
    observed_agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_a_true = sum(labels_a) / n
    p_b_true = sum(labels_b) / n
    expected_agreement = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if expected_agreement >= 1.0:
        return 1.0 if observed_agreement >= 1.0 else 0.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def _evidence_strength_map(assessment: Assessment) -> dict[str, str]:
    return {
        ev.requirement.strip().lower(): ev.evidence_strength
        for ev in assessment.must_have_evaluations
    }


def repeat_qualification_gate(
    jp: JobProfile,
    candidate: Candidate,
    jev_client: "JevClient",
    repeats: int = 3,
) -> list[Assessment]:
    """Run the qualification gate `repeats` independent times for one (JD,
    candidate) pair — every call hits Jev fresh, bypassing the on-disk
    cache entirely, since the cache is keyed to return the same answer forever
    and would make every "repeat" trivially identical."""
    return [
        generate_assessment(jp, candidate, jev_client)
        for _ in range(repeats)
    ]


def build_reliability_report(
    repeated_assessments: dict[str, list[Assessment]],
) -> ReliabilityReport:
    """`repeated_assessments` maps a pair id (e.g. "jd_id/candidate_id") to the
    list of Assessment objects produced by independent repeat calls for that
    exact pair."""
    per_pair_qualified: dict[str, list[bool]] = {}
    paired_qualified_a: list[bool] = []
    paired_qualified_b: list[bool] = []
    strength_matches = 0
    strength_total = 0

    for pair_id, assessments in repeated_assessments.items():
        if len(assessments) < 2:
            logger.info("Skipping %s: fewer than 2 repeats", pair_id)
            continue
        per_pair_qualified[pair_id] = [a.qualified for a in assessments]
        strength_maps = [_evidence_strength_map(a) for a in assessments]

        for i in range(len(assessments)):
            for j in range(i + 1, len(assessments)):
                paired_qualified_a.append(assessments[i].qualified)
                paired_qualified_b.append(assessments[j].qualified)
                shared_requirements = set(strength_maps[i]) & set(strength_maps[j])
                for requirement in shared_requirements:
                    strength_total += 1
                    if strength_maps[i][requirement] == strength_maps[j][requirement]:
                        strength_matches += 1

    if not per_pair_qualified:
        raise ValueError("No pairs with ≥ 2 repeats found")

    n_comparisons = len(paired_qualified_a)
    qualified_agreement_rate = (
        sum(1 for a, b in zip(paired_qualified_a, paired_qualified_b) if a == b) / n_comparisons
    )
    qualified_kappa = cohens_kappa(paired_qualified_a, paired_qualified_b)
    evidence_strength_agreement_rate = strength_matches / strength_total if strength_total else 1.0

    return ReliabilityReport(
        n_pairs=len(per_pair_qualified),
        n_repeats=max(len(v) for v in per_pair_qualified.values()),
        qualified_agreement_rate=qualified_agreement_rate,
        qualified_cohens_kappa=qualified_kappa,
        evidence_strength_agreement_rate=evidence_strength_agreement_rate,
        per_pair_qualified=per_pair_qualified,
    )
