from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from candidate_ranking.models import Assessment


@dataclass
class AssessmentAuditReport:
    """Zero-LLM-call quality metrics derived from cached assessments.

    Designed for publication-level evaluation: every field is reference-free
    (no ground truth required) and computable from a single pipeline run.
    """

    n_jps: int
    n_assessments: int

    # --- BM25 extraction quality ---
    must_have_evals_total: int
    must_have_evals_with_evidence: int
    evidence_coverage_rate: float  # must_have_evals_with_evidence / total

    # --- Jev assessment distribution ---
    # How Jev distributes evidence_strength across must-have evaluations.
    # A healthy distribution should not be dominated by "none" or "strong" alone.
    evidence_strength_distribution: dict[str, float]  # {strength: fraction}

    # --- qualified flag integrity ---
    qualified_flag_errors: int  # assessments where flag != evidence_strength-derived value

    # --- qualification funnel ---
    qualification_rate_per_jp: dict[str, float]
    mean_qualification_rate: float
    fully_qualified_jps: list[str]   # 100% — gate may be too loose
    zero_qualified_jps: list[str]    # 0%   — gate may be too strict / must-haves misconfigured

    # --- disqualification bottlenecks ---
    # Which must-have requirements most often have evidence_strength="none".
    # Key = requirement text, value = fraction of all such requirements.
    # Useful for identifying ambiguous or unrealistic must-haves.
    most_common_disqualifiers: list[tuple[str, int]]  # [(requirement, count)] descending


def audit_assessments(
    assessments_by_jp: dict[str, dict[str, Assessment]],
) -> AssessmentAuditReport:
    """Compute publication-quality metrics from cached assessments — zero LLM calls.

    Covers three dimensions:
    1. BM25 evidence extraction quality (coverage rate)
    2. Jev assessment distribution (evidence_strength distribution)
    3. Qualification funnel (rate per JD, anomaly detection, disqualifier analysis)
    """
    n_assessments = sum(len(v) for v in assessments_by_jp.values())
    must_have_total = 0
    must_have_with_evidence = 0
    strength_counter: Counter[str] = Counter()
    disqualifier_counter: Counter[str] = Counter()
    flag_errors = 0
    qualification_rate_per_jp: dict[str, float] = {}

    for jp_id, assessments in assessments_by_jp.items():
        n_qualified = 0
        for candidate_id, assessment in assessments.items():
            if assessment.qualified:
                n_qualified += 1

            # Qualified flag must equal evidence_strength-derived value
            computed = all(ev.evidence_strength != "none" for ev in assessment.must_have_evaluations)
            if computed != assessment.qualified:
                flag_errors += 1

            for ev in assessment.must_have_evaluations:
                must_have_total += 1

                # BM25 coverage
                if ev.evidence and ev.evidence.strip():
                    must_have_with_evidence += 1

                # Evidence strength distribution
                if ev.evidence_strength:
                    strength_counter[ev.evidence_strength] += 1

                # Disqualification bottlenecks
                if ev.evidence_strength == "none":
                    disqualifier_counter[ev.requirement] += 1

        n = len(assessments)
        if n > 0:
            qualification_rate_per_jp[jp_id] = n_qualified / n

    # Normalise strength distribution to fractions
    total_strengths = sum(strength_counter.values())
    evidence_strength_distribution: dict[str, float] = {
        k: v / total_strengths for k, v in strength_counter.items()
    } if total_strengths else {}

    return AssessmentAuditReport(
        n_jps=len(assessments_by_jp),
        n_assessments=n_assessments,
        must_have_evals_total=must_have_total,
        must_have_evals_with_evidence=must_have_with_evidence,
        evidence_coverage_rate=(
            must_have_with_evidence / must_have_total if must_have_total else 1.0
        ),
        evidence_strength_distribution=evidence_strength_distribution,
        qualified_flag_errors=flag_errors,
        qualification_rate_per_jp=qualification_rate_per_jp,
        mean_qualification_rate=(
            sum(qualification_rate_per_jp.values()) / len(qualification_rate_per_jp)
            if qualification_rate_per_jp else 0.0
        ),
        fully_qualified_jps=[jp for jp, r in qualification_rate_per_jp.items() if r >= 1.0],
        zero_qualified_jps=[jp for jp, r in qualification_rate_per_jp.items() if r == 0.0],
        most_common_disqualifiers=disqualifier_counter.most_common(),
    )
