from __future__ import annotations

import pytest

from candidate_ranking.evaluation.criteria_optimization import (
    REQUIREMENT_LEVEL_COUNT,
    RequirementTriple,
    load_requirement_triples,
    triple_key,
    validate_criteria_shape,
)


def test_requirement_level_count_is_five():
    assert REQUIREMENT_LEVEL_COUNT == 5


def test_validate_criteria_shape_accepts_five_nonempty_levels():
    validate_criteria_shape(["a", "b", "c", "d", "e"])  # must not raise


def test_validate_criteria_shape_rejects_wrong_count():
    with pytest.raises(ValueError, match="exactly 5"):
        validate_criteria_shape(["a", "b", "c"])


def test_validate_criteria_shape_rejects_empty_level():
    with pytest.raises(ValueError, match="empty"):
        validate_criteria_shape(["a", "b", "", "d", "e"])


def test_triple_key_is_stable_and_distinct():
    t1 = RequirementTriple(job_description_id="jd-1", candidate_id="c-1", requirement="Python")
    t2 = RequirementTriple(job_description_id="jd-1", candidate_id="c-1", requirement="SQL")
    assert triple_key(t1) != triple_key(t2)
    assert triple_key(t1) == triple_key(RequirementTriple(job_description_id="jd-1", candidate_id="c-1", requirement="Python"))


def test_load_requirement_triples_extracts_requirement_keys_only():
    assessments_by_jd = {
        "jd-1": {
            "cand-1": {
                "confidence": {
                    "overall_recommendation": 0.8,
                    "requirement::Python": 0.9,
                    "requirement::SQL": 0.7,
                },
            },
            "cand-2": {
                "confidence": {"requirement::Python": 0.6},
            },
        },
    }

    triples = load_requirement_triples(assessments_by_jd)

    assert len(triples) == 3
    assert RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python") in triples
    assert RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="SQL") in triples
    assert RequirementTriple(job_description_id="jd-1", candidate_id="cand-2", requirement="Python") in triples
