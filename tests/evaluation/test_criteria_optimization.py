from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from candidate_ranking.evaluation.criteria_optimization import (
    REQUIREMENT_LEVEL_COUNT,
    CriteriaEvalRecord,
    RequirementTriple,
    build_requirement_question,
    compute_metric,
    evaluate_criteria,
    load_requirement_triples,
    select_hard_triples,
    triple_key,
    validate_criteria_shape,
)
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.jev_client import JevAnswer


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


_CRITERIA = ["never", "rarely", "sometimes", "often", "always"]


def test_build_requirement_question_matches_existing_instructions_format():
    question = build_requirement_question("Python", _CRITERIA)
    assert question.key == "requirement::Python"
    assert question.kind == "score"
    assert question.instructions == "How well does the candidate's CV support the requirement 'Python'?"
    assert question.criteria == _CRITERIA


def test_build_requirement_question_rejects_bad_shape():
    with pytest.raises(ValueError, match="exactly 5"):
        build_requirement_question("Python", ["only", "two"])


def test_evaluate_criteria_calls_jev_once_per_triple_and_maps_results():
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python.", source_path="jd.pdf")
    candidate = Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok",
    )
    triples = [RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")]
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.85)
    ]

    records = evaluate_criteria(_CRITERIA, triples, {"jd-1": jd}, {"cand-1": candidate}, jev_client)

    assert records == [
        CriteriaEvalRecord(triple=triples[0], score=3.0, confidence=0.85)
    ]
    jev_client.evaluate.assert_called_once()
    call_state, call_questions = jev_client.evaluate.call_args.args
    assert "Needs Python." in call_state
    assert "I know Python." in call_state
    assert call_questions[0].key == "requirement::Python"


# Tests for compute_metric and select_hard_triples


def _record(jd_id: str, cand_id: str, requirement: str, score: float, confidence: float) -> CriteriaEvalRecord:
    return CriteriaEvalRecord(
        triple=RequirementTriple(job_description_id=jd_id, candidate_id=cand_id, requirement=requirement),
        score=score,
        confidence=confidence,
    )


def test_compute_metric_credits_confidence_only_when_agreeing_with_proxy_label():
    correct = _record("jd-1", "c-1", "Python", score=3.0, confidence=0.9)
    wrong = _record("jd-1", "c-2", "Python", score=0.0, confidence=0.9)
    proxy_labels = {
        triple_key(correct.triple): 3,  # agrees (within 1): credited
        triple_key(wrong.triple): 4,    # disagrees by 4: not credited
    }

    metric = compute_metric([correct, wrong], proxy_labels)

    assert metric == pytest.approx((0.9 + 0.0) / 2)


def test_compute_metric_confidently_wrong_scores_lower_than_confidently_right():
    right = _record("jd-1", "c-1", "Python", score=3.0, confidence=0.9)
    wrong = _record("jd-1", "c-2", "Python", score=0.0, confidence=0.9)
    labels_all_right = {triple_key(right.triple): 3, triple_key(wrong.triple): 0}
    labels_one_wrong = {triple_key(right.triple): 3, triple_key(wrong.triple): 4}

    metric_both_right = compute_metric([right, wrong], labels_all_right)
    metric_one_wrong = compute_metric([right, wrong], labels_one_wrong)

    assert metric_one_wrong < metric_both_right


def test_compute_metric_empty_records_returns_zero():
    assert compute_metric([], {}) == 0.0


def test_select_hard_triples_returns_k_lowest_confidence():
    records = [
        _record("jd-1", "c-1", "Python", score=3.0, confidence=0.9),
        _record("jd-1", "c-2", "Python", score=2.0, confidence=0.3),
        _record("jd-1", "c-3", "Python", score=1.0, confidence=0.6),
    ]

    hardest_two = select_hard_triples(records, k=2)

    assert hardest_two == [records[1].triple, records[2].triple]


def test_select_hard_triples_k_larger_than_records_returns_all():
    records = [_record("jd-1", "c-1", "Python", score=3.0, confidence=0.5)]
    assert select_hard_triples(records, k=5) == [records[0].triple]


def test_load_assessments_by_jd_reads_each_jd_file(tmp_path):
    from candidate_ranking.evaluation.criteria_optimization import load_assessments_by_jd

    jd_dir = tmp_path / "jd-1"
    jd_dir.mkdir()
    (jd_dir / "assessments.json").write_text(
        json.dumps({"cand-1": {"confidence": {"requirement::Python": 0.9}}}), encoding="utf-8"
    )

    result = load_assessments_by_jd(tmp_path, ["jd-1"])

    assert result == {"jd-1": {"cand-1": {"confidence": {"requirement::Python": 0.9}}}}


def test_load_assessments_by_jd_skips_missing_files(tmp_path):
    from candidate_ranking.evaluation.criteria_optimization import load_assessments_by_jd

    result = load_assessments_by_jd(tmp_path, ["jd-missing"])

    assert result == {}
