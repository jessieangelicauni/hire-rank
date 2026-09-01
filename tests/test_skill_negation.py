
import numpy as np

from candidate_ranking.scoring.skills import bridge_negated_jd_skills_to_candidate, find_negated_skill_mentions


def test_short_skill_name_is_not_falsely_matched_inside_an_unrelated_word():
    text = "Lacks strong SQL skills and experience building ETL/ELT pipelines using Airflow, Dagster, dbt"
    assert find_negated_skill_mentions(text, ["R"]) == []


def test_short_skill_name_is_not_falsely_matched_inside_version_control():
    text = "Lacks experience with Power BI, AS400, version control (Git) and CI/CD for data pipelines"
    assert find_negated_skill_mentions(text, ["R"]) == []


def test_short_skill_name_is_detected_when_genuinely_negated_as_its_own_word():
    text = "No experience with R for statistical analysis."
    assert find_negated_skill_mentions(text, ["R"]) == ["R"]


def test_contrastive_phrasing_suppresses_a_false_positive():
    text = "Lacks formal certification, though Terraform is evident throughout the CV."
    assert find_negated_skill_mentions(text, ["Terraform"]) == []


def test_category_label_for_a_more_specific_tool_is_not_flagged():
    text = "No experience with Infrastructure as Code tools."
    assert find_negated_skill_mentions(text, ["Terraform"]) == []


def _fake_embedder(strings: list[str]) -> np.ndarray:
    canonical_index = {"rest api design": 0, "restful api design": 0, "kubernetes": 1}
    dim = 3
    vectors = np.zeros((len(strings), dim))
    for row, s in enumerate(strings):
        vectors[row, canonical_index.get(s.lower(), dim - 1)] = 1.0
    return vectors


def test_bridge_finds_a_negated_jd_skill_matching_a_candidate_skill_under_different_wording():
    bridged = bridge_negated_jd_skills_to_candidate(
        weaknesses=["No specific mention of RESTful API design."],
        candidate_skills=["REST API Design"],
        jd_technical_skills=["RESTful API design"],
        embedder=_fake_embedder,
    )
    assert bridged == {"RESTful API design": "REST API Design"}


def test_bridge_returns_empty_when_negated_jd_skill_has_no_matching_candidate_skill():
    bridged = bridge_negated_jd_skills_to_candidate(
        weaknesses=["No specific mention of Kubernetes."],
        candidate_skills=["REST API Design"],
        jd_technical_skills=["Kubernetes"],
        embedder=_fake_embedder,
    )
    assert bridged == {}


def test_bridge_returns_empty_when_jd_skill_is_not_negated():
    bridged = bridge_negated_jd_skills_to_candidate(
        weaknesses=["Strong experience with RESTful API design."],
        candidate_skills=["REST API Design"],
        jd_technical_skills=["RESTful API design"],
        embedder=_fake_embedder,
    )
    assert bridged == {}
