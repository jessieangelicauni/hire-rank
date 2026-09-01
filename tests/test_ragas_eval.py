
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candidate_ranking.config import RunConfig
from candidate_ranking.evaluation.ragas_eval import (
    LOW_SCORE_CUTOFF,
    NEUTRAL_USER_INPUT,
    FaithfulnessRow,
    build_rows,
    find_weakness_contradictions,
    merge_and_report,
    run_ragas_faithfulness_eval,
    run_weakness_contradiction_check,
    stratified_sample,
)
from candidate_ranking.ingestion.cv import _ExtractedSkills, load_candidates
from candidate_ranking.models import Assessment, Candidate, JDSkills
from candidate_ranking.scoring.jd_skills import _GeneratedSkills

_JD_TITLE = "Frontend Engineer"


class _FakeJDSkillsChain:
    def invoke(self, payload):
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["React", "Vue"])


class _FakeCvSkillsChain:
    def invoke(self, payload):
        return _ExtractedSkills(reasoning="fake reasoning", skills=["React"])

REAL_CV_DIR = Path(__file__).resolve().parents[1] / "cv"


def _fake_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        preset="full", jd_dir=tmp_path / "jd", cv_dir=tmp_path / "cv", cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs", max_jds=None, max_candidates=None, tournament_iterations=1,
        stability_repeats=1, tournament_subset_size=1, num_subset_samples=1, num_mc_draws=1,
        pl_prior_variance=1.0, ollama_model="fake-model", ollama_base_url="http://localhost:11434",
        ollama_num_parallel=4, skill_embedding_model="fake-embed-model",
    )


def _write_jd(jd_dir: Path, title: str) -> None:
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / f"{title}.txt").write_text(f"{title}\nBuild things.", encoding="utf-8")


def _write_real_cv(cv_dir: Path, filename: str = "cv_00201.pdf") -> None:
    cv_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_CV_DIR / filename, cv_dir / filename)


def _write_manifest(runs_dir: Path, run_id: str, jd_ids: list[str]) -> None:
    path = runs_dir / run_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jd_ids": jd_ids}), encoding="utf-8")


def _write_assessment(runs_dir: Path, run_id: str, jd_id: str, assessment: Assessment) -> None:
    path = runs_dir / run_id / jd_id / "assessments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing[assessment.candidate_id] = assessment.model_dump()
    path.write_text(json.dumps(existing), encoding="utf-8")


def test_build_rows_produces_one_row_per_strength_and_weakness(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]
    assert candidate.id == "cv-00201"

    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    assessment = Assessment(
        job_description_id="backend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Built APIs in Python."], weaknesses=["No Kubernetes experience noted."],
    )
    _write_assessment(cfg.runs_dir, "run-1", "backend-engineer", assessment)

    rows = build_rows(cfg, "run-1")

    assert rows == [
        FaithfulnessRow(
            jd_id="backend-engineer", candidate_id=candidate.id, item_type="strength",
            response="Built APIs in Python.", retrieved_contexts=[candidate.raw_text],
            user_input=NEUTRAL_USER_INPUT,
        ),
        FaithfulnessRow(
            jd_id="backend-engineer", candidate_id=candidate.id, item_type="weakness",
            response="No Kubernetes experience noted.", retrieved_contexts=[candidate.raw_text],
            user_input=NEUTRAL_USER_INPUT,
        ),
    ]


def test_build_rows_skips_jd_removed_from_job_description(tmp_path, caplog):
    cfg = _fake_cfg(tmp_path)
    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]
    assessment = Assessment(
        job_description_id="backend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Built APIs in Python."], weaknesses=[],
    )
    _write_assessment(cfg.runs_dir, "run-1", "backend-engineer", assessment)

    with caplog.at_level("WARNING"):
        rows = build_rows(cfg, "run-1")

    assert rows == []
    assert any("backend-engineer" in record.message for record in caplog.records)


def test_build_rows_skips_candidate_removed_from_cv_dir(tmp_path, caplog):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    assessment = Assessment(
        job_description_id="backend-engineer", candidate_id="cv-00201", generated_by_model="fake-model",
        strengths=["Built APIs in Python."], weaknesses=[],
    )
    _write_assessment(cfg.runs_dir, "run-1", "backend-engineer", assessment)

    with caplog.at_level("WARNING"):
        rows = build_rows(cfg, "run-1")

    assert rows == []
    assert any("cv-00201" in record.message for record in caplog.records)


def _row(jd_id="backend-engineer", candidate_id="cv-001", item_type="strength") -> FaithfulnessRow:
    return FaithfulnessRow(
        jd_id=jd_id, candidate_id=candidate_id, item_type=item_type,
        response="Built APIs in Python.", retrieved_contexts=["Jane: Python."],
        user_input=f"Backend Engineer — {item_type}",
    )


def test_merge_and_report_computes_mean_score():
    rows = [_row(item_type="strength"), _row(item_type="weakness")]
    report = merge_and_report("run-1", rows, ragas_scores=[1.0, 0.5])

    assert report.run_id == "run-1"
    assert report.n_rows == 2
    assert report.n_ragas_failures == 0
    assert report.mean_score == pytest.approx(0.75)
    assert report.low_score_items == []


def test_merge_and_report_flags_low_score_items():
    rows = [_row(item_type="strength")]
    report = merge_and_report("run-1", rows, ragas_scores=[0.2])

    assert report.n_ragas_failures == 0
    assert len(report.low_score_items) == 1
    item = report.low_score_items[0]
    assert item["jd_id"] == "backend-engineer"
    assert item["candidate_id"] == "cv-001"
    assert item["item_type"] == "strength"
    assert item["item_text"] == "Built APIs in Python."
    assert item["ragas_score"] == 0.2


def test_merge_and_report_does_not_flag_scores_at_or_above_cutoff():
    rows = [_row(item_type="strength")]
    report = merge_and_report("run-1", rows, ragas_scores=[LOW_SCORE_CUTOFF])

    assert report.low_score_items == []


def test_merge_and_report_per_jd_breakdown():
    rows = [
        _row(jd_id="backend-engineer", item_type="strength"),
        _row(jd_id="frontend-engineer", item_type="strength"),
    ]
    report = merge_and_report("run-1", rows, ragas_scores=[1.0, 0.0])

    assert report.per_jd["backend-engineer"] == pytest.approx(1.0)
    assert report.per_jd["frontend-engineer"] == pytest.approx(0.0)


def test_merge_and_report_handles_zero_rows():
    report = merge_and_report("run-1", [], [])

    assert report.n_rows == 0
    assert report.n_ragas_failures == 0
    assert report.mean_score is None
    assert report.low_score_items == []
    assert report.per_jd == {}


def test_faithfulness_report_to_dict_is_json_serializable():
    report = merge_and_report("run-1", [_row()], [0.2])
    json.dumps(report.to_dict())


def test_merge_and_report_defaults_weakness_contradictions_to_empty_list():
    report = merge_and_report("run-1", [_row()], [0.2])

    assert report.weakness_contradictions == []
    assert report.to_dict()["weakness_contradictions"] == []


def test_merge_and_report_excludes_nan_ragas_scores_from_stats():
    rows = [_row(item_type="strength"), _row(item_type="weakness")]
    report = merge_and_report("run-1", rows, ragas_scores=[1.0, float("nan")])

    assert report.n_rows == 2
    assert report.n_ragas_failures == 1
    assert report.mean_score == pytest.approx(1.0)
    assert report.low_score_items == []


def test_merge_and_report_reports_all_failures_when_every_score_is_nan():
    rows = [_row()]
    report = merge_and_report("run-1", rows, ragas_scores=[float("nan")])

    assert report.n_rows == 1
    assert report.n_ragas_failures == 1
    assert report.mean_score is None
    assert report.low_score_items == []


def test_run_ragas_faithfulness_eval_writes_report(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]

    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    assessment = Assessment(
        job_description_id="backend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Knows Python."], weaknesses=[],
    )
    _write_assessment(cfg.runs_dir, "run-1", "backend-engineer", assessment)

    def fake_evaluate_fn(dataset, llm):
        return pd.DataFrame([{"response": sample.response, "faithfulness": 0.9} for sample in dataset.samples])

    output_path = tmp_path / "report.json"
    result = run_ragas_faithfulness_eval(cfg, "run-1", output_path, evaluate_fn=fake_evaluate_fn)

    assert result.n_rows == 1
    assert result.n_ragas_failures == 0
    assert result.mean_score == pytest.approx(0.9)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["n_rows"] == 1
    assert report["n_ragas_failures"] == 0
    assert report["mean_score"] == pytest.approx(0.9)


def test_run_ragas_faithfulness_eval_never_sends_weakness_rows_to_ragas(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]

    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    assessment = Assessment(
        job_description_id="backend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Knows Python."], weaknesses=["Lacks Kubernetes experience."],
    )
    _write_assessment(cfg.runs_dir, "run-1", "backend-engineer", assessment)

    seen_responses: list[str] = []

    def fake_evaluate_fn(dataset, llm):
        seen_responses.extend(sample.response for sample in dataset.samples)
        return pd.DataFrame([{"response": sample.response, "faithfulness": 0.9} for sample in dataset.samples])

    output_path = tmp_path / "report.json"
    result = run_ragas_faithfulness_eval(cfg, "run-1", output_path, evaluate_fn=fake_evaluate_fn)

    assert seen_responses == ["Knows Python."]
    assert result.n_rows == 1
    assert result.population_size == 1


def test_run_ragas_faithfulness_eval_checkpoints_progress_before_a_later_group_fails(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_jd(cfg.jd_dir, "Frontend Engineer")
    _write_real_cv(cfg.cv_dir, "cv_00201.pdf")
    _write_real_cv(cfg.cv_dir, "cv_00202.pdf")
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    backend_candidate = candidates_by_id["cv-00201"]
    frontend_candidate = candidates_by_id["cv-00202"]

    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer", "frontend-engineer"])
    _write_assessment(
        cfg.runs_dir, "run-1", "backend-engineer",
        Assessment(
            job_description_id="backend-engineer", candidate_id=backend_candidate.id, generated_by_model="fake-model",
            strengths=["Knows Python."], weaknesses=[],
        ),
    )
    _write_assessment(
        cfg.runs_dir, "run-1", "frontend-engineer",
        Assessment(
            job_description_id="frontend-engineer", candidate_id=frontend_candidate.id, generated_by_model="fake-model",
            strengths=["Knows React."], weaknesses=[],
        ),
    )

    def flaky_evaluate_fn(dataset, llm):
        response = dataset.samples[0].response
        if response == "Knows React.":
            raise RuntimeError("simulated Ollama failure on the second JD")
        return pd.DataFrame([{"response": s.response, "faithfulness": 0.9} for s in dataset.samples])

    output_path = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="simulated Ollama failure"):
        run_ragas_faithfulness_eval(cfg, "run-1", output_path, evaluate_fn=flaky_evaluate_fn)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["n_rows"] == 1
    assert "backend-engineer" in report["per_jd"]
    assert "frontend-engineer" not in report["per_jd"]


def test_run_ragas_faithfulness_eval_raises_clear_error_when_no_rows(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])

    with pytest.raises(SystemExit, match="No Ragas-evaluable data"):
        run_ragas_faithfulness_eval(cfg, "run-1", tmp_path / "report.json", evaluate_fn=lambda *a, **k: None)


def test_stratified_sample_returns_everything_when_sample_size_covers_the_population():
    rows = [_row(candidate_id=f"cv-{i}") for i in range(5)]

    sampled = stratified_sample(rows, sample_size=10, seed=1)

    assert sampled == rows


def test_stratified_sample_samples_equally_per_jd_regardless_of_pool_size():
    rows = [_row(jd_id="jd-a", candidate_id=f"cv-a{i}") for i in range(80)] + [
        _row(jd_id="jd-b", candidate_id=f"cv-b{i}") for i in range(20)
    ]

    sampled = stratified_sample(rows, sample_size=10, seed=1)

    n_a = sum(1 for r in sampled if r.jd_id == "jd-a")
    n_b = sum(1 for r in sampled if r.jd_id == "jd-b")
    assert len(sampled) == 10
    assert n_a == 5
    assert n_b == 5


def test_stratified_sample_distributes_remainder_deterministically():
    rows = [_row(jd_id="jd-a", candidate_id=f"cv-a{i}") for i in range(80)] + [
        _row(jd_id="jd-b", candidate_id=f"cv-b{i}") for i in range(80)
    ]

    sampled = stratified_sample(rows, sample_size=11, seed=1)

    n_a = sum(1 for r in sampled if r.jd_id == "jd-a")
    n_b = sum(1 for r in sampled if r.jd_id == "jd-b")
    assert len(sampled) == 11
    assert n_a == 6
    assert n_b == 5


def test_stratified_sample_never_drops_a_small_jd_entirely():
    rows = [_row(jd_id="jd-a", candidate_id=f"cv-a{i}") for i in range(95)] + [
        _row(jd_id="jd-b", candidate_id="cv-b0")
    ]

    sampled = stratified_sample(rows, sample_size=10, seed=1)

    assert any(r.jd_id == "jd-b" for r in sampled)


def test_stratified_sample_is_reproducible_with_the_same_seed():
    rows = [_row(candidate_id=f"cv-{i}") for i in range(50)]

    first = stratified_sample(rows, sample_size=10, seed=7)
    second = stratified_sample(rows, sample_size=10, seed=7)

    assert [r.candidate_id for r in first] == [r.candidate_id for r in second]


def test_run_ragas_faithfulness_eval_with_sample_size_evaluates_fewer_rows(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, "Backend Engineer")
    for i in range(20):
        _write_real_cv(cfg.cv_dir, f"cv_{201 + i:05d}.pdf")
    candidates = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")

    _write_manifest(cfg.runs_dir, "run-1", ["backend-engineer"])
    for candidate in candidates:
        _write_assessment(
            cfg.runs_dir, "run-1", "backend-engineer",
            Assessment(
                job_description_id="backend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
                strengths=["Knows Python."], weaknesses=[],
            ),
        )

    def fake_evaluate_fn(dataset, llm):
        return pd.DataFrame([{"response": sample.response, "faithfulness": 0.9} for sample in dataset.samples])

    output_path = tmp_path / "report.json"
    result = run_ragas_faithfulness_eval(
        cfg, "run-1", output_path, evaluate_fn=fake_evaluate_fn, sample_size=5, sample_seed=1
    )

    assert result.n_rows == 5
    assert result.population_size == 20
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["n_rows"] == 5
    assert report["population_size"] == 20


_SKILL_VOCAB = [
    "React", "Vue", "Redux", "Node.js", "Kubernetes", "Docker", "TypeScript",
    "Terraform", "Infrastructure as Code", "GitOps", "ArgoCD", "Flux",
]


def _one_hot(skill: str) -> np.ndarray:
    vec = np.zeros(len(_SKILL_VOCAB), dtype="float32")
    vec[_SKILL_VOCAB.index(skill)] = 1.0
    return vec


def _fake_skill_embedder(texts: list[str]) -> np.ndarray:
    return np.stack([_one_hot(t) for t in texts])


def _candidate(candidate_id: str, skills: list[str]) -> Candidate:
    return Candidate(
        id=candidate_id, source_path=f"{candidate_id}.pdf", raw_text="irrelevant",
        num_pages=1, char_count=10, parse_status="ok", skills=skills,
    )


def _jd_skills(jd_id: str, technical_skills: list[str]) -> JDSkills:
    return JDSkills(job_description_id=jd_id, generated_by_model="fake-model", technical_skills=technical_skills)


def test_find_weakness_contradictions_flags_a_skill_the_candidate_actually_has():
    rows = [
        FaithfulnessRow(
            jd_id="frontend-engineer", candidate_id="cv-001", item_type="weakness",
            response="No explicit mention of experience with React.", retrieved_contexts=["irrelevant"],
            user_input="Frontend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"frontend-engineer": _jd_skills("frontend-engineer", ["React", "Vue"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["React"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert len(contradictions) == 1
    assert contradictions[0]["jd_id"] == "frontend-engineer"
    assert contradictions[0]["candidate_id"] == "cv-001"
    assert contradictions[0]["contradicted_skill"] == "React"


def test_find_weakness_contradictions_does_not_flag_a_skill_the_candidate_lacks():
    rows = [
        FaithfulnessRow(
            jd_id="frontend-engineer", candidate_id="cv-001", item_type="weakness",
            response="No explicit mention of experience with React.", retrieved_contexts=["irrelevant"],
            user_input="Frontend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"frontend-engineer": _jd_skills("frontend-engineer", ["React", "Vue"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Vue"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_ignores_strength_rows():
    rows = [
        FaithfulnessRow(
            jd_id="frontend-engineer", candidate_id="cv-001", item_type="strength",
            response="Strong experience with React.", retrieved_contexts=["irrelevant"],
            user_input="Frontend Engineer — strength",
        )
    ]
    jd_skills_by_id = {"frontend-engineer": _jd_skills("frontend-engineer", ["React"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["React"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_skips_weakness_naming_no_jd_skill():
    rows = [
        FaithfulnessRow(
            jd_id="frontend-engineer", candidate_id="cv-001", item_type="weakness",
            response="Limited professional experience overall.", retrieved_contexts=["irrelevant"],
            user_input="Frontend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"frontend-engineer": _jd_skills("frontend-engineer", ["React"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["React"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_matches_skill_names_case_insensitively():
    rows = [
        FaithfulnessRow(
            jd_id="frontend-engineer", candidate_id="cv-001", item_type="weakness",
            response="no experience with react found in the cv.", retrieved_contexts=["irrelevant"],
            user_input="Frontend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"frontend-engineer": _jd_skills("frontend-engineer", ["React"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["React"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert len(contradictions) == 1


def test_find_weakness_contradictions_does_not_flag_a_skill_mentioned_only_as_present():
    rows = [
        FaithfulnessRow(
            jd_id="backend-engineer", candidate_id="cv-001", item_type="weakness",
            response="Experience with Docker but not specifically with Kubernetes.",
            retrieved_contexts=["irrelevant"], user_input="Backend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"backend-engineer": _jd_skills("backend-engineer", ["Docker", "Kubernetes"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Docker", "Kubernetes"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    flagged_skills = [c["contradicted_skill"] for c in contradictions]
    assert flagged_skills == ["Kubernetes"]


def test_find_weakness_contradictions_does_not_flag_a_skill_reintroduced_after_a_contrast_word():
    rows = [
        FaithfulnessRow(
            jd_id="backend-engineer", candidate_id="cv-001", item_type="weakness",
            response="Lack of direct mention of TypeScript, though proficiency in Node.js is evident.",
            retrieved_contexts=["irrelevant"], user_input="Backend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"backend-engineer": _jd_skills("backend-engineer", ["TypeScript", "Node.js"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Node.js"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_does_not_flag_a_skill_reintroduced_after_only():
    rows = [
        FaithfulnessRow(
            jd_id="backend-engineer", candidate_id="cv-001", item_type="weakness",
            response="No direct experience with Docker, only Kubernetes.",
            retrieved_contexts=["irrelevant"], user_input="Backend Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"backend-engineer": _jd_skills("backend-engineer", ["Docker", "Kubernetes"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Kubernetes"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_does_not_flag_a_category_term_preceded_by_for():
    rows = [
        FaithfulnessRow(
            jd_id="devops-engineer", candidate_id="cv-001", item_type="weakness",
            response="No explicit experience or mention of Terraform for Infrastructure as Code.",
            retrieved_contexts=["irrelevant"], user_input="Devops Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"devops-engineer": _jd_skills("devops-engineer", ["Terraform", "Infrastructure as Code"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Infrastructure as Code"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_does_not_flag_a_category_term_followed_by_based():
    rows = [
        FaithfulnessRow(
            jd_id="devops-engineer", candidate_id="cv-001", item_type="weakness",
            response="No explicit mention of hands-on experience with GitOps-based delivery tools such as ArgoCD or Flux.",
            retrieved_contexts=["irrelevant"], user_input="Devops Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"devops-engineer": _jd_skills("devops-engineer", ["GitOps", "ArgoCD", "Flux"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["GitOps"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    assert contradictions == []


def test_find_weakness_contradictions_still_flags_a_specific_tool_next_to_a_category_term():
    rows = [
        FaithfulnessRow(
            jd_id="devops-engineer", candidate_id="cv-001", item_type="weakness",
            response="No explicit experience or mention of Terraform for Infrastructure as Code.",
            retrieved_contexts=["irrelevant"], user_input="Devops Engineer — weakness",
        )
    ]
    jd_skills_by_id = {"devops-engineer": _jd_skills("devops-engineer", ["Terraform", "Infrastructure as Code"])}
    candidates_by_id = {"cv-001": _candidate("cv-001", ["Terraform"])}

    contradictions = find_weakness_contradictions(rows, jd_skills_by_id, candidates_by_id, _fake_skill_embedder)

    flagged_skills = [c["contradicted_skill"] for c in contradictions]
    assert flagged_skills == ["Terraform"]


def test_run_weakness_contradiction_check_loads_jd_and_candidate_skills_and_flags_contradictions(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, _JD_TITLE)
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]

    _write_manifest(cfg.runs_dir, "run-1", ["frontend-engineer"])
    assessment = Assessment(
        job_description_id="frontend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Built things."], weaknesses=["No explicit mention of experience with React."],
    )
    _write_assessment(cfg.runs_dir, "run-1", "frontend-engineer", assessment)

    contradictions = run_weakness_contradiction_check(
        cfg, "run-1",
        jd_skills_chain=_FakeJDSkillsChain(),
        skill_extraction_chain=_FakeCvSkillsChain(),
        embedder=_fake_skill_embedder,
    )

    assert len(contradictions) == 1
    assert contradictions[0]["candidate_id"] == candidate.id
    assert contradictions[0]["contradicted_skill"] == "React"


def test_run_weakness_contradiction_check_returns_empty_list_with_no_weaknesses(tmp_path):
    cfg = _fake_cfg(tmp_path)
    _write_jd(cfg.jd_dir, _JD_TITLE)
    _write_real_cv(cfg.cv_dir)
    candidate = load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")[0]

    _write_manifest(cfg.runs_dir, "run-1", ["frontend-engineer"])
    assessment = Assessment(
        job_description_id="frontend-engineer", candidate_id=candidate.id, generated_by_model="fake-model",
        strengths=["Built things."], weaknesses=[],
    )
    _write_assessment(cfg.runs_dir, "run-1", "frontend-engineer", assessment)

    contradictions = run_weakness_contradiction_check(
        cfg, "run-1",
        jd_skills_chain=_FakeJDSkillsChain(),
        skill_extraction_chain=_FakeCvSkillsChain(),
        embedder=_fake_skill_embedder,
    )

    assert contradictions == []
