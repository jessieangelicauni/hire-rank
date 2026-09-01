
import json
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from candidate_ranking import cli
from candidate_ranking.scoring.assessment import ASSESSMENT_SCOPE_VERSION, _GeneratedAssessment
from candidate_ranking.config import ENV_OVERRIDE_VARS, RunConfig
from candidate_ranking.scoring.jd_skills import _GeneratedSkills
from candidate_ranking.models import Assessment, Candidate, JobDescription
from candidate_ranking.ranking.tournament import RANKING_PROMPT_VERSION


@pytest.fixture(autouse=True)
def _isolate_from_real_env(monkeypatch):
    for var in ENV_OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)


JD_MATCHED = JobDescription(
    id="backend-engineer", title="Backend Engineer", raw_text="Build APIs.", source_path="/jd/be.txt"
)
JD_UNMATCHED = JobDescription(
    id="frontend-engineer", title="Frontend Engineer", raw_text="Build UIs.", source_path="/jd/fe.txt"
)

C1 = Candidate(
    id="cv-001", source_path="/cv/1.pdf", raw_text="Jane: Python.", num_pages=1, char_count=13,
    parse_status="ok", skills=["shared-skill"],
)
C2 = Candidate(
    id="cv-002", source_path="/cv/2.pdf", raw_text="Sam: React.", num_pages=1, char_count=11,
    parse_status="ok", skills=["shared-skill"],
)

_SKILL_VOCAB = ["shared-skill", "unmatched-jd-skill"]


def _one_hot(word: str) -> np.ndarray:
    vec = np.zeros(len(_SKILL_VOCAB), dtype="float32")
    vec[_SKILL_VOCAB.index(word)] = 1.0
    return vec


def _fake_skill_embedder(texts: list[str]) -> np.ndarray:
    return np.stack([_one_hot(t) for t in texts])


def _complete_result() -> _GeneratedAssessment:
    return _GeneratedAssessment(reasoning="fake reasoning", strengths=["Relevant experience noted."], weaknesses=[])


class _FakeJDSkillsChain:

    def invoke(self, payload):
        if payload["title"] == JD_UNMATCHED.title:
            return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["unmatched-jd-skill"])
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["shared-skill"])


class _FakeJDSkillsChainWithOneEmptyResult:

    def invoke(self, payload):
        if payload["title"] == JD_UNMATCHED.title:
            return _GeneratedSkills(reasoning="fake reasoning", technical_skills=[])
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["shared-skill"])


class _FakeAssessmentChain:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def invoke(self, _payload):
        with self._lock:
            self.calls += 1
        return _complete_result()


class _OrderPreservingRankingChain:
    def __init__(self, ranking_model):
        self._ranking_model = ranking_model

    def invoke(self, payload):
        import re

        ids = re.findall(r"(?m)^(\d+):", payload["candidates_text"])
        return self._ranking_model(reasoning="fake reasoning", ranking=ids)


def _fake_cfg(tmp_path: Path, preset: str = "full") -> RunConfig:
    return RunConfig(
        preset=preset,
        jd_dir=tmp_path / "jd",
        cv_dir=tmp_path / "cv",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        max_jds=None,
        max_candidates=None,
        tournament_iterations=1,
        stability_repeats=1,
        tournament_subset_size=1,
        num_subset_samples=5,
        num_mc_draws=5,
        pl_prior_variance=1.0,
        ollama_model="fake-model",
        ollama_base_url="http://localhost:11434",
        ollama_num_parallel=4,
        skill_embedding_model="fake-embed-model",
    )


def _patch_expensive_dependencies(monkeypatch, tmp_path, jds, candidates, preset: str = "full"):
    fake_cfg = _fake_cfg(tmp_path, preset)
    monkeypatch.setattr(RunConfig, "full", classmethod(lambda cls, root: fake_cfg))

    monkeypatch.setattr(cli, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(cli, "build_jd_skills_chain", lambda llm: _FakeJDSkillsChain())
    monkeypatch.setattr(cli, "build_assessment_chain", lambda llm: _FakeAssessmentChain())
    monkeypatch.setattr(
        cli, "build_listwise_ranking_chain", lambda llm: (lambda ranking_model: _OrderPreservingRankingChain(ranking_model))
    )
    monkeypatch.setattr(cli, "build_skill_extraction_chain", lambda llm: object())
    monkeypatch.setattr(cli, "build_skill_embedder", lambda model_name: _fake_skill_embedder)
    monkeypatch.setattr(
        cli,
        "enrich_candidates_with_skills",
        lambda cands, chain, model_name, cache_path, max_workers=1: cands,
    )
    from candidate_ranking.scoring.skills import build_candidate_skill_index as real_build_candidate_skill_index

    monkeypatch.setattr(cli, "build_candidate_skill_index", real_build_candidate_skill_index)
    monkeypatch.setattr(cli, "load_corpus", lambda cfg: (jds, candidates))
    monkeypatch.setattr(cli, "seed_console_web_roles", lambda cfg: None)

    return fake_cfg


def test_run_constructs_chat_ollama_with_the_configured_context_window(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])
    captured_kwargs = {}

    def _capturing_chat_ollama(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "ChatOllama", _capturing_chat_ollama)

    cli.run(run_id="test-run-num-ctx", min_skill_matches=1)

    assert captured_kwargs["num_ctx"] == fake_cfg.ollama_num_ctx


def test_run_writes_manifest_with_skill_shortlist_fields(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])

    run_dir = cli.run(run_id="test-run-manifest", min_skill_matches=1)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skill_embedding_model"] == fake_cfg.skill_embedding_model
    assert manifest["skill_match_threshold"] == 0.8
    assert manifest["min_skill_matches"] == 1
    assert manifest["shortlist_sizes"] == {"backend-engineer": 2}
    assert manifest["ranking_prompt_version"] == RANKING_PROMPT_VERSION
    assert manifest["assessment_scope"] == ASSESSMENT_SCOPE_VERSION


def test_run_raises_when_total_shortlisted_is_zero_but_still_writes_output(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_UNMATCHED], [C1, C2])

    with pytest.raises(RuntimeError, match="zero shortlisted candidates"):
        cli.run(run_id="test-run-zero-shortlist", min_skill_matches=1)

    run_dir = fake_cfg.runs_dir / "test-run-zero-shortlist"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shortlist_sizes"] == {"frontend-engineer": 0}
    assert (run_dir / "summary.json").exists()


def test_run_warns_per_jd_for_empty_shortlist_even_when_total_nonzero(monkeypatch, tmp_path, capsys):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED, JD_UNMATCHED], [C1, C2])

    run_dir = cli.run(run_id="test-run-partial-empty", min_skill_matches=1)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shortlist_sizes"] == {"backend-engineer": 2, "frontend-engineer": 0}

    stderr = capsys.readouterr().err
    assert "frontend-engineer" in stderr
    assert "empty" in stderr.lower()


def test_run_excludes_a_jd_whose_skill_extraction_fails_without_crashing_the_run(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED, JD_UNMATCHED], [C1, C2])
    monkeypatch.setattr(cli, "build_jd_skills_chain", lambda llm: _FakeJDSkillsChainWithOneEmptyResult())

    run_dir = cli.run(run_id="test-run-skills-failure", min_skill_matches=1)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shortlist_sizes"] == {"backend-engineer": 2}
    assert (run_dir / "backend-engineer" / "ranking.md").exists()
    assert not (run_dir / "frontend-engineer").exists()


def test_run_resume_records_original_thresholds_not_freshly_passed_ones(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])

    first_run_dir = cli.run(
        run_id="test-run-resume", skill_match_threshold=0.8, min_skill_matches=1
    )
    first_manifest = json.loads((first_run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["skill_match_threshold"] == 0.8
    assert first_manifest["min_skill_matches"] == 1

    second_run_dir = cli.run(
        run_id="test-run-resume", skill_match_threshold=0.95, min_skill_matches=5
    )
    second_manifest = json.loads((second_run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["skill_match_threshold"] == 0.8
    assert second_manifest["min_skill_matches"] == 1


def test_sqlite_checkpointer_serde_allows_own_model_types(tmp_path, caplog):
    assessment = Assessment(
        job_description_id="jd",
        candidate_id="c",
        generated_by_model="m",
        strengths=["Relevant experience."],
        weaknesses=[],
    )
    with cli.sqlite_checkpointer(tmp_path / "checkpoints.db") as saver:
        with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
            typed = saver.serde.dumps_typed(assessment)
            saver.serde.loads_typed(typed)

    assert "Deserializing unregistered type" not in caplog.text


def test_run_writes_ranking_files_per_jd(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])

    run_dir = cli.run(run_id="test-run-ranking", min_skill_matches=1)

    assert (run_dir / "backend-engineer" / "ranking.md").exists()
    assert (run_dir / "backend-engineer" / "ranking.json").exists()


def test_run_writes_one_assessments_file_per_jd(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])

    run_dir = cli.run(run_id="test-run-assessments", min_skill_matches=1)

    assessments_path = run_dir / "backend-engineer" / "assessments.json"
    assert assessments_path.exists()
    data = json.loads(assessments_path.read_text(encoding="utf-8"))
    assert data["cv-001"]["strengths"] == ["Relevant experience noted."]
    assert data["cv-001"]["weaknesses"] == []


def test_run_applies_candidate_ranking_env_overrides_on_top_of_preset(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2])
    monkeypatch.setenv("CANDIDATE_RANKING_SKILL_EMBEDDING_MODEL", "env-override-embed-model")

    run_dir = cli.run(run_id="test-run-env-override", min_skill_matches=1)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skill_embedding_model"] == "env-override-embed-model"
    assert manifest["skill_embedding_model"] != fake_cfg.skill_embedding_model


def test_run_exports_console_web_data_for_full_preset(monkeypatch, tmp_path):
    fake_cfg = _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2], preset="full")
    export_calls = []

    def _fake_export(cfg, run_id):
        export_calls.append((cfg, run_id))
        return Path("fake-real-data.json")

    monkeypatch.setattr(cli, "export_console_web_data", _fake_export)

    cli.run(run_id="export-wiring-full")

    assert export_calls == [(fake_cfg, "export-wiring-full")]


def test_run_survives_console_web_export_raising(monkeypatch, tmp_path, capsys):
    _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2], preset="full")

    def _boom(cfg, run_id):
        raise RuntimeError("export blew up")

    monkeypatch.setattr(cli, "export_console_web_data", _boom)

    run_dir = cli.run(run_id="export-wiring-raises")

    assert run_dir.exists()
    captured = capsys.readouterr()
    assert "console-web export failed" in captured.err
    assert "export blew up" in captured.err


def test_run_seeds_console_web_roles_before_building_pipeline_for_full_preset(monkeypatch, tmp_path):
    _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2], preset="full")
    call_order = []

    def _spy_load_corpus(cfg):
        call_order.append("load_corpus")
        return [JD_MATCHED], [C1, C2]

    monkeypatch.setattr(cli, "load_corpus", _spy_load_corpus)

    def _spy_seed(cfg):
        call_order.append("seed_console_web_roles")
        return Path("fake-real-data.json")

    monkeypatch.setattr(cli, "seed_console_web_roles", _spy_seed)

    def _spy_chat_ollama(**kwargs):
        call_order.append("ChatOllama")
        return object()

    monkeypatch.setattr(cli, "ChatOllama", _spy_chat_ollama)

    cli.run(run_id="seed-order-full")

    assert call_order == ["load_corpus", "seed_console_web_roles", "ChatOllama"]


def test_run_survives_console_web_seed_raising(monkeypatch, tmp_path, capsys):
    _patch_expensive_dependencies(monkeypatch, tmp_path, [JD_MATCHED], [C1, C2], preset="full")

    def _boom(cfg):
        raise RuntimeError("seed blew up")

    monkeypatch.setattr(cli, "seed_console_web_roles", _boom)

    run_dir = cli.run(run_id="seed-wiring-raises")

    assert run_dir.exists()
    captured = capsys.readouterr()
    assert "console-web role seeding failed" in captured.err
    assert "seed blew up" in captured.err
