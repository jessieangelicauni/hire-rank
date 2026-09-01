
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from candidate_ranking import cli
from candidate_ranking.output import console_export
from candidate_ranking.scoring.assessment import _GeneratedAssessment
from candidate_ranking.config import ENV_OVERRIDE_VARS, RunConfig
from candidate_ranking.scoring.jd_skills import _GeneratedSkills
from candidate_ranking.ingestion.cv import _ExtractedName
from candidate_ranking.models import Candidate, JobDescription

REAL_CV_DIR = Path(__file__).resolve().parents[1] / "cv"


@pytest.fixture(autouse=True)
def _isolate_from_real_env(monkeypatch):
    for var in ENV_OVERRIDE_VARS:
        monkeypatch.delenv(var, raising=False)


JD1 = JobDescription(
    id="backend-engineer", title="Backend Engineer", raw_text="Build APIs.", source_path="/jd/be.txt"
)
JD2 = JobDescription(
    id="frontend-engineer", title="Frontend Engineer", raw_text="Build UIs.", source_path="/jd/fe.txt"
)
JD_ALPHA = JobDescription(
    id="alpha-engineer", title="Alpha Engineer", raw_text="Build widgets.", source_path="/jd/ae.txt"
)

C1 = Candidate(
    id="cv-00201", source_path=str(REAL_CV_DIR / "cv_00201.pdf"), raw_text="Jane: Python.", num_pages=1,
    char_count=13, parse_status="ok", skills=["shared-skill"],
)
C2 = Candidate(
    id="cv-00202", source_path=str(REAL_CV_DIR / "cv_00202.pdf"), raw_text="John: Python.", num_pages=1,
    char_count=13, parse_status="ok", skills=["shared-skill"],
)

_SKILL_VOCAB = ["shared-skill", "unmatched-skill"]


def _one_hot(word: str) -> np.ndarray:
    vec = np.zeros(len(_SKILL_VOCAB), dtype="float32")
    vec[_SKILL_VOCAB.index(word)] = 1.0
    return vec


def _fake_skill_embedder(texts: list[str]) -> np.ndarray:
    return np.stack([_one_hot(t) for t in texts])


class _FakeJDSkillsChain:
    def invoke(self, payload):
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["shared-skill"])


class _FakeJDSkillsChainWithUnmatchedJD:

    def invoke(self, payload):
        if payload["title"] == "Frontend Engineer":
            return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["unmatched-skill"])
        return _GeneratedSkills(reasoning="fake reasoning", technical_skills=["shared-skill"])


class _FakeAssessmentChain:
    def __init__(self):
        self._lock = threading.Lock()

    def invoke(self, payload):
        with self._lock:
            pass
        return _GeneratedAssessment(reasoning="fake reasoning", strengths=["Relevant experience noted."], weaknesses=[])


class _OrderPreservingRankingChain:
    def __init__(self, ranking_model):
        self._ranking_model = ranking_model

    def invoke(self, payload):
        import re

        ids = re.findall(r"(?m)^(\d+):", payload["candidates_text"])
        return self._ranking_model(reasoning="fake reasoning", ranking=ids)


class _FakeNameChain:
    def invoke(self, payload):
        return _ExtractedName(reasoning="fake reasoning", name="Jane Doe")


def _fake_cfg(tmp_path: Path, preset: str = "full") -> RunConfig:
    return RunConfig(
        preset=preset, jd_dir=tmp_path / "jd", cv_dir=tmp_path / "cv", cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs", max_jds=None, max_candidates=None, tournament_iterations=2,
        stability_repeats=2, tournament_subset_size=2, num_subset_samples=3, num_mc_draws=3,
        pl_prior_variance=1.0, ollama_model="fake-model", ollama_base_url="http://localhost:11434",
        ollama_num_parallel=4, skill_embedding_model="fake-embed-model",
    )


def _patch_pipeline(monkeypatch, tmp_path, jds, candidates, preset: str = "full") -> RunConfig:
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
        cli, "enrich_candidates_with_skills", lambda cands, chain, model_name, cache_path, max_workers=1: cands
    )
    from candidate_ranking.scoring.skills import build_candidate_skill_index as real_build_candidate_skill_index

    monkeypatch.setattr(cli, "build_candidate_skill_index", real_build_candidate_skill_index)
    monkeypatch.setattr(cli, "load_corpus", lambda cfg: (jds, candidates))
    return fake_cfg


def _patch_export_llm(monkeypatch):
    monkeypatch.setattr(console_export, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(console_export, "build_name_extraction_chain", lambda llm: _FakeNameChain())


def _write_jd_file(jd_dir: Path, filename: str, title: str, body: str) -> None:
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / filename).write_text(f"{title}\n{body}\n", encoding="utf-8")


def _write_real_cv(cv_dir: Path) -> None:
    cv_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(REAL_CV_DIR / "cv_00201.pdf", cv_dir / "cv_00201.pdf")


def test_export_constructs_chat_ollama_with_the_configured_context_window(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1])
    cli.run(run_id="export-num-ctx", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    monkeypatch.setattr(console_export, "build_name_extraction_chain", lambda llm: _FakeNameChain())
    captured_kwargs = {}

    def _capturing_chat_ollama(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(console_export, "ChatOllama", _capturing_chat_ollama)

    console_export.export_console_web_data(fake_cfg, "export-num-ctx", output_path=tmp_path / "out.json")

    assert captured_kwargs["num_ctx"] == fake_cfg.ollama_num_ctx


def test_export_writes_roles_candidates_assessments_and_comparison(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1])
    cli.run(run_id="export-normal", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    output_path = tmp_path / "real-data.json"
    result = console_export.export_console_web_data(fake_cfg, "export-normal", output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["backend-engineer"]
    assert data["roles"][0]["title"] == "Backend Engineer"
    assert data["roles"][0]["candidateCount"] == 1
    assert [c["id"] for c in data["candidates"]] == ["cv-00201::backend-engineer"]
    assert data["candidates"][0]["name"] == "Jane Doe"
    assert data["candidates"][0]["rank"] == 1
    assert "groundedness" not in data["candidates"][0]
    assert data["assessments"]["cv-00201::backend-engineer"] == {
        "strengths": ["Relevant experience noted."], "weaknesses": [], "additional_skills": [],
    }
    assert data["comparison"]["backend-engineer"] == {
        "jdId": "backend-engineer",
        "kendallTau": None,
        "deltaU": None,
        "faithfulness": None,
    }


def test_export_picks_up_faithfulness_report_when_present(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1])
    cli.run(run_id="export-faithfulness", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    report_path = fake_cfg.runs_dir / "export-faithfulness" / "ragas_faithfulness_report.json"
    report_path.write_text(json.dumps({"per_jd": {"backend-engineer": 0.7}}), encoding="utf-8")

    output_path = tmp_path / "real-data.json"
    console_export.export_console_web_data(fake_cfg, "export-faithfulness", output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["comparison"]["backend-engineer"]["faithfulness"] == 0.7


def test_export_ignores_a_malformed_faithfulness_report(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1])
    cli.run(run_id="export-faithfulness-bad", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    report_path = fake_cfg.runs_dir / "export-faithfulness-bad" / "ragas_faithfulness_report.json"
    report_path.write_text("not valid json", encoding="utf-8")

    output_path = tmp_path / "real-data.json"
    console_export.export_console_web_data(fake_cfg, "export-faithfulness-bad", output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["comparison"]["backend-engineer"]["faithfulness"] is None

    warnings_log = json.loads((fake_cfg.runs_dir / "export-faithfulness-bad" / "warnings.json").read_text(encoding="utf-8"))
    assert any("ragas" in entry["message"].lower() for entry in warnings_log)


def test_export_skips_candidate_removed_from_cv_dir(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1, C2])
    cli.run(run_id="export-candidate-removed", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    output_path = tmp_path / "real-data.json"
    result = console_export.export_console_web_data(fake_cfg, "export-candidate-removed", output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["backend-engineer"]
    assert data["roles"][0]["candidateCount"] == 1
    assert [c["id"] for c in data["candidates"]] == ["cv-00201::backend-engineer"]
    assert list(data["assessments"].keys()) == ["cv-00201::backend-engineer"]


def test_export_returns_none_when_run_directory_has_no_jd_subdirectories(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)
    empty_run_dir = fake_cfg.runs_dir / "no-assessments-run"
    empty_run_dir.mkdir(parents=True)
    (empty_run_dir / "manifest.json").write_text(json.dumps({}), encoding="utf-8")

    result = console_export.export_console_web_data(
        fake_cfg, "no-assessments-run", output_path=tmp_path / "real-data.json"
    )

    assert result is None
    assert not (tmp_path / "real-data.json").exists()


def test_export_returns_none_when_run_directory_does_not_exist(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)

    result = console_export.export_console_web_data(
        fake_cfg, "no-such-run", output_path=tmp_path / "real-data.json"
    )

    assert result is None
    assert not (tmp_path / "real-data.json").exists()


def test_export_excludes_a_jd_whose_file_was_removed_from_job_description(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1, JD2], [C1])
    cli.run(run_id="export-deleted-jd", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    output_path = tmp_path / "real-data.json"
    result = console_export.export_console_web_data(fake_cfg, "export-deleted-jd", output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["backend-engineer"]
    assert all(c["roleId"] == "backend-engineer" for c in data["candidates"])
    assert "frontend-engineer" not in data["comparison"]


def test_export_keeps_a_role_with_an_empty_shortlist_at_zero_candidates(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1, JD2], [C1])
    monkeypatch.setattr(cli, "build_jd_skills_chain", lambda llm: _FakeJDSkillsChainWithUnmatchedJD())
    cli.run(run_id="export-empty-shortlist", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_jd_file(fake_cfg.jd_dir, "Frontend-Engineer.txt", "Frontend Engineer", "Build UIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    output_path = tmp_path / "real-data.json"
    result = console_export.export_console_web_data(fake_cfg, "export-empty-shortlist", output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    role_ids = [r["id"] for r in data["roles"]]
    assert "backend-engineer" in role_ids
    assert "frontend-engineer" in role_ids
    frontend_role = next(r for r in data["roles"] if r["id"] == "frontend-engineer")
    assert frontend_role["candidateCount"] == 0
    assert not any(c["roleId"] == "frontend-engineer" for c in data["candidates"])
    assert data["comparison"]["frontend-engineer"] == {
        "jdId": "frontend-engineer", "kendallTau": None, "deltaU": None, "faithfulness": None,
    }


def test_export_orders_unprocessed_jd_by_corpus_position_not_appended_last(monkeypatch, tmp_path):
    fake_cfg = _patch_pipeline(monkeypatch, tmp_path, [JD1], [C1])
    cli.run(run_id="export-order-fix", min_skill_matches=1)

    _write_jd_file(fake_cfg.jd_dir, "Alpha-Engineer.txt", JD_ALPHA.title, JD_ALPHA.raw_text)
    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_real_cv(fake_cfg.cv_dir)
    _patch_export_llm(monkeypatch)

    output_path = tmp_path / "real-data.json"
    result = console_export.export_console_web_data(fake_cfg, "export-order-fix", output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["alpha-engineer", "backend-engineer"]


def test_seed_writes_a_role_for_every_jd_when_output_missing(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)
    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_jd_file(fake_cfg.jd_dir, "Frontend-Engineer.txt", "Frontend Engineer", "Build UIs.")

    output_path = tmp_path / "real-data.json"
    result = console_export.seed_console_web_roles(fake_cfg, output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["backend-engineer", "frontend-engineer"]
    assert data["roles"][0]["title"] == "Backend Engineer"
    assert data["roles"][0]["candidateCount"] == 0
    assert data["candidates"] == []
    assert data["assessments"] == {}
    assert data["comparison"] == {
        "backend-engineer": {
            "jdId": "backend-engineer", "kendallTau": None, "deltaU": None, "faithfulness": None,
        },
        "frontend-engineer": {
            "jdId": "frontend-engineer", "kendallTau": None, "deltaU": None, "faithfulness": None,
        },
    }


def test_seed_preserves_existing_role_with_real_candidate_data(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)
    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")

    output_path = tmp_path / "real-data.json"
    existing = {
        "roles": [
            {"id": "backend-engineer", "title": "Backend Engineer", "description": "Build reliable APIs.", "candidateCount": 1}
        ],
        "candidates": [
            {
                "id": "cv-00201::backend-engineer", "cvId": "cv-00201", "roleId": "backend-engineer",
                "name": "Jane Doe", "initials": "JD", "rank": 1, "utility": 0.9,
            }
        ],
        "assessments": {
            "cv-00201::backend-engineer": {"strengths": ["Strong Python background."], "weaknesses": []}
        },
        "comparison": {
            "backend-engineer": {
                "jdId": "backend-engineer", "kendallTau": None, "stability": 1.0, "deltaU": None, "faithfulness": None,
            }
        },
    }
    output_path.write_text(json.dumps(existing), encoding="utf-8")

    result = console_export.seed_console_web_roles(fake_cfg, output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["roles"] == existing["roles"]
    assert data["candidates"] == existing["candidates"]
    assert data["assessments"] == existing["assessments"]
    assert data["comparison"] == existing["comparison"]


def test_seed_backfills_null_comparison_for_new_jd_without_touching_existing(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)
    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")
    _write_jd_file(fake_cfg.jd_dir, "Frontend-Engineer.txt", "Frontend Engineer", "Build UIs.")

    output_path = tmp_path / "real-data.json"
    existing_comparison_entry = {
        "jdId": "backend-engineer", "kendallTau": None, "stability": 1.0, "deltaU": None, "faithfulness": None,
    }
    existing = {
        "roles": [
            {"id": "backend-engineer", "title": "Backend Engineer", "description": "Build reliable APIs.", "candidateCount": 1}
        ],
        "candidates": [],
        "assessments": {},
        "comparison": {"backend-engineer": existing_comparison_entry},
    }
    output_path.write_text(json.dumps(existing), encoding="utf-8")

    result = console_export.seed_console_web_roles(fake_cfg, output_path=output_path)

    assert result == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["comparison"]["backend-engineer"] == existing_comparison_entry
    assert data["comparison"]["frontend-engineer"] == {
        "jdId": "frontend-engineer", "kendallTau": None, "deltaU": None, "faithfulness": None,
    }


def test_seed_drops_role_whose_jd_file_was_removed(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)
    _write_jd_file(fake_cfg.jd_dir, "Backend-Engineer.txt", "Backend Engineer", "Build reliable APIs.")

    output_path = tmp_path / "real-data.json"
    existing = {
        "roles": [
            {"id": "backend-engineer", "title": "Backend Engineer", "description": "...", "candidateCount": 1},
            {"id": "frontend-engineer", "title": "Frontend Engineer", "description": "...", "candidateCount": 2},
        ],
        "candidates": [],
        "assessments": {},
        "comparison": {},
    }
    output_path.write_text(json.dumps(existing), encoding="utf-8")

    result = console_export.seed_console_web_roles(fake_cfg, output_path=output_path)

    data = json.loads(result.read_text(encoding="utf-8"))
    assert [r["id"] for r in data["roles"]] == ["backend-engineer"]


def test_seed_leaves_output_untouched_when_jd_dir_has_no_jds(tmp_path):
    fake_cfg = _fake_cfg(tmp_path)

    output_path = tmp_path / "real-data.json"
    existing = {
        "roles": [
            {"id": "backend-engineer", "title": "Backend Engineer", "description": "Build reliable APIs.", "candidateCount": 1}
        ],
        "candidates": [
            {
                "id": "cv-00201::backend-engineer", "cvId": "cv-00201", "roleId": "backend-engineer",
                "name": "Jane Doe", "initials": "JD", "rank": 1, "utility": 0.9,
            }
        ],
        "assessments": {
            "cv-00201::backend-engineer": {"strengths": ["Strong Python background."], "weaknesses": []}
        },
        "comparison": {
            "backend-engineer": {
                "jdId": "backend-engineer", "kendallTau": None, "stability": 1.0, "deltaU": None, "faithfulness": None,
            }
        },
    }
    existing_bytes = json.dumps(existing).encode("utf-8")
    output_path.write_bytes(existing_bytes)

    result = console_export.seed_console_web_roles(fake_cfg, output_path=output_path)

    assert result == output_path
    assert output_path.read_bytes() == existing_bytes
