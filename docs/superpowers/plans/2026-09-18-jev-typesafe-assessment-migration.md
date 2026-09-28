# Jev Typesafe Assessment Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free-text LLM assessment-generation flow with Jev (TypeSafe AI's structured Score/Choice/Noul model, via Cloudflare Workers AI), and simplify candidate ranking to a direct sort by Jev's score instead of an LLM tournament.

**Architecture:** A new thin `JevClient` HTTP wrapper (no LangChain) replaces the `ChatOllama` assessment chain. `Assessment` becomes a structured-score schema (no more `strengths`/`weaknesses`/`reasoning`). The pipeline drops its tournament stage entirely and instead sorts each JD's candidates by `overall_fit_score` directly. `jd_skills_chain` and `skill_extraction_chain` continue to run on Ollama, unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, `requests` (new direct dependency), LangGraph (pipeline orchestration, unchanged), pytest (new `tests/` tree — none currently exists in this repo).

## Global Constraints

- Never hardcode a Cloudflare account ID or API token anywhere (code, tests, fixtures, docs). Read both only from `RunConfig.cf_account_id` / `RunConfig.cf_api_token`, populated via `CANDIDATE_RANKING_CF_ACCOUNT_ID` / `CANDIDATE_RANKING_CF_API_TOKEN` environment variables.
- Do not modify `src/candidate_ranking/injection/` (attacks.py, defenses.py, measurement.py) — the active rank-shift research paper depends on the old flow; it will go stale and is adapted separately later.
- Do not modify `src/candidate_ranking/evaluation/ragas_eval.py` — its faithfulness metric is conceptually tied to free-text claims and no longer applies; leave in place.
- Do not delete `src/candidate_ranking/ranking/tournament.py`, `mc_kg.py`, or `plackett_luce.py` — stop calling them from the pipeline, but keep the files since the injection research still uses them standalone.
- Do not modify `src/candidate_ranking/scoring/skills.py` — `find_negated_skill_mentions` / `bridge_negated_jd_skills_to_candidate` stay as-is; `ragas_eval.py` still depends on the former.
- All new/changed Python files use `from __future__ import annotations` at the top, matching every existing file in this package.
- Run tests with `.venv/bin/python -m pytest <path> -v` from the repo root (`/home/jessie/AXPilot-Internal`) — the package is installed editable, so `import candidate_ranking...` works without a `src` path hack.

---

### Task 1: Jev HTTP Client

**Files:**
- Create: `src/candidate_ranking/scoring/jev_client.py`
- Modify: `pyproject.toml` (add `requests` dependency)
- Test: `tests/scoring/test_jev_client.py`

**Interfaces:**
- Produces:
  - `class JevQuestion(BaseModel)`: fields `key: str`, `kind: Literal["noul", "choice", "score"]`, `instructions: str`, `criteria: dict[str, str] | list[str]`
  - `class JevAnswer(BaseModel)`: fields `key: str`, `kind: Literal["noul", "choice", "score"]`, `value: bool | str | float`, `confidence: float`
  - `class JevClientError(Exception)`
  - `class JevClient`: `__init__(self, account_id: str, api_token: str, timeout: float = 30.0) -> None`; `evaluate(self, state: str, questions: list[JevQuestion]) -> list[JevAnswer]`

- [ ] **Step 1: Add `requests` to `pyproject.toml` dependencies**

In `pyproject.toml`, add `"requests>=2.32.0",` to the `dependencies` list (alongside `"pypdf>=5.1.0",` etc.):

```toml
dependencies = [
    "langgraph>=0.2.60",
    "langgraph-checkpoint-sqlite>=2.0.1",
    "langchain>=0.3.0",
    "langchain-ollama>=0.2.0",
    "pydantic>=2.9.0",
    "pypdf>=5.1.0",
    "requests>=2.32.0",
    "torch>=2.13.0",
    "sentence-transformers>=5.7.0",
    "transformers>=4.44.0",
    "numpy>=2.5.2",
    "scipy>=1.14.0",
    "python-dotenv>=1.2.3",
    "python-docx==1.1.0",
    "dspy>=2.6.0,<4",
    "nltk>=3.10.3",
]
```

Run: `.venv/bin/python -c "import requests; print(requests.__version__)"`
Expected: prints a version string (requests is already present transitively; this just makes it an explicit direct dependency).

- [ ] **Step 2: Write the failing test for a successful `evaluate()` call**

Create `tests/scoring/test_jev_client.py`:

```python
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from candidate_ranking.scoring.jev_client import JevAnswer, JevClient, JevClientError, JevQuestion


def _fake_response(answers: dict) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value={"model": "jev-1.13.0", "answers": answers, "usage": {}})
    return response


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_parses_noul_choice_and_score_answers(mock_post):
    mock_post.return_value = _fake_response(
        {
            "meets_min_qualifications": {"type": "noul", "noul": 0.95},
            "overall_recommendation": {
                "type": "choice",
                "choice": "hire",
                "confidence": 0.8,
                "probabilities": {"hire": 0.8, "maybe": 0.15, "no": 0.05},
            },
            "overall_fit_score": {
                "type": "score",
                "score": 3.0,
                "confidence": 0.9,
                "legend": {"0": "0", "1": "25", "2": "50", "3": "75", "4": "100"},
                "probabilities": {"0": 0, "1": 0, "2": 0.1, "3": 0.9, "4": 0},
            },
        }
    )

    client = JevClient(account_id="acc123", api_token="token123")
    questions = [
        JevQuestion(key="meets_min_qualifications", kind="noul", instructions="...", criteria={"true": "...", "false": "..."}),
        JevQuestion(key="overall_recommendation", kind="choice", instructions="...", criteria={"hire": "...", "maybe": "...", "no": "..."}),
        JevQuestion(key="overall_fit_score", kind="score", instructions="...", criteria=["0", "25", "50", "75", "100"]),
    ]

    answers = client.evaluate("some state text", questions)

    by_key = {a.key: a for a in answers}
    assert by_key["meets_min_qualifications"] == JevAnswer(
        key="meets_min_qualifications", kind="noul", value=True, confidence=0.95
    )
    assert by_key["overall_recommendation"] == JevAnswer(
        key="overall_recommendation", kind="choice", value="hire", confidence=0.8
    )
    assert by_key["overall_fit_score"] == JevAnswer(
        key="overall_fit_score", kind="score", value=3.0, confidence=0.9
    )

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer token123"
    assert call_kwargs["json"]["model"] == "typesafe/jev"
    assert call_kwargs["json"]["input"]["state"] == "some state text"
    assert set(call_kwargs["json"]["input"]["questions"]) == {
        "meets_min_qualifications", "overall_recommendation", "overall_fit_score",
    }


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_raises_jev_client_error_on_http_failure(mock_post):
    import requests

    mock_post.side_effect = requests.ConnectionError("boom")
    client = JevClient(account_id="acc123", api_token="token123")

    with pytest.raises(JevClientError, match="boom"):
        client.evaluate("state", [JevQuestion(key="k", kind="noul", instructions="i", criteria={"true": "t", "false": "f"})])


@patch("candidate_ranking.scoring.jev_client.requests.post")
def test_evaluate_raises_jev_client_error_on_missing_answers_key(mock_post):
    mock_post.return_value = _fake_response({})
    mock_post.return_value.json = Mock(return_value={"model": "jev-1.13.0"})
    client = JevClient(account_id="acc123", api_token="token123")

    with pytest.raises(JevClientError, match="answers"):
        client.evaluate("state", [JevQuestion(key="k", kind="noul", instructions="i", criteria={"true": "t", "false": "f"})])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/scoring/test_jev_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidate_ranking.scoring.jev_client'`

- [ ] **Step 4: Implement `jev_client.py`**

Create `src/candidate_ranking/scoring/jev_client.py`:

```python
from __future__ import annotations

from typing import Literal

import requests
from pydantic import BaseModel

_ENDPOINT_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
JEV_MODEL_ID = "typesafe/jev"


class JevQuestion(BaseModel):
    key: str
    kind: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, str] | list[str]


class JevAnswer(BaseModel):
    key: str
    kind: Literal["noul", "choice", "score"]
    value: bool | str | float
    confidence: float


class JevClientError(Exception):
    pass


def _parse_answer(key: str, raw: dict) -> JevAnswer:
    kind = raw.get("type")
    try:
        if kind == "noul":
            probability_true = float(raw["noul"])
            confidence = probability_true if probability_true >= 0.5 else 1.0 - probability_true
            return JevAnswer(key=key, kind="noul", value=probability_true >= 0.5, confidence=confidence)
        if kind == "choice":
            return JevAnswer(key=key, kind="choice", value=raw["choice"], confidence=float(raw["confidence"]))
        if kind == "score":
            return JevAnswer(key=key, kind="score", value=float(raw["score"]), confidence=float(raw["confidence"]))
    except KeyError as exc:
        raise JevClientError(f"Jev answer for {key!r} is missing expected field: {exc}") from exc
    raise JevClientError(f"Jev answer for {key!r} has unrecognized type: {kind!r}")


class JevClient:
    def __init__(self, account_id: str, api_token: str, timeout: float = 30.0) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._timeout = timeout

    def evaluate(self, state: str, questions: list[JevQuestion]) -> list[JevAnswer]:
        payload = {
            "model": JEV_MODEL_ID,
            "input": {
                "state": state,
                "questions": {
                    q.key: {"type": q.kind, "instructions": q.instructions, "criteria": q.criteria}
                    for q in questions
                },
            },
        }
        url = _ENDPOINT_TEMPLATE.format(account_id=self._account_id)
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_token}"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise JevClientError(f"Jev request failed: {exc}") from exc
        except ValueError as exc:
            raise JevClientError(f"Jev response was not valid JSON: {exc}") from exc

        try:
            raw_answers = data["answers"]
        except KeyError as exc:
            raise JevClientError(f"Jev response is missing 'answers': {data}") from exc

        return [_parse_answer(key, raw) for key, raw in raw_answers.items()]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/scoring/test_jev_client.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/candidate_ranking/scoring/jev_client.py tests/scoring/test_jev_client.py
git commit -m "feat: add Jev HTTP client for typesafe structured evaluation"
```

---

### Task 2: Config — Jev Credentials, Drop Tournament Fields

**Files:**
- Modify: `src/candidate_ranking/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from other tasks
- Produces: `RunConfig` gains `cf_account_id: str`, `cf_api_token: str`; loses `tournament_iterations`, `stability_repeats`, `tournament_subset_size`, `num_subset_samples`, `num_mc_draws`, `pl_prior_variance`, `target_appearances_per_candidate`, `tournament_iterations_min`. `_ENV_OVERRIDES` gains `CANDIDATE_RANKING_CF_ACCOUNT_ID` / `CANDIDATE_RANKING_CF_API_TOKEN`, loses the corresponding tournament env vars.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from candidate_ranking.config import RunConfig, apply_env_overrides


def test_full_config_has_empty_jev_credentials_by_default():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.cf_account_id == ""
    assert cfg.cf_api_token == ""
    assert not hasattr(cfg, "tournament_iterations")
    assert not hasattr(cfg, "pl_prior_variance")


def test_env_overrides_apply_jev_credentials(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_CF_ACCOUNT_ID", "acc-from-env")
    monkeypatch.setenv("CANDIDATE_RANKING_CF_API_TOKEN", "token-from-env")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.cf_account_id == "acc-from-env"
    assert cfg.cf_api_token == "token-from-env"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `RunConfig.full()` raises `TypeError` (unexpected/missing keyword) since `cf_account_id`/`cf_api_token` don't exist yet, or the `hasattr` assertions fail because the old fields are still present.

- [ ] **Step 3: Update `config.py`**

Replace the full contents of `src/candidate_ranking/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RunConfig:
    preset: str
    jd_dir: Path
    cv_dir: Path
    cache_dir: Path
    runs_dir: Path
    max_jds: int | None
    max_candidates: int | None
    ollama_model: str
    ollama_base_url: str
    ollama_num_parallel: int
    cf_account_id: str
    cf_api_token: str
    faithfulness_model: str | None = None
    skill_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_num_ctx: int = 8192

    @classmethod
    def full(cls, project_root: Path) -> "RunConfig":
        return cls(
            preset="full",
            jd_dir=project_root / "job-description",
            cv_dir=project_root / "cv",
            cache_dir=project_root / "runs" / "_cache",
            runs_dir=project_root / "runs",
            max_jds=None,
            max_candidates=None,
            ollama_model="qwen2.5:14b-instruct-q4_K_M",
            ollama_base_url="http://localhost:11434",
            ollama_num_parallel=4,
            cf_account_id="",
            cf_api_token="",
            skill_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )


_ENV_OVERRIDES: dict[str, tuple[str, Callable[[str], object]]] = {
    "CANDIDATE_RANKING_JD_DIR": ("jd_dir", Path),
    "CANDIDATE_RANKING_CV_DIR": ("cv_dir", Path),
    "CANDIDATE_RANKING_CACHE_DIR": ("cache_dir", Path),
    "CANDIDATE_RANKING_RUNS_DIR": ("runs_dir", Path),
    "CANDIDATE_RANKING_MAX_JDS": ("max_jds", int),
    "CANDIDATE_RANKING_MAX_CANDIDATES": ("max_candidates", int),
    "CANDIDATE_RANKING_MODEL": ("ollama_model", str),
    "CANDIDATE_RANKING_OLLAMA_BASE_URL": ("ollama_base_url", str),
    "CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL": ("ollama_num_parallel", int),
    "CANDIDATE_RANKING_FAITHFULNESS_MODEL": ("faithfulness_model", str),
    "CANDIDATE_RANKING_OLLAMA_NUM_CTX": ("ollama_num_ctx", int),
    "CANDIDATE_RANKING_SKILL_EMBEDDING_MODEL": ("skill_embedding_model", str),
    "CANDIDATE_RANKING_CF_ACCOUNT_ID": ("cf_account_id", str),
    "CANDIDATE_RANKING_CF_API_TOKEN": ("cf_api_token", str),
}

ENV_OVERRIDE_VARS: tuple[str, ...] = tuple(_ENV_OVERRIDES)


def apply_env_overrides(cfg: RunConfig) -> RunConfig:
    overrides: dict[str, object] = {}
    for env_var, (field_name, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if not raw:
            continue
        try:
            overrides[field_name] = caster(raw)
        except ValueError as exc:
            raise ValueError(f"invalid {env_var}={raw!r}: {exc}") from exc
    return replace(cfg, **overrides)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/config.py tests/test_config.py
git commit -m "feat: add Jev credentials to RunConfig, drop tournament-only fields"
```

---

### Task 3: `Assessment` Model

**Files:**
- Modify: `src/candidate_ranking/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: new `Assessment` shape — `job_description_id: str`, `candidate_id: str`, `generated_by_model: str`, `overall_fit_score: float` (0-100), `overall_recommendation: Literal["hire", "maybe", "no"]`, `meets_min_qualifications: bool`, `requirement_scores: dict[str, float]` (default `{}`), `confidence: dict[str, float]` (default `{}`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from candidate_ranking.models import Assessment


def test_assessment_accepts_new_score_based_fields():
    assessment = Assessment(
        job_description_id="jd-1",
        candidate_id="cand-1",
        generated_by_model="typesafe/jev",
        overall_fit_score=87.5,
        overall_recommendation="hire",
        meets_min_qualifications=True,
        requirement_scores={"python": 100.0, "sql": 50.0},
        confidence={"overall_fit_score": 0.9},
    )
    assert assessment.overall_fit_score == 87.5
    assert assessment.requirement_scores["python"] == 100.0


def test_assessment_rejects_score_out_of_range():
    with pytest.raises(ValidationError):
        Assessment(
            job_description_id="jd-1",
            candidate_id="cand-1",
            generated_by_model="typesafe/jev",
            overall_fit_score=150.0,
            overall_recommendation="hire",
            meets_min_qualifications=True,
        )


def test_assessment_rejects_unknown_recommendation():
    with pytest.raises(ValidationError):
        Assessment(
            job_description_id="jd-1",
            candidate_id="cand-1",
            generated_by_model="typesafe/jev",
            overall_fit_score=50.0,
            overall_recommendation="strongly_hire",
            meets_min_qualifications=True,
        )


def test_assessment_no_longer_has_free_text_fields():
    assert "strengths" not in Assessment.model_fields
    assert "weaknesses" not in Assessment.model_fields
    assert "reasoning" not in Assessment.model_fields
    assert "additional_skills" not in Assessment.model_fields
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — `ValidationError` on construction (unexpected keyword `overall_fit_score`) since the current `Assessment` still requires `strengths`.

- [ ] **Step 3: Update `Assessment` in `models.py`**

In `src/candidate_ranking/models.py`, replace the `Assessment` class:

```python
class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)
    overall_fit_score: float = Field(ge=0, le=100)
    overall_recommendation: Literal["hire", "maybe", "no"]
    meets_min_qualifications: bool
    requirement_scores: dict[str, float] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
```

(The `Literal` import at the top of the file already exists — it's used by `Candidate.parse_status` and `TournamentResult.status`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "feat: replace Assessment free-text fields with Jev structured scores"
```

---

### Task 4: Assessment Generation via Jev

**Files:**
- Modify: `src/candidate_ranking/scoring/assessment.py` (full rewrite of the generation/cache logic; `filter_assessable_candidates` is unchanged)
- Test: `tests/scoring/test_assessment.py`

**Interfaces:**
- Consumes: `JevClient`, `JevQuestion`, `JevAnswer`, `JevClientError` from `candidate_ranking.scoring.jev_client` (Task 1); new `Assessment` from `candidate_ranking.models` (Task 3)
- Produces:
  - `JEV_MODEL_NAME: str = "typesafe/jev"`
  - `class AssessmentGenerationError(Exception)`
  - `def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]` (unchanged behavior)
  - `def generate_assessment(jd: JobDescription, candidate: Candidate, jev_client: JevClient, model_name: str = JEV_MODEL_NAME, jd_skills: JDSkills | None = None) -> Assessment`
  - `def load_or_generate_assessment(jd: JobDescription, candidate: Candidate, jev_client: JevClient, model_name: str, cache_dir: Path, jd_skills: JDSkills | None = None) -> Assessment`

- [ ] **Step 1: Write the failing tests**

Create `tests/scoring/test_assessment.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.assessment import (
    AssessmentGenerationError,
    JEV_MODEL_NAME,
    generate_assessment,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.jev_client import JevAnswer, JevClientError


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python and SQL.", source_path="jd.pdf")


def _candidate() -> Candidate:
    return Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok", skills=["Python"],
    )


def _jd_skills() -> JDSkills:
    return JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python", "SQL"])


def _high_confidence_answers() -> list[JevAnswer]:
    return [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="requirement::SQL", kind="score", value=1.0, confidence=0.8),
    ]


def test_generate_assessment_maps_jev_answers_onto_assessment():
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert assessment.job_description_id == "jd-1"
    assert assessment.candidate_id == "cand-1"
    assert assessment.generated_by_model == JEV_MODEL_NAME
    assert assessment.overall_fit_score == 75.0  # score 3 of 4 max -> 3 * (100/4)
    assert assessment.overall_recommendation == "hire"
    assert assessment.meets_min_qualifications is True
    assert assessment.requirement_scores == {"Python": 100.0, "SQL": 25.0}
    assert assessment.confidence["overall_fit_score"] == 0.9
    jev_client.evaluate.assert_called_once()
    state, questions = jev_client.evaluate.call_args.args
    assert "Backend Engineer" in state
    assert "I know Python." in state
    question_keys = {q.key for q in questions}
    assert question_keys == {
        "overall_fit_score", "overall_recommendation", "meets_min_qualifications",
        "requirement::Python", "requirement::SQL",
    }


def test_generate_assessment_skips_requirement_scores_without_jd_skills():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=2.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="maybe", confidence=0.7),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=False, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills=None)

    assert assessment.requirement_scores == {}
    _, questions = jev_client.evaluate.call_args.args
    assert all(not q.key.startswith("requirement::") for q in questions)


def test_generate_assessment_retries_once_on_low_confidence_then_accepts():
    jev_client = Mock()
    low_confidence_answers = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.3),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]
    jev_client.evaluate.side_effect = [low_confidence_answers, _high_confidence_answers()]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())

    assert jev_client.evaluate.call_count == 2
    assert assessment.confidence["overall_fit_score"] == 0.9
    second_state, _ = jev_client.evaluate.call_args_list[1].args
    assert "low-confidence" in second_state


def test_generate_assessment_wraps_jev_client_error():
    jev_client = Mock()
    jev_client.evaluate.side_effect = JevClientError("network down")

    with pytest.raises(AssessmentGenerationError, match="network down"):
        generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills())


def test_load_or_generate_assessment_uses_cache_on_second_call(tmp_path: Path):
    jev_client = Mock()
    jev_client.evaluate.return_value = _high_confidence_answers()

    first = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())
    second = load_or_generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, tmp_path, _jd_skills())

    assert first == second
    jev_client.evaluate.assert_called_once()
    cache_file = tmp_path / "assessments" / "jd-1" / "cand-1.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["assessment"]["overall_fit_score"] == 75.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/scoring/test_assessment.py -v`
Expected: FAIL — `ImportError: cannot import name 'JEV_MODEL_NAME'` (and the old `generate_assessment` signature takes a `chain: Runnable`, not a `jev_client`).

- [ ] **Step 3: Rewrite `assessment.py`**

Replace the full contents of `src/candidate_ranking/scoring/assessment.py`:

```python
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.jev_client import JevAnswer, JevClient, JevClientError, JevQuestion

logger = logging.getLogger(__name__)

JEV_MODEL_NAME = "typesafe/jev"

ASSESSMENT_SCOPE_VERSION = "jev-score-recommendation-v1"

_REQUIREMENT_KEY_PREFIX = "requirement::"
_OVERALL_FIT_KEY = "overall_fit_score"
_RECOMMENDATION_KEY = "overall_recommendation"
_MIN_QUALIFICATIONS_KEY = "meets_min_qualifications"
_RETRY_ON_LOW_CONFIDENCE_KEYS = (_OVERALL_FIT_KEY, _RECOMMENDATION_KEY, _MIN_QUALIFICATIONS_KEY)
_CONFIDENCE_RETRY_THRESHOLD = 0.5

_SCORE_CRITERIA = ["0", "25", "50", "75", "100"]
_SCORE_MAX_INDEX = len(_SCORE_CRITERIA) - 1


class AssessmentGenerationError(Exception):
    pass


def _requirement_question_key(requirement: str) -> str:
    return f"{_REQUIREMENT_KEY_PREFIX}{requirement}"


def _score_to_percent(raw_score: float) -> float:
    return max(0.0, min(100.0, raw_score * (100.0 / _SCORE_MAX_INDEX)))


def _format_candidate_skills(candidate: Candidate) -> str:
    return ", ".join(candidate.skills) if candidate.skills else "(none extracted)"


def _build_state(jd: JobDescription, candidate: Candidate, low_confidence_note: str = "") -> str:
    return (
        f"Job Title: {jd.title}\n\n"
        f"Job Description:\n{jd.raw_text}\n\n"
        f"Candidate's identified skills: {_format_candidate_skills(candidate)}\n\n"
        f"Candidate CV:\n{candidate.raw_text}{low_confidence_note}"
    )


def _build_questions(jd_technical_skills: list[str] | None) -> list[JevQuestion]:
    questions = [
        JevQuestion(
            key=_OVERALL_FIT_KEY,
            kind="score",
            instructions="How well does this candidate's CV fit the job description overall?",
            criteria=_SCORE_CRITERIA,
        ),
        JevQuestion(
            key=_RECOMMENDATION_KEY,
            kind="choice",
            instructions="What is the hiring recommendation for this candidate against this job description?",
            criteria={
                "hire": "Candidate clearly meets or exceeds the role's requirements",
                "maybe": "Candidate partially meets the role's requirements",
                "no": "Candidate does not meet the role's requirements",
            },
        ),
        JevQuestion(
            key=_MIN_QUALIFICATIONS_KEY,
            kind="noul",
            instructions="Does the candidate meet the job description's minimum qualifications?",
            criteria={
                "true": "Meets every minimum qualification stated in the job description",
                "false": "Fails at least one minimum qualification stated in the job description",
            },
        ),
    ]
    for requirement in jd_technical_skills or []:
        questions.append(
            JevQuestion(
                key=_requirement_question_key(requirement),
                kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_SCORE_CRITERIA,
            )
        )
    return questions


def _answers_to_assessment(
    jd: JobDescription, candidate: Candidate, model_name: str, answers: list[JevAnswer]
) -> Assessment:
    by_key = {a.key: a for a in answers}
    confidence = {a.key: a.confidence for a in answers}
    requirement_scores = {
        key[len(_REQUIREMENT_KEY_PREFIX):]: _score_to_percent(a.value)
        for key, a in by_key.items()
        if key.startswith(_REQUIREMENT_KEY_PREFIX)
    }
    return Assessment(
        job_description_id=jd.id,
        candidate_id=candidate.id,
        generated_by_model=model_name,
        overall_fit_score=_score_to_percent(by_key[_OVERALL_FIT_KEY].value),
        overall_recommendation=by_key[_RECOMMENDATION_KEY].value,
        meets_min_qualifications=by_key[_MIN_QUALIFICATIONS_KEY].value,
        requirement_scores=requirement_scores,
        confidence=confidence,
    )


def generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str = JEV_MODEL_NAME,
    jd_skills: JDSkills | None = None,
) -> Assessment:
    jd_technical_skills = jd_skills.technical_skills if jd_skills is not None else None
    questions = _build_questions(jd_technical_skills)

    low_confidence_note = ""
    assessment: Assessment | None = None
    for _attempt in range(2):
        state = _build_state(jd, candidate, low_confidence_note)
        try:
            answers = jev_client.evaluate(state, questions)
        except JevClientError as exc:
            raise AssessmentGenerationError(str(exc)) from exc

        assessment = _answers_to_assessment(jd, candidate, model_name, answers)
        low_confidence_keys = [
            key for key in _RETRY_ON_LOW_CONFIDENCE_KEYS
            if assessment.confidence.get(key, 1.0) < _CONFIDENCE_RETRY_THRESHOLD
        ]
        if not low_confidence_keys:
            break
        logger.warning(
            "Low-confidence Jev answer(s) for %s/%s: %s -- retrying once",
            jd.id, candidate.id, low_confidence_keys,
        )
        low_confidence_note = (
            "\n\nNote: a previous evaluation of this same candidate/job pair returned "
            f"low-confidence answers for: {', '.join(low_confidence_keys)}. Re-evaluate carefully."
        )

    assert assessment is not None
    return assessment


def filter_assessable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    assessable = [c for c in candidates if c.parse_status == "ok"]
    excluded_count = len(candidates) - len(assessable)
    if excluded_count > 0:
        logger.warning(
            "Excluded %d candidate(s) with failed CV parsing from the assessable pool",
            excluded_count,
        )
    return assessable


def _assessment_cache_path(cache_dir: Path, jd_id: str, candidate_id: str) -> Path:
    return cache_dir / "assessments" / jd_id / f"{candidate_id}.json"


def _assessment_cache_key(
    jd: JobDescription,
    candidate: Candidate,
    model_name: str,
    jd_skills: JDSkills | None = None,
) -> str:
    jd_technical_skills = jd_skills.technical_skills if jd_skills is not None else None
    state = _build_state(jd, candidate)
    questions_repr = repr([q.model_dump() for q in _build_questions(jd_technical_skills)])
    digest_input = f"{state}||{questions_repr}||{model_name}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _read_cached_assessment(path: Path, expected_key: str) -> Assessment | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Discarding unreadable assessment cache %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("cache_key") != expected_key:
        return None
    try:
        return Assessment.model_validate(data["assessment"])
    except (KeyError, ValidationError) as exc:
        logger.warning("Discarding invalid cached assessment %s: %s", path, exc)
        return None


def _write_cached_assessment(path: Path, cache_key: str, assessment: Assessment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"cache_key": cache_key, "assessment": assessment.model_dump()}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_or_generate_assessment(
    jd: JobDescription,
    candidate: Candidate,
    jev_client: JevClient,
    model_name: str,
    cache_dir: Path,
    jd_skills: JDSkills | None = None,
) -> Assessment:
    path = _assessment_cache_path(cache_dir, jd.id, candidate.id)
    key = _assessment_cache_key(jd, candidate, model_name, jd_skills)

    cached = _read_cached_assessment(path, key)
    if cached is not None:
        return cached

    assessment = generate_assessment(jd, candidate, jev_client, model_name, jd_skills)
    _write_cached_assessment(path, key, assessment)
    return assessment
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/scoring/test_assessment.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/assessment.py tests/scoring/test_assessment.py
git commit -m "feat: generate assessments via Jev structured questions instead of free-text LLM"
```

---

### Task 5: Ranking Output Formatting

**Files:**
- Modify: `src/candidate_ranking/output/formatter.py` (full rewrite)
- Test: `tests/output/test_formatter.py`

**Interfaces:**
- Consumes: `Assessment` (Task 3)
- Produces:
  - `def format_jd_ranking(jd: JobDescription, assessments: dict[str, Assessment]) -> tuple[str, dict]`
  - `def write_jd_ranking(run_dir: Path, jd: JobDescription, assessments: dict[str, Assessment]) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/output/test_formatter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription
from candidate_ranking.output.formatter import format_jd_ranking, write_jd_ranking


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _assessment(candidate_id: str, score: float, recommendation: str, meets_min: bool) -> Assessment:
    return Assessment(
        job_description_id="jd-1",
        candidate_id=candidate_id,
        generated_by_model="typesafe/jev",
        overall_fit_score=score,
        overall_recommendation=recommendation,
        meets_min_qualifications=meets_min,
        requirement_scores={"Python": score},
        confidence={"overall_fit_score": 0.9},
    )


def test_format_jd_ranking_sorts_descending_by_score():
    assessments = {
        "cand-a": _assessment("cand-a", 40.0, "maybe", False),
        "cand-b": _assessment("cand-b", 90.0, "hire", True),
        "cand-c": _assessment("cand-c", 65.0, "maybe", True),
    }

    markdown_text, json_payload = format_jd_ranking(_jd(), assessments)

    rows = json_payload["rankings"]
    assert [row["candidate_id"] for row in rows] == ["cand-b", "cand-c", "cand-a"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert rows[0]["overall_fit_score"] == 90.0
    assert rows[0]["overall_recommendation"] == "hire"
    assert rows[0]["meets_min_qualifications"] is True
    assert rows[0]["requirement_scores"] == {"Python": 90.0}
    assert json_payload["job_description_id"] == "jd-1"
    assert "cand-b" in markdown_text
    assert markdown_text.index("cand-b") < markdown_text.index("cand-c") < markdown_text.index("cand-a")


def test_write_jd_ranking_writes_markdown_and_json(tmp_path: Path):
    assessments = {"cand-a": _assessment("cand-a", 55.0, "maybe", False)}

    write_jd_ranking(tmp_path, _jd(), assessments)

    assert (tmp_path / "jd-1" / "ranking.md").exists()
    json_payload = json.loads((tmp_path / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert json_payload["rankings"][0]["candidate_id"] == "cand-a"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/output/test_formatter.py -v`
Expected: FAIL — `TypeError: format_jd_ranking() missing 1 required positional argument: 'tournament_result'` (current signature still requires it).

- [ ] **Step 3: Rewrite `formatter.py`**

Replace the full contents of `src/candidate_ranking/output/formatter.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from candidate_ranking.models import Assessment, JobDescription


def format_jd_ranking(jd: JobDescription, assessments: dict[str, Assessment]) -> tuple[str, dict]:
    ranked = sorted(assessments, key=lambda cid: assessments[cid].overall_fit_score, reverse=True)

    lines = [
        f"# Ranking: {jd.title} ({jd.id})",
        "",
        "> Scores, recommendations, and per-requirement fit are produced directly by Jev.",
        "",
    ]
    rows = []
    for rank, candidate_id in enumerate(ranked, start=1):
        assessment = assessments[candidate_id]
        lines.append(
            f"{rank}. **{candidate_id}** (score={assessment.overall_fit_score:.1f}, "
            f"{assessment.overall_recommendation}) — meets_min_qualifications="
            f"{assessment.meets_min_qualifications}"
        )
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "overall_fit_score": assessment.overall_fit_score,
                "overall_recommendation": assessment.overall_recommendation,
                "meets_min_qualifications": assessment.meets_min_qualifications,
                "requirement_scores": assessment.requirement_scores,
            }
        )

    markdown_text = "\n".join(lines) + "\n"
    json_payload = {"job_description_id": jd.id, "rankings": rows}
    return markdown_text, json_payload


def write_jd_ranking(run_dir: Path, jd: JobDescription, assessments: dict[str, Assessment]) -> None:
    markdown_text, json_payload = format_jd_ranking(jd, assessments)
    jd_dir = run_dir / jd.id
    jd_dir.mkdir(parents=True, exist_ok=True)
    (jd_dir / "ranking.md").write_text(markdown_text, encoding="utf-8")
    (jd_dir / "ranking.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/output/test_formatter.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/output/formatter.py tests/output/test_formatter.py
git commit -m "feat: format rankings by Jev score, drop tournament-based formatting"
```

---

### Task 6: Console-Web Export

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py`
- Test: `tests/output/test_console_export.py`

**Interfaces:**
- Consumes: `ranking.json` shape from Task 5 (`rank`, `candidate_id`, `overall_fit_score`, `overall_recommendation`, `meets_min_qualifications`, `requirement_scores`); `assessments.json` shape from Task 3/4's `Assessment.model_dump()`
- Produces: `export_console_web_data` unchanged signature; internal mapping updated

**Note:** the exported `real-data.json`'s `candidates[].utility` key name is kept as-is (avoids an untracked console-web frontend change) but its value now comes from `overall_fit_score`. Only the `assessments` dict's per-candidate shape changes, per the design spec.

- [ ] **Step 1: Write the failing test**

Create `tests/output/test_console_export.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from candidate_ranking.config import RunConfig
from candidate_ranking.output.console_export import export_console_web_data


@pytest.fixture
def run_with_jev_ranking(tmp_path: Path, monkeypatch) -> tuple[RunConfig, str]:
    project_root = tmp_path / "project"
    jd_dir = project_root / "job-description"
    cv_dir = project_root / "cv"
    runs_dir = project_root / "runs"
    jd_dir.mkdir(parents=True)
    cv_dir.mkdir(parents=True)

    (jd_dir / "jd-1.txt").write_text("Backend Engineer\n\nNeeds Python.", encoding="utf-8")
    # load_candidates() only globs *.pdf and parses via pypdf, so the fixture CV
    # must be a real (if minimal/blank) PDF, not a .txt file.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(cv_dir / "cand-a.pdf", "wb") as f:
        writer.write(f)

    run_id = "run-1"
    run_dir = runs_dir / run_id
    (run_dir / "jd-1").mkdir(parents=True)

    manifest = {"ollama_model": "qwen2.5:14b", "jd_ids": ["jd-1"], "candidate_ids": ["cand-a"]}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ranking = {
        "job_description_id": "jd-1",
        "rankings": [
            {
                "rank": 1, "candidate_id": "cand-a", "overall_fit_score": 88.0,
                "overall_recommendation": "hire", "meets_min_qualifications": True,
                "requirement_scores": {"Python": 100.0},
            }
        ],
    }
    (run_dir / "jd-1" / "ranking.json").write_text(json.dumps(ranking), encoding="utf-8")

    assessments = {
        "cand-a": {
            "job_description_id": "jd-1", "candidate_id": "cand-a", "generated_by_model": "typesafe/jev",
            "overall_fit_score": 88.0, "overall_recommendation": "hire", "meets_min_qualifications": True,
            "requirement_scores": {"Python": 100.0}, "confidence": {"overall_fit_score": 0.9},
        }
    }
    (run_dir / "jd-1" / "assessments.json").write_text(json.dumps(assessments), encoding="utf-8")

    cfg = RunConfig.full(project_root)

    monkeypatch.setattr(
        "candidate_ranking.output.console_export.load_or_extract_candidate_name",
        lambda candidate, chain, model_name, cache_path: None,
    )
    monkeypatch.setattr(
        "candidate_ranking.output.console_export.build_name_extraction_chain", lambda llm: None
    )
    monkeypatch.setattr("candidate_ranking.output.console_export.ChatOllama", lambda **kwargs: None)
    monkeypatch.setattr("candidate_ranking.output.console_export.evaluate_run", lambda cfg, run_id: [])

    return cfg, run_id


def test_export_console_web_data_maps_jev_scores(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    output_path = tmp_path / "real-data.json"

    result_path = export_console_web_data(cfg, run_id, output_path=output_path)

    assert result_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["candidates"][0]["utility"] == 88.0
    assessment = data["assessments"]["cand-a::jd-1"]
    assert assessment["overall_fit_score"] == 88.0
    assert assessment["overall_recommendation"] == "hire"
    assert assessment["requirement_scores"] == {"Python": 100.0}
    assert "strengths" not in assessment
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/output/test_console_export.py -v`
Expected: FAIL — `KeyError: 'utility'` (current code reads `row["utility"]` from `ranking.json`, which Task 5's new shape doesn't have) or `KeyError: 'strengths'`.

- [ ] **Step 3: Update `console_export.py`**

In `src/candidate_ranking/output/console_export.py`, replace the body of `export_console_web_data` from the `ranking = json.loads(...)` line through the `assessments[row_id] = {...}` block:

```python
        ranking = json.loads((run_dir / jd_id / "ranking.json").read_text(encoding="utf-8"))
        rank_by_cv_id = {row["candidate_id"]: row["rank"] for row in ranking["rankings"]}
        score_by_cv_id = {row["candidate_id"]: row["overall_fit_score"] for row in ranking["rankings"]}

        jd_assessments = {
            cv_id: entry for cv_id, entry in sorted(_load_jd_assessments(run_dir, jd_id).items())
            if cv_id in candidates_by_id
        }
        exported_count = 0
        for cv_id, assessment_entry in jd_assessments.items():
            row_id = f"{cv_id}::{jd_id}"
            name = names_by_cv_id[cv_id]

            candidates.append(
                {
                    "id": row_id,
                    "cvId": cv_id,
                    "roleId": jd_id,
                    "name": name if name else cv_id.upper(),
                    "initials": _initials_for(name, cv_id),
                    "rank": rank_by_cv_id.get(cv_id),
                    "utility": score_by_cv_id.get(cv_id),
                }
            )

            assessments[row_id] = {
                "overall_fit_score": assessment_entry["overall_fit_score"],
                "overall_recommendation": assessment_entry["overall_recommendation"],
                "meets_min_qualifications": assessment_entry["meets_min_qualifications"],
                "requirement_scores": assessment_entry.get("requirement_scores", {}),
            }
            exported_count += 1
```

Also update the `export_console_web_data` signature's default to keep it keyword-compatible with the test's `output_path=` call — it already accepts `output_path: Path = DEFAULT_OUTPUT_PATH` as a keyword-or-positional argument, so no signature change is needed.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/output/test_console_export.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/output/console_export.py tests/output/test_console_export.py
git commit -m "feat: export Jev scores to console-web instead of strengths/weaknesses"
```

---

### Task 7: Pipeline — Drop Tournament Stage, Add Rank-by-Score Stage

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py`
- Test: `tests/graphs/test_pipeline.py`

**Interfaces:**
- Consumes: `load_or_generate_assessment` (Task 4), `write_jd_ranking` (Task 5), `JevClient` (Task 1)
- Produces:
  - `PipelineState` (TypedDict) — no longer has a `tournament_results` key
  - `AssessmentResult` (TypedDict) — no longer has a `retry_audit` key
  - `def build_pipeline_graph(cfg: RunConfig, jd_skills_chain: Runnable, jev_client: JevClient, skill_index: np.ndarray, skill_row_map: list[tuple[str, str]], skill_embedder: Callable[[list[str]], np.ndarray], run_id: str, skill_match_threshold: float = 0.8, min_skill_matches: int = 5) -> StateGraph`
  - `def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]` — module-level helper, independently testable
  - `def rank_and_format_jd(runs_dir: Path, run_id: str, jd: JobDescription, assessments: dict[str, Assessment]) -> None` — module-level helper, independently testable

- [ ] **Step 1: Write the failing tests**

Create `tests/graphs/test_pipeline.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_ranking.config import RunConfig
from candidate_ranking.graphs.pipeline import PipelineState, assessments_by_jd, build_pipeline_graph, rank_and_format_jd
from candidate_ranking.models import Assessment, JobDescription


def _ok_result(jd_id: str, candidate_id: str, score: float) -> dict:
    return {
        "jd_id": jd_id,
        "candidate_id": candidate_id,
        "status": "ok",
        "assessment": Assessment(
            job_description_id=jd_id, candidate_id=candidate_id, generated_by_model="typesafe/jev",
            overall_fit_score=score, overall_recommendation="hire", meets_min_qualifications=True,
        ),
        "error": None,
    }


def _failed_result(jd_id: str, candidate_id: str) -> dict:
    return {"jd_id": jd_id, "candidate_id": candidate_id, "status": "failed", "assessment": None, "error": "boom"}


def test_pipeline_state_has_no_tournament_results_key():
    assert "tournament_results" not in PipelineState.__annotations__
    assert "assessment_results" in PipelineState.__annotations__


def test_assessments_by_jd_groups_ok_results_and_skips_failed():
    results = [_ok_result("jd-1", "cand-a", 80.0), _ok_result("jd-1", "cand-b", 60.0), _failed_result("jd-1", "cand-c")]

    grouped = assessments_by_jd(results)

    assert set(grouped["jd-1"]) == {"cand-a", "cand-b"}
    assert grouped["jd-1"]["cand-a"].overall_fit_score == 80.0


def test_rank_and_format_jd_writes_ranking_files(tmp_path: Path):
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")
    assessments = {
        "cand-a": Assessment(
            job_description_id="jd-1", candidate_id="cand-a", generated_by_model="typesafe/jev",
            overall_fit_score=80.0, overall_recommendation="hire", meets_min_qualifications=True,
        )
    }

    rank_and_format_jd(tmp_path, "run-1", jd, assessments)

    ranking = json.loads((tmp_path / "run-1" / "jd-1" / "ranking.json").read_text(encoding="utf-8"))
    assert ranking["rankings"][0]["candidate_id"] == "cand-a"


def test_build_pipeline_graph_has_no_tournament_nodes(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain=None,
        jev_client=None,
        skill_index=np.zeros(0),
        skill_row_map=[],
        skill_embedder=lambda texts: np.zeros((len(texts), 0)),
        run_id="run-1",
    )

    node_names = set(graph.nodes)
    assert "rank_and_format_jd_node" in node_names
    assert not any("tournament" in name for name in node_names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/graphs/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'assessments_by_jd' from 'candidate_ranking.graphs.pipeline'`

- [ ] **Step 3: Rewrite `pipeline.py`**

Replace the full contents of `src/candidate_ranking/graphs/pipeline.py`:

```python
from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from candidate_ranking.scoring.assessment import AssessmentGenerationError, JEV_MODEL_NAME, load_or_generate_assessment
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.config import RunConfig
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, load_or_generate_jd_skills
from candidate_ranking.output.formatter import write_jd_ranking
from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.skills import shortlist_candidates

logger = logging.getLogger(__name__)


class AssessmentResult(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


class PipelineState(TypedDict):
    jds: list[JobDescription]
    candidates: list[Candidate]
    jd_skills: Annotated[dict[str, JDSkills], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]


def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]:
    grouped: dict[str, dict[str, Assessment]] = {}
    for result in assessment_results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["jd_id"], {})[result["candidate_id"]] = result["assessment"]
    return grouped


def rank_and_format_jd(runs_dir: Path, run_id: str, jd: JobDescription, assessments: dict[str, Assessment]) -> None:
    write_jd_ranking(runs_dir / run_id, jd, assessments)


def build_pipeline_graph(
    cfg: RunConfig,
    jd_skills_chain: Runnable,
    jev_client: JevClient,
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
) -> StateGraph:
    jd_skills_cache_path = cfg.cache_dir / "jd_skills.json"

    def _noop(state: PipelineState) -> dict:
        return {}

    def fanout_skill_extraction(state: PipelineState) -> list[Send]:
        return [Send("extract_skills_for_jd", {"jd": jd}) for jd in state["jds"]]

    def extract_skills_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            jd_skills = load_or_generate_jd_skills(jd, jd_skills_chain, cfg.ollama_model, jd_skills_cache_path)
        except JDSkillsGenerationError as exc:
            logger.warning("Excluding %s from this run: %s", jd.id, exc)
            return {}
        return {"jd_skills": {jd.id: jd_skills}}

    def fanout_shortlist(state: PipelineState) -> list[Send]:
        return [
            Send("build_shortlist_for_jd", {"jd": jd, "jd_skills": state["jd_skills"][jd.id]})
            for jd in state["jds"]
            if jd.id in state["jd_skills"]
        ]

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        jd_skills = payload["jd_skills"]
        candidate_ids = shortlist_candidates(
            jd,
            jd_skills,
            skill_index,
            skill_row_map,
            skill_embedder,
            threshold=skill_match_threshold,
            min_matches=min_skill_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {"jd": jd, "candidate": candidates_by_id[candidate_id], "jd_skills": state["jd_skills"].get(jd.id)},
            )
            for jd in state["jds"]
            for candidate_id in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd = payload["jd"]
        candidate = payload["candidate"]
        jd_skills = payload.get("jd_skills")
        try:
            assessment = load_or_generate_assessment(
                jd, candidate, jev_client, JEV_MODEL_NAME, cfg.cache_dir,
                jd_skills=jd_skills,
            )
            result: AssessmentResult = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "ok",
                "assessment": assessment, "error": None,
            }
        except AssessmentGenerationError as exc:
            result = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "failed",
                "assessment": None, "error": str(exc),
            }
        return {"assessment_results": [result]}

    def fanout_rank_and_format(state: PipelineState) -> list[Send]:
        grouped = assessments_by_jd(state["assessment_results"])
        sends = []
        for jd in state["jds"]:
            jd_assessments = grouped.get(jd.id, {})
            if not jd_assessments:
                continue
            sends.append(Send("rank_and_format_jd_node", {"jd": jd, "assessments": jd_assessments}))
        return sends

    def rank_and_format_jd_node(payload: dict) -> dict:
        rank_and_format_jd(cfg.runs_dir, run_id, payload["jd"], payload["assessments"])
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_skills_for_jd", extract_skills_for_jd)
    graph.add_node("skills_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("shortlist_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_skill_extraction, ["extract_skills_for_jd"])
    graph.add_edge("extract_skills_for_jd", "skills_barrier")
    graph.add_conditional_edges("skills_barrier", fanout_shortlist, ["build_shortlist_for_jd"])
    graph.add_edge("build_shortlist_for_jd", "shortlist_barrier")
    graph.add_conditional_edges("shortlist_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/graphs/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py tests/graphs/test_pipeline.py
git commit -m "feat: drop tournament stage from pipeline, rank candidates by Jev score"
```

---

### Task 8: CLI Wiring

**Files:**
- Modify: `src/candidate_ranking/cli.py`

**Interfaces:**
- Consumes: `JevClient` (Task 1), `JEV_MODEL_NAME` and the new `load_or_generate_assessment`/`generate_assessment` signatures (Task 4), the new `build_pipeline_graph` signature (Task 7)
- Produces: `run()` — same public signature (`run_id`, `skill_match_threshold`, `min_skill_matches`) — and `main()` are otherwise unchanged from the CLI's perspective

- [ ] **Step 1: Update imports and remove tournament/retry-audit wiring in `cli.py`**

In `src/candidate_ranking/cli.py`, replace this import block:

```python
from candidate_ranking.scoring.assessment import (
    ASSESSMENT_SCOPE_VERSION,
    build_assessment_chain,
    filter_assessable_candidates,
    write_retry_audit_report,
)
from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import build_skill_extraction_chain, enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.output.run_output import write_run_output
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder
from candidate_ranking.ranking.tournament import (
    RANKING_PROMPT_VERSION,
    build_listwise_ranking_chain,
)
```

with:

```python
from candidate_ranking.scoring.assessment import ASSESSMENT_SCOPE_VERSION, JEV_MODEL_NAME, filter_assessable_candidates
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data, seed_console_web_roles
from candidate_ranking.scoring.jd_skills import build_jd_skills_chain
from candidate_ranking.graphs.pipeline import build_pipeline_graph
from candidate_ranking.ingestion.cv import build_skill_extraction_chain, enrich_candidates_with_skills, load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.logging_setup import configure_logging
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.output.run_output import write_run_output
from candidate_ranking.scoring.skills import build_candidate_skill_index, build_skill_embedder
```

- [ ] **Step 2: Replace LLM/tournament chain construction with the Jev client**

Replace this block in `run()`:

```python
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    jd_skills_chain = build_jd_skills_chain(llm)
    assessment_chain = build_assessment_chain(llm)
    build_ranking_chain = build_listwise_ranking_chain(llm)
    skill_extraction_chain = build_skill_extraction_chain(llm)
```

with:

```python
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    jd_skills_chain = build_jd_skills_chain(llm)
    jev_client = JevClient(account_id=cfg.cf_account_id, api_token=cfg.cf_api_token)
    skill_extraction_chain = build_skill_extraction_chain(llm)
```

- [ ] **Step 3: Update the `build_pipeline_graph` call and `initial_state`**

Replace:

```python
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain,
        assessment_chain,
        skill_index,
        skill_row_map,
        skill_embedder,
        build_ranking_chain,
        run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
    )
```

with:

```python
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain,
        jev_client,
        skill_index,
        skill_row_map,
        skill_embedder,
        run_id,
        skill_match_threshold=skill_match_threshold,
        min_skill_matches=min_skill_matches,
    )
```

Replace:

```python
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "shortlists": {},
        "assessment_results": [],
        "tournament_results": [],
    }
```

with:

```python
    initial_state = {
        "jds": jds,
        "candidates": candidates,
        "jd_skills": {},
        "shortlists": {},
        "assessment_results": [],
    }
```

- [ ] **Step 4: Drop the retry-audit report and the tournament manifest field**

Replace:

```python
    manifest = {
        "run_id": run_id,
        "preset": cfg.preset,
        "ollama_model": cfg.ollama_model,
        "ranking_prompt_version": RANKING_PROMPT_VERSION,
        "assessment_scope": ASSESSMENT_SCOPE_VERSION,
        "skill_embedding_model": cfg.skill_embedding_model,
        "skill_match_threshold": skill_match_threshold,
        "min_skill_matches": min_skill_matches,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(
        cfg.runs_dir, run_id, manifest, final_state["assessment_results"]
    )
    retry_audits = [
        r["retry_audit"] for r in final_state["assessment_results"] if r.get("retry_audit") is not None
    ]
    write_retry_audit_report(cfg.runs_dir, run_id, retry_audits)
```

with:

```python
    manifest = {
        "run_id": run_id,
        "preset": cfg.preset,
        "ollama_model": cfg.ollama_model,
        "jev_model": JEV_MODEL_NAME,
        "assessment_scope": ASSESSMENT_SCOPE_VERSION,
        "skill_embedding_model": cfg.skill_embedding_model,
        "skill_match_threshold": skill_match_threshold,
        "min_skill_matches": min_skill_matches,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
    run_dir = write_run_output(
        cfg.runs_dir, run_id, manifest, final_state["assessment_results"]
    )
```

- [ ] **Step 5: Verify the module imports cleanly and the CLI still parses arguments**

Run: `.venv/bin/python -c "import candidate_ranking.cli"`
Expected: no output, exit code 0 (no `ImportError`/`NameError` from the now-removed `RANKING_PROMPT_VERSION`, `build_assessment_chain`, `build_listwise_ranking_chain`, or `write_retry_audit_report` symbols).

Run: `.venv/bin/python -m candidate_ranking.cli run --help`
Expected: prints the `run` subcommand's `--run-id`, `--skill-match-threshold`, `--min-skill-matches` help text and exits 0.

Run: `grep -rn "build_assessment_chain\|build_listwise_ranking_chain\|write_retry_audit_report\|RANKING_PROMPT_VERSION" src/candidate_ranking/cli.py`
Expected: no matches (all four symbols fully removed from `cli.py`).

- [ ] **Step 6: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests from Tasks 1-7 pass (21 total: 3 + 2 + 4 + 5 + 2 + 1 + 4), 0 failures.

- [ ] **Step 7: Commit**

```bash
git add src/candidate_ranking/cli.py
git commit -m "feat: wire Jev client into CLI, drop tournament/retry-audit wiring"
```

---

## Post-Plan Notes

- `src/candidate_ranking/ranking/tournament.py`, `mc_kg.py`, `plackett_luce.py`, and `src/candidate_ranking/evaluation/evaluation.py` are no longer called from the pipeline but remain importable for the injection research to use standalone. `evaluate_run()` in `evaluation.py` will now always return an empty list for new runs (no `repeats.json` is written) — `console_export.py`'s `comparison[jdId].kendallTau`/`.deltaU` will be `None` for every JD; this is expected, not a bug.
- `src/candidate_ranking/injection/` and `evaluation/ragas_eval.py` were not touched and will fail or behave incorrectly if invoked against runs produced after this migration (they still expect `strengths`/`weaknesses` on `Assessment`). Adapting them is separate follow-up work.
- Rotate the Cloudflare API token that was pasted into the chat during the design phase, if that has not already been done.
- Do not resume a pre-migration `--run-id`. Its LangGraph checkpoint carries a `tournament_results` key no longer in `PipelineState`, and any `Assessment` objects in it are the old free-text schema — resume will fail or behave oddly. Start a fresh run id for any run after this migration. (Assessment *caches* degrade gracefully instead: an old cache entry's key won't match the new `state`+`questions` digest, so it's simply treated as a miss and regenerated.)
- Final whole-branch review (2026-09-18) found the blast radius of leaving `injection/` untouched is wider than this plan originally documented: `src/candidate_ranking/injection/defenses.py` fails at import (`ImportError` on `ASSESSMENT_GENERATION_PROMPT`/`_GeneratedAssessment`, both removed from `assessment.py` by Task 4), which breaks the whole `injection` package (`__init__.py` imports `defenses`). Additionally, `scripts/run_injection_study.py`, `scripts/run_extended_injection_study.py`, and `scripts/run_adaptive_cross_defense.py` each import `build_assessment_chain` directly and call the OLD `generate_assessment(jd, candidate, chain, model_name)` (chain-based, 4 positional args) — a function Task 4 replaced with a different signature (`generate_assessment(jd, candidate, jev_client, model_name, jd_skills=None)`). A true fix requires resurrecting the whole old chain-based generation path as a standalone legacy module, not just relocating two names — the user decided against doing that here (out of scope for this migration) and will commit their in-progress `injection/` research WIP separately and adapt it to the new `Assessment` schema as follow-up work. Until that adaptation happens, `injection/` and the three scripts above will not import or run.
