# Automated, Accuracy-Guarded Criteria Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated optimizer that revises the per-requirement Score-question criteria (`_REQUIREMENT_FIT_CRITERIA`), guarded against confidence-only Goodharting by an independent-model (Claude) proxy-label accuracy check and confidence-based hard-case mining that refreshes low-confidence boundary cases into each round, then compare the optimized criteria against the current hand-written baseline on held-out data.

**Architecture:** A round-based proposer/evaluate loop. A LangChain-backed proposer (Qwen2.5-14B via Ollama, `ChatOllama` + structured output, following the existing `jd_skills.py` pattern) revises a 5-level criteria list each round; `evaluate_criteria` runs it through the real Jev API over a sample; `compute_metric` credits Jev's confidence only when it agrees with an independent Claude proxy label (the anti-Goodhart guard); `select_hard_triples` mines the lowest-confidence validation pairs each round and feeds them back as hard cases for the next round (confidence-based hard-case mining — a standard, well-established technique, not an adversarial agent). A final script compares the optimizer's best criteria against the hand-written baseline on a held-out split.

**Tech Stack:** Python 3.12, Pydantic, LangChain (`langchain-core`, `langchain-ollama` — already project dependencies) over Ollama (`qwen2.5:14b-instruct-q4_K_M`), Anthropic SDK (`anthropic`) for the proxy labeler, existing `JevClient`/`requests`, `scipy.stats` for the final Wilcoxon comparison, `pytest` + `unittest.mock`.

## Global Constraints

- Scope is the per-technical-requirement Score question only (`_REQUIREMENT_FIT_CRITERIA` in `src/candidate_ranking/scoring/assessment.py`). Seniority, education, must-have/nice-to-have classification, shortlisting, and ranking are unchanged — do not touch them.
- Every proposed criteria list must have exactly 5 entries (matches the existing `_SCORE_MAX_INDEX = len(criteria) - 1` assumption already asserted in `assessment.py`).
- Confidence is only credited toward the optimization metric when Jev's picked level is within 1 of the independent Claude proxy label — this is the load-bearing anti-Goodhart guard from the spec and must not be simplified to "confidence alone."
- No network calls in unit tests — mock `requests.post` (existing `JevClient` pattern), the Anthropic client, and the LangChain chain's `.invoke`/structured-output call site, exactly as `tests/scoring/test_jev_client.py` already mocks `requests.post`.
- Reuse existing helpers instead of duplicating logic: `_build_state`, `_REQUIREMENT_FIT_CRITERIA` from `assessment.py`; `load_json_cache`/`save_json_cache` from `json_cache.py`; `GenerationError`/`invoke_and_validate` from `generation.py`; the `ChatPromptTemplate` + `llm.with_structured_output(...)` chain pattern from `jd_skills.py`. Cross-module import of underscore-prefixed helpers from `assessment.py` is an established pattern in this codebase (`scripts/run_jev_evaluation_study.py` already does it) — follow it, don't rename things to avoid the underscore.
- New model identifiers (Claude proxy-label model) are config values with env overrides, not hardcoded strings, since exact API model IDs can change — follow the existing `RunConfig` / `_ENV_OVERRIDES` pattern in `config.py`. The proposer's LLM reuses the existing `cfg.ollama_model`/`cfg.ollama_base_url` — no new config field needed for it.
- DSPy is explicitly not a dependency of this project. Do not introduce it or any other LLM-orchestration framework — the proposer is built on the existing LangChain pattern, which already meets the need.
- `select_hard_triples` and its role in the loop must be named and documented as confidence-based hard-case mining (a standard technique), never as "red-team" or adversarial — this project explicitly does not claim that framing as novel or accurate to what's built.

---

## Task 1: Dependencies and config for the proxy labeler

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/candidate_ranking/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `RunConfig.anthropic_api_key: str` (default `""`), `RunConfig.proxy_label_model: str` (default `"claude-opus-5"`), env vars `CANDIDATE_RANKING_ANTHROPIC_API_KEY`, `CANDIDATE_RANKING_PROXY_LABEL_MODEL`. Later tasks read these two fields to construct the Anthropic client.

- [ ] **Step 1: Write the failing config tests**

Add to `tests/test_config.py`:

```python
def test_full_config_has_empty_anthropic_credentials_by_default():
    cfg = RunConfig.full(Path("/tmp/project"))
    assert cfg.anthropic_api_key == ""
    assert cfg.proxy_label_model == "claude-opus-5"


def test_env_overrides_apply_anthropic_credentials(monkeypatch):
    monkeypatch.setenv("CANDIDATE_RANKING_ANTHROPIC_API_KEY", "anthropic-key-from-env")
    monkeypatch.setenv("CANDIDATE_RANKING_PROXY_LABEL_MODEL", "claude-opus-5-test")
    cfg = apply_env_overrides(RunConfig.full(Path("/tmp/project")))
    assert cfg.anthropic_api_key == "anthropic-key-from-env"
    assert cfg.proxy_label_model == "claude-opus-5-test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: 'RunConfig' object has no attribute 'anthropic_api_key'`

- [ ] **Step 3: Add the fields to `RunConfig` and the env override table**

In `src/candidate_ranking/config.py`, add two fields to the `RunConfig` dataclass (after `jev_api_key: str`):

```python
    jev_api_key: str
    anthropic_api_key: str = ""
    proxy_label_model: str = "claude-opus-5"
```

In `RunConfig.full()`, add the defaults:

```python
            jev_api_key="",
            anthropic_api_key="",
            proxy_label_model="claude-opus-5",
```

In `_ENV_OVERRIDES`, add:

```python
    "CANDIDATE_RANKING_ANTHROPIC_API_KEY": ("anthropic_api_key", str),
    "CANDIDATE_RANKING_PROXY_LABEL_MODEL": ("proxy_label_model", str),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Add the Anthropic dependency**

In `pyproject.toml`, add to the `dependencies` list (after `"python-dotenv>=1.2.3",`):

```toml
    "anthropic>=0.40.0",
```

Do not add `dspy` — this project does not use it (see Global Constraints).

- [ ] **Step 6: Install and verify the import**

Run: `uv sync` (or `pip install -e .` if not using uv)
Run: `python -c "import anthropic; print(anthropic.__version__)"`
Expected: prints a version string, no `ImportError`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/candidate_ranking/config.py tests/test_config.py
git commit -m "feat: add Anthropic dependency and config for the proxy labeler"
```

---

## Task 2: Requirement triples and criteria-shape validation

**Files:**
- Create: `src/candidate_ranking/evaluation/criteria_optimization.py`
- Test: `tests/evaluation/test_criteria_optimization.py`

**Interfaces:**
- Produces: `REQUIREMENT_LEVEL_COUNT: int = 5`; `RequirementTriple` (pydantic model: `job_description_id: str`, `candidate_id: str`, `requirement: str`); `triple_key(triple: RequirementTriple) -> str`; `validate_criteria_shape(criteria: list[str]) -> None` (raises `ValueError`); `load_requirement_triples(assessments_by_jd: dict[str, dict], requirement_prefix: str = "requirement::") -> list[RequirementTriple]`.

- [ ] **Step 1: Create the test directory and write the failing tests**

Check first whether `tests/scoring/__init__.py` exists (`ls tests/scoring/__init__.py 2>&1`); create `tests/evaluation/__init__.py` only if that convention has one, to match the existing pattern exactly.

Create `tests/evaluation/test_criteria_optimization.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidate_ranking.evaluation.criteria_optimization'`

- [ ] **Step 3: Implement `criteria_optimization.py`**

Create `src/candidate_ranking/evaluation/criteria_optimization.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

REQUIREMENT_LEVEL_COUNT = 5
_REQUIREMENT_KEY_PREFIX = "requirement::"


class RequirementTriple(BaseModel):
    job_description_id: str
    candidate_id: str
    requirement: str


def triple_key(triple: RequirementTriple) -> str:
    return f"{triple.job_description_id}||{triple.candidate_id}||{triple.requirement}"


def validate_criteria_shape(criteria: list[str]) -> None:
    if len(criteria) != REQUIREMENT_LEVEL_COUNT:
        raise ValueError(f"criteria must have exactly {REQUIREMENT_LEVEL_COUNT} levels, got {len(criteria)}")
    for index, level in enumerate(criteria):
        if not level.strip():
            raise ValueError(f"criteria level {index} is empty")


def load_requirement_triples(
    assessments_by_jd: dict[str, dict], requirement_prefix: str = _REQUIREMENT_KEY_PREFIX
) -> list[RequirementTriple]:
    triples: list[RequirementTriple] = []
    for jd_id, candidates in assessments_by_jd.items():
        for candidate_id, assessment in candidates.items():
            for key in assessment.get("confidence", {}):
                if key.startswith(requirement_prefix):
                    triples.append(
                        RequirementTriple(
                            job_description_id=jd_id,
                            candidate_id=candidate_id,
                            requirement=key[len(requirement_prefix):],
                        )
                    )
    return triples
```

Create `src/candidate_ranking/evaluation/__init__.py` if it does not already exist (check first: `ls src/candidate_ranking/evaluation/__init__.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_optimization.py tests/evaluation/
git commit -m "feat: add requirement-triple loading and criteria-shape validation"
```

---

## Task 3: Requirement-question builder and `evaluate_criteria`

**Files:**
- Modify: `src/candidate_ranking/evaluation/criteria_optimization.py`
- Test: `tests/evaluation/test_criteria_optimization.py`

**Interfaces:**
- Consumes: `RequirementTriple`, `validate_criteria_shape` (Task 2); `JevQuestion`, `JevClient` (`candidate_ranking.scoring.jev_client`); `_build_state` from `candidate_ranking.scoring.assessment`; `JobDescription`, `Candidate` (`candidate_ranking.models`).
- Produces: `build_requirement_question(requirement: str, criteria: list[str]) -> JevQuestion`; `CriteriaEvalRecord` (pydantic model: `triple: RequirementTriple`, `score: float`, `confidence: float`); `evaluate_criteria(criteria: list[str], triples: list[RequirementTriple], jds_by_id: dict[str, JobDescription], candidates_by_id: dict[str, Candidate], jev_client: JevClient) -> list[CriteriaEvalRecord]`. Later tasks (metric, hard-case mining, loop) consume `CriteriaEvalRecord`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/evaluation/test_criteria_optimization.py`:

```python
from unittest.mock import Mock

from candidate_ranking.evaluation.criteria_optimization import (
    CriteriaEvalRecord,
    build_requirement_question,
    evaluate_criteria,
)
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.jev_client import JevAnswer


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_requirement_question'`

- [ ] **Step 3: Implement**

Add to `src/candidate_ranking/evaluation/criteria_optimization.py` (new imports at top, functions at bottom):

```python
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.assessment import _build_state
from candidate_ranking.scoring.jev_client import JevClient, JevQuestion


class CriteriaEvalRecord(BaseModel):
    triple: RequirementTriple
    score: float
    confidence: float


def build_requirement_question(requirement: str, criteria: list[str]) -> JevQuestion:
    validate_criteria_shape(criteria)
    return JevQuestion(
        key=f"{_REQUIREMENT_KEY_PREFIX}{requirement}",
        kind="score",
        instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
        criteria=criteria,
    )


def evaluate_criteria(
    criteria: list[str],
    triples: list[RequirementTriple],
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jev_client: JevClient,
) -> list[CriteriaEvalRecord]:
    records: list[CriteriaEvalRecord] = []
    for triple in triples:
        jd = jds_by_id[triple.job_description_id]
        candidate = candidates_by_id[triple.candidate_id]
        state = _build_state(jd, candidate)
        question = build_requirement_question(triple.requirement, criteria)
        answers = jev_client.evaluate(state, [question])
        answer = answers[0]
        records.append(CriteriaEvalRecord(triple=triple, score=float(answer.value), confidence=answer.confidence))
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_optimization.py tests/evaluation/test_criteria_optimization.py
git commit -m "feat: add requirement-question builder and criteria evaluation over Jev"
```

---

## Task 4: Anti-Goodhart metric and confidence-based hard-case mining

**Files:**
- Modify: `src/candidate_ranking/evaluation/criteria_optimization.py`
- Test: `tests/evaluation/test_criteria_optimization.py`

**Interfaces:**
- Consumes: `CriteriaEvalRecord`, `RequirementTriple`, `triple_key` (Task 2/3).
- Produces: `compute_metric(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float`; `select_hard_triples(records: list[CriteriaEvalRecord], k: int) -> list[RequirementTriple]`. Later tasks (optimization loop, Task 8) call both.

- [ ] **Step 1: Write the failing tests**

Add to `tests/evaluation/test_criteria_optimization.py`:

```python
from candidate_ranking.evaluation.criteria_optimization import compute_metric, select_hard_triples


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_metric'`

- [ ] **Step 3: Implement**

Add to `src/candidate_ranking/evaluation/criteria_optimization.py`:

```python
import statistics


def compute_metric(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float:
    if not records:
        return 0.0
    credited_confidences = []
    for record in records:
        label = proxy_labels[triple_key(record.triple)]
        agrees = abs(record.score - label) <= 1
        credited_confidences.append(record.confidence if agrees else 0.0)
    return statistics.mean(credited_confidences)


def select_hard_triples(records: list[CriteriaEvalRecord], k: int) -> list[RequirementTriple]:
    """Confidence-based hard-case mining: the K lowest-confidence triples under the
    current best criteria. A standard, well-established technique -- not an adversarial
    agent, and not claimed as novel on its own (see the design spec's Motivation)."""
    ordered = sorted(records, key=lambda record: record.confidence)
    return [record.triple for record in ordered[:k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_optimization.py tests/evaluation/test_criteria_optimization.py
git commit -m "feat: add anti-Goodhart metric and confidence-based hard-case mining"
```

---

## Task 5: Claude proxy labeler

**Files:**
- Create: `src/candidate_ranking/evaluation/proxy_labeler.py`
- Test: `tests/evaluation/test_proxy_labeler.py`

**Interfaces:**
- Consumes: `RequirementTriple`, `triple_key`, `REQUIREMENT_LEVEL_COUNT` (Task 2); `_REQUIREMENT_FIT_CRITERIA` (`candidate_ranking.scoring.assessment`); `load_json_cache`/`save_json_cache` (`candidate_ranking.json_cache`); `JobDescription`, `Candidate` (`candidate_ranking.models`).
- Produces: `ProxyLabelClient(anthropic_client, cache_path, model)` with `.label(triple, jd, candidate) -> int` (0-4). Later tasks (optimization loop, Task 8; comparison script, Task 10) call `.label(...)` and read `proxy_labels` dicts keyed by `triple_key`.

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/test_proxy_labeler.py`:

```python
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from candidate_ranking.evaluation.criteria_optimization import RequirementTriple, triple_key
from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.models import Candidate, JobDescription


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python.", source_path="jd.pdf")


def _candidate() -> Candidate:
    return Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok",
    )


def _anthropic_response(text: str) -> Mock:
    response = Mock()
    response.content = [Mock(text=text)]
    return response


def test_label_calls_anthropic_and_parses_level(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("3")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    level = labeler.label(triple, _jd(), _candidate())

    assert level == 3
    client.messages.create.assert_called_once()
    assert client.messages.create.call_args.kwargs["model"] == "claude-opus-5"


def test_label_uses_cache_on_second_call(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("2")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    first = labeler.label(triple, _jd(), _candidate())
    second = labeler.label(triple, _jd(), _candidate())

    assert first == second == 2
    client.messages.create.assert_called_once()


def test_label_reuses_cache_across_client_instances(tmp_path):
    cache_path = tmp_path / "cache.json"
    client = Mock()
    client.messages.create.return_value = _anthropic_response("2")
    labeler1 = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")
    labeler1.label(triple, _jd(), _candidate())

    labeler2 = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    result = labeler2.label(triple, _jd(), _candidate())

    assert result == 2
    client.messages.create.assert_called_once()


def test_label_persists_cache_to_disk(tmp_path):
    cache_path = tmp_path / "cache.json"
    client = Mock()
    client.messages.create.return_value = _anthropic_response("1")
    labeler = ProxyLabelClient(client, cache_path=cache_path, model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    labeler.label(triple, _jd(), _candidate())

    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved[triple_key(triple)] == 1


def test_label_raises_on_unparseable_response(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("not a number")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="not parseable"):
        labeler.label(triple, _jd(), _candidate())


def test_label_raises_on_out_of_range_level(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("9")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="out of range"):
        labeler.label(triple, _jd(), _candidate())


def test_label_raises_on_multi_digit_malformed_response(tmp_path):
    client = Mock()
    client.messages.create.return_value = _anthropic_response("10")
    labeler = ProxyLabelClient(client, cache_path=tmp_path / "cache.json", model="claude-opus-5")
    triple = RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")

    with pytest.raises(ValueError, match="out of range"):
        labeler.label(triple, _jd(), _candidate())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_proxy_labeler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidate_ranking.evaluation.proxy_labeler'`

- [ ] **Step 3: Implement**

Create `src/candidate_ranking/evaluation/proxy_labeler.py`. Note the response parsing validates the *whole* response string, not just its first character — a malformed multi-digit reply like `"10"` must raise, not silently become `"1"`:

```python
from __future__ import annotations

from pathlib import Path

from candidate_ranking.evaluation.criteria_optimization import (
    REQUIREMENT_LEVEL_COUNT,
    RequirementTriple,
    triple_key,
)
from candidate_ranking.json_cache import load_json_cache, save_json_cache
from candidate_ranking.models import Candidate, JobDescription
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA


class ProxyLabelClient:
    def __init__(self, anthropic_client, cache_path: Path, model: str) -> None:
        self._client = anthropic_client
        self._cache_path = cache_path
        self._model = model
        self._cache = load_json_cache(cache_path, "proxy label cache")

    def label(self, triple: RequirementTriple, jd: JobDescription, candidate: Candidate) -> int:
        key = triple_key(triple)
        if key in self._cache:
            return self._cache[key]
        level = self._call_model(triple, jd, candidate)
        self._cache[key] = level
        save_json_cache(self._cache_path, self._cache)
        return level

    def _call_model(self, triple: RequirementTriple, jd: JobDescription, candidate: Candidate) -> int:
        rubric = "\n".join(f"{i}: {desc}" for i, desc in enumerate(_REQUIREMENT_FIT_CRITERIA))
        prompt = (
            f"Job Title: {jd.title}\n\nJob Description:\n{jd.raw_text}\n\n"
            f"Candidate CV:\n{candidate.raw_text}\n\n"
            f"Requirement: '{triple.requirement}'\n\n"
            "Rate how well the candidate's CV supports this requirement, using exactly one of these "
            f"{REQUIREMENT_LEVEL_COUNT} levels (reply with only the number, nothing else):\n{rubric}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if not text.isdigit():
            raise ValueError(f"proxy label response not parseable as 0-4: {text!r}")
        level = int(text)
        if not (0 <= level <= REQUIREMENT_LEVEL_COUNT - 1):
            raise ValueError(f"proxy label out of range: {level}")
        return level
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_proxy_labeler.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/proxy_labeler.py tests/evaluation/test_proxy_labeler.py
git commit -m "feat: add Claude proxy labeler with disk cache"
```

---

## Task 6: LangChain-backed criteria proposer

**Files:**
- Create: `src/candidate_ranking/evaluation/criteria_proposer.py`
- Test: `tests/evaluation/test_criteria_proposer.py`

**Interfaces:**
- Consumes: `validate_criteria_shape` (Task 2); `GenerationError`, `invoke_and_validate` (`candidate_ranking.generation`); `BaseChatModel`, `ChatPromptTemplate`, `Runnable` (`langchain_core`).
- Produces: `PROPOSE_CRITERIA_PROMPT` (`ChatPromptTemplate`); `build_criteria_proposer_chain(llm: BaseChatModel) -> Runnable`; `propose_criteria(current_criteria: list[str], hard_case_summaries: list[str], chain: Runnable) -> list[str]` (raises `GenerationError` if the LLM's proposed criteria fail shape validation). Later tasks (optimization loop, Task 8; CLI script, Task 9) call `propose_criteria`; Task 9 builds the `chain` once via `build_criteria_proposer_chain(ChatOllama(...))`.

This follows `src/candidate_ranking/scoring/jd_skills.py`'s existing chain-building pattern exactly — read that file first if unfamiliar with it. Do not use DSPy or any other LLM-orchestration framework (see Global Constraints).

- [ ] **Step 1: Write the failing tests**

Create `tests/evaluation/test_criteria_proposer.py`:

```python
from __future__ import annotations

from unittest.mock import Mock

import pytest

from candidate_ranking.evaluation.criteria_proposer import propose_criteria
from candidate_ranking.generation import GenerationError

_CURRENT = ["never", "rarely", "sometimes", "often", "always"]


def test_propose_criteria_returns_chain_levels():
    chain = Mock()
    chain.invoke.return_value = Mock(levels=["Level A", "Level B", "Level C", "Level D", "Level E"])

    result = propose_criteria(_CURRENT, [], chain)

    assert result == ["Level A", "Level B", "Level C", "Level D", "Level E"]


def test_propose_criteria_passes_current_criteria_and_hard_cases_to_chain():
    chain = Mock()
    chain.invoke.return_value = Mock(levels=["A", "B", "C", "D", "E"])

    propose_criteria(_CURRENT, ["hard case 1"], chain)

    call_payload = chain.invoke.call_args.args[0]
    assert "never" in call_payload["current_criteria"]
    assert "hard case 1" in call_payload["hard_cases"]


def test_propose_criteria_uses_none_yet_placeholder_when_no_hard_cases():
    chain = Mock()
    chain.invoke.return_value = Mock(levels=["A", "B", "C", "D", "E"])

    propose_criteria(_CURRENT, [], chain)

    call_payload = chain.invoke.call_args.args[0]
    assert call_payload["hard_cases"] == "(none yet)"


def test_propose_criteria_rejects_wrong_line_count():
    chain = Mock()
    chain.invoke.return_value = Mock(levels=["Only", "Three", "Lines"])

    with pytest.raises(GenerationError, match="exactly 5"):
        propose_criteria(_CURRENT, [], chain)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_proposer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'candidate_ranking.evaluation.criteria_proposer'`

- [ ] **Step 3: Implement**

Create `src/candidate_ranking/evaluation/criteria_proposer.py`, following `src/candidate_ranking/scoring/jd_skills.py`'s exact chain-building pattern:

```python
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from candidate_ranking.evaluation.criteria_optimization import validate_criteria_shape
from candidate_ranking.generation import GenerationError, invoke_and_validate

PROPOSE_CRITERIA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are designing the criteria for a Score-type question that judges how well a CV supports a "
            "job requirement, for a model that matches evidence against concrete situational descriptions "
            "rather than bare labels.\n\n"
            "Propose a revised 5-level criteria list, one concrete situational description per level, "
            "weakest (level 0) to strongest (level 4). Each level must be a distinct, evidence-based "
            "situation the model can match against CV text -- not a bare label or number. List the 5 "
            "descriptions in the `levels` field, weakest first.",
        ),
        (
            "human",
            "Current 5-level criteria (level 0 to level 4):\n{current_criteria}\n\n"
            "Real cases where the current criteria produced low-confidence, ambiguous judgments:\n{hard_cases}\n\n"
            "Propose a revised 5-level criteria list.",
        ),
    ]
)


class _ProposedCriteria(BaseModel):
    levels: list[str] = Field(min_length=1)


def build_criteria_proposer_chain(llm: BaseChatModel) -> Runnable:
    return PROPOSE_CRITERIA_PROMPT | llm.with_structured_output(_ProposedCriteria)


def propose_criteria(current_criteria: list[str], hard_case_summaries: list[str], chain: Runnable) -> list[str]:
    def validate(result: _ProposedCriteria) -> str | None:
        try:
            validate_criteria_shape(result.levels)
        except ValueError as exc:
            return str(exc)
        return None

    result = invoke_and_validate(
        chain,
        {
            "current_criteria": "\n".join(current_criteria),
            "hard_cases": "\n".join(hard_case_summaries) if hard_case_summaries else "(none yet)",
        },
        _ProposedCriteria,
        validate,
        log_context="criteria proposal",
    )
    return result.levels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_proposer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_proposer.py tests/evaluation/test_criteria_proposer.py
git commit -m "feat: add LangChain-backed criteria proposer"
```

---

## Task 7: Requirement triples from real corpus (script-facing loader)

**Files:**
- Modify: `src/candidate_ranking/evaluation/criteria_optimization.py`
- Test: `tests/evaluation/test_criteria_optimization.py`

**Interfaces:**
- Consumes: `RequirementTriple`, `load_requirement_triples` (Task 2).
- Produces: `load_assessments_by_jd(run_dir: Path, jd_ids: list[str]) -> dict[str, dict]` (reads `run_dir / jd_id / "assessments.json"` for each `jd_id`, matching the existing `load_run_pairs` pattern in `scripts/run_jev_evaluation_study.py`). Later tasks (CLI scripts, Task 9/10) call this then pass the result to `load_requirement_triples`.

- [ ] **Step 1: Write the failing test**

Add to `tests/evaluation/test_criteria_optimization.py`:

```python
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
```

Add `import json` at the top of `tests/evaluation/test_criteria_optimization.py` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_assessments_by_jd'`

- [ ] **Step 3: Implement**

Add to `src/candidate_ranking/evaluation/criteria_optimization.py`:

```python
import json
from pathlib import Path


def load_assessments_by_jd(run_dir: Path, jd_ids: list[str]) -> dict[str, dict]:
    assessments_by_jd: dict[str, dict] = {}
    for jd_id in jd_ids:
        path = run_dir / jd_id / "assessments.json"
        if not path.exists():
            continue
        assessments_by_jd[jd_id] = json.loads(path.read_text(encoding="utf-8"))
    return assessments_by_jd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_optimization.py tests/evaluation/test_criteria_optimization.py
git commit -m "feat: add assessments-by-jd loader for criteria optimization scripts"
```

---

## Task 8: Optimization loop

**Files:**
- Modify: `src/candidate_ranking/evaluation/criteria_optimization.py`
- Test: `tests/evaluation/test_criteria_optimization.py`

**Interfaces:**
- Consumes: `RequirementTriple`, `CriteriaEvalRecord`, `evaluate_criteria`, `compute_metric`, `select_hard_triples` (Tasks 2-4); `JevClient` (`candidate_ranking.scoring.jev_client`); `Candidate`, `JobDescription` (`candidate_ranking.models`).
- Produces: `OptimizationRound` (pydantic model: `round_index: int`, `criteria: list[str]`, `metric: float`); `run_optimization(seed_criteria: list[str], train_triples: list[RequirementTriple], validation_triples: list[RequirementTriple], jds_by_id: dict[str, JobDescription], candidates_by_id: dict[str, Candidate], jev_client: JevClient, proxy_labels: dict[str, int], propose_fn: Callable[[list[str], list[str]], list[str]], max_rounds: int, hard_case_count: int, patience: int) -> tuple[list[str], list[OptimizationRound]]`. Later tasks (CLI script, Task 9) call `run_optimization` with a real `propose_criteria` bound to a real LangChain chain (`build_criteria_proposer_chain`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/evaluation/test_criteria_optimization.py`:

```python
from typing import Callable

from candidate_ranking.evaluation.criteria_optimization import OptimizationRound, run_optimization


def test_run_optimization_returns_best_criteria_and_full_history():
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python.", source_path="jd.pdf")
    candidate = Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok",
    )
    triples = [RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")]
    proxy_labels = {triple_key(triples[0]): 3}

    seed = ["never", "rarely", "sometimes", "often", "always"]
    proposed_round_1 = ["a", "b", "c", "d", "e"]

    jev_client = Mock()
    # Round 0 uses seed criteria (metric-worthy: score matches label, high confidence).
    # Round 1 uses proposed criteria (worse: score matches label, but lower confidence).
    # Validation call after each round reuses the same triple.
    jev_client.evaluate.side_effect = [
        [JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.9)],   # round 0 train
        [JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.9)],   # round 0 validation
        [JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.4)],   # round 1 train
        [JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.4)],   # round 1 validation
    ]

    propose_fn: Callable[[list[str], list[str]], list[str]] = Mock(return_value=proposed_round_1)

    best_criteria, history = run_optimization(
        seed_criteria=seed,
        train_triples=triples,
        validation_triples=triples,
        jds_by_id={"jd-1": jd},
        candidates_by_id={"cand-1": candidate},
        jev_client=jev_client,
        proxy_labels=proxy_labels,
        propose_fn=propose_fn,
        max_rounds=2,
        hard_case_count=1,
        patience=5,
    )

    assert best_criteria == seed  # round 0's higher confidence wins
    assert len(history) == 2
    assert history[0] == OptimizationRound(round_index=0, criteria=seed, metric=pytest.approx(0.9))
    assert history[1] == OptimizationRound(round_index=1, criteria=proposed_round_1, metric=pytest.approx(0.4))


def test_run_optimization_stops_early_after_patience_rounds_without_improvement():
    jd = JobDescription(id="jd-1", title="Backend Engineer", raw_text="Needs Python.", source_path="jd.pdf")
    candidate = Candidate(
        id="cand-1", source_path="cv.pdf", raw_text="I know Python.", num_pages=1, char_count=20,
        parse_status="ok",
    )
    triples = [RequirementTriple(job_description_id="jd-1", candidate_id="cand-1", requirement="Python")]
    proxy_labels = {triple_key(triples[0]): 3}
    seed = ["never", "rarely", "sometimes", "often", "always"]

    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="requirement::Python", kind="score", value=3.0, confidence=0.5)
    ]
    propose_fn = Mock(return_value=seed)

    _best, history = run_optimization(
        seed_criteria=seed,
        train_triples=triples,
        validation_triples=triples,
        jds_by_id={"jd-1": jd},
        candidates_by_id={"cand-1": candidate},
        jev_client=jev_client,
        proxy_labels=proxy_labels,
        propose_fn=propose_fn,
        max_rounds=10,
        hard_case_count=1,
        patience=1,
    )

    assert len(history) == 2  # round 0 (baseline), round 1 (no improvement, patience=1 stops after this)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_optimization'`

- [ ] **Step 3: Implement**

Add to `src/candidate_ranking/evaluation/criteria_optimization.py`:

```python
from typing import Callable


class OptimizationRound(BaseModel):
    round_index: int
    criteria: list[str]
    metric: float


def run_optimization(
    seed_criteria: list[str],
    train_triples: list[RequirementTriple],
    validation_triples: list[RequirementTriple],
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jev_client: JevClient,
    proxy_labels: dict[str, int],
    propose_fn: Callable[[list[str], list[str]], list[str]],
    max_rounds: int,
    hard_case_count: int,
    patience: int,
) -> tuple[list[str], list[OptimizationRound]]:
    history: list[OptimizationRound] = []
    best_metric: float | None = None
    best_criteria = seed_criteria
    hard_triples: list[RequirementTriple] = []
    rounds_without_improvement = 0

    for round_index in range(max_rounds):
        if round_index == 0:
            candidate_criteria = seed_criteria
        else:
            hard_case_summaries = [
                f"requirement={t.requirement}, jd={t.job_description_id}, candidate={t.candidate_id}"
                for t in hard_triples
            ]
            candidate_criteria = propose_fn(best_criteria, hard_case_summaries)

        train_sample = train_triples + hard_triples
        train_records = evaluate_criteria(candidate_criteria, train_sample, jds_by_id, candidates_by_id, jev_client)
        round_metric = compute_metric(train_records, proxy_labels)
        history.append(OptimizationRound(round_index=round_index, criteria=candidate_criteria, metric=round_metric))

        if best_metric is None or round_metric > best_metric:
            best_metric = round_metric
            best_criteria = candidate_criteria
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        validation_records = evaluate_criteria(
            best_criteria, validation_triples, jds_by_id, candidates_by_id, jev_client
        )
        hard_triples = select_hard_triples(validation_records, hard_case_count)

        if rounds_without_improvement >= patience:
            break

    return best_criteria, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_criteria_optimization.py -v`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/evaluation/criteria_optimization.py tests/evaluation/test_criteria_optimization.py
git commit -m "feat: add round-based criteria optimization loop with hard-case mining refresh"
```

---

## Task 9: `scripts/optimize_criteria.py` CLI driver

**Files:**
- Create: `scripts/optimize_criteria.py`
- Test: manual dry-run (this is a CLI wiring script; its logic is already covered by Task 2-8 unit tests, so no new unit test file — matches the existing convention where `run_jev_evaluation_study.py` has no dedicated test file and is validated via `--dry-run`)

**Interfaces:**
- Consumes: everything from Tasks 1-8 (`RunConfig`/`apply_env_overrides`, `load_assessments_by_jd`, `load_requirement_triples`, `run_optimization`, `propose_criteria`, `build_criteria_proposer_chain`, `ProxyLabelClient`, `JevClient`); `load_corpus` (copy the pattern from `scripts/run_jev_evaluation_study.py`); `ChatOllama` (`langchain_ollama`, constructed the same way as `src/candidate_ranking/cli.py:84`).
- Produces: `runs/<run-id>/evaluation/criteria_optimization.json` containing the best criteria and full round history.

- [ ] **Step 1: Write the script**

Create `scripts/optimize_criteria.py`:

```python
"""Run the automated, accuracy-guarded criteria-design optimizer for the
per-requirement Score question, and write the winning criteria plus the full
round history to runs/<run-id>/evaluation/criteria_optimization.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import anthropic
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.evaluation.criteria_optimization import (
    load_assessments_by_jd,
    load_requirement_triples,
    run_optimization,
    triple_key,
)
from candidate_ranking.evaluation.criteria_proposer import build_criteria_proposer_chain, propose_criteria
from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA
from candidate_ranking.scoring.jev_client import JevClient

load_dotenv()

# Re-declared here rather than importing from run_jev_evaluation_study.py: scripts/ is a
# collection of standalone CLIs, not a shared package, so cross-script imports would need a
# second sys.path hack for no real benefit over these ~10 lines.
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JobDescription


def load_corpus(cfg: RunConfig, jd_ids: list[str]) -> tuple[dict[str, JobDescription], dict[str, Candidate]]:
    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})
    return jds_by_id, candidates_by_id


def main(
    run_id: str, max_rounds: int, hard_case_count: int, patience: int,
    train_fraction: float, validation_fraction: float, seed: int, dry_run: bool,
) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jd_ids = manifest["jd_ids"]

    assessments_by_jd = load_assessments_by_jd(run_dir, jd_ids)
    all_triples = load_requirement_triples(assessments_by_jd)
    print(f"Loaded {len(all_triples)} requirement triple(s) from run {run_id}")

    rng = random.Random(seed)
    shuffled = all_triples[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_fraction)
    n_val = int(len(shuffled) * validation_fraction)
    train_triples = shuffled[:n_train]
    validation_triples = shuffled[n_train:n_train + n_val]
    test_triples = shuffled[n_train + n_val:]
    print(f"Split: {len(train_triples)} train, {len(validation_triples)} validation, {len(test_triples)} held-out test")

    if dry_run:
        print(f"Dry run: would run up to {max_rounds} optimization round(s) with hard_case_count={hard_case_count}.")
        return

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY before running.")
    if not cfg.anthropic_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_ANTHROPIC_API_KEY before running.")

    jev_client = JevClient(api_token=cfg.jev_api_key)
    anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    proxy_labeler = ProxyLabelClient(
        anthropic_client, cache_path=cfg.cache_dir / "proxy_labels.json", model=cfg.proxy_label_model
    )
    llm = ChatOllama(model=cfg.ollama_model, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    proposer_chain = build_criteria_proposer_chain(llm)

    jds_by_id, candidates_by_id = load_corpus(cfg, jd_ids)

    proxy_labels: dict[str, int] = {}
    for triple in train_triples + validation_triples + test_triples:
        jd = jds_by_id[triple.job_description_id]
        candidate = candidates_by_id[triple.candidate_id]
        proxy_labels[triple_key(triple)] = proxy_labeler.label(triple, jd, candidate)
    print(f"Collected {len(proxy_labels)} proxy label(s)")

    def bound_propose_fn(current_criteria: list[str], hard_case_summaries: list[str]) -> list[str]:
        return propose_criteria(current_criteria, hard_case_summaries, proposer_chain)

    best_criteria, history = run_optimization(
        seed_criteria=_REQUIREMENT_FIT_CRITERIA,
        train_triples=train_triples,
        validation_triples=validation_triples,
        jds_by_id=jds_by_id,
        candidates_by_id=candidates_by_id,
        jev_client=jev_client,
        proxy_labels=proxy_labels,
        propose_fn=bound_propose_fn,
        max_rounds=max_rounds,
        hard_case_count=hard_case_count,
        patience=patience,
    )

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "criteria_optimization.json"
    out_path.write_text(
        json.dumps(
            {
                "best_criteria": best_criteria,
                "history": [round_.model_dump() for round_ in history],
                "test_triples": [t.model_dump() for t in test_triples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote optimization result to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--hard-case-count", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(
        args.run_id, args.max_rounds, args.hard_case_count, args.patience,
        args.train_fraction, args.validation_fraction, args.seed, args.dry_run,
    )
```

- [ ] **Step 2: Verify the script imports and dry-runs cleanly**

Run: `python scripts/optimize_criteria.py --run-id <an-existing-run-id> --dry-run`
(Use a `run_id` under `runs/` that already has a `manifest.json` and per-jd `assessments.json` from a prior full pipeline run — check `ls runs/` for a candidate.)
Expected: prints the loaded triple count and train/validation/test split sizes, no traceback, exits without calling any API.

- [ ] **Step 3: Commit**

```bash
git add scripts/optimize_criteria.py
git commit -m "feat: add optimize_criteria.py CLI driver"
```

---

## Task 10: `scripts/evaluate_criteria_comparison.py` — final held-out comparison

**Files:**
- Create: `scripts/evaluate_criteria_comparison.py`
- Test: `tests/evaluation/test_evaluate_criteria_comparison.py` (unit-tests the pure comparison function; the script's `main()` is a thin CLI wrapper around it, matching the `run_optimization` / CLI split in Tasks 8-9)

**Interfaces:**
- Consumes: `CriteriaEvalRecord`, `RequirementTriple`, `evaluate_criteria`, `triple_key` (Tasks 2-4); `_REQUIREMENT_FIT_CRITERIA` (`candidate_ranking.scoring.assessment`).
- Produces: `compare_criteria(baseline_records: list[CriteriaEvalRecord], optimized_records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> dict` returning `{"baseline": {"mean_confidence": float, "accuracy": float}, "optimized": {...}, "wilcoxon": {"statistic": float, "p_value": float, "rank_biserial_r": float} | None}`.

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_evaluate_criteria_comparison.py`:

```python
from __future__ import annotations

import pytest

from candidate_ranking.evaluation.criteria_optimization import CriteriaEvalRecord, RequirementTriple, triple_key
from scripts.evaluate_criteria_comparison import compare_criteria


def _record(cand_id: str, score: float, confidence: float) -> CriteriaEvalRecord:
    return CriteriaEvalRecord(
        triple=RequirementTriple(job_description_id="jd-1", candidate_id=cand_id, requirement="Python"),
        score=score,
        confidence=confidence,
    )


def test_compare_criteria_reports_mean_confidence_and_accuracy_for_each_side():
    baseline = [_record("c-1", score=2.0, confidence=0.6), _record("c-2", score=1.0, confidence=0.4)]
    optimized = [_record("c-1", score=3.0, confidence=0.9), _record("c-2", score=3.0, confidence=0.85)]
    proxy_labels = {triple_key(baseline[0].triple): 3, triple_key(baseline[1].triple): 3}

    report = compare_criteria(baseline, optimized, proxy_labels)

    assert report["baseline"]["mean_confidence"] == pytest.approx(0.5)
    assert report["baseline"]["accuracy"] == pytest.approx(0.5)  # only c-1 (2 vs 3) is within 1
    assert report["optimized"]["mean_confidence"] == pytest.approx(0.875)
    assert report["optimized"]["accuracy"] == pytest.approx(1.0)  # both within 1 of label 3
    assert report["wilcoxon"] is not None
    assert "p_value" in report["wilcoxon"]


def test_compare_criteria_returns_none_wilcoxon_for_identical_confidences():
    baseline = [_record("c-1", score=2.0, confidence=0.5), _record("c-2", score=2.0, confidence=0.5)]
    optimized = [_record("c-1", score=2.0, confidence=0.5), _record("c-2", score=2.0, confidence=0.5)]
    proxy_labels = {triple_key(baseline[0].triple): 2, triple_key(baseline[1].triple): 2}

    report = compare_criteria(baseline, optimized, proxy_labels)

    assert report["wilcoxon"] is None  # scipy.wilcoxon raises on all-zero differences
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_evaluate_criteria_comparison.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate_criteria_comparison'` (or `No module named 'scripts'` — if so, create `scripts/__init__.py` as an empty file first, and re-run)

- [ ] **Step 3: Implement**

Create `scripts/evaluate_criteria_comparison.py`:

```python
"""Compare the automatically-optimized per-requirement criteria against the hand-written
baseline on a held-out test split, reporting mean confidence, accuracy against
Claude proxy labels, and a paired Wilcoxon signed-rank test.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scipy.stats import wilcoxon

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.evaluation.criteria_optimization import (
    CriteriaEvalRecord,
    RequirementTriple,
    evaluate_criteria,
    load_assessments_by_jd,
    triple_key,
)
from candidate_ranking.scoring.assessment import _REQUIREMENT_FIT_CRITERIA
from candidate_ranking.scoring.jev_client import JevClient


def _accuracy(records: list[CriteriaEvalRecord], proxy_labels: dict[str, int]) -> float:
    agreements = [abs(r.score - proxy_labels[triple_key(r.triple)]) <= 1 for r in records]
    return statistics.mean(agreements) if agreements else 0.0


def _rank_biserial(first: list[float], second: list[float]) -> float:
    diffs = [b - a for a, b in zip(first, second)]
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def compare_criteria(
    baseline_records: list[CriteriaEvalRecord],
    optimized_records: list[CriteriaEvalRecord],
    proxy_labels: dict[str, int],
) -> dict:
    baseline_confidences = [r.confidence for r in baseline_records]
    optimized_confidences = [r.confidence for r in optimized_records]

    wilcoxon_result = None
    try:
        stat, p_value = wilcoxon(baseline_confidences, optimized_confidences)
        wilcoxon_result = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "rank_biserial_r": _rank_biserial(baseline_confidences, optimized_confidences),
        }
    except ValueError:
        pass  # all-zero differences: scipy.wilcoxon has nothing to test

    return {
        "baseline": {
            "mean_confidence": statistics.mean(baseline_confidences) if baseline_confidences else 0.0,
            "accuracy": _accuracy(baseline_records, proxy_labels),
        },
        "optimized": {
            "mean_confidence": statistics.mean(optimized_confidences) if optimized_confidences else 0.0,
            "accuracy": _accuracy(optimized_records, proxy_labels),
        },
        "wilcoxon": wilcoxon_result,
    }


def main(run_id: str) -> None:
    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    run_dir = cfg.runs_dir / run_id
    opt_path = run_dir / "evaluation" / "criteria_optimization.json"
    opt_result = json.loads(opt_path.read_text(encoding="utf-8"))
    test_triples = [RequirementTriple.model_validate(t) for t in opt_result["test_triples"]]
    optimized_criteria = opt_result["best_criteria"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assessments_by_jd = load_assessments_by_jd(run_dir, manifest["jd_ids"])

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY before running.")
    jev_client = JevClient(api_token=cfg.jev_api_key)

    # jds_by_id / candidates_by_id / proxy_labels loading mirrors optimize_criteria.py's
    # load_corpus + ProxyLabelClient wiring (Task 9) -- wired in Step 5 below.
    raise NotImplementedError(
        "wire jds_by_id, candidates_by_id, and proxy_labels the same way as optimize_criteria.py "
        "(Task 9), then call evaluate_criteria for both _REQUIREMENT_FIT_CRITERIA and "
        "optimized_criteria over test_triples, and compare_criteria() the two record lists."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    main(args.run_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/evaluation/test_evaluate_criteria_comparison.py -v`
Expected: PASS (2 tests) — these cover `compare_criteria` directly and do not exercise `main()`.

- [ ] **Step 5: Finish `main()` by mirroring the Task 9 wiring**

Replace the `raise NotImplementedError(...)` block in `main()` with the same corpus/proxy-label wiring `optimize_criteria.py` already has (copy `load_corpus` from Task 9's script into this one, the same way `run_jev_evaluation_study.py`'s helpers are self-contained per script):

```python
import anthropic

from candidate_ranking.evaluation.proxy_labeler import ProxyLabelClient
from candidate_ranking.ingestion.cv import load_candidates
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import Candidate, JobDescription


def load_corpus(cfg: RunConfig, jd_ids: list[str]) -> tuple[dict[str, JobDescription], dict[str, Candidate]]:
    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir) if jd.id in jd_ids}
    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    cv_skills_cache = json.loads((cfg.cache_dir / "cv_skills.json").read_text(encoding="utf-8"))
    for cid, candidate in candidates_by_id.items():
        if cid in cv_skills_cache:
            candidates_by_id[cid] = candidate.model_copy(update={"skills": cv_skills_cache[cid]["skills"]})
    return jds_by_id, candidates_by_id
```

Then in `main()`, replace the `raise NotImplementedError(...)` with:

```python
    if not cfg.anthropic_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_ANTHROPIC_API_KEY before running.")
    anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    proxy_labeler = ProxyLabelClient(
        anthropic_client, cache_path=cfg.cache_dir / "proxy_labels.json", model=cfg.proxy_label_model
    )

    jds_by_id, candidates_by_id = load_corpus(cfg, manifest["jd_ids"])
    proxy_labels = {
        triple_key(t): proxy_labeler.label(t, jds_by_id[t.job_description_id], candidates_by_id[t.candidate_id])
        for t in test_triples
    }

    baseline_records = evaluate_criteria(
        _REQUIREMENT_FIT_CRITERIA, test_triples, jds_by_id, candidates_by_id, jev_client
    )
    optimized_records = evaluate_criteria(
        optimized_criteria, test_triples, jds_by_id, candidates_by_id, jev_client
    )

    report = compare_criteria(baseline_records, optimized_records, proxy_labels)
    out_path = run_dir / "evaluation" / "criteria_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote comparison to {out_path}")
```

- [ ] **Step 6: Run the full test suite once more**

Run: `pytest tests/ -v`
Expected: all tests PASS, including the two `test_evaluate_criteria_comparison.py` tests (still testing `compare_criteria` only, `main()` remains integration-only per the existing scripts convention).

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_criteria_comparison.py tests/evaluation/test_evaluate_criteria_comparison.py scripts/__init__.py
git commit -m "feat: add held-out baseline-vs-optimized criteria comparison script"
```

---

## Task 11: Paper results subsection

**Files:**
- Modify: `docs/paper2_jev.tex`

**Interfaces:**
- Consumes: the `criteria_comparison.json` report produced by Task 10, once a real optimization + comparison run has been executed against the live Jev/Anthropic APIs (this task only prepares the LaTeX structure and placeholder-free prose; it does not itself run the optimizer — that is a research execution step outside this plan's automated test cycle).

- [ ] **Step 1: Add the new results subsection**

In `docs/paper2_jev.tex`, after the existing §V-B (Criteria-Grounded Question Design results), add a new subsection reporting the held-out comparison. Do not fill in numeric results yet — that requires an actual run of `scripts/optimize_criteria.py` followed by `scripts/evaluate_criteria_comparison.py` against the real Jev and Anthropic APIs, which is a research execution step, not a code-writing step. Leave a clearly-marked run instruction as a LaTeX comment (not prose in the paper body — this is the one acceptable place for a placeholder, since it is an instruction to the paper's author, not paper text). Keep the wording plainly scoped to what was built — no "red-team" or adversarial language, and no claim that criteria-grounded design itself is this paper's invention:

```latex
\subsection{Automated, Accuracy-Guarded Criteria Optimization}
% RUN BEFORE FILLING IN THIS SECTION:
%   1. python scripts/optimize_criteria.py --run-id <run-id>
%   2. python scripts/evaluate_criteria_comparison.py --run-id <run-id>
%   3. Read runs/<run-id>/evaluation/criteria_comparison.json for the numbers below.
Manually written criteria (Section~III-B) were validated once, after the fact,
following TypeSafe's own guidance for writing Score-question criteria. This
section reports an automated alternative: a round-based propose/evaluate loop
(Section~III-C) that revises the per-requirement criteria under a
confidence-plus-independent-proxy-label metric, guarded against rewarding
confident-but-wrong answers, refreshed each round by mining the lowest-
confidence validation pairs -- a standard technique, not claimed as novel on
its own. What is reported here as new is the automated optimization method
itself: no public source on Jev describes a process for automatically
discovering better Score-question criteria, only static human-facing
guidance for writing them by hand.
```

- [ ] **Step 2: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: add paper subsection structure for automated criteria optimization results"
```

---

## Post-Implementation: Research Execution (not part of this code plan)

Once Tasks 1-11 are merged, running the actual optimization and filling in the paper's numbers requires:
1. A live Jev API key, Anthropic API key, and a running Ollama server with `qwen2.5:14b-instruct-q4_K_M` pulled.
2. `python scripts/optimize_criteria.py --run-id <run-id>` (real run, not `--dry-run`).
3. `python scripts/evaluate_criteria_comparison.py --run-id <run-id>`.
4. Transcribing `runs/<run-id>/evaluation/criteria_comparison.json` into `docs/paper2_jev.tex`'s new subsection (Task 11) and into a new results table alongside `tab_score_ablation`.

This is a data-collection step with real API cost, not a coding task — it happens after this plan's tasks are implemented and reviewed, not as one of them.
