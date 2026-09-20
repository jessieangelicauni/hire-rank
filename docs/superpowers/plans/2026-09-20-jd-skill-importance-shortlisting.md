# Must-Have / Nice-to-Have Skill Shortlisting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `shortlist_candidates` from admitting a candidate on peripheral/nice-to-have skills alone by classifying each JD's technical skills as must-have or nice-to-have during extraction, and gating shortlisting on a separate must-have match threshold.

**Architecture:** Extend the existing JD-skill-extraction LLM call (one prompt, one schema, one call — no new call) to also emit `must_have_skills`, a validated subset of `technical_skills`. Thread a new `min_must_have_matches` threshold through `shortlist_candidates` → the LangGraph pipeline → the CLI, mirroring the existing `min_skill_matches` plumbing exactly. `must_have_skills` defaults to empty everywhere, so every existing caller's behavior is unchanged until the new field is populated.

**Tech Stack:** Python 3.12, Pydantic v2, LangChain (`ChatPromptTemplate`, `with_structured_output`), NumPy (cosine similarity via dot product on pre-normalized vectors), pytest.

## Global Constraints

- `must_have_skills` must always be a subset of `technical_skills` — enforced by filtering in `generate_jd_skills`, not left to the LLM's output to guarantee.
- No new LLM call: the must-have classification is added to the existing JD-skill-extraction prompt/schema in `jd_skills.py`.
- `technical_skills` itself, and every existing consumer of it (`shortlist_candidates`'s total-match count, `assessment.py`'s per-required Score question generation, `ragas_eval.py`'s negation bridging), must remain unchanged.
- `min_must_have_matches` follows the exact same "shrink to fit" rule already used for `min_matches`: `required_must_have = min(min_must_have_matches, len(must_have_normalized))`, so a JD with fewer must-have skills than the configured threshold never becomes unshortlistable.
- Empty `must_have_skills` (the default) must reproduce today's shortlisting behavior exactly (`required_must_have` collapses to 0).
- Do not touch `src/candidate_ranking/scoring/assessment.py`, `scripts/run_jev_evaluation_study.py`, `scripts/analyze_jev_evaluation_study.py`, or `docs/paper2_jev.tex` — out of scope per the design spec.

---

### Task 1: Add `must_have_skills` to the `JDSkills` model

**Files:**
- Modify: `src/candidate_ranking/models.py:31-37`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `JDSkills.must_have_skills: list[str]` (default `[]`), consumed by Task 2 (`generate_jd_skills`) and Task 3 (`shortlist_candidates`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models.py` (near the existing `test_jd_skills_*` tests around line 86):

```python
def test_jd_skills_accepts_must_have_skills():
    jd_skills = JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python"],
    )
    assert jd_skills.must_have_skills == ["Python"]


def test_jd_skills_must_have_skills_defaults_empty():
    jd_skills = JDSkills(job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=["Python"])
    assert jd_skills.must_have_skills == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -k must_have_skills -v`
Expected: FAIL — `test_jd_skills_accepts_must_have_skills` raises because `must_have_skills` is not a recognized field (Pydantic silently drops unknown kwargs by default, so `jd_skills.must_have_skills` raises `AttributeError: 'JDSkills' object has no attribute 'must_have_skills'`).

- [ ] **Step 3: Add the field**

In `src/candidate_ranking/models.py`, change:

```python
class JDSkills(BaseModel):
    job_description_id: str
    generated_by_model: str
    technical_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    education_requirement: str | None = None
```

to:

```python
class JDSkills(BaseModel):
    job_description_id: str
    generated_by_model: str
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    education_requirement: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -k must_have_skills -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full model test file to check for regressions**

Run: `pytest tests/test_models.py -v`
Expected: all tests pass (no existing test constructs `JDSkills` positionally, so adding a field with a default cannot break them)

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/models.py tests/test_models.py
git commit -m "feat: add must_have_skills field to JDSkills"
```

---

### Task 2: Classify must-have skills during JD extraction

**Files:**
- Modify: `src/candidate_ranking/scoring/jd_skills.py`
- Test: `tests/scoring/test_jd_skills.py`

**Interfaces:**
- Consumes: `JDSkills.must_have_skills` (Task 1)
- Produces: `generate_jd_skills(jd, chain, model_name) -> JDSkills` now populates `must_have_skills`, filtered to a verified subset of `technical_skills`. `_GeneratedSkills.must_have_skills: list[str]` is the new raw LLM-output field consumed by that filter.

- [ ] **Step 1: Write the failing tests**

Add to `tests/scoring/test_jd_skills.py`:

```python
def test_generate_jd_skills_extracts_must_have_skills():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python"],
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == ["Python"]


def test_generate_jd_skills_filters_must_have_not_in_technical_skills(caplog):
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python", "SQL"],
        must_have_skills=["Python", "Rust"],
    )

    with caplog.at_level("WARNING"):
        jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == ["Python"]
    assert "Rust" in caplog.text


def test_generate_jd_skills_must_have_skills_defaults_empty():
    chain = Mock()
    chain.invoke.return_value = _GeneratedSkills(
        reasoning="analysis",
        technical_skills=["Python"],
    )

    jd_skills = generate_jd_skills(_jd(), chain, "qwen2.5:14b")

    assert jd_skills.must_have_skills == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scoring/test_jd_skills.py -k must_have -v`
Expected: FAIL — `test_generate_jd_skills_extracts_must_have_skills` fails with `AssertionError: assert [] == ['Python']` (since `_GeneratedSkills` silently drops the unrecognized `must_have_skills` kwarg today, and `generate_jd_skills` never reads it even once added, so the result defaults to `[]`).

- [ ] **Step 3: Update the prompt, schema, and filtering logic**

In `src/candidate_ranking/scoring/jd_skills.py`, replace the `JD_SKILL_EXTRACTION_PROMPT` system message:

```python
JD_SKILL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert HR consultant extracting structured requirements a job description asks for.\n\n"
            "Task:\n"
            "- Read the job title and description and identify every technology and technical subject explicitly mentioned.\n"
            "- Reason through the description section by section in the `reasoning` field before giving your final answer.\n"
            "- Note which text backs each skill, certification, seniority requirement, or education requirement you identify.\n"
            "- List the identified skills in the `technical_skills` field.\n"
            "- Of those, list in `must_have_skills` the subset the description states or clearly implies is required, mandatory, or essential; a skill mentioned as a plus, preferred, or with no importance qualifier at all is not must-have.\n"
            "- List any named professional certifications (e.g. \"AWS Certified Solutions Architect\", \"PMP\") in the `certifications` field.\n"
            "- If the description states a seniority or years-of-experience requirement, summarize it in one sentence in the `seniority_requirement` field; otherwise leave it null.\n"
            "- If the description states an education requirement, summarize it in one sentence in the `education_requirement` field; otherwise leave it null.\n\n"
            "Constraints:\n"
            "- Name each skill as the single atomic technology or subject it refers to.\n"
            "- Write each skill the way it would appear as a standalone item on a resume.\n"
            "- Do not phrase a skill as a description of proficiency, usage, or context.\n"
            "- Omit generic process or methodology phrases that do not name a specific technology or subject.\n"
            "- Every entry in `must_have_skills` must also appear in `technical_skills`; do not name a must-have skill that isn't already in that list.\n"
            "- Do not name a certification as a technical skill, or a technical skill as a certification.\n"
            "- A degree requirement belongs only in `education_requirement`, never in `technical_skills`.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)
```

Update `_GeneratedSkills`:

```python
class _GeneratedSkills(BaseModel):
    reasoning: str = Field(min_length=1)
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    education_requirement: str | None = None
```

Add a filter function above `generate_jd_skills`:

```python
def _filter_must_have_skills(must_have_skills: list[str], technical_skills: list[str], jd_id: str) -> list[str]:
    technical_normalized = {s.strip().lower() for s in technical_skills}
    filtered: list[str] = []
    for skill in must_have_skills:
        if skill.strip().lower() in technical_normalized:
            filtered.append(skill)
        else:
            logger.warning(
                "Discarding must-have skill %r for JD %s: not present in extracted technical_skills",
                skill, jd_id,
            )
    return filtered
```

Update `generate_jd_skills`'s return statement:

```python
    return JDSkills(
        job_description_id=jd.id,
        generated_by_model=model_name,
        technical_skills=result.technical_skills,
        must_have_skills=_filter_must_have_skills(result.must_have_skills, result.technical_skills, jd.id),
        certifications=result.certifications,
        seniority_requirement=result.seniority_requirement,
        education_requirement=result.education_requirement,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scoring/test_jd_skills.py -v`
Expected: all pass (5 passed — 2 pre-existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/scoring/jd_skills.py tests/scoring/test_jd_skills.py
git commit -m "feat: classify must-have skills during JD extraction"
```

---

### Task 3: Gate shortlisting on a must-have match threshold

**Files:**
- Modify: `src/candidate_ranking/scoring/skills.py:139-162`
- Test: `tests/scoring/test_skills.py` (new file)

**Interfaces:**
- Consumes: `JDSkills.must_have_skills` (Task 1), `build_candidate_skill_index(candidates, embedder) -> (np.ndarray, list[tuple[str, str]])` (existing, unchanged)
- Produces: `shortlist_candidates(jd, jd_skills, index, row_map, embedder, threshold=0.8, min_matches=2, min_must_have_matches=2) -> list[str]` — new `min_must_have_matches` parameter, consumed by Task 4 (`pipeline.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/scoring/test_skills.py`:

```python
from __future__ import annotations

from typing import Callable

import numpy as np

from candidate_ranking.models import Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.skills import build_candidate_skill_index, shortlist_candidates

_JD_SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"]
_MUST_HAVE = ["Python", "SQL"]


def _one_hot_embedder(vocab: list[str]) -> Callable[[list[str]], np.ndarray]:
    """Deterministic stand-in for a real sentence-transformer embedder: an exact
    (case-insensitive) string match gets cosine similarity 1.0, anything else 0.0."""
    index = {v.strip().lower(): i for i, v in enumerate(vocab)}
    dim = len(index)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            key = text.strip().lower()
            if key in index:
                vectors[row, index[key]] = 1.0
        return vectors

    return embed


def _jd() -> JobDescription:
    return JobDescription(id="jd-1", title="Backend Engineer", raw_text="...", source_path="jd.pdf")


def _jd_skills(must_have: list[str]) -> JDSkills:
    return JDSkills(
        job_description_id="jd-1",
        generated_by_model="qwen2.5:14b",
        technical_skills=_JD_SKILLS,
        must_have_skills=must_have,
    )


def _candidate(candidate_id: str, skills: list[str]) -> Candidate:
    return Candidate(
        id=candidate_id, source_path="cv.pdf", raw_text="...", num_pages=1, char_count=10,
        parse_status="ok", skills=skills,
    )


def test_shortlist_excludes_candidate_who_meets_total_but_not_must_have():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_excludes_candidate_who_meets_must_have_but_not_total():
    core_only = _candidate("core-only", ["Python", "SQL", "AWS"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([core_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == []


def test_shortlist_includes_candidate_who_meets_both():
    both = _candidate("both", ["Python", "SQL", "AWS", "Docker", "Kubernetes"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([both], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills(_MUST_HAVE), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["both"]


def test_shortlist_empty_must_have_skills_preserves_prior_behavior():
    peripheral_only = _candidate("peripheral-only", ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins"])
    embedder = _one_hot_embedder(_JD_SKILLS)
    index, row_map = build_candidate_skill_index([peripheral_only], embedder)

    shortlisted = shortlist_candidates(
        _jd(), _jd_skills([]), index, row_map, embedder,
        min_matches=5, min_must_have_matches=2,
    )

    assert shortlisted == ["peripheral-only"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scoring/test_skills.py -v`
Expected: FAIL — `TypeError: shortlist_candidates() got an unexpected keyword argument 'min_must_have_matches'` on every test.

- [ ] **Step 3: Implement the must-have gate**

In `src/candidate_ranking/scoring/skills.py`, replace `shortlist_candidates`:

```python
def shortlist_candidates(
    jd: JobDescription,
    jd_skills: JDSkills,
    index: np.ndarray,
    row_map: list[tuple[str, str]],
    embedder: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
    min_matches: int = 2,
    min_must_have_matches: int = 2,
) -> list[str]:
    if not jd_skills.technical_skills:
        logger.warning("JD %s has no extracted technical skills; shortlist is empty", jd.id)
        return []

    matched = match_candidate_skills(jd_skills, index, row_map, embedder, threshold)

    deduplicated_count = len({s.strip().lower() for s in jd_skills.technical_skills})
    required = min(min_matches, deduplicated_count)

    must_have_normalized = {s.strip().lower() for s in jd_skills.must_have_skills}
    required_must_have = min(min_must_have_matches, len(must_have_normalized))

    def must_have_match_count(skills: set[str]) -> int:
        return sum(1 for skill in skills if skill.strip().lower() in must_have_normalized)

    shortlisted = sorted(
        candidate_id
        for candidate_id, skills in matched.items()
        if len(skills) >= required and must_have_match_count(skills) >= required_must_have
    )
    if not shortlisted:
        logger.warning(
            "JD %s's skill-match shortlist is empty at threshold=%s, required matches=%s, "
            "required must-have matches=%s", jd.id, threshold, required, required_must_have,
        )
    return shortlisted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scoring/test_skills.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: all pass (no other file calls `shortlist_candidates` positionally past `min_matches`, so the new keyword-only-by-convention parameter with a default cannot break existing callers)

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/scoring/skills.py tests/scoring/test_skills.py
git commit -m "feat: gate shortlisting on a must-have skill match threshold"
```

---

### Task 4: Wire `min_must_have_matches` through the pipeline and CLI

**Files:**
- Modify: `src/candidate_ranking/graphs/pipeline.py:53-100`
- Modify: `src/candidate_ranking/cli.py:64-219`
- Test: `tests/graphs/test_pipeline.py`

**Interfaces:**
- Consumes: `shortlist_candidates(..., min_must_have_matches=...)` (Task 3)
- Produces: `build_pipeline_graph(..., min_must_have_matches: int = 2)`, `run(..., min_must_have_matches: int = 2)`, CLI flag `--min-must-have-matches`, manifest key `"min_must_have_matches"` — nothing downstream of this task consumes these; this is the outermost layer.

- [ ] **Step 1: Write the failing test**

Add to `tests/graphs/test_pipeline.py`:

```python
def test_build_pipeline_graph_accepts_min_must_have_matches(tmp_path: Path):
    cfg = RunConfig.full(tmp_path)
    graph = build_pipeline_graph(
        cfg,
        jd_skills_chain=None,
        jev_client=None,
        skill_index=np.zeros(0),
        skill_row_map=[],
        skill_embedder=lambda texts: np.zeros((len(texts), 0)),
        run_id="run-1",
        min_must_have_matches=3,
    )

    assert "build_shortlist_for_jd" in set(graph.nodes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/graphs/test_pipeline.py -k min_must_have_matches -v`
Expected: FAIL — `TypeError: build_pipeline_graph() got an unexpected keyword argument 'min_must_have_matches'`

- [ ] **Step 3: Thread the parameter through `pipeline.py`**

In `src/candidate_ranking/graphs/pipeline.py`, change the `build_pipeline_graph` signature:

```python
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
    min_must_have_matches: int = 2,
) -> StateGraph:
```

and update `build_shortlist_for_jd`:

```python
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
            min_must_have_matches=min_must_have_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/graphs/test_pipeline.py -v`
Expected: all pass

- [ ] **Step 5: Thread the parameter through `cli.py`**

In `src/candidate_ranking/cli.py`, update the `run` signature (currently at line 64):

```python
def run(
    run_id: str | None = None,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
) -> Path:
```

Update the `build_pipeline_graph` call (currently lines 99-109):

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
        min_must_have_matches=min_must_have_matches,
    )
```

Update the empty-shortlist warning (currently lines 126-133):

```python
    for jd in jds:
        if len(final_state["shortlists"].get(jd.id, [])) == 0:
            print(
                f"WARNING: JD {jd.id}'s skill-match shortlist is empty (skill_match_threshold="
                f"{skill_match_threshold}, min_skill_matches={min_skill_matches}, "
                f"min_must_have_matches={min_must_have_matches}); no candidates will be "
                "assessed for this JD.",
                file=sys.stderr,
            )
```

Update the resume-from-manifest block (currently lines 135-147):

```python
    manifest_path = cfg.runs_dir / run_id / "manifest.json"
    if resuming and manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            skill_match_threshold = prior_manifest.get("skill_match_threshold", skill_match_threshold)
            min_skill_matches = prior_manifest.get("min_skill_matches", min_skill_matches)
            min_must_have_matches = prior_manifest.get("min_must_have_matches", min_must_have_matches)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"WARNING: could not read prior manifest at {manifest_path} to preserve original "
                f"skill-shortlist thresholds on resume; using the values passed to this invocation "
                f"instead: {exc}",
                file=sys.stderr,
            )
```

Update the manifest dict (currently lines 149-161):

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
        "min_must_have_matches": min_must_have_matches,
        "shortlist_sizes": {jd_id: len(cids) for jd_id, cids in final_state["shortlists"].items()},
        "jd_ids": [jd.id for jd in jds],
        "candidate_ids": [c.id for c in candidates],
    }
```

Update the zero-shortlisted error (currently lines 166-173):

```python
    total_shortlisted = sum(len(v) for v in final_state["shortlists"].values())
    if total_shortlisted == 0:
        raise RuntimeError(
            f"Run produced zero shortlisted candidates across all {len(jds)} JDs at "
            f"skill_match_threshold={skill_match_threshold}, min_skill_matches={min_skill_matches}, "
            f"min_must_have_matches={min_must_have_matches} -- no "
            "assessments were generated. Check extracted skill vocabularies or loosen the thresholds. "
            f"Run output (manifest, empty summary) was still written to {run_dir}."
        )
```

Add the CLI argument (currently after the `--min-skill-matches` block, around line 211):

```python
    run_parser.add_argument(
        "--min-must-have-matches",
        type=int,
        default=2,
        help=(
            "Number of a JD's must-have technical skills a candidate must match to be shortlisted for "
            "assessment (default: 2; automatically reduced to the JD's total must-have-skill count if "
            "that's smaller; has no effect if the JD has no classified must-have skills)."
        ),
    )
```

Update the `run(...)` call in `main()` (currently lines 214-219):

```python
    if args.command == "run":
        run_dir = run(
            run_id=args.run_id,
            skill_match_threshold=args.skill_match_threshold,
            min_skill_matches=args.min_skill_matches,
            min_must_have_matches=args.min_must_have_matches,
        )
        print(f"Run complete: {run_dir / 'summary.json'}")
```

- [ ] **Step 6: Verify the CLI wires up correctly**

Run: `python -m candidate_ranking.cli run --help`
Expected: help text includes a `--min-must-have-matches` line with the help string written above, alongside the existing `--min-skill-matches` line.

- [ ] **Step 7: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/candidate_ranking/graphs/pipeline.py src/candidate_ranking/cli.py tests/graphs/test_pipeline.py
git commit -m "feat: wire min_must_have_matches through the pipeline and CLI"
```
