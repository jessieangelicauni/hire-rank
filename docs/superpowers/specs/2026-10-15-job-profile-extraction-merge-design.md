# Merge JD Requirements + Skills Extraction into One Job Profile Call — Design

Date: 2026-10-15

## Motivation

Today, extracting structured data from a job description takes two
separate LLM calls per JD:

- `scoring/jd_requirements.py` — `generate_jd_requirements` — extracts
  `must_have`/`nice_to_have` requirement items (free text + context).
- `scoring/jd_skills.py` — `generate_jd_skills` — extracts
  `technical_skills`, `must_have_skills`, `certifications`,
  `seniority_requirement`, `seniority_min_years`, `education_requirement`.

Both read the exact same input (`jd.title`, `jd.raw_text`). Auditing
actual consumers (grep across `src/`) found that `certifications`,
`seniority_requirement`, `seniority_min_years`, and `education_requirement`
on `JDSkills` have zero downstream readers — they're generated and
cached but never read by `shortlist_candidates`, the qualification
gate, or any output writer. The facts they'd carry (degree, years of
experience) were already captured as ordinary items in
`JDRequirements.must_have`/`nice_to_have`, which the qualification gate
(`llm_assessment.py`) does read. So today's two-call design pays for a
second LLM call partly to re-derive information the first call already
produced in a form nothing downstream uses.

This design merges both extractions into one LLM call producing one
`JobProfile`, cutting LLM calls for JD processing from 2 to 1 per JD,
and drops the four unused fields instead of carrying them forward.
(`jd_requirements.py`'s retry-on-empty-list loop was already removed in
a prior change, uncommitted in the working tree at the start of this
work — this design carries that same single-shot-no-retry approach into
the merged call, and removes the equivalent retry loop from
`jd_skills.py` too.)

## Scope

**In scope:**
- `src/candidate_ranking/models.py` — remove `JDRequirement`,
  `JDRequirements`, `JDSkills`; add `JobProfile`
- New `src/candidate_ranking/scoring/job_profile.py` — replaces
  `jd_requirements.py` and `jd_skills.py`
- `src/candidate_ranking/scoring/skills.py` — `shortlist_candidates`
  and `match_candidate_skills` take `JobProfile`, read
  `must_have_skills`/`nice_to_have_skills` instead of
  `technical_skills`/`must_have_skills`
- `src/candidate_ranking/scoring/llm_assessment.py` — qualification
  gate's `must_have_json`/`nice_to_have_json` built from
  `profile.must_have_skills`/`nice_to_have_skills` only
- `src/candidate_ranking/graphs/pipeline.py` — one extraction node,
  one cache path, replaces the two existing fanned-out nodes
- `src/candidate_ranking/cli.py` — one chain replaces two
- Tests for all of the above

**Out of scope:**
- `scoring/active_listwise_ranking.py` — still reads
  `Assessment.must_have_evaluations`/`nice_to_have_evaluations`
  exactly as today; this design does not change what ranking consumes,
  only what feeds the gate that produces those evaluations.
- The qualification gate's disqualification flow — already removed
  from pipeline control flow in `085a31a`; not revisited here.
  `qualified`/`disqualification_reason` remain informational fields on
  `Assessment`.
- `education_requirement`, `seniority_requirement`,
  `other_requirements` are carried on `JobProfile` for display only —
  not sent to the qualification gate. (Decided explicitly: these have
  no required/preferred marker, so there is no principled way to route
  them into the gate's non-compensatory MUST_HAVE evaluation; doing so
  risks repeating the over-strict-disqualification failure mode that
  `085a31a` already had to work around.)
- `bridge_negated_jd_skills_to_candidate` (`skills.py`) — already
  unused by any caller; signature (`jd_technical_skills: list[str]`)
  is unchanged by this design, a future caller passes
  `must_have_skills + nice_to_have_skills`.

## Data Model

Replace `JDRequirement`, `JDRequirements`, `JDSkills` with one model:

```python
class JobProfile(BaseModel):
    job_description_id: str
    generated_by_model: str
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)
```

`must_have_skills` and `nice_to_have_skills` are a disjoint partition
of what `technical_skills` used to be — every extracted skill lands in
exactly one, never both. (Previously `must_have_skills` was a subset
*also* duplicated inside `technical_skills`.) Anywhere the old code
needs "all skills," it uses `must_have_skills + nice_to_have_skills`.

## Extraction Prompt & Schema (`job_profile.py`)

```python
JOB_PROFILE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior technical recruiter extracting a structured job profile "
            "from a job description, for candidate evaluation and skill matching.\n\n"
            "In `reasoning` (max 3 sentences), note anything non-obvious before answering.\n\n"
            "Skills → list every named technology, split into:\n"
            "- `must_have_skills`: the core the role is built around — main language(s) plus "
            "1-3 defining frameworks or platforms. Infra/tooling (Docker, cloud, CI/CD) is not "
            "core unless it IS the role.\n"
            "- `nice_to_have_skills`: every other named technology. Most skills belong here.\n\n"
            "Non-skill requirements:\n"
            "- `education_requirement`: one sentence summary if the JD states one, else null.\n"
            "- `seniority_requirement`: one sentence summary of seniority/years-of-experience "
            "if stated, else null.\n"
            "- `other_requirements`: anything else the JD requires or prefers (certifications, "
            "languages, domain knowledge, etc.), each as a short phrase.",
        ),
        ("human", "Job Title: {title}\n\nJob Description:\n{description}"),
    ]
)


class _JobProfileOutput(BaseModel):
    reasoning: str = Field(min_length=1)
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    seniority_requirement: str | None = None
    other_requirements: list[str] = Field(default_factory=list)


class JobProfileGenerationError(Exception):
    pass


def build_job_profile_chain(llm: BaseChatModel) -> Runnable:
    return JOB_PROFILE_PROMPT | llm.with_structured_output(_JobProfileOutput)


def generate_job_profile(jd: JobDescription, chain: Runnable, model_name: str) -> JobProfile:
    try:
        result = invoke_and_validate(
            chain,
            {"title": jd.title, "description": jd.raw_text},
            _JobProfileOutput,
            lambda _: None,
            log_context=f"Job profile extraction for {jd.id}",
        )
    except GenerationError as exc:
        raise JobProfileGenerationError(str(exc)) from exc

    return JobProfile(
        job_description_id=jd.id,
        generated_by_model=model_name,
        must_have_skills=result.must_have_skills,
        nice_to_have_skills=result.nice_to_have_skills,
        education_requirement=result.education_requirement,
        seniority_requirement=result.seniority_requirement,
        other_requirements=result.other_requirements,
    )
```

Single-shot, no retry loop — matches the already-applied change to
`jd_requirements.py` and removes the equivalent loop currently in
`jd_skills.py` (`for attempt in range(2): ...`). A
`JobProfileGenerationError` on the one attempt causes that JD to be
skipped (same behavior the pipeline already has for
`JDRequirementsGenerationError`/`JDSkillsGenerationError` today).

`load_or_generate_job_profile` follows the same cache pattern as the
two functions it replaces (hash of formatted prompt + model name →
cache key), writing to one cache file, `job_profile.json`, instead of
`jd_requirements.json` + `jd_skills.json`.

## Downstream Changes

### `skills.py`

`match_candidate_skills(jd_skills: JDSkills, ...)` →
`match_candidate_skills(profile: JobProfile, ...)`, reading
`profile.must_have_skills + profile.nice_to_have_skills` everywhere it
used to read `jd_skills.technical_skills`.

`shortlist_candidates(jd, jd_skills: JDSkills, ...)` →
`shortlist_candidates(jd, profile: JobProfile, ...)`. The two
threshold computations are unchanged in logic, only the source field
names move:

```python
all_skills = profile.must_have_skills + profile.nice_to_have_skills
deduplicated_count = len({s.strip().lower() for s in all_skills})
required = min(min_matches, deduplicated_count)

must_have_normalized = {s.strip().lower() for s in profile.must_have_skills}
required_must_have = min(min_must_have_matches, len(must_have_normalized))
```

The empty-skills guard (`if not jd_skills.technical_skills: return []`)
becomes `if not all_skills: return []`.

### `llm_assessment.py`

`generate_assessment(jd, candidate, jd_requirements: JDRequirements | None, ...)`
→ `generate_assessment(jd, candidate, profile: JobProfile | None, ...)`.
Build the gate's requirement JSON directly from the skill lists —
no `context` field is available for a bare skill name, so each item's
`context` is a fixed placeholder rather than LLM-generated text:

```python
must_have_json = json.dumps(
    [{"requirement": s, "context": "Core skill required by the job description"} for s in profile.must_have_skills]
    if profile else []
)
nice_to_have_json = json.dumps(
    [{"requirement": s, "context": "Skill mentioned in the job description"} for s in profile.nice_to_have_skills]
    if profile else []
)
```

Everything else in the qualification gate (evidence rules, verdict
mapping, non-compensatory rule, `_QualificationOutput`) is unchanged —
it already only ever consumed `must_have_json`/`nice_to_have_json` as
opaque requirement text, not anything specific to
`JDRequirement.context`'s prior free-text content.

### `pipeline.py`

Replace the two fanned-out nodes (`extract_skills_for_jd`,
`extract_requirements_for_jd`) with one:

```python
def extract_job_profile_for_jd(payload: dict) -> dict:
    jd = payload["jd"]
    try:
        profile = load_or_generate_job_profile(jd, job_profile_chain, cfg.ollama_model, job_profile_cache_path)
    except JobProfileGenerationError as exc:
        logger.warning("Excluding %s from this run: %s", jd.id, exc)
        return {}
    return {"job_profiles": {jd.id: profile}}
```

`PipelineState.jd_skills: Annotated[dict[str, JDSkills], operator.or_]`
and the equivalent `jd_requirements` field both become one
`job_profiles: Annotated[dict[str, JobProfile], operator.or_]`.
`fanout_shortlist_and_requirements` collapses similarly: shortlisting
and (what used to be) requirements-dependent steps both now key off
the same `job_profiles[jd.id]`, since one extraction now produces
everything either step needs — so the second `Send` (previously
`extract_requirements_for_jd`) is simply removed, not replaced.

### `cli.py`

```python
job_profile_chain = build_job_profile_chain(llm)
```
replaces the two existing `jd_skills_chain` / `jd_requirements_chain`
construction lines, and the pydantic-model name list used for schema
export drops `"JDSkills"`, `"JDRequirements"` in favor of
`"JobProfile"`.

## Caching

Both old cache files (`jd_skills.json`, `jd_requirements.json`) are
superseded by one `job_profile.json`. Nothing migrates old cache
entries forward — the prompt text changed, so `_cache_key`'s hash
changes too, and every JD naturally misses and regenerates under the
new cache file on first run after this change, exactly as any other
prompt change already does today. The two old cache files are simply
left behind on disk (not read, not deleted by this change) — stale but
harmless.

## Error Handling

- One `JobProfileGenerationError` (structured-output validation
  failure, or the underlying LLM call failing) causes that JD to be
  excluded from the run (`logger.warning`, pipeline fanout returns
  `{}`), same as today's behavior when either of the two calls it
  replaces fails — except now a single failure is the only way to lose
  a JD, instead of two independent failure points.
- No retry: a transient failure is not retried in-process. (This
  matches the prior, already-applied removal of the retry loop in
  `jd_requirements.py`; this design extends the same choice to the
  skills half of the merged call.)
- `shortlist_candidates`/`match_candidate_skills` treat an empty
  `must_have_skills + nice_to_have_skills` the same way the old code
  treated an empty `technical_skills`: log a warning, return an empty
  shortlist/match set, not an error.

## Testing

- New `tests/scoring/test_job_profile.py` (replaces
  `test_jd_requirements.py` + `test_jd_skills.py`): prompt formatting,
  `generate_job_profile` happy path, `JobProfileGenerationError` on
  LLM/validation failure (single attempt, no retry), cache
  hit/miss/invalid-entry behavior for `load_or_generate_job_profile`.
- `tests/test_models.py`: `JobProfile` field defaults; old
  `JDRequirement`/`JDRequirements`/`JDSkills` tests removed.
- `tests/scoring/test_skills.py`: update fixtures from
  `JDSkills(technical_skills=..., must_have_skills=...)` to
  `JobProfile(must_have_skills=..., nice_to_have_skills=...)`; same
  four shortlisting cases as before (both-thresholds-met,
  must-have-only, total-only, empty-must-have) now phrased against the
  new field names.
- `tests/scoring/test_llm_assessment.py`: update fixtures to pass
  `JobProfile` instead of `JDRequirements`; assert the placeholder
  `context` strings for must-have/nice-to-have gate input.
- `tests/graphs/test_pipeline.py`: update to the single
  `extract_job_profile_for_jd` node and `job_profiles` state key;
  remove assertions tied to the now-deleted second fanout send.
