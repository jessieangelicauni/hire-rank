# Seniority/Education Score-Criteria Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `meets_seniority_requirement`/`meets_education_requirement` from binary Noul questions to 5-level, criteria-grounded Score questions (reusing the pattern already validated for technical requirements), surface the resulting scores through console-web, adapt the evaluation-study scripts' ablation to the new question type, update the paper's methodology prose, then re-run the study and fill in real numbers.

**Architecture:** `assessment.py`'s `_build_questions` changes the seniority/education `JevQuestion.kind` from `"noul"` to `"score"` with new 5-item criteria lists (mirroring `_REQUIREMENT_FIT_CRITERIA`); `Assessment.meets_seniority_requirement: bool | None` / `meets_education_requirement: bool | None` become `seniority_fit_score: float | None` / `education_fit_score: float | None`, aggregated across calls by arithmetic mean (a new `_aggregate_mean_optional` helper) instead of majority vote. Certification and `meets_min_qualifications` are untouched. `console_export.py` and `console-web` surface the two new scores the same way `requirement_scores` is already surfaced. The evaluation-study ablation scripts drop the Noul-specific "circular vs. concrete" comparison and instead run the same "concrete vs. bare-label Score" ablation already used for technical requirements, over *all* eligible pairs (not a subsample), fixing the statistical-power gap observed for the old Seniority ablation row (`n=66`, `p=0.134`).

**Tech Stack:** Python 3 (pydantic, pytest), TypeScript/React (Vite, no existing test framework — verify via `tsc -b`), LaTeX (IEEEtran).

## Global Constraints

- Certification stays per-cert Noul; `meets_min_qualifications` stays a single holistic Noul question — neither is touched by this plan.
- No backward-compatibility shims: `meets_seniority_requirement`/`meets_education_requirement` are removed outright, not deprecated-and-kept.
- Seniority/education ablation in the evaluation-study scripts must run on all eligible pairs (every pair whose job profile states that requirement), not a subsample — this is the specific fix for the old ablation's `n=66` underpowered Seniority row.
- The design doc is `docs/superpowers/specs/2026-09-19-seniority-education-score-criteria-design.md` — re-read it if a task's rationale is unclear.

---

### Task 1: Score-type seniority/education questions, model fields, aggregation

**Files:**
- Modify: `src/candidate_ranking/models.py:50-51`
- Modify: `src/candidate_ranking/scoring/assessment.py:42-51` (criteria constants), `:131-172` (`_build_questions`), `:191-203` (`_answers_to_assessment`), `:284-320` (aggregation + `generate_assessment`)
- Test: `tests/scoring/test_assessment.py:205-297`

**Interfaces:**
- Produces: `Assessment.seniority_fit_score: float | None`, `Assessment.education_fit_score: float | None` (0-100, `None` when the job profile states no such requirement) — read by Task 2 (`console_export.py`) and Task 4 (`run_jev_evaluation_study.py`'s `_answers_to_assessment` reuse).
- Produces: `_SENIORITY_FIT_CRITERIA: list[str]`, `_EDUCATION_FIT_CRITERIA: list[str]` (5 items each) in `assessment.py` — not consumed elsewhere in this plan, but must exist for the question builder.

- [ ] **Step 1: Update the existing tests to expect Score-type seniority/education**

Edit `tests/scoring/test_assessment.py`. Replace the test at lines 205-227:

```python
def test_generate_assessment_builds_certification_seniority_education_questions():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
        JevAnswer(key="requirement::Python", kind="score", value=4.0, confidence=0.9),
        JevAnswer(key="certification::AWS Certified Solutions Architect", kind="noul", value=True, confidence=0.9),
        JevAnswer(key="seniority", kind="score", value=3.0, confidence=0.85),
        JevAnswer(key="education", kind="score", value=1.0, confidence=0.8),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills_full(), n_calls=1)

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "certification::AWS Certified Solutions Architect" in question_keys
    assert "seniority" in question_keys
    assert "education" in question_keys
    assert assessment.certification_results == {"AWS Certified Solutions Architect": True}
    assert assessment.seniority_fit_score == 75.0
    assert assessment.education_fit_score == 25.0
```

Replace the test at lines 229-247 (only the last two assertions change):

```python
def test_generate_assessment_omits_seniority_education_questions_when_not_stated():
    jev_client = Mock()
    jev_client.evaluate.return_value = [
        JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
        JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.85),
        JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.95),
    ]

    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, _jd_skills(), n_calls=1)

    _, questions = jev_client.evaluate.call_args.args
    question_keys = {q.key for q in questions}
    assert "seniority" not in question_keys
    assert "education" not in question_keys
    assert not any(k.startswith("certification::") for k in question_keys)
    assert assessment.certification_results == {}
    assert assessment.seniority_fit_score is None
    assert assessment.education_fit_score is None
```

Replace the test at lines 249-269 entirely:

```python
def test_seniority_and_education_criteria_are_five_level_evidence_based_scores():
    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=[],
        seniority_requirement="5+ years of backend experience",
        education_requirement="Bachelor's degree in Computer Science",
    )
    questions = _build_questions(jd_skills)
    by_key = {q.key: q for q in questions}

    assert by_key["seniority"].kind == "score"
    assert by_key["education"].kind == "score"
    assert len(by_key["seniority"].criteria) == 5
    assert len(by_key["education"].criteria) == 5

    seniority_criteria = " ".join(by_key["seniority"].criteria).lower()
    education_criteria = " ".join(by_key["education"].criteria).lower()

    # The old Noul criteria just restated the question ("supports"/"does not support" this
    # requirement) -- circular, giving Jev no concrete evidence to look for. The Score criteria
    # must instead name the kind of evidence that distinguishes each level.
    for banned_phrase in ("supports that the candidate", "does not support that the candidate"):
        assert banned_phrase not in seniority_criteria
        assert banned_phrase not in education_criteria

    assert any(term in seniority_criteria for term in ("years", "level", "domain", "role"))
    assert any(term in education_criteria for term in ("degree", "field", "credential"))
```

Replace the test at lines 272-297 (only the seniority answers/assertion change from bool votes to score means):

```python
def test_generate_assessment_aggregates_certification_and_seniority_across_calls():
    jev_client = Mock()

    def _answers(cert_value, seniority_value):
        return [
            JevAnswer(key="overall_fit_score", kind="score", value=3.0, confidence=0.9),
            JevAnswer(key="overall_recommendation", kind="choice", value="hire", confidence=0.9),
            JevAnswer(key="meets_min_qualifications", kind="noul", value=True, confidence=0.9),
            JevAnswer(key="certification::AWS Certified Solutions Architect", kind="noul", value=cert_value, confidence=0.9),
            JevAnswer(key="seniority", kind="score", value=seniority_value, confidence=0.9),
        ]

    jev_client.evaluate.side_effect = [
        _answers(True, 4.0),
        _answers(True, 2.0),
        _answers(False, 3.0),
    ]

    jd_skills = JDSkills(
        job_description_id="jd-1", generated_by_model="qwen2.5:14b", technical_skills=[],
        certifications=["AWS Certified Solutions Architect"], seniority_requirement="5+ years",
    )
    assessment = generate_assessment(_jd(), _candidate(), jev_client, JEV_MODEL_NAME, jd_skills)

    assert assessment.certification_results == {"AWS Certified Solutions Architect": True}  # 2 of 3 votes
    assert assessment.seniority_fit_score == pytest.approx((100.0 + 50.0 + 75.0) / 3)  # mean of 4.0/2.0/3.0 -> percent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scoring/test_assessment.py -v`
Expected: FAIL — `test_generate_assessment_builds_certification_seniority_education_questions` and friends fail because `Assessment` has no `seniority_fit_score` attribute yet, and `_build_questions` still emits `kind="noul"` criteria dicts.

- [ ] **Step 3: Replace the model fields**

Edit `src/candidate_ranking/models.py`, replace lines 50-51:

```python
    seniority_fit_score: float | None = Field(default=None, ge=0, le=100)
    education_fit_score: float | None = Field(default=None, ge=0, le=100)
```

- [ ] **Step 4: Add the criteria constants**

Edit `src/candidate_ranking/scoring/assessment.py`. Insert immediately after `_REQUIREMENT_FIT_CRITERIA` (after line 48, before the existing `assert len(_OVERALL_FIT_CRITERIA) == len(_REQUIREMENT_FIT_CRITERIA)` on line 49):

```python
_SENIORITY_FIT_CRITERIA = [
    "No relevant experience or years mentioned toward this requirement",
    "Some relevant experience but clearly short of the stated years/level, or relevant only "
    "in an adjacent domain (e.g., internship-level, or senior experience in an unrelated field)",
    "Close to but slightly under the stated years/level requirement, in a clearly related role",
    "Meets the stated years/level requirement in a directly matching role",
    "Exceeds the stated years/level requirement in a directly matching role, with demonstrated "
    "seniority (e.g., ownership scope, leadership)",
]
_EDUCATION_FIT_CRITERIA = [
    "No relevant education mentioned",
    "Education mentioned but neither the field nor the degree level matches the requirement",
    "Matches the requirement on field or degree level, but not both",
    "Matches the requirement's field and degree level",
    "Exceeds the requirement (higher degree level) in the same or a closely related field",
]
```

And extend the existing assert line (49) to:

```python
assert len(_OVERALL_FIT_CRITERIA) == len(_REQUIREMENT_FIT_CRITERIA) == len(_SENIORITY_FIT_CRITERIA) == len(_EDUCATION_FIT_CRITERIA)
```

- [ ] **Step 5: Change `_build_questions` to emit Score questions for seniority/education**

Edit `src/candidate_ranking/scoring/assessment.py`, replace lines 131-172 (the `if jd_skills and jd_skills.seniority_requirement:` block through the end of the `if jd_skills and jd_skills.education_requirement:` block):

```python
    if jd_skills and jd_skills.seniority_requirement:
        questions.append(
            JevQuestion(
                key=_SENIORITY_KEY,
                kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated seniority/experience "
                    f"requirement: '{jd_skills.seniority_requirement}'?"
                ),
                criteria=_SENIORITY_FIT_CRITERIA,
            )
        )
    if jd_skills and jd_skills.education_requirement:
        questions.append(
            JevQuestion(
                key=_EDUCATION_KEY,
                kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated education "
                    f"requirement: '{jd_skills.education_requirement}'?"
                ),
                criteria=_EDUCATION_FIT_CRITERIA,
            )
        )
```

- [ ] **Step 6: Update `_answers_to_assessment` to map Score answers instead of Noul**

Edit `src/candidate_ranking/scoring/assessment.py`, replace lines 201-202:

```python
        seniority_fit_score=_score_to_percent(by_key[_SENIORITY_KEY].value) if _SENIORITY_KEY in by_key else None,
        education_fit_score=_score_to_percent(by_key[_EDUCATION_KEY].value) if _EDUCATION_KEY in by_key else None,
```

- [ ] **Step 7: Replace `_aggregate_optional_bool` with a mean-based aggregator**

Edit `src/candidate_ranking/scoring/assessment.py`, replace the `_aggregate_optional_bool` function (lines 284-288):

```python
def _aggregate_mean_optional(calls: list[Assessment], field: str) -> float | None:
    values = [v for a in calls if (v := getattr(a, field)) is not None]
    if not values:
        return None
    return statistics.mean(values)
```

- [ ] **Step 8: Use the new aggregator in `generate_assessment`**

Edit `src/candidate_ranking/scoring/assessment.py`, replace lines 318-319:

```python
        seniority_fit_score=_aggregate_mean_optional(calls, "seniority_fit_score"),
        education_fit_score=_aggregate_mean_optional(calls, "education_fit_score"),
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/scoring/test_assessment.py -v`
Expected: PASS — all tests in `tests/scoring/test_assessment.py` green.

- [ ] **Step 10: Commit**

```bash
git add src/candidate_ranking/models.py src/candidate_ranking/scoring/assessment.py tests/scoring/test_assessment.py
git commit -m "feat: score seniority/education as 5-level criteria-grounded questions

Replaces the binary meets_seniority_requirement/meets_education_requirement
Noul questions with Score questions (5 ordinal levels, concrete evidence
per level), reusing the pattern already validated for technical
requirements. A single bit couldn't express '4 years when 5 required' vs
'0 years', or 'senior but wrong domain' vs 'senior, matching domain'."
```

---

### Task 2: Export seniority/education scores to console-web data

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py:250-255`
- Test: `tests/output/test_console_export.py:47-53,70-90`

**Interfaces:**
- Consumes: `Assessment.seniority_fit_score`, `Assessment.education_fit_score` from Task 1 (read via the JSON dict `assessment_entry` loaded from `assessments.json`, since `console_export.py` works off cached JSON, not live `Assessment` objects).
- Produces: `assessments[row_id]["seniority_fit_score"]`, `assessments[row_id]["education_fit_score"]` in the exported `real-data.json` — consumed by Task 3 (`console-web/src/data.ts`).

- [ ] **Step 1: Update the fixture and add failing assertions**

Edit `tests/output/test_console_export.py`. Replace the `assessments` fixture dict at lines 47-53:

```python
    assessments = {
        "cand-a": {
            "job_description_id": "jd-1", "candidate_id": "cand-a", "generated_by_model": "typesafe/jev",
            "overall_fit_score": 88.0, "overall_recommendation": "hire", "meets_min_qualifications": True,
            "requirement_scores": {"Python": 100.0}, "confidence": {"overall_fit_score": 0.9},
            "seniority_fit_score": 75.0, "education_fit_score": None,
        }
    }
```

Add two assertions after line 82 (`assert assessment["requirement_scores"] == {"Python": 100.0}`) in `test_export_console_web_data_maps_jev_scores`:

```python
    assert assessment["seniority_fit_score"] == 75.0
    assert assessment["education_fit_score"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/output/test_console_export.py::test_export_console_web_data_maps_jev_scores -v`
Expected: FAIL with `KeyError: 'seniority_fit_score'`.

- [ ] **Step 3: Export the new fields**

Edit `src/candidate_ranking/output/console_export.py`, replace lines 250-255:

```python
            assessments[row_id] = {
                "overall_fit_score": assessment_entry["overall_fit_score"],
                "overall_recommendation": assessment_entry["overall_recommendation"],
                "meets_min_qualifications": assessment_entry["meets_min_qualifications"],
                "requirement_scores": assessment_entry.get("requirement_scores", {}),
                "seniority_fit_score": assessment_entry.get("seniority_fit_score"),
                "education_fit_score": assessment_entry.get("education_fit_score"),
            }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/output/test_console_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_ranking/output/console_export.py tests/output/test_console_export.py
git commit -m "feat: export seniority/education fit scores to console-web data"
```

---

### Task 3: Display seniority/education fit in the console-web applicant detail panel

**Files:**
- Modify: `console-web/src/data.ts:20-25,75-80`
- Modify: `console-web/src/views/Leaderboard.tsx:252-253` (insertion point), new `QualificationScores` component

**Interfaces:**
- Consumes: `Assessment.seniority_fit_score`, `Assessment.education_fit_score` (Task 2's export shape).

- [ ] **Step 1: Add the fields to the `Assessment` type and its empty default**

Edit `console-web/src/data.ts`, replace the `Assessment` interface at lines 20-25:

```ts
export interface Assessment {
  overall_fit_score: number;
  overall_recommendation: 'hire' | 'maybe' | 'no';
  meets_min_qualifications: boolean;
  requirement_scores: Record<string, number>;
  seniority_fit_score: number | null;
  education_fit_score: number | null;
}
```

Replace `EMPTY_ASSESSMENT` at lines 75-80:

```ts
const EMPTY_ASSESSMENT: Assessment = {
  overall_fit_score: 0,
  overall_recommendation: 'no',
  meets_min_qualifications: false,
  requirement_scores: {},
  seniority_fit_score: null,
  education_fit_score: null,
};
```

- [ ] **Step 2: Add a `QualificationScores` component to `Leaderboard.tsx`**

Edit `console-web/src/views/Leaderboard.tsx`. Insert this new function immediately after the `ScoreSummary` function (i.e., right before `function RequirementScores(...)` at line 296):

```tsx
function QualificationScores({ assessment }: { assessment: Assessment }) {
  const entries: [string, number][] = [];
  if (assessment.seniority_fit_score !== null) entries.push(['Seniority', assessment.seniority_fit_score]);
  if (assessment.education_fit_score !== null) entries.push(['Education', assessment.education_fit_score]);
  if (entries.length === 0) return null;

  return (
    <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden', marginBottom: 16 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, padding: '16px 20px 12px' }}>
        Qualification fit
      </div>
      <div>
        {entries.map(([label, score]) => (
          <div
            key={label}
            style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', fontSize: 14, color: colorText,
              borderTop: `1px solid ${colorBorder}`,
            }}
          >
            <span style={{ flex: 1, minWidth: 0 }}>{label}</span>
            <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorSurfaceMuted, overflow: 'hidden' }}>
              <div style={{
                width: `${Math.max(0, Math.min(100, score))}%`, height: '100%', borderRadius: radiusPill,
                background: score >= 50 ? colorAccent : colorDanger,
              }} />
            </div>
            <span style={{ flexShrink: 0, width: 32, textAlign: 'right', color: colorTextMuted, fontSize: 13 }}>{score.toFixed(0)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Render it in `DetailPanel`**

Edit `console-web/src/views/Leaderboard.tsx`, replace lines 252-253:

```tsx
      <ScoreSummary assessment={assessment} />
      <QualificationScores assessment={assessment} />
      <RequirementScores requirementScores={assessment.requirement_scores} />
```

- [ ] **Step 4: Type-check and build**

Run: `cd console-web && npm run build`
Expected: builds cleanly (no TypeScript errors). There is no existing test framework for `console-web` (no `.test.*` files in the repo); `tsc -b` inside `npm run build` is the only automated check available, so this step is the verification gate for this task.

- [ ] **Step 5: Manually verify in the dev server**

Run: `cd console-web && npm run dev`, open the app, navigate to a role's leaderboard, click an applicant whose job profile states a seniority or education requirement, confirm the new "Qualification fit" card renders with a bar for each present field and is absent when both are `null`.

- [ ] **Step 6: Commit**

```bash
git add console-web/src/data.ts console-web/src/views/Leaderboard.tsx
git commit -m "feat: display seniority/education fit score in applicant detail panel"
```

---

### Task 4: Adapt the evaluation-study data collector to Score-type seniority/education

**Files:**
- Modify: `scripts/run_jev_evaluation_study.py` (whole-file edits below)

**Interfaces:**
- Consumes: `_build_state` from `candidate_ranking.scoring.assessment` (unchanged import); the new `seniority_fit_score`/`education_fit_score` fields are not read directly here — this task only changes *ablation question construction*, not the real pipeline path (`_build_questions`, already updated in Task 1, is what `collect_test_retest`/`collect_ablation` call for the non-ablation paths).
- Produces: `run_dir/evaluation/seniority_education_ablation.json` (replaces the old `noul_criteria_ablation.json`), with records shaped `{"jd_id", "candidate_id", "concrete_confidence", "vague_confidence", "vague_latency_seconds"}` — consumed by Task 5.

No existing test file covers this script (it is a live-API data-collection driver, not unit-tested); there is no TDD loop here — implement directly and verify with `--dry-run`.

- [ ] **Step 1: Replace `_circular_noul_questions` with a Score-type vague-criteria builder**

Edit `scripts/run_jev_evaluation_study.py`, replace the `_circular_noul_questions` function (lines 64-94):

```python
def _vague_seniority_education_questions(jd_skills: JDSkills | None) -> list[JevQuestion]:
    questions = []
    if jd_skills and jd_skills.seniority_requirement:
        questions.append(
            JevQuestion(
                key="seniority", kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated seniority/experience "
                    f"requirement: '{jd_skills.seniority_requirement}'?"
                ),
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    if jd_skills and jd_skills.education_requirement:
        questions.append(
            JevQuestion(
                key="education", kind="score",
                instructions=(
                    "How well does the candidate meet the job description's stated education "
                    f"requirement: '{jd_skills.education_requirement}'?"
                ),
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    return questions
```

Note: the `instructions` text here must match `_build_questions`' seniority/education `instructions` in `assessment.py` exactly (Task 1, Step 5) — the ablation only varies `criteria`, never `instructions`.

- [ ] **Step 2: Replace `collect_noul_criteria_ablation` with `collect_seniority_education_ablation`**

Edit `scripts/run_jev_evaluation_study.py`, replace the `collect_noul_criteria_ablation` function (lines 210-254):

```python
def collect_seniority_education_ablation(
    jev_client: JevClient,
    jds_by_id: dict[str, JobDescription],
    candidates_by_id: dict[str, Candidate],
    jd_skills_by_jd: dict[str, JDSkills],
    run_dir: Path,
    pairs: list[tuple[str, str]],
) -> list[dict]:
    eligible_pairs = [
        pair
        for pair in pairs
        if _vague_seniority_education_questions(jd_skills_by_jd.get(pair[0]))
    ]

    def run_one(pair: tuple[str, str]) -> dict:
        jd_id, candidate_id = pair
        jd = jds_by_id[jd_id]
        candidate = candidates_by_id[candidate_id]
        jd_skills = jd_skills_by_jd.get(jd_id)

        cached = json.loads((run_dir / jd_id / "assessments.json").read_text(encoding="utf-8"))[candidate_id]
        vague_questions = _vague_seniority_education_questions(jd_skills)
        concrete_confidence = {
            q.key: cached["confidence"][q.key] for q in vague_questions if q.key in cached["confidence"]
        }

        state = _build_state(jd, candidate)
        answers, elapsed = _timed_evaluate(jev_client, state, vague_questions)
        vague_confidence = {a.key: a.confidence for a in answers}

        return {
            "jd_id": jd_id,
            "candidate_id": candidate_id,
            "concrete_confidence": concrete_confidence,
            "vague_confidence": vague_confidence,
            "vague_latency_seconds": elapsed,
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(run_one, eligible_pairs), start=1):
            results.append(result)
            if i % 25 == 0 or i == len(eligible_pairs):
                print(f"seniority-education-ablation: {i}/{len(eligible_pairs)} pairs done")
    return results
```

This is unchanged in structure from the function it replaces: `eligible_pairs` still filters over the *full* `pairs` list passed in (all successfully-assessed pairs for the run), not a subsample — so "all eligible pairs" is preserved automatically.

- [ ] **Step 3: Update `main()` to use the renamed function, flag, and output file**

Edit `scripts/run_jev_evaluation_study.py`. Replace line 268:

```python
    eligible_count = sum(1 for jd_id, _ in pairs if _vague_seniority_education_questions(jd_skills_by_jd.get(jd_id)))
```

Replace lines 257 and 270-298 (the `main` signature and the `noul_criteria_only` dry-run/execution branch):

```python
def main(run_id: str, repeats: int, ablation_sample_size: int, seed: int, max_workers: int, dry_run: bool, seniority_education_only: bool) -> None:
```

```python
    if dry_run:
        if seniority_education_only:
            print(
                f"Dry run: would collect seniority/education ablation for {eligible_count} pair(s) whose job "
                f"description has a seniority or education requirement (out of {len(pairs)} total pairs)."
            )
        else:
            print(
                f"Dry run: would collect test-retest for {len(pairs)} pair(s) x {repeats} repeat(s) "
                f"({len(pairs) * repeats} call(s)), plus ablation for {ablation_size} pair(s) "
                f"({ablation_size} call(s))."
            )
        return

    if not cfg.jev_api_key:
        raise RuntimeError("Set CANDIDATE_RANKING_JEV_API_KEY (e.g. in .env) before running.")
    jev_client = JevClient(api_token=cfg.jev_api_key)

    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if seniority_education_only:
        print("Collecting seniority/education criteria-design ablation data (concrete vs. bare-label Score)...")
        sen_edu_results = collect_seniority_education_ablation(
            jev_client, jds_by_id, candidates_by_id, jd_skills_by_jd, run_dir, pairs
        )
        (out_dir / "seniority_education_ablation.json").write_text(json.dumps(sen_edu_results, indent=2), encoding="utf-8")
        print(f"Wrote {len(sen_edu_results)} seniority/education ablation record(s) to {out_dir / 'seniority_education_ablation.json'}")
        return
```

- [ ] **Step 4: Rename the CLI flag**

Edit `scripts/run_jev_evaluation_study.py`, replace lines 325-329 and 332-333:

```python
    parser.add_argument(
        "--seniority-education-only", action="store_true",
        help="Skip test-retest and technical-requirement ablation; only collect the seniority/education "
        "concrete-vs-bare-label Score criteria ablation, over all eligible pairs.",
    )
    args = parser.parse_args()
    main(
        args.run_id, args.repeats, args.ablation_sample_size, args.seed, args.max_workers, args.dry_run,
        args.seniority_education_only,
    )
```

- [ ] **Step 5: Verify with a dry run**

Run: `uv run python scripts/run_jev_evaluation_study.py --run-id <an-existing-run-id> --dry-run`
Run: `uv run python scripts/run_jev_evaluation_study.py --run-id <an-existing-run-id> --dry-run --seniority-education-only`
Expected: both print their pair-count summaries without error and without calling Jev (no `CANDIDATE_RANKING_JEV_API_KEY` required for `--dry-run`). Use a `run_id` already present under `runs/` from a prior real run (e.g. `20260918-104531`, referenced in `README.md`'s evaluation-harness example).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_jev_evaluation_study.py
git commit -m "feat: run seniority/education ablation as Score (concrete vs. bare-label), not circular Noul"
```

---

### Task 5: Adapt the evaluation-study analyzer to the renamed ablation

**Files:**
- Modify: `scripts/analyze_jev_evaluation_study.py:295-323` (analysis function), `:428-442` (markdown rendering), `:485-488` (main wiring)

**Interfaces:**
- Consumes: `run_dir/evaluation/seniority_education_ablation.json` (Task 4's output shape).
- Produces: `report["seniority_education_ablation"]` in `report.json`, and the corresponding markdown section in `report.md` — read manually in Task 7 when filling in the paper.

No existing test file covers this script either; verify by running it end-to-end against a real run's output in Task 7.

- [ ] **Step 1: Rename and adapt `analyze_noul_criteria_ablation`**

Edit `scripts/analyze_jev_evaluation_study.py`, replace the function at lines 295-322:

```python
def analyze_seniority_education_ablation(records: list[dict]) -> dict:
    concrete_by_key: dict[str, list[float]] = {"seniority": [], "education": []}
    vague_by_key: dict[str, list[float]] = {"seniority": [], "education": []}
    vague_latencies: list[float] = []

    for record in records:
        concrete_conf = record["concrete_confidence"]
        vague_conf = record["vague_confidence"]
        for key in ("seniority", "education"):
            if key in concrete_conf and key in vague_conf:
                concrete_by_key[key].append(concrete_conf[key])
                vague_by_key[key].append(vague_conf[key])
        vague_latencies.append(record["vague_latency_seconds"])

    buckets = {
        key: _paired_bucket(concrete_by_key[key], vague_by_key[key], f"{key}_confidence")
        for key in ("seniority", "education")
    }
    concrete_pooled = concrete_by_key["seniority"] + concrete_by_key["education"]
    vague_pooled = vague_by_key["seniority"] + vague_by_key["education"]

    return {
        "n_pairs": len(records),
        "seniority": buckets["seniority"],
        "education": buckets["education"],
        "pooled": _paired_bucket(concrete_pooled, vague_pooled, "seniority_education_pooled_confidence"),
        "vague_latency_seconds_mean": statistics.mean(vague_latencies) if vague_latencies else None,
    }
```

- [ ] **Step 2: Rename the markdown section**

Edit `scripts/analyze_jev_evaluation_study.py`, replace lines 428-442:

```python
    sen_edu_abl = report.get("seniority_education_ablation")
    if sen_edu_abl:
        lines += ["", "## Seniority/education criteria ablation (concrete vs. bare-label Score)", f"- n={sen_edu_abl['n_pairs']} eligible pairs"]
        for label, key in (("seniority", "seniority"), ("education", "education"), ("pooled", "pooled")):
            bucket = sen_edu_abl[key]
            if bucket["wilcoxon"]:
                w = bucket["wilcoxon"]
                lines.append(
                    f"- {label} confidence (n_obs={bucket['n_observations']}): "
                    f"concrete mean={w['mean_first']:.3f} vs bare-label mean={w['mean_second']:.3f} "
                    f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f} "
                    f"(<0.5: concrete {bucket['concrete_pct_below_0.5']:.0f}% vs bare-label {bucket['vague_pct_below_0.5']:.0f}%)"
                )
            else:
                lines.append(f"- {label} confidence: insufficient paired data (n_obs={bucket['n_observations']})")
```

- [ ] **Step 3: Update `main()`'s file wiring**

Edit `scripts/analyze_jev_evaluation_study.py`, replace lines 485-488:

```python
    seniority_education_ablation_path = eval_dir / "seniority_education_ablation.json"
    if seniority_education_ablation_path.exists():
        seniority_education_records = json.loads(seniority_education_ablation_path.read_text(encoding="utf-8"))
        report["seniority_education_ablation"] = analyze_seniority_education_ablation(seniority_education_records)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_jev_evaluation_study.py
git commit -m "feat: analyze seniority/education ablation as Score criteria, not circular Noul"
```

---

### Task 6: Paper methodology prose (no numbers yet)

**Files:**
- Modify: `docs/paper2_jev.tex:44` (bullet 1), `:64` (§III-A pipeline description), `:92` (§III-B seniority/education paragraph), `:221` (Conclusion)

This task only touches prose that does not depend on rerun results. Table `tab_score_ablation` and every numeric claim (abstract, §V-B, Conclusion's effect-size comparison) are handled in Task 7, after real numbers exist.

- [ ] **Step 1: Update bullet 1 (Criteria-Grounded Question Design) to drop the Noul framing**

Edit `docs/paper2_jev.tex`, replace line 44:

```latex
\item \textbf{Criteria-Grounded Question Design}: a methodology for writing Score-type questions with concrete, situational evidence descriptions instead of bare ordinal labels. A paired ablation shows this raises answer confidence for every Score question type -- overall fit, per-requirement, seniority, and education -- and a separate internal-coherence check shows it improves group separation for Score-derived judgments, on held-out (job, applicant) pairs.
```

- [ ] **Step 2: Update §III-A's question-type inventory**

Edit `docs/paper2_jev.tex`, in the paragraph starting "The pipeline changes at the next stage..." (line 64), replace:

```
an overall-fit Score, a hiring-recommendation Choice, a minimum-qualifications Noul, one Score per technical requirement, and one Noul per stated certification, seniority, or education requirement. Every Score question, overall-fit or per-requirement, offers a fixed set of ordinal levels (Section III-B);
```

with:

```
an overall-fit Score, a hiring-recommendation Choice, a minimum-qualifications Noul, one Score per technical requirement, one Score per stated seniority or education requirement, and one Noul per stated certification. Every Score question -- overall-fit, per-requirement, or per-qualification -- offers a fixed set of ordinal levels (Section III-B);
```

- [ ] **Step 3: Rewrite the §III-B seniority/education paragraph**

Edit `docs/paper2_jev.tex`, replace line 92 in full:

```latex
The same fix -- naming evidence instead of leaving the model to guess -- generalizes to the seniority and education Score questions, which had the same problem in a different shape: their five levels were initially given the same bare ordinal labels (``0,'' ``25,'' ``50,'' ``75,'' ``100'') as the unfixed Score questions above, rather than the concrete evidence descriptions used for technical requirements. Rewriting the levels to name concrete evidence -- explicit years, dates, job titles, or domain match for seniority; a matching field, a matching degree level, or both, for education -- follows the same logic as the technical-requirement fix, tested with a second paired ablation on all (job, applicant) pairs whose job profile states either requirement. One alternative was rejected: generating levels per job profile with an LLM, since that reintroduces the free-text generation this pipeline exists to remove, and would make scores for the same requirement incomparable across differently worded job profiles.
```

- [ ] **Step 4: Update the Conclusion's cross-question-type claim**

Edit `docs/paper2_jev.tex`, in the Conclusion paragraph (line 221), replace:

```
Criteria-grounded design, validated first for Score questions, also holds for Noul questions with a smaller benefit, and score decomposition's diagnostic value is confirmed quantitatively,
```

with:

```
Criteria-grounded design, validated first for the overall-fit and per-requirement Score questions, also holds for the seniority and education Score questions, and score decomposition's diagnostic value is confirmed quantitatively,
```

- [ ] **Step 5: Commit**

```bash
git add docs/paper2_jev.tex
git commit -m "docs: update paper2 methodology prose for Score-type seniority/education"
```

---

### Task 7: Re-run the evaluation study and fill in real numbers

**This task requires a live `CANDIDATE_RANKING_JEV_API_KEY` and makes real, billed Jev API calls — confirm with the user before running it, and expect it to take real wall-clock time (hundreds of calls).** Do not run this unattended as part of an automated task sweep.

**Files:**
- Modify: `docs/paper2_jev.tex` (abstract, `tab_score_ablation`, §V-B prose, Conclusion effect-size line)
- Generates: a new run's `runs/<run_id>/evaluation/{test_retest,ablation,seniority_education_ablation,report}.{json,md}`, and a refreshed `console-web/src/data/real-data.json` via `export_console_web_data`

- [ ] **Step 1: Run the full pipeline for a fresh run** (produces `assessments.json`/`ranking.json` per JD, needed as the "concrete" baseline for ablation)

Ensure Ollama is running and `CANDIDATE_RANKING_JEV_API_KEY` is set (`.env`), then:

Run: `uv run python -m candidate_ranking.cli run`
Expected: prints a new `Run ID` (UTC timestamp, e.g. `20260919-140000`) and writes `runs/<new-run-id>/`. This also auto-exports a first-pass `console-web/src/data/real-data.json` with `evaluationSummary: null` (no `report.json` exists yet) — that gets refreshed with real evaluation numbers in Step 5. Confirm the run's own output reports 346 (or close to it) successfully-assessed pairs across all 10 job profiles before continuing.

- [ ] **Step 2: Collect test-retest and technical-requirement ablation data**

Run: `uv run python scripts/run_jev_evaluation_study.py --run-id <new-run-id>`
Expected: writes `test_retest.json` and `ablation.json` under `runs/<new-run-id>/evaluation/`.

- [ ] **Step 3: Collect the seniority/education ablation over all eligible pairs**

Run: `uv run python scripts/run_jev_evaluation_study.py --run-id <new-run-id> --seniority-education-only`
Expected: writes `seniority_education_ablation.json` under `runs/<new-run-id>/evaluation/`, covering every pair whose job profile states a seniority or education requirement (not a subsample).

- [ ] **Step 4: Generate the report**

Run: `uv run python scripts/analyze_jev_evaluation_study.py --run-id <new-run-id>`
Run: `uv run python scripts/validate_multi_call_averaging.py --run-id <new-run-id>` (existing third script in the evaluation harness, per `README.md`'s documented sequence — unaffected by this plan's changes, run for completeness since Task 7 regenerates the full evaluation-study output set)
Expected: writes and prints `runs/<new-run-id>/evaluation/report.json` and `report.md`, including a "## Seniority/education criteria ablation (concrete vs. bare-label Score)" section with `seniority`, `education`, and `pooled` Wilcoxon results.

- [ ] **Step 5: Re-export console-web data (now including the evaluation report)**

`export_console_web_data` has no standalone CLI wrapper (it's only auto-invoked inside `cli.py run`, before `report.json` existed). Re-run it directly now that Steps 2-4 have produced `report.json`, so `evaluationSummary` is populated:

Run:
```bash
uv run python -c "
from pathlib import Path
from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data
cfg = apply_env_overrides(RunConfig.full(Path('.')))
print(export_console_web_data(cfg, '<new-run-id>'))
"
```
Expected: prints the path to `console-web/src/data/real-data.json` and overwrites it. Confirm via `cd console-web && npm run dev` that the "Qualification fit" card (Task 3) now renders real scores for applicants against job profiles with seniority/education requirements.

- [ ] **Step 6: Fill in the paper's numbers**

Using `report.json`'s `seniority_education_ablation` bucket (`seniority`, `education`, `pooled`, each with `wilcoxon.{n_pairs,mean_first,mean_second,p_value,rank_biserial_r}`) and `ablation`/`test_retest`/`ranking_convergence` for everything else, update `docs/paper2_jev.tex`:

- Abstract (currently: `pooled seniority/education confidence from 0.828 to 0.839 ($p=0.0063$)`): replace `0.828`/`0.839`/`0.0063` with the new run's `seniority_education_ablation.pooled.wilcoxon.{mean_second,mean_first,p_value}`. Also replace every other abstract number (`0.695`→`0.891`, `346`, `869`, `0.747`, `0.794`, `0.0061`, `0.900`→`0.941`, `0.957`, `0.680`) with this run's equivalents from `ablation`, `score_decomposition_diagnostic`, and `ranking_convergence`.
- Table `tab_score_ablation` (§V-B, `\label{tab_score_ablation}`): replace the current "Noul criteria (concrete vs.\ circular)" block (the `Seniority`/`Education`/`Pooled` rows) with rows in the same shape as the table's existing "Score criteria" rows — columns `Question group`, `$n$`, `Mean conf.\ (concrete / other)`, `$p$ ($r$)` — populated from `seniority_education_ablation.{seniority,education,pooled}.wilcoxon`. Update the table's `\caption` to say "concrete vs.\ bare-label" instead of "concrete vs.\ vague/circular" if the whole table is now homogeneous in comparison type.
- §V-B prose (the paragraph after the table, currently: `The same principle holds for Noul questions, reaching significance pooled and for education alone but not seniority alone (p=0.134, n=66); the direction matches the Score-question result, but the effect is an order of magnitude smaller (r=0.18 vs. r=0.75)...`): rewrite to describe the new Score-vs-Score comparison using the new `p`/`r`/`n` values, dropping "Noul questions" language entirely.
- Conclusion (line 221, already updated qualitatively in Task 6): if the new seniority/education effect size is still meaningfully smaller than the per-requirement effect size, add back a specific quantitative comparison (e.g., "...with a smaller effect size (r=X vs. r=Y)"); if it is now comparable, say so instead — this sentence must reflect what the new numbers actually show, not the old Noul-era pattern.
- Table `tab_comparison` and any other place citing `346`/`869`/specific p-values: check `grep -n "346\|869\|0\.828\|0\.839\|0\.0063\|0\.134" docs/paper2_jev.tex` after the above edits to confirm no stale number remains.

- [ ] **Step 7: Rebuild the PDF and verify it compiles**

No documented build command exists in this repo (`README.md` has none, `git status` shows `docs/paper2_jev.pdf` was previously committed as a build artifact rather than generated by a checked-in script). Build it with `latexmk`, which handles the citation/reference re-run passes IEEEtran needs automatically:

Run: `cd docs && latexmk -pdf paper2_jev.tex && cd ..`
Expected: no LaTeX errors, `docs/paper2_jev.pdf` regenerated, table renders, all citations resolve. If `latexmk` isn't installed, fall back to `cd docs && pdflatex paper2_jev.tex && pdflatex paper2_jev.tex && cd ..` (run twice so cross-references resolve).

- [ ] **Step 8: Commit**

```bash
git add docs/paper2_jev.tex docs/paper2_jev.pdf console-web/src/data/real-data.json
git commit -m "feat: re-run evaluation study with Score-type seniority/education, fill in paper results"
```
