# Jev Typesafe Assessment Migration — Design

Date: 2026-09-18

## Motivation

Replace the assessment-generation flow in `candidate_ranking` (currently a
free-text LLM chain via `ChatOllama` + LangChain structured output) with
**Jev**, TypeSafe AI's "System One Model." Jev answers predefined
Noul (true/false), Choice (multi-option), and Score (scaled) questions
against a `state` (input text) and returns calibrated confidence per
answer, instead of generating open-ended text. It is accessed via
Cloudflare Workers AI (`typesafe/jev`).

Because Jev cannot produce open-ended prose, the current
`strengths` / `weaknesses` / `additional_skills` / `reasoning` fields
cannot be preserved as free text. The assessment schema is redesigned
around structured Score/Choice/Noul answers, and downstream ranking
(which currently reads that free text to drive an LLM tournament) is
simplified to sort directly by the resulting score.

## Scope

**In scope (migrated to Jev):**
- Assessment generation: `src/candidate_ranking/scoring/assessment.py`
- `Assessment` model: `src/candidate_ranking/models.py`
- Ranking: replace tournament/Plackett-Luce listwise comparison with a
  direct sort by Jev score
- Pipeline graph: `src/candidate_ranking/graphs/pipeline.py`
- Output: `src/candidate_ranking/output/formatter.py`,
  `src/candidate_ranking/output/console_export.py`
- Config/credentials: `src/candidate_ranking/config.py`

**Explicitly out of scope (left as-is, will go stale/break, adapted separately later):**
- `jd_skills_chain`, `skill_extraction_chain` — remain on Ollama
- `src/candidate_ranking/injection/` (attacks.py, defenses.py, measurement.py)
  — the active rank-shift/mitigation research paper is built on the old
  free-text assessment + tournament flow; not touched by this migration
- `src/candidate_ranking/evaluation/ragas_eval.py` — faithfulness
  evaluation is conceptually tied to free-text claims and no longer
  applies to structured scores; left in place, not removed
- `src/candidate_ranking/ranking/tournament.py`, `mc_kg.py`,
  `plackett_luce.py` — files remain in the repo (not deleted) since the
  injection research depends on them, but the pipeline stops calling them

## Jev Client

New module: `src/candidate_ranking/scoring/jev_client.py`

```python
class JevQuestion(BaseModel):
    key: str
    kind: Literal["noul", "choice", "score"]
    prompt: str
    choices: list[str] | None = None   # required for "choice"

class JevAnswer(BaseModel):
    key: str
    value: bool | str | float
    confidence: float

class JevClientError(Exception): ...

class JevClient:
    def __init__(self, account_id: str, api_token: str, timeout: float = 30.0): ...
    def evaluate(self, state: str, questions: list[JevQuestion]) -> list[JevAnswer]:
        """POST to Cloudflare Workers AI `typesafe/jev`. Raises
        JevClientError on HTTP failure or unparseable response."""
```

- `state` is built from job title, job description, the candidate's
  identified skills, and the CV text — equivalent content to the
  current prompt, as plain text.
- One `evaluate()` call per candidate carries all questions at once:
  the fixed questions plus one Score question per JD requirement.
- No LangChain involvement in this path — Jev's `state`+`questions`
  contract doesn't fit the chat-completion-shaped `BaseChatModel`
  interface, so a dedicated thin HTTP client is used instead of forcing
  it into a `Runnable`.
- `JevClientError` is caught at the call site and re-raised as
  `AssessmentGenerationError`, matching the existing `GenerationError`
  pattern in `generation.py`.

## Assessment Schema

`models.py` — `Assessment` replaced:

```python
class Assessment(BaseModel):
    job_description_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    generated_by_model: str = Field(min_length=1)   # "typesafe/jev"
    overall_fit_score: float = Field(ge=0, le=100)          # Score
    overall_recommendation: Literal["hire", "maybe", "no"]  # Choice
    meets_min_qualifications: bool                          # Noul
    requirement_scores: dict[str, float]                    # Score, one per jd_skills.technical_skills
    confidence: dict[str, float]                            # per-key confidence from Jev, for audit/retry
```

`reasoning`, `strengths`, `weaknesses`, `additional_skills` are removed
entirely — no derived/synthetic text is generated in their place.

## `generate_assessment()` Rewrite

`src/candidate_ranking/scoring/assessment.py`:

1. Build `state` from JD title/description, candidate's identified
   skills, and CV text (same content as the current prompt, as plain
   text).
2. Build `questions`: three fixed questions
   (`overall_fit_score`, `overall_recommendation`,
   `meets_min_qualifications`) plus one Score question per entry in
   `jd_skills.technical_skills` (skip `requirement_scores` entirely if
   `jd_skills` is `None`).
3. Call `JevClient.evaluate(state, questions)` and map the returned
   `list[JevAnswer]` onto `Assessment`.
4. Retry once (same two-attempt shape as today): if `confidence` on
   `overall_fit_score`, `overall_recommendation`, or
   `meets_min_qualifications` falls below a fixed threshold (0.5),
   retry with a note appended to `state` that the previous answer was
   low-confidence.
5. Removed entirely, with no replacement (they only made sense against
   free-text weaknesses): `_find_contradictions`,
   `_drop_contradicting_weaknesses`, `_build_retry_feedback`, and the
   retry-audit report (`_classify_retry_attempts`, `_build_retry_audit`,
   `write_retry_audit_report`).
6. `_assessment_cache_path` / `_read_cached_assessment` /
   `_write_cached_assessment` keep their current shape; only
   `_assessment_cache_key` changes — it hashes a representation of the
   new `state` + `questions` instead of
   `ASSESSMENT_GENERATION_PROMPT.format_messages(...)`.

## Ranking & Output

**Pipeline graph** (`graphs/pipeline.py`): after `assessments_barrier`,
remove the tournament stage (`run_tournament_for_jd_repeat_node`,
`tournament_barrier`, `build_ensemble_tournament_result`,
`write_jd_repeats`) entirely. Add a new `rank_and_format_jd_node` per
JD: take every assessment for that JD's shortlisted candidates, sort
descending by `overall_fit_score`, and write the result. No
Monte-Carlo/stability-repeat averaging is needed since Jev's score is a
direct per-candidate output, not the product of pairwise LLM
comparisons.

**`output/formatter.py`**: `format_jd_ranking` no longer takes a
`TournamentResult` — utility/variance/borderline/times_ranked are
tournament-specific concepts that no longer apply. New line shape:

```
1. **cand_id** (score=87.5, hire) — meets_min_qualifications=True
```

JSON payload per row: `rank`, `candidate_id`, `overall_fit_score`,
`overall_recommendation`, `meets_min_qualifications`,
`requirement_scores`.

**`output/console_export.py`**: `strengths` / `weaknesses` /
`additional_skills` fields replaced with `overall_fit_score` /
`overall_recommendation` / `requirement_scores`.

**`config.py`**: tournament-only fields removed from `RunConfig` and
`_ENV_OVERRIDES`: `tournament_iterations`, `tournament_subset_size`,
`num_subset_samples`, `num_mc_draws`, `pl_prior_variance`,
`stability_repeats`, `target_appearances_per_candidate`,
`tournament_iterations_min`. Added: `cf_account_id: str`,
`cf_api_token: str`, with env overrides
`CANDIDATE_RANKING_CF_ACCOUNT_ID` and `CANDIDATE_RANKING_CF_API_TOKEN`.
Actual credential values live in a local `.env` (not committed) or a
secrets manager — never hardcoded, never logged.

## Credentials Handling

A Cloudflare API token was pasted directly into this session's chat
during design discussion. It was not stored in any file, memory, or
this spec, and the user was advised to rotate/revoke it immediately.
Implementation must read credentials only from environment variables
at runtime — never embed a literal token in code, tests, fixtures, or
documentation.

## Testing Plan

- Unit tests for `JevClient.evaluate()` against a mocked HTTP transport
  (success, HTTP error, malformed response → `JevClientError`).
- Unit tests for `generate_assessment()`: question construction from
  `JDSkills`, low-confidence retry path, cache-key stability.
- Unit tests for the new `format_jd_ranking` (no `TournamentResult`
  input) and `console_export` field mapping.
- Pipeline graph test confirming the tournament nodes are gone and
  `rank_and_format_jd_node` produces a descending-by-score order.
