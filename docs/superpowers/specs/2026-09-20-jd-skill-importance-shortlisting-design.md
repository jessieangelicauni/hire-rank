# Must-Have / Nice-to-Have Skill Classification for Shortlisting — Design

Date: 2026-09-20

## Motivation

`shortlist_candidates` (`src/candidate_ranking/scoring/skills.py`)
admits a candidate once their matched-skill count reaches
`min_skill_matches` (default 5), counting every matched skill equally.
The JD extraction prompt (`src/candidate_ranking/scoring/jd_skills.py`)
currently instructs the LLM to "list required and nice-to-have skills
together, without distinguishing between them," so nothing upstream
carries the distinction either.

This lets a candidate be shortlisted on peripheral/utility skills
alone while scoring near-zero on the role's actual core requirements.
`docs/paper2_jev.tex` documents this defect concretely: 8 of 76
shortlisted data-engineer applicants (11%) were near-zero specifically
on Python despite being shortlisted, and 2 of 33 full-stack-engineer
applicants (6%) were near-zero on REST APIs and Node.js while scoring
adequately on peripheral tools (Docker, Kubernetes, AWS). The paper's
Future Work section names the fix directly: "separating a role's core
technical skills from peripheral ones during job profile extraction, so
shortlisting stops admitting applicants on utility skills alone."

This design implements that fix.

## Scope

**In scope:**
- `src/candidate_ranking/models.py` — new `JDSkills.must_have_skills` field
- `src/candidate_ranking/scoring/jd_skills.py` — extraction prompt + schema
- `src/candidate_ranking/scoring/skills.py` — `shortlist_candidates` gains a second, independent match requirement
- `src/candidate_ranking/graphs/pipeline.py` — thread the new threshold through
- `src/candidate_ranking/cli.py` — new `--min-must-have-matches` flag, manifest field, resume support
- Unit tests for all of the above

**Out of scope:**
- Jev assessment/scoring (`scoring/assessment.py`) — per-requirement Score questions, overall-fit scoring, and recommendation are untouched. Must-have/nice-to-have information is used only pre-Jev, at the shortlisting gate, and never reaches Jev.
- Re-running or amending the study already reported in `docs/paper2_jev.tex` — that paper's shortlisting stage was held fixed by design for the controlled comparison it reports; this is a forward-looking pipeline change, not a retroactive correction to a published result.
- `scripts/run_jev_evaluation_study.py` / `scripts/analyze_jev_evaluation_study.py` — no changes needed; they exercise the assessment stage, not shortlisting.

## Data Model

`JDSkills` (`models.py`) gains one field, additive and backward-compatible:

```python
class JDSkills(BaseModel):
    job_description_id: str
    generated_by_model: str
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)  # new: subset of technical_skills
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    education_requirement: str | None = None
```

`technical_skills` itself is unchanged — every existing consumer
(`shortlist_candidates`'s total-match count, `assessment.py`'s
per-requirement question generation, `ragas_eval.py`'s negation
bridging) keeps reading it exactly as before. `must_have_skills` is
purely additive; an empty list (the default) means "no must-have
distinction available," which downstream must degrade to today's
behavior (see Shortlisting Logic below) rather than error.

## Extraction Prompt & Schema (`jd_skills.py`)

Replace the constraint:
> "List required and nice-to-have skills together, without distinguishing between them."

with an instruction to classify: a skill is must-have only if the JD
states or clearly implies it is required/mandatory/essential; every
other extracted skill (mentioned as a plus, preferred, or without any
qualifier of importance) is nice-to-have by default.

`_GeneratedSkills` gains a matching field:

```python
class _GeneratedSkills(BaseModel):
    reasoning: str = Field(min_length=1)
    technical_skills: list[str] = Field(default_factory=list)
    must_have_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    seniority_requirement: str | None = None
    education_requirement: str | None = None
```

`generate_jd_skills` filters `result.must_have_skills` down to the
intersection with `result.technical_skills` (case-insensitive, same
normalization already used for dedup in `skills.py`) before
constructing `JDSkills`, discarding anything the LLM names as
must-have that isn't in its own extracted skill list, with a
`logger.warning` when this happens. This keeps `must_have_skills` a
true subset by construction — no downstream code needs to re-validate
that invariant.

Validation in `generate_jd_skills.validate()` stays a warning-only
guard (log, don't fail generation) — an empty `must_have_skills` after
filtering is a valid outcome (e.g., a JD that lists everything as
"nice to have"), not a generation failure, and must not trigger a
retry loop.

## Shortlisting Logic (`skills.py`)

`shortlist_candidates` gains one new parameter:

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
```

After computing `matched` (unchanged), also compute each candidate's
matched must-have subset:

```python
must_have_normalized = {s.strip().lower() for s in jd_skills.must_have_skills}
required_must_have = min(min_must_have_matches, len(must_have_normalized))

def must_have_match_count(skills: set[str]) -> int:
    return sum(1 for s in skills if s.strip().lower() in must_have_normalized)
```

A candidate is shortlisted only if **both** hold:
- `len(skills) >= required` (existing total-match rule, unchanged)
- `must_have_match_count(skills) >= required_must_have` (new)

`required_must_have = min(min_must_have_matches, len(must_have_normalized))`
mirrors the existing `required = min(min_matches, deduplicated_count)`
pattern exactly: if a job profile's must-have set is smaller than the
configured threshold, the threshold shrinks to match rather than
making the job profile unshortlistable. When `must_have_skills` is empty
(old-format JDSkills, or a JD the LLM classified with no must-haves),
`required_must_have` is 0 and every candidate automatically satisfies
the must-have gate — today's behavior is preserved exactly.

## Wiring (`pipeline.py`, `cli.py`)

`min_must_have_matches` threads through exactly where `min_skill_matches`
already does:
- `build_pipeline_graph(..., min_must_have_matches: int = 2)`
- `build_shortlist_for_jd` passes it to `shortlist_candidates`
- `run()` in `cli.py` gains `min_must_have_matches: int = 2`, forwarded to `build_pipeline_graph`
- CLI argparse gains `--min-must-have-matches` (default 2)
- Manifest gains a `"min_must_have_matches"` key, written alongside
  `min_skill_matches`, and read back the same way on resume
  (`prior_manifest.get("min_must_have_matches", min_must_have_matches)`)
- The empty-shortlist warning message includes the new value alongside
  the existing `skill_match_threshold`/`min_skill_matches` for debugging

## Caching

No manual cache invalidation needed: `_cache_key` in `jd_skills.py`
hashes the fully-formatted prompt text plus model name. Changing the
prompt changes that hash, so every JD's cached extraction misses and
regenerates automatically on first run after this change — existing
cache entries are simply never matched again, not corrupted.

## Error Handling

- LLM names a must-have skill that isn't in its own `technical_skills`
  output: filtered out silently (from the caller's perspective) with a
  `logger.warning`; does not fail generation or trigger retry.
- A JD with zero must-have skills after filtering: valid state,
  `required_must_have` collapses to 0, no candidate is blocked on this
  gate. No warning needed here — this is a legitimate JD, not an error.
- `min_must_have_matches` behaves identically to `min_skill_matches` on
  invalid input (e.g., negative) — no new validation beyond what
  already exists for `min_skill_matches`, since none exists today.

## Testing

- `tests/scoring/test_jd_skills.py`: a must-have skill outside
  `technical_skills` is dropped with a warning; a normal case where
  must-have is a proper subset passes through unchanged; an empty
  `must_have_skills` result is accepted without triggering the
  empty-list retry (that retry is keyed on `technical_skills` only).
- `tests/test_models.py`: `JDSkills` accepts and defaults
  `must_have_skills` correctly.
- `tests/scoring/test_skills.py` (new file — `shortlist_candidates`
  has no dedicated test file today; follows the naming convention of
  the sibling `tests/scoring/test_jd_skills.py`): four cases —
  1. candidate meets total-match threshold but not must-have threshold → excluded
  2. candidate meets must-have threshold but not total-match threshold → excluded
  3. candidate meets both → included
  4. `jd_skills.must_have_skills` empty → behavior identical to before this change (regression guard)
- Existing tests for `shortlist_candidates`, `generate_jd_skills`, and
  `JDSkills` continue to pass unmodified, since every new field
  defaults to empty/0-effect.
