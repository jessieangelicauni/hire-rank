# Remove Recommendation/Qualifications from Paper & Docs (Sub-project 5/5, Final)

## Purpose

Sub-projects 1-4/5 removed `overall_recommendation`/`meets_min_qualifications`/`certification_results`/`recommendation_probabilities` from the production pipeline, its output layer, its frontend, and its evaluation scripts (all done, merged). This final sub-project brings the prose that describes the system in line with what it now actually does.

Applying the lesson from sub-project 4's final review (whose own scoping missed two stale doc lines because its spec grepped only one hand-picked file for two English phrases instead of the exact removed identifiers across the whole repo): this sub-project's scope was determined by a repo-wide, case-insensitive grep for `overall_recommendation`, `meets_min_qualifications`, `certification_results`, `recommendation_probabilities`, `hireRate`, `meetsMinRate`, `recommendationAgreementRate`, `recommendation_full_agreement`, `unaffected_control`, `hiring recommendation`, and `minimum qualification` across every `.py`/`.ts`/`.tsx`/`.md`/`.tex` file, excluding `node_modules`, `dist`, `runs/`, `.superpowers/` (session scratch), and `docs/superpowers/` (historical plan/spec records, which correctly still describe past states and are not living documentation). The sweep found exactly three files needing real edits — `docs/paper2_jev.tex`, `README.md`, `docs/jev_vs_paper1_comparison.md` — and confirmed every other hit (in `tests/`) is an intentional absence-assertion, a deliberately-generic `jev_client.py` test, or a deliberately-preserved legacy-data fixture, none of which need changing.

## 1. `docs/paper2_jev.tex`

Three edits, all fixing text that is now factually wrong about what the pipeline does (not a framing/scope judgment call — verified against the sweep and against the actual Score-only `_build_questions` in `assessment.py`):

- **Section III-D (Stage 3 methodology)**: the sentence describing what a Jev call asks currently lists a hiring-recommendation Choice, a minimum-qualifications Noul, and a per-certification Noul alongside the Score questions — none of these exist anymore. Rewrite to describe only the Score questions actually sent (per-requirement, seniority, education).
- **`tab_comparison`, "Assessment" row**: "Typed Score/Choice/Noul answered in one parallel pass" → "Typed Score answered in one parallel pass" (this work's assessment is Score-only now).
- **Conclusion, future work**: remove the clause "and extending the same criteria-grounded principle to Choice questions, where TypeSafe's own guidance is comparatively undeveloped" — this reads as a still-open research direction on an architecture the pipeline still uses Choice within, but the pipeline no longer uses Choice at all (a product decision, not a deferred-research gap), so proposing to "extend" criteria-grounded design to it no longer makes sense as this paper's future work.

**Explicitly kept, per user decision during design**: Section II's two paragraphs describing Jev's three typed primitives (Noul/Choice/Score) in general terms, and Section III-D's "Match-degree questions use Score, not Choice, since a match degree is a position on a spectrum" sentence. Both are accurate background/design-rationale about Jev's architecture and why Score was chosen among the available primitives — not claims about what else this specific pipeline currently uses — and stay as useful context for the reader.

## 2. `README.md`

Two edits, both in prose describing what Jev is asked (not the already-corrected "N calls" framing from sub-project 2, which is unaffected):

- Intro paragraph ("Jev answers a fixed set of typed questions..."): drop "a hiring recommendation, a minimum-qualifications judgment, and" from the parenthetical list, leaving "an overall fit score... and one score per extracted requirement."
- Pipeline stage-4 bullet ("Structured assessment via Jev — ... one Jev call is issued (answering the recommendation, minimum-qualifications, per-requirement, and any certification/seniority/education questions..."): drop "the recommendation, minimum-qualifications," and "certification/", leaving "answering the per-requirement, seniority, and education questions."

## 3. `docs/jev_vs_paper1_comparison.md`

Two edits:

- **`tab_comparison`-style "Output shape" row**: "Structured: overall_fit_score, overall_recommendation, meets_min_qualifications, per-requirement scores, confidence" → "Structured: overall_fit_score, per-requirement scores, confidence" (the two removed fields dropped from the list of what Jev's output actually contains).
- **Full-stack-engineer case-study paragraph**: per user decision, the clause "and 0% of its shortlisted candidates meeting minimum qualifications" is removed (the metric it names no longer exists), leaving the sentence's substantive finding intact via the `overall_fit_score` figure and the explanatory parenthetical that follows it (which already describes the mechanism — matched generic skills but scored zero on the role's actual core skills — independent of the removed metric).

## Out of scope

- Every other repo-wide grep hit (`tests/output/test_formatter.py`, `tests/test_models.py`, `tests/scoring/test_jev_client.py`, `tests/output/test_console_export.py`) is intentional: absence-assertions (`assert "overall_recommendation" not in ...`), `jev_client.py`'s deliberately-generic client tests (unrelated to what production actually asks), or `test_console_export.py`'s deliberately-preserved legacy `test_retest.json`-shape fixture (sub-project 4's design explicitly kept this to prove old data doesn't break new code). None are edited.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` (including this sub-project's own future spec/plan files) are historical records of what was decided and built at each point in time — they correctly retain old field names as part of that record and are not edited.
- No code changes anywhere — this sub-project is prose-only.
- `pdflatex` is never run and `docs/paper2_jev.pdf` is never committed, per standing project convention — `.tex` source is edited as text only.

## Testing

No automated verification exists for `.tex`/`.md` prose. Verification is:
- Re-run the same repo-wide grep from this design's Purpose section after editing; confirm the only remaining hits are the explicitly-out-of-scope files listed above.
- Read each edited paragraph/row in full context after the edit to confirm it still reads grammatically and doesn't leave a dangling reference (e.g. a sentence that used to end differently, or a table row whose column alignment assumptions changed).
