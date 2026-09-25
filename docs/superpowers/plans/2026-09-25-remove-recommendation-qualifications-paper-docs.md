# Remove Recommendation/Qualifications from Paper & Docs (Sub-project 5/5, Final) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every remaining prose description of Jev's assessment as including a hiring recommendation, a minimum-qualifications judgment, or a certification check — the pipeline is Score-only since sub-project 1/5 — across the paper and two supporting docs.

**Architecture:** Pure text edits, no code. Task 1 covers `docs/paper2_jev.tex` (the paper itself). Task 2 covers `README.md` and `docs/jev_vs_paper1_comparison.md` (project docs, scoped by the same repo-wide identifier sweep that scoped the whole sub-project). This is the final sub-project of a 5-part removal spanning core pipeline → output layer → frontend → evaluation scripts → paper/docs.

**Tech Stack:** LaTeX (paper), Markdown (docs). No build step, no compiler run.

## Global Constraints

- Never run `pdflatex` or any LaTeX compiler, and never create or commit a `docs/paper2_jev.pdf` — edit `.tex` source as text only; the user compiles it themselves.
- `docs/` is gitignored at the repo root; committing changes to already-tracked files under it (all three files in this plan are already tracked) needs `git add -f`.
- Do not touch Section II's two paragraphs describing Jev's three typed primitives (Noul/Choice/Score) in general terms, or Section III-D's "Match-degree questions use Score, not Choice..." sentence — these are accurate background/design-rationale about Jev's architecture in general, not claims about what this pipeline currently uses, and the user explicitly decided to keep them during design.
- Do not touch any file under `docs/superpowers/` (historical spec/plan records — they correctly retain old field names as part of that record) or any code file — this sub-project is prose-only in exactly the three files named below.

---

### Task 1: Fix `docs/paper2_jev.tex`

**Files:**
- Modify: `docs/paper2_jev.tex`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing consumed by Task 2 (independent file).

- [ ] **Step 1: Rewrite Section III-D's description of what a Jev call asks**

```latex
% OLD
Where the prior system's free text lived, the pipeline sends one call to Jev per shortlisted pair (C1). Each call gives Jev one state (job profile, extracted skills, resume text) and typed questions answered in one pass: a hiring-recommendation Choice, a minimum-qualifications Noul, one Score per technical requirement, one Score per stated seniority or education requirement, and one Noul per stated certification -- one judgment per field, combined afterward. Each answer is retained in full, including its per-level (Score) or per-option (Choice) probability distribution alongside the collapsed value and confidence, so downstream stages can use either representation.

% NEW
Where the prior system's free text lived, the pipeline sends one call to Jev per shortlisted pair (C1). Each call gives Jev one state (job profile, extracted skills, resume text) and typed Score questions answered in one pass: one Score per technical requirement, and one Score per stated seniority or education requirement -- one judgment per field, combined afterward. Each answer is retained in full, including its per-level probability distribution alongside the collapsed value and confidence, so downstream stages can use either representation.
```

Note the second sentence also drops "or per-option (Choice)" from the probability-distribution description, since Choice answers no longer exist anywhere in this pipeline's output.

- [ ] **Step 2: Fix the comparison table's "Assessment" row**

```latex
% OLD
Assessment & Free-text strengths/weaknesses + retry-on-contradiction & Typed Score/Choice/Noul answered in one parallel pass\\

% NEW
Assessment & Free-text strengths/weaknesses + retry-on-contradiction & Typed Score answered in one parallel pass\\
```

- [ ] **Step 3: Remove the Choice-questions future-work item from the Conclusion**

```latex
% OLD
Future work includes independently testing RLCD's calibration guarantee against ground-truth correctness rather than self-consistency alone, since calibration does not itself guarantee any individual prediction is correct; testing whether the per-level independence property behind criteria-grounded design holds across TypeSafe's full documented 2--10 level range, not only the five used here; and extending the same criteria-grounded principle to Choice questions, where TypeSafe's own guidance is comparatively undeveloped.

% NEW
Future work includes independently testing RLCD's calibration guarantee against ground-truth correctness rather than self-consistency alone, since calibration does not itself guarantee any individual prediction is correct; and testing whether the per-level independence property behind criteria-grounded design holds across TypeSafe's full documented 2--10 level range, not only the five used here.
```

- [ ] **Step 4: Verify**

Run: `grep -n -i "hiring-recommendation\|minimum-qualifications Noul\|per stated certification\|Score/Choice/Noul\|Choice questions" docs/paper2_jev.tex`

Expected: no output (empty). This deliberately does NOT search for bare "Choice" or "Noul" — those still appear legitimately in the Section II background paragraphs and the Section III-D "Match-degree questions use Score, not Choice" sentence, both explicitly kept per Global Constraints.

- [ ] **Step 5: Commit**

```bash
git add -f docs/paper2_jev.tex
git commit -m "$(cat <<'EOF'
docs: correct paper2_jev.tex's description of Jev's question set

Section III-D, the comparison table, and the future-work list all
described the assessment call as including a hiring-recommendation
Choice, a minimum-qualifications Noul, and a per-certification Noul --
none of which exist since sub-project 1/5 made the pipeline Score-only.
Section II's general background on Jev's three primitives, and Section
III-D's Score-vs-Choice design-rationale sentence, are kept unchanged
per explicit design decision -- they describe Jev's architecture in
general, not this pipeline's current usage. Sub-project 5/5 (final).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix `README.md` and `docs/jev_vs_paper1_comparison.md`

**Files:**
- Modify: `README.md`
- Modify: `docs/jev_vs_paper1_comparison.md`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: nothing consumed by later tasks (last task in this plan and in the whole 5-part removal effort).

- [ ] **Step 1: Fix `README.md`'s intro paragraph**

```markdown
# OLD
Jev answers a fixed set of typed questions (an overall fit score, a hiring
recommendation, a minimum-qualifications judgment, and one score per
extracted requirement) in a single parallel pass per call, with a
model-reported confidence on every answer.

# NEW
Jev answers a fixed set of typed questions (an overall fit score, and
one score per extracted requirement) in a single parallel pass per call, with a
model-reported confidence on every answer.
```

- [ ] **Step 2: Fix `README.md`'s Stage 4 pipeline bullet**

```markdown
# OLD
4. **Structured assessment via Jev** — per shortlisted (JD, candidate) pair,
   one Jev call is issued (answering the recommendation,
   minimum-qualifications, per-requirement, and any
   certification/seniority/education questions in one parallel pass),
   producing a single assessment (cached per JD/candidate under
   `runs/_cache/assessments/`).

# NEW
4. **Structured assessment via Jev** — per shortlisted (JD, candidate) pair,
   one Jev call is issued (answering the per-requirement, and any
   seniority/education questions in one parallel pass),
   producing a single assessment (cached per JD/candidate under
   `runs/_cache/assessments/`).
```

- [ ] **Step 3: Fix `docs/jev_vs_paper1_comparison.md`'s "Output shape" row**

```markdown
# OLD
| Output shape | Prose: strengths, weaknesses, additional skills | Structured: overall_fit_score, overall_recommendation, meets_min_qualifications, per-requirement scores, confidence |

# NEW
| Output shape | Prose: strengths, weaknesses, additional skills | Structured: overall_fit_score, per-requirement scores, confidence |
```

- [ ] **Step 4: Fix `docs/jev_vs_paper1_comparison.md`'s full-stack-engineer case-study paragraph**

```markdown
# OLD
Paper 1's per-role Faithfulness scores rank **full-stack-engineer lowest**
(0.764, vs. a 0.880 run-wide mean). Independently, the Jev evaluation on
this repo's current corpus also found full-stack-engineer among the
weakest roles: mean `overall_fit_score` 34.2 and 0% of its shortlisted
candidates meeting minimum qualifications (see the shortlisting
investigation in this session -- shortlisted candidates matched generic
skills like GitHub Actions/Kubernetes/AWS but scored zero on React/Next.js/
PostgreSQL, the role's actual core skills).

# NEW
Paper 1's per-role Faithfulness scores rank **full-stack-engineer lowest**
(0.764, vs. a 0.880 run-wide mean). Independently, the Jev evaluation on
this repo's current corpus also found full-stack-engineer among the
weakest roles: mean `overall_fit_score` 34.2 (see the shortlisting
investigation in this session -- shortlisted candidates matched generic
skills like GitHub Actions/Kubernetes/AWS but scored zero on React/Next.js/
PostgreSQL, the role's actual core skills).
```

- [ ] **Step 5: Verify**

Run: `grep -n -i "hiring recommendation\|minimum-qualifications judgment\|answering the recommendation\|overall_recommendation\|meets_min_qualifications\|meeting minimum qualifications" README.md docs/jev_vs_paper1_comparison.md`

Expected: no output (empty).

- [ ] **Step 6: Run the full repo-wide sweep from the spec, across the whole repo, to confirm sub-project 5/5 (and the whole 5-part removal effort) is complete**

Run:
```bash
grep -rln --include="*.py" --include="*.ts" --include="*.tsx" --include="*.md" --include="*.tex" \
  -i "overall_recommendation\|meets_min_qualifications\|certification_results\|recommendation_probabilities\|hireRate\|meetsMinRate\|recommendationAgreementRate\|recommendation_full_agreement\|unaffected_control\|hiring recommendation\|minimum.qualification" \
  --exclude-dir=node_modules --exclude-dir=dist --exclude-dir=runs --exclude-dir=.superpowers . \
  | grep -v "docs/superpowers/\|real-data.json"
```

Expected output: exactly these four lines (all confirmed-intentional per this plan's spec — absence-assertions, a deliberately-generic client test, and a deliberately-preserved legacy fixture; none require changes):
```
tests/output/test_formatter.py
tests/test_models.py
tests/scoring/test_jev_client.py
tests/output/test_console_export.py
```

If anything else appears, stop and investigate before committing — it means something was missed.

- [ ] **Step 7: Commit**

```bash
git add README.md
git add -f docs/jev_vs_paper1_comparison.md
git commit -m "$(cat <<'EOF'
docs: correct README and paper comparison doc's Jev question descriptions

Both still described Jev's assessment call as including a hiring
recommendation and a minimum-qualifications judgment; the comparison
doc's case-study paragraph also cited a "0% meeting minimum
qualifications" figure for a metric that no longer exists. All three
corrected to match the Score-only pipeline. Completes sub-project 5/5
and the full 5-part recommendation/qualifications removal effort.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** all three of the spec's named files and every edit it lists map to a task step — `docs/paper2_jev.tex`'s three edits → Task 1; `README.md`'s two edits and `docs/jev_vs_paper1_comparison.md`'s two edits → Task 2. The spec's "explicitly kept" items (Section II background, Section III-D's Score-vs-Choice sentence) are named in Global Constraints so neither task touches them, and Task 1's Step 4 verification grep is deliberately narrow (specific phrases, not bare "Choice"/"Noul") so it doesn't false-positive on the kept text.
- **Placeholder scan:** none — every step has literal text and exact commands.
- **Consistency check:** Task 2's Step 6 final sweep command is the same one that scoped this whole sub-project during design (per the spec's Purpose section), giving an end-to-end, repo-wide confirmation that this is genuinely the last piece of the 5-part effort — not just this task's own narrow claim.
