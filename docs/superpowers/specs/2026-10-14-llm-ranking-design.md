# LLM-Based Candidate Ranking System — Design Spec

**Date:** 2026-10-14  
**Status:** Draft — pending implementation

---

## 1. Goal

Replace the existing Jev-based assessment pipeline with a fully self-hosted LLM pipeline that:

- Performs **non-compensatory qualification gating** — a candidate missing any must-have requirement is rejected regardless of other strengths
- Produces **evidence-based structured evaluation** — every verdict is backed by a direct citation from the CV
- Ranks qualified candidates through **comparative listwise judgment** rather than absolute scoring
- Runs entirely on **Qwen2.5-14B-Instruct (4-bit, Ollama)** — no external API dependency beyond Ollama

This is a fresh design. It does not extend or continue any prior system in this repository.

---

## 2. Architecture

```
CV pool + JD files
        │
        ▼
┌─────────────────────────────┐
│  Ingestion + Skill          │  unchanged
│  Extraction + Shortlisting  │  cosine ≥ 0.8, ≥5 matches, ≥2 must-have
└─────────────────────────────┘
        │ shortlisted (JD, CV) pairs
        ▼
┌─────────────────────────────┐
│  PROMPT 1                   │  1 call per JD
│  JD Requirement Extractor   │  → must_have[], nice_to_have[]
└─────────────────────────────┘
        │ requirements JSON (cached per JD)
        ▼
┌─────────────────────────────┐
│  PROMPT 2                   │  1 call per (JD, CV), parallel
│  CV Qualification Gate      │  → structured evidence + qualified verdict
└─────────────────────────────┘
        │ qualified candidates with structured evidence
        ▼
┌─────────────────────────────────────────────────────┐
│  PROMPT 3 — Adaptive Ranking                        │
│                                                     │
│  pool ≤ 20:  one-shot listwise × 3 (shuffled order) │
│  pool > 20:  MC-KG tournament (subset size 10)      │
│                                                     │
│  Both paths → Plackett-Luce aggregation             │
└─────────────────────────────────────────────────────┘
        │ ranked list + utility scores + variance
        ▼
     Output (ranking.md, ranking.json, console export)
```

---

## 3. Model Configuration

| Parameter | Value |
|---|---|
| Model | `qwen2.5:14b-instruct-q4_K_M` |
| Inference | Ollama (self-hosted) |
| `CANDIDATE_RANKING_OLLAMA_NUM_CTX` | `16384` |
| Temperature — Prompt 1 | `0` |
| Temperature — Prompt 2 | `0` |
| Temperature — Prompt 3 | `0.3–0.5` (diversity across runs) |

Temperature > 0 on Prompt 3 is intentional: three runs must produce different orderings so Plackett-Luce aggregation is meaningful. At temperature 0 all three runs are identical and aggregation adds nothing.

---

## 4. Prompt 1 — JD Requirement Extractor

**Runs:** once per JD, output cached.

```
You are a senior technical recruiter parsing a job description.

Extract all requirements and classify each as:
- MUST_HAVE: non-negotiable; absence alone disqualifies a candidate
- NICE_TO_HAVE: preferred but absence does not disqualify

Rules:
- Be conservative with MUST_HAVE — only use it if the JD treats the
  requirement as essential, not merely preferred
- "Preferred", "a plus", "bonus" in JD text → NICE_TO_HAVE at most
- Minimum years of experience and seniority level → MUST_HAVE if stated
- Degree/education → MUST_HAVE only if explicitly required, not just preferred

JOB DESCRIPTION:
{jd_text}
```

**Output schema:**

```json
{
  "must_have": [
    {"requirement": "Python (3+ years)", "context": "Primary language, listed as required"},
    {"requirement": "REST API design", "context": "Core daily responsibility"}
  ],
  "nice_to_have": [
    {"requirement": "Docker", "context": "Listed as preferred"},
    {"requirement": "AWS experience", "context": "Mentioned as a plus"}
  ]
}
```

---

## 5. Prompt 2 — CV Qualification Gate

**Runs:** once per shortlisted (JD, CV) pair, all pairs in parallel.

```
You are a senior technical recruiter screening a CV.

EVIDENCE RULES — apply consistently:
  STRONG   = skill demonstrated in work experience with specific system or project
  ADEQUATE = skill mentioned in experience context but without project specifics
  WEAK     = skill listed in skills/summary section ONLY, never shown in any
             experience or project

Verdict mapping:
  MET           = strong or adequate evidence
  PARTIALLY_MET = weak evidence only (skills section, never shown in experience)
  NOT_MET       = no evidence anywhere in CV

NON-COMPENSATORY RULE:
  If any MUST_HAVE verdict is NOT_MET → candidate is disqualified immediately.
  Strong performance elsewhere does not compensate.

EVIDENCE CITATION RULE:
  For every verdict, quote or reference the exact section of the CV where
  the evidence appears. If you cannot find specific text → verdict is NOT_MET
  or PARTIALLY_MET (weak), never MET.

MISSED EVIDENCE RULE:
  Before concluding NOT_MET for any requirement, re-read ALL sections:
  skills, experience, projects, education, certifications, and summary.
  Skills are often implied — "Django" implies Python, "Spring Boot" implies Java.

DURATION RULE:
  Compute total_relevant_years by summing durations of roles where the
  candidate used skills relevant to this JD. Use CV dates directly.
  years_demonstrated per requirement = years that skill appears in active use.

Evaluate every MUST_HAVE requirement first. If candidate qualifies, also
evaluate NICE_TO_HAVE requirements and fill all remaining fields.

ROLE: {jd_title}

MUST_HAVE REQUIREMENTS:
{must_have_json}

NICE_TO_HAVE REQUIREMENTS:
{nice_to_have_json}

CV:
{cv_text}
```

**Output schema:**

```json
{
  "must_have_evaluations": [
    {
      "requirement": "Python (3+ years)",
      "verdict": "MET",
      "evidence": "FastAPI microservices at PT XYZ (2022–present); SQLAlchemy, pytest, async Python in project descriptions",
      "evidence_strength": "strong",
      "recency": "current",
      "years_demonstrated": 3
    }
  ],
  "nice_to_have_evaluations": [
    {
      "requirement": "Docker",
      "verdict": "MET",
      "evidence": "Dockerized 3 production services (mentioned in 2 project descriptions)",
      "evidence_strength": "adequate",
      "recency": "current",
      "years_demonstrated": 2
    }
  ],
  "total_relevant_years": 5,
  "career_trajectory": {
    "direction": "ascending",
    "evidence": "Junior Dev 2019 → Mid 2021 → Senior 2023; avg ~2yr/role",
    "tenure_pattern": "stable",
    "average_tenure_months": 24,
    "most_recent_role_duration_months": 18,
    "has_multinational_experience": false,
    "multinational_evidence": null
  },
  "strongest_achievements": [
    "Reduced p99 latency 800ms→120ms serving 2M DAU",
    "Led payment module processing Rp500M/day"
  ],
  "notable_signals": "Open source contributor to FastAPI core (2 merged PRs)",
  "red_flags": [],
  "qualified": true,
  "disqualification_reason": null
}
```

**Disqualification output (when qualified = false):**

```json
{
  "must_have_evaluations": [
    {
      "requirement": "Python (3+ years)",
      "verdict": "NOT_MET",
      "evidence": "Python listed in skills section only. All experience descriptions use Java exclusively. No Python in any project or role.",
      "evidence_strength": "weak",
      "recency": "unknown",
      "years_demonstrated": 0
    }
  ],
  "nice_to_have_evaluations": [],
  "total_relevant_years": 4,
  "career_trajectory": null,
  "strongest_achievements": [],
  "notable_signals": null,
  "red_flags": [],
  "qualified": false,
  "disqualification_reason": "Python not demonstrated in any work experience — listed in skills section only; all roles use Java."
}
```

---

## 6. Prompt 3 — Listwise Ranking

**Runs:** 3× per JD (pool ≤ 20) or ~N iterations (pool > 20). Candidate order shuffled per run.

Input per candidate: structured evidence from Prompt 2 output (not full CV text).

`{jd_summary}` = JD title + key responsibilities section (≤ 200 words), extracted by truncating JD text — not a separate LLM call. Gives the ranker enough context without burning tokens on boilerplate like company culture paragraphs.

```
You are a panel of senior HR professionals making a final ranking decision.

All candidates below have already passed qualification screening — they all
meet the minimum requirements. Rank them from STRONGEST to WEAKEST fit
for this specific role.

RANKING CRITERIA (in order of importance):
1. Relevance — whose experience most directly matches what this role does daily
2. Evidence quality — concrete achievements and measurable impact over vague responsibilities
3. Career trajectory — growing scope, promotions, increasing seniority over time
4. Recency — recent hands-on experience with core stack outweighs older exposure
5. Nice-to-have coverage — breadth of preferred skills as a tiebreaker

RULES:
- Rank ALL candidates. No ties. Forced ordering.
- For each rank position, state the key reason this candidate ranks above the next.
- Judge only on information provided. Do not assume unlisted skills.
- When two candidates feel equal: "Who would I call for interview first?" —
  that person ranks higher.

ROLE: {jd_title}

JOB SUMMARY: {jd_summary}

CANDIDATES:
{candidate_evidence_json}
```

**Output schema:**

```json
{
  "ranking": [
    {
      "rank": 1,
      "candidate_id": "cv_003",
      "key_differentiator": "Only candidate with quantified latency impact at scale (2M DAU). Career trajectory consistently ascending with promotions."
    },
    {
      "rank": 2,
      "candidate_id": "cv_001",
      "key_differentiator": "Strong Python depth but achievements are vague — no numbers. Less recent Docker experience."
    }
  ],
  "overall_reasoning": "..."
}
```

---

## 7. Ranking Aggregation

Rankings from multiple Prompt 3 runs are aggregated using **Plackett-Luce** to produce stable utility scores per candidate.

```
Pool ≤ 20 candidates:
  3 independent one-shot runs (full pool, shuffled order each run)
  → P-L fit over 3 complete rankings
  → Output: utility score + variance per candidate

Pool > 20 candidates:
  MC-KG subset selection (subset size = 10)
  → iterative tournament until Δu < threshold or budget exhausted
  → P-L refit after each iteration
  → Output: utility score + variance per candidate
```

The three-run design for small pools mitigates **position bias** (candidates listed first/last in prompt tend to rank differently regardless of quality). Shuffling + P-L aggregation averages this out.

---

## 8. Structured Evidence Fields — Design Rationale

### Fields included and why

| Field | Why included |
|---|---|
| `verdict` + `evidence` + `evidence_strength` + `recency` | Core qualification signal; verifiable from CV text |
| `years_demonstrated` | Computed from dates; validates seniority requirements |
| `total_relevant_years` | Computed from dates; seniority comparison across candidates |
| `career_trajectory` | Verifiable from job titles + dates; growth signal |
| `average_tenure_months` | Computed from dates; stability signal |
| `most_recent_role_duration_months` | Computed from dates; current commitment signal |
| `has_multinational_experience` | Valid exposure signal — only flagged with explicit CV evidence |
| `strongest_achievements` | Directly cited from CV; impact differentiator |
| `notable_signals` | Free-text catch-all for things not in requirements |
| `red_flags` + `duration_months` | Verifiable concerns with magnitude |

### Fields explicitly excluded and why

| Field excluded | Reason |
|---|---|
| `company_scale` | Proxy for prestige, not quality of work; geographic/economic bias |
| `is_multinational` (as boolean weight) | Replaced by `has_multinational_experience` with required explicit evidence |
| `ownership_level` | Inferred from CV writing style, not actual ownership; writing skill bias |
| `technical_complexity` | Inferred from buzzwords, not actual complexity; terminology bias |
| `university_name` | Institution prestige bias |
| `gpa` | Not relevant for experienced candidates |
| Absolute scores (0–100) | False precision; compensatory assumption; LLM has no calibrated scale |

---

## 9. No Retry Loop

Paper 1 required a retry loop because free-text generation can produce claims that contradict the CV. This system does not generate free-text claims — it classifies and cites. The evidence citation requirement is the built-in check: if the model cannot produce specific cited text, the verdict must be PARTIALLY_MET or NOT_MET.

A retry loop is not needed and not included.

---

## 10. Files to Build / Change

```
NEW:
  scoring/jd_requirements.py     Prompt 1 runner — JD requirement extraction
  scoring/llm_assessment.py      Prompt 2 runner — CV qualification gate
                                 Prompt 3 runner — listwise ranking call

REPLACE:
  scoring/jev_client.py     →    delete entirely
  scoring/assessment.py     →    update Assessment model (new schema, no Jev fields)

UPDATE:
  models.py                      update Assessment + new Qualification dataclasses
  graphs/pipeline.py             wire new stages, remove Jev call
  output/console_export.py       include evidence fields in export

UNCHANGED:
  scoring/skills.py              shortlisting — thresholds stay as-is
  scoring/jd_skills.py           JD skill extraction
  ranking/plackett_luce.py       P-L fitting
  ranking/mc_kg.py               MC-KG subset selection
  ranking/tournament.py          tournament runner
  ingestion/                     CV and JD loading
  json_cache.py                  caching layer
  config.py                      add OLLAMA_NUM_CTX default = 16384
```

---

## 11. Cost Profile

| Stage | Calls | Est. tokens/call | Scale |
|---|---|---|---|
| Prompt 1 — JD extraction | 1 per JD | ~900 | O(J) |
| Prompt 2 — Qualification gate | 1 per shortlisted pair | ~1,900 | O(N) |
| Prompt 3 — Ranking (pool ≤ 20) | 3 per JD | ~2,700 | O(J) |
| Prompt 3 — Ranking (pool > 20) | ~35 per JD | ~2,300 | O(J × k) |

**At current scale (500 CV, 10 JD, ~15% shortlist rate):**
~350 qualification calls + 30 ranking calls + 10 JD extraction = ~390 total calls

**At future scale (5,000 CV, 50 JD, ~15% shortlist rate):**
~3,750 qualification calls + 150 ranking calls (assuming pools mostly ≤ 20 post-gate) + 50 JD extraction = ~3,950 total calls

Shortlisting thresholds are kept at current values (cosine ≥ 0.8, ≥ 5 matches, ≥ 2 must-have) for cost control at scale. These are not loosened.

---

## 12. Open Questions (not blocking implementation)

1. **Ground truth validation** — no human-expert ranking exists. Internal consistency metrics (Kendall-tau convergence, test-retest stability) can be measured; external validity cannot without HR expert annotation.

2. **Multinational evidence detection accuracy** — model reliability for identifying multinational companies from CV text has not been empirically measured.

3. **Qualification gate false negative rate** — how many genuinely qualified candidates are incorrectly disqualified? Cannot be measured without ground truth.
