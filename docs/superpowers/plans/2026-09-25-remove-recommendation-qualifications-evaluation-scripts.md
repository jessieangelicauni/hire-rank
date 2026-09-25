# Remove Recommendation/Qualifications from Evaluation Scripts (Sub-project 4/5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `collect_test_retest`'s `AttributeError` crash, remove the now-dead "unaffected control" ablation mechanism, and remove the recommendation-agreement metric end-to-end from evaluation scripts through to the Analytics UI.

**Architecture:** Task 1 fixes the two Python evaluation scripts (the actual crash, plus the two agreed methodology removals). Task 2 fixes `console_export.py`'s corresponding read (`.get()` already made this non-crashing, but the field should stop being produced at all). Task 3 removes the now-permanently-null field/card from the frontend, completing the same removal chain sub-project 3/5 already applied to the other recommendation-related fields.

**Tech Stack:** Python 3 (pytest for `console_export.py` only — the two evaluation scripts have no automated test coverage in this repo, verified manually), TypeScript/React.

## Global Constraints

- `collect_seniority_education_ablation`/`analyze_seniority_education_ablation` are untouched — they never had an "unaffected control" mechanism; this plan brings `collect_ablation`/`analyze_ablation` in line with their existing pattern, not the other way around.
- `docs/paper2_jev.tex` is untouched — sub-project 5/5's job. A grep already confirmed the paper has no reference to "unaffected control" or "recommendation agreement," so there's nothing for sub-project 5 to clean up from this specific change.
- `src/candidate_ranking/scoring/assessment.py` and `models.py` are untouched — sub-project 1/5 already finished that; this plan only touches code downstream of it.
- No automated test file exists for `scripts/run_jev_evaluation_study.py` or `scripts/analyze_jev_evaluation_study.py` in this repo (confirmed: `tests/test_scripts_import.py` only import-smoke-tests every file under `scripts/`, which would not have caught this task's bug since it's inside a function body, not at import time). Task 1's verification is a manual `python -c` sanity check using the same dynamic-import technique `tests/test_scripts_import.py` already uses, not a new pytest file — this matches existing project convention for these two scripts.

---

### Task 1: Fix `scripts/run_jev_evaluation_study.py` and `scripts/analyze_jev_evaluation_study.py`

**Files:**
- Modify: `scripts/run_jev_evaluation_study.py`
- Modify: `scripts/analyze_jev_evaluation_study.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `collect_test_retest`'s per-repeat dicts now contain only `{"composite_fit_score", "requirement_scores", "latency_seconds"}` (no `overall_recommendation`/`meets_min_qualifications`) — this is the `test_retest.json` shape Task 2/3 and any future run must assume. `analyze_test_retest`'s return dict no longer has `recommendation_full_agreement_rate`. `analyze_ablation`'s return dict no longer has `unaffected_control`.

- [ ] **Step 1: Fix `_vague_build_questions` in `scripts/run_jev_evaluation_study.py`**

```python
# OLD
def _vague_build_questions(jd_technical_skills: list[str] | None) -> list[JevQuestion]:
    questions = [
        JevQuestion(
            key="overall_recommendation", kind="choice",
            instructions="What is the hiring recommendation for this candidate against this job description?",
            criteria={
                "hire": "Candidate clearly meets or exceeds the role's requirements",
                "maybe": "Candidate partially meets the role's requirements",
                "no": "Candidate does not meet the role's requirements",
            },
        ),
        JevQuestion(
            key="meets_min_qualifications", kind="noul",
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
                key=f"requirement::{requirement}", kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    return questions

# NEW
def _vague_build_questions(jd_technical_skills: list[str] | None) -> list[JevQuestion]:
    questions: list[JevQuestion] = []
    for requirement in jd_technical_skills or []:
        questions.append(
            JevQuestion(
                key=f"requirement::{requirement}", kind="score",
                instructions=f"How well does the candidate's CV support the requirement '{requirement}'?",
                criteria=_VAGUE_SCORE_CRITERIA,
            )
        )
    return questions
```

- [ ] **Step 2: Fix `collect_test_retest`'s crash in `scripts/run_jev_evaluation_study.py`**

```python
# OLD
        repeats_out = []
        for _ in range(repeats):
            answers, elapsed = _timed_evaluate(jev_client, state, questions)
            assessment = _answers_to_assessment(jd, candidate, JEV_MODEL_NAME, answers)
            repeats_out.append(
                {
                    "composite_fit_score": assessment.composite_fit_score,
                    "overall_recommendation": assessment.overall_recommendation,
                    "meets_min_qualifications": assessment.meets_min_qualifications,
                    "requirement_scores": assessment.requirement_scores,
                    "latency_seconds": elapsed,
                }
            )
        return {"jd_id": jd_id, "candidate_id": candidate_id, "repeats": repeats_out}

# NEW
        repeats_out = []
        for _ in range(repeats):
            answers, elapsed = _timed_evaluate(jev_client, state, questions)
            assessment = _answers_to_assessment(jd, candidate, JEV_MODEL_NAME, answers)
            repeats_out.append(
                {
                    "composite_fit_score": assessment.composite_fit_score,
                    "requirement_scores": assessment.requirement_scores,
                    "latency_seconds": elapsed,
                }
            )
        return {"jd_id": jd_id, "candidate_id": candidate_id, "repeats": repeats_out}
```

- [ ] **Step 3: Verify Steps 1-2 with a manual sanity check**

Run:
```bash
uv run python -c "
import sys, importlib.util
sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('run_study', 'scripts/run_jev_evaluation_study.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

questions = m._vague_build_questions(['Python', 'SQL'])
assert all(q.kind == 'score' for q in questions), 'vague questions must be score-only now'
assert {q.key for q in questions} == {'requirement::Python', 'requirement::SQL'}
print('_vague_build_questions OK:', [q.key for q in questions])
"
```

Expected output: `_vague_build_questions OK: ['requirement::Python', 'requirement::SQL']`, no errors.

(`collect_test_retest` itself calls the real Jev API and cannot be sanity-checked offline; Step 2's edit is a direct, narrow dict-literal change reviewable by inspection — the surrounding retry/threading logic is unchanged.)

- [ ] **Step 4: Remove `_UNAFFECTED_CONTROL_KEYS` and simplify `analyze_ablation` in `scripts/analyze_jev_evaluation_study.py`**

```python
# OLD
_UNAFFECTED_CONTROL_KEYS = ("overall_recommendation", "meets_min_qualifications")


def _paired_bucket(concrete: list[float], vague: list[float], label: str) -> dict:

# NEW
def _paired_bucket(concrete: list[float], vague: list[float], label: str) -> dict:
```

```python
# OLD
def analyze_ablation(records: list[dict]) -> dict:
    concrete_control: list[float] = []
    vague_control: list[float] = []
    concrete_requirement: list[float] = []
    vague_requirement: list[float] = []
    vague_latencies: list[float] = []

    for record in records:
        concrete_conf = record["concrete_confidence"]
        vague_conf = record["vague_confidence"]

        for key in _UNAFFECTED_CONTROL_KEYS:
            if key in concrete_conf and key in vague_conf:
                concrete_control.append(concrete_conf[key])
                vague_control.append(vague_conf[key])

        shared_requirement_keys = {
            k for k in concrete_conf if k.startswith("requirement::")
        } & {k for k in vague_conf if k.startswith("requirement::")}
        for key in shared_requirement_keys:
            concrete_requirement.append(concrete_conf[key])
            vague_requirement.append(vague_conf[key])

        vague_latencies.append(record["vague_latency_seconds"])

    return {
        "n_pairs": len(records),
        "unaffected_control": _paired_bucket(concrete_control, vague_control, "unaffected_control_confidence"),
        "requirement_questions": _paired_bucket(concrete_requirement, vague_requirement, "requirement_confidence"),
        "vague_latency_seconds_mean": statistics.mean(vague_latencies) if vague_latencies else None,
    }

# NEW
def analyze_ablation(records: list[dict]) -> dict:
    concrete_requirement: list[float] = []
    vague_requirement: list[float] = []
    vague_latencies: list[float] = []

    for record in records:
        concrete_conf = record["concrete_confidence"]
        vague_conf = record["vague_confidence"]

        shared_requirement_keys = {
            k for k in concrete_conf if k.startswith("requirement::")
        } & {k for k in vague_conf if k.startswith("requirement::")}
        for key in shared_requirement_keys:
            concrete_requirement.append(concrete_conf[key])
            vague_requirement.append(vague_conf[key])

        vague_latencies.append(record["vague_latency_seconds"])

    return {
        "n_pairs": len(records),
        "requirement_questions": _paired_bucket(concrete_requirement, vague_requirement, "requirement_confidence"),
        "vague_latency_seconds_mean": statistics.mean(vague_latencies) if vague_latencies else None,
    }
```

- [ ] **Step 5: Remove the "unaffected_control" line from `render_markdown`'s ablation section**

```python
# OLD
    if abl:
        lines += ["", "## Criteria-design ablation (concrete vs. vague Score criteria)", f"- n={abl['n_pairs']} sampled pairs"]
        if abl["unaffected_control"]["wilcoxon"]:
            w = abl["unaffected_control"]["wilcoxon"]
            lines.append(
                f"- unaffected_control confidence (overall_recommendation + meets_min_qualifications, criteria "
                f"NOT changed by the ablation -- expected null effect): "
                f"concrete mean={w['mean_first']:.3f} vs vague mean={w['mean_second']:.3f} "
                f"-- Wilcoxon p={w['p_value']:.4g}, r={w['rank_biserial_r']:.3f}"
            )
        if abl["requirement_questions"]["wilcoxon"]:

# NEW
    if abl:
        lines += ["", "## Criteria-design ablation (concrete vs. vague Score criteria)", f"- n={abl['n_pairs']} sampled pairs"]
        if abl["requirement_questions"]["wilcoxon"]:
```

- [ ] **Step 6: Remove `recommendation_full_agreement` tracking from `analyze_test_retest`**

```python
# OLD
def analyze_test_retest(records: list[dict]) -> dict:
    score_stdevs: list[float] = []
    requirement_score_stdevs: list[float] = []
    recommendation_full_agreement = 0
    latencies: list[float] = []

    for record in records:
        repeats = record["repeats"]
        scores = [r["composite_fit_score"] for r in repeats]
        if len(scores) >= 2:
            score_stdevs.append(statistics.stdev(scores))

        recommendations = {r["overall_recommendation"] for r in repeats}
        if len(recommendations) == 1:
            recommendation_full_agreement += 1

        requirement_keys = set(repeats[0]["requirement_scores"])

# NEW
def analyze_test_retest(records: list[dict]) -> dict:
    score_stdevs: list[float] = []
    requirement_score_stdevs: list[float] = []
    latencies: list[float] = []

    for record in records:
        repeats = record["repeats"]
        scores = [r["composite_fit_score"] for r in repeats]
        if len(scores) >= 2:
            score_stdevs.append(statistics.stdev(scores))

        requirement_keys = set(repeats[0]["requirement_scores"])
```

```python
# OLD
        "requirement_score_stdev": {
            "mean": statistics.mean(requirement_score_stdevs) if requirement_score_stdevs else None,
            "median": statistics.median(requirement_score_stdevs) if requirement_score_stdevs else None,
            "n_requirement_observations": len(requirement_score_stdevs),
        },
        "recommendation_full_agreement_rate": recommendation_full_agreement / n if n else None,
        "latency_seconds": {

# NEW
        "requirement_score_stdev": {
            "mean": statistics.mean(requirement_score_stdevs) if requirement_score_stdevs else None,
            "median": statistics.median(requirement_score_stdevs) if requirement_score_stdevs else None,
            "n_requirement_observations": len(requirement_score_stdevs),
        },
        "latency_seconds": {
```

- [ ] **Step 7: Remove the "recommendation full agreement rate" line from `render_markdown`'s test-retest section**

```python
# OLD
            f"- per-requirement score stdev: mean={tr['requirement_score_stdev']['mean']:.2f} "
            f"(n={tr['requirement_score_stdev']['n_requirement_observations']} requirement observations)",
            f"- recommendation full agreement rate: {tr['recommendation_full_agreement_rate']:.1%}",
        ]

# NEW
            f"- per-requirement score stdev: mean={tr['requirement_score_stdev']['mean']:.2f} "
            f"(n={tr['requirement_score_stdev']['n_requirement_observations']} requirement observations)",
        ]
```

- [ ] **Step 8: Verify Steps 4-7 with a manual sanity check**

Run:
```bash
uv run python -c "
import sys, importlib.util
sys.path.insert(0, 'src')
spec = importlib.util.spec_from_file_location('analyze_study', 'scripts/analyze_jev_evaluation_study.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

tr = m.analyze_test_retest([
    {'jd_id': 'jd-1', 'candidate_id': 'cand-a', 'repeats': [
        {'composite_fit_score': 80.0, 'requirement_scores': {'Python': 80.0}, 'latency_seconds': 1.0},
        {'composite_fit_score': 82.0, 'requirement_scores': {'Python': 82.0}, 'latency_seconds': 1.1},
    ]},
])
assert 'recommendation_full_agreement_rate' not in tr
print('analyze_test_retest OK:', tr['n_pairs'])

abl = m.analyze_ablation([
    {'concrete_confidence': {'requirement::Python': 0.9}, 'vague_confidence': {'requirement::Python': 0.6}, 'vague_latency_seconds': 1.0},
])
assert 'unaffected_control' not in abl
print('analyze_ablation OK:', abl['n_pairs'])
"
```

Expected output:
```
analyze_test_retest OK: 1
analyze_ablation OK: 1
```
No errors, no `AssertionError`.

- [ ] **Step 9: Run the existing import smoke test to confirm nothing else broke**

Run: `uv run pytest tests/test_scripts_import.py -v`

Expected: both scripts still import cleanly (this test would not have caught Step 2's original bug, since it only exercises module-level import, but it's a cheap regression guard against a new import-time mistake introduced by this task's edits).

- [ ] **Step 10: Commit**

```bash
git add scripts/run_jev_evaluation_study.py scripts/analyze_jev_evaluation_study.py
git commit -m "$(cat <<'EOF'
fix: stop evaluation scripts reading removed Assessment fields

collect_test_retest AttributeError'd reading assessment.overall_
recommendation/meets_min_qualifications, removed from Assessment in
sub-project 1/5 -- fixed. Also removes the now-dead "unaffected
control" mechanism from the technical-requirement ablation (its
concrete-side data no longer exists in production assessments.json,
so it was silently validating nothing) and the recommendation-full-
agreement-rate computation from test-retest analysis, bringing this
ablation in line with how collect_seniority_education_ablation/
analyze_seniority_education_ablation already work (direct concrete-
vs-vague comparison, no control field). Sub-project 4/5.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix `console_export.py`

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py`
- Modify: `tests/output/test_console_export.py`

**Interfaces:**
- Consumes: `report.json`'s `test_retest` no longer has `recommendation_full_agreement_rate` (per Task 1) — this task's own edit doesn't strictly depend on Task 1 landing first (`.get()` already handles a missing key gracefully either way), but conceptually completes the same removal.
- Produces: `_evaluation_summary`'s returned dict — and therefore the `evaluationSummary` object in the exported `real-data.json` — no longer has `recommendationAgreementRate`. This is the shape Task 3 must assume.

- [ ] **Step 1: Update the test fixture and assertions in `tests/output/test_console_export.py`**

```python
# OLD
    report = {
        "run_id": run_id,
        "test_retest": {
            "n_pairs": 338, "n_repeats_per_pair": 3,
            "composite_fit_score_stdev": {"mean": 0.8, "median": 0.66},
            "recommendation_full_agreement_rate": 0.935,
        },
        "ranking_convergence": {
            "per_job_profile": {"jd-1": {"n_candidates": 22, "n_repeats": 3, "mean_kendall_tau": 0.93}},
            "mean_kendall_tau_across_all_profiles": 0.906,
        },
        "internal_coherence": {
            "n_assessments": 338,
            "requirement_vs_composite_score_correlation": {"spearman_rho": 0.667, "p_value": 6.7e-45, "n": 338},
        },
    }

# NEW
    report = {
        "run_id": run_id,
        "test_retest": {
            "n_pairs": 338, "n_repeats_per_pair": 3,
            "composite_fit_score_stdev": {"mean": 0.8, "median": 0.66},
        },
        "ranking_convergence": {
            "per_job_profile": {"jd-1": {"n_candidates": 22, "n_repeats": 3, "mean_kendall_tau": 0.93}},
            "mean_kendall_tau_across_all_profiles": 0.906,
        },
        "internal_coherence": {
            "n_assessments": 338,
            "requirement_vs_composite_score_correlation": {"spearman_rho": 0.667, "p_value": 6.7e-45, "n": 338},
        },
    }
```

```python
# OLD
    summary = data["evaluationSummary"]
    assert summary["recommendationAgreementRate"] == 0.935
    assert summary["meanRankingConvergence"] == 0.906
    assert summary["compositeScoreStdev"] == 0.8
    assert summary["coherenceSpearmanRho"] == 0.667

# NEW
    summary = data["evaluationSummary"]
    assert "recommendationAgreementRate" not in summary
    assert summary["meanRankingConvergence"] == 0.906
    assert summary["compositeScoreStdev"] == 0.8
    assert summary["coherenceSpearmanRho"] == 0.667
```

- [ ] **Step 2: Run the test and confirm it fails against the current implementation**

Run: `uv run pytest tests/output/test_console_export.py -v`

Expected: `test_export_console_web_data_includes_evaluation_summary_when_report_exists` fails on `assert "recommendationAgreementRate" not in summary` — the current `console_export.py` still populates that key (as `None`, since the fixture no longer has `recommendation_full_agreement_rate` for it to read, but the *key itself* is still present in the returned dict, which is what the new assertion checks for).

- [ ] **Step 3: Update `src/candidate_ranking/output/console_export.py`**

```python
# OLD
    return {
        "nPairs": test_retest.get("n_pairs"),
        "nRepeats": test_retest.get("n_repeats_per_pair"),
        "recommendationAgreementRate": test_retest.get("recommendation_full_agreement_rate"),
        "compositeScoreStdev": test_retest.get("composite_fit_score_stdev", {}).get("mean"),
        "meanRankingConvergence": ranking_convergence.get("mean_kendall_tau_across_all_profiles"),
        "coherenceSpearmanRho": coherence.get("spearman_rho") if coherence else None,
    }

# NEW
    return {
        "nPairs": test_retest.get("n_pairs"),
        "nRepeats": test_retest.get("n_repeats_per_pair"),
        "compositeScoreStdev": test_retest.get("composite_fit_score_stdev", {}).get("mean"),
        "meanRankingConvergence": ranking_convergence.get("mean_kendall_tau_across_all_profiles"),
        "coherenceSpearmanRho": coherence.get("spearman_rho") if coherence else None,
    }
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `uv run pytest tests/output/test_console_export.py -v`

Expected: all tests `PASS`.

- [ ] **Step 5: Run the full Python test suite**

Run: `uv run pytest`

Expected: all tests `PASS`.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/output/console_export.py tests/output/test_console_export.py
git commit -m "$(cat <<'EOF'
fix: stop console_export.py producing recommendationAgreementRate

Completes the removal from the evaluation-report side: the exported
evaluationSummary no longer has a recommendationAgreementRate key at
all (not just a null value). Sub-project 4/5; the frontend fix
(dropping the field/card entirely) is the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Fix `console-web/src/data.ts` and `console-web/src/views/Comparison.tsx`

**Files:**
- Modify: `console-web/src/data.ts`
- Modify: `console-web/src/views/Comparison.tsx`

**Interfaces:**
- Consumes: `console_export.py`'s `evaluationSummary` shape from Task 2 (no `recommendationAgreementRate`).
- Produces: nothing consumed by later tasks (last task in this plan).

- [ ] **Step 1: Update `console-web/src/data.ts`**

```typescript
// OLD
export interface EvaluationSummary {
  nPairs: number | null;
  nRepeats: number | null;
  recommendationAgreementRate: number | null;
  compositeScoreStdev: number | null;
  meanRankingConvergence: number | null;
  coherenceSpearmanRho: number | null;
}

// NEW
export interface EvaluationSummary {
  nPairs: number | null;
  nRepeats: number | null;
  compositeScoreStdev: number | null;
  meanRankingConvergence: number | null;
  coherenceSpearmanRho: number | null;
}
```

- [ ] **Step 2: Update `console-web/src/views/Comparison.tsx`**

```typescript
// OLD
          <div style={{ display: 'flex', gap: 12 }}>
            <StatCard
              label="Recommendation agreement (repeat calls)"
              value={EVALUATION_SUMMARY.recommendationAgreementRate !== null ? `${(EVALUATION_SUMMARY.recommendationAgreementRate * 100).toFixed(0)}%` : '—'}
            />
            <StatCard
              label="Mean ranking stability (τ)"
              value={EVALUATION_SUMMARY.meanRankingConvergence !== null ? EVALUATION_SUMMARY.meanRankingConvergence.toFixed(3) : '—'}
            />

// NEW
          <div style={{ display: 'flex', gap: 12 }}>
            <StatCard
              label="Mean ranking stability (τ)"
              value={EVALUATION_SUMMARY.meanRankingConvergence !== null ? EVALUATION_SUMMARY.meanRankingConvergence.toFixed(3) : '—'}
            />
```

No substitute metric is added — the "Reliability & validation" row keeps its remaining three cards (mean ranking stability, composite-score stdev, requirement/composite coherence), matching the "remove, don't replace" rule already applied throughout this removal effort.

- [ ] **Step 3: Verify**

Run: `grep -n "overall_recommendation\|meets_min_qualifications\|recommendationAgreementRate\|Recommendation agreement" console-web/src/data.ts console-web/src/views/Comparison.tsx`

Expected: no output (empty).

Run: `cd console-web && npx tsc -b --noEmit`

Expected: exits with no errors.

- [ ] **Step 4: Commit**

```bash
git add console-web/src/data.ts console-web/src/views/Comparison.tsx
git commit -m "$(cat <<'EOF'
refactor: remove recommendationAgreementRate from frontend

EvaluationSummary drops the field and Comparison.tsx drops the
"Recommendation agreement (repeat calls)" StatCard -- it would
otherwise have rendered a permanent "—" for a metric the system no
longer computes. Completes sub-project 4/5.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** `collect_test_retest` crash fix → Task 1 Step 2; `_vague_build_questions` simplification → Task 1 Step 1; "unaffected control" removal (both `analyze_ablation` and its `render_markdown` line) → Task 1 Steps 4-5; `recommendation_full_agreement_rate` removal (both `analyze_test_retest` and its `render_markdown` line) → Task 1 Steps 6-7; `console_export.py` → Task 2; `data.ts`/`Comparison.tsx` → Task 3. The spec's "out of scope" items (`collect_seniority_education_ablation`, `docs/paper2_jev.tex`, `assessment.py`/`models.py`) are each named in Global Constraints so no task drifts into them.
- **Placeholder scan:** none — every step has literal code, exact commands, and expected output.
- **Type/name consistency:** the per-repeat dict shape Task 1 Step 2 produces (`composite_fit_score`, `requirement_scores`, `latency_seconds`) matches exactly what Task 1 Step 8's sanity-check fixture uses and what `analyze_test_retest` (Step 6) and `analyze_score_decomposition_diagnostic` (unchanged, already Score-only) expect. `EvaluationSummary`'s field set in Task 3 Step 1 matches what Task 2's `_evaluation_summary` now returns.
