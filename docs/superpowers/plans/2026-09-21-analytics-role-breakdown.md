# Analytics Role Breakdown & Multicall View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-role expandable breakdown to the console-web Analytics page, and a per-applicant "repeatability sample" (3 independent Jev calls) view to the Leaderboard detail card.

**Architecture:** Both features are additive, read-only views over data that already exists on disk. The Python export (`console_export.py`) gains one new join (over the existing `evaluation/test_retest.json`) and exports it as a new top-level field. The React app gains a typed accessor for that field, a pure aggregation function for the role breakdown (computed client-side from data already exported), and two new UI sections that render only when their data is non-empty.

**Tech Stack:** Python 3.12 (pytest), React 19 + TypeScript (Vite), no TS test runner in this project — TypeScript changes are verified with `npx tsc -b --noEmit` plus manual checks in the dev server.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-21-analytics-role-breakdown-design.md`.
- No new Jev API calls. `runs/20260921-085111/evaluation/test_retest.json` already exists on disk for the current production run and is the sole new data source.
- The repeatability sample is a *different* set of 3 calls than the ones behind the displayed `composite_fit_score` — the UI must say so explicitly, per the spec.
- Follow existing code style exactly: Python — `snake_case` internal, `camelCase` JSON keys, small `_load_*` / `_*_by_*` helper pairs matching `_load_shortlisting_audit` / `_shortlisting_audit_by_jd`. TypeScript — inline style objects using the existing `tokens.ts` constants, no CSS files, no new dependencies.
- Do not modify `.tex`/PDF files, do not run `pdflatex` (unrelated to this feature, carried over from project convention).

---

### Task 1: Export repeatability samples from `console_export.py`

**Files:**
- Modify: `src/candidate_ranking/output/console_export.py`
- Test: `tests/output/test_console_export.py`

**Interfaces:**
- Produces: `export_console_web_data(...)` output dict gains key `"repeatSamples": dict[str, list[dict]]`, where each row's list contains `{"compositeFitScore": float, "overallRecommendation": str, "meetsMinQualifications": bool}` and the dict key is the same `f"{cv_id}::{jd_id}"` row id used by `"assessments"`. Empty dict `{}` when `evaluation/test_retest.json` doesn't exist for the run.

- [ ] **Step 1: Write the failing tests**

Add to `tests/output/test_console_export.py` (append at end of file):

```python
def test_export_console_web_data_includes_repeat_samples_when_test_retest_exists(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    run_dir = cfg.runs_dir / run_id
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True)
    test_retest = [
        {
            "jd_id": "jd-1",
            "candidate_id": "cand-a",
            "repeats": [
                {
                    "composite_fit_score": 86.0, "overall_recommendation": "hire",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 95.0},
                    "latency_seconds": 1.1,
                },
                {
                    "composite_fit_score": 88.5, "overall_recommendation": "hire",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 100.0},
                    "latency_seconds": 1.0,
                },
                {
                    "composite_fit_score": 87.0, "overall_recommendation": "maybe",
                    "meets_min_qualifications": True, "requirement_scores": {"Python": 98.0},
                    "latency_seconds": 1.3,
                },
            ],
        }
    ]
    (eval_dir / "test_retest.json").write_text(json.dumps(test_retest), encoding="utf-8")
    output_path = tmp_path / "real-data.json"

    export_console_web_data(cfg, run_id, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    samples = data["repeatSamples"]["cand-a::jd-1"]
    assert len(samples) == 3
    assert samples[0] == {"compositeFitScore": 86.0, "overallRecommendation": "hire", "meetsMinQualifications": True}
    assert samples[2]["overallRecommendation"] == "maybe"


def test_export_console_web_data_repeat_samples_empty_when_no_test_retest(run_with_jev_ranking, tmp_path):
    cfg, run_id = run_with_jev_ranking
    output_path = tmp_path / "real-data.json"

    export_console_web_data(cfg, run_id, output_path=output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["repeatSamples"] == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/output/test_console_export.py -v -k repeat_samples`
Expected: both tests FAIL with `KeyError: 'repeatSamples'`

- [ ] **Step 3: Implement the export**

In `src/candidate_ranking/output/console_export.py`, add these two functions directly after `_load_shortlisting_audit` (after the function ending at line 74, before `_shortlisting_audit_by_jd`):

```python
def _load_test_retest(run_dir: Path) -> list[dict] | None:
    path = run_dir / "evaluation" / "test_retest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("console-web export: discarding unreadable test-retest data %s: %s", path, exc)
        return None


def _repeat_samples_by_row(records: list[dict] | None) -> dict[str, list[dict]]:
    if records is None:
        return {}
    result: dict[str, list[dict]] = {}
    for record in records:
        row_id = f"{record['candidate_id']}::{record['jd_id']}"
        result[row_id] = [
            {
                "compositeFitScore": repeat["composite_fit_score"],
                "overallRecommendation": repeat["overall_recommendation"],
                "meetsMinQualifications": repeat["meets_min_qualifications"],
            }
            for repeat in record["repeats"]
        ]
    return result
```

Then in `export_console_web_data`, find this line (currently right after loading the manifest/JD/candidate data, near the evaluation report loading):

```python
    evaluation_report = _load_evaluation_report(run_dir)
    ranking_stability_by_jd = _ranking_stability_by_jd(evaluation_report)
    shortlisting_audit_by_jd = _shortlisting_audit_by_jd(_load_shortlisting_audit(run_dir))
```

Replace it with:

```python
    evaluation_report = _load_evaluation_report(run_dir)
    ranking_stability_by_jd = _ranking_stability_by_jd(evaluation_report)
    shortlisting_audit_by_jd = _shortlisting_audit_by_jd(_load_shortlisting_audit(run_dir))
    repeat_samples_by_row = _repeat_samples_by_row(_load_test_retest(run_dir))
```

Finally, find the `output = {...}` dict near the end of the function:

```python
    output = {
        "roles": roles,
        "candidates": candidates,
        "assessments": assessments,
        "comparison": comparison_out,
        "evaluationSummary": _evaluation_summary(evaluation_report),
        "shortlistingAudit": shortlisting_audit_by_jd,
    }
```

Replace it with:

```python
    output = {
        "roles": roles,
        "candidates": candidates,
        "assessments": assessments,
        "comparison": comparison_out,
        "evaluationSummary": _evaluation_summary(evaluation_report),
        "shortlistingAudit": shortlisting_audit_by_jd,
        "repeatSamples": repeat_samples_by_row,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/output/test_console_export.py -v`
Expected: all tests PASS (the two new ones, plus the two pre-existing ones unaffected)

- [ ] **Step 5: Re-export the current production run so the frontend has real data to work against**

Run:
```bash
uv run python -c "
from pathlib import Path
from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data
cfg = apply_env_overrides(RunConfig.full(Path('.').resolve()))
p = export_console_web_data(cfg, '20260921-085111')
print('exported to', p)
"
```
Expected: prints `exported to .../console-web/src/data/real-data.json`

Then verify the new field landed:
```bash
python3 -c "
import json
d = json.load(open('console-web/src/data/real-data.json'))
print(len(d['repeatSamples']))
print(next(iter(d['repeatSamples'].items())))
"
```
Expected: prints `259` and one `(row_id, [3 dicts])` sample.

- [ ] **Step 6: Commit**

```bash
git add src/candidate_ranking/output/console_export.py tests/output/test_console_export.py console-web/src/data/real-data.json
git commit -m "feat: export per-applicant repeatability samples to console-web"
```

---

### Task 2: Add `repeatSamplesFor` to the TypeScript data layer

**Files:**
- Modify: `console-web/src/data.ts`

**Interfaces:**
- Consumes: `console-web/src/data/real-data.json` field `repeatSamples: Record<string, {compositeFitScore, overallRecommendation, meetsMinQualifications}[]>` (Task 1).
- Produces: `RepeatSample` type, `repeatSamplesFor(applicantRowId: string): RepeatSample[]` — used by Task 3.

- [ ] **Step 1: Add the type and accessor**

In `console-web/src/data.ts`, add this interface after the existing `Assessment` interface (after line 27, before `ComparisonRow`):

```typescript
export interface RepeatSample {
  compositeFitScore: number;
  overallRecommendation: Assessment['overall_recommendation'];
  meetsMinQualifications: boolean;
}
```

Update the `RealData` interface to add the new field:

```typescript
interface RealData {
  roles: Role[];
  candidates: Applicant[];
  assessments: Record<string, Assessment>;
  comparison: Record<string, ComparisonRow>;
  evaluationSummary: EvaluationSummary | null;
  repeatSamples: Record<string, RepeatSample[]>;
}
```

Add the export and helper function at the end of the file (after `assessmentFor`):

```typescript
export const REPEAT_SAMPLES: Record<string, RepeatSample[]> = data.repeatSamples ?? {};

export function repeatSamplesFor(applicantRowId: string): RepeatSample[] {
  return REPEAT_SAMPLES[applicantRowId] ?? [];
}
```

- [ ] **Step 2: Type-check**

Run: `cd console-web && npx tsc -b --noEmit`
Expected: no output (clean compile)

- [ ] **Step 3: Commit**

```bash
git add console-web/src/data.ts
git commit -m "feat: add repeatSamplesFor accessor to console-web data layer"
```

---

### Task 3: Render the repeatability sample in the Leaderboard detail card

**Files:**
- Modify: `console-web/src/views/Leaderboard.tsx`

**Interfaces:**
- Consumes: `repeatSamplesFor(applicantRowId: string): RepeatSample[]`, `RepeatSample` type (Task 2).

- [ ] **Step 1: Import the new accessor and type**

In `console-web/src/views/Leaderboard.tsx`, find the top import:

```typescript
import { assessmentFor, type Applicant, type Assessment, type Role } from '../data';
```

Replace it with:

```typescript
import { assessmentFor, repeatSamplesFor, type Applicant, type Assessment, type RepeatSample, type Role } from '../data';
```

- [ ] **Step 2: Render the new section in `DetailPanel`**

Find:

```typescript
      <ScoreSummary assessment={assessment} />
      <QualificationScores assessment={assessment} />
      <RequirementScores requirementScores={assessment.requirement_scores} />
    </div>
  );
}
```

Replace with:

```typescript
      <ScoreSummary assessment={assessment} />
      <QualificationScores assessment={assessment} />
      <RequirementScores requirementScores={assessment.requirement_scores} />
      <RepeatabilitySample samples={repeatSamplesFor(applicant.id)} />
    </div>
  );
}
```

- [ ] **Step 3: Add the `RepeatabilitySample` component**

Find the end of the `RequirementScores` function (the closing `}` after its returned JSX, currently the last function in the file). Add this new function immediately after it:

```typescript
function RepeatabilitySample({ samples }: { samples: RepeatSample[] }) {
  if (samples.length === 0) return null;

  return (
    <div style={{ background: colorSurface, border: `1px solid ${colorBorder}`, borderRadius: radius, boxShadow: shadowMicro, overflow: 'hidden', marginTop: 16 }}>
      <div style={{ fontSize: 18, fontWeight: 700, color: colorText, padding: '16px 20px 4px' }}>
        Repeatability sample (3 independent calls)
      </div>
      <div style={{ fontSize: 12, color: colorTextMuted, padding: '0 20px 12px', lineHeight: 1.5 }}>
        From a separate reliability study, not the exact calls behind the score above -- its average may differ slightly.
      </div>
      <div>
        {samples.map((sample, i) => {
          const color = sample.overallRecommendation === 'hire'
            ? colorAccent
            : sample.overallRecommendation === 'no' ? colorDanger : colorTextMuted;
          return (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'center', gap: 12, padding: '10px 20px', fontSize: 14, color: colorText,
                borderTop: `1px solid ${colorBorder}`,
              }}
            >
              <span style={{ flex: 1, minWidth: 0 }}>Call {i + 1}</span>
              <span style={{ fontWeight: 700, color: colorText }}>{sample.compositeFitScore.toFixed(1)}</span>
              <span style={{ flexShrink: 0, width: 60, textAlign: 'right', color, fontWeight: 600, fontSize: 13 }}>
                {RECOMMENDATION_LABEL[sample.overallRecommendation]}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

This reuses the module-scope `RECOMMENDATION_LABEL` constant and the `colorAccent`/`colorDanger`/`colorTextMuted`/`colorSurface`/`colorBorder`/`colorText`/`radius`/`shadowMicro` tokens already imported at the top of this file — no new imports needed beyond Step 1.

- [ ] **Step 4: Type-check**

Run: `cd console-web && npx tsc -b --noEmit`
Expected: no output (clean compile)

- [ ] **Step 5: Manual verification in the dev server**

Run: `cd console-web && npm run dev` (or confirm the already-running dev server at `http://localhost:5173` picked up the change via HMR)

In the browser: open any role's leaderboard, click into an applicant's detail card, and confirm a "Repeatability sample (3 independent calls)" section appears below "Requirement fit", showing 3 rows with a score and a recommendation badge each, and the muted caption sentence.

- [ ] **Step 6: Commit**

```bash
git add console-web/src/views/Leaderboard.tsx
git commit -m "feat: show 3-call repeatability sample in applicant detail card"
```

---

### Task 4: Add the pure `roleBreakdownFor` aggregation function

**Files:**
- Create: `console-web/src/lib/roleBreakdown.ts`

**Interfaces:**
- Consumes: `APPLICANTS: Applicant[]`, `assessmentFor(id: string): Assessment` (existing, from `../data`).
- Produces: `RoleBreakdown` type, `roleBreakdownFor(role: Role): RoleBreakdown` — used by Task 5.

- [ ] **Step 1: Write the file**

Create `console-web/src/lib/roleBreakdown.ts`:

```typescript
import { APPLICANTS, assessmentFor, type Assessment, type Role } from '../data';

export interface RoleBreakdown {
  seniorityMean: number | null;
  educationMean: number | null;
  recommendationCounts: { hire: number; maybe: number; no: number };
  requirementMeans: [string, number][];
}

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

export function roleBreakdownFor(role: Role): RoleBreakdown {
  const assessments: Assessment[] = APPLICANTS
    .filter((a) => a.roleId === role.id)
    .map((a) => assessmentFor(a.id));

  const seniorityValues = assessments
    .map((a) => a.seniority_years_fit_score)
    .filter((v): v is number => v !== null);
  const educationValues = assessments
    .map((a) => a.education_fit_score)
    .filter((v): v is number => v !== null);

  const recommendationCounts = { hire: 0, maybe: 0, no: 0 };
  for (const a of assessments) recommendationCounts[a.overall_recommendation] += 1;

  const requirementValues = new Map<string, number[]>();
  for (const a of assessments) {
    for (const [requirement, score] of Object.entries(a.requirement_scores)) {
      const list = requirementValues.get(requirement) ?? [];
      list.push(score);
      requirementValues.set(requirement, list);
    }
  }
  const requirementMeans: [string, number][] = Array.from(requirementValues.entries())
    .map(([requirement, values]): [string, number] => [requirement, mean(values) ?? 0])
    .sort(([, a], [, b]) => b - a);

  return {
    seniorityMean: mean(seniorityValues),
    educationMean: mean(educationValues),
    recommendationCounts,
    requirementMeans,
  };
}
```

- [ ] **Step 2: Type-check**

Run: `cd console-web && npx tsc -b --noEmit`
Expected: no output (clean compile)

- [ ] **Step 3: Commit**

```bash
git add console-web/src/lib/roleBreakdown.ts
git commit -m "feat: add roleBreakdownFor aggregation for console-web analytics"
```

---

### Task 5: Expandable per-role breakdown panel in Analytics

**Files:**
- Modify: `console-web/src/views/Comparison.tsx`

**Interfaces:**
- Consumes: `roleBreakdownFor(role: Role): RoleBreakdown` (Task 4).

- [ ] **Step 1: Update imports and add expand state**

Find the top of `console-web/src/views/Comparison.tsx`:

```typescript
import { COMPARISON, EVALUATION_SUMMARY, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { colorAccent, colorBorder, colorSurface, colorText, colorTextMuted, radius, radiusPill, shadowMicro } from '../tokens';
```

Replace with:

```typescript
import { Fragment, useState } from 'react';
import { COMPARISON, EVALUATION_SUMMARY, type Role } from '../data';
import { avatarColorOf, initialsOf } from '../lib/avatar';
import { roleBreakdownFor } from '../lib/roleBreakdown';
import {
  colorAccent, colorBorder, colorDanger, colorSurface, colorSurfaceMuted,
  colorText, colorTextMuted, colorTextSoft, radius, radiusPill, shadowMicro,
} from '../tokens';
```

Inside the `Comparison` component function, find:

```typescript
export default function Comparison({ roles }: Props) {
  const metrics = roles.map(metricFor);
```

Replace with:

```typescript
export default function Comparison({ roles }: Props) {
  const [expandedRoleId, setExpandedRoleId] = useState<string | null>(null);
  const metrics = roles.map(metricFor);
```

- [ ] **Step 2: Make each role row clickable and wrap in a `Fragment`**

Find:

```typescript
        {roles.map((role, i) => {
          const m = metricFor(role);
          return (
            <div
              key={role.id}
              className="row-hover"
              style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '14px 20px',
                borderTop: i === 0 ? undefined : `1px solid ${colorBorder}`,
              }}
            >
```

Replace with:

```typescript
        {roles.map((role, i) => {
          const m = metricFor(role);
          const isExpanded = expandedRoleId === role.id;
          return (
            <Fragment key={role.id}>
            <div
              className="row-hover"
              onClick={() => setExpandedRoleId(isExpanded ? null : role.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 14, padding: '14px 20px', cursor: 'pointer',
                borderTop: i === 0 ? undefined : `1px solid ${colorBorder}`,
              }}
            >
```

Then find the closing of that same row (end of the role-row map callback):

```typescript
                </span>
              </div>
            </div>
          );
        })}
```

Replace with:

```typescript
                </span>
              </div>
            </div>
            {isExpanded && <RoleBreakdownPanel role={role} />}
            </Fragment>
          );
        })}
```

- [ ] **Step 3: Add the `RoleBreakdownPanel` component**

Add this new function after the `Comparison` default-export function (at the end of the file):

```typescript
function RoleBreakdownPanel({ role }: { role: Role }) {
  const { seniorityMean, educationMean, recommendationCounts, requirementMeans } = roleBreakdownFor(role);
  const totalRecommendations = recommendationCounts.hire + recommendationCounts.maybe + recommendationCounts.no;

  return (
    <div style={{ padding: '16px 20px 20px', background: colorSurfaceMuted, borderTop: `1px solid ${colorBorder}` }}>
      {(seniorityMean !== null || educationMean !== null) && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
          {seniorityMean !== null && (
            <StatCard label="Avg. years-of-experience score" value={seniorityMean.toFixed(1)} />
          )}
          {educationMean !== null && (
            <StatCard label="Avg. education score" value={educationMean.toFixed(1)} />
          )}
        </div>
      )}

      {totalRecommendations > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colorText, marginBottom: 6 }}>Recommendation breakdown</div>
          <div style={{ display: 'flex', height: 8, borderRadius: radiusPill, overflow: 'hidden', marginBottom: 6 }}>
            <div style={{ width: `${(recommendationCounts.hire / totalRecommendations) * 100}%`, background: colorAccent }} />
            <div style={{ width: `${(recommendationCounts.maybe / totalRecommendations) * 100}%`, background: colorTextSoft }} />
            <div style={{ width: `${(recommendationCounts.no / totalRecommendations) * 100}%`, background: colorDanger }} />
          </div>
          <div style={{ display: 'flex', gap: 16, fontSize: 12, color: colorTextMuted }}>
            <span>Hire: {recommendationCounts.hire}</span>
            <span>Maybe: {recommendationCounts.maybe}</span>
            <span>No: {recommendationCounts.no}</span>
          </div>
        </div>
      )}

      {requirementMeans.length > 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: colorText, marginBottom: 6 }}>Avg. score per requirement</div>
          {requirementMeans.map(([requirement, score]) => (
            <div key={requirement} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '6px 0', fontSize: 13, color: colorText }}>
              <span style={{ flex: 1, minWidth: 0 }}>{requirement}</span>
              <div style={{ flexShrink: 0, width: 100, height: 6, borderRadius: radiusPill, background: colorBorder, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.max(0, Math.min(100, score))}%`, height: '100%', borderRadius: radiusPill,
                  background: score >= 50 ? colorAccent : colorDanger,
                }} />
              </div>
              <span style={{ flexShrink: 0, width: 32, textAlign: 'right', color: colorTextMuted, fontSize: 12 }}>{score.toFixed(0)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

This reuses the existing `StatCard` function already defined earlier in this same file — no new import needed for it.

- [ ] **Step 4: Type-check**

Run: `cd console-web && npx tsc -b --noEmit`
Expected: no output (clean compile)

- [ ] **Step 5: Manual verification in the dev server**

In the browser at `http://localhost:5173`, go to Analytics. Click a role row (e.g. "Cloud Engineer") and confirm:
- The row expands showing avg years-of-experience / education `StatCard`s (when applicable to that role),
- A 3-segment hire/maybe/no bar with counts,
- A list of that role's technical requirements with average-score bars, sorted strongest first.

Click the same row again and confirm it collapses. Click a different role and confirm only one panel is expanded at a time.

Spot-check correctness: pick one role's top requirement average from the panel, then open 2-3 of that role's applicant detail cards in the Leaderboard and confirm their individual scores for that same requirement are in the right ballpark of the displayed average.

- [ ] **Step 6: Commit**

```bash
git add console-web/src/views/Comparison.tsx
git commit -m "feat: add expandable per-role breakdown panel to Analytics"
```
