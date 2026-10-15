# Sliding Window Ranking — Design Spec

**Date:** 2026-10-15
**Status:** Draft — pending implementation

---

## 1. Goal

Replace the MC-KG tournament (pool > 20 path) with a sliding window ranking
approach that is cheaper, more reliable on a 14B local model, and produces
better ranking quality through chain-of-thought reasoning before the permutation
decision.

The pool ≤ 20 path (`rank_one_shot` × 3 + P-L) is unchanged.

---

## 2. Why Sliding Window Over MC-KG Tournament

| Dimension | MC-KG Tournament | Sliding Window |
|---|---|---|
| LLM calls (N=100) | ~35 | ~38 |
| Context per call | subset of 10, all at once | window of 10, structured |
| Position bias mitigation | shuffling per iteration | forward + backward pass |
| Convergence guarantee | Kendall-τ via P-L | 2-pass bubble convergence |
| Reasoning quality | post-hoc `overall_reasoning` | per-candidate CoT before ranking |
| Failure mode | discard iteration | keep original window order |

The MC-KG tournament is a statistically principled approach but its quality
is bounded by the LLM's listwise judgment. Sliding window pairs smaller
context windows with chain-of-thought reasoning before the ranking decision —
producing better quality per call on a 14B model.

---

## 3. Algorithm

### 3.1 Initial Sort (0 LLM calls)

Sort qualified candidates by:
1. Count of must-have evaluations with `verdict=MET` and `evidence_strength=strong` (descending)
2. `total_relevant_years` (descending) as tiebreaker

This is initialization only — not the final ranking. It ensures the sliding
window starts from a reasonable ordering rather than arbitrary input order,
reducing the number of passes needed for convergence.

### 3.2 Sliding Window Passes

```
Parameters: window_size=10, step=5, passes=2

Forward pass (left → right):
  windows: [0..9], [5..14], [10..19], ..., [90..99]

Backward pass (right → left):
  windows: [90..99], [85..94], [80..89], ..., [0..9]
```

Per window:
1. Extract the slice of candidates at those positions
2. Number them 1..window_size (positional tokens)
3. LLM ranks the window → permutation of 1..window_size
4. Reorder global list for those positions
5. On failure (invalid permutation): keep original window order, log warning

Non-overlapping windows within a pass can run in parallel.

### 3.3 Total Calls

For N candidates, window=10, step=5:
```
windows_per_pass = ceil((N - window_size) / step) + 1
total_calls = windows_per_pass × 2
```

| N | Calls |
|---|---|
| 25 | 10 |
| 50 | 20 |
| 100 | 38 |

---

## 4. Schema Change: Chain-of-Thought Before Ranking

### Current (reasoning after → post-hoc rationalization)
```json
{
  "ranking": [3, 1, 2],
  "overall_reasoning": "..."
}
```

### New (reasoning before → genuine chain-of-thought)
```json
{
  "candidate_evaluations": [
    {"id": 1, "key_strengths": "Strong FastAPI evidence, 4yr relevant exp", "key_gaps": "No cloud experience"},
    {"id": 2, "key_strengths": "Docker + Kubernetes depth", "key_gaps": "Python only 1yr demonstrated"},
    {"id": 3, "key_strengths": "Python 6yr, ascending trajectory", "key_gaps": "No nice-to-have coverage"}
  ],
  "ranking": [3, 1, 2]
}
```

`candidate_evaluations` forces the model to assess each candidate individually
before committing to an ordering. This reduces position bias (the model cannot
just anchor on position 1 and rank down) and produces more reliable permutations.

---

## 5. Prompt

```
You are a senior hiring panel ranking a group of pre-screened candidates for
a role. All candidates below have passed qualification screening.

For EACH candidate, first write a brief evaluation: their key strengths and
key gaps for THIS specific role. Then rank all candidates from strongest to
weakest fit.

RANKING CRITERIA (in priority order):
1. Relevance — experience directly matching daily responsibilities
2. Evidence quality — concrete achievements over vague responsibilities
3. Career trajectory — ascending scope and seniority
4. Recency — recent hands-on experience over older exposure
5. Nice-to-have coverage — breadth of preferred skills as tiebreaker

RULES:
- Candidates are numbered 1..N. `ranking` must contain every number exactly once.
- Rank ALL candidates. No ties allowed.
- Judge only on information provided.

ROLE: {jd_title}
JOB SUMMARY: {jd_summary}
CANDIDATES:
{candidates_text}
```

---

## 6. Evidence Rendering

Reuse `_render_evidence_for_ranking` from `llm_assessment.py` unchanged.
Each candidate block is already structured and uniform — suitable for window context.

---

## 7. Output Utilities

Sliding window produces one ordered list, not multiple complete rankings.
P-L aggregation is not applicable.

Utilities for the output formatter:
```python
{cid: float(N - i) for i, cid in enumerate(final_ordered_ids)}
```

Pool ≤ 20 path retains actual P-L utilities from `rank_one_shot`.

---

## 8. Files

```
CREATE:
  src/candidate_ranking/scoring/sliding_window_ranking.py
  tests/scoring/test_sliding_window_ranking.py

MODIFY:
  src/candidate_ranking/graphs/pipeline.py
    → replace run_tournament_for_jd_repeat call with rank_sliding_window
  src/candidate_ranking/config.py
    → add sliding_window_size: int = 10
    → add sliding_window_step: int = 5
  .env.example
    → add CANDIDATE_RANKING_SLIDING_WINDOW_SIZE
    → add CANDIDATE_RANKING_SLIDING_WINDOW_STEP
```

---

## 9. Key Design Decisions

**Window size 10, not 20.**
RankGPT paper validated window=20 on GPT-4. This system runs Qwen2.5-14B-Instruct
4-bit. Paper 1 validated subset_size=5 with 0 failures on this model. Window=10
is a conservative middle ground — larger than Paper 1's subset for efficiency,
smaller than RankGPT's window for reliability.

**Step = window/2 = 5.**
50% overlap ensures every candidate appears in at least 2 windows per pass,
reducing the chance that a good candidate is stuck in a hard window.

**2 passes only.**
Forward pass bubbles strong candidates upward. Backward pass corrects
over-ranked weak candidates. Empirically sufficient for pool ≤ 100 with a
decent initial ordering.

**No retry.**
On invalid permutation output, keep original window order and log a warning.
Retrying adds cost and was responsible for Paper 1's additional complexity.
The forward+backward structure provides natural error correction — a mis-ranked
window will be revisited and corrected in the next pass or adjacent window.

**Initial sort is initialization, not ranking.**
The initial sort by MET strong count + years is not the final answer — it just
gives the sliding window a reasonable starting point so 2 passes are sufficient.
HR users never see this intermediate state.
