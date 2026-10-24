# Candidate Ranking

LLM-driven candidate screening and ranking: given a folder of job profiles
and a folder of CVs, the pipeline extracts a technical-skills rubric per
job profile, filters candidates through a local, zero-API-cost shortlist
gate, then assesses every surviving candidate against that job using Jev
(TypeSafe AI's structured-output model). Jev answers one per-criterion
evidence-strength question (strong/partial/none) per rubric item, in a
single parallel call, returning a full probability distribution on every
answer — no free-text generation, so no hallucination risk from the output
format itself.

Ranking is primarily a transparent `criterion_score`, computed locally from
that probability distribution and auditable back to a named requirement.
Jev's more expensive holistic pairwise judgment ("which candidate is the
stronger fit") only runs for candidates whose transparent score is too
close to call, and its influence on the final order is bounded so it can
reorder within that near-tie cluster but never override the transparent
score across it. See `docs/PRD.md` for the full architecture history and
current status.

## How the pipeline works

Each run executes a [LangGraph](https://github.com/langchain-ai/langgraph)
graph (`src/candidate_ranking/graphs/pipeline.py`):

1. **Ingestion** — load JP `.txt` files from `job-profile/`, parse CV
   `.pdf` files from `cv/`, and extract each candidate's skills profile
   (all cached under `runs/_cache/`).
2. **Rubric extraction** — a local Ollama-hosted LLM turns each JP's raw
   posting into an `evaluation_criteria` rubric plus a deduplicated
   `technical_skills` list. No must-have/nice-to-have split.
3. **Local shortlist gate** (`scoring/shortlist.py`) — before any Jev call,
   each candidate is checked against the job's `technical_skills` (verbatim
   match, alias table, or embedding similarity via `sentence-transformers`).
   Needs `min(CANDIDATE_RANKING_SHORTLIST_MIN_SKILLS, job's skill count)`
   matches to proceed — pure local computation, no Jev cost.
4. **Structured assessment via Jev** — one Jev call per (JP, candidate)
   pair that clears the gate, producing one assessment per criterion with
   its evidence-strength label and probability distribution.
5. **Ranking** (`scoring/active_pairwise_ranking.py`) — `criterion_score`
   per candidate from probability-weighted expected evidence strength
   (`scoring/criterion_score.py`). Candidates within a tie-tolerance band
   form a tie cluster; only inside a cluster does an active-learning
   pairwise tournament ask Jev for the most-uncertain comparison, refitting
   a Bayesian Plackett-Luce utility model after each one.

The graph is checkpointed to `runs/_cache/checkpoints.db`, so an
interrupted run resumes instead of restarting.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally (or reachable at
  `CANDIDATE_RANKING_OLLAMA_BASE_URL`), with the rubric model pulled:

  ```bash
  ollama pull qwen3:14b
  ```

  Ollama is only used for rubric generation; assessment and ranking call
  TypeSafe AI's hosted Jev API, and the shortlist gate's embeddings run
  locally via `sentence-transformers` (no external call).
- A TypeSafe AI API key for Jev, set as `CANDIDATE_RANKING_JEV_API_KEY` —
  the run fails fast if unset.
- A GPU is recommended for the rubric LLM and shortlist embeddings, but not
  required.

## Setup

```bash
uv sync
cp .env.example .env   # all variables optional, see Configuration
```

## Running the pipeline

Populate `job-profile/` with one `.txt` per role and `cv/` with one `.pdf`
per candidate (sample corpus already included), then:

```bash
uv run python -m candidate_ranking.cli run
```

Prints a `Run ID` (UTC timestamp, e.g. `20260826-010039`) and writes to
`runs/<run_id>/`. Cap scope for a smoke test with `CANDIDATE_RANKING_MAX_JPS`
/ `CANDIDATE_RANKING_MAX_CANDIDATES`. Resume an interrupted run with:

```bash
uv run python -m candidate_ranking.cli run --run-id 20260826-010039
```

## Output layout

```
runs/<run_id>/<jp_id>/
  assessments.json       # every candidate's Jev assessment, keyed by candidate id
  ranking.md             # human-readable final ranking
  ranking.json           # structured ranking: rank, utility, criterion_score,
                          # per-criterion breakdown
  reliability/<jp_id>.json  # written by scripts/run_reliability_suite.py, if run

runs/_cache/              # cross-run caches, keyed by content+model
  cv.json                 # parsed CV text
  cv_profile.json         # extracted candidate skills profiles
  job_profile.json        # per-JP rubric + technical_skills
  assessments/<jp_id>/<cv_id>.json
  checkpoints.db          # LangGraph resumability checkpoint
```

## Configuration

Read from the environment (`.env` loaded automatically); unset variables
fall back to `RunConfig.full()` (`src/candidate_ranking/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `CANDIDATE_RANKING_JEV_API_KEY` | unset (**required**) | TypeSafe AI API key for Jev. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | unset (optional) | Jev call cost tracing via Langfuse Cloud. Tracing only turns on when `LANGFUSE_PUBLIC_KEY` is set. |
| `CANDIDATE_RANKING_MODEL` | `qwen3:14b` | Ollama model for rubric generation. |
| `CANDIDATE_RANKING_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL` | `4` | Max concurrent LLM calls. |
| `CANDIDATE_RANKING_OLLAMA_NUM_CTX` | `16384` | Context window requested from Ollama. |
| `CANDIDATE_RANKING_JP_DIR` | `job-profile/` | JP `.txt` directory. |
| `CANDIDATE_RANKING_CV_DIR` | `cv/` | CV `.pdf` directory. |
| `CANDIDATE_RANKING_CACHE_DIR` | `runs/_cache/` | Cross-run cache directory. |
| `CANDIDATE_RANKING_RUNS_DIR` | `runs/` | Per-run output directory. |
| `CANDIDATE_RANKING_MAX_JPS` | unset (all) | Cap JPs processed. |
| `CANDIDATE_RANKING_MAX_CANDIDATES` | unset (all) | Cap candidates processed. |
| `CANDIDATE_RANKING_SHORTLIST_MIN_SKILLS` | `5` | Minimum matched skills to clear the shortlist gate. `0` disables the gate. |
| `CANDIDATE_RANKING_TIE_EPSILON` | unset (auto) | Tie-cluster tolerance band. |
| `CANDIDATE_RANKING_ENSEMBLE_REPEATS` | `1` | Independent Jev calls per within-cluster comparison. |
| `CANDIDATE_RANKING_RANKING_PL_PRIOR_VARIANCE` | `1.0` | Prior variance for the Plackett-Luce model. |
| `CANDIDATE_RANKING_MIN_DELTA_U` | `0.01` | Convergence threshold that stops the tournament early. |

## Running tests

```bash
uv run pytest
```

Tests mock the LLM/Jev/embedding dependencies — no Ollama, GPU, Jev API
key, or network access required.

## Scripts

All require a prior `run` so `runs/_cache/assessments/<jp_id>/*.json`
exists; all call the live Jev API (no cache) unless noted.

| Script | What it does |
|---|---|
| `run_shortlist_only.py` | Previews shortlist gate pass rates — pure local computation, zero Jev calls. |
| `run_reliability_suite.py --jp-id <id> [--full]` | Runs the 4 checks below concurrently for one JP and merges them into `runs/<run_id>/reliability/<jp_id>.json`, consumed by the console dashboard's Reliability panel. |
| `check_ranking_consistency.py` | Test-retest + cosmetic (evidence-order) invariance of pairwise comparisons. |
| `check_tiering_validity.py` | Does Jev's pairwise judgment agree with `criterion_score` on pairs clearly outside any tie cluster? |
| `check_confidence_calibration.py` | Does higher reported confidence correlate with higher self-agreement across repeats? |
| `check_ranking_repeatability.py` | Does the full ranking tournament reproduce the same order across independent runs (Kendall's tau)? |
| `reextract_evidence.py` | Recomputes the evidence snippet on cached assessments after a fix to evidence extraction, without re-calling Jev. |

## Console dashboard

`console-web/` is a React/Vite app that reads a static export
(`console-web/src/data/real-data.json`) and shows roles, ranked applicants,
per-criterion assessment detail, and the reliability suite's results.
`output/console_export.py`'s `export_console_web_data` writes that export
after a pipeline run. To view it locally:

```bash
cd console-web && npm install && npm run dev
```

## Project layout

```
src/candidate_ranking/
  cli.py                     # entry point: `run`
  config.py                  # RunConfig + environment variable overrides
  models.py                  # shared Pydantic data model
  graphs/pipeline.py         # the LangGraph pipeline definition
  ingestion/                  # JP/CV loading (jp.py) and CV PDF parsing (cv.py)
  scoring/
    job_profile.py             # rubric (evaluation-criteria) generation
    shortlist.py                # local, zero-Jev-cost shortlist gate
    cv_profile.py                # candidate skills profile extraction
    assessment.py                 # Jev-based structured assessment
    criterion_score.py             # probability-weighted transparent scoring
    active_pairwise_ranking.py       # tie-cluster formation + active pairwise tournament
    jev_client.py                     # Jev API client
    render.py                          # evidence rendering for ranking prompts
  ranking/plackett_luce.py    # Bayesian Plackett-Luce fitting + Laplace covariance
  output/
    writer.py                  # per-run/per-JP output writers
    console_export.py           # exports console-web/src/data/real-data.json
console-web/                 # React dashboard that reads that export
scripts/                      # see Scripts above
job-profile/, cv/            # sample input corpus
runs/                        # run outputs and cross-run caches (generated)
tests/                       # pytest suite (mocked dependencies, no live services)
docs/                        # PRD + literature-review reference PDFs
```
