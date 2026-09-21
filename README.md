# Candidate Ranking

LLM-driven agentic candidate ranking research: given a folder of job
descriptions and a folder of CVs, the pipeline shortlists candidates by skill
overlap, then assesses each shortlisted candidate against the job
description using Jev, TypeSafe AI's non-autoregressive "System One Model."
Jev answers a fixed set of typed questions (an overall fit score, a hiring
recommendation, a minimum-qualifications judgment, and one score per
extracted requirement) in a single parallel pass per call, with a
model-reported confidence on every answer. Because the output space is
predefined (Noul/Choice/Score), a schema violation is impossible by
construction rather than statistically reduced -- there is no free-text
generation to hallucinate. Three independent Jev calls per (job, candidate)
pair are aggregated (mean for numeric fields, confidence-weighted majority
vote for the recommendation, simple majority for binary judgments) and
candidates are ranked per job by sorting on the resulting overall fit score.
See `docs/paper2_jev.tex` for the full write-up, and
`docs/jev_vs_paper1_comparison.md` for how this compares to the prior
free-text + tournament system documented in `docs/paper1.tex`.

## How the pipeline works

Each run executes a [LangGraph](https://github.com/langchain-ai/langgraph)
graph (`src/candidate_ranking/graphs/pipeline.py`) with these stages, run
per job description (JD) and/or per candidate where noted:

1. **Ingestion** — load JD `.txt` files from `job-description/` and parse CV
   `.pdf` files from `cv/` (cached to `runs/_cache/cv.json` so PDFs aren't
   re-parsed on the next run).
2. **Skill extraction** — a local Ollama-hosted LLM extracts a normalized
   skill list from each CV (cached to `runs/_cache/cv_skills.json`) and from
   each JD, including certifications and any stated seniority/education
   requirement (cached to `runs/_cache/jd_skills.json`), then a
   sentence-transformer embeds every extracted candidate skill into an index
   for fast similarity search.
3. **Shortlisting** — per JD, candidates are shortlisted by cosine
   similarity between their extracted skills and the JD's extracted
   technical skills (`--skill-match-threshold`, `--min-skill-matches`, `--min-must-have-matches`). Only
   shortlisted candidates get assessed — this is what keeps the expensive
   LLM/Jev stages below from running against the whole CV corpus for every
   JD.
4. **Structured assessment via Jev** — per shortlisted (JD, candidate) pair,
   3 independent Jev calls are issued (each answering the recommendation,
   minimum-qualifications, per-requirement, and any
   certification/seniority/education questions in one parallel pass) and
   aggregated into a single assessment (cached per JD/candidate under
   `runs/_cache/assessments/`).
5. **Ranking** — per JD, candidates are ranked by sorting directly on the
   aggregated `composite_fit_score` (the mean of the per-requirement,
   seniority-years, and education sub-scores). There is no iterative
   tournament stage; Jev's structured output makes one removable.
6. **Console export** — after every run, `console-web/src/data/real-data.json`
   is regenerated automatically so the console-web dashboard reflects the
   latest run.

The whole graph is checkpointed to `runs/_cache/checkpoints.db` (SQLite,
keyed by run id), so an interrupted run can be resumed from where it left
off instead of restarting from scratch.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally (or reachable at
  `CANDIDATE_RANKING_OLLAMA_BASE_URL`), with the skill/JD-extraction model
  pulled:

  ```bash
  ollama pull qwen2.5:14b-instruct-q4_K_M
  ```

  (Override the model with `CANDIDATE_RANKING_MODEL` — see
  [Configuration](#configuration) — if you want to use a different one.)
  Ollama is only used for skill/JD extraction; the assessment stage calls
  TypeSafe AI's hosted Jev API instead.
- A TypeSafe AI API key for Jev, set as `CANDIDATE_RANKING_JEV_API_KEY` (see
  [Configuration](#configuration)) — the run fails fast if this is unset.
- A GPU is recommended for the skill-extraction LLM and the skill-embedding
  model, but not required.
- Node.js 18+ and npm, only if you want to run the console-web dashboard.

## Setup

Install Python dependencies with `uv`:

```bash
uv sync
```

Copy the example environment file and adjust as needed (at minimum, the
`CANDIDATE_RANKING_*_DIR` paths should point at this repo's absolute path,
and `CANDIDATE_RANKING_JEV_API_KEY` must be set):

```bash
cp .env.example .env
```

## Running the pipeline

Populate `job-description/` with one `.txt` file per role and `cv/` with
one `.pdf` per candidate (both directories already contain a sample corpus
in this repo), make sure Ollama is running and `CANDIDATE_RANKING_JEV_API_KEY`
is set, then:

```bash
uv run python -m candidate_ranking.cli run
```

This prints a `Run ID` (a UTC timestamp, e.g. `20260826-010039`) and writes
everything to `runs/<run_id>/`. A full run over a real corpus makes 3 Jev
calls per shortlisted candidate (plus one LLM call each for skill/JD
extraction) — expect it to take a while for a large corpus, though each Jev
call itself is fast (low seconds, see `docs/paper2_jev.tex` Section IV for
measured latency).

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--run-id ID` | new timestamp | Resume a prior run from its checkpoint instead of starting fresh. Pass the `Run ID` printed by the run you want to continue. |
| `--skill-match-threshold FLOAT` | `0.8` | Cosine similarity at or above which a candidate's skill counts as matching a JD's technical skill. |
| `--min-skill-matches INT` | `5` | Number of a JD's technical skills a candidate must match to be shortlisted for assessment (auto-reduced to the JD's total technical-skill count if that's smaller). |
| `--min-must-have-matches INT` | `2` | Number of a JD's must-have technical skills a candidate must match to be shortlisted for assessment (auto-reduced to the JD's total must-have-skill count if that's smaller; has no effect if the JD has no classified must-have skills). |

Most other pipeline parameters (model names, concurrency, etc.) are set via
environment variables — see [Configuration](#configuration).

### Resuming an interrupted run

If a run is killed or crashes partway through, resume it with the same
run id:

```bash
uv run python -m candidate_ranking.cli run --run-id 20260826-010039
```

This restores pipeline state from `runs/_cache/checkpoints.db` and
continues from the last completed stage. On resume, the original run's
`--skill-match-threshold`/`--min-skill-matches`/`--min-must-have-matches` (read back from that run's
`manifest.json`) are preserved even if different values are passed on the
command line.

### Watching a run in progress

The terminal only prints the run id and final summary path — everything
else (including warnings) is logged elsewhere:

- `runs/<run_id>/warnings.json` — every WARNING+ log record from the run,
  appended live.
- `runs/_cache/checkpoints.db` — LangGraph's checkpoint DB; new writes mean
  the graph is actively progressing through a stage.
- `runs/<run_id>/<jd_id>/assessments.json` — appears once that JD's
  shortlisted candidates start finishing assessment.

## Output layout

```
runs/<run_id>/
  manifest.json          # config used for this run (model, thresholds, corpus ids, ...)
  summary.json            # succeeded/failed assessment counts, per-JD breakdown
  failures.json            # assessment generation failures, if any
  warnings.json            # full WARNING+ log for this run
  <jd_id>/
    assessments.json         # every candidate's aggregated Jev assessment for this JD, keyed by candidate id
    ranking.md               # human-readable final ranking for this JD
    ranking.json              # same ranking, structured
  evaluation/                # only present after running the Jev evaluation study, see below

runs/_cache/                # cross-run caches (safe to keep between runs, keyed by content+model)
  cv.json                    # parsed CV text
  cv_skills.json               # extracted candidate skill lists
  jd_skills.json                # extracted JD technical-skill/certification/seniority/education lists
  assessments/<jd_id>/<cv_id>.json  # generated (aggregated) assessments
  checkpoints.db                # LangGraph resumability checkpoint
```

## Viewing results (console-web dashboard)

`console-web/` is a small React + Vite app that renders
`console-web/src/data/real-data.json` — the file `cli.py run` regenerates
automatically after every run.

```bash
cd console-web
npm install
npm run dev
```

Then open the printed local URL (typically `http://localhost:5173`).

## Configuration

All of these are read from the environment (`.env` is loaded automatically
via `python-dotenv`); unset variables fall back to the defaults in
`RunConfig.full()` (`src/candidate_ranking/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `CANDIDATE_RANKING_JEV_API_KEY` | unset (**required**) | TypeSafe AI API key for the Jev assessment stage. The run raises an error immediately if this is unset. |
| `CANDIDATE_RANKING_MODEL` | `qwen2.5:14b-instruct-q4_K_M` | Ollama model used for skill/JD extraction. |
| `CANDIDATE_RANKING_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `CANDIDATE_RANKING_OLLAMA_NUM_PARALLEL` | `4` | Max concurrent LLM calls (also caps LangGraph's `max_concurrency`). |
| `CANDIDATE_RANKING_OLLAMA_NUM_CTX` | `8192` | Context window requested from Ollama. |
| `CANDIDATE_RANKING_SKILL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for skill-based shortlisting. |
| `CANDIDATE_RANKING_JD_DIR` | `job-description/` | Directory of JD `.txt` files. |
| `CANDIDATE_RANKING_CV_DIR` | `cv/` | Directory of CV `.pdf` files. |
| `CANDIDATE_RANKING_CACHE_DIR` | `runs/_cache/` | Cross-run cache directory (must stay consistent between `run` invocations that should share caches). |
| `CANDIDATE_RANKING_RUNS_DIR` | `runs/` | Where per-run output directories are written. |
| `CANDIDATE_RANKING_MAX_JDS` | unset (all) | Cap the number of JDs processed — useful for a quick smoke run. |
| `CANDIDATE_RANKING_MAX_CANDIDATES` | unset (all) | Cap the number of candidates processed. |
| `CANDIDATE_RANKING_FAITHFULNESS_MODEL` | unset | Unused by the current Jev-based pipeline; retained only for the legacy Ragas evaluation code path in `evaluation/ragas_eval.py` (see [Project layout](#project-layout)). |

For a fast local smoke test, set `CANDIDATE_RANKING_MAX_JDS=1` and
`CANDIDATE_RANKING_MAX_CANDIDATES=10` (or similar) before running.

## Running tests

```bash
uv run pytest
```

Tests mock out the LLM/embedding/Jev dependencies (no Ollama instance, GPU,
or Jev API key required to run the test suite).

## Evaluation harness: Jev evaluation study

The numbers reported in `docs/paper2_jev.tex` come from a standalone,
read-only evaluation harness that runs against an already-completed run's
cached data (it never touches the production pipeline):

```bash
uv run python scripts/run_jev_evaluation_study.py --run-id 20260918-104531
uv run python scripts/analyze_jev_evaluation_study.py --run-id 20260918-104531
uv run python scripts/validate_multi_call_averaging.py --run-id 20260918-104531
```

- `run_jev_evaluation_study.py` collects raw data: test-retest reliability
  (3 fresh single-call repeats per shortlisted pair), the criteria-design
  ablation (concrete vs. bare-label Score criteria, `--ablation-sample-size`
  pairs, default 40), and, with `--noul-criteria-only`, the seniority/
  education Noul criteria ablation (concrete vs. circular). Output goes to
  `runs/<run_id>/evaluation/*.json`.
- `analyze_jev_evaluation_study.py` computes the actual statistics (Kendall-
  tau ranking convergence, Spearman internal coherence, Mann-Whitney U
  group separation, Wilcoxon signed-rank ablation effects) from that raw
  data and writes `runs/<run_id>/evaluation/report.{json,md}`.
- `validate_multi_call_averaging.py` collects two more independent 3-call
  trials on top of the test-retest data, then compares single-call ranking
  convergence against 3-call-averaged ranking convergence, writing
  `runs/<run_id>/evaluation/multi_call_averaging_validation.json`.

Each script requires `CANDIDATE_RANKING_JEV_API_KEY` except
`analyze_jev_evaluation_study.py`, which only reads already-collected data.

## Project layout

```
src/candidate_ranking/
  cli.py                # entry point: `python -m candidate_ranking.cli run`
  config.py                # RunConfig + environment variable overrides
  models.py                 # shared Pydantic/TypedDict data model
  graphs/pipeline.py         # the LangGraph pipeline definition
  ingestion/                  # JD/CV loading and CV skill/name extraction
  scoring/                      # JD skill extraction, skill matching, and Jev-based
                                  # structured assessment + multi-call aggregation
  ranking/                        # tournament ranking, Plackett-Luce fitting, Monte
                                    # Carlo knowledge-gradient subset selection --
                                    # from the prior free-text pipeline (docs/paper1.tex),
                                    # not used by the current Jev-based pipeline
  evaluation/                      # evaluation.py: legacy tournament convergence/
                                     # stability metrics, also unused by the current
                                     # pipeline; ragas_eval.py: legacy Faithfulness
                                     # check, likewise unused
  output/                           # run/JD output writers, console-web export
scripts/
  run_jev_evaluation_study.py       # collects test-retest + ablation raw data, see above
  analyze_jev_evaluation_study.py    # computes statistics from that raw data
  validate_multi_call_averaging.py    # single-call vs. 3-call-averaged ranking convergence
console-web/                # React dashboard for browsing a run's results
job-description/, cv/       # sample input corpus
runs/                        # run outputs and cross-run caches (generated)
tests/                        # pytest suite (mocked dependencies, no live services)
docs/                          # literature review, paper drafts, and comparison notes
```
