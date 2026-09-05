# Candidate Ranking

LLM-driven agentic candidate ranking research: given a folder of job
descriptions and a folder of CVs, the pipeline shortlists candidates by skill
overlap, has a local LLM assess each shortlisted candidate's strengths and
weaknesses against the job description (with a retry loop that catches and
corrects weaknesses contradicting the candidate's own extracted skills), and
ranks candidates per job with an LLM-judged, Plackett-Luce tournament.
Assessment quality is checked offline, after the fact, against Ragas'
Faithfulness metric (strengths) and a skill-negation contradiction check
(weaknesses) -- see [Evaluation harness](#evaluation-harness-ragas-faithfulness-check).

## How the pipeline works

Each run executes a [LangGraph](https://github.com/langchain-ai/langgraph)
graph (`src/candidate_ranking/graphs/pipeline.py`) with these stages, run
per job description (JD) and/or per candidate where noted:

1. **Ingestion** — load JD `.txt` files from `job-description/` and parse CV
   `.pdf` files from `cv/` (cached to `runs/_cache/cv.json` so PDFs aren't
   re-parsed on the next run).
2. **Skill extraction** — an LLM extracts a normalized skill list from each
   CV (cached to `runs/_cache/cv_skills.json`) and from each JD (cached to
   `runs/_cache/jd_skills.json`), then a sentence-transformer embeds every
   extracted candidate skill into an index for fast similarity search.
3. **Shortlisting** — per JD, candidates are shortlisted by cosine
   similarity between their extracted skills and the JD's extracted
   technical skills (`--skill-match-threshold`, `--min-skill-matches`). Only
   shortlisted candidates get assessed — this is what keeps the expensive
   LLM stages below from running against the whole CV corpus for every JD.
4. **Assessment generation** — per shortlisted (JD, candidate) pair, an LLM
   writes strengths and weaknesses grounded in the CV text, plus an
   `additional_skills` list for bare skill-list mentions the CV gives no
   sentence-level context for (cached per JD/candidate under
   `runs/_cache/assessments/`). Before accepting the result, each weakness
   is checked against the candidate's own extracted skill list for
   self-contradiction (e.g. claiming an absent skill the candidate's skill
   list actually has); if any are found, the LLM is asked to retry once with
   that feedback, and any weakness that still contradicts after the retry is
   dropped rather than kept.
5. **Tournament ranking** — per JD, run 1+ independent "stability repeats."
   Each repeat samples small candidate subsets, has an LLM listwise-rank
   each subset by strengths/weaknesses, fits Plackett-Luce utilities from
   those rankings (via `scipy.optimize`), and picks the next subset to
   sample using a Monte Carlo knowledge-gradient criterion — repeated until
   the iteration budget is exhausted or every unique subset has been seen.
   Each repeat is checkpointed to disk as it runs.
6. **Evaluation & output** — per JD, writes a final ranking
   (`ranking.md`/`ranking.json`), the raw per-repeat tournament traces
   (`repeats.json`), and within-repeat convergence metrics, averaged across
   repeats for reporting.
7. **Console export** — after every run, `console-web/src/data/real-data.json`
   is regenerated automatically so the console-web dashboard reflects the
   latest run.

The whole graph is checkpointed to `runs/_cache/checkpoints.db` (SQLite,
keyed by run id), so an interrupted run can be resumed from where it left
off instead of restarting from scratch.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally (or reachable at
  `CANDIDATE_RANKING_OLLAMA_BASE_URL`), with the ranking model pulled:

  ```bash
  ollama pull qwen2.5:14b-instruct-q4_K_M
  ```

  (Override the model with `CANDIDATE_RANKING_MODEL` — see
  [Configuration](#configuration) — if you want to use a different one.)

  If Ollama's default parallelism auto-detection ends up serializing
  concurrent requests on your GPU instead of running them in parallel,
  `scripts/start_ollama.sh` starts `ollama serve` with `OLLAMA_NUM_PARALLEL`
  forced to a benchmarked-safe value instead.
- A GPU is recommended (the LLM inference and skill-embedding model both
  benefit from one) but not required.
- Node.js 18+ and npm, only if you want to run the console-web dashboard.

## Setup

Install Python dependencies with `uv`:

```bash
uv sync
```

Copy the example environment file and adjust as needed (at minimum, the
`CANDIDATE_RANKING_*_DIR` paths should point at this repo's absolute path):

```bash
cp .env.example .env
```

## Running the pipeline

Populate `job-description/` with one `.txt` file per role and `cv/` with
one `.pdf` per candidate (both directories already contain a sample corpus
in this repo), make sure Ollama is running, then:

```bash
uv run python -m candidate_ranking.cli run
```

This prints a `Run ID` (a UTC timestamp, e.g. `20260826-010039`) and writes
everything to `runs/<run_id>/`. A full run over a real corpus can take
hours, since it does at least one LLM call per shortlisted candidate for
assessment, plus many more for tournament ranking — expect it to run for a
while in the background.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--run-id ID` | new timestamp | Resume a prior run from its checkpoint instead of starting fresh. Pass the `Run ID` printed by the run you want to continue. |
| `--skill-match-threshold FLOAT` | `0.8` | Cosine similarity at or above which a candidate's skill counts as matching a JD's technical skill. |
| `--min-skill-matches INT` | `5` | Number of a JD's technical skills a candidate must match to be shortlisted for assessment (auto-reduced to the JD's total technical-skill count if that's smaller). |

Most other pipeline parameters (tournament iterations, subset size, model
names, concurrency, etc.) are set via environment variables — see
[Configuration](#configuration).

### Resuming an interrupted run

If a run is killed or crashes partway through, resume it with the same
run id:

```bash
uv run python -m candidate_ranking.cli run --run-id 20260826-010039
```

This restores pipeline state from `runs/_cache/checkpoints.db` and
continues from the last completed stage. On resume, the original run's
`--skill-match-threshold`/`--min-skill-matches` (read back from that run's
`manifest.json`) are preserved even if different values are passed on the
command line.

### Watching a run in progress

The terminal only prints the run id and final summary path — everything
else (including warnings) is logged elsewhere:

- `runs/<run_id>/warnings.json` — every WARNING+ log record from the run,
  appended live.
- `runs/_cache/checkpoints.db` — LangGraph's checkpoint DB; new writes mean
  the graph is actively progressing through a stage.
- `runs/<run_id>/<jd_id>/tournament/repeat_<i>/state.json` — live
  tournament progress for a given JD/repeat, once that stage is reached.

## Output layout

```
runs/<run_id>/
  manifest.json          # config used for this run (model, thresholds, corpus ids, ...)
  summary.json            # succeeded/failed assessment counts, per-JD breakdown
  failures.json            # assessment generation failures, if any
  warnings.json            # full WARNING+ log for this run
  <jd_id>/
    assessments.json         # every candidate's strengths/weaknesses/additional_skills for this JD, keyed by candidate id
    ranking.md               # human-readable final ranking for this JD
    ranking.json              # same ranking, structured
    repeats.json               # every stability repeat's tournament trace
    tournament/repeat_<i>/state.json  # live/resumable per-repeat tournament state
  ragas_faithfulness_report.json  # only present after running the eval harness, see below

runs/_cache/                # cross-run caches (safe to keep between runs, keyed by content+model)
  cv.json                    # parsed CV text
  cv_skills.json               # extracted candidate skill lists
  jd_skills.json                # extracted JD technical-skill lists
  assessments/<jd_id>/<cv_id>.json  # generated assessments
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

Then open the printed local URL (typically `http://localhost:5174`).

### Re-exporting an older run manually

The export normally happens automatically at the end of `cli.py run`, but
you can regenerate it for any already-completed run without re-running the
pipeline:

```bash
uv run python scripts/export_console_web_data.py --run-id 20260826-010039
```

## Configuration

All of these are read from the environment (`.env` is loaded automatically
via `python-dotenv`); unset variables fall back to the defaults in
`RunConfig.full()` (`src/candidate_ranking/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `CANDIDATE_RANKING_MODEL` | `qwen2.5:14b-instruct-q4_K_M` | Ollama model used for all generation stages (skill extraction, assessment, ranking). |
| `CANDIDATE_RANKING_FAITHFULNESS_MODEL` | unset (falls back to `CANDIDATE_RANKING_MODEL`) | Judge model used only by `scripts/ragas_faithfulness_eval.py`. |
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
| `CANDIDATE_RANKING_TOURNAMENT_ITERATIONS` | `30` | Iteration budget per tournament repeat. Acts as a ceiling, not a fixed count, whenever `TARGET_APPEARANCES_PER_CANDIDATE` is set (the default) — see below. |
| `CANDIDATE_RANKING_TARGET_APPEARANCES_PER_CANDIDATE` | `8` | Scales the actual iteration count to each JD's shortlist size so every candidate is seen roughly this many times on average, clamped to `[TOURNAMENT_ITERATIONS_MIN, TOURNAMENT_ITERATIONS]`. Set to unset/empty to fall back to `TOURNAMENT_ITERATIONS` as a fixed count instead. |
| `CANDIDATE_RANKING_TOURNAMENT_ITERATIONS_MIN` | `30` | Floor on the scaled iteration count, so scaling only ever adds iterations for JDs that need them. |
| `CANDIDATE_RANKING_STABILITY_REPEATS` | `3` | Independent tournament repeats per JD, averaged together to reduce the noise a single repeat's point estimate would otherwise carry into the final ranking. This is noise reduction via averaging, not a stability *measurement* — nothing compares repeat-to-repeat rankings. `mean_kendall_tau` (see `evaluation/evaluation.py`) is the separate within-repeat convergence metric: how much a single repeat's utility ordering changes between consecutive iterations, then averaged across repeats for reporting. |
| `CANDIDATE_RANKING_TOURNAMENT_SUBSET_SIZE` | `5` | Candidates per sampled subset shown to the LLM ranker. |
| `CANDIDATE_RANKING_NUM_SUBSET_SAMPLES` | `30` | Candidate subsets sampled per knowledge-gradient selection step. |
| `CANDIDATE_RANKING_NUM_MC_DRAWS` | `50` | Monte Carlo draws used to score each candidate subset. |
| `CANDIDATE_RANKING_PL_PRIOR_VARIANCE` | `1.0` | Prior variance in the Plackett-Luce utility fit. |

For a fast local smoke test, set `CANDIDATE_RANKING_MAX_JDS=1` and
`CANDIDATE_RANKING_MAX_CANDIDATES=10` (or similar) before running.

## Running tests

```bash
uv run pytest
```

Tests mock out the LLM/embedding dependencies (no Ollama instance or GPU
required to run the test suite).

## Evaluation harness: Ragas Faithfulness check

A standalone, offline script checks assessment quality on data from an
already-completed run. It's read-only — it never touches the production
pipeline or modifies the run it evaluates, other than writing its own
report file. It runs two checks:

- **Strengths**, via [Ragas](https://docs.ragas.io/)' `Faithfulness` metric
  (an LLM judge scores how well each strength is entailed by the
  candidate's CV text).
- **Weaknesses**, via a separate non-LLM check (`find_weakness_contradictions`)
  that flags a weakness claiming the candidate lacks a JD skill they
  actually have listed in their own extracted skills — Ragas Faithfulness
  can only verify claims positively entailed by the context, and an absence
  claim ("lacks X") never is, so it's structurally unable to check
  weaknesses. This check is cheap (it reuses already-cached extracted
  skills, no LLM judge call) and runs by default; skip it with
  `--skip-weakness-check`.

Install the extra dependencies (not part of the default install):

```bash
uv sync --group eval
```

Run it against a completed run:

```bash
uv run --group eval python scripts/ragas_faithfulness_eval.py --run-id 20260826-010039
```

This writes `runs/<run-id>/ragas_faithfulness_report.json` with the mean
faithfulness score across strengths, a per-JD breakdown, low-scoring items
(score < 0.5) for manual review, and the weakness-contradiction list. Pass
`--sample-size N` to evaluate a stratified random sample (an equal number of
rows per JD, not proportional to JD size) instead of the full population (a
full corpus can be thousands of rows at one LLM call each), `--sample-seed
INT` to control that sample's RNG seed (default: `42`), and `--output PATH`
to write the report somewhere other than the default location.

## Project layout

```
src/candidate_ranking/
  cli.py                # entry point: `python -m candidate_ranking.cli run`
  config.py                # RunConfig + environment variable overrides
  models.py                 # shared Pydantic/TypedDict data model
  graphs/pipeline.py         # the LangGraph pipeline definition
  ingestion/                  # JD/CV loading and CV skill/name extraction
  scoring/                      # JD skill extraction, assessment generation
                                  # (with self-correction retry), skill matching
  ranking/                        # tournament ranking, Plackett-Luce fitting,
                                    # Monte Carlo knowledge-gradient subset selection
  evaluation/                      # convergence/stability metrics, run evaluation,
                                     # Ragas Faithfulness / weakness-contradiction checks
  output/                           # run/JD output writers, console-web export
scripts/
  export_console_web_data.py  # manual/backfill console-web re-export for an older run
  ragas_faithfulness_eval.py  # the evaluation harness, see above
  generate_evaluation_figures.py  # regenerates the paper's evaluation figures (docs/figures/)
  start_ollama.sh              # starts `ollama serve` with a benchmarked OLLAMA_NUM_PARALLEL
console-web/                # React dashboard for browsing a run's results
job-description/, cv/       # sample input corpus
runs/                        # run outputs and cross-run caches (generated)
tests/                        # pytest suite (mocked dependencies, no live services)
docs/                          # literature review and paper drafts (not tracked in git)
```
