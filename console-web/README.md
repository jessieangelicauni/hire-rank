# Console Web

Read-only React/Vite dashboard for the candidate-ranking pipeline. It reads
a single static export, `src/data/real-data.json` — written by
`output/console_export.py`'s `export_console_web_data` after a pipeline
run — and has no backend or build-time API calls of its own.

## Views

- **Dashboard** (`views/Dashboard.tsx`) — open roles, searchable by title.
- **Leaderboard** (`views/Leaderboard.tsx`) — ranked applicants for one
  role, with per-candidate assessment detail (criteria met, skills,
  per-requirement breakdown).
- **Comparison** / Analytics (`views/Comparison.tsx`) — per-role fit-score
  summary, expandable into per-requirement averages and the reliability
  suite's results (tiering validity, confidence calibration, ranking
  consistency, ranking repeatability) when `run_reliability_suite.py` has
  been run for that role.

Data shapes and accessors live in `src/data.ts`; design tokens (colors,
radii, shadows) are CSS variables defined in `src/index.css` and re-exported
from `src/tokens.ts`.

## Running it

```bash
npm install
npm run dev
```

Regenerate `src/data/real-data.json` by running the pipeline (see the
root `README.md`) — it's gitignored since it's a build output, not source.

## Scripts

```bash
npm run dev       # start the Vite dev server
npm run build     # type-check (tsc -b) and build
npm run lint      # oxlint
npm run preview   # preview the production build
```
