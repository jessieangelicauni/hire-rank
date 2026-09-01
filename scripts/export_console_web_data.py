"""Manual/backfill entry point for regenerating console-web's
real-data.json from an already-completed run, without re-running the
pipeline. The automatic path is candidate_ranking.cli.run() calling
candidate_ranking.output.console_export.export_console_web_data() directly
at the end of every run -- this script exists for pointing at an older run
or re-exporting on demand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from candidate_ranking.config import RunConfig, apply_env_overrides
from candidate_ranking.output.console_export import export_console_web_data

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run id under runs_dir to export, e.g. 20260821-094526")
    args = parser.parse_args()

    cfg = apply_env_overrides(RunConfig.full(PROJECT_ROOT))
    result = export_console_web_data(cfg, args.run_id)
    if result is None:
        sys.exit(
            f"No exportable data for run {args.run_id!r} -- either it has no assessments.json "
            "files, or every JD it covers has since "
            "been removed from job-description/. See log output above for details."
        )
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
