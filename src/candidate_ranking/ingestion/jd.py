from __future__ import annotations

from pathlib import Path

from candidate_ranking.models import JobDescription, slugify


def load_job_descriptions(jd_dir: Path) -> list[JobDescription]:
    job_descriptions: list[JobDescription] = []
    for path in sorted(jd_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8").strip()
        first_line = next(
            (line.strip() for line in raw_text.splitlines() if line.strip()),
            path.stem,
        )
        job_descriptions.append(
            JobDescription(
                id=slugify(path.stem),
                title=first_line,
                raw_text=raw_text,
                source_path=str(path),
            )
        )
    return job_descriptions
