from __future__ import annotations

from pathlib import Path

from candidate_ranking.models import JobProfile, slugify


def load_job_profiles(jp_dir: Path) -> list[JobProfile]:
    job_profiles: list[JobProfile] = []
    for path in sorted(jp_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8").strip()
        job_profiles.append(JobProfile(id=slugify(path.stem), raw_text=raw_text))
    return job_profiles
