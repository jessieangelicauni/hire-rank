from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig
from candidate_ranking.ingestion.cv import build_name_extraction_chain, load_candidates, load_or_extract_candidate_name
from candidate_ranking.ingestion.jp import load_job_profiles
from candidate_ranking.models import JobProfile

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "console-web" / "src" / "data" / "real-data.json"


def _role_stub(jp: JobProfile, candidate_count: int) -> dict:
    return {"id": jp.id, "title": jp.title, "description": jp.raw_text, "candidateCount": candidate_count}


def _initials_for(name: str | None, cv_id: str) -> str:
    if not name:
        return cv_id[:2].upper()
    words = name.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return name[:2].upper()


def _load_jd_assessments(run_dir: Path, jp_id: str) -> dict[str, dict]:
    path = run_dir / jp_id / "assessments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def seed_console_web_roles(cfg: RunConfig, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        existing = {"roles": [], "candidates": [], "assessments": {}, "comparison": {}}

    existing_roles_by_id = {r["id"]: r for r in existing["roles"]}
    jps = load_job_profiles(cfg.jp_dir)
    if not jps:
        logger.warning(
            "seed_console_web_roles: no JPs found in %s; leaving %s unchanged.", cfg.jp_dir, output_path
        )
        return output_path

    roles = [existing_roles_by_id.get(jp.id, _role_stub(jp, candidate_count=0)) for jp in jps]

    comparison = dict(existing["comparison"])
    for role in roles:
        comparison.setdefault(role["id"], {"jpId": role["id"], "qualifiedCount": None, "totalAssessed": None})

    output = {**existing, "roles": roles, "comparison": comparison}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Seeded %d role(s) into %s", len(jps), output_path)
    return output_path


def export_console_web_data(
    cfg: RunConfig, run_id: str, output_path: Path = DEFAULT_OUTPUT_PATH
) -> Path | None:
    run_dir = cfg.runs_dir / run_id
    if not run_dir.is_dir():
        logger.warning(
            "console-web export: run directory %s does not exist; skipping console-web export.", run_dir,
        )
        return None

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model_name = manifest["ollama_model"]

    jps_by_id = {jp.id: jp for jp in load_job_profiles(cfg.jp_dir)}
    run_jp_ids = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and (p / "assessments.json").is_file())
    jp_ids = [jp_id for jp_id in run_jp_ids if jp_id in jps_by_id]

    skipped = sorted(set(run_jp_ids) - set(jp_ids))
    if skipped:
        logger.warning(
            "console-web export: excluding %d JD(s) with run data but no matching file in %s: %s",
            len(skipped), cfg.jp_dir, skipped,
        )
    if not jp_ids:
        logger.warning(
            "No exportable JDs for run %s (no assessment data, or all matching JDs have since been "
            "removed from %s); skipping console-web export.", run_id, cfg.jp_dir,
        )
        return None

    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    run_candidate_ids = sorted({cv_id for jp_id in jp_ids for cv_id in _load_jd_assessments(run_dir, jp_id)})
    target_candidate_ids = [cv_id for cv_id in run_candidate_ids if cv_id in candidates_by_id]

    skipped_candidates = sorted(set(run_candidate_ids) - set(target_candidate_ids))
    if skipped_candidates:
        logger.warning(
            "console-web export: excluding %d candidate(s) with run data but no matching file in %s: %s",
            len(skipped_candidates), cfg.cv_dir, skipped_candidates,
        )

    llm = ChatOllama(model=model_name, base_url=cfg.ollama_base_url, temperature=0, num_ctx=cfg.ollama_num_ctx)
    name_chain = build_name_extraction_chain(llm)
    name_cache_path = cfg.cache_dir / "candidate_names.json"

    def _extract_name(cv_id: str) -> tuple[str, str | None]:
        candidate = candidates_by_id[cv_id]
        return cv_id, load_or_extract_candidate_name(candidate, name_chain, model_name, name_cache_path)

    with ThreadPoolExecutor(max_workers=cfg.ollama_num_parallel) as executor:
        names_by_cv_id: dict[str, str | None] = dict(executor.map(_extract_name, target_candidate_ids))

    roles: list[dict] = []
    candidates: list[dict] = []
    assessments: dict[str, dict] = {}
    comparison_out: dict[str, dict] = {}

    for jp_id in jp_ids:
        jp = jps_by_id[jp_id]

        ranking = json.loads((run_dir / jp_id / "ranking.json").read_text(encoding="utf-8"))
        rank_by_cv_id = {row["candidate_id"]: row["rank"] for row in ranking["rankings"]}
        score_by_cv_id = {row["candidate_id"]: row.get("utility", 0.0) for row in ranking.get("rankings", [])}

        jd_assessments = {
            cv_id: entry for cv_id, entry in sorted(_load_jd_assessments(run_dir, jp_id).items())
            if cv_id in candidates_by_id
        }
        exported_count = 0
        for cv_id, assessment_entry in jd_assessments.items():
            row_id = f"{cv_id}::{jp_id}"
            name = names_by_cv_id[cv_id]
            candidates.append({
                "id": row_id,
                "cvId": cv_id,
                "roleId": jp_id,
                "name": name if name else cv_id.upper(),
                "initials": _initials_for(name, cv_id),
                "rank": rank_by_cv_id.get(cv_id),
                "utility": score_by_cv_id.get(cv_id),
            })
            assessments[row_id] = {
                "qualified": assessment_entry.get("qualified"),
                "total_relevant_years": assessment_entry.get("total_relevant_years", 0),
                "must_have_met": sum(
                    1 for ev in assessment_entry.get("must_have_evaluations", [])
                    if ev.get("verdict") == "MET"
                ),
                "nice_to_have_met": sum(
                    1 for ev in assessment_entry.get("nice_to_have_evaluations", [])
                    if ev.get("verdict") == "MET"
                ),
                "has_multinational_experience": (
                    assessment_entry.get("career_trajectory") or {}
                ).get("has_multinational_experience"),
            }
            exported_count += 1

        roles.append(_role_stub(jp, candidate_count=exported_count))
        comparison_out[jp_id] = {
            "jdId": jp_id,
            "qualifiedCount": sum(1 for a in jd_assessments.values() if a.get("qualified")),
            "totalAssessed": len(jd_assessments),
        }

    for jp_id in sorted(set(jps_by_id) - set(jp_ids)):
        roles.append(_role_stub(jps_by_id[jp_id], candidate_count=0))
        comparison_out[jp_id] = {"jdId": jp_id, "qualifiedCount": None, "totalAssessed": None}

    roles_by_id = {r["id"]: r for r in roles}
    output = {
        "roles": [roles_by_id[jp_id] for jp_id in jps_by_id if jp_id in roles_by_id],
        "candidates": candidates,
        "assessments": assessments,
        "comparison": comparison_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %d roles, %d candidates to %s", len(roles), len(candidates), output_path)
    return output_path
