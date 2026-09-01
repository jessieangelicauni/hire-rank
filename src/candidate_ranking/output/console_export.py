from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig
from candidate_ranking.evaluation.evaluation import evaluate_run, load_manifest
from candidate_ranking.ingestion.cv import build_name_extraction_chain, load_candidates, load_or_extract_candidate_name
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import JobDescription

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "console-web" / "src" / "data" / "real-data.json"


def _role_stub(jd: JobDescription, candidate_count: int) -> dict:
    return {"id": jd.id, "title": jd.title, "description": jd.raw_text, "candidateCount": candidate_count}


def _null_comparison(jd_id: str) -> dict:
    return {"jdId": jd_id, "kendallTau": None, "deltaU": None, "faithfulness": None}


def _load_faithfulness_per_jd(run_dir: Path) -> dict[str, float]:
    path = run_dir / "ragas_faithfulness_report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data["per_jd"])
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning("console-web export: discarding unreadable Ragas report %s: %s", path, exc)
        return {}


def _initials_for(name: str | None, cv_id: str) -> str:
    if not name:
        return cv_id[:2].upper()
    words = name.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    return name[:2].upper()


def _load_jd_assessments(run_dir: Path, jd_id: str) -> dict[str, dict]:
    path = run_dir / jd_id / "assessments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def seed_console_web_roles(cfg: RunConfig, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        existing = {"roles": [], "candidates": [], "assessments": {}, "comparison": {}}

    existing_roles_by_id = {r["id"]: r for r in existing["roles"]}
    jds = load_job_descriptions(cfg.jd_dir)
    if not jds:
        logger.warning(
            "seed_console_web_roles: no JDs found in %s; leaving %s unchanged.", cfg.jd_dir, output_path
        )
        return output_path

    roles = [existing_roles_by_id.get(jd.id, _role_stub(jd, candidate_count=0)) for jd in jds]

    comparison = dict(existing["comparison"])
    for role in roles:
        comparison.setdefault(role["id"], _null_comparison(role["id"]))

    output = {**existing, "roles": roles, "comparison": comparison}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Seeded %d role(s) into %s", len(roles), output_path)
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

    manifest = load_manifest(run_dir)

    jds_by_id = {jd.id: jd for jd in load_job_descriptions(cfg.jd_dir)}
    run_jd_ids = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and (p / "assessments.json").is_file())
    jd_ids = [jd_id for jd_id in run_jd_ids if jd_id in jds_by_id]

    skipped = sorted(set(run_jd_ids) - set(jd_ids))
    if skipped:
        logger.warning(
            "console-web export: excluding %d JD(s) with run data but no matching file in %s: %s",
            len(skipped), cfg.jd_dir, skipped,
        )
    if not jd_ids:
        logger.warning(
            "No exportable JDs for run %s (no assessment data, or all matching JDs have since been "
            "removed from %s); skipping console-web export.", run_id, cfg.jd_dir,
        )
        return None

    model_name = manifest["ollama_model"]

    candidates_by_id = {c.id: c for c in load_candidates(cfg.cv_dir, cfg.cache_dir / "cv.json")}
    run_candidate_ids = sorted({cv_id for jd_id in jd_ids for cv_id in _load_jd_assessments(run_dir, jd_id)})
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

    jd_evaluations = {e.job_description_id: e for e in evaluate_run(cfg, run_id)}
    faithfulness_per_jd = _load_faithfulness_per_jd(run_dir)

    roles: list[dict] = []
    candidates: list[dict] = []
    assessments: dict[str, dict] = {}
    comparison_out: dict[str, dict] = {}

    for jd_id in jd_ids:
        jd = jds_by_id[jd_id]

        ranking = json.loads((run_dir / jd_id / "ranking.json").read_text(encoding="utf-8"))
        rank_by_cv_id = {row["candidate_id"]: row["rank"] for row in ranking["rankings"]}
        utility_by_cv_id = {row["candidate_id"]: row["utility"] for row in ranking["rankings"]}

        jd_assessments = {
            cv_id: entry for cv_id, entry in sorted(_load_jd_assessments(run_dir, jd_id).items())
            if cv_id in candidates_by_id
        }
        exported_count = 0
        for cv_id, assessment_entry in jd_assessments.items():
            row_id = f"{cv_id}::{jd_id}"
            name = names_by_cv_id[cv_id]

            candidates.append(
                {
                    "id": row_id,
                    "cvId": cv_id,
                    "roleId": jd_id,
                    "name": name if name else cv_id.upper(),
                    "initials": _initials_for(name, cv_id),
                    "rank": rank_by_cv_id.get(cv_id),
                    "utility": utility_by_cv_id.get(cv_id),
                }
            )

            assessments[row_id] = {
                "strengths": assessment_entry["strengths"],
                "weaknesses": assessment_entry["weaknesses"],
                "additional_skills": assessment_entry.get("additional_skills", []),
            }
            exported_count += 1

        roles.append(_role_stub(jd, candidate_count=exported_count))

        evaluation = jd_evaluations.get(jd_id)
        comparison_out[jd_id] = {
            "jdId": jd_id,
            "kendallTau": evaluation.mean_kendall_tau if evaluation else None,
            "deltaU": evaluation.mean_abs_delta_u if evaluation else None,
            "faithfulness": faithfulness_per_jd.get(jd_id),
        }

    unprocessed_jd_ids = sorted(set(jds_by_id) - set(jd_ids))
    for jd_id in unprocessed_jd_ids:
        jd = jds_by_id[jd_id]
        roles.append(_role_stub(jd, candidate_count=0))
        comparison_out[jd_id] = _null_comparison(jd_id)

    roles_by_id = {r["id"]: r for r in roles}
    roles = [roles_by_id[jd_id] for jd_id in jds_by_id if jd_id in roles_by_id]

    output = {"roles": roles, "candidates": candidates, "assessments": assessments, "comparison": comparison_out}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %d roles, %d candidates to %s", len(roles), len(candidates), output_path)
    return output_path
