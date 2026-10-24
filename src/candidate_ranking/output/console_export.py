from __future__ import annotations

import json
import logging
from pathlib import Path

from candidate_ranking.config import RunConfig
from candidate_ranking.ingestion.jp import load_job_profiles
from candidate_ranking.models import CandidateProfile, JobProfile

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "console-web" / "src" / "data" / "real-data.json"


def _role_stub(jp: JobProfile, candidate_count: int) -> dict:
    return {"id": jp.id, "title": jp.id, "description": jp.raw_text, "candidateCount": candidate_count}


def _initials_for(cv_id: str) -> str:
    return cv_id[:2].upper()


def _load_jd_assessments(run_dir: Path, jp_id: str) -> dict[str, dict]:
    path = run_dir / jp_id / "assessments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cv_profiles(cache_path: Path) -> dict[str, CandidateProfile]:
    if not cache_path.exists():
        return {}
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return {cv_id: CandidateProfile.model_validate(entry["profile"]) for cv_id, entry in data.items()}


def _load_reliability(run_dir: Path, jp_id: str) -> dict | None:
    path = run_dir / "reliability" / f"{jp_id}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})

    tiering = checks.get("tiering_validity")
    calibration = checks.get("confidence_calibration")
    consistency = checks.get("ranking_consistency")
    repeatability = checks.get("ranking_repeatability")

    return {
        "tieringValidity": (
            {"total": tiering["total"], "agreed": tiering["agreed"], "agreementRate": tiering.get("agreement_rate")}
            if tiering else None
        ),
        "confidenceCalibration": {"buckets": calibration["buckets"]} if calibration else None,
        "rankingConsistency": (
            {
                "totalPairs": consistency["total_pairs"],
                "flippedPairs": consistency["flipped_pairs"],
                "inconsistentRate": consistency["inconsistent_rate"],
                "cosmeticTotal": consistency.get("cosmetic_total", 0),
                "cosmeticFlipped": consistency.get("cosmetic_flipped", 0),
                "cosmeticInconsistentRate": consistency.get("cosmetic_inconsistent_rate", 0.0),
            }
            if consistency and "total_pairs" in consistency else None
        ),
        "rankingRepeatability": (
            {
                "nRepeatPairs": repeatability["n_repeat_pairs"],
                "meanKendallTau": repeatability.get("mean_kendall_tau"),
                "nCandidates": repeatability.get("n_candidates"),
                "nCandidatesInTieClusters": repeatability.get("n_candidates_in_tie_clusters"),
                "nJevCalls": repeatability.get("n_jev_calls"),
            }
            if repeatability else None
        ),
        "coverage": report.get("coverage"),
    }


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
        comparison.setdefault(role["id"], {"jdId": role["id"], "totalAssessed": None, "meanCriterionScore": None})

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

    profiles_by_cv_id = _load_cv_profiles(cfg.cache_dir / "cv_profile.json")

    roles: list[dict] = []
    candidates: list[dict] = []
    assessments: dict[str, dict] = {}
    comparison_out: dict[str, dict] = {}

    for jp_id in jp_ids:
        jp = jps_by_id[jp_id]

        ranking = json.loads((run_dir / jp_id / "ranking.json").read_text(encoding="utf-8"))
        rank_by_cv_id = {row["candidate_id"]: row["rank"] for row in ranking["rankings"]}
        utility_by_cv_id = {row["candidate_id"]: row.get("utility") for row in ranking["rankings"]}
        score_by_cv_id = {row["candidate_id"]: row.get("criterion_score") for row in ranking["rankings"]}
        breakdown_by_cv_id = {row["candidate_id"]: row.get("criterion_breakdown", []) for row in ranking["rankings"]}

        jd_assessments = _load_jd_assessments(run_dir, jp_id)
        exported_count = 0
        criterion_scores: list[float] = []

        for cv_id, assessment_entry in sorted(jd_assessments.items()):
            row_id = f"{cv_id}::{jp_id}"
            criterion_score = score_by_cv_id.get(cv_id)
            if criterion_score is not None:
                criterion_scores.append(criterion_score)

            profile = profiles_by_cv_id.get(cv_id)
            skills_grouped = (
                [g.model_dump() for g in profile.skills_grouped_by_context()] if profile is not None else []
            )

            candidates.append({
                "id": row_id,
                "cvId": cv_id,
                "roleId": jp_id,
                "name": cv_id.upper(),
                "initials": _initials_for(cv_id),
                "rank": rank_by_cv_id.get(cv_id),
                "utility": utility_by_cv_id.get(cv_id),
                "criterionScore": criterion_score,
            })
            evaluations = assessment_entry.get("evaluations", [])
            assessments[row_id] = {
                "criteriaMet": sum(1 for ev in evaluations if ev.get("evidence_strength") in ("strong", "partial")),
                "criteriaTotal": len(evaluations),
                "criterionScore": criterion_score if criterion_score is not None else 0.0,
                "breakdown": [
                    {
                        "requirement": item["requirement"],
                        "evidenceStrength": item["evidence_strength"],
                        "expectedStrength": item["expected_strength"],
                        "probabilities": item.get("probabilities"),
                    }
                    for item in breakdown_by_cv_id.get(cv_id, [])
                ],
                "skillsGrouped": skills_grouped,
            }
            exported_count += 1

        roles.append(_role_stub(jp, candidate_count=exported_count))
        comparison_out[jp_id] = {
            "jdId": jp_id,
            "totalAssessed": len(jd_assessments),
            "meanCriterionScore": (
                sum(criterion_scores) / len(criterion_scores) if criterion_scores else None
            ),
        }

    for jp_id in sorted(set(jps_by_id) - set(jp_ids)):
        roles.append(_role_stub(jps_by_id[jp_id], candidate_count=0))
        comparison_out[jp_id] = {"jdId": jp_id, "totalAssessed": None, "meanCriterionScore": None}

    reliability_out = {
        jp_id: reliability
        for jp_id in jps_by_id
        if (reliability := _load_reliability(run_dir, jp_id)) is not None
    }

    roles_by_id = {r["id"]: r for r in roles}
    output = {
        "roles": [roles_by_id[jp_id] for jp_id in jps_by_id if jp_id in roles_by_id],
        "candidates": candidates,
        "assessments": assessments,
        "comparison": comparison_out,
        "reliability": reliability_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %d roles, %d candidates to %s", len(roles), len(candidates), output_path)
    return output_path
