from __future__ import annotations

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_ollama import ChatOllama

from candidate_ranking.config import RunConfig
from candidate_ranking.evaluation.evaluation import load_manifest
from candidate_ranking.ingestion.cv import build_name_extraction_chain, load_candidates, load_or_extract_candidate_name
from candidate_ranking.ingestion.jd import load_job_descriptions
from candidate_ranking.models import JobDescription

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = _PROJECT_ROOT / "console-web" / "src" / "data" / "real-data.json"


def _role_stub(jd: JobDescription, candidate_count: int) -> dict:
    return {"id": jd.id, "title": jd.title, "description": jd.raw_text, "candidateCount": candidate_count}


def _composite_fit_score(entry: dict) -> float:
    components = list(entry.get("requirement_scores", {}).values())
    for key in ("seniority_years_fit_score", "education_fit_score"):
        value = entry.get(key)
        if value is not None:
            components.append(value)
    return statistics.mean(components) if components else 0.0


def _null_comparison(jd_id: str) -> dict:
    return {"jdId": jd_id, "meanFitScore": None, "rankingStability": None}


def _comparison_from_assessments(jd_id: str, jd_assessments: dict[str, dict], ranking_stability: float | None) -> dict:
    if not jd_assessments:
        return _null_comparison(jd_id)
    fit_scores = [_composite_fit_score(a) for a in jd_assessments.values()]
    return {
        "jdId": jd_id,
        "meanFitScore": statistics.mean(fit_scores),
        "rankingStability": ranking_stability,
    }


def _load_evaluation_report(run_dir: Path) -> dict | None:
    path = run_dir / "evaluation" / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("console-web export: discarding unreadable evaluation report %s: %s", path, exc)
        return None


def _load_shortlisting_audit(run_dir: Path) -> dict | None:
    path = run_dir / "evaluation" / "shortlisting_audit.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("console-web export: discarding unreadable shortlisting audit %s: %s", path, exc)
        return None


def _load_test_retest(run_dir: Path) -> list[dict] | None:
    path = run_dir / "evaluation" / "test_retest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("console-web export: discarding unreadable test-retest data %s: %s", path, exc)
        return None


def _repeat_samples_by_row(records: list[dict] | None) -> dict[str, list[dict]]:
    if records is None:
        return {}
    result: dict[str, list[dict]] = {}
    for record in records:
        row_id = f"{record['candidate_id']}::{record['jd_id']}"
        result[row_id] = [
            {"compositeFitScore": repeat["composite_fit_score"]}
            for repeat in record["repeats"]
        ]
    return result


def _shortlisting_audit_by_jd(audit: dict | None) -> dict[str, dict]:
    if audit is None:
        return {}
    result = {}
    for jd in audit.get("per_job_profile", []):
        result[jd["jd_id"]] = {
            "nCandidates": jd["n_candidates"],
            "nFlagged": jd["n_flagged"],
            "flaggedPoolFraction": jd["flagged_pool_fraction"],
            "profileFlagged": jd["profile_flagged"],
            "flaggedCandidates": [
                {
                    "candidateId": c["candidate_id"],
                    "compositeFitScore": c["composite_fit_score"],
                    "nearZeroRequirements": c["near_zero_requirements"],
                }
                for c in jd["flagged_candidates"][:5]
            ],
        }
    return result


def _ranking_stability_by_jd(report: dict | None) -> dict[str, float]:
    if report is None:
        return {}
    per_job_profile = report.get("ranking_convergence", {}).get("per_job_profile", {})
    return {
        jd_id: stats["mean_kendall_tau"]
        for jd_id, stats in per_job_profile.items()
        if stats.get("mean_kendall_tau") is not None
    }


def _evaluation_summary(report: dict | None) -> dict | None:
    if report is None:
        return None
    test_retest = report.get("test_retest", {})
    ranking_convergence = report.get("ranking_convergence", {})
    coherence = report.get("internal_coherence", {}).get("requirement_vs_composite_score_correlation")
    return {
        "nPairs": test_retest.get("n_pairs"),
        "nRepeats": test_retest.get("n_repeats_per_pair"),
        "recommendationAgreementRate": test_retest.get("recommendation_full_agreement_rate"),
        "compositeScoreStdev": test_retest.get("composite_fit_score_stdev", {}).get("mean"),
        "meanRankingConvergence": ranking_convergence.get("mean_kendall_tau_across_all_profiles"),
        "coherenceSpearmanRho": coherence.get("spearman_rho") if coherence else None,
    }


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

    evaluation_report = _load_evaluation_report(run_dir)
    ranking_stability_by_jd = _ranking_stability_by_jd(evaluation_report)
    shortlisting_audit_by_jd = _shortlisting_audit_by_jd(_load_shortlisting_audit(run_dir))
    repeat_samples_by_row = _repeat_samples_by_row(_load_test_retest(run_dir))

    roles: list[dict] = []
    candidates: list[dict] = []
    assessments: dict[str, dict] = {}
    comparison_out: dict[str, dict] = {}

    for jd_id in jd_ids:
        jd = jds_by_id[jd_id]

        ranking = json.loads((run_dir / jd_id / "ranking.json").read_text(encoding="utf-8"))
        rank_by_cv_id = {row["candidate_id"]: row["rank"] for row in ranking["rankings"]}
        score_by_cv_id = {row["candidate_id"]: row["composite_fit_score"] for row in ranking["rankings"]}

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
                    "utility": score_by_cv_id.get(cv_id),
                }
            )

            assessments[row_id] = {
                "composite_fit_score": _composite_fit_score(assessment_entry),
                "requirement_scores": assessment_entry.get("requirement_scores", {}),
                "seniority_years_fit_score": assessment_entry.get("seniority_years_fit_score"),
                "education_fit_score": assessment_entry.get("education_fit_score"),
            }
            exported_count += 1

        roles.append(_role_stub(jd, candidate_count=exported_count))

        comparison_out[jd_id] = _comparison_from_assessments(
            jd_id, jd_assessments, ranking_stability_by_jd.get(jd_id)
        )

    unprocessed_jd_ids = sorted(set(jds_by_id) - set(jd_ids))
    for jd_id in unprocessed_jd_ids:
        jd = jds_by_id[jd_id]
        roles.append(_role_stub(jd, candidate_count=0))
        comparison_out[jd_id] = _null_comparison(jd_id)

    roles_by_id = {r["id"]: r for r in roles}
    roles = [roles_by_id[jd_id] for jd_id in jds_by_id if jd_id in roles_by_id]

    output = {
        "roles": roles,
        "candidates": candidates,
        "assessments": assessments,
        "comparison": comparison_out,
        "evaluationSummary": _evaluation_summary(evaluation_report),
        "shortlistingAudit": shortlisting_audit_by_jd,
        "repeatSamples": repeat_samples_by_row,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote %d roles, %d candidates to %s", len(roles), len(candidates), output_path)
    return output_path
