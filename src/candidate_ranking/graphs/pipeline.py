from __future__ import annotations

import logging
import operator
from typing import Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from candidate_ranking.scoring.assessment import AssessmentGenerationError, load_or_generate_assessment
from candidate_ranking.config import RunConfig
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, load_or_generate_jd_skills
from candidate_ranking.evaluation.evaluation import build_ensemble_tournament_result, write_jd_repeats
from candidate_ranking.output.formatter import write_jd_ranking
from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription, TournamentResult
from candidate_ranking.scoring.skills import shortlist_candidates
from candidate_ranking.ranking.tournament import run_tournament_for_jd_repeat

logger = logging.getLogger(__name__)


class AssessmentResult(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None
    retry_audit: dict | None


class PipelineState(TypedDict):
    jds: list[JobDescription]
    candidates: list[Candidate]
    jd_skills: Annotated[dict[str, JDSkills], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]
    tournament_results: Annotated[list[TournamentResult], operator.add]


def build_pipeline_graph(
    cfg: RunConfig,
    jd_skills_chain: Runnable,
    assessment_chain: Runnable,
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    build_ranking_chain: Callable[[type[BaseModel]], Runnable],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
) -> StateGraph:
    jd_skills_cache_path = cfg.cache_dir / "jd_skills.json"

    def _noop(state: PipelineState) -> dict:
        return {}

    def fanout_skill_extraction(state: PipelineState) -> list[Send]:
        return [Send("extract_skills_for_jd", {"jd": jd}) for jd in state["jds"]]

    def extract_skills_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            jd_skills = load_or_generate_jd_skills(jd, jd_skills_chain, cfg.ollama_model, jd_skills_cache_path)
        except JDSkillsGenerationError as exc:
            logger.warning("Excluding %s from this run: %s", jd.id, exc)
            return {}
        return {"jd_skills": {jd.id: jd_skills}}

    def fanout_shortlist(state: PipelineState) -> list[Send]:
        return [
            Send("build_shortlist_for_jd", {"jd": jd, "jd_skills": state["jd_skills"][jd.id]})
            for jd in state["jds"]
            if jd.id in state["jd_skills"]
        ]

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        jd_skills = payload["jd_skills"]
        candidate_ids = shortlist_candidates(
            jd,
            jd_skills,
            skill_index,
            skill_row_map,
            skill_embedder,
            threshold=skill_match_threshold,
            min_matches=min_skill_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {"jd": jd, "candidate": candidates_by_id[candidate_id], "jd_skills": state["jd_skills"].get(jd.id)},
            )
            for jd in state["jds"]
            for candidate_id in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd = payload["jd"]
        candidate = payload["candidate"]
        jd_skills = payload.get("jd_skills")
        try:
            assessment, retry_audit = load_or_generate_assessment(
                jd, candidate, assessment_chain, cfg.ollama_model, cfg.cache_dir,
                jd_skills=jd_skills, skill_embedder=skill_embedder,
            )
            result: AssessmentResult = {
                "jd_id": jd.id,
                "candidate_id": candidate.id,
                "status": "ok",
                "assessment": assessment,
                "error": None,
                "retry_audit": retry_audit,
            }
        except AssessmentGenerationError as exc:
            result = {
                "jd_id": jd.id,
                "candidate_id": candidate.id,
                "status": "failed",
                "assessment": None,
                "error": str(exc),
                "retry_audit": None,
            }
        return {"assessment_results": [result]}

    def fanout_tournament(state: PipelineState) -> list[Send]:
        assessments_by_jd: dict[str, dict[str, Assessment]] = {}
        for r in state["assessment_results"]:
            if r["status"] != "ok":
                continue
            assessments_by_jd.setdefault(r["jd_id"], {})[r["candidate_id"]] = r["assessment"]

        sends = []
        for jd in state["jds"]:
            jd_assessments = assessments_by_jd.get(jd.id, {})
            shortlisted_ids = [cid for cid in state["shortlists"].get(jd.id, []) if cid in jd_assessments]
            if not shortlisted_ids:
                continue
            for repeat_index in range(cfg.stability_repeats):
                sends.append(
                    Send(
                        "run_tournament_for_jd_repeat_node",
                        {
                            "jd": jd,
                            "assessments": {cid: jd_assessments[cid] for cid in shortlisted_ids},
                            "repeat_index": repeat_index,
                        },
                    )
                )
        return sends

    def run_tournament_for_jd_repeat_node(payload: dict) -> dict:
        jd = payload["jd"]
        repeat_index = payload["repeat_index"]
        checkpoint_path = cfg.runs_dir / run_id / jd.id / "tournament" / f"repeat_{repeat_index}" / "state.json"
        rng_seed = abs(hash((run_id, jd.id, repeat_index))) % (2**32)
        result = run_tournament_for_jd_repeat(
            jd,
            payload["assessments"],
            repeat_index,
            cfg.tournament_iterations,
            cfg.tournament_subset_size,
            cfg.num_subset_samples,
            cfg.num_mc_draws,
            cfg.pl_prior_variance,
            build_ranking_chain,
            checkpoint_path,
            rng_seed,
            target_appearances_per_candidate=cfg.target_appearances_per_candidate,
            tournament_iterations_min=cfg.tournament_iterations_min,
        )
        return {"tournament_results": [result]}

    def fanout_finalize(state: PipelineState) -> list[Send]:
        assessments_by_jd: dict[str, dict[str, Assessment]] = {}
        for r in state["assessment_results"]:
            if r["status"] == "ok":
                assessments_by_jd.setdefault(r["jd_id"], {})[r["candidate_id"]] = r["assessment"]

        results_by_jd: dict[str, list[TournamentResult]] = {}
        for tr in state["tournament_results"]:
            results_by_jd.setdefault(tr.job_description_id, []).append(tr)

        sends: list[Send] = []

        for jd_id, results in results_by_jd.items():
            sends.append(Send("write_jd_repeats_node", {"jd_id": jd_id, "results": results}))

        for jd in state["jds"]:
            ok_results = [tr for tr in results_by_jd.get(jd.id, []) if tr.status == "ok"]
            if not ok_results:
                continue
            ensemble_result = build_ensemble_tournament_result(ok_results)
            sends.append(
                Send(
                    "format_jd_results_node",
                    {"jd": jd, "tournament_result": ensemble_result, "assessments": assessments_by_jd.get(jd.id, {})},
                )
            )
        return sends

    def format_jd_results_node(payload: dict) -> dict:
        write_jd_ranking(cfg.runs_dir / run_id, payload["jd"], payload["tournament_result"], payload["assessments"])
        return {}

    def write_jd_repeats_node(payload: dict) -> dict:
        write_jd_repeats(cfg.runs_dir / run_id, payload["jd_id"], payload["results"])
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_skills_for_jd", extract_skills_for_jd)
    graph.add_node("skills_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("shortlist_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_skill_extraction, ["extract_skills_for_jd"])
    graph.add_edge("extract_skills_for_jd", "skills_barrier")
    graph.add_conditional_edges("skills_barrier", fanout_shortlist, ["build_shortlist_for_jd"])
    graph.add_edge("build_shortlist_for_jd", "shortlist_barrier")
    graph.add_conditional_edges("shortlist_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")

    graph.add_node("run_tournament_for_jd_repeat_node", run_tournament_for_jd_repeat_node)
    graph.add_node("tournament_barrier", _noop)
    graph.add_node("format_jd_results_node", format_jd_results_node)
    graph.add_node("write_jd_repeats_node", write_jd_repeats_node)

    graph.add_conditional_edges("assessments_barrier", fanout_tournament, ["run_tournament_for_jd_repeat_node"])
    graph.add_edge("run_tournament_for_jd_repeat_node", "tournament_barrier")
    graph.add_conditional_edges(
        "tournament_barrier", fanout_finalize, ["format_jd_results_node", "write_jd_repeats_node"]
    )
    graph.add_edge("format_jd_results_node", END)
    graph.add_edge("write_jd_repeats_node", END)

    return graph
