from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from candidate_ranking.scoring.assessment import AssessmentGenerationError, JEV_MODEL_NAME, load_or_generate_assessment
from candidate_ranking.scoring.jev_client import JevClient
from candidate_ranking.config import RunConfig
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, load_or_generate_jd_skills
from candidate_ranking.output.formatter import write_jd_ranking
from candidate_ranking.models import Assessment, Candidate, JDSkills, JobDescription
from candidate_ranking.scoring.skills import shortlist_candidates

logger = logging.getLogger(__name__)


class AssessmentResult(TypedDict):
    jd_id: str
    candidate_id: str
    status: Literal["ok", "failed"]
    assessment: Assessment | None
    error: str | None


class PipelineState(TypedDict):
    jds: list[JobDescription]
    candidates: list[Candidate]
    jd_skills: Annotated[dict[str, JDSkills], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]


def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]:
    grouped: dict[str, dict[str, Assessment]] = {}
    for result in assessment_results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["jd_id"], {})[result["candidate_id"]] = result["assessment"]
    return grouped


def rank_and_format_jd(runs_dir: Path, run_id: str, jd: JobDescription, assessments: dict[str, Assessment]) -> None:
    write_jd_ranking(runs_dir / run_id, jd, assessments)


def build_pipeline_graph(
    cfg: RunConfig,
    jd_skills_chain: Runnable,
    jev_client: JevClient,
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
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
            min_must_have_matches=min_must_have_matches,
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
            assessment = load_or_generate_assessment(jd, candidate, jev_client, JEV_MODEL_NAME, cfg.cache_dir, jd_skills=jd_skills)
            result: AssessmentResult = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "ok",
                "assessment": assessment, "error": None,
            }
        except AssessmentGenerationError as exc:
            result = {
                "jd_id": jd.id, "candidate_id": candidate.id, "status": "failed",
                "assessment": None, "error": str(exc),
            }
        return {"assessment_results": [result]}

    def fanout_rank_and_format(state: PipelineState) -> list[Send]:
        grouped = assessments_by_jd(state["assessment_results"])
        sends = []
        for jd in state["jds"]:
            jd_assessments = grouped.get(jd.id, {})
            if not jd_assessments:
                continue
            sends.append(Send("rank_and_format_jd_node", {"jd": jd, "assessments": jd_assessments}))
        return sends

    def rank_and_format_jd_node(payload: dict) -> dict:
        rank_and_format_jd(cfg.runs_dir, run_id, payload["jd"], payload["assessments"])
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_skills_for_jd", extract_skills_for_jd)
    graph.add_node("skills_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("shortlist_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_skill_extraction, ["extract_skills_for_jd"])
    graph.add_edge("extract_skills_for_jd", "skills_barrier")
    graph.add_conditional_edges("skills_barrier", fanout_shortlist, ["build_shortlist_for_jd"])
    graph.add_edge("build_shortlist_for_jd", "shortlist_barrier")
    graph.add_conditional_edges("shortlist_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
