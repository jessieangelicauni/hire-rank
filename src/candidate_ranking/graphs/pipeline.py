# src/candidate_ranking/graphs/pipeline.py
from __future__ import annotations

import logging
import operator
from typing import Annotated, Callable, Literal, TypedDict

import numpy as np
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from candidate_ranking.config import RunConfig
from candidate_ranking.models import Assessment, Candidate, JDRequirements, JDSkills, JobDescription
from candidate_ranking.output.formatter import write_jd_ranking
from candidate_ranking.scoring.jd_requirements import (
    JDRequirementsGenerationError,
    load_or_generate_jd_requirements,
)
from candidate_ranking.scoring.jd_skills import JDSkillsGenerationError, load_or_generate_jd_skills
from candidate_ranking.scoring.llm_assessment import (
    AssessmentGenerationError,
    load_or_generate_assessment,
)
from candidate_ranking.scoring.sliding_window_ranking import rank_sliding_window
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
    jd_requirements: Annotated[dict[str, JDRequirements], operator.or_]
    shortlists: Annotated[dict[str, list[str]], operator.or_]
    assessment_results: Annotated[list[AssessmentResult], operator.add]


def assessments_by_jd(assessment_results: list[AssessmentResult]) -> dict[str, dict[str, Assessment]]:
    grouped: dict[str, dict[str, Assessment]] = {}
    for result in assessment_results:
        if result["status"] != "ok":
            continue
        grouped.setdefault(result["jd_id"], {})[result["candidate_id"]] = result["assessment"]
    return grouped


def build_pipeline_graph(
    cfg: RunConfig,
    jd_skills_chain: Runnable,
    jd_requirements_chain: Runnable,
    qualification_chain: Runnable,
    build_sliding_window_chain: Callable[[type[BaseModel]], Runnable],
    skill_index: np.ndarray,
    skill_row_map: list[tuple[str, str]],
    skill_embedder: Callable[[list[str]], np.ndarray],
    run_id: str,
    skill_match_threshold: float = 0.8,
    min_skill_matches: int = 5,
    min_must_have_matches: int = 2,
) -> StateGraph:
    jd_skills_cache_path = cfg.cache_dir / "jd_skills.json"
    jd_requirements_cache_path = cfg.cache_dir / "jd_requirements.json"

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

    def fanout_shortlist_and_requirements(state: PipelineState) -> list[Send]:
        sends = []
        for jd in state["jds"]:
            if jd.id in state["jd_skills"]:
                sends.append(Send("build_shortlist_for_jd", {"jd": jd, "jd_skills": state["jd_skills"][jd.id]}))
                sends.append(Send("extract_requirements_for_jd", {"jd": jd}))
        return sends

    def build_shortlist_for_jd(payload: dict) -> dict:
        jd, jd_skills = payload["jd"], payload["jd_skills"]
        candidate_ids = shortlist_candidates(
            jd, jd_skills, skill_index, skill_row_map, skill_embedder,
            threshold=skill_match_threshold, min_matches=min_skill_matches,
            min_must_have_matches=min_must_have_matches,
        )
        return {"shortlists": {jd.id: candidate_ids}}

    def extract_requirements_for_jd(payload: dict) -> dict:
        jd = payload["jd"]
        try:
            jd_requirements = load_or_generate_jd_requirements(
                jd, jd_requirements_chain, cfg.ollama_model, jd_requirements_cache_path
            )
        except JDRequirementsGenerationError as exc:
            logger.warning("Skipping requirements for %s: %s", jd.id, exc)
            return {}
        return {"jd_requirements": {jd.id: jd_requirements}}

    def fanout_assessments(state: PipelineState) -> list[Send]:
        candidates_by_id = {c.id: c for c in state["candidates"]}
        return [
            Send(
                "generate_assessment_for_pair",
                {
                    "jd": jd,
                    "candidate": candidates_by_id[cid],
                    "jd_requirements": state["jd_requirements"].get(jd.id),
                },
            )
            for jd in state["jds"]
            for cid in state["shortlists"].get(jd.id, [])
        ]

    def generate_assessment_for_pair(payload: dict) -> dict:
        jd, candidate = payload["jd"], payload["candidate"]
        jd_requirements = payload.get("jd_requirements")
        try:
            assessment = load_or_generate_assessment(
                jd, candidate, jd_requirements, qualification_chain, cfg.ollama_model, cfg.cache_dir
            )
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
        return [
            Send("rank_and_format_jd_node", {"jd": jd, "assessments": grouped.get(jd.id, {})})
            for jd in state["jds"] if grouped.get(jd.id)
        ]

    def rank_and_format_jd_node(payload: dict) -> dict:
        jd: JobDescription = payload["jd"]
        all_assessments: dict[str, Assessment] = payload["assessments"]

        qualified = {cid: a for cid, a in all_assessments.items() if a.qualified}
        disqualified_count = len(all_assessments) - len(qualified)
        if disqualified_count > 0:
            logger.info("JD %s: %d/%d candidates disqualified", jd.id, disqualified_count, len(all_assessments))

        if not qualified:
            logger.warning("No qualified candidates for JD %s — skipping ranking", jd.id)
            write_jd_ranking(cfg.runs_dir / run_id, jd, {}, {})
            return {}

        utilities = rank_sliding_window(
            jd,
            qualified,
            build_sliding_window_chain,
            window_size=cfg.sliding_window_size,
            step=cfg.sliding_window_step,
        )

        write_jd_ranking(cfg.runs_dir / run_id, jd, qualified, utilities)
        return {}

    graph = StateGraph(PipelineState)
    graph.add_node("start", _noop)
    graph.add_node("extract_skills_for_jd", extract_skills_for_jd)
    graph.add_node("skills_barrier", _noop)
    graph.add_node("build_shortlist_for_jd", build_shortlist_for_jd)
    graph.add_node("extract_requirements_for_jd", extract_requirements_for_jd)
    graph.add_node("shortlist_requirements_barrier", _noop)
    graph.add_node("generate_assessment_for_pair", generate_assessment_for_pair)
    graph.add_node("assessments_barrier", _noop)
    graph.add_node("rank_and_format_jd_node", rank_and_format_jd_node)

    graph.set_entry_point("start")
    graph.add_conditional_edges("start", fanout_skill_extraction, ["extract_skills_for_jd"])
    graph.add_edge("extract_skills_for_jd", "skills_barrier")
    graph.add_conditional_edges(
        "skills_barrier",
        fanout_shortlist_and_requirements,
        ["build_shortlist_for_jd", "extract_requirements_for_jd"],
    )
    graph.add_edge("build_shortlist_for_jd", "shortlist_requirements_barrier")
    graph.add_edge("extract_requirements_for_jd", "shortlist_requirements_barrier")
    graph.add_conditional_edges("shortlist_requirements_barrier", fanout_assessments, ["generate_assessment_for_pair"])
    graph.add_edge("generate_assessment_for_pair", "assessments_barrier")
    graph.add_conditional_edges("assessments_barrier", fanout_rank_and_format, ["rank_and_format_jd_node"])
    graph.add_edge("rank_and_format_jd_node", END)

    return graph
