from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator, Literal

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL_ID = "jev-latest"
JEV_PRICE_PER_MILLION_INPUT_TOKENS = 0.042

_USAGE_TOKEN_KEYS = ("input_tokens", "prompt_tokens", "tokens")


def _tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def _parse_input_tokens(raw: dict) -> int | None:
    usage = raw.get("usage") or {}
    for key in _USAGE_TOKEN_KEYS:
        if key in usage:
            return int(usage[key])
    return None


class JevQuestion(BaseModel):
    key: str
    kind: Literal["noul", "choice", "score"]
    instructions: str
    criteria: dict[str, str] | list[str]


class JevAnswer(BaseModel):
    key: str
    kind: Literal["noul", "choice", "score"]
    value: bool | str | float
    confidence: float
    probabilities: dict[str, float] | None = None


class JevClientError(Exception):
    pass


@contextmanager
def _traced_generation(state: str, questions: list[JevQuestion]) -> Iterator[object | None]:
    if not _tracing_enabled():
        yield None
        return
    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        name="jev-evaluate",
        as_type="generation",
        model=JEV_MODEL_ID,
        input={"state": state, "questions": {q.key: q.instructions for q in questions}},
    ) as generation:
        yield generation


def _parse_probabilities(raw: dict) -> dict[str, float] | None:
    probabilities = raw.get("probabilities")
    if not probabilities:
        return None
    return {k: float(v) for k, v in probabilities.items()}


def _parse_answer(key: str, raw: dict) -> JevAnswer:
    kind = raw.get("type")
    try:
        if kind == "noul":
            probability_true = float(raw["noul"])
            confidence = probability_true if probability_true >= 0.5 else 1.0 - probability_true
            return JevAnswer(key=key, kind="noul", value=probability_true >= 0.5, confidence=confidence)
        if kind == "choice":
            return JevAnswer(
                key=key, kind="choice", value=raw["choice"], confidence=float(raw["confidence"]),
                probabilities=_parse_probabilities(raw),
            )
        if kind == "score":
            return JevAnswer(
                key=key, kind="score", value=float(raw["score"]), confidence=float(raw["confidence"]),
                probabilities=_parse_probabilities(raw),
            )
    except KeyError as exc:
        raise JevClientError(f"Jev answer for {key!r} is missing expected field: {exc}") from exc
    raise JevClientError(f"Jev answer for {key!r} has unrecognized type: {kind!r}")


class JevClient:
    def __init__(self, api_token: str, timeout: float = 30.0) -> None:
        self._api_token = api_token
        self._timeout = timeout

    def evaluate(self, state: str, questions: list[JevQuestion]) -> list[JevAnswer]:
        payload = {
            "state": state,
            "model": JEV_MODEL_ID,
            "questions": {
                q.key: {"type": q.kind, "instructions": q.instructions, "criteria": q.criteria}
                for q in questions
            },
        }
        with _traced_generation(state, questions) as generation:
            try:
                t0 = time.perf_counter()
                response = requests.post(
                    _ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_token}"},
                    timeout=self._timeout,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                if generation is not None:
                    generation.update(level="ERROR", status_message=str(exc))
                raise JevClientError(f"Jev request failed: {exc}") from exc
            except ValueError as exc:
                if generation is not None:
                    generation.update(level="ERROR", status_message=str(exc))
                raise JevClientError(f"Jev response was not valid JSON: {exc}") from exc

            n_questions = len(questions)
            resolved_model = data.get("model", "unknown")
            logger.debug(
                "Jev latency: %.0f ms (%d question(s)), resolved model: %s",
                latency_ms, n_questions, resolved_model,
            )

            try:
                raw_answers = data["answers"]
            except KeyError as exc:
                if generation is not None:
                    generation.update(level="ERROR", status_message=f"missing 'answers': {data}")
                raise JevClientError(f"Jev response is missing 'answers': {data}") from exc

            answers = [_parse_answer(key, raw) for key, raw in raw_answers.items()]

            if generation is not None:
                input_tokens = _parse_input_tokens(data)
                usage_details = {"input": input_tokens, "output": 0} if input_tokens is not None else None
                cost_input = (
                    input_tokens / 1_000_000 * JEV_PRICE_PER_MILLION_INPUT_TOKENS
                    if input_tokens is not None else None
                )
                cost_details = {"input": cost_input, "total": cost_input} if cost_input is not None else None
                generation.update(
                    model=resolved_model,
                    output=[a.model_dump() for a in answers],
                    usage_details=usage_details,
                    cost_details=cost_details,
                    metadata={"latency_ms": latency_ms, "n_questions": n_questions},
                )

            return answers
