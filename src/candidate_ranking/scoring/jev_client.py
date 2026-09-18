from __future__ import annotations

from typing import Literal

import requests
from pydantic import BaseModel

_ENDPOINT_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
JEV_MODEL_ID = "typesafe/jev"


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


class JevClientError(Exception):
    pass


def _parse_answer(key: str, raw: dict) -> JevAnswer:
    kind = raw.get("type")
    try:
        if kind == "noul":
            probability_true = float(raw["noul"])
            confidence = probability_true if probability_true >= 0.5 else 1.0 - probability_true
            return JevAnswer(key=key, kind="noul", value=probability_true >= 0.5, confidence=confidence)
        if kind == "choice":
            return JevAnswer(key=key, kind="choice", value=raw["choice"], confidence=float(raw["confidence"]))
        if kind == "score":
            return JevAnswer(key=key, kind="score", value=float(raw["score"]), confidence=float(raw["confidence"]))
    except KeyError as exc:
        raise JevClientError(f"Jev answer for {key!r} is missing expected field: {exc}") from exc
    raise JevClientError(f"Jev answer for {key!r} has unrecognized type: {kind!r}")


class JevClient:
    def __init__(self, account_id: str, api_token: str, timeout: float = 30.0) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._timeout = timeout

    def evaluate(self, state: str, questions: list[JevQuestion]) -> list[JevAnswer]:
        payload = {
            "model": JEV_MODEL_ID,
            "input": {
                "state": state,
                "questions": {
                    q.key: {"type": q.kind, "instructions": q.instructions, "criteria": q.criteria}
                    for q in questions
                },
            },
        }
        url = _ENDPOINT_TEMPLATE.format(account_id=self._account_id)
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_token}"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise JevClientError(f"Jev request failed: {exc}") from exc
        except ValueError as exc:
            raise JevClientError(f"Jev response was not valid JSON: {exc}") from exc

        try:
            raw_answers = data["answers"]
        except KeyError as exc:
            raise JevClientError(f"Jev response is missing 'answers': {data}") from exc

        return [_parse_answer(key, raw) for key, raw in raw_answers.items()]
