"""The judge conformance suite. Every judge, fake or real, must pass it.

An adapter test calls ``assert_conformant(make_judge)``. ``make_judge(scenario)`` returns a
judge whose provider plays ``scenario.steps`` in order, one step per provider request, the
last step repeating once the list runs out. Each step is ``(kind, payload)``:

    ("reply", text)   the provider answers with ``text`` as the model output
    ("retryable", "") a transient error (HTTP 429 or 5xx)
    ("refusal", "")   a safety block or refusal
    ("error", "")     a permanent error (HTTP 400, bad key)
    ("hang", "")      no reply at all, ever

A real adapter maps these onto recorded responses of its own wire format; the reply texts
are the canonical model outputs below.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import numpy as np

from percept.contracts import FAILURES, Condition, CounterExample, Judge, JudgeItem, JudgeRequest, JudgeResult

SLACK_S = 1.0

ITEMS = (
    JudgeItem(Condition("door_open", "The front door is open")),
    JudgeItem(Condition("fall", "A person falls to the floor", kind="event", not_when=("lying on a sofa",),
                        version=2), counter_examples=(CounterExample("a dog lying down"),)),
    JudgeItem(Condition("package", "A package is left at the door"), subject="person-42", occurrence_gen=3),
)
IDS = tuple(item.condition.id for item in ITEMS)
ANSWERS = {"door_open": ("yes", 0.9, "door is open"), "fall": ("no", 0.8, "standing"),
           "package": ("unknown", 0.3, "door hidden")}


def _results(answers: dict) -> list[dict]:
    return [{"id": cid, "answer": a, "confidence": c, "reason": r} for cid, (a, c, r) in answers.items()]


OK = json.dumps({"results": _results(ANSWERS)})
FENCED = "```json\n" + OK + "\n```"
PROSE = "Here is my assessment of the images.\n" + OK + "\nLet me know if you need more."
MISSING = json.dumps({"results": _results({k: v for k, v in ANSWERS.items() if k != "fall"})})
BAD_CONFIDENCE = json.dumps({"results": _results({**ANSWERS, "fall": ("no", 1.7, "x")})})
EXTRA_ID = json.dumps({"results": _results(ANSWERS) + [{"id": "other", "answer": "yes", "confidence": 1, "reason": ""}]})


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[tuple[str, str], ...]
    expect: dict          # condition id -> failure kind (None: the scripted answer)
    deadline_s: float = 2.0
    min_attempts: int = 1  # for the first provider request


def _all(failure):
    return {cid: failure for cid in IDS}


SCENARIOS = (
    Scenario("ok", (("reply", OK),), _all(None)),
    Scenario("fenced", (("reply", FENCED),), _all(None)),
    Scenario("prose", (("reply", PROSE),), _all(None)),
    Scenario("extra_id", (("reply", EXTRA_ID),), _all(None)),
    Scenario("missing_id", (("reply", MISSING),), {**_all(None), "fall": "missing"}),
    Scenario("bad_confidence", (("reply", BAD_CONFIDENCE),), {**_all(None), "fall": "malformed"}),
    Scenario("prose_only", (("reply", "I can't tell from these images."),), _all("malformed")),
    Scenario("empty", (("reply", ""),), _all("malformed")),
    Scenario("refusal", (("refusal", ""),), _all("refused")),
    Scenario("error", (("error", ""),), _all("unavailable")),
    Scenario("retryable_then_ok", (("retryable", ""), ("reply", OK)), _all(None), min_attempts=2),
    Scenario("retryable_always", (("retryable", ""),), _all("unavailable"), min_attempts=2),
    Scenario("hang", (("hang", ""),), _all("timeout"), deadline_s=0.3),
)


def make_request(scenario: Scenario) -> JudgeRequest:
    images = tuple(np.full((48, 64, 3), 40 * i, dtype=np.uint8) for i in range(3))
    return JudgeRequest(request_id=f"req-{scenario.name}", stream_id="cam-1", stream_epoch=4, items=ITEMS,
                        images=images, evidence_t=(1000.0, 1002.0), purpose="verify",
                        deadline_s=scenario.deadline_s)


def check_result(scenario: Scenario, judge: Judge, request: JudgeRequest, result: JudgeResult) -> None:
    where = f"[{judge.name}/{scenario.name}]"
    assert isinstance(result, JudgeResult), where
    assert tuple(o.condition_id for o in result.observations) == IDS, f"{where} one observation per id, in order"
    for item, ob in zip(ITEMS, result.observations):
        expected = scenario.expect[item.condition.id]
        assert ob.failure == expected, f"{where} {ob.condition_id}: failure {ob.failure!r}, expected {expected!r}"
        assert ob.failure is None or ob.failure in FAILURES
        if expected is None:
            answer, confidence, _ = ANSWERS[item.condition.id]
            assert (ob.answer, ob.confidence) == (answer, confidence), f"{where} {ob.condition_id} answer"
        else:
            assert ob.answer == "unknown", where
        assert (ob.request_id, ob.stream_id, ob.stream_epoch) == (request.request_id, "cam-1", 4), where
        assert (ob.subject, ob.occurrence_gen) == (item.subject, item.occurrence_gen), where
        assert ob.condition_version == item.condition.version, where
        assert (ob.evidence_t, ob.purpose, ob.judge) == (request.evidence_t, "verify", judge.name), where
    assert result.usage, f"{where} usage is reported for every request sent"
    for ru in result.usage:
        assert (ru.request_id, ru.judge) == (request.request_id, judge.name), where
        assert ru.usage.attempts >= 1, where
    assert result.usage[0].usage.attempts >= scenario.min_attempts, f"{where} retries are counted as attempts"
    if judge.price is None:
        assert all(ru.usage.cost_usd == 0 for ru in result.usage), where


async def _run(judge: Judge, request: JudgeRequest) -> tuple[JudgeResult, float]:
    start = time.monotonic()
    try:
        result = await judge.judge(request)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"[{judge.name}] judge() raised {exc!r}; judges never raise") from exc
    return result, time.monotonic() - start


def assert_conformant(make_judge) -> None:
    """Run every scenario against ``make_judge(scenario)``; raise AssertionError on a violation."""
    for scenario in SCENARIOS:
        judge = make_judge(scenario)
        assert isinstance(judge, Judge), "the judge satisfies the Judge protocol"
        request = make_request(scenario)
        result, elapsed = asyncio.run(_run(judge, request))
        assert elapsed <= scenario.deadline_s + SLACK_S, f"[{scenario.name}] took {elapsed:.2f} s"
        check_result(scenario, judge, request, result)
