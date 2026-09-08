"""The frozen conductor must dry-arm every selected plan before a paid judge exists."""

from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import live_run as LR  # noqa: E402
from percept.conductor import PlanConductor  # noqa: E402


FROZEN_PIN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "primitive-authoring"
    / "corpus-v1.2.frozen.json"
)


@pytest.fixture
def frozen_pin(monkeypatch):
    monkeypatch.setattr(LR, "ACTION_CONDUCTOR_PIN", str(FROZEN_PIN))
    LR._authoring_pin.cache_clear()
    yield
    LR._authoring_pin.cache_clear()


def _frozen_primitives(cases: list[dict]) -> list[object]:
    conductor = PlanConductor(cache=LR._authoring_pin())

    async def resolve() -> list[object]:
        return [await conductor.author(str(case["goal_text"])) for case in cases]

    return asyncio.run(resolve())


def test_exact_frozen_corpus_dry_arm_names_every_unsupported_runtime_route(frozen_pin):
    cases = LR.SM.load_corpus()
    authored = _frozen_primitives(cases)

    issues = LR.authored_plan_armability_issues(cases, authored)

    assert len(cases) == 96
    assert [issue.split(": WatchPlan", 1)[0] for issue in issues] == [
        "omnipro::6RfXt26r1kQ::49",
        "omnipro::HF8nrtwFVA8::151",
        "omnipro::-ZvLZrWsFAc::3",
        "omnipro::QtK2gxsTNyM::148",
        "omnipro::C3Wx2o4LZko::79",
        "omnipro::GjDTEjAHu_Y::101",
        "omnipro::MqarNjlLyvk::135",
    ]
    assert all("unsupported_runtime_route" in issue for issue in issues)
    assert all("external_boundary" in issue for issue in issues)
    assert all("Percept.perceive_boundary" in issue for issue in issues)


def test_active_pin_aborts_on_unarmable_count_even_without_expect_flag(
    frozen_pin, monkeypatch
):
    async def provider_path_must_not_start(*_args, **_kwargs):
        raise AssertionError("paid case path started despite failed dry-arm preflight")

    monkeypatch.setattr(LR, "run_case", provider_path_must_not_start)

    with pytest.raises(SystemExit) as caught:
        LR.main([
            "--cases", "omnipro::6RfXt26r1kQ::49",
            "--no-webhook-probe",
            "--max-provider-usd", "1",
            "--max-provider-calls", "1",
            "--conservative-usd-per-call", "1",
            "--global-prior-committed-usd", "0",
            "--global-prior-commitment-source", "fresh-test-run",
            "--global-prior-commitment-basis", "conservative-reservations",
        ])

    message = str(caught.value)
    assert "ARM ABORTED before provider calls" in message
    assert "unsupported_runtime_route" in message
    assert "Percept.perceive_boundary" in message
