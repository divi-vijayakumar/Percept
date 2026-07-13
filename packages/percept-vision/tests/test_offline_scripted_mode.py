"""Acceptance tests for the offline/scripted-mode RFC (RFC-offline-scripted-mode.md §9).

Prime directive: every bug in this codebase is a default that RUNS, LOOKS correct, and silently
does nothing. The one configuration that silently LOOKS like success — a scripted brain behind a
live voice — must be impossible to reach by accident, and scripted backends must always be loud.

The suite's conftest sets ``PERCEPT_MODE=offline`` (keyless). Tests that exercise LIVE behaviour
delete it explicitly (a bare production run has no PERCEPT_MODE).
"""
from __future__ import annotations

import asyncio
import logging

import pytest

import percept
from percept import ModeError
from percept.adapters import Goal
from percept.backends.fake import FakeConductor, FakeVision, ScriptedConductor
from percept.edge.tier0 import gray_frame
from percept.observability import get_logger

_FRAME = gray_frame(8, 8, 0)


# --------------------------------------------------------------------------- helpers ----
class _CaptureWarnings:
    """Capture WARNING+ records off the 'percept' logger (propagate=False, so caplog can't see it)."""

    def __init__(self, name: str = "percept.agent") -> None:
        self.records: list[logging.LogRecord] = []
        self._logger = get_logger(name)

        class _H(logging.Handler):
            def emit(_self, record):  # noqa: N805
                self.records.append(record)

        self._handler = _H(level=logging.WARNING)

    def __enter__(self):
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


async def _one_frame_source():
    # (t, frame, judge-token): a real gray frame for the tier-0 gate, "empty" token for FakeVision.
    yield (0.0, _FRAME, "empty")


# --------------------------------------------------- §9.1 bare live create() -> ModeError ----
def test_bare_create_live_no_key_raises_modeerror(monkeypatch):
    # A bare Percept.create() in LIVE mode must refuse to silently fake — ModeError naming the fix.
    monkeypatch.delenv("PERCEPT_MODE", raising=False)
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ModeError) as ei:
        percept.Percept.create(discover_plugins=False)
    msg = str(ei.value)
    assert "offline" in msg and "vision" in msg  # names the fix


def test_bare_create_live_selected_backend_missing_credential(monkeypatch):
    # A SELECTED real backend with no credential -> ModeError naming the exact env var (per-backend).
    monkeypatch.delenv("PERCEPT_MODE", raising=False)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AISTUDIO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ModeError) as ei:
        percept.Percept.create(vision="gemini", discover_plugins=False)
    assert "GEMINI_API_KEY" in str(ei.value)


# ------------------------ §9.2 THE TEST THAT MATTERS: scripted brain behind a live voice ----
def test_run_on_utterance_scripted_conductor_raises(monkeypatch):
    monkeypatch.delenv("PERCEPT_MODE", raising=False)  # LIVE
    core = percept.Percept.create(
        mode="live", vision=FakeVision(), conductor=FakeConductor(), discover_plugins=False
    )

    async def _drive():
        async for _ in percept.run(core, source=_one_frame_source(), on_utterance=lambda a, t: None):
            pass

    with pytest.raises(ModeError) as ei:
        asyncio.run(_drive())
    assert "scripted brain behind a live voice" in str(ei.value)


def test_start_on_utterance_scripted_conductor_raises(monkeypatch):
    from percept.runtime.session import start

    monkeypatch.delenv("PERCEPT_MODE", raising=False)
    core = percept.Percept.create(
        mode="live", vision=FakeVision(), conductor=FakeConductor(), discover_plugins=False
    )
    with pytest.raises(ModeError):
        asyncio.run(start(core, source=_one_frame_source(), on_utterance=lambda a, t: None))


def test_attach_transport_scripted_conductor_raises(monkeypatch):
    from percept.runtime.session import RuntimeSession
    from percept.backends.fake import FakeVoiceTransport

    monkeypatch.delenv("PERCEPT_MODE", raising=False)
    core = percept.Percept.create(
        mode="live", vision=FakeVision(), conductor=FakeConductor(), discover_plugins=False
    )
    session = RuntimeSession(core)
    with pytest.raises(ModeError):
        asyncio.run(session.attach_transport(FakeVoiceTransport()))


def test_language_seam_scripted_conductor_raises_the_fourth_door(monkeypatch):
    # session.ask() / on_speech_endpoint and arm_from_language route straight to the conductor, bypassing
    # run()/start()/attach_transport. Guarding only those three wiring points left this FOURTH door open
    # (a custom-STT integration: `start(core, source=…)` then `session.ask(text)` -> scripted brain ->
    # route="ignore" -> silence). The guard now lives at the language<->conductor crossing, so every door
    # is covered by one check, not each wiring point separately.
    monkeypatch.delenv("PERCEPT_MODE", raising=False)  # LIVE
    core = percept.Percept.create(
        mode="live", vision=FakeVision(), conductor=FakeConductor(), discover_plugins=False
    )
    with pytest.raises(ModeError):  # the session.ask(...) path
        asyncio.run(core.on_speech_endpoint("tell me when someone arrives"))
    with pytest.raises(ModeError):  # the raw-language arming path
        asyncio.run(core.arm_from_language("tell me when someone arrives"))


def test_offline_conductor_is_the_honest_double_not_a_null_object():
    # The RFC's thesis: a double must never return the value indistinguishable from success. mode="offline"
    # is the door people explore through, so it must resolve the conductor to the HONEST ScriptedConductor
    # (no-match -> route="respond_direct"), NOT FakeConductor (route="ignore", which looks exactly like a
    # real conductor correctly declining — the one value that silently looks like it worked).
    core = percept.Percept.create(mode="offline", discover_plugins=False)
    assert type(core.conductor).__name__ == "ScriptedConductor"
    decision = asyncio.run(core.conductor.run("speech", "tell me when the kettle boils", {}))
    assert decision.route == "respond_direct"  # honest no-author, NEVER "ignore"


# --------------------------- §9.3 code-armed watch (no language path) -> constructs, loud ----
def test_code_armed_real_vision_scripted_conductor_constructs_loud():
    # vision="gemini" (real object) + default scripted conductor + NO language path: the common
    # watch-agent pattern. Must CONSTRUCT (not crash) and announce that authoring is off.
    events: list = []
    with _CaptureWarnings("percept.agent") as cap:
        a = percept.Percept.create(vision="gemini", on_health=events.append, discover_plugins=False)
    assert type(a.vision).__name__ == "GeminiVision"        # a real vision backend
    assert "conductor" in a.health()["scripted"]            # scripted brain, announced
    assert "scripted_conductor" in events
    assert any("authoring" in m or "SCRIPTED" in m for m in cap.messages())


def test_code_armed_run_no_utterance_does_not_raise():
    # A code-armed watch drives run() with NO on_utterance -> the scripted conductor is fine (§5).
    core = percept.Percept.create(vision=FakeVision(), conductor=FakeConductor(), discover_plugins=False)
    core.add_goal(Goal(id="g", condition="x", say="hi"))

    async def _drive():
        return [inc async for inc in percept.run(core, source=_one_frame_source())]

    asyncio.run(_drive())  # no ModeError


# ------------------------------------------------- §9.4 mode="offline" -> all scripted, health ----
def test_mode_offline_all_scripted_health():
    a = percept.Percept.create(mode="offline", discover_plugins=False)
    h = a.health()
    assert h["mode"] == "offline"
    assert "vision" in h["scripted"] and "conductor" in h["scripted"]


# ---------------------------------------------- §9.5 PERCEPT_MODE == kwarg; kwarg wins ----
def test_env_mode_equivalent_to_kwarg(monkeypatch):
    monkeypatch.setenv("PERCEPT_MODE", "offline")
    a = percept.Percept.create(discover_plugins=False)
    assert a.health()["mode"] == "offline"
    assert "vision" in a.health()["scripted"]


def test_kwarg_mode_wins_over_env(monkeypatch):
    # env=offline but kwarg=live: the kwarg wins -> LIVE -> a bare create() raises (no fallback).
    monkeypatch.setenv("PERCEPT_MODE", "offline")
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ModeError):
        percept.Percept.create(mode="live", discover_plugins=False)


# ------------------------------------------ §9.6 vision="scripted" canonical; "fake" still works ----
def test_scripted_selector_is_canonical():
    a = percept.Percept.create(vision="scripted", discover_plugins=False)
    assert type(a.vision).__name__ == "FakeVision"


def test_fake_selector_still_resolves():
    a = percept.Percept.create(vision="fake", discover_plugins=False)
    assert type(a.vision).__name__ == "FakeVision"


# ----------------------------------------- §9.7 allow_scripted_brain hatch: case 2 constructs ----
def test_allow_scripted_brain_hatch_constructs(monkeypatch):
    from percept.runtime.session import start

    monkeypatch.delenv("PERCEPT_MODE", raising=False)  # LIVE
    core = percept.Percept.create(
        mode="live", vision=FakeVision(), conductor=FakeConductor(),
        allow_scripted_brain=True, discover_plugins=False,
    )

    async def _go():
        session = await start(core, source=_one_frame_source(), on_utterance=lambda a, t: None)
        await session.stop()

    asyncio.run(_go())  # no ModeError — the hatch is honored


# ------------------------------------- §9.9 ScriptedConductor authors a real goal; no-match honest ----
def test_scripted_conductor_authors_a_goal():
    # The keyless goal-authoring test that did not exist: utterance -> a goal exists.
    goal = Goal(id="caffeine", condition="the person is drinking coffee or tea", say="caffeine.")
    conductor = ScriptedConductor({"caffeine": goal})
    decision = asyncio.run(conductor.run("speech", "tell me when I'm drinking caffeine", {}))
    assert decision.route == "respond_direct"
    assert len(decision.goal_ops) == 1
    op = decision.goal_ops[0]
    assert op.op == "create" and op.goal.id == "caffeine"


def test_scripted_conductor_no_match_is_respond_direct_not_ignore():
    # The whole point: a double must never return the value indistinguishable from success.
    conductor = ScriptedConductor({"caffeine": Goal(id="c", condition="c", say="c")})
    decision = asyncio.run(conductor.run("speech", "what's the weather", {}))
    assert decision.route == "respond_direct"      # honest no-author
    assert decision.route != "ignore"              # NEVER the silent-success value
    assert decision.goal_ops == []


def test_scripted_conductor_arms_through_the_core():
    # End-to-end keyless: a scripted utterance flowing through the core arms a real, active goal.
    goal = Goal(id="kettle", condition="the kettle is boiling", say="Kettle's boiling.")
    core = percept.Percept.create(
        vision=FakeVision(), conductor=ScriptedConductor({"kettle": goal}), discover_plugins=False
    )
    asyncio.run(core.arm_from_language("tell me when the kettle boils"))
    assert core.goals.get("kettle") is not None
    assert "kettle" in [g.id for g in core.goals.active()]
