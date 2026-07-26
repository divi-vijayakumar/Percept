"""— fixture-bound offline mode; no implicit fake judge.

Two contracts, tested against the real runtime (no mocks of our own code):

1. EXPLICIT ``mode="offline"`` binds to a BUNDLED deterministic fixture that ships in the wheel, and
   replaying it through the frozen ``watch()`` seam produces exactly the events the fixture declares.
2. ONLINE (live) mode with no key raises the typed ``ModeError`` — it never silently lands on a
   scripted/fake judge.

The suite has no pytest-asyncio, so coroutines are driven with ``asyncio.run`` (same style as
``tests/test_watch.py``).
"""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path

import pytest

import percept
from percept.adapters import Goal, Verdict
from percept.config import ModeError, Settings
from percept.offline import (
    DEFAULT_FIXTURE,
    FixtureVision,
    OfflineFixture,
    OfflineFixtureError,
    assert_live_or_offline,
    load_fixture,
    offline_core,
    offline_frames,
    offline_goals,
)
from percept.watch import watch

_CRED_ENV = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "DEEPGRAM_API_KEY",
)


@pytest.fixture()
def keyless(monkeypatch):
    """A pristine keyless environment: no credentials, no PERCEPT_* mode/backend overrides.

    The suite-wide conftest declares ``PERCEPT_MODE=offline``; these tests are precisely about what
    happens WITHOUT that declaration, so they strip it back off.
    """
    for key in _CRED_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in ("PERCEPT_MODE", "PERCEPT_VISION_BACKEND", "PERCEPT_CONDUCTOR_BACKEND",
                "PERCEPT_SPECIALIST_BACKEND"):
        monkeypatch.delenv(key, raising=False)


async def _collect(frames, goals, core) -> list:
    return [ev async for ev in watch(frames, goals, percept=core)]


# --- 1. the bundled fixture ---------------------------------------------------------------------


def test_default_fixture_loads_from_the_installed_package():
    fx = load_fixture()
    assert isinstance(fx, OfflineFixture)
    assert fx.fixture_id == DEFAULT_FIXTURE
    assert fx.frames, "a fixture with no frames demonstrates nothing"
    assert fx.expect, "a fixture must declare what the runtime should emit (self-checking)"
    assert set(fx.frames) <= set(fx.script)


def test_fixture_ships_in_the_wheel_package_data():
    """The fixture is only useful if `pip install` actually carries it."""
    root = Path(percept.__file__).resolve().parent
    assert (root / "fixtures" / "__init__.py").exists()
    assert list((root / "fixtures").glob("*.json")), "no bundled fixture json"
    pyproject = tomllib.loads((root.parent / "pyproject.toml").read_text())
    pkg_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "*.json" in pkg_data["percept.fixtures"]


def test_fixture_is_readable_via_importlib_resources_not_a_source_path():
    """Loading must not depend on a source checkout layout (wheel/zip installs have no source dir)."""
    from importlib import resources

    text = (resources.files("percept.fixtures") / "kettle_boils.json").read_text()
    assert json.loads(text)["fixture_id"] == "kettle-boils"


def test_unknown_fixture_name_raises_typed_error():
    with pytest.raises(OfflineFixtureError) as exc:
        load_fixture("no-such-fixture")
    assert "no-such-fixture" in str(exc.value)


def test_offline_fixture_error_is_a_mode_error():
    assert issubclass(OfflineFixtureError, ModeError)


def test_fixture_verdicts_are_labelled_as_replay_not_perception():
    fx = load_fixture()
    for token, verdict in fx.script.items():
        assert isinstance(verdict, Verdict)
        assert "NOT a perception result" in verdict.evidence
        assert token in verdict.evidence


# --- 2. the replayer ----------------------------------------------------------------------------


def test_fixture_vision_replays_the_scripted_verdict():
    fx = load_fixture()
    vision = FixtureVision(fx)
    for token, expected in fx.script.items():
        got = asyncio.run(vision.judge(fx.goal.condition, token))
        assert (got.satisfied, got.confidence) == (expected.satisfied, expected.confidence)


def test_fixture_vision_refuses_a_frame_it_does_not_script():
    """A real camera pointed at offline mode must fail loudly, never invent a plausible verdict."""
    vision = FixtureVision()
    with pytest.raises(OfflineFixtureError) as exc:
        asyncio.run(vision.judge("anything", "a-frame-the-fixture-never-heard-of"))
    msg = str(exc.value)
    assert "does not perceive" in msg
    assert "a-frame-the-fixture-never-heard-of" in msg


def test_fixture_vision_asserts_no_object_intelligence():
    vision = FixtureVision()
    assert asyncio.run(vision.detect("kettle-cold", [])) == []
    assert asyncio.run(vision.detect_events([(0.0, "kettle-cold")], [])) == []


def test_an_unscripted_frame_never_fires_through_the_runtime(keyless):
    """Even though the runtime fails a judge CLOSED, an unknown frame must never produce a FIRE."""
    fx = load_fixture()
    events = asyncio.run(_collect(["not-in-this-fixture"], offline_goals(fx), offline_core(fx)))
    assert [e for e in events if e.action == "fire"] == []


# --- 3. offline end to end through the frozen watch() seam --------------------------------------


def test_offline_run_is_keyless_and_reproduces_the_fixture_expectation(keyless):
    """Installed package -> offline mode -> real gate path fires, with no API key set."""
    fx = load_fixture()
    core = offline_core(fx)
    assert core._mode == "offline"

    events = asyncio.run(_collect(offline_frames(fx), offline_goals(fx), core))

    assert [(e.goal_id, e.action) for e in events] == [
        (e["goal_id"], e["action"]) for e in fx.expect
    ]
    fired = events[0]
    assert fired.text == fx.goal.say
    assert fired.verdict is not None
    assert "NOT a perception result" in fired.verdict.evidence


def test_offline_run_is_deterministic_across_two_fresh_cores(keyless):
    fx = load_fixture()

    def run() -> list:
        return [
            (ev.goal_id, ev.action)
            for ev in asyncio.run(_collect(offline_frames(fx), offline_goals(fx), offline_core(fx)))
        ]

    assert run() == run()


def test_offline_fires_only_on_the_rising_edge(keyless):
    """The fixture holds the condition true for two frames; the gate must fire once, not twice."""
    fx = load_fixture()
    true_frames = [f for f in fx.frames if fx.script[f].satisfied]
    assert len(true_frames) >= 2, "fixture must hold the condition true for >1 frame to prove this"
    events = asyncio.run(_collect(offline_frames(fx), offline_goals(fx), offline_core(fx)))
    assert sum(1 for e in events if e.action == "fire") == 1


def test_offline_core_owns_mode_and_vision():
    with pytest.raises(TypeError):
        offline_core(vision="claude")
    with pytest.raises(TypeError):
        offline_core(mode="live")


def test_offline_core_forwards_other_kwargs(keyless):
    core = offline_core(settings=Settings())
    assert isinstance(core.vision, FixtureVision)


def test_registry_exposes_the_fixture_vision_selector(keyless):
    from percept import registry

    built = registry.make_vision("fixture", Settings())
    assert isinstance(built, FixtureVision)


# --- 4. no implicit fake judge in a live run ----------------------------------------------------


def test_live_run_with_no_key_raises_typed_error_and_never_fakes(keyless):
    with pytest.raises(ModeError) as exc:
        percept.Percept.create()
    assert "offline" in str(exc.value)


def test_live_run_with_a_selected_backend_but_no_credential_raises(keyless):
    with pytest.raises(ModeError):
        percept.Percept.create(vision="claude")


def test_watch_without_a_core_does_not_silently_fake_in_live_mode(keyless):
    """watch() builds its own core when `percept=` is omitted — that path must fail closed too."""
    goal = Goal(id="g", condition="x", say="y")

    async def _drive():
        async for _ in watch(["kettle-cold"], [goal]):
            pass

    with pytest.raises(ModeError):
        asyncio.run(_drive())


def test_assert_live_or_offline_contract(keyless, monkeypatch):
    assert assert_live_or_offline("offline") == "offline"
    with pytest.raises(ModeError):
        assert_live_or_offline("live")
    with pytest.raises(ModeError):
        assert_live_or_offline("live", vision="claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert assert_live_or_offline("live", vision="claude") == "live"


def test_offline_mode_reads_no_credential(keyless, monkeypatch):
    """Offline must not even LOOK at a key — set a bogus one and prove nothing changes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-used")
    core = offline_core()
    assert isinstance(core.vision, FixtureVision)
    assert core._mode == "offline"
