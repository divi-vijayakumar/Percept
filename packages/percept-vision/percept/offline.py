"""Fixture-bound OFFLINE mode — the keyless first run.

Two halves of one rule: **you may run Percept with no API key, and you may never mistake that run
for perception.**

* ``mode="offline"`` is EXPLICIT and bound to a BUNDLED DETERMINISTIC FIXTURE
  (``percept/fixtures/*.json``, shipped in the wheel). :func:`offline_core` builds a core whose
  vision seam is a :class:`FixtureVision` — a replayer that looks each frame token up in the
  fixture's script and returns the verdict a human wrote there. A frame the fixture does not
  contain raises :class:`OfflineFixtureError` rather than inventing an answer, so piping a real
  camera into offline mode fails loudly instead of producing plausible-looking judgements.
* ONLINE (live) mode with no credential raises the typed :class:`~percept.config.ModeError` and
  NEVER falls back to a scripted judge — that check already lives in
  :meth:`percept.agent.Percept.create` (an unset vision seam in a live run, and a selected backend
  whose key is absent). :func:`assert_live_or_offline` is the same contract as a callable
  pre-flight for front doors that want to fail before touching a stream.

Nothing here changes the frozen :func:`percept.watch.watch` signature: an offline demo is
``watch(offline_frames(), [goal], percept=offline_core())`` — the existing ``percept=`` seam.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .adapters import Goal, Verdict
from .config import ModeError, missing_credential, resolve_mode

#: The fixture used when none is named — the bundled kettle replay.
DEFAULT_FIXTURE = "kettle-boils"

#: Registry-visible backend selector for the bundled-fixture vision seam.
FIXTURE_BACKEND = "fixture"

_FIXTURE_FILES: dict[str, str] = {
    "kettle-boils": "kettle_boils.json",
}


class OfflineFixtureError(ModeError):
    """The offline replay was asked for something the fixture cannot honestly answer.

    A :class:`~percept.config.ModeError` subclass because it is the same failure family: a run is
    about to look like success without any perception behind it. Raised for an unknown fixture name,
    a malformed fixture file, and — the important one — a frame token that is not part of the
    fixture's script (offline REPLAYS, it does not perceive).
    """


@dataclass(frozen=True)
class OfflineFixture:
    """A loaded bundled fixture: the goal, the frame sequence, the scripted verdicts, the expectation.

    ``script`` maps a frame token to the :class:`~percept.adapters.Verdict` the replay returns for it.
    ``expect`` is the fixture author's own statement of what the runtime should emit
    (``[{"frame_index": int, "goal_id": str, "action": "fire"|"ask"}, ...]``) — it is what makes a
    fixture self-checking in CI rather than a demo nobody verifies.
    """

    fixture_id: str
    fixture_version: int
    description: str
    goal: Goal
    frames: tuple[str, ...]
    script: Mapping[str, Verdict]
    expect: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def verdict_for(self, token: str) -> Verdict:
        """The scripted verdict for ``token``, or :class:`OfflineFixtureError` if it isn't in the script."""
        try:
            return self.script[token]
        except KeyError:
            raise OfflineFixtureError(
                f"offline fixture {self.fixture_id!r} has no scripted verdict for frame {token!r}. "
                f"Offline mode REPLAYS a bundled fixture — it does not perceive, so it will not "
                f"invent a verdict for an unknown frame. Fix: feed offline_frames() (the fixture's "
                f"own frames), or run live with a real vision backend and its API key."
            ) from None


def _evidence(fixture_id: str, token: str) -> str:
    return (
        f"offline fixture {fixture_id!r}: scripted verdict for frame {token!r} — "
        f"replay of a human-authored file, NOT a perception result"
    )


def _fixture_text(name: str) -> str:
    filename = _FIXTURE_FILES.get(name)
    if filename is None:
        raise OfflineFixtureError(
            f"unknown offline fixture {name!r}; bundled fixtures are {sorted(_FIXTURE_FILES)}."
        )
    from importlib import resources

    try:
        return (resources.files("percept.fixtures") / filename).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - broken install only
        raise OfflineFixtureError(
            f"bundled offline fixture {name!r} ({filename}) is missing from this install — the "
            f"wheel should ship percept/fixtures/*.json. (cause: {exc})"
        ) from exc


def load_fixture(name: str | None = None) -> OfflineFixture:
    """Load a BUNDLED fixture by name (default :data:`DEFAULT_FIXTURE`) off the installed package.

    Read via :mod:`importlib.resources`, so it works from a wheel, a zip, or a source checkout alike.
    """
    name = (name or DEFAULT_FIXTURE).strip()
    raw = json.loads(_fixture_text(name))
    try:
        goal_raw = raw["goal"]
        goal = Goal(id=goal_raw["id"], condition=goal_raw["condition"], say=goal_raw["say"])
        frames = tuple(str(f) for f in raw["frames"])
        fixture_id = str(raw["fixture_id"])
        script = {
            token: Verdict(
                satisfied=bool(v["satisfied"]),
                confidence=float(v["confidence"]),
                evidence=_evidence(fixture_id, token),
            )
            for token, v in raw["script"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise OfflineFixtureError(
            f"bundled offline fixture {name!r} is malformed: {exc}"
        ) from exc
    missing = [f for f in frames if f not in script]
    if missing:
        raise OfflineFixtureError(
            f"offline fixture {name!r} lists frames with no scripted verdict: {sorted(set(missing))}"
        )
    return OfflineFixture(
        fixture_id=fixture_id,
        fixture_version=int(raw.get("fixture_version", 1)),
        description=str(raw.get("description", "")),
        goal=goal,
        frames=frames,
        script=script,
        expect=tuple(dict(e) for e in raw.get("expect", ())),
    )


class FixtureVision:
    """The offline vision seam: replays a bundled :class:`OfflineFixture`, and refuses anything else.

    Deliberately NOT :class:`percept.backends.fake.FakeVision`: the fake answers an unknown frame with
    a low-confidence "unknown frame" verdict (fine for unit tests, wrong for a first-run demo — it
    reads as a model being unsure). This replayer answers ONLY what its fixture scripts and raises
    :class:`OfflineFixtureError` otherwise, so a real stream pointed at offline mode cannot produce a
    single plausible-looking judgement. ``detect``/``detect_events`` return nothing: the fixture
    asserts a gate-visible verdict, never object intelligence.
    """

    def __init__(self, fixture: OfflineFixture | str | None = None) -> None:
        self.fixture = fixture if isinstance(fixture, OfflineFixture) else load_fixture(fixture)

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        return self.fixture.verdict_for(str(frame))

    async def detect(self, frame: Any, goals: list[Goal]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        return []


def offline_frames(fixture: OfflineFixture | str | None = None) -> list[str]:
    """The fixture's own frame sequence, oldest -> newest — what you feed :func:`percept.watch.watch`."""
    fx = fixture if isinstance(fixture, OfflineFixture) else load_fixture(fixture)
    return list(fx.frames)


def offline_goals(fixture: OfflineFixture | str | None = None) -> list[Goal]:
    """The fixture's goal(s), ready to pass to :func:`percept.watch.watch`."""
    fx = fixture if isinstance(fixture, OfflineFixture) else load_fixture(fixture)
    return [fx.goal]


def offline_core(fixture: OfflineFixture | str | None = None, **kwargs: Any):
    """Build a :class:`~percept.agent.Percept` in EXPLICIT ``mode="offline"``, bound to ``fixture``.

    This is the keyless first run: no credential is read, no network call is possible, and the vision
    seam is a fixture replayer (not a general-purpose fake). ``kwargs`` are forwarded to
    :meth:`Percept.create` (``settings``, ``clock``, ``sense_policy``, ...); ``vision`` and ``mode``
    are owned by this function — pass your own core to :func:`percept.watch.watch` if you want other
    wiring.
    """
    from .agent import Percept

    for reserved in ("vision", "mode"):
        if reserved in kwargs:
            raise TypeError(
                f"offline_core() owns {reserved!r} (mode='offline' bound to a bundled fixture). "
                f"Use Percept.create() directly if you need different wiring."
            )
    return Percept.create(vision=FixtureVision(fixture), mode="offline", **kwargs)


def assert_live_or_offline(
    mode: str | None = None,
    *,
    vision: Any = None,
) -> str:
    """Pre-flight the mode contract, returning the resolved mode; raise :class:`ModeError` otherwise.

    The same rule :meth:`Percept.create` enforces, callable BEFORE a front door opens a camera or
    spends a frame: an ONLINE run must name a real vision backend and have its credential; there is
    no implicit fake judge. Offline is always allowed — it is explicit and keyless by construction.
    """
    resolved = resolve_mode(mode)
    if resolved == "offline":
        return resolved
    if vision is None:
        raise ModeError(
            "no vision backend selected and this is a LIVE run — Percept will not silently fall "
            "back to a scripted double. Fix: run mode='offline' (keyless, replays a bundled fixture "
            "— proves WIRING, not intelligence), or select a real backend and set its API key."
        )
    keys = missing_credential(vision)
    if keys:
        raise ModeError(
            f"vision backend {vision!r} needs a credential and none is set. Fix: export one of "
            f"{list(keys)}, or run mode='offline' (keyless bundled fixture)."
        )
    return resolved


__all__ = [
    "DEFAULT_FIXTURE",
    "FIXTURE_BACKEND",
    "FixtureVision",
    "OfflineFixture",
    "OfflineFixtureError",
    "assert_live_or_offline",
    "load_fixture",
    "offline_core",
    "offline_frames",
    "offline_goals",
]
