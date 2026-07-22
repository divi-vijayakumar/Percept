"""P-QUAL-1 (Viability §7/§5a; board card A22) — SCAFFOLD PROBE, G2 judgment quality.

SCAFFOLD-ONLY: wires the probe's own logic against fakes/fixtures and asserts it. Does NOT run a
for-record benchmark (gated on bars-locked — see ``eval/viability/reference_config.yaml``, whose
top-level ``bars_locked: false`` and per-archetype ``recall_min``/``precision_min`` are still
``TODO``). ``RECALL_MIN``/``PRECISION_MIN`` below are PLACEHOLDER parameters that give the QUALITY
assertion a threshold shape to wire NOW; swap in the real, per-archetype, locked numbers once that
file's ``bars_locked: true``.

Property under test (Viability §7/§5a): on the CORE archetype fixtures (card A16's six neutral
archetypes) —

  1. QUALITY — event-level recall/precision, computed via ``golden_eval.score()`` (card A14) over a
     deterministically-scripted ``FakeJudge`` (card A12) verdict pattern, are USABLE and match a
     KNOWN, exactly-computed expected value (not just "close to something").
  2. ERROR-SAFETY — a provider error/timeout injected into the judge NEVER degrades into a
     phantom/stale FIRE; it must resolve to ASK / SILENT / a provider-failure classification.

Reused (imported, never edited):
  * ``eval/viability/fixtures/core_archetypes.py`` (card A16) — the six core archetype fixtures +
    their by-construction ``GroundTruth``.
  * ``percept.harness.golden_eval`` (card A14) — the §5a event-level scorer.
  * ``percept.backends.fake_judge.FakeJudge`` (card A12) — deterministic verdicts + error injection.
  * ``percept.harness.metrics`` / ``percept.clock`` are the other pieces named in this card's reuse
    list; this probe's own timeline math doesn't end up needing live queue-depth tracking or an
    injected clock (every timestamp here is a fixed fixture/offset number, never wall-clock), so
    they are deliberately NOT imported rather than force-fit — noted here so that omission reads as
    a decision, not an oversight.

IMPORTANT COORDINATION (card A30, now landed): a sibling probe (P-JUDGE-SAFE-1, card A23,
``tests/probes/test_p_judge_safe.py``) originally established that driving the REAL
``Percept.perceive_judged()`` judge->gate path with an injected ``malformed``/``partial`` FakeJudge
response CRASHED the run with an uncaught ``AttributeError`` (``percept.gate.classify`` read
``verdict.confidence`` on a plain dict with no shape validation), while ``timeout``/
``provider_error`` raised a TYPED, documented failure (``asyncio.TimeoutError`` / ``ProviderError``)
that nonetheless also propagated uncaught. Card A30 (``Percept._safe_timed_judge`` /
``_JudgeDegraded`` in agent.py) closed that gap: EVERY one of the four modes now degrades THAT TICK
to an ASK/SILENT/provider-failure abstention (``perceive_judged`` returns its normal, empty-of-fires
``list[FireEvent]``, never raises, and the gate is never even touched) instead of crashing the run.

This probe therefore asserts the ERROR-SAFETY property (never phantom/stale FIRE, AND a graceful,
non-crashing resolution) for ALL FOUR modes — ``timeout``/``provider_error`` in
``TestErrorSafetyNeverFires`` (the pre-existing, always-must-pass class) and ``malformed``/
``partial`` in ``TestErrorSafetyGracefulHandlingPendingA30`` below, whose ``@unittest.expectedFailure``
has been REMOVED now that A30 has landed — an XPASS here would have meant this probe was silently
masking a fix instead of confirming it.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any

import pytest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake_judge import ErrorInjectionConfig, FakeJudge, LatencyProfile
from percept.contracts import GroundTruth, JudgmentEvent, JudgmentTiming, Verdict
from percept.harness.golden_eval import GroundTruthEpisode, score

# `eval/viability` is not (yet) an installed package -- same sys.path pattern
# `eval/viability/fixtures/test_core_archetypes.py` itself uses, so `from core_archetypes import
# ...` resolves regardless of pytest's rootdir/import mode.
_VIABILITY_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "eval" / "viability" / "fixtures"
sys.path.insert(0, str(_VIABILITY_FIXTURES_DIR))

# `eval/viability/` is intentionally git-untracked and gitignored (see commit cc46b37,
# "untrack internal product-strategy docs before any public push") — it holds the founder
# bars / GO-NOGO memo / quality-gate protocol and is kept LOCAL-ONLY on machines that have run
# the viability harness, never shipped or checked into the public repo. On a clean checkout (and
# in CI) `core_archetypes.py` genuinely does not exist, so this whole scaffold — which reuses
# those fixtures verbatim, by design (see module docstring) — is unimportable and must skip
# cleanly rather than error at collection. When the fixture IS present (e.g. locally, with the
# viability harness checked out alongside the repo), the tests run for real and assert exactly
# what they always did.
pytest.importorskip(
    "core_archetypes",
    reason="eval/viability/fixtures/core_archetypes.py is git-untracked/gitignored (internal "
    "viability harness, kept local-only) — not present on a clean checkout/CI",
)

from core_archetypes import build_core_archetype_fixtures  # noqa: E402


# ---------------------------------------------------------------------------
# Part 1 -- QUALITY: known-correctness-pattern scoring via golden_eval.score()
# ---------------------------------------------------------------------------

# A generous, fixed separation between fixtures' own LOCAL timelines (each independently starts
# near t=0) projected onto one shared GLOBAL timeline, so golden_eval.score()'s window checks for
# one fixture's episode never accidentally collide with another, unrelated fixture's nearby local
# timestamps. Far larger than any fixture's own span (< 5s) plus allowed lateness.
_OFFSET_STEP_S = 1_000.0
_ALLOWED_LATENESS_S = 0.5

# The ONE known-wrong verdict per class, chosen so the resulting recall/precision are non-trivial
# (neither 1.0 nor 0.0) AND exactly, analytically known -- not just "high enough".
_KNOWN_MISS_FIXTURE_ID = "persistence_door_held_open"  # real positive -> scripted False (a miss)
_KNOWN_FALSE_FIRE_FIXTURE_ID = "transition_touch_without_release"  # real negative -> scripted True

# PLACEHOLDER quality-gate parameters. The REAL per-archetype recall_min/precision_min numbers live
# in eval/viability/reference_config.yaml under numeric_bars, and are still TODO there with
# bars_locked: false -- this probe is scaffold-only (not a for-record run) and is not gated on
# those real bars. These placeholders exist so the QUALITY assertion has a threshold shape to wire
# NOW; swap in the locked, per-archetype values once that file's bars_locked: true.
RECALL_MIN = 0.75
PRECISION_MIN = 0.75


def _offsets_for(fixtures) -> dict[str, float]:
    """A deterministic fixture_id -> global-time-offset map (alphabetical fixture_id order, so two
    calls over the same fixture set always produce the identical assignment)."""
    ordered = sorted(fixtures, key=lambda f: f.fixture_id)
    return {f.fixture_id: i * _OFFSET_STEP_S for i, f in enumerate(ordered)}


def _episode_for(fixture, offset: float) -> GroundTruthEpisode:
    gt = fixture.ground_truth
    assert gt is not None
    return GroundTruthEpisode(
        episode_id=fixture.fixture_id,
        ground_truth=GroundTruth(
            event_start_ts=gt.event_start_ts + offset,
            decision_eligible_ts=gt.decision_eligible_ts + offset,
            annotation_version=gt.annotation_version,
            annotator_or_fixture_id=gt.annotator_or_fixture_id,
        ),
        episode_end_ts=fixture.timeline[-1].t + offset,
        allowed_lateness_s=_ALLOWED_LATENESS_S,
    )


def _scripted_verdict_satisfied(fixture) -> bool:
    """The KNOWN correctness pattern this probe scores against: every fixture's judge verdict
    matches its ground truth (satisfied <-> positive) EXCEPT the two named exceptions above, which
    are deliberately scripted WRONG so the resulting recall/precision are non-trivial and exactly
    computable (see the module-level constants)."""
    if fixture.fixture_id == _KNOWN_MISS_FIXTURE_ID:
        return False
    if fixture.fixture_id == _KNOWN_FALSE_FIRE_FIXTURE_ID:
        return True
    return fixture.positive


def _judgment_event_for(fixture, offset: float, satisfied: bool) -> JudgmentEvent | None:
    """The predicted JudgmentEvent a `satisfied=True` verdict emits ("fire"); `None` for
    `satisfied=False` (nothing is predicted -- a silent tick emits no event that
    `golden_eval.score()`'s default `predicted_actions={"fire"}` would count anyway)."""
    if not satisfied:
        return None
    gt = fixture.ground_truth
    # A correct positive fires at its own known-eligible instant; the one deliberately-wrong
    # negative (the known false fire) fires at its own last-frame instant instead -- there is no
    # ground truth to anchor it to, it is a genuine false fire by construction.
    ts = (gt.decision_eligible_ts if gt is not None else fixture.timeline[-1].t) + offset
    timing = JudgmentTiming(
        ingest_observed_ts=ts,
        admitted_ts=ts,
        judge_started_ts=ts,
        judge_completed_ts=ts,
        decision_emitted_ts=ts,
    )
    return JudgmentEvent(key=fixture.fixture_id, evidence_seq=0, action="fire", timing=timing)


def _quality_script(fixtures) -> dict[str, Verdict]:
    return {
        f.fixture_id: Verdict(
            satisfied=_scripted_verdict_satisfied(f),
            confidence=0.9,
            evidence=f"scripted:{f.fixture_id}",
        )
        for f in fixtures
    }


async def _drive_fixtures_through_fake_judge(fixtures, judge: FakeJudge) -> dict[str, bool]:
    """Calls `FakeJudge.judge()` once per fixture (condition=archetype name, frame=fixture_id --
    FakeVision keys verdicts off `str(frame)`) and returns `{fixture_id: verdict.satisfied}`."""
    out: dict[str, bool] = {}
    for f in fixtures:
        verdict = await judge.judge(f.archetype, f.fixture_id)
        out[f.fixture_id] = verdict.satisfied
    return out


class TestQualityRecallPrecision(unittest.TestCase):
    """Drives the six core archetype fixtures through a deterministic FakeJudge scripted to a KNOWN
    correctness pattern, scores the result with `golden_eval.score()`, and asserts the recovered
    event-level recall/precision equal the analytically-known expected value."""

    def test_recall_and_precision_match_known_expected_value(self) -> None:
        fixtures = build_core_archetype_fixtures()
        positives = [f for f in fixtures if f.positive]
        self.assertEqual(len(positives), 6, "expected exactly one positive fixture per core archetype")

        offsets = _offsets_for(fixtures)
        episodes = tuple(_episode_for(f, offsets[f.fixture_id]) for f in positives)

        judge = FakeJudge(seed=0, script=_quality_script(fixtures), simulate_sleep=False)
        satisfied_by_id = asyncio.run(_drive_fixtures_through_fake_judge(fixtures, judge))

        predicted = [
            ev
            for f in fixtures
            if (ev := _judgment_event_for(f, offsets[f.fixture_id], satisfied_by_id[f.fixture_id]))
            is not None
        ]

        result = score(episodes, predicted)

        # Known, exact expected values: 5 of the 6 positives are correctly matched (one scripted
        # miss), and exactly 1 false fire (the one scripted false positive) alongside those 5 true
        # positives -- recall = 5/6 matched episodes, precision = 5/6 correct-of-total predictions.
        expected_recall = 5 / 6
        expected_precision = 5 / 6
        self.assertAlmostEqual(result.recall, expected_recall, places=9)
        self.assertAlmostEqual(result.precision, expected_precision, places=9)
        self.assertEqual(result.true_positives, 5)
        self.assertEqual(result.misses, 1)
        self.assertEqual(result.missed_episode_ids, (_KNOWN_MISS_FIXTURE_ID,))
        self.assertEqual(result.unmatched_false_fires, 1)
        self.assertEqual(result.duplicate_false_fires, 0)

        # The quality-gate shape this probe wires in for later: recall/precision must clear
        # RECALL_MIN/PRECISION_MIN (placeholders -- see module docstring/constants above; the real
        # per-archetype bars come from eval/viability/reference_config.yaml once bars_locked).
        self.assertGreaterEqual(result.recall, RECALL_MIN, "recall fell below the (placeholder) quality bar")
        self.assertGreaterEqual(
            result.precision, PRECISION_MIN, "precision fell below the (placeholder) quality bar"
        )


# ---------------------------------------------------------------------------
# Part 2 -- ERROR-SAFETY: injected judge failures must never reach FIRE
# ---------------------------------------------------------------------------

# The two modes that mirror a REAL backend's failure contract (a typed, documented
# exception -- see the module docstring's A30 coordination note).
_SAFE_FAILURE_MODES: tuple[str, ...] = ("timeout", "provider_error")
# The two modes that currently CRASH unclassified (agent.py:941, pending card A30) -- NOT asserted
# safe here; see `TestErrorSafetyGracefulHandlingPendingA30` below.
_UNSAFE_FAILURE_MODES: tuple[str, ...] = ("malformed", "partial")


class _ProbeVision:
    """Adapts a judge-only double (FakeJudge) to the full `percept.contracts.Vision` protocol
    `Percept.create()` expects -- the same adapter shape `tests/probes/test_p_judge_safe.py` (card
    A23) already established; reproduced here (not imported) since it is test-local wiring, not
    production code. `detect`/`detect_events` are no-ops: this probe only cares what a `judge()`
    failure does to the gate."""

    def __init__(self, judge_double: FakeJudge) -> None:
        self._judge_double = judge_double

    async def judge(self, *args: Any, **kwargs: Any):
        return await self._judge_double.judge(*args, **kwargs)

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []


def _build_error_agent(mode: str, condition: str, *, seed: int = 0) -> tuple[Percept, FakeJudge]:
    """One watch goal built off a CORE ARCHETYPE fixture's own description, wired to a FakeJudge
    that ALWAYS fails with `mode` (`rate=1.0`) -- deterministic, key-free, zero real vendor
    traffic, driving the REAL `Percept.perceive_judged()` judge->gate path (agent.py:817-960,
    unedited by this card)."""
    judge = FakeJudge(
        seed=seed,
        latency=LatencyProfile(kind="fixed", value_s=0.0),
        errors=ErrorInjectionConfig(rate=1.0, modes=(mode,)),
        simulate_sleep=False,
    )
    agent = Percept.create(vision=_ProbeVision(judge), discover_plugins=False)
    agent.add_goal(Goal(id="g", condition=condition, say=f"{condition}!"))
    return agent, judge


async def _drive_one_frame(agent: Percept, frame_token: str, *, t: float = 1.0):
    """Feeds exactly one frame through the REAL production judge->gate path. Returns
    `(events, exc)`: on a normal return, `events` is the `list[FireEvent]` and `exc` is `None`; on a
    raised exception, `events` is `None` and `exc` is the exception instance."""
    try:
        events = await agent.perceive_judged(frame_token, t=t)
        return events, None
    except Exception as exc:  # noqa: BLE001 -- deliberately observes ANY exception shape
        return None, exc


def _positive_fixtures_by_archetype() -> dict[str, Any]:
    return {f.archetype: f for f in build_core_archetype_fixtures() if f.positive}


class TestErrorSafetyNeverFires(unittest.TestCase):
    """THE hard safety property this probe's error-safety half exists to prove: whatever a judge
    backend raises on the six core archetype fixtures' own conditions, the real judge->gate path
    must never emit a FIRE, and must not leave the gate's edge state mutated (no
    stale-armed-then-a-later-clean-call-fires-wrongly residue). Covers `timeout`/`provider_error` --
    the two modes that resolve to a TYPED, documented provider-failure exception (see module
    docstring's A30 coordination note). MUST pass; a failure here is release-blocking."""

    def test_never_fires_for_any_core_archetype_under_timeout_or_provider_error(self) -> None:
        fixtures = _positive_fixtures_by_archetype()
        for mode in _SAFE_FAILURE_MODES:
            for archetype, fixture in fixtures.items():
                with self.subTest(mode=mode, archetype=archetype):
                    agent, _judge = _build_error_agent(mode, fixture.description)
                    events, exc = asyncio.run(_drive_one_frame(agent, fixture.fixture_id))
                    if events is not None:
                        fired = [e for e in events if e.action == "fire"]
                        self.assertEqual(
                            fired,
                            [],
                            f"RELEASE-BLOCKING: {archetype!r} under injected {mode!r} reached FIRE: "
                            f"{fired!r} -- a timed-out/erroring judge response must NEVER fire.",
                        )
                    else:
                        self.assertIsNotNone(
                            exc,
                            f"{archetype!r}/{mode!r}: perceive_judged returned neither events nor "
                            f"an exception",
                        )

    def test_gate_edge_state_unmutated_under_timeout_or_provider_error(self) -> None:
        """Belt-and-suspenders 'never STALE' check: an injected failure must not silently arm/disarm
        the rising edge either -- that would let a LATER clean call fire/suppress incorrectly
        because of a fault that should never have touched gate state at all."""
        fixtures = _positive_fixtures_by_archetype()
        for mode in _SAFE_FAILURE_MODES:
            for archetype, fixture in fixtures.items():
                with self.subTest(mode=mode, archetype=archetype):
                    agent, _judge = _build_error_agent(mode, fixture.description)
                    before = agent.gate.snapshot()
                    asyncio.run(_drive_one_frame(agent, fixture.fixture_id))
                    after = agent.gate.snapshot()
                    self.assertEqual(before, after, f"{archetype!r}/{mode!r}: gate edge state changed")


class TestErrorSafetyGracefulHandlingPendingA30(unittest.TestCase):
    """NOT the hard safety invariant (that holds structurally today for ALL four modes -- see
    `TestErrorSafetyNeverFires` and the module docstring's A30 coordination note). This class
    asserts the DESIRED end state -- a graceful ASK/SILENT/provider-failure classification, never an
    uncaught crash -- for `malformed`/`partial`. Card A30 (the fail-closed-boundary fix,
    `Percept._safe_timed_judge`) has LANDED, so the `@unittest.expectedFailure` has been dropped and
    this now asserts (and holds) the graceful end state directly."""

    def test_malformed_and_partial_resolve_gracefully_after_a30(self) -> None:
        # One representative archetype is enough here -- card A23's own probe already proved this
        # crash exhaustively (all four failure modes, end to end); this scaffold only needs to keep
        # the gap visible and self-checking, not re-prove it per archetype.
        fixture = _positive_fixtures_by_archetype()["state"]
        for mode in _UNSAFE_FAILURE_MODES:
            agent, _judge = _build_error_agent(mode, fixture.description)
            events, exc = asyncio.run(_drive_one_frame(agent, fixture.fixture_id))
            self.assertIsNone(
                exc,
                f"{mode!r}: expected a graceful ASK/SILENT/provider-failure classification (no "
                f"uncaught exception), got {exc!r} -- see card A30 (fail-closed-boundary fix)",
            )
            fired = [e for e in (events or []) if e.action == "fire"]
            self.assertEqual(fired, [])


if __name__ == "__main__":
    unittest.main()
