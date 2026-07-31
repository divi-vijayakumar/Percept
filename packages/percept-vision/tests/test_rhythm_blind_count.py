"""EXTERNAL_RHYTHM's honesty tier — when a VLM-free count must admit it is BLIND.

``EXTERNAL_RHYTHM`` is the one behavior route that reports a number to the user without ever
spending a judge: the eye measures channels, ``RhythmTracker`` counts prominence-gated excursions,
and nothing downstream ever looks at a pixel again. That makes it the cheapest route to run and the
easiest one to be confidently wrong on, because the counter has no notion of "this signal was not a
rhythm at all" — it counts whatever excursions clear the prominence gate. Fed pure noise it does not
refuse, it returns a plausible-looking integer.

``rhythm_hero.escalate_count`` is the design's answer: trust the edge count while its periodicity
``confidence`` clears ``tau``, otherwise call the count BLIND (occluded / non-periodic / wrong
subject) and defer to a backstop. The existing suite proves that policy as arithmetic — it is called
with literal confidences of ``0.1`` and ``0.9``. What was never pinned is the part that decides
whether it ever fires in production: **that a genuinely blind stream actually measures below tau,
and a real rhythm above it.** A separation that does not hold turns the whole tier into decoration,
and the counter reports a fabricated number as clean. That is the failure this file exists to catch.

Everything here is a synthetic scalar stream — a sine for a rhythm, seeded Gaussian noise for the
blind case — so there is no media, no provider, no key and no numpy: the honesty contract is
asserted on the pure-stdlib measurement layer the route already ships.

Also pinned here, because nothing else pins them: the route's refusal edges. ``perceive_rhythm`` is
a public entry point that silently returns ``None`` for an unknown goal or an empty channel vector,
and a caller has to be able to rely on those being *inert* rather than half-applied.
"""

from __future__ import annotations

import math
import random
import unittest

from percept import Goal, Percept
from percept.behavior_routes import (
    BehaviorRoute,
    drives_rhythm,
    goal_behavior_route,
    needs_watch_condition,
    worker_can_drive,
)
from percept.harness.detectors import PersonMeasure
from percept.measurement import MeasurementSeries, periodicity_score
from percept.rhythm import RhythmTracker
from percept.rhythm_hero import RhythmHero, escalate_count

FPS = 6.0
CHANNEL = "limb-vertical"
PERIOD_SAMPLES = 12  # 12 samples at 6 fps = a 2 s rep, comfortably inside BAND_S
N_SAMPLES = 72  # 12 s of stream = 6 clean reps
TAU = 0.4  # escalate_count's shipped default confidence threshold
BAND_S = (0.3, 5.0)  # the tracker's own plausible rep-period band (RhythmTracker.select_band_s)
CANON_HZ = 30.0  # score every stream on one canonical grid, per measurement.periodicity_score


def _rhythm_state(**extra) -> dict:
    """The armed (apply-coerced) state that declares the route: flat ``type``, no nested block."""

    state = {"type": "rhythm", "count": 0, "rhythm": {"channel": CHANNEL}}
    state.update(extra)
    return state


def _sine(n: int = N_SAMPLES, period: int = PERIOD_SAMPLES) -> list[float]:
    """A clean rhythm: ``n / period`` complete rest-peak-rest cycles."""

    return [math.sin(2 * math.pi * i / period) for i in range(n)]


def _noise(seed: int, n: int = N_SAMPLES) -> list[float]:
    """Broadband Gaussian noise — the blind case with the largest counting error.

    Not "a quiet stream": it has full amplitude, so every prominence gate in the counter passes and
    the tracker confirms reps. It simply has no cycle, so any count it produces is fabricated.
    """

    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n)]


def _core() -> Percept:
    core = Percept.create(mode="offline", discover_plugins=False)
    core.add_goal(
        Goal(id="reps", condition="a completed rep", say="rep", state=_rhythm_state())
    )
    return core


def _observe(core: Percept, values, *, goal_id: str = "reps"):
    """Drive the route's sense entry point one frame at a time; return the fires it emitted."""

    return [
        fire
        for i, v in enumerate(values)
        if (fire := core.perceive_rhythm(goal_id, t=i / FPS, channels={CHANNEL: v})) is not None
    ]


def _hero_result(values):
    """The same stream through ``RhythmHero`` — the producer that publishes the observability.

    ``PersonMeasure`` is built directly rather than decoded from pixels: the association seam is
    already covered elsewhere, and what this file needs is the ``HeroResult`` a confidence is read
    from, without pulling in a pose model or a frame buffer.
    """

    core = _core()
    hero = RhythmHero(core, "reps", prior_channel=CHANNEL)
    for i, v in enumerate(values):
        measure = PersonMeasure(
            box=(0.2, 0.2, 0.1, 0.4),
            channels={CHANNEL: v},
            pose_conf=0.9,
            coverage={CHANNEL: 0.95},
            person_id="0",
        )
        hero.observe(i / FPS, [measure], frame=None)
    return hero.result()


def _confidence(signal) -> float:
    """The periodicity confidence of a performer's channel stream, in ``[0, 1]``.

    This is what ``HeroResult.signal`` is published for — its own field comment says the stream is
    carried "so a periodicity confidence can be measured downstream". ``periodicity_score`` is the
    package's canonical answer to "how periodic is this", it is pure stdlib, and it is rate-invariant
    by construction, so it is the measure a keyless deployment can actually feed ``escalate_count``.
    """

    return periodicity_score(
        MeasurementSeries.from_pairs(CHANNEL, signal), band_s=BAND_S, hz=CANON_HZ
    )


class RhythmRouteSelectionTests(unittest.TestCase):
    """Which goals reach the counting driver at all — the precondition for everything below."""

    def test_a_rhythm_goal_selects_the_measured_route_and_never_compiles_a_watch_condition(self):
        # The route's defining property: the frame worker drives it, but it is MEASURED, not judged,
        # so it must never be handed to tier-0 as an admission condition. If it ever needed one, the
        # "counts without spending a judge" guarantee this file's honesty tier protects would be gone.
        goal = Goal(id="reps", condition="a completed rep", say="rep", state=_rhythm_state())
        route = goal_behavior_route(goal)
        self.assertIs(route, BehaviorRoute.EXTERNAL_RHYTHM)
        self.assertTrue(worker_can_drive(route))
        self.assertTrue(drives_rhythm(route))
        self.assertFalse(needs_watch_condition(route))

    def test_the_neighbouring_counting_types_do_not_reach_the_rhythm_driver(self):
        # 'count' also produces a tally, and 'watch' is the default authored type, so both are the
        # plausible mis-selections. Neither may be driven by the rhythm counter: a 'count' goal is
        # fed by the host one boundary at a time and has no channel stream to measure periodicity on.
        for authored, expected in (
            ("count", BehaviorRoute.EXTERNAL_BOUNDARY),
            ("watch", BehaviorRoute.FRAME_WATCH),
            ("event", BehaviorRoute.EXTERNAL_EVENT_WINDOW),
        ):
            with self.subTest(authored=authored):
                goal = Goal(id="g", condition="c", say="s", state={"type": authored})
                route = goal_behavior_route(goal)
                self.assertIs(route, expected)
                self.assertFalse(drives_rhythm(route))


class RhythmCountsAnySignalTests(unittest.TestCase):
    """The counter does not refuse a non-rhythm — which is precisely why the blind tier is needed."""

    def test_a_clean_periodic_stream_counts_every_complete_rep(self):
        # The control arm for the blind comparison below: with a real rhythm the count is exact, so a
        # wrong count on noise cannot be blamed on the counter being loose in general.
        core = _core()
        _observe(core, _sine())
        self.assertEqual(core.rhythm["reps"].count, N_SAMPLES // PERIOD_SAMPLES)
        self.assertEqual(core.goals.get("reps").state["count"], core.rhythm["reps"].count)

    def test_a_non_periodic_stream_is_counted_anyway_and_looks_identical_from_the_count(self):
        # The honest fact this route ships with: fed noise, the counter returns a number in exactly
        # the range a real set of reps would produce. Nothing in the count, the goal state or the
        # edge tally distinguishes it. A caller reading only `count` cannot tell the two apart, which
        # is the entire reason a separate confidence measure has to exist.
        real = _core()
        _observe(real, _sine())
        for seed in range(4):
            with self.subTest(seed=seed):
                blind = _core()
                _observe(blind, _noise(seed))
                fabricated = blind.rhythm["reps"].count
                self.assertGreater(fabricated, 0)  # never refuses, never returns 0
                self.assertEqual(blind.goals.get("reps").state["count"], fabricated)
                self.assertLess(abs(fabricated - real.rhythm["reps"].count), 5)


class BlindCountEscalationTests(unittest.TestCase):
    """The keystone: a blind count must MEASURE as blind, not merely be escalatable in principle."""

    def test_a_real_rhythm_measures_confident_and_keeps_the_edge_count(self):
        result = _hero_result(_sine())
        confidence = _confidence(result.signal)
        self.assertGreaterEqual(confidence, TAU)

        def recount():  # pragma: no cover - a confident count must not reach the backstop
            raise AssertionError("a confident count must never spend the backstop")

        count, escalated = escalate_count(confidence, result.count, recount, tau=TAU)
        self.assertFalse(escalated)
        self.assertEqual(count, result.count)
        self.assertEqual(count, N_SAMPLES // PERIOD_SAMPLES)

    def test_a_blind_count_measures_low_and_defers_to_the_backstop(self):
        result = _hero_result(_noise(seed=0))
        self.assertGreater(result.count, 0)  # the fabricated count the user would otherwise be told
        calls = []

        def recount():
            calls.append(1)
            return -1  # a sentinel the edge counter could never produce

        count, escalated = escalate_count(_confidence(result.signal), result.count, recount, tau=TAU)
        self.assertTrue(escalated)
        self.assertEqual(count, -1)  # the reported number came from the backstop, not the edge
        self.assertEqual(len(calls), 1)

    def test_the_confidence_separates_rhythm_from_noise_on_every_seed(self):
        # One lucky seed proves nothing — a threshold is only a threshold if the two populations do
        # not overlap it. Pinned across a seed sweep with the observed margin on both sides, so a
        # regression that merely narrows the gap fails here rather than in a client's rep count.
        rhythm_confidence = _confidence(_hero_result(_sine()).signal)
        blind = [_confidence(_hero_result(_noise(seed)).signal) for seed in range(20)]
        self.assertGreater(rhythm_confidence, 0.6)
        self.assertLess(max(blind), TAU)
        self.assertGreater(rhythm_confidence - max(blind), 0.25)

    def test_the_hero_publishes_the_stream_the_confidence_is_measured_from(self):
        # The wiring the separation depends on: `signal` must be the performer's own per-frame
        # channel stream as timestamped pairs, complete and in order. A truncated or index-keyed
        # stream would still score, just meaninglessly, and the tier would fail silently.
        result = _hero_result(_sine())
        self.assertEqual(len(result.signal), N_SAMPLES)
        self.assertEqual(result.prior_channel, CHANNEL)
        self.assertEqual(result.n_entities, 1)
        times = [t for t, _ in result.signal]
        self.assertEqual(times, sorted(times))
        self.assertAlmostEqual(times[-1] - times[0], (N_SAMPLES - 1) / FPS, places=6)
        self.assertGreater(result.hero_track_coverage, 0.9)
        self.assertGreater(result.channel_coverage, 0.9)

    @unittest.expectedFailure
    def test_the_core_native_regularity_escalates_a_blind_count(self):
        """The obvious composition, which does not work — expected to FAIL until the measure changes.

        ``HeroResult.regularity`` is the only confidence-shaped scalar the route itself computes, it
        sits in the dataclass whose docstring says a count scoring low on these fields is blind, and
        it is documented in-field as "core-native periodicity". A deployment wiring the honesty tier
        with what it has therefore writes ``escalate_count(result.regularity, result.count, ...)``
        and gets a counter that never escalates.

        It cannot escalate: ``regularity`` is ``1 / (1 + CV)`` over *confirmed inter-rep intervals*,
        and those intervals survive a prominence gate that already discards the irregular excursions,
        so their coefficient of variation stays well under 1 even on pure noise. The value is
        effectively floored around 0.5, while ``tau`` defaults to 0.4 — reaching it would require a
        CV above 1.5. Measured over 20 noise seeds the minimum is 0.62 and the branch fires 0 times,
        each on a count of 3 to 7 reps fabricated from a signal with no cycle.

        The route's usable confidence is measured from ``HeroResult.signal`` instead (see
        ``_confidence``), which is what the tests above pin. When ``regularity`` is made a real
        periodicity measure, or the field is renamed so it cannot be mistaken for one, this test
        starts passing and is reported as an unexpected success — delete the marker then.
        """

        for seed in range(20):
            result = _hero_result(_noise(seed))
            _, escalated = escalate_count(
                result.regularity, result.count, lambda: -1, tau=TAU
            )
            self.assertTrue(escalated, f"seed {seed}: regularity {result.regularity:.4f} >= {TAU}")


class RhythmRefusalEdgeTests(unittest.TestCase):
    """The inert cases. Each returns ``None``; each must also leave no half-applied state behind."""

    def test_an_unknown_goal_id_is_inert_and_mints_no_tracker(self):
        # perceive_rhythm is public and takes a bare string, so a typo'd or already-disarmed goal id
        # is reachable. It must not lazily create a tracker: a tracker keyed on a goal that does not
        # exist would accumulate forever and be readable as a count nobody armed.
        core = _core()
        self.assertIsNone(core.perceive_rhythm("not-a-goal", t=0.0, channels={CHANNEL: 1.0}))
        self.assertIsNone(core.rhythm.get("not-a-goal"))
        self.assertNotIn("not-a-goal", core.edge_counts)

    def test_an_empty_channel_vector_leaves_the_goal_untouched(self):
        # The eye legitimately measures nothing on a frame with no subject in it. That frame must be
        # a no-op, not a zero sample: folding an implicit 0.0 into the tracker would inject a
        # phantom trough and let an occlusion gap manufacture a rep.
        core = _core()
        before = dict(core.goals.get("reps").state)
        for i in range(10):
            self.assertIsNone(core.perceive_rhythm("reps", t=i / FPS, channels={}))
        self.assertIsNone(core.rhythm.get("reps"))
        self.assertEqual(core.goals.get("reps").state, before)
        self.assertNotIn("reps", core.edge_counts)

    def test_an_empty_stream_leaves_no_tracker_and_no_count(self):
        core = _core()
        self.assertEqual(_observe(core, []), [])
        self.assertIsNone(core.rhythm.get("reps"))
        self.assertEqual(core.goals.get("reps").state["count"], 0)

    def test_a_flat_stream_counts_nothing_and_reports_no_confidence(self):
        # A still subject: no channel is ever selected, so there is nothing to be confident about.
        # The confidence must read 0.0 (deferring), not the 1.0 that a "perfectly regular because
        # there are no intervals to disagree" reading would give.
        tracker = RhythmTracker()
        for i in range(30):
            self.assertIsNone(tracker.observe(i / FPS, {CHANNEL: 0.0}))
        self.assertIsNone(tracker.selected)
        self.assertEqual(tracker.count, 0)
        self.assertEqual(tracker.regularity, 0.0)

    def test_a_single_rep_is_not_yet_confident(self):
        # One rep gives one endpoint and zero intervals. The tracker must report 0.0 rather than the
        # 1.0 its per-channel helper returns for "fewer than two reps", so the first rep of a set is
        # never handed downstream as a confident measurement.
        tracker = RhythmTracker()
        for i, v in enumerate(_sine(n=PERIOD_SAMPLES + 4)):
            tracker.observe(i / FPS, {CHANNEL: v})
        self.assertEqual(tracker.count, 1)
        self.assertEqual(tracker.regularity, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
