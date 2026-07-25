"""VID interval lifecycle: deterministic boundaries over scalar/tier-0 interest signals."""

from __future__ import annotations

import random
import unittest

from percept import (
    INTEREST_ENTERED,
    INTEREST_EXITED,
    INTEREST_UPDATED,
    Box,
    Goal,
    InterestIntervalConfig,
    InterestIntervalEngine,
    InterestSample,
    Tier0MotionInterest,
)
from percept.edge.tier0 import GatedWatchLoop, gray_frame


def _sample(
    t: float,
    score: float,
    *,
    source_at: float | None = None,
    reason: str = "synthetic",
    confidence: float | None = 0.8,
    coverage: float = 1.0,
    regions: tuple[Box, ...] = (),
    track_ids: tuple[str, ...] = (),
) -> InterestSample:
    return InterestSample(
        source_at=t if source_at is None else source_at,
        observed_at=t,
        score=score,
        reason=reason,
        confidence=confidence,
        coverage=coverage,
        regions=regions,
        track_ids=track_ids,
    )


def _engine(
    *,
    enter: float = 0.7,
    exit_: float = 0.3,
    enter_debounce: int = 1,
    exit_debounce: int = 1,
    expiry_s: float | None = None,
) -> InterestIntervalEngine:
    return InterestIntervalEngine(
        source="camera:front",
        admission_id="tier0.signal.motion",
        detector_id="motion_score.mean",
        config=InterestIntervalConfig(
            enter_threshold=enter,
            exit_threshold=exit_,
            enter_debounce=enter_debounce,
            exit_debounce=exit_debounce,
            expiry_s=expiry_s,
        ),
    )


class InterestIntervalTableTests(unittest.TestCase):
    def test_hysteresis_and_debounce_boundaries(self) -> None:
        engine = _engine(enter_debounce=2, exit_debounce=2)
        rows = [
            (0.0, 0.69, []),
            (1.0, 0.70, []),
            (2.0, 0.80, [INTEREST_ENTERED]),
            (3.0, 0.50, [INTEREST_UPDATED]),  # hysteresis band stays active
            (4.0, 0.30, []),
            (5.0, 0.20, [INTEREST_EXITED]),
        ]
        events = []
        for t, score, kinds in rows:
            emitted = engine.observe(_sample(t, score))
            self.assertEqual([event.kind for event in emitted], kinds)
            events.extend(emitted)

        self.assertEqual([event.interval_id for event in events], [events[0].interval_id] * 3)
        self.assertEqual((events[0].source_start, events[0].source_end), (1.0, 2.0))
        # Exit debounce confirms at t=5, but no positive evidence extends past t=3.
        self.assertEqual((events[-1].observed_end, events[-1].event_observed_at), (3.0, 5.0))
        self.assertEqual(events[-1].reason, "signal_lost")
        self.assertGreaterEqual(events[-1].duration, 0.0)

    def test_thresholds_are_caller_selected_and_hysteretic(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower"):
            InterestIntervalConfig(enter_threshold=0.4, exit_threshold=0.4)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            InterestIntervalConfig(enter_threshold=1.1, exit_threshold=0.2)
        with self.assertRaisesRegex(ValueError, "integer"):
            InterestIntervalConfig(
                enter_threshold=0.7,
                exit_threshold=0.3,
                enter_debounce=1.5,  # type: ignore[arg-type]
            )

    def test_region_and_track_metadata_is_validated_and_stable(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalized box"):
            _sample(0.0, 1.0, regions=(Box(x=0.9, y=0.0, w=0.2, h=0.2),))

        engine = _engine()
        regions = (
            Box(x=0.5, y=0.5, w=0.2, h=0.2),
            Box(x=0.1, y=0.1, w=0.2, h=0.2),
        )
        entered = engine.observe(
            _sample(0.0, 0.9, regions=regions, track_ids=("track-b", "track-a", "track-a"))
        )[0]
        self.assertEqual(
            entered.regions,
            (
                Box(x=0.1, y=0.1, w=0.2, h=0.2),
                Box(x=0.5, y=0.5, w=0.2, h=0.2),
            ),
        )
        self.assertEqual(entered.track_ids, ("track-a", "track-b"))
        # Missing metadata on a later sample preserves the latest available interval metadata.
        updated = engine.observe(_sample(1.0, 0.5, regions=(), track_ids=()))[0]
        self.assertEqual(updated.regions, entered.regions)
        self.assertEqual(updated.track_ids, entered.track_ids)

    def test_invalid_or_out_of_order_input_cannot_mutate_identity(self) -> None:
        invalid_then_valid = _engine()
        with self.assertRaisesRegex(ValueError, "normalized box"):
            invalid_then_valid.observe(
                _sample(0.0, 0.9, regions=(Box(x=-0.1, y=0.0, w=0.2, h=0.2),))
            )
        first = invalid_then_valid.observe(_sample(0.0, 0.9))[0]

        clean = _engine()
        replay_first = clean.observe(_sample(0.0, 0.9))[0]
        self.assertEqual(first.interval_id, replay_first.interval_id)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            invalid_then_valid.observe(_sample(-1.0, 0.9))
        self.assertTrue(invalid_then_valid.active)
        self.assertEqual(invalid_then_valid.interval_id, first.interval_id)

    def test_gap_merge_and_expiry_split_policy(self) -> None:
        engine = _engine(expiry_s=3.0)
        entered = engine.observe(_sample(0.0, 0.9))[0]
        merged = engine.observe(_sample(2.0, 0.8))[0]
        self.assertEqual(merged.kind, INTEREST_UPDATED)
        self.assertEqual(merged.interval_id, entered.interval_id)

        # Exactly at the expiry boundary: old interval exits, recurring signal opens a new interval.
        split = engine.observe(_sample(5.0, 0.9))
        self.assertEqual([event.kind for event in split], [INTEREST_EXITED, INTEREST_ENTERED])
        self.assertEqual(split[0].reason, "expired")
        self.assertEqual(split[0].event_observed_at, 5.0)
        self.assertNotEqual(split[0].interval_id, split[1].interval_id)

    def test_expiry_gap_cannot_bridge_partial_enter_debounce(self) -> None:
        engine = _engine(enter_debounce=2, expiry_s=3.0)
        self.assertEqual(engine.observe(_sample(0.0, 0.9)), [])
        # The first high sample is stale at t=3; this is a fresh first debounce sample, not an enter.
        self.assertEqual(engine.observe(_sample(3.0, 0.9)), [])
        entered = engine.observe(_sample(4.0, 0.9))[0]
        self.assertEqual(entered.observed_start, 3.0)

    def test_polling_advance_does_not_push_the_expiry_deadline(self) -> None:
        engine = _engine(expiry_s=3.0)
        entered = engine.observe(_sample(0.0, 0.9))[0]
        self.assertEqual(engine.advance(observed_at=1.0), [])
        self.assertEqual(engine.advance(observed_at=2.0), [])
        expired = engine.advance(observed_at=3.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].interval_id, entered.interval_id)
        self.assertEqual(expired[0].reason, "expired")
        self.assertEqual(expired[0].event_observed_at, 3.0)

    def test_explicit_expiry_without_configured_idle_timeout(self) -> None:
        engine = _engine()
        entered = engine.observe(_sample(1.0, 0.9))[0]
        exited = engine.expire(observed_at=4.0)
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0].interval_id, entered.interval_id)
        self.assertEqual(exited[0].reason, "expired")
        self.assertEqual(exited[0].observed_end, 1.0)
        self.assertEqual(engine.expire(observed_at=5.0), [])

    def test_reconnect_closes_and_allows_source_clock_restart(self) -> None:
        engine = _engine()
        first = engine.observe(_sample(100.0, 0.9, source_at=10.0))[0]
        reconnect = engine.reconnect(observed_at=101.0, source_at=11.0)
        self.assertEqual([event.reason for event in reconnect], ["reconnect"])
        second = engine.observe(_sample(102.0, 0.9, source_at=0.0))[0]
        self.assertNotEqual(first.interval_id, second.interval_id)
        self.assertEqual((first.source_start, first.observed_start), (10.0, 100.0))
        self.assertEqual(second.source_start, 0.0)

    def test_eos_discards_pending_enter_and_terminal_state_is_explicit(self) -> None:
        pending = _engine(enter_debounce=2)
        self.assertEqual(pending.observe(_sample(0.0, 0.9)), [])
        self.assertEqual(pending.end_of_stream(), [])  # no orphan exit

        engine = _engine()
        entered = engine.observe(_sample(0.0, 0.9))[0]
        eos = engine.end_of_stream(observed_at=2.0)
        self.assertEqual(len(eos), 1)
        self.assertEqual(eos[0].reason, "end_of_stream")
        self.assertEqual(eos[0].interval_id, entered.interval_id)
        self.assertEqual(engine.end_of_stream(observed_at=3.0), [])
        with self.assertRaisesRegex(RuntimeError, "end_of_stream"):
            engine.observe(_sample(4.0, 0.9))
        self.assertEqual(engine.reconnect(observed_at=4.0), [])
        self.assertEqual(engine.observe(_sample(5.0, 0.9, source_at=0.0))[0].kind, INTEREST_ENTERED)


class InterestIntervalPropertyTests(unittest.TestCase):
    @staticmethod
    def _replay(scores: list[float]) -> list:
        engine = _engine(enter_debounce=2, exit_debounce=2, expiry_s=5.0)
        out = []
        for index, score in enumerate(scores):
            out.extend(engine.observe(_sample(float(index), score)))
        out.extend(engine.end_of_stream())
        return out

    def test_random_signal_sequences_preserve_lifecycle_invariants_and_replay(self) -> None:
        rng = random.Random(0xC0FFEE)
        for _ in range(100):
            scores = [rng.random() for _ in range(rng.randint(0, 80))]
            events = self._replay(scores)
            replay = self._replay(scores)
            self.assertEqual(
                [event.as_dict() for event in events],
                [event.as_dict() for event in replay],
            )

            active: str | None = None
            seen_ids: set[str] = set()
            for event in events:
                self.assertGreaterEqual(event.source_end, event.source_start)
                self.assertGreaterEqual(event.observed_end, event.observed_start)
                self.assertGreaterEqual(event.duration, 0.0)
                if event.kind == INTEREST_ENTERED:
                    self.assertIsNone(active, "duplicate enter")
                    self.assertNotIn(event.interval_id, seen_ids)
                    active = event.interval_id
                    seen_ids.add(event.interval_id)
                elif event.kind == INTEREST_UPDATED:
                    self.assertEqual(event.interval_id, active, "orphan update")
                else:
                    self.assertEqual(event.interval_id, active, "orphan exit")
                    active = None
            self.assertIsNone(active, "EOS must close every open interval")


class _Core:
    metrics = None

    async def perceive_judged(self, frame, source, *, t=None, goal_ids=None):
        return []


class Tier0InterestIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_source_mismatch_is_rejected_before_processing(self) -> None:
        interest = Tier0MotionInterest(
            source="camera:other",
            config=InterestIntervalConfig(enter_threshold=0.7, exit_threshold=0.2),
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            GatedWatchLoop(
                _Core(),
                [Goal(id="watch", condition="something changes", say="changed")],
                threshold=0.7,
                source="camera:front",
                interest=interest,
            )

    async def test_exact_pixel_sequence_transitions_and_eos(self) -> None:
        interest = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.7, exit_threshold=0.2),
        )
        loop = GatedWatchLoop(
            _Core(),
            [Goal(id="watch", condition="something changes", say="changed")],
            threshold=0.7,
            source="camera:front",
            interest=interest,
        )
        # Exact mean absolute diffs: first=0.0, black->white=1.0, white->white=0.0,
        # white->black=1.0. This yields enter, exit, enter, then an explicit EOS exit.
        report = await loop.run(
            [
                (0.0, gray_frame(2, 2, 0)),
                (1.0, gray_frame(2, 2, 255)),
                (2.0, gray_frame(2, 2, 255)),
                (3.0, gray_frame(2, 2, 0)),
            ]
        )
        self.assertEqual(
            [signal.motion for signal in report.signals],
            [0.0, 1.0, 0.0, 1.0],
        )
        self.assertEqual(
            [event.kind for event in report.interest_events],
            [INTEREST_ENTERED, INTEREST_EXITED, INTEREST_ENTERED, INTEREST_EXITED],
        )
        self.assertEqual(
            [event.reason for event in report.interest_events],
            ["tier0_motion", "signal_lost", "tier0_motion", "end_of_stream"],
        )
        self.assertIsNone(report.interest_events[0].confidence)
        self.assertEqual(report.interest_events[0].coverage, 1.0)
        self.assertEqual(report.interest_events[0].admission_id, "tier0.signal.motion")
        self.assertEqual(report.interest_events[0].detector_id, "motion_score.mean")


if __name__ == "__main__":
    unittest.main()
