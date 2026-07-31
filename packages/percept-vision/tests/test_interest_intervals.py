"""VID interval lifecycle: deterministic boundaries over scalar/tier-0 interest signals."""

from __future__ import annotations

import dataclasses
import pathlib
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
from percept.edge.tier0 import (
    Frame,
    GatedWatchLoop,
    MotionGate,
    SALIENCE_STAT_KEY,
    compile_watchspec,
    gray_frame,
)
from percept.harness.tier0 import ADMISSION_CAUSES, NON_ADMISSION_REASONS
from percept.interest import (
    _LIVE_CAUSES,
    _LIVE_EXIT_REASONS,
    _LIVE_NON_ADMISSION_REASONS,
    InterestBackpressureError,
    InterestSubscription,
    LiveInterestAdmission,
    LiveInterestEvent,
    LiveInterestTracker,
    _discontinuity_reason,
    _live_active_reason,
    _live_non_admission_reason,
    interest_admission_from_outcome,
)
from percept.runtime import SourceEnvelope, SourceFormat


def _sample(
    t: float,
    score: float,
    *,
    source_at: float | None = None,
    observed_at: float | None = None,
    reason: str = "synthetic",
    evidence_id: str | None = None,
    confidence: float | None = 0.8,
    coverage: float = 1.0,
    regions: tuple[Box, ...] = (),
    track_ids: tuple[str, ...] = (),
) -> InterestSample:
    source_time = t if source_at is None else source_at
    observed_time = t if observed_at is None else observed_at
    return InterestSample(
        source_at=source_time,
        observed_at=observed_time,
        score=score,
        reason=reason,
        evidence_id=(
            evidence_id
            if evidence_id is not None
            else f"evidence:{source_time}:{observed_time}:{reason}"
        ),
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
    stream_epoch: str = "stream-epoch-1",
) -> InterestIntervalEngine:
    return InterestIntervalEngine(
        source="camera:front",
        stream_epoch=stream_epoch,
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
        for bad in ("0.7", True, None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "int or float"):
                    InterestIntervalConfig(
                        enter_threshold=bad,  # type: ignore[arg-type]
                        exit_threshold=0.2,
                    )
        with self.assertRaisesRegex(ValueError, "int or float"):
            InterestIntervalConfig(
                enter_threshold=0.7,
                exit_threshold=0.2,
                expiry_s="3",  # type: ignore[arg-type]
            )
        normalized = InterestIntervalConfig(
            enter_threshold=1,
            exit_threshold=0,
            expiry_s=3,
        )
        self.assertIs(type(normalized.enter_threshold), float)
        self.assertIs(type(normalized.exit_threshold), float)
        self.assertIs(type(normalized.expiry_s), float)
        # Construction cannot succeed with a latent string that fails only on first observation.
        self.assertEqual(
            _engine(enter=normalized.enter_threshold, exit_=normalized.exit_threshold)
            .observe(_sample(0.0, 0.5)),
            [],
        )

    def test_ids_bind_epoch_evidence_reason_while_clocks_remain_independent(self) -> None:
        base = _engine(stream_epoch="session-a").observe(
            _sample(
                0.0,
                0.9,
                source_at=10.0,
                observed_at=100.0,
                reason="motion",
                evidence_id="frame-sha-a",
            )
        )[0]
        other_epoch = _engine(stream_epoch="session-b").observe(
            _sample(
                0.0,
                0.9,
                source_at=10.0,
                observed_at=100.0,
                reason="motion",
                evidence_id="frame-sha-a",
            )
        )[0]
        other_evidence = _engine(stream_epoch="session-a").observe(
            _sample(
                0.0,
                0.9,
                source_at=10.0,
                observed_at=100.0,
                reason="motion",
                evidence_id="frame-sha-b",
            )
        )[0]
        other_reason = _engine(stream_epoch="session-a").observe(
            _sample(
                0.0,
                0.9,
                source_at=10.0,
                observed_at=100.0,
                reason="audio",
                evidence_id="frame-sha-a",
            )
        )[0]
        other_observer_clock = _engine(stream_epoch="session-a").observe(
            _sample(
                0.0,
                0.9,
                source_at=10.0,
                observed_at=101.0,
                reason="motion",
                evidence_id="frame-sha-a",
            )
        )[0]
        self.assertEqual((base.source_start, base.observed_start), (10.0, 100.0))
        self.assertEqual(base.stream_epoch, "session-a")
        self.assertEqual(base.evidence_id, "frame-sha-a")
        # Observer receipt time is a distinct bound, not evidence identity: replaying the same source
        # evidence on another observer clock keeps the stable id without aliasing either timestamp.
        self.assertEqual(other_observer_clock.observed_start, 101.0)
        self.assertEqual(base.interval_id, other_observer_clock.interval_id)
        self.assertEqual(
            len(
                {
                    base.interval_id,
                    other_epoch.interval_id,
                    other_evidence.interval_id,
                    other_reason.interval_id,
                }
            ),
            4,
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
        expired = engine.advance(observed_at=10.0)
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].interval_id, entered.interval_id)
        self.assertEqual(expired[0].reason, "expired")
        # Deadline was t=3, but the transition was not observed/emitted until the t=10 poll.
        self.assertEqual(expired[0].observed_end, 0.0)
        self.assertEqual(expired[0].event_observed_at, 10.0)

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
        with self.assertRaisesRegex(ValueError, "must differ"):
            engine.reconnect(
                stream_epoch="stream-epoch-1",
                observed_at=101.0,
                source_at=11.0,
            )
        self.assertTrue(engine.active)
        reconnect = engine.reconnect(
            stream_epoch="stream-epoch-2",
            observed_at=101.0,
            source_at=11.0,
        )
        self.assertEqual([event.reason for event in reconnect], ["reconnect"])
        self.assertEqual(reconnect[0].stream_epoch, "stream-epoch-1")
        second = engine.observe(_sample(102.0, 0.9, source_at=0.0))[0]
        self.assertNotEqual(first.interval_id, second.interval_id)
        self.assertEqual((first.stream_epoch, second.stream_epoch), ("stream-epoch-1", "stream-epoch-2"))
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
        self.assertEqual(
            engine.reconnect(stream_epoch="stream-epoch-2", observed_at=4.0),
            [],
        )
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


class _FailOnceCore:
    metrics = None

    def __init__(self) -> None:
        self.failed = False

    async def perceive_judged(self, frame, source, *, t=None, goal_ids=None):
        if not self.failed:
            self.failed = True
            raise RuntimeError("synthetic judge failure")
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

    async def test_interest_replay_requires_explicit_epoch_and_observer_clock(self) -> None:
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
        frames = [(0.0, gray_frame(2, 2, 0))]
        with self.assertRaisesRegex(ValueError, "interest_stream_epoch"):
            await loop.run(frames)
        with self.assertRaisesRegex(ValueError, "interest_observed_times"):
            await loop.run(frames, interest_stream_epoch="replay-1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            await loop.run(
                frames,
                interest_stream_epoch="replay-1",
                interest_observed_times=(),
            )
        self.assertIsNone(interest.engine)

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
            ],
            interest_stream_epoch="fixture-exact-pixels",
            interest_observed_times=(100.0, 101.0, 102.0, 103.0),
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
        # `motion_score.block_max`, NOT `.mean`: this goal carries no per-goal salience override, so
        # the wired loop compiles it at the shipped deployment default, which is `block_max`. The
        # timeline used to read `.mean` only because the loop passed no `include_outcomes`, and the
        # adapter's mean-gauge fallback mislabelled every event with a statistic the gate never
        # consulted. On a uniform flip the two statistics are the same NUMBER (1.0), which is why
        # the interval sequence above is unchanged — only the label was ever wrong.
        self.assertEqual(report.interest_events[0].detector_id, "motion_score.block_max")
        self.assertEqual(report.interest_events[0].stream_epoch, "fixture-exact-pixels")
        self.assertTrue(report.interest_events[0].evidence_id.startswith("frame_"))
        self.assertEqual(
            (
                report.interest_events[0].source_start,
                report.interest_events[0].observed_start,
            ),
            (1.0, 101.0),
        )
        self.assertIsNone(interest.engine)

    async def test_equal_motion_same_current_different_previous_pair_cannot_collide(self) -> None:
        async def _one(previous_pixels: tuple[int, ...]):
            interest = Tier0MotionInterest(
                source="camera:front",
                config=InterestIntervalConfig(enter_threshold=0.4, exit_threshold=0.1),
            )
            loop = GatedWatchLoop(
                _Core(),
                [Goal(id="watch", condition="something changes", say="changed")],
                threshold=0.4,
                source="camera:front",
                interest=interest,
            )
            return await loop.run(
                [
                    (0.0, (2, 1, previous_pixels)),
                    (1.0, (2, 1, (128, 128))),
                ],
                interest_stream_epoch="same-explicit-epoch",
                interest_observed_times=(100.0, 101.0),
            )

        first = await _one((0, 255))
        second = await _one((255, 0))
        self.assertEqual(first.signals[-1].motion, second.signals[-1].motion)
        self.assertNotEqual(
            first.interest_events[0].evidence_id,
            second.interest_events[0].evidence_id,
        )
        self.assertNotEqual(
            first.interest_events[0].interval_id,
            second.interest_events[0].interval_id,
        )

    async def test_successful_epochs_reset_gate_before_new_first_frame(self) -> None:
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
        black = await loop.run(
            [(0.0, gray_frame(2, 2, 0))],
            interest_stream_epoch="successful-black",
            interest_observed_times=(10.0,),
        )
        new_white = await loop.run(
            [(0.0, gray_frame(2, 2, 255))],
            interest_stream_epoch="new-white",
            interest_observed_times=(20.0,),
        )
        self.assertEqual([signal.motion for signal in black.signals], [0.0])
        self.assertEqual([signal.motion for signal in new_white.signals], [0.0])
        self.assertEqual(black.interest_events, [])
        self.assertEqual(new_white.interest_events, [])

    async def test_successive_chunks_are_independent_reusable_epochs(self) -> None:
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
        first = await loop.run(
            [
                (0.0, gray_frame(2, 2, 0)),
                (1.0, gray_frame(2, 2, 255)),
            ],
            interest_stream_epoch="chunk-1",
            interest_observed_times=(100.0, 101.0),
        )
        self.assertIsNone(interest.engine)
        second = await loop.run(
            [
                (0.0, gray_frame(2, 2, 255)),
                (1.0, gray_frame(2, 2, 0)),
            ],
            interest_stream_epoch="chunk-2",
            interest_observed_times=(200.0, 201.0),
        )
        self.assertIsNone(interest.engine)

        for report, epoch in ((first, "chunk-1"), (second, "chunk-2")):
            self.assertEqual(
                [event.kind for event in report.interest_events],
                [INTEREST_ENTERED, INTEREST_EXITED],
            )
            self.assertEqual(
                [event.stream_epoch for event in report.interest_events],
                [epoch, epoch],
            )
            self.assertEqual(
                report.interest_events[0].interval_id,
                report.interest_events[1].interval_id,
            )
        self.assertNotEqual(
            first.interest_events[0].interval_id,
            second.interest_events[0].interval_id,
        )

    async def test_a_localized_goal_can_no_longer_be_replayed_under_the_wrong_statistic(
        self,
    ) -> None:
        """The full round trip: the shipped loop replays a localized Watch as a real timeline.

        A per-goal salience override reaches ``compile_watchspec`` through the wired loop, so this is
        the shipped loop and not a hand-built spec: the goal admits on the localized statistic.
        ``GatedWatchLoop`` now calls ``MotionGate.evaluate`` with ``include_outcomes=True`` whenever
        an interest adapter is wired (``percept/harness/tier0.py``), so the decisions it hands the
        adapter carry the per-Watch outcome and the mean-gauge fallback is genuinely a fallback
        rather than the only path. Before that opt-in this timeline came back EMPTY: the whole-frame
        mean of a one-block event is ``1/48``, below this policy's ``0.10`` enter threshold, so the
        frame that spent a judge call was invisible to replay. The unit-level proof that the adapter
        records the admitting statistic is in ``Tier0MotionInterestGaugeTests``.
        """
        interest = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.10, exit_threshold=0.02),
        )
        loop = GatedWatchLoop(
            _Core(),
            [
                Goal(
                    id="watch",
                    condition="something changes in one corner",
                    say="changed",
                    state={SALIENCE_STAT_KEY: "block_max"},
                )
            ],
            threshold=0.10,
            source="camera:front",
            interest=interest,
        )
        report = await loop.run(
            [(0.0, _still_gate_frame()), (1.0, _one_block_lit())],
            interest_stream_epoch="localized-goal",
            interest_observed_times=(100.0, 101.0),
        )
        self.assertEqual(report.escalations, 1)
        self.assertEqual(report.vision_calls, 1)
        self.assertEqual(report.admission_by_goal, {"watch": 1})
        self.assertAlmostEqual(report.signals[1].motion, 1.0 / 48.0, places=12)
        # Whatever the loop replays, it can no longer describe this block_max admission with the
        # mean gauge. This holds on both sides of the loop's opt-in, and is the invariant A-07-01
        # bought: the mislabelled middle case is gone.
        for event in report.interest_events:
            with self.subTest(kind=event.kind):
                self.assertEqual(event.detector_id, "motion_score.block_max")
                self.assertEqual(event.score, 1.0)
        # ...and the loop now opts in, so the frame that spent the judge call is VISIBLE: one
        # interval, entered on the admitting frame and closed by end-of-stream. This assertion
        # replaces `assertEqual(report.interest_events, [])`, which pinned the half-closed state.
        self.assertEqual(
            [event.kind for event in report.interest_events],
            ["interest.entered", "interest.exited"],
        )
        entered, exited = report.interest_events
        self.assertEqual(entered.admission_id, "tier0.signal.motion")
        self.assertEqual(entered.reason, "tier0_motion")
        self.assertEqual(exited.reason, "end_of_stream")
        self.assertEqual(entered.interval_id, exited.interval_id)
        self.assertEqual((entered.source_start, entered.observed_start), (1.0, 101.0))
        # The whole-frame gauge for this very frame is 1/48 — below the 0.10 enter threshold. That
        # is exactly why the fallback could not carry this Watch, and why the opt-in is the fix.
        self.assertAlmostEqual(report.signals[1].motion, 1.0 / 48.0, places=12)
        self.assertLess(report.signals[1].motion, 0.10)

    async def test_zero_frame_epoch_closes_with_no_observer_clock_and_leaves_the_adapter_reusable(
        self,
    ) -> None:
        # The wired loop's end-of-stream timestamp is `interest_observed_times[-1] or None`, so an
        # empty stream is the one path that reaches `end_of_stream(observed_at=None)` with an engine
        # that never saw a sample. It must terminate the epoch, not leave the adapter half-open.
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
        empty = await loop.run(
            [],
            interest_stream_epoch="zero-frame-epoch",
            interest_observed_times=(),
        )
        self.assertEqual(empty.frames, 0)
        self.assertEqual(empty.signals, [])
        self.assertEqual(empty.interest_events, [])
        self.assertEqual(empty.escalations, 0)
        # Reusable: the zero-frame epoch released the engine, so the next epoch is not a lifecycle
        # error and produces a normal enter/exit pair.
        self.assertIsNone(interest.engine)
        after = await loop.run(
            [
                (0.0, gray_frame(2, 2, 0)),
                (1.0, gray_frame(2, 2, 255)),
            ],
            interest_stream_epoch="after-zero-frame",
            interest_observed_times=(10.0, 11.0),
        )
        self.assertEqual(
            [event.kind for event in after.interest_events],
            [INTEREST_ENTERED, INTEREST_EXITED],
        )
        self.assertIsNone(interest.engine)

    async def test_judge_failure_aborts_hidden_epoch_without_orphan_exit_or_terminal_adapter(self) -> None:
        interest = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.7, exit_threshold=0.2),
        )
        loop = GatedWatchLoop(
            _FailOnceCore(),
            [Goal(id="watch", condition="something changes", say="changed")],
            threshold=0.7,
            source="camera:front",
            interest=interest,
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic judge failure"):
            await loop.run(
                [
                    (0.0, gray_frame(2, 2, 0)),
                    (1.0, gray_frame(2, 2, 255)),
                ],
                interest_stream_epoch="failed-epoch",
                interest_observed_times=(10.0, 11.0),
            )
        self.assertIsNone(interest.engine)

        # The failed report (including its enter) was never exposed. A fresh epoch therefore begins
        # with ENTER, not an orphan EXIT from failed-epoch, and the adapter is not terminal.
        recovered = await loop.run(
            [
                (0.0, gray_frame(2, 2, 0)),
                (1.0, gray_frame(2, 2, 255)),
            ],
            interest_stream_epoch="recovered-epoch",
            interest_observed_times=(20.0, 21.0),
        )
        self.assertEqual(
            [event.kind for event in recovered.interest_events],
            [INTEREST_ENTERED, INTEREST_EXITED],
        )
        # Failed epoch ended on white; recovery starts on black. A carried predecessor would make
        # recovery's first frame motion=1.0. The reset makes it a fresh 0.0, while black->white
        # within the recovery epoch remains the exact 1.0 transition.
        self.assertEqual([signal.motion for signal in recovered.signals], [0.0, 1.0])
        self.assertEqual(
            {event.stream_epoch for event in recovered.interest_events},
            {"recovered-epoch"},
        )
        self.assertIsNone(interest.engine)


GATE_W, GATE_H = 48, 36
#: 48x36 under the hardcoded (8, 6) block grid is exactly 48 blocks of 6x6 px.
_BLOCK_W, _BLOCK_H = GATE_W // 8, GATE_H // 6


def _still_gate_frame() -> Frame:
    return gray_frame(GATE_W, GATE_H, 0)


def _one_block_lit() -> Frame:
    """Exactly one whole 6x6 block flips 0 -> 255: block_max is 1.0, the whole-frame mean is 1/48."""
    pixels = [0] * (GATE_W * GATE_H)
    for y in range(_BLOCK_H):
        for x in range(_BLOCK_W):
            pixels[y * GATE_W + x] = 255
    return Frame(w=GATE_W, h=GATE_H, pixels=tuple(pixels))


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """The gate's per-Watch outcome shape, without a gate — see WatchAdmissionOutcome."""

    goal_id: str
    score: float
    threshold: float = 0.5
    admitted: bool = True
    cause: str | None = "motion"
    non_admission_reason: str | None = None
    admission_id: str = "tier0.signal.motion"
    detector_id: str = "motion_score.block_max"


class _OutcomeDecision:
    """A GateDecision built WITH per-Watch outcomes, without a gate."""

    def __init__(self, *outcomes: _Outcome, ts: float = 0.0, motion: float = 0.0) -> None:
        self.signal = _ScalarSignal(motion, ts)
        self.outcomes = tuple(outcomes)
        self.escalate = any(outcome.admitted for outcome in outcomes)


def _block_max_gate() -> MotionGate:
    """A gate whose only Watch admits on the localized block statistic, at its own higher cutoff."""
    goal = Goal(
        id="watch",
        condition="something changes in one corner",
        say="changed",
        state={SALIENCE_STAT_KEY: "block_max"},
    )
    spec = compile_watchspec(
        [goal],
        threshold=0.10,
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20},
    )
    return MotionGate(spec)


class Tier0MotionInterestGaugeTests(unittest.TestCase):
    """The replay adapter scores an interval from the statistic that admitted, not from the gauge.

    ``Tier0Signal.motion`` remains the reported whole-frame MEAN (pinned by
    ``test_tier0.py::MotionGateModeTests``); using it as the interval SCORE was the defect, because
    a Watch may admit on ``block_max``, ``p95``, an ego residual, a structural/semantic channel or a
    fused wake, none of which the mean can stand in for.
    """

    def test_block_max_admission_is_recorded_with_the_statistic_that_admitted(self) -> None:
        gate = _block_max_gate()
        gate.evaluate(_still_gate_frame(), ts=0.0, include_outcomes=True)
        decision = gate.evaluate(_one_block_lit(), ts=1.0, include_outcomes=True)

        # The gate admitted, and it already knows exactly which statistic did it.
        self.assertTrue(decision.escalate)
        outcome = decision.outcomes[0]
        self.assertTrue(outcome.admitted)
        self.assertEqual(outcome.score, 1.0)
        self.assertEqual(outcome.detector_id, "motion_score.block_max")
        # The reported gauge is a different statistic on a different value scale, and stays one: the
        # fix moves the interval SCORE off the gauge, it does not move the gauge.
        self.assertAlmostEqual(decision.signal.motion, 1.0 / 48.0, places=12)
        self.assertLess(decision.signal.motion, outcome.threshold)

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("block-max-epoch")
        events = adapter.observe(decision, observed_at=100.0, evidence_id="ev-1")
        self.assertEqual([event.kind for event in events], [INTEREST_ENTERED])
        self.assertEqual(events[0].score, 1.0)
        self.assertEqual(events[0].detector_id, "motion_score.block_max")
        self.assertEqual(events[0].admission_id, "tier0.signal.motion")
        self.assertEqual(events[0].reason, "tier0_motion")
        self.assertEqual(
            [event.kind for event in adapter.end_of_stream(observed_at=101.0)],
            [INTEREST_EXITED],
        )
        # The mean-gauge identity is untouched: it is the fallback's label, and it is not what
        # labelled this interval.
        self.assertEqual(adapter.detector_id, "motion_score.mean")

    def test_interval_describes_the_statistic_that_actually_admitted(self) -> None:
        gate = _block_max_gate()
        gate.evaluate(_still_gate_frame(), ts=0.0, include_outcomes=True)
        decision = gate.evaluate(_one_block_lit(), ts=1.0, include_outcomes=True)
        outcome = decision.outcomes[0]

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("block-max-epoch")
        events = adapter.observe(decision, observed_at=100.0, evidence_id="ev-1")

        entered = [event for event in events if event.kind == INTEREST_ENTERED]
        self.assertEqual(len(entered), 1, "an admitted frame must open an interval")
        self.assertEqual(entered[0].score, outcome.score)
        self.assertEqual(entered[0].detector_id, outcome.detector_id)

    def test_mean_admission_on_the_same_geometry_does_open_an_interval(self) -> None:
        # Control: the same adapter, the same pixels, admitted on the plain mean. It held before the
        # fix and holds after it — which is what makes the case above a gauge/statistic mismatch and
        # not a threshold typo, and what shows the fix did not simply lower the bar for everyone.
        goal = Goal(id="watch", condition="the whole frame changes", say="changed")
        gate = MotionGate(compile_watchspec([goal], threshold=0.20))
        gate.evaluate(_still_gate_frame(), ts=0.0, include_outcomes=True)
        decision = gate.evaluate(gray_frame(GATE_W, GATE_H, 255), ts=1.0, include_outcomes=True)
        self.assertTrue(decision.escalate)
        self.assertEqual(decision.signal.motion, 1.0)
        self.assertEqual(decision.outcomes[0].score, 1.0)

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("mean-epoch")
        events = adapter.observe(decision, observed_at=100.0, evidence_id="ev-1")
        self.assertEqual([event.kind for event in events], [INTEREST_ENTERED])
        self.assertEqual(events[0].score, 1.0)

    def test_replay_and_live_recorders_agree_about_the_same_admission(self) -> None:
        # Both records come out of percept.interest.interest_admission_from_outcome, fed the same
        # WatchAdmissionOutcome. The replay adapter used to build its own from the reported gauge, so
        # the same admitted unit produced an interval on the live path and nothing at all on this one.
        gate = _block_max_gate()
        gate.evaluate(_still_gate_frame(), ts=0.0, include_outcomes=True)
        decision = gate.evaluate(_one_block_lit(), ts=1.0, include_outcomes=True)
        outcome = decision.outcomes[0]

        replay = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        replay.begin_epoch("agreement")
        replay_events = replay.observe(decision, observed_at=100.0, evidence_id="ev-1")

        live_events = LiveInterestTracker(run_id="run-1").observe(
            _envelope(0),
            [
                interest_admission_from_outcome(
                    outcome, evidence_id="ev-1", cause=outcome.cause
                )
            ],
        )
        self.assertEqual([event.kind for event in live_events], [INTEREST_ENTERED])
        self.assertEqual([event.kind for event in replay_events], [INTEREST_ENTERED])
        # Every admission field the two records share, on the same admitted unit. (`evidence_id` is
        # not one of them: the live tracker re-derives its own from the source envelope.)
        for field in ("score", "admission_id", "detector_id", "reason"):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(replay_events[0], field), getattr(live_events[0], field)
                )
        self.assertEqual(replay_events[0].detector_id, "motion_score.block_max")
        self.assertEqual(replay_events[0].score, 1.0)

    def test_confidence_and_coverage_are_fixed_and_carry_no_admission_detail(self) -> None:
        gate = _block_max_gate()
        gate.evaluate(_still_gate_frame(), ts=0.0)
        decision = gate.evaluate(gray_frame(GATE_W, GATE_H, 255), ts=1.0)

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("fixed-metadata-epoch")
        entered = adapter.observe(decision, observed_at=100.0, evidence_id="ev-1")[0]
        # Salience is not a calibrated confidence, so the adapter refuses to manufacture one; and it
        # cannot express partial coverage even when only part of the frame carried the admission.
        self.assertIsNone(entered.confidence)
        self.assertEqual(entered.coverage, 1.0)
        self.assertEqual(entered.regions, ())
        self.assertEqual(entered.track_ids, ())

    def test_a_decision_without_outcomes_still_replays_from_the_documented_mean_gauge(self) -> None:
        # `include_outcomes` is a lifecycle opt-in; a decision built without it (and a legacy
        # hand-built one) carries no per-Watch detail at all. The fallback is the pre-fix behaviour,
        # named by the adapter's own constructor identity, not a silent no-op.
        gate = _block_max_gate()
        gate.evaluate(_still_gate_frame(), ts=0.0)
        decision = gate.evaluate(gray_frame(GATE_W, GATE_H, 255), ts=1.0)
        self.assertEqual(decision.outcomes, ())

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("no-outcomes-epoch")
        events = adapter.observe(decision, observed_at=100.0, evidence_id="ev-1")
        self.assertEqual([event.kind for event in events], [INTEREST_ENTERED])
        self.assertEqual(events[0].score, decision.signal.motion)
        self.assertEqual(events[0].detector_id, "motion_score.mean")
        self.assertEqual(events[0].admission_id, "tier0.signal.motion")
        self.assertEqual(adapter.statistic_engines, {})

    def test_two_statistics_on_one_source_get_one_labelled_timeline_each(self) -> None:
        # The cardinality decision: engines are keyed by the admitting statistic, not by the Watch.
        # A localized Watch and a whole-frame Watch on the same source produce two independent,
        # correctly labelled timelines rather than one timeline labelled with whichever won.
        localized = Goal(
            id="corner",
            condition="something changes in one corner",
            say="changed",
            state={SALIENCE_STAT_KEY: "block_max"},
        )
        whole = Goal(id="whole", condition="the whole frame changes", say="changed")
        gate = MotionGate(
            compile_watchspec(
                [localized, whole], threshold=0.10, stat_thresholds={"block_max": 0.20}
            )
        )
        gate.evaluate(_still_gate_frame(), ts=0.0, include_outcomes=True)
        lit = gate.evaluate(_one_block_lit(), ts=1.0, include_outcomes=True)
        self.assertEqual(lit.goal_ids, ("corner",))  # only the localized Watch admitted

        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.20, exit_threshold=0.05),
        )
        adapter.begin_epoch("two-statistics")
        events = adapter.observe(lit, observed_at=100.0, evidence_id="ev-1")
        # Both statistics are tracked; only the one whose score cleared the interval policy opened.
        self.assertEqual(
            sorted(adapter.statistic_engines),
            [
                ("tier0.signal.motion", "motion_score.block_max"),
                ("tier0.signal.motion", "motion_score.mean"),
            ],
        )
        self.assertEqual([event.kind for event in events], [INTEREST_ENTERED])
        self.assertEqual(events[0].detector_id, "motion_score.block_max")
        self.assertEqual(events[0].score, 1.0)

        # Now flip the whole frame: the mean timeline opens on its own id, the block_max one extends.
        white = gate.evaluate(gray_frame(GATE_W, GATE_H, 255), ts=2.0, include_outcomes=True)
        events = adapter.observe(white, observed_at=101.0, evidence_id="ev-2")
        self.assertEqual(
            {(event.kind, event.detector_id) for event in events},
            {
                (INTEREST_UPDATED, "motion_score.block_max"),
                (INTEREST_ENTERED, "motion_score.mean"),
            },
        )
        self.assertEqual(len({event.interval_id for event in events}), 2)
        closing = adapter.end_of_stream(observed_at=102.0)
        self.assertEqual([event.kind for event in closing], [INTEREST_EXITED, INTEREST_EXITED])
        self.assertEqual(
            {event.detector_id for event in closing},
            {"motion_score.block_max", "motion_score.mean"},
        )

    def test_watches_sharing_one_statistic_are_unioned_by_the_highest_score(self) -> None:
        # The other half of the cardinality decision. InterestIntervalEvent has no watch_id and its
        # interval id has no watch field, so two Watches on one statistic cannot be told apart in
        # this record; they are deliberately unioned rather than emitted as colliding intervals.
        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.50, exit_threshold=0.10),
        )
        adapter.begin_epoch("union")
        events = adapter.observe(
            _OutcomeDecision(
                _Outcome(goal_id="quiet", score=0.20, admitted=False, cause=None,
                         non_admission_reason="signal_below_threshold"),
                _Outcome(goal_id="loud", score=0.90),
                ts=1.0,
            ),
            observed_at=100.0,
            evidence_id="ev-1",
        )
        self.assertEqual([event.kind for event in events], [INTEREST_ENTERED])
        self.assertEqual(events[0].score, 0.90)
        self.assertEqual(len(adapter.statistic_engines), 1)
        # Both Watches quiet -> the single shared timeline closes once, not twice.
        closing = adapter.observe(
            _OutcomeDecision(
                _Outcome(goal_id="quiet", score=0.0, admitted=False, cause=None,
                         non_admission_reason="signal_below_threshold"),
                _Outcome(goal_id="loud", score=0.05, admitted=False, cause=None,
                         non_admission_reason="signal_below_threshold"),
                ts=2.0,
            ),
            observed_at=101.0,
            evidence_id="ev-2",
        )
        self.assertEqual([event.kind for event in closing], [INTEREST_EXITED])
        self.assertEqual(closing[0].reason, "signal_lost")
        self.assertEqual(adapter.end_of_stream(observed_at=102.0), [])

    def test_a_policy_denial_is_scored_as_no_interest_not_as_the_last_admission(self) -> None:
        # Every condition reports an outcome on every evaluated unit, so the engine keeps seeing a
        # continuous scalar and can still close on signal_lost. A schedule/cadence denial reports
        # 0.0, which is what the live recorder does with the same denial: not looking is not interest.
        adapter = Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.50, exit_threshold=0.10),
        )
        adapter.begin_epoch("denied")
        adapter.observe(
            _OutcomeDecision(_Outcome(goal_id="w", score=0.90), ts=1.0),
            observed_at=100.0,
            evidence_id="ev-1",
        )
        denied = adapter.observe(
            _OutcomeDecision(
                _Outcome(goal_id="w", score=0.0, admitted=False, cause=None,
                         non_admission_reason="cadence_limited"),
                ts=2.0,
            ),
            observed_at=101.0,
            evidence_id="ev-2",
        )
        self.assertEqual([event.kind for event in denied], [INTEREST_EXITED])
        self.assertEqual(denied[0].reason, "signal_lost")


class InterestProjectionSharingTests(unittest.TestCase):
    """Live and replay records are one projection, not two that happen to agree today."""

    def test_the_worker_has_no_second_projection_left_to_drift(self) -> None:
        import percept.runtime.worker as worker_module

        source = pathlib.Path(worker_module.__file__).read_text(encoding="utf-8")
        self.assertIn("interest_admission_from_outcome(", source)
        # The bug was two call sites performing the same projection independently. A hand-built
        # LiveInterestAdmission at this seam is exactly that shape returning; the name may appear
        # only as an import and a type annotation.
        self.assertEqual(
            source.count("LiveInterestAdmission("),
            0,
            "worker.py must build its live interest record through interest_admission_from_outcome",
        )

    def test_a_worker_owned_route_relabels_the_detector_on_both_recorders(self) -> None:
        # coast/audio are worker routes, not gate detectors, so the gate's detector identity must not
        # be borrowed for them. One function decides that, so replay cannot disagree.
        outcome = _Outcome(goal_id="w", score=0.9, detector_id="motion_score.block_max")
        gate_owned = interest_admission_from_outcome(outcome, evidence_id="ev", cause="motion")
        self.assertEqual(gate_owned.detector_id, "motion_score.block_max")
        self.assertEqual(gate_owned.admission_id, "tier0.signal.motion")
        routed = interest_admission_from_outcome(outcome, evidence_id="ev", cause="coast")
        self.assertEqual(routed.detector_id, "runtime.coast")
        self.assertEqual(routed.admission_id, "runtime.admission.coast")
        dropped = interest_admission_from_outcome(outcome, evidence_id="ev", cause=None)
        self.assertFalse(dropped.admitted)
        # The gate admitted this Watch and the caller then dropped it: there is no gate reason to
        # keep, so it is the catch-all.
        self.assertEqual(dropped.non_admission_reason, "not_admitted")
        refused = interest_admission_from_outcome(
            _Outcome(goal_id="w", score=0.1, admitted=False, cause=None,
                     non_admission_reason="schedule_inactive"),
            evidence_id="ev",
            cause=None,
        )
        self.assertEqual(refused.non_admission_reason, "schedule_inactive")

    def test_the_projection_refuses_an_outcome_it_cannot_read(self) -> None:
        for field, kwargs in (
            ("outcome.score", {"score": 1.5}),
            ("outcome.threshold", {"threshold": float("nan")}),
            ("outcome.goal_id", {"goal_id": ""}),
            ("outcome.detector_id", {"detector_id": ""}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field.replace(".", r"\.")):
                    interest_admission_from_outcome(
                        _Outcome(**{"goal_id": "w", "score": 0.5, **kwargs}),
                        evidence_id="ev",
                        cause="motion",
                    )


class _NoSignalDecision:
    signal = None


class _ScalarSignal:
    def __init__(self, motion: float, ts: float = 0.0) -> None:
        self.motion = motion
        self.ts = ts


class _SyntheticDecision:
    def __init__(self, motion: float, ts: float = 0.0) -> None:
        self.signal = _ScalarSignal(motion, ts)


class Tier0MotionInterestLifecycleTests(unittest.TestCase):
    def _adapter(self) -> Tier0MotionInterest:
        return Tier0MotionInterest(
            source="camera:front",
            config=InterestIntervalConfig(enter_threshold=0.7, exit_threshold=0.2),
        )

    def test_construction_validates_identity_and_config(self) -> None:
        with self.assertRaisesRegex(TypeError, "InterestIntervalConfig"):
            Tier0MotionInterest(source="camera:front", config=object())  # type: ignore[arg-type]
        for field, kwargs in (
            ("source", {"source": "  "}),
            ("admission_id", {"admission_id": ""}),
            ("detector_id", {"detector_id": ""}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    Tier0MotionInterest(
                        **{
                            "source": "camera:front",
                            "config": InterestIntervalConfig(
                                enter_threshold=0.7, exit_threshold=0.2
                            ),
                            **kwargs,
                        }
                    )

    def test_observe_rejects_a_decision_without_a_tier0_signal_before_any_epoch_check(self) -> None:
        adapter = self._adapter()
        # The shape check precedes the lifecycle check: a caller that also forgot begin_epoch is
        # still told the real problem.
        with self.assertRaisesRegex(TypeError, "tier-0 signal"):
            adapter.observe(_NoSignalDecision(), observed_at=1.0, evidence_id="ev")
        with self.assertRaisesRegex(RuntimeError, "begin_epoch"):
            adapter.observe(_SyntheticDecision(0.9), observed_at=1.0, evidence_id="ev")

    def test_out_of_range_or_non_finite_signal_values_are_refused_not_clamped(self) -> None:
        adapter = self._adapter()
        adapter.begin_epoch("bad-values")
        for label, decision in (
            ("above", _SyntheticDecision(1.5)),
            ("below", _SyntheticDecision(-0.1)),
            ("nan", _SyntheticDecision(float("nan"))),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "decision.signal.motion"):
                    adapter.observe(decision, observed_at=1.0, evidence_id="ev")
        with self.assertRaisesRegex(ValueError, "decision.signal.ts"):
            adapter.observe(
                _SyntheticDecision(0.9, ts=float("inf")), observed_at=1.0, evidence_id="ev"
            )
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            adapter.observe(_SyntheticDecision(0.9), observed_at=1.0, evidence_id="")
        # None of the rejected inputs opened an interval or advanced the engine.
        self.assertIsNotNone(adapter.engine)
        self.assertFalse(adapter.engine.active)

    def test_a_second_begin_epoch_raises_and_only_an_explicit_release_recovers(self) -> None:
        adapter = self._adapter()
        adapter.begin_epoch("epoch-1")
        first_engine = adapter.engine
        with self.assertRaisesRegex(RuntimeError, "already active"):
            adapter.begin_epoch("epoch-2")
        # The failed call left the FIRST epoch in place; a caller that aborts a run without
        # releasing the adapter keeps hitting this on every later epoch.
        self.assertIs(adapter.engine, first_engine)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            adapter.begin_epoch("epoch-3")
        # Exactly two public releases exist, and both leave the adapter ready for the next epoch.
        adapter.abort_epoch()
        self.assertIsNone(adapter.engine)
        adapter.abort_epoch()  # idempotent
        self.assertIsNone(adapter.engine)
        adapter.begin_epoch("epoch-4")
        adapter.observe(_SyntheticDecision(0.9), observed_at=1.0, evidence_id="ev")
        self.assertEqual(
            [event.reason for event in adapter.end_of_stream(observed_at=2.0)],
            ["end_of_stream"],
        )
        self.assertIsNone(adapter.engine)
        self.assertEqual(adapter.end_of_stream(observed_at=3.0), [])
        adapter.begin_epoch("epoch-5")


def _envelope(
    sequence: int,
    *,
    epoch: int = 0,
    source_id: str = "video:test",
    source_time: float | None = None,
    observed_time: float | None = None,
) -> SourceEnvelope:
    return SourceEnvelope(
        source_id=source_id,
        modality="video",
        epoch=epoch,
        sequence=sequence,
        source_time=10.0 + sequence if source_time is None else source_time,
        observed_time=100.0 + sequence if observed_time is None else observed_time,
        format=SourceFormat(media_type="application/x-percept-opaque"),
        payload="clear",
        signal=None,
    )


def _admission(
    watch_id: str,
    *,
    admitted: bool = True,
    cause: str | None = "motion",
    non_admission_reason: str | None = None,
    score: float = 0.9,
    threshold: float = 0.5,
    evidence_id: str = "gate-evidence",
    admission_id: str = "tier0.signal.motion",
    detector_id: str = "motion_score.mean",
) -> LiveInterestAdmission:
    return LiveInterestAdmission(
        watch_id=watch_id,
        admitted=admitted,
        cause=cause,
        non_admission_reason=non_admission_reason,
        score=score,
        threshold=threshold,
        evidence_id=evidence_id,
        admission_id=admission_id,
        detector_id=detector_id,
    )


@dataclasses.dataclass(frozen=True)
class _Coverage:
    kind: str = "observed_after_discontinuity"
    missing_sequences: int = 2


@dataclasses.dataclass(frozen=True)
class _Inspection:
    events: tuple[str, ...] = ()
    coverage: _Coverage = dataclasses.field(default_factory=_Coverage)


class LiveInterestVocabularyTests(unittest.TestCase):
    """The three closed live frozensets have no single source of truth — pin the relationship."""

    def test_live_causes_are_exactly_the_gate_causes_plus_the_worker_owned_routes(self) -> None:
        self.assertEqual(set(_LIVE_CAUSES), set(ADMISSION_CAUSES) | {"coast", "audio"})

    def test_live_non_admission_reasons_are_exactly_the_gate_reasons_plus_the_catch_all(
        self,
    ) -> None:
        self.assertEqual(
            set(_LIVE_NON_ADMISSION_REASONS), set(NON_ADMISSION_REASONS) | {"not_admitted"}
        )

    def test_every_cause_names_a_distinct_active_reason_and_never_falls_through(self) -> None:
        # `_live_active_reason` returns "not_admitted" for anything it does not know. A cause added
        # to the gate but not to this map would label an ADMITTED interval as not admitted.
        actives = {cause: _live_active_reason(cause) for cause in sorted(_LIVE_CAUSES)}
        self.assertNotIn("not_admitted", actives.values())
        self.assertEqual(len(set(actives.values())), len(actives))
        self.assertEqual(_live_active_reason("thermal"), "not_admitted")
        self.assertEqual(_live_active_reason(None), "not_admitted")

    def test_every_non_admission_reason_maps_onto_a_legal_exit_reason(self) -> None:
        for reason in sorted(_LIVE_NON_ADMISSION_REASONS):
            with self.subTest(reason=reason):
                self.assertIn(_live_non_admission_reason(reason), _LIVE_EXIT_REASONS)
        # Only the signal case is renamed; the rest pass through verbatim.
        self.assertEqual(_live_non_admission_reason("signal_below_threshold"), "signal_lost")
        self.assertEqual(_live_non_admission_reason("schedule_inactive"), "schedule_inactive")
        self.assertEqual(_live_non_admission_reason("invented_reason"), "not_admitted")
        self.assertEqual(_live_non_admission_reason(None), "not_admitted")

    def test_discontinuity_reasons_are_ranked_and_an_unknown_break_is_loud(self) -> None:
        self.assertEqual(
            _discontinuity_reason(("source_gap", "source_reconnect")), "source_reconnect"
        )
        self.assertEqual(
            _discontinuity_reason(("source_format_change", "source_capture_overflow")),
            "source_capture_overflow",
        )
        with self.assertRaisesRegex(ValueError, "recognized source event"):
            _discontinuity_reason(("something_new",))
        with self.assertRaisesRegex(ValueError, "recognized source event"):
            _discontinuity_reason(())


class LiveInterestAdmissionTests(unittest.TestCase):
    def test_an_admitted_outcome_needs_one_cause_from_the_closed_set_and_no_reason(self) -> None:
        for cause in sorted(_LIVE_CAUSES):
            with self.subTest(cause=cause):
                self.assertEqual(_admission("w", cause=cause).cause, cause)
        with self.assertRaisesRegex(ValueError, "one cause"):
            _admission("w", cause="thermal")
        with self.assertRaisesRegex(ValueError, "one cause"):
            _admission("w", cause=None)
        with self.assertRaisesRegex(ValueError, "one cause"):
            _admission("w", cause="motion", non_admission_reason="cadence_limited")

    def test_a_non_admitted_outcome_needs_one_reason_from_the_closed_set_and_no_cause(self) -> None:
        for reason in sorted(_LIVE_NON_ADMISSION_REASONS):
            with self.subTest(reason=reason):
                built = _admission("w", admitted=False, cause=None, non_admission_reason=reason)
                self.assertEqual(built.non_admission_reason, reason)
        with self.assertRaisesRegex(ValueError, "one reason"):
            _admission("w", admitted=False, cause=None, non_admission_reason="bored")
        with self.assertRaisesRegex(ValueError, "one reason"):
            _admission("w", admitted=False, cause=None, non_admission_reason=None)
        with self.assertRaisesRegex(ValueError, "one reason"):
            _admission(
                "w", admitted=False, cause="motion", non_admission_reason="cadence_limited"
            )

    def test_scalar_and_identity_fields_are_validated_not_coerced(self) -> None:
        with self.assertRaisesRegex(TypeError, "admitted must be a bool"):
            _admission("w", admitted=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "score"):
            _admission("w", score=1.5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            _admission("w", threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "watch_id"):
            _admission("   ")
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            _admission("w", evidence_id="")


class LiveInterestTrackerTests(unittest.TestCase):
    def test_lifecycle_over_one_source_covers_every_per_watch_close_route(self) -> None:
        tracker = LiveInterestTracker(run_id="run-1")
        self.assertEqual(tracker.active, ())

        opened = tracker.observe(_envelope(0), [_admission("watch-b"), _admission("watch-a")])
        # Emission order is deterministic by watch id, not by the caller's iteration order.
        self.assertEqual(
            [(event.kind, event.watch_id) for event in opened],
            [(INTEREST_ENTERED, "watch-a"), (INTEREST_ENTERED, "watch-b")],
        )
        self.assertEqual(
            tracker.active, (("video:test", "watch-a"), ("video:test", "watch-b"))
        )

        # watch-b is absent from this update -> watch_removed; watch-a's cause changes -> the old
        # interval closes and a new one opens on the same unit.
        changed = tracker.observe(_envelope(1), [_admission("watch-a", cause="coast")])
        self.assertEqual(
            [(event.kind, event.watch_id, event.reason) for event in changed],
            [
                (INTEREST_EXITED, "watch-b", "watch_removed"),
                (INTEREST_EXITED, "watch-a", "admission_changed"),
                (INTEREST_ENTERED, "watch-a", "coast"),
            ],
        )
        self.assertNotEqual(opened[0].interval_id, changed[2].interval_id)
        # "coast" is a worker-owned route the gate's own cause vocabulary never names.
        self.assertNotIn("coast", ADMISSION_CAUSES)
        self.assertEqual(changed[2].admission_cause, "coast")

        held = tracker.observe(_envelope(2), [_admission("watch-a", cause="coast")])
        self.assertEqual([event.kind for event in held], [INTEREST_UPDATED])
        self.assertEqual(held[0].interval_id, changed[2].interval_id)
        self.assertEqual((held[0].sequence_start, held[0].sequence_end), (1, 2))

        dropped = tracker.observe(
            _envelope(3),
            [
                _admission(
                    "watch-a",
                    admitted=False,
                    cause=None,
                    non_admission_reason="signal_below_threshold",
                )
            ],
        )
        self.assertEqual(
            [(event.kind, event.reason) for event in dropped],
            [(INTEREST_EXITED, "signal_lost")],
        )
        # The interval end names the last ADMITTED unit; the event names the unit that revealed it.
        self.assertEqual(dropped[0].sequence_end, 2)
        self.assertEqual(dropped[0].event_sequence, 3)
        self.assertEqual(dropped[0].observed_end, 102.0)
        self.assertEqual(dropped[0].event_observed_at, 103.0)
        self.assertEqual(tracker.active, ())

    def test_observe_rejects_impostor_admissions_and_duplicate_watch_ids(self) -> None:
        tracker = LiveInterestTracker(run_id="run-1")

        class _AdmissionImpostor(LiveInterestAdmission):
            pass

        impostor = _AdmissionImpostor(
            watch_id="watch-a",
            admitted=True,
            cause="motion",
            non_admission_reason=None,
            score=0.9,
            threshold=0.5,
            evidence_id="gate-evidence",
            admission_id="tier0.signal.motion",
            detector_id="motion_score.mean",
        )
        with self.assertRaisesRegex(TypeError, "exact LiveInterestAdmission"):
            tracker.observe(_envelope(0), [impostor])
        with self.assertRaisesRegex(ValueError, "duplicate Watch outcome"):
            tracker.observe(_envelope(0), [_admission("watch-a"), _admission("watch-a")])
        # Neither rejection left partial state behind.
        self.assertEqual(tracker.active, ())

    def test_close_all_uses_the_latest_accepted_unit_and_rejects_an_unknown_reason(self) -> None:
        tracker = LiveInterestTracker(run_id="run-1")
        self.assertEqual(tracker.close_all("end_of_stream"), [])
        with self.assertRaisesRegex(ValueError, "unknown terminal reason"):
            tracker.close_all("gave_up")  # type: ignore[arg-type]

        tracker.observe(_envelope(0), [_admission("watch-a")])
        # Units accepted after the last admission still advance the terminal event boundary.
        tracker.record_accepted(_envelope(5))
        closed = tracker.close_all("cancelled")
        self.assertEqual([(event.kind, event.reason) for event in closed],
                         [(INTEREST_EXITED, "cancelled")])
        self.assertEqual(closed[0].observed_end, 100.0)
        self.assertEqual(closed[0].sequence_end, 0)
        self.assertEqual(closed[0].event_observed_at, 105.0)
        self.assertEqual(closed[0].event_sequence, 5)
        self.assertEqual(tracker.active, ())
        self.assertEqual(tracker.close_all("cancelled"), [])

    def test_discontinuity_and_pause_close_only_their_own_source(self) -> None:
        tracker = LiveInterestTracker(run_id="run-1")
        tracker.observe(_envelope(0, source_id="video:one"), [_admission("watch-a")])
        tracker.observe(_envelope(0, source_id="video:two"), [_admission("watch-a")])
        self.assertEqual(
            tracker.active, (("video:one", "watch-a"), ("video:two", "watch-a"))
        )

        broke = tracker.discontinuity(
            _envelope(4, source_id="video:one", epoch=1),
            _Inspection(events=("source_gap", "source_reconnect")),
        )
        self.assertEqual([(event.kind, event.reason) for event in broke],
                         [(INTEREST_EXITED, "source_reconnect")])
        self.assertEqual(broke[0].coverage_kind, "observed_after_discontinuity")
        self.assertEqual(broke[0].missing_sequences, 2)
        self.assertEqual(broke[0].source_events, ("source_gap", "source_reconnect"))
        self.assertEqual(tracker.active, (("video:two", "watch-a"),))

        paused = tracker.pause(
            _envelope(4, source_id="video:two"),
            _Inspection(events=(), coverage=_Coverage(kind="observed", missing_sequences=0)),
        )
        self.assertEqual([(event.kind, event.reason) for event in paused],
                         [(INTEREST_EXITED, "paused")])
        self.assertEqual(tracker.active, ())

    def test_interval_and_source_ids_are_run_local_and_never_echo_the_raw_source(self) -> None:
        secret = "rtsp://user:pass@camera.internal/stream"
        first = LiveInterestTracker(run_id="run-1").observe(
            _envelope(0, source_id=secret), [_admission("watch-a")]
        )[0]
        second = LiveInterestTracker(run_id="run-2").observe(
            _envelope(0, source_id=secret), [_admission("watch-a")]
        )[0]
        self.assertNotIn(secret, first.source_id)
        self.assertNotIn("pass", first.source_id)
        self.assertNotIn(secret, first.evidence_id)
        # Same source, same unit, different run -> different ids. Nothing is globally guessable.
        self.assertNotEqual(first.source_id, second.source_id)
        self.assertNotEqual(first.interval_id, second.interval_id)
        self.assertNotEqual(first.evidence_id, second.evidence_id)
        # Deterministic within a run: an identical replay reproduces the same identity.
        replay = LiveInterestTracker(run_id="run-1").observe(
            _envelope(0, source_id=secret), [_admission("watch-a")]
        )[0]
        self.assertEqual(first.as_dict(), replay.as_dict())


class InterestSubscriptionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _open(max_batches: int = 256) -> InterestSubscription:
        subscription = InterestSubscription(max_batches=max_batches)
        InterestSubscription._bind(subscription, "run-1")
        return subscription

    async def test_max_batches_rejects_non_int_bool_and_below_one(self) -> None:
        self.assertEqual(InterestSubscription()._max_batches, 256)
        for bad in (0, -1, True, False, 1.0, "8", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "max_batches"):
                    InterestSubscription(max_batches=bad)  # type: ignore[arg-type]

    async def test_a_subscription_or_event_impostor_is_refused(self) -> None:
        with self.assertRaisesRegex(TypeError, "final SDK type"):
            type("_SubscriptionImpostor", (InterestSubscription,), {})

        tracker = LiveInterestTracker(run_id="run-1")
        real = tracker.observe(_envelope(0), [_admission("watch-a")])[0]

        class _EventImpostor(LiveInterestEvent):
            pass

        impostor = _EventImpostor(
            **{field.name: getattr(real, field.name) for field in dataclasses.fields(real)}
        )
        subscription = self._open()
        with self.assertRaisesRegex(TypeError, "exact LiveInterestEvent"):
            InterestSubscription._publish(subscription, [impostor])
        self.assertFalse(subscription.closed)

    async def test_overflow_of_a_multi_watch_batch_is_reserved_for_the_terminal_publish(
        self,
    ) -> None:
        subscription = self._open(max_batches=1)
        tracker = LiveInterestTracker(run_id="run-1")

        InterestSubscription._publish(
            subscription, tracker.observe(_envelope(0), [_admission("watch-a")])
        )
        # One frame, two Watches, one atomic batch: capacity is counted in batches so this can never
        # be split in half.
        in_flight = tracker.observe(
            _envelope(1), [_admission("watch-a"), _admission("watch-b")]
        )
        self.assertEqual(len(in_flight), 2)
        with self.assertRaisesRegex(InterestBackpressureError, "capacity"):
            InterestSubscription._publish(subscription, in_flight)
        # The overflow terminates the run, and a producer that ignored that must be refused rather
        # than allowed to append after the reserved transition.
        with self.assertRaisesRegex(InterestBackpressureError, "already rejected"):
            InterestSubscription._publish(subscription, in_flight)

        InterestSubscription._finish(subscription, tracker.close_all("worker_error"))
        drained = [event async for event in subscription]
        self.assertEqual(
            [(event.kind, event.watch_id, event.reason) for event in drained],
            [
                (INTEREST_ENTERED, "watch-a", "tier0_motion"),
                (INTEREST_UPDATED, "watch-a", "tier0_motion"),
                (INTEREST_ENTERED, "watch-b", "tier0_motion"),
                (INTEREST_EXITED, "watch-a", "worker_error"),
                (INTEREST_EXITED, "watch-b", "worker_error"),
            ],
        )
        # The reserve exists so no enter that reached a consumer is left open, and no exit is an
        # orphan: watch-b's enter arrived only via the rejected batch, and it is still closed.
        opened = {event.watch_id for event in drained if event.kind == INTEREST_ENTERED}
        exited = {event.watch_id for event in drained if event.kind == INTEREST_EXITED}
        self.assertEqual(opened, exited)

    async def test_publication_requires_a_bound_open_subscription_and_is_single_use(self) -> None:
        subscription = InterestSubscription()
        tracker = LiveInterestTracker(run_id="run-1")
        batch = tracker.observe(_envelope(0), [_admission("watch-a")])
        with self.assertRaisesRegex(RuntimeError, "not open for publication"):
            InterestSubscription._publish(subscription, batch)
        with self.assertRaisesRegex(RuntimeError, "was not bound"):
            InterestSubscription._finish(subscription, ())

        InterestSubscription._bind(subscription, "run-1")
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            InterestSubscription._bind(subscription, "run-2")
        InterestSubscription._publish(subscription, batch)
        InterestSubscription._publish(subscription, ())  # empty batches are a no-op, not an error
        InterestSubscription._finish(subscription, tracker.close_all("end_of_stream"))
        self.assertTrue(subscription.closed)
        InterestSubscription._finish(subscription, ())  # idempotent

        self.assertEqual(len([event async for event in subscription]), 2)
        with self.assertRaisesRegex(RuntimeError, "exactly one async iterator"):
            subscription.__aiter__()


if __name__ == "__main__":
    unittest.main()
