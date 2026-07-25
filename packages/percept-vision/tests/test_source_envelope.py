"""Versioned source ordering, clock provenance, and live-runtime coverage effects."""
from __future__ import annotations

import asyncio
import itertools
import unittest
from unittest.mock import patch

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.runtime import (
    SOURCE_ENVELOPE_VERSION,
    RuntimeStats,
    SourceEnvelope,
    SourceFormat,
    SourceTracker,
    run,
)
from percept.runtime.envelope import legacy_video_envelope
from percept.runtime.source import _source_time

STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)
OPAQUE = SourceFormat(media_type="application/x-percept-opaque")


def _video(
    sequence: int,
    *,
    epoch: int = 0,
    source_time: float | None = None,
    observed_time: float | None = None,
    payload: object = "clear",
    frame=STILL,
    format: SourceFormat = OPAQUE,
    version: int = SOURCE_ENVELOPE_VERSION,
) -> SourceEnvelope:
    return SourceEnvelope(
        source_id="video",
        modality="video",
        epoch=epoch,
        sequence=sequence,
        observed_time=float(sequence + epoch * 100 if observed_time is None else observed_time),
        source_time=float(sequence if source_time is None else source_time),
        format=format,
        payload=payload,
        signal=frame,
        version=version,
    )


class SourceTrackerTableTests(unittest.TestCase):
    def test_discontinuity_table_and_exact_coverage_effects(self):
        changed = SourceFormat(media_type="application/x-percept-opaque", width=16, height=9)
        cases = [
            (
                "duplicate",
                [_video(0, source_time=10.0), _video(0, source_time=10.0, observed_time=1.0)],
                False,
                ("source_duplicate",),
                "excluded",
                0,
            ),
            (
                "reorder",
                [_video(2, source_time=12.0), _video(1, source_time=11.0, observed_time=3.0)],
                False,
                ("source_reorder",),
                "excluded",
                0,
            ),
            (
                "gap",
                [_video(0, source_time=10.0), _video(3, source_time=13.0)],
                True,
                ("source_gap",),
                "observed_after_discontinuity",
                2,
            ),
            (
                "timestamp reset",
                [_video(0, source_time=10.0), _video(1, source_time=9.0)],
                False,
                ("source_timestamp_reset",),
                "excluded",
                0,
            ),
            (
                "reconnect",
                [_video(7, source_time=17.0), _video(0, epoch=1, source_time=0.0)],
                True,
                ("source_reconnect",),
                "observed_after_discontinuity",
                0,
            ),
            (
                "format change",
                [_video(0), _video(1, format=changed)],
                True,
                ("source_format_change",),
                "observed_after_discontinuity",
                0,
            ),
            (
                "unknown source time",
                [
                    SourceEnvelope(
                        source_id="video",
                        modality="video",
                        epoch=0,
                        sequence=0,
                        observed_time=20.0,
                        source_time=None,
                        format=OPAQUE,
                        payload="clear",
                        signal=STILL,
                    )
                ],
                True,
                ("source_time_unknown",),
                "observed",
                0,
            ),
            (
                "corrupt payload",
                [_video(0, payload=None)],
                False,
                ("source_corrupt_payload",),
                "excluded",
                0,
            ),
            (
                "unsupported version",
                [_video(0, version=SOURCE_ENVELOPE_VERSION + 1)],
                False,
                ("source_version_unsupported",),
                "excluded",
                0,
            ),
        ]
        for name, stream, accepted, events, coverage, missing in cases:
            with self.subTest(name=name):
                tracker = SourceTracker()
                result = None
                for envelope in stream:
                    result = tracker.inspect(envelope)
                self.assertIsNotNone(result)
                self.assertEqual(result.accepted, accepted)
                self.assertEqual(result.events, events)
                self.assertEqual(result.coverage.kind, coverage)
                self.assertEqual(result.coverage.missing_sequences, missing)

    def test_media_clock_requirement_excludes_unknown_time(self):
        envelope = SourceEnvelope(
            source_id="video",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=20.0,
            source_time=None,
            format=OPAQUE,
            payload="clear",
            signal=STILL,
        )
        result = SourceTracker().inspect(envelope, require_source_time=True)
        self.assertFalse(result.accepted)
        self.assertEqual(result.events, ("source_time_unknown",))
        self.assertEqual(result.coverage.kind, "excluded")

    def test_first_visible_sequence_accounts_for_preconsumer_queue_loss(self):
        result = SourceTracker().inspect(_video(4))
        self.assertTrue(result.accepted)
        self.assertEqual(result.events, ("source_gap",))
        self.assertEqual(result.coverage.kind, "observed_after_discontinuity")
        self.assertEqual(result.coverage.missing_sequences, 4)

    def test_legacy_adapter_preserves_only_the_explicit_source_time(self):
        envelope = legacy_video_envelope(
            (42.5, STILL, "clear"),
            source_id="video",
            sequence=3,
            observed_time=900.0,
        )
        self.assertEqual(envelope.source_time, 42.5)
        self.assertEqual(envelope.observed_time, 900.0)
        self.assertEqual(envelope.sequence, 3)
        observed = legacy_video_envelope(
            (42.5, STILL, "clear"),
            source_id="video",
            sequence=3,
            observed_time=900.0,
            timebase="observed",
        )
        self.assertIsNone(observed.source_time)
        self.assertEqual(observed.observed_time, 42.5)

    def test_capture_adapter_does_not_derive_time_from_reported_fps(self):
        class Cv:
            pass

        class Cap:
            def get(self, _prop):
                return 30.0

        self.assertIsNone(_source_time(Cap(), Cv(), "clip.mp4"))

    def test_declared_jpeg_rejects_non_jpeg_bytes(self):
        envelope = SourceEnvelope(
            source_id="video",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=1.0,
            source_time=0.0,
            format=SourceFormat(media_type="image/jpeg"),
            payload=("image/jpeg", b"not-a-jpeg"),
            signal=STILL,
        )
        result = SourceTracker().inspect(envelope)
        self.assertFalse(result.accepted)
        self.assertEqual(result.events, ("source_corrupt_payload",))

    def test_audio_uses_the_same_format_and_corruption_contract(self):
        def audio(sequence, sample_rate, *, declared_rate=None, pcm=b"\x00\x00"):
            frame = AudioFrame(pcm=pcm, sample_rate=sample_rate, channels=1, sample_format="s16le")
            return SourceEnvelope(
                source_id="audio",
                modality="audio",
                epoch=0,
                sequence=sequence,
                observed_time=100.0 + sequence,
                source_time=0.25 * sequence,
                format=SourceFormat(
                    media_type="audio/pcm",
                    sample_rate=sample_rate if declared_rate is None else declared_rate,
                    channels=1,
                    sample_format="s16le",
                ),
                payload=frame,
                signal=0.0,
            )

        tracker = SourceTracker()
        self.assertTrue(tracker.inspect(audio(0, 16_000)).accepted)
        changed = tracker.inspect(audio(1, 8_000))
        self.assertTrue(changed.accepted)
        self.assertEqual(changed.events, ("source_format_change",))
        corrupt = tracker.inspect(audio(2, 8_000, declared_rate=16_000))
        self.assertFalse(corrupt.accepted)
        self.assertEqual(corrupt.events, ("source_corrupt_payload",))


class SourceTrackerPropertyTests(unittest.TestCase):
    def test_all_small_reorderings_are_accounted_once_and_never_reaccepted_backwards(self):
        for order in itertools.permutations((1, 2, 3)):
            with self.subTest(order=order):
                tracker = SourceTracker()
                inspections = [tracker.inspect(_video(0))]
                inspections.extend(tracker.inspect(_video(seq)) for seq in order)
                accepted_sequences = [
                    seq for seq, inspection in zip((0, *order), inspections) if inspection.accepted
                ]
                self.assertEqual(accepted_sequences, sorted(set(accepted_sequences)))
                self.assertEqual(
                    tracker.coverage.observed + tracker.coverage.excluded,
                    4,
                )
                self.assertEqual(
                    tracker.coverage.missing_sequences,
                    sum(i.coverage.missing_sequences for i in inspections),
                )


class LiveRuntimeSourceCoverageTests(unittest.TestCase):
    @staticmethod
    def _core():
        vision = FakeVision(
            script={
                "clear": Verdict(satisfied=False, confidence=0.95, evidence="clear"),
                "moved": Verdict(satisfied=True, confidence=0.95, evidence="moved"),
            }
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="g", condition="the watched condition is visible", say="seen"))
        return core

    def test_synthetic_discontinuities_have_exact_live_coverage(self):
        changed = SourceFormat(media_type="application/x-percept-opaque", width=16, height=9)
        stream = [
            _video(0, source_time=10.0, observed_time=100.0),
            _video(0, source_time=10.0, observed_time=101.0),  # duplicate
            _video(2, source_time=12.0, observed_time=102.0, payload="moved", frame=MOVED),  # gap
            _video(1, source_time=11.0, observed_time=103.0),  # reorder
            _video(3, source_time=9.0, observed_time=104.0),  # reset without epoch
            _video(0, epoch=1, source_time=0.0, observed_time=105.0),  # reconnect
            _video(1, epoch=1, source_time=1.0, observed_time=106.0, format=changed),
            _video(2, epoch=1, source_time=2.0, observed_time=107.0, payload=None),  # corrupt
        ]
        events = []
        stats = RuntimeStats()

        async def collect():
            async for _ in run(
                self._core(),
                source=stream,
                judge_dispatch="inline",
                on_health=events.append,
                stats=stats,
            ):
                pass

        asyncio.run(collect())
        ended = next(event for event in events if event.kind == "runtime_ended")
        self.assertEqual(ended.data["frames_seen"], 4)
        self.assertEqual(
            ended.data["source_coverage"]["video"],
            {
                "observed": 4,
                "excluded": 4,
                "discontinuities": 3,
                "missing_sequences": 1,
            },
        )
        self.assertEqual(stats.source_observed, 4)
        self.assertEqual(stats.source_excluded, 4)
        self.assertEqual(stats.source_discontinuities, 3)
        self.assertEqual(stats.source_missing_sequences, 1)
        kinds = [event.kind for event in events]
        for kind in (
            "source_duplicate",
            "source_gap",
            "source_reorder",
            "source_timestamp_reset",
            "source_reconnect",
            "source_format_change",
            "source_corrupt_payload",
        ):
            self.assertIn(kind, kinds)
        gap = next(event for event in events if event.kind == "source_gap")
        self.assertEqual(gap.data["coverage_effect"], "observed_after_discontinuity")
        self.assertEqual(gap.data["missing_sequences"], 1)

    def test_live_unknown_source_time_uses_observation_clock_without_fabricating_media_time(self):
        envelope = SourceEnvelope(
            source_id="video",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=1234.5,
            source_time=None,
            format=OPAQUE,
            payload="clear",
            signal=STILL,
            legacy_timebase="observed",
        )

        async def fake_open_source(_uri, **_kwargs):
            yield envelope

        events = []
        stats = RuntimeStats()

        async def collect():
            with patch("percept.runtime.source.open_source", fake_open_source):
                async for _ in run(
                    self._core(),
                    source="webcam:0",
                    judge_dispatch="inline",
                    on_health=events.append,
                    stats=stats,
                ):
                    pass

        asyncio.run(collect())
        unknown = [event for event in events if event.kind == "source_time_unknown"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0].data["coverage_effect"], "observed")
        self.assertEqual(stats.last_wall_t, 1234.5)
        self.assertIsNone(stats.last_media_t)
        ended = next(event for event in events if event.kind == "runtime_ended")
        self.assertEqual(ended.data["source_coverage"]["video"]["observed"], 1)
        self.assertEqual(ended.data["source_coverage"]["video"]["excluded"], 0)

    def test_discontinuity_cancels_an_inflight_pre_gap_judgment(self):
        started = asyncio.Event()

        class BlockingVision(FakeVision):
            async def judge(self, condition, frame, **kwargs):
                started.set()
                await asyncio.Event().wait()

        core = Percept.create(vision=BlockingVision(), discover_plugins=False)
        core.add_goal(Goal(id="g", condition="the watched condition is visible", say="seen"))
        events = []
        stats = RuntimeStats()

        async def stream():
            yield _video(0, source_time=0.0, observed_time=100.0, frame=STILL)
            yield _video(1, source_time=1.0, observed_time=101.0, payload="moved", frame=MOVED)
            await started.wait()
            yield _video(3, source_time=3.0, observed_time=103.0, frame=STILL)

        async def collect():
            async for _ in run(
                core,
                source=stream(),
                judge_dispatch="async",
                on_health=events.append,
                stats=stats,
            ):
                pass

        asyncio.run(collect())
        self.assertEqual(stats.judge_source_discontinuity_dropped, 1)
        self.assertEqual(stats.judge_source_discontinuity_active_dropped, 1)
        self.assertEqual(stats.judge_source_discontinuity_pending_dropped, 0)
        dropped = [
            event for event in events
            if event.kind == "judge_source_discontinuity_dropped"
        ]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].data["slot"], "active")
        self.assertEqual(stats.incidents, 0)


if __name__ == "__main__":
    unittest.main()
