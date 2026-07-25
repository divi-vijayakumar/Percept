"""Versioned source ordering, clock provenance, and live-runtime coverage effects."""
from __future__ import annotations

import asyncio
import io
import itertools
import queue
import sys
import tempfile
import threading
import unittest
import wave
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
from percept.runtime.audio import AudioSourceConfig, _capture_mic, _capture_wav
from percept.runtime.envelope import legacy_video_envelope
from percept.runtime.source import _source_time
from percept.runtime.worker import _aiter

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

    def test_declared_jpeg_requires_a_complete_container(self):
        envelope = SourceEnvelope(
            source_id="video:fixture",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=1.0,
            source_time=0.0,
            format=SourceFormat(media_type="image/jpeg"),
            payload=("image/jpeg", b"\xff\xd8truncated"),
            signal=STILL,
        )
        self.assertEqual(
            SourceTracker().inspect(envelope).events,
            ("source_corrupt_payload",),
        )

    def test_version_must_be_exact_int_not_bool(self):
        result = SourceTracker().inspect(_video(0, version=True))
        self.assertFalse(result.accepted)
        self.assertEqual(result.events, ("source_version_unsupported",))

    def test_mutable_encoded_payload_is_owned_immutably(self):
        mutable = bytearray(b"\xff\xd8jpeg\xff\xd9")
        envelope = SourceEnvelope(
            source_id="video:fixture",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=1.0,
            source_time=0.0,
            format=SourceFormat(media_type="image/jpeg"),
            payload=("image/jpeg", mutable),
            signal=STILL,
        )
        mutable[:] = b"corrupted later"
        self.assertIsInstance(envelope.payload[1], bytes)
        self.assertEqual(envelope.payload[1], b"\xff\xd8jpeg\xff\xd9")
        self.assertTrue(SourceTracker().inspect(envelope).accepted)

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

    def test_encoded_wav_requires_complete_riff_chunks_and_lengths(self):
        def envelope(blob):
            frame = AudioFrame(
                encoded=blob,
                mime="audio/wav",
                sample_rate=8_000,
                channels=1,
                sample_format="s16le",
            )
            return SourceEnvelope(
                source_id="audio:wav",
                modality="audio",
                epoch=0,
                sequence=0,
                observed_time=1.0,
                source_time=0.0,
                format=SourceFormat(
                    media_type="audio/wav",
                    sample_rate=8_000,
                    channels=1,
                    sample_format="s16le",
                ),
                payload=frame,
                signal=0.0,
            )

        invalid = [
            b"RIFF\x04\x00\x00\x00WAVE",  # signature-only, no fmt/data chunks
            b"RIFF\x24\x00\x00\x00WAVEfmt ",  # declared body is truncated
        ]
        for blob in invalid:
            with self.subTest(blob=blob):
                result = SourceTracker().inspect(envelope(blob))
                self.assertFalse(result.accepted)
                self.assertEqual(result.events, ("source_corrupt_payload",))

        encoded = io.BytesIO()
        with wave.open(encoded, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8_000)
            wav.writeframes(b"\x00\x00")
        self.assertTrue(SourceTracker().inspect(envelope(encoded.getvalue())).accepted)

    def test_explicit_wall_legacy_tuple_preserves_its_timestamp(self):
        async def normalize():
            return [
                envelope
                async for envelope in _aiter(
                    [(42.0, STILL, "clear")],
                    4.0,
                    timebase="wall",
                )
            ]

        envelope = asyncio.run(normalize())[0]
        self.assertEqual(envelope.observed_time, 42.0)
        self.assertIsNone(envelope.source_time)
        self.assertEqual(envelope.event_time, 42.0)


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

    def test_explicit_wall_time_accepts_observation_only_custom_envelope(self):
        envelope = SourceEnvelope(
            source_id="video:custom",
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
        events = []
        stats = RuntimeStats()

        async def collect():
            async for _ in run(
                self._core(),
                source=[envelope],
                timebase="wall",
                judge_dispatch="inline",
                on_health=events.append,
                stats=stats,
            ):
                pass

        asyncio.run(collect())
        ended = next(event for event in events if event.kind == "runtime_ended")
        self.assertEqual(ended.data["source_coverage"]["video"]["observed"], 1)
        self.assertEqual(ended.data["source_coverage"]["video"]["excluded"], 0)
        self.assertEqual(stats.last_wall_t, 1234.5)
        self.assertIsNone(stats.last_media_t)

    def test_wrong_modality_is_excluded_before_video_tracker_and_gate(self):
        audio = SourceEnvelope(
            source_id="audio:wrong-lane",
            modality="audio",
            epoch=0,
            sequence=0,
            observed_time=1.0,
            source_time=0.0,
            format=OPAQUE,
            payload="not-video",
            signal=0.0,
        )
        events = []
        stats = RuntimeStats()

        async def collect():
            async for _ in run(
                self._core(),
                source=[audio],
                on_health=events.append,
                stats=stats,
            ):
                pass

        asyncio.run(collect())
        self.assertEqual(stats.frames, 0)
        mismatch = [event for event in events if event.kind == "source_modality_mismatch"]
        self.assertEqual(len(mismatch), 1)
        ended = next(event for event in events if event.kind == "runtime_ended")
        self.assertEqual(
            ended.data["source_coverage"]["video"],
            {
                "observed": 0,
                "excluded": 1,
                "discontinuities": 0,
                "missing_sequences": 0,
            },
        )

    def test_wrong_modality_is_excluded_before_audio_tracker_and_window(self):
        mismatch_seen = asyncio.Event()
        events = []
        stats = RuntimeStats()

        async def audio_stream():
            yield _video(
                0,
                source_time=None,
                observed_time=10.0,
                payload="clear",
                frame=STILL,
            )

        async def video_stream():
            await mismatch_seen.wait()
            yield SourceEnvelope(
                source_id="video:custom",
                modality="video",
                epoch=0,
                sequence=0,
                observed_time=11.0,
                source_time=None,
                format=OPAQUE,
                payload="clear",
                signal=STILL,
                legacy_timebase="observed",
            )

        def health(event):
            events.append(event)
            if event.kind == "source_modality_mismatch":
                mismatch_seen.set()

        async def collect():
            async for _ in run(
                self._core(),
                source=video_stream(),
                audio_source=audio_stream(),
                timebase="wall",
                on_health=health,
                stats=stats,
            ):
                pass

        asyncio.run(collect())
        ended = next(event for event in events if event.kind == "runtime_ended")
        self.assertEqual(
            ended.data["source_coverage"]["audio"],
            {
                "observed": 0,
                "excluded": 1,
                "discontinuities": 0,
                "missing_sequences": 0,
            },
        )
        self.assertEqual(stats.audio_chunks, 0)

    def test_malformed_fields_are_sanitized_before_health_deduplication(self):
        malformed = SourceEnvelope(
            source_id=["unhashable"],
            modality="video",
            epoch=["unhashable"],
            sequence=0,
            observed_time=1.0,
            source_time=None,
            format=OPAQUE,
            payload="clear",
            signal=STILL,
        )
        events = []

        async def collect():
            async for _ in run(
                self._core(),
                source=[malformed],
                timebase="wall",
                on_health=events.append,
            ):
                pass

        asyncio.run(collect())
        corrupt = next(event for event in events if event.kind == "source_corrupt_payload")
        self.assertEqual(corrupt.data["source_id"], "<invalid-source-id>")
        self.assertIsNone(corrupt.data["epoch"])

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

    def test_cancellation_resistant_stale_true_cannot_spend_post_gap_rising_edge(self):
        started = asyncio.Event()

        class ResistantVision(FakeVision):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def judge(self, condition, frame, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # Suppress cancellation and return a stale confident true. The worker's result
                        # filter is too late if this verdict has already disarmed the core gate.
                        return Verdict(satisfied=True, confidence=0.99, evidence="stale")
                return Verdict(satisfied=True, confidence=0.99, evidence="fresh")

        vision = ResistantVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="g", condition="the watched condition is visible", say="seen"))
        stats = RuntimeStats()

        async def stream():
            yield _video(0, source_time=0.0, observed_time=100.0, frame=STILL)
            yield _video(1, source_time=1.0, observed_time=101.0, payload="moved", frame=MOVED)
            await started.wait()
            yield _video(3, source_time=3.0, observed_time=103.0, frame=STILL)
            yield _video(4, source_time=4.0, observed_time=104.0, payload="moved", frame=MOVED)

        async def collect():
            return [
                incident
                async for incident in run(
                    core,
                    source=stream(),
                    judge_dispatch="async",
                    stats=stats,
                )
            ]

        incidents = asyncio.run(collect())
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].message, "seen")
        self.assertEqual(vision.calls, 2)
        self.assertEqual(stats.judge_source_discontinuity_active_dropped, 1)


class CaptureEnvelopeHardeningTests(unittest.TestCase):
    def test_opencv_nonadvancing_pos_msec_becomes_unknown(self):
        class Cv:
            CAP_PROP_POS_MSEC = 1

        class Cap:
            def get(self, _prop):
                return 0.0

        first = _source_time(Cap(), Cv(), "clip.mp4")
        second = _source_time(
            Cap(), Cv(), "clip.mp4", previous=first, after_first=True
        )
        self.assertEqual(first, 0.0)
        self.assertIsNone(second)

    def test_wav_widths_are_preserved_or_converted_truthfully(self):
        cases = [
            (1, b"\x00\x80\xff", "u8", b"\x00\x80\xff"),
            (3, b"\x00\x00\x00\xff\xff\x7f", "s16le", b"\x00\x00\xff\x7f"),
            (
                4,
                b"\x00\x00\x00\x00\xff\xff\xff\x7f",
                "s16le",
                b"\x00\x00\xff\x7f",
            ),
        ]
        for width, raw, expected_format, expected_pcm in cases:
            with self.subTest(width=width):
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    with wave.open(f.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(width)
                        wav.setframerate(8_000)
                        wav.writeframes(raw)
                    q = queue.Queue()
                    _capture_wav(
                        f.name,
                        AudioSourceConfig(chunk_s=1.0),
                        q,
                        threading.Event(),
                        None,
                    )
                    envelope = q.get_nowait()
                    self.assertEqual(envelope.payload.sample_format, expected_format)
                    self.assertEqual(envelope.format.sample_format, expected_format)
                    self.assertEqual(envelope.payload.pcm, expected_pcm)
                    self.assertEqual(envelope.source_id, f"audio:{f.name}")
                    self.assertTrue(SourceTracker().inspect(envelope).accepted)
                    self.assertIsNone(q.get_nowait())

    def test_microphone_overflow_starts_an_explicit_new_epoch(self):
        stop = threading.Event()

        class Stream:
            def __init__(self, **_kwargs):
                self.reads = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _step):
                self.reads += 1
                if self.reads == 3:
                    stop.set()
                return (b"\x00\x00", self.reads == 2)

        class SoundDevice:
            RawInputStream = Stream

        q = queue.Queue()
        with patch.dict(sys.modules, {"sounddevice": SoundDevice()}):
            _capture_mic(
                7,
                AudioSourceConfig(sample_rate=8_000, chunk_s=0.1),
                q,
                stop,
                None,
            )
        envelopes = [q.get_nowait(), q.get_nowait(), q.get_nowait()]
        with self.assertRaises(queue.Empty):
            q.get_nowait()  # stop was already set, so teardown does not publish an unused sentinel
        self.assertEqual(
            [(item.epoch, item.sequence) for item in envelopes],
            [(0, 0), (1, 0), (1, 1)],
        )
        self.assertEqual(envelopes[1].discontinuity, "capture_overflow")
        self.assertEqual(envelopes[1].source_id, "audio:mic:7")
        tracker = SourceTracker()
        self.assertTrue(tracker.inspect(envelopes[0]).accepted)
        overflow = tracker.inspect(envelopes[1])
        self.assertTrue(overflow.accepted)
        self.assertEqual(overflow.events, ("source_capture_overflow",))
        self.assertEqual(overflow.coverage.kind, "observed_after_discontinuity")
        self.assertEqual(overflow.coverage.missing_sequences, 0)
        self.assertTrue(tracker.inspect(envelopes[2]).accepted)


if __name__ == "__main__":
    unittest.main()
