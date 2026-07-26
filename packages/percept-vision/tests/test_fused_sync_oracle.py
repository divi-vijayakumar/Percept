"""Adversarial contract tests for the fused audio/video synchronizer.

These tests intentionally exercise the private, pure synchronization seam.  The public product
surface remains ``WatchPlan(sense="fused")`` plus ``percept.run``; keeping the pure oracle here lets
ordering, watermarks, and continuity be tested without timing a live capture loop.
"""

from __future__ import annotations

import itertools
import unittest
from dataclasses import FrozenInstanceError, replace

from percept.fused_sync import (
    _AcceptedMediaUnit,
    _FusedCoverageUncertainty,
    _FusedDiscontinuity,
    _FusedSyncPolicy,
    _FusedSynchronizer,
)
from percept.contracts import AudioFrame
from percept.events import AudioUnit


APP = "com.example.kitchen"
REVISION = 7


def policy(**overrides):
    values = {
        "window_s": 2.0,
        "allowed_lateness_s": 0.25,
        "max_skew_s": 0.5,
        "min_clock_confidence": 0.9,
        "max_units_per_modality": 16,
        "max_bytes_per_modality": 1024,
        "version": "oracle-v1",
    }
    values.update(overrides)
    return _FusedSyncPolicy(**values)


def unit(
    modality: str,
    sequence: int,
    start: float,
    end: float | None = None,
    *,
    payload=None,
    source_id: str | None = None,
    source_epoch: int = 3,
    generation: int = 0,
    watch_revision: int = REVISION,
    observed_start: float | None = None,
    observed_end: float | None = None,
    clock_domain: str = "source",
    clock_authority: str = "fixture-shared-clock",
    clock_confidence: float = 1.0,
    provenance=None,
    discontinuity=_FusedDiscontinuity.NONE,
    source_watermark: float | None = None,
    byte_size: int | None = None,
):
    end = (
        start + 0.25
        if end is None and modality == "audio"
        else start
        if end is None
        else end
    )
    observed_start = 100.0 + start if observed_start is None else observed_start
    observed_end = 100.0 + end if observed_end is None else observed_end
    payload = (
        f"{modality}-{sequence}".encode("ascii")
        if payload is None
        else payload
    )
    if modality == "audio" and isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        if raw and len(raw) % 2:
            raw += b"\0"
        sample_count = len(raw) // 2
        duration = end - start
        sample_rate = (
            max(1, int(round(sample_count / duration)))
            if duration > 0.0
            else 16_000
        )
        payload = AudioFrame(
            pcm=raw,
            sample_rate=sample_rate,
            channels=1,
            sample_format="s16le",
            t_start=start,
            t_end=end,
        )
        payload = AudioUnit(
            seq=sequence,
            t_start=start,
            t_end=end,
            event_t=end,
            chunk=payload,
            sample_rate=sample_rate,
            channels=1,
            sample_format="s16le",
            direction="input",
            source_id=source_id or modality,
            source_epoch=source_epoch,
            source_sequence=sequence,
            observed_time=observed_end,
            source_time=end,
        )
        if byte_size is None:
            byte_size = len(raw)
    return _AcceptedMediaUnit(
        modality=modality,
        payload=payload,
        source_id=source_id or modality,
        source_epoch=source_epoch,
        source_sequence=sequence,
        generation=generation,
        watch_application=APP,
        watch_revision=watch_revision,
        event_start=start,
        event_end=end,
        observed_start=observed_start,
        observed_end=observed_end,
        clock_domain=clock_domain,
        clock_authority=clock_authority,
        clock_confidence=clock_confidence,
        provenance=(
            {"fixture": modality, "sequence": sequence}
            if provenance is None
            else provenance
        ),
        discontinuity=discontinuity,
        source_watermark=source_watermark,
        byte_size=byte_size,
    )


def synchronizer(**policy_overrides):
    return _FusedSynchronizer(
        policy(**policy_overrides),
        watch_application=APP,
        watch_revision=REVISION,
    )


def sequences(bundle, modality: str) -> tuple[int, ...]:
    units = bundle.video_units if modality == "video" else bundle.audio_units
    return tuple(item.source_sequence for item in units)


def uncertainty_codes(bundle) -> set[str]:
    reasons = set()
    for reason in bundle.coverage_reasons:
        if isinstance(reason, _FusedCoverageUncertainty):
            reasons.add(str(getattr(reason, "code", reason)))
        else:
            reasons.add(str(reason))
    return reasons


def media_bytes(item) -> bytes:
    payload = item.payload
    if isinstance(payload, AudioUnit):
        payload = payload.chunk
    if isinstance(payload, AudioFrame):
        return payload.pcm or payload.encoded or b""
    if isinstance(payload, tuple):
        return payload[1]
    return payload


class FusedPureOrderingTests(unittest.TestCase):
    def test_every_ingress_permutation_produces_one_canonical_order(self) -> None:
        candidates = (
            unit("video", 1, 1.0, 1.1),
            unit("video", 2, 1.8, 1.9),
            unit("audio", 4, 0.9, 1.4),
            unit("audio", 5, 1.4, 2.0),
        )
        observed = set()
        for permutation in itertools.permutations(candidates):
            sync = synchronizer(allowed_lateness_s=100.0)
            self.assertTrue(all(sync.accept(item) for item in permutation))
            sync.bundle_for(candidates[1])
            bundle = sync.close()[0]
            self.assertTrue(bundle.complete)
            observed.add(
                (
                    sequences(bundle, "video"),
                    sequences(bundle, "audio"),
                    bundle.video_event_interval,
                    bundle.audio_event_interval,
                )
            )
        self.assertEqual(
            observed,
            {((1, 2), (4, 5), (1.0, 1.9), (0.9, 2.0))},
        )

    def test_skew_limit_is_closed_and_one_microsecond_past_is_uncertain(self) -> None:
        exact = synchronizer(max_skew_s=0.5)
        exact_video = unit("video", 0, 10.0, source_watermark=11.0)
        exact.accept(exact_video)
        exact.accept(unit("audio", 0, 10.5, source_watermark=11.0))
        at_limit = exact.bundle_for(exact_video)
        self.assertAlmostEqual(at_limit.actual_skew_s, 0.5)

        past = synchronizer(max_skew_s=0.5)
        past_video = unit("video", 0, 10.0, source_watermark=11.0)
        past.accept(past_video)
        past.accept(
            unit("audio", 0, 10.500001, source_watermark=11.0)
        )
        outside = past.bundle_for(past_video)
        self.assertFalse(outside.complete)
        self.assertGreater(outside.actual_skew_s, 0.5)
        codes = uncertainty_codes(outside)
        self.assertIn("skew_exceeded", " ".join(codes).lower())

    def test_event_watermark_not_observed_time_controls_lateness(self) -> None:
        sync = synchronizer(allowed_lateness_s=0.25)
        self.assertTrue(
            sync.accept(
                unit(
                    "audio",
                    2,
                    9.9,
                    10.0,
                    observed_start=1.0,
                    observed_end=1.1,
                )
            )
        )
        # Derived watermark is max_event_end_seen - allowed_lateness == 9.75.
        self.assertTrue(
            sync.accept(
                unit(
                    "audio",
                    1,
                    9.7,
                    9.75,
                    observed_start=999.0,
                    observed_end=999.1,
                )
            )
        )
        self.assertFalse(
            sync.accept(
                unit(
                    "audio",
                    0,
                    9.7,
                    9.749999,
                    observed_start=1000.0,
                    observed_end=1000.1,
                )
            )
        )

    def test_real_source_watermark_is_monotonic_and_cannot_regress(self) -> None:
        sync = synchronizer(allowed_lateness_s=10.0)
        video = unit("video", 2, 2.0, source_watermark=1.5)
        self.assertTrue(
            sync.accept(video)
        )
        self.assertTrue(
            sync.accept(unit("video", 3, 3.0, source_watermark=1.0))
        )
        self.assertFalse(
            sync.accept(unit("video", 1, 1.4, source_watermark=1.4))
        )
        bundle = sync.bundle_for(video)
        self.assertGreaterEqual(bundle.video_watermark, 1.5)

    def test_anchor_closes_only_after_both_watermarks_cover_window_and_skew(self) -> None:
        sync = synchronizer(window_s=2.0, max_skew_s=0.5)
        anchor = unit("video", 0, 10.0, source_watermark=10.49)
        audio = unit("audio", 0, 8.0, 10.5, source_watermark=10.49)
        sync.accept(anchor)
        sync.accept(audio)
        sync.bundle_for(anchor)
        self.assertEqual(sync.drain_ready(), ())

        # New accepted units move both real source watermarks through the complete decision
        # horizon (window start through maximum skew), without relying on observed time.
        sync.accept(unit("video", 1, 10.5, source_watermark=10.5))
        sync.accept(unit("audio", 1, 10.5, source_watermark=10.5))
        closed = sync.drain_ready()
        self.assertEqual(
            sum(bundle.video_units[0].source_sequence == 0 for bundle in closed),
            1,
        )
        self.assertEqual(sync.drain_ready(), ())

    def test_rechunking_changes_units_but_not_provider_bytes_or_intervals(self) -> None:
        video = unit(
            "video", 0, 1.0, payload=b"frame", source_watermark=3.0
        )
        one_chunk = synchronizer()
        one_chunk.accept(video)
        one_chunk.accept(
            unit(
                "audio",
                0,
                0.0,
                2.0,
                payload=b"abcdefgh",
                source_watermark=3.0,
            )
        )
        first = one_chunk.bundle_for(video)

        two_chunks = synchronizer()
        two_chunks.accept(video)
        two_chunks.accept(
            unit(
                "audio", 0, 0.0, 1.0, payload=b"abcd", source_watermark=3.0
            )
        )
        two_chunks.accept(
            unit(
                "audio", 1, 1.0, 2.0, payload=b"efgh", source_watermark=3.0
            )
        )
        second = two_chunks.bundle_for(video)

        self.assertTrue(first.complete and second.complete)
        self.assertEqual(
            b"".join(media_bytes(item) for item in first.audio_units),
            b"".join(media_bytes(item) for item in second.audio_units),
        )
        self.assertEqual(first.audio_event_interval, second.audio_event_interval)
        self.assertEqual(first.actual_skew_s, second.actual_skew_s)


class FusedPureContinuityTests(unittest.TestCase):
    def test_generation_change_rejects_old_work_and_never_borrows_audio(self) -> None:
        sync = synchronizer()
        before_video = unit("video", 0, 1.0, source_watermark=2.0)
        self.assertTrue(sync.accept(before_video))
        self.assertTrue(
            sync.accept(unit("audio", 0, 1.0, source_watermark=2.0))
        )
        self.assertTrue(sync.bundle_for(before_video).complete)

        self.assertEqual(sync.advance_generation(_FusedDiscontinuity.RECONNECT), 1)
        self.assertFalse(sync.accept(unit("audio", 1, 2.0, generation=0)))
        after_video = unit("video", 1, 2.0, generation=1)
        self.assertTrue(sync.accept(after_video))
        after = sync.bundle_for(after_video)
        self.assertFalse(after.complete)
        self.assertEqual(sequences(after, "audio"), ())
        self.assertTrue(uncertainty_codes(after))

    def test_reconnect_does_not_permanently_poison_the_new_generation(self) -> None:
        sync = synchronizer()
        sync.accept(unit("video", 0, 1.0))
        sync.advance_generation(_FusedDiscontinuity.RECONNECT)
        video = unit(
            "video",
            0,
            2.0,
            source_epoch=4,
            generation=1,
            source_watermark=3.0,
        )
        audio = unit(
            "audio",
            0,
            2.0,
            source_epoch=4,
            generation=1,
            source_watermark=3.0,
        )
        self.assertTrue(sync.accept(video))
        self.assertTrue(sync.accept(audio))
        recovered = sync.bundle_for(video)
        self.assertTrue(recovered.complete)
        self.assertEqual(recovered.coverage_reasons, ())

    def test_every_runtime_discontinuity_reason_advances_before_new_evidence(self) -> None:
        reasons = (
            _FusedDiscontinuity.GAP,
            _FusedDiscontinuity.RECONNECT,
            _FusedDiscontinuity.EPOCH_GAP,
            _FusedDiscontinuity.FORMAT_CHANGE,
            _FusedDiscontinuity.CAPTURE_OVERFLOW,
            _FusedDiscontinuity.PAUSE,
            _FusedDiscontinuity.RESUME,
            _FusedDiscontinuity.WATCH_REVISION,
            _FusedDiscontinuity.CANCELLATION,
            _FusedDiscontinuity.WORKER_ERROR,
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                sync = synchronizer()
                sync.accept(unit("audio", 0, 1.0))
                revision = REVISION + 1 if reason is _FusedDiscontinuity.WATCH_REVISION else REVISION
                generation = sync.advance_generation(
                    reason,
                    watch_revision=(
                        revision
                        if reason is _FusedDiscontinuity.WATCH_REVISION
                        else None
                    ),
                )
                video = unit(
                    "video",
                    0,
                    2.0,
                    generation=generation,
                    watch_revision=revision,
                )
                sync.accept(video)
                bundle = sync.bundle_for(video)
                self.assertFalse(bundle.complete)
                self.assertEqual(bundle.audio_units, ())

    def test_watch_revision_advances_generation_before_accepting_new_media(self) -> None:
        sync = synchronizer()
        sync.accept(unit("audio", 0, 1.0))
        self.assertEqual(
            sync.advance_generation(
                _FusedDiscontinuity.WATCH_REVISION,
                watch_revision=REVISION + 1,
            ),
            1,
        )
        self.assertFalse(sync.accept(unit("video", 0, 1.0, watch_revision=REVISION)))
        current_video = unit(
            "video",
            0,
            2.0,
            generation=1,
            watch_revision=REVISION + 1,
        )
        self.assertTrue(
            sync.accept(current_video)
        )
        bundle = sync.bundle_for(current_video)
        self.assertEqual(bundle.watch_revision, REVISION + 1)
        self.assertFalse(bundle.complete)
        self.assertEqual(bundle.audio_units, ())

    def test_revision_purges_both_pending_and_ready_but_undrained_results(self) -> None:
        sync = synchronizer()
        ready_video = unit(
            "video",
            0,
            1.0,
            source_watermark=2.0,
        )
        sync.accept(ready_video)
        sync.accept(unit("audio", 0, 1.0, source_watermark=2.0))
        sync.bundle_for(ready_video)

        pending_video = unit("video", 1, 3.0)
        sync.accept(pending_video)
        sync.bundle_for(pending_video)
        sync.advance_generation(
            _FusedDiscontinuity.WATCH_REVISION,
            watch_revision=REVISION + 1,
        )

        self.assertEqual(sync.drain_ready(), ())
        self.assertEqual(sync.close(), ())

    def test_epoch_and_source_are_part_of_identity(self) -> None:
        sync = synchronizer()
        initial_video = unit(
            "video",
            0,
            1.0,
            source_id="cam-a",
            source_epoch=1,
            source_watermark=2.0,
        )
        sync.accept(initial_video)
        sync.accept(
            unit(
                "audio",
                0,
                1.0,
                source_id="mic-a",
                source_epoch=1,
                source_watermark=2.0,
            )
        )
        self.assertTrue(sync.bundle_for(initial_video).complete)

        sync.advance_generation(_FusedDiscontinuity.EPOCH_GAP)
        current_video = unit(
            "video",
            0,
            2.0,
            source_id="cam-b",
            source_epoch=2,
            generation=1,
        )
        sync.accept(current_video)
        sync.accept(
            unit(
                "audio",
                0,
                2.0,
                source_id="mic-a",
                source_epoch=1,
                generation=1,
            )
        )
        bundle = sync.bundle_for(current_video)
        self.assertFalse(bundle.complete)
        self.assertTrue(uncertainty_codes(bundle))

    def test_generic_equal_clock_labels_do_not_claim_cross_device_calibration(self) -> None:
        sync = synchronizer()
        video = unit(
            "video",
            0,
            1.0,
            source_id="usb-camera",
            clock_domain="source",
            clock_authority="usb-camera",
        )
        audio = unit(
            "audio",
            0,
            1.0,
            source_id="usb-microphone",
            clock_domain="source",
            clock_authority="usb-microphone",
        )
        sync.accept(video)
        sync.accept(audio)
        bundle = sync.bundle_for(video)
        self.assertFalse(bundle.complete)
        self.assertTrue(uncertainty_codes(bundle))

    def test_second_source_per_modality_is_rejected_as_unsupported_n_way_fusion(self) -> None:
        sync = synchronizer()
        first = unit("video", 0, 1.0, source_id="cam-a")
        second = unit("video", 0, 1.0, source_id="cam-b")
        self.assertTrue(sync.accept(first))
        try:
            accepted = sync.accept(second)
        except ValueError:
            accepted = False
        self.assertFalse(accepted)
        sources = sync.bundle_for(first).video_sources
        self.assertEqual(len(sources), 1)
        self.assertNotIn("cam-b", repr(sources))

    def test_close_flushes_each_pending_anchor_once_and_is_idempotent(self) -> None:
        sync = synchronizer()
        first = unit("video", 0, 1.0)
        sync.accept(first)
        sync.accept(unit("audio", 0, 1.0))
        second = unit("video", 1, 2.0)
        sync.accept(second)
        sync.accept(unit("audio", 1, 2.0))
        direct = (sync.bundle_for(first), sync.bundle_for(second))

        # ``bundle_for`` may itself return a terminal ASK receipt once both watermarks close even
        # when media coverage is incomplete.  Only WATERMARK_PENDING remains owned by the queue;
        # count every other direct receipt exactly once instead of equating terminal with complete.
        terminal_direct = tuple(
            bundle
            for bundle in direct
            if "watermark_pending"
            not in " ".join(uncertainty_codes(bundle)).lower()
        )
        closed = (
            terminal_direct
            + sync.drain_ready()
            + sync.close()
        )
        self.assertEqual(len(closed), 2)
        self.assertTrue(
            all(
                "watermark_pending"
                not in " ".join(uncertainty_codes(bundle)).lower()
                for bundle in closed
            )
        )
        self.assertEqual(sync.close(), ())

    def test_eos_closes_unwatermarked_anchor_but_reports_nonempty_truncated_tail(self) -> None:
        sync = synchronizer(window_s=2.0, max_skew_s=0.5)
        anchor = unit("video", 0, 10.0)
        # Non-empty audio is not sufficient: it ends before the complete fused tail.
        sync.accept(anchor)
        sync.accept(unit("audio", 0, 8.0, 9.4, payload=b"partial!"))
        sync.bundle_for(anchor)
        self.assertEqual(sync.drain_ready(), ())
        closed = sync.close()
        self.assertEqual(len(closed), 1)
        self.assertFalse(closed[0].complete)
        self.assertTrue(uncertainty_codes(closed[0]))

    def test_missing_low_confidence_and_clock_mismatch_are_typed_uncertainty(self) -> None:
        cases = []

        missing = synchronizer()
        missing_video = unit("video", 0, 1.0)
        missing.accept(missing_video)
        cases.append(missing.bundle_for(missing_video))

        low_confidence = synchronizer(min_clock_confidence=0.9)
        low_video = unit("video", 0, 1.0)
        low_confidence.accept(low_video)
        low_confidence.accept(unit("audio", 0, 1.0, clock_confidence=0.89))
        cases.append(low_confidence.bundle_for(low_video))

        mismatched = synchronizer()
        mismatch_video = unit("video", 0, 1.0, clock_domain="source")
        mismatched.accept(mismatch_video)
        mismatched.accept(unit("audio", 0, 1.0, clock_domain="observed"))
        cases.append(mismatched.bundle_for(mismatch_video))

        for bundle in cases:
            with self.subTest(reasons=bundle.coverage_reasons):
                self.assertFalse(bundle.complete)
                self.assertTrue(bundle.coverage_reasons)
                self.assertTrue(
                    all(
                        isinstance(reason, _FusedCoverageUncertainty)
                        for reason in bundle.coverage_reasons
                    )
                )


class FusedPureBoundsAndIdentityTests(unittest.TestCase):
    def test_unit_and_byte_limits_retain_no_more_than_the_declared_bounds(self) -> None:
        sync = synchronizer(
            window_s=100.0,
            allowed_lateness_s=100.0,
            max_units_per_modality=2,
            max_bytes_per_modality=5,
        )
        for seq, payload in enumerate((b"aa", b"bb", b"ccc")):
            sync.accept(unit("audio", seq, float(seq), payload=payload))
        video = unit("video", 0, 2.0, payload=b"v")
        sync.accept(video)

        bundle = sync.bundle_for(video)
        self.assertLessEqual(len(bundle.audio_units), 2)
        self.assertLessEqual(sum(item.byte_size for item in bundle.audio_units), 5)
        self.assertNotIn(0, sequences(bundle, "audio"))
        self.assertTrue(uncertainty_codes(bundle))

    def test_bound_pressure_is_window_scoped_and_later_complete_window_recovers(self) -> None:
        sync = synchronizer(
            window_s=1.0,
            allowed_lateness_s=10.0,
            max_units_per_modality=2,
            max_bytes_per_modality=4,
        )
        for seq in range(3):
            sync.accept(unit("audio", seq, seq * 0.1, payload=b"aa"))
        old_video = unit("video", 0, 0.2, payload=b"v")
        sync.accept(old_video)
        self.assertFalse(sync.bundle_for(old_video).complete)

        fresh_video = unit(
            "video", 1, 10.0, payload=b"v", source_watermark=11.0
        )
        fresh_audio = unit(
            "audio", 3, 10.0, payload=b"aa", source_watermark=11.0
        )
        sync.accept(fresh_video)
        sync.accept(fresh_audio)
        recovered = sync.bundle_for(fresh_video)
        self.assertTrue(recovered.complete)
        self.assertEqual(recovered.coverage_reasons, ())

    def test_unknown_mutable_and_zero_byte_payloads_never_enter_state(self) -> None:
        sync = synchronizer()
        with self.assertRaises((TypeError, ValueError)):
            sync.accept(unit("audio", 0, 1.0, payload=[], byte_size=1))
        with self.assertRaises((TypeError, ValueError)):
            sync.accept(unit("audio", 1, 1.0, payload=b"", byte_size=0))

    def test_exact_qualified_byte_carriers_are_accepted_without_coercion(self) -> None:
        raw = b"\xff\xd8fixture\xff\xd9"
        typed = ("image/jpeg", raw)
        plain_unit = unit("video", 0, 1.0, payload=raw, byte_size=len(raw))
        typed_unit = unit("video", 1, 2.0, payload=typed, byte_size=len(raw))
        self.assertIs(plain_unit.payload, raw)
        self.assertEqual(typed_unit.payload, typed)

    def test_stalled_audio_cannot_make_pending_anchor_count_or_bytes_unbounded(self) -> None:
        sync = synchronizer(
            window_s=100.0,
            allowed_lateness_s=100.0,
            max_units_per_modality=3,
            max_bytes_per_modality=12,
        )
        for seq in range(100):
            anchor = unit("video", seq, float(seq), payload=b"vvvv", byte_size=4)
            sync.accept(anchor)
            sync.bundle_for(anchor)
        pending = sync.close()
        self.assertLessEqual(len(pending), 3)
        self.assertLessEqual(
            sum(
                sum(item.byte_size for item in bundle.video_units)
                for bundle in pending
            ),
            36,
        )
        self.assertTrue(all(uncertainty_codes(bundle) for bundle in pending))

    def test_many_emitted_anchors_have_unique_bounded_explicit_identity(self) -> None:
        sync = synchronizer(
            window_s=0.1,
            allowed_lateness_s=0.0,
            max_units_per_modality=4,
            max_bytes_per_modality=64,
        )
        emitted_ids = []
        for seq in range(50):
            video = unit(
                "video",
                seq,
                float(seq),
                payload=b"v",
                source_watermark=float(seq) + 1.0,
            )
            audio = unit(
                "audio",
                seq,
                float(seq),
                payload=b"aa",
                source_watermark=float(seq) + 1.0,
            )
            sync.accept(video)
            sync.accept(audio)
            bundle = sync.bundle_for(video).with_judgment_identity(
                evidence_seq=seq + 1,
                judgment_event_id=f"jid-{seq + 1}",
            )
            emitted_ids.append((bundle.evidence_seq, bundle.judgment_event_id))
        self.assertEqual(len(set(emitted_ids)), 50)
        self.assertEqual(emitted_ids[0], (1, "jid-1"))
        self.assertEqual(emitted_ids[-1], (50, "jid-50"))

    def test_accepted_payload_and_provenance_are_immutable_snapshots(self) -> None:
        payload = bytearray(b"abcd")
        provenance = {"carrier": {"seq": 4}}
        accepted = unit(
            "audio",
            4,
            1.0,
            payload=payload,
            provenance=provenance,
            byte_size=4,
        )
        payload[:] = b"zzzz"
        provenance["carrier"]["seq"] = 99

        self.assertEqual(media_bytes(accepted), b"abcd")
        self.assertEqual(accepted.provenance["carrier"]["seq"], 4)
        with self.assertRaises(FrozenInstanceError):
            accepted.source_sequence = 9

    def test_bundle_identity_is_frozen_once_and_preserved_by_identity_stamp(self) -> None:
        sync = synchronizer()
        video = unit(
            "video",
            0,
            1.0,
            payload=b"exact-video",
            source_watermark=2.0,
        )
        sync.accept(video)
        sync.accept(
            unit(
                "audio",
                0,
                1.0,
                payload=b"exact-audio!",
                source_watermark=2.0,
            )
        )
        bundle = sync.bundle_for(video)
        stamped = bundle.with_judgment_identity(
            evidence_seq=41,
            judgment_event_id="judgment-opaque-41",
        )

        self.assertEqual(stamped.video_units, bundle.video_units)
        self.assertEqual(stamped.audio_units, bundle.audio_units)
        self.assertEqual(stamped.watch_application, APP)
        self.assertEqual(stamped.watch_revision, REVISION)
        self.assertEqual(stamped.policy_version, "oracle-v1")
        self.assertEqual(stamped.evidence_seq, 41)
        self.assertEqual(stamped.judgment_event_id, "judgment-opaque-41")
        with self.assertRaises(FrozenInstanceError):
            stamped.evidence_seq = 42

    def test_same_shaped_but_unaccepted_anchor_cannot_forge_a_receipt(self) -> None:
        sync = synchronizer()
        anchor = unit("video", 0, 1.0, payload=b"frame")
        sync.accept(anchor)
        sync.accept(unit("audio", 0, 1.0, payload=b"audio!"))
        forged = replace(anchor)
        self.assertIsNot(forged, anchor)
        self.assertEqual(forged, anchor)
        with self.assertRaises((KeyError, ValueError)):
            sync.bundle_for(forged)

    def test_emitted_anchor_cannot_repend_after_bounded_history_churn(self) -> None:
        sync = synchronizer(
            window_s=0.1,
            allowed_lateness_s=0.0,
            max_units_per_modality=2,
            max_bytes_per_modality=16,
        )
        anchor = unit("video", 0, 1.0, payload=b"v", source_watermark=2.0)
        sync.accept(anchor)
        sync.accept(
            unit("audio", 0, 1.0, payload=b"aa", source_watermark=2.0)
        )
        first = sync.bundle_for(anchor)
        self.assertTrue(first.complete)
        self.assertEqual(sync.drain_ready(), ())

        for seq in range(1, 20):
            sync.accept(unit("video", seq, float(seq + 1), payload=b"v"))
            sync.accept(unit("audio", seq, float(seq + 1), payload=b"aa"))

        replayed = unit(
            "video",
            0,
            1.0,
            payload=b"v",
            source_watermark=2.0,
        )
        self.assertIsNot(replayed, anchor)
        self.assertEqual(
            (
                replayed.source_id,
                replayed.source_epoch,
                replayed.generation,
                replayed.source_sequence,
            ),
            (
                anchor.source_id,
                anchor.source_epoch,
                anchor.generation,
                anchor.source_sequence,
            ),
        )
        try:
            replay_accepted = sync.accept(replayed)
        except ValueError:
            replay_accepted = False
        self.assertFalse(replay_accepted)
        with self.assertRaises((KeyError, ValueError)):
            sync.bundle_for(replayed)

        try:
            sync.bundle_for(anchor)
        except (KeyError, ValueError):
            pass
        self.assertEqual(sync.drain_ready(), ())
        self.assertEqual(sync.close(), ())


if __name__ == "__main__":
    unittest.main()
