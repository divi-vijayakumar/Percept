from __future__ import annotations

import asyncio
import hashlib
import math
import struct
import unittest
from dataclasses import replace
from typing import Any

import percept.audio_input as audio_input_module
from percept import Percept
from percept.adapters import Verdict
from percept.audio_input import (
    AudioInputCapability,
    audio_frame_from_units,
    audio_script_token,
    canonicalize_audio_frame,
    canonicalize_audio_unit,
)
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame, WatchPlan
from percept.events import AudioBoundary, AudioUnit
from percept.goals import GoalOp
from percept.harness.tier0 import CivilTime
from percept.runtime.worker import RuntimeStats, run
from percept.runtime.envelope import SourceEnvelope, SourceFormat
from percept.tracing import (
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGMENT_EVENT,
    STAGE_SOURCE_AUDIO,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)
from percept.watchplan_apply import apply_watch_plan


CAP = AudioInputCapability(modes=frozenset({"audio_only"}))


def frame(
    samples: tuple[int, ...] = (32767,),
    *,
    start: float = 0.0,
    rate: int = 4,
    fmt: str = "s16le",
) -> AudioFrame:
    if fmt == "s16le":
        raw = struct.pack("<" + "h" * len(samples), *samples)
    elif fmt == "u8":
        raw = bytes(samples)
    else:
        raw = struct.pack("<" + "f" * len(samples), *samples)
    return AudioFrame(
        pcm=raw,
        sample_rate=rate,
        channels=1,
        sample_format=fmt,
        t_start=start,
        t_end=start + len(samples) / rate,
    )


def unit(
    seq: int,
    *,
    start: float = 0.0,
    samples: tuple[int, ...] = (32767,),
    event_t: float | None = None,
    energy: float = 1.0,
    boundaries: tuple[AudioBoundary, ...] = (),
) -> AudioUnit:
    chunk = frame(samples, start=start)
    return AudioUnit(
        seq=seq,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        event_t=chunk.t_end if event_t is None else event_t,
        chunk=chunk,
        energy=energy,
        boundaries=boundaries,
    )


#: Sample magnitude whose mean-square energy (8.38e-05) is REAL audible signal that still sits an
#: order of magnitude below the shipped acoustic wake bar — the regime the whole AUDIO-GATE-DEAF
#: corpus lived in (p99 chunk energy 0.0097, loudest 0.0277, against a bar of 0.09).
QUIET_SAMPLE = 300
QUIET_ENERGY = (QUIET_SAMPLE / 32768) ** 2


def QUIET_UNITS(n: int) -> list[AudioUnit]:
    """``n`` consecutive chunks that carry signal but never clear the wake bar."""
    return [
        unit(i, start=0.25 * i, samples=(QUIET_SAMPLE,), energy=QUIET_ENERGY)
        for i in range(n)
    ]


def canonical(candidate: AudioUnit, *, observed: float = 10.0) -> AudioUnit:
    return canonicalize_audio_unit(
        candidate,
        capability=CAP,
        source_id="audio",
        source_epoch=0,
        source_sequence=candidate.seq,
        observed_time=observed,
        source_time=candidate.t_end,
    )


def token_for(*units: AudioUnit) -> str:
    return audio_script_token(
        audio_frame_from_units(tuple(canonical(item) for item in units), capability=CAP)
    )


def audio_plan(goal_id: str = "audio", **kwargs: Any) -> WatchPlan:
    return WatchPlan(id=goal_id, condition="the sound matters", sense="audio", **kwargs)


def wav_bytes(data: bytes, *, rate: int = 4, channels: int = 1, bits: int = 16) -> bytes:
    align = channels * (bits // 8)
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * align, align, bits)
    body = (
        b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def riff_chunk(tag: bytes, payload: bytes, *, pad: bool = True) -> bytes:
    return (
        tag
        + struct.pack("<I", len(payload))
        + payload
        + (b"\0" if pad and len(payload) % 2 else b"")
    )


def wav_with_chunks(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


async def collect(core: Percept, source: Any, **kwargs: Any) -> list:
    return [
        incident
        async for incident in run(
            core,
            source=None,
            audio_source=source,
            audio_settle_s=0.0,
            audio_wake={"high": 0.1, "low": 0.05},
            **kwargs,
        )
    ]


class CanonicalAudioTests(unittest.TestCase):
    def test_supported_pcm_is_immutable_and_deterministic(self) -> None:
        raw = bytearray(struct.pack("<hhh", -32768, 0, 32767))
        source = AudioFrame(
            pcm=raw,
            sample_rate=3,
            channels=1,
            sample_format="s16le",
            t_start=1.0,
            t_end=2.0,
        )
        got = canonicalize_audio_frame(source, t_start=1.0, t_end=2.0, capability=CAP)
        raw[:] = b"\0" * len(raw)
        self.assertEqual(got.pcm, struct.pack("<hhh", -32768, 0, 32767))
        self.assertEqual(
            canonicalize_audio_frame(
                frame((0, 128, 255), rate=3, fmt="u8"),
                t_start=0.0,
                t_end=1.0,
                capability=CAP,
            ).pcm,
            struct.pack("<hhh", -32768, 0, 32512),
        )
        f32 = frame((-1.0, 0.0, 1.0), rate=3, fmt="f32le")
        self.assertEqual(
            canonicalize_audio_frame(
                f32, t_start=0.0, t_end=1.0, capability=CAP
            ).pcm,
            struct.pack("<hhh", -32768, 0, 32767),
        )

    def test_carrier_metadata_and_float_attacks_reject(self) -> None:
        bad = [
            AudioFrame(pcm=b"", sample_rate=4, channels=1, sample_format="s16le"),
            AudioFrame(
                pcm=b"\0\0",
                encoded=b"x",
                mime="audio/wav",
                sample_rate=4,
                channels=1,
                sample_format="s16le",
            ),
            AudioFrame(pcm=b"\0", sample_rate=4, channels=1, sample_format="s16le"),
            AudioFrame(
                pcm=struct.pack("<f", math.nan),
                sample_rate=1,
                channels=1,
                sample_format="f32le",
                t_start=0.0,
                t_end=1.0,
            ),
        ]
        for candidate in bad:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    canonicalize_audio_frame(
                        candidate, t_start=0.0, t_end=1.0, capability=CAP
                    )

    def test_dual_clock_boundaries_remain_on_event_clock(self) -> None:
        candidate = unit(
            0,
            start=2.0,
            event_t=50.25,
            boundaries=(
                AudioBoundary("onset", 50.0),
                AudioBoundary("endpoint", 50.2),
            ),
        )
        got = canonicalize_audio_unit(
            candidate,
            capability=CAP,
            source_id="rtsp://secret@example.test/live",
            source_epoch=2,
            source_sequence=7,
            observed_time=50.25,
            source_time=2.25,
        )
        self.assertEqual((got.t_start, got.t_end, got.event_t), (2.0, 2.25, 50.25))
        self.assertEqual(tuple(item.t for item in got.boundaries), (50.0, 50.2))

    def test_forged_boundary_direction_and_provenance_reject(self) -> None:
        forged = unit(0)
        object.__setattr__(forged, "boundaries", [AudioBoundary("onset", 1.0)])
        with self.assertRaises(ValueError):
            canonical(forged)
        forged = unit(0)
        object.__setattr__(forged, "direction", "sideways")
        with self.assertRaises(ValueError):
            canonical(forged)
        with self.assertRaises(ValueError):
            canonicalize_audio_unit(
                unit(0),
                capability=CAP,
                source_id="",
                source_epoch=True,
                source_sequence=0,
                observed_time=1.0,
                source_time=0.25,
            )

    def test_strict_wav_length_mime_format_and_outer_metadata(self) -> None:
        raw = struct.pack("<h", 12)
        valid = wav_bytes(raw)
        encoded = AudioFrame(
            pcm=b"",
            encoded=valid,
            mime="audio/wav",
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=0.0,
            t_end=0.25,
        )
        self.assertEqual(
            canonicalize_audio_frame(
                encoded, t_start=0.0, t_end=0.25, capability=CAP
            ).pcm,
            raw,
        )
        attacks = [
            {**encoded.__dict__, "encoded": valid + b"x"},
            {**encoded.__dict__, "encoded": valid[:-1]},
            {**encoded.__dict__, "mime": "audio/wav; charset=binary"},
            {**encoded.__dict__, "sample_rate": True},
            {
                **encoded.__dict__,
                "encoded": wav_bytes(struct.pack("<f", 0.0), bits=32),
                "sample_format": "f32le",
                "t_end": 1.0,
            },
        ]
        for values in attacks:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    canonicalize_audio_frame(
                        AudioFrame(**values),
                        t_start=values["t_start"],
                        t_end=values["t_end"],
                        capability=CAP,
                    )

    def test_riff_chunk_structure_and_padding_are_strict(self) -> None:
        fmt = riff_chunk(
            b"fmt ", struct.pack("<HHIIHH", 1, 1, 4, 8, 2, 16)
        )
        data = riff_chunk(b"data", struct.pack("<h", 7))
        valid_with_metadata = wav_with_chunks(
            fmt, riff_chunk(b"JUNK", b"abc"), data
        )

        def encoded(raw: bytes) -> AudioFrame:
            return AudioFrame(
                pcm=b"",
                encoded=raw,
                mime="audio/wav",
                sample_rate=4,
                channels=1,
                sample_format="s16le",
                t_start=0.0,
                t_end=0.25,
            )

        self.assertEqual(
            canonicalize_audio_frame(
                encoded(valid_with_metadata),
                t_start=0.0,
                t_end=0.25,
                capability=CAP,
            ).pcm,
            struct.pack("<h", 7),
        )
        attacks = (
            wav_with_chunks(fmt, fmt, data),
            wav_with_chunks(fmt, data, data),
            wav_with_chunks(fmt),
            wav_with_chunks(fmt, riff_chunk(b"data", b"")),
            wav_with_chunks(fmt, riff_chunk(b"JUNK", b"x", pad=False), data),
        )
        for raw in attacks:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    canonicalize_audio_frame(
                        encoded(raw), t_start=0.0, t_end=0.25, capability=CAP
                    )

    def test_input_and_predicted_output_caps_reject(self) -> None:
        tiny_input = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=2,
            max_provider_bytes=100,
        )
        with self.assertRaises(ValueError):
            canonicalize_audio_frame(
                frame((1, 2, 3), rate=3, fmt="u8"),
                t_start=0.0,
                t_end=1.0,
                capability=tiny_input,
            )
        tiny_output = AudioInputCapability(
            modes=frozenset({"audio_only"}),
            max_input_bytes=100,
            max_provider_bytes=49,
        )
        with self.assertRaises(ValueError):
            canonicalize_audio_frame(
                frame((1, 2, 3), rate=3, fmt="u8"),
                t_start=0.0,
                t_end=1.0,
                capability=tiny_output,
            )

    def test_capability_rejects_mutable_sets_bool_and_non_int_caps(self) -> None:
        attacks = (
            {"modes": {"audio_only"}},
            {
                "modes": frozenset({"audio_only"}),
                "pcm_formats": {"s16le"},
            },
            {
                "modes": frozenset({"audio_only"}),
                "encoded_mimes": {"audio/wav", "audio/x-wav"},
            },
            {"modes": frozenset({"audio_only"}), "max_input_bytes": True},
            {"modes": frozenset({"audio_only"}), "max_provider_bytes": 50.0},
        )
        for kwargs in attacks:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AudioInputCapability(**kwargs)


class BindingTests(unittest.TestCase):
    def test_exact_sdk_adapter_only_and_prepare_is_atomic(self) -> None:
        class Subclass(FakeVision):
            pass

        class MarkerLiar:
            SUPPORTS_FUSED_AUDIO_INPUT = True

            async def judge(self, *args: Any, **kwargs: Any) -> Verdict:
                return Verdict(True, 1.0)

        for adapter in (Subclass(), MarkerLiar()):
            core = Percept.create(vision=adapter, discover_plugins=False)
            receipt = apply_watch_plan(core, audio_plan())
            self.assertEqual(receipt.status, "rejected")
            self.assertEqual(core.goals.all(), [])
        shadow = FakeVision()
        shadow.judge = lambda *args, **kwargs: None  # type: ignore[method-assign]
        core = Percept.create(vision=shadow, discover_plugins=False)
        self.assertEqual(apply_watch_plan(core, audio_plan()).status, "rejected")
        self.assertEqual(core.goals.all(), [])

    def test_post_apply_core_instance_and_class_mutation_cannot_redirect(self) -> None:
        candidate = unit(0)
        token = token_for(candidate)
        fake = FakeVision(
            script={token: Verdict(True, 0.99, evidence="captured")}
        )
        core = Percept.create(vision=fake, discover_plugins=False)
        self.assertEqual(apply_watch_plan(core, audio_plan()).status, "applied")

        async def poisoned(*args: Any, **kwargs: Any) -> Verdict:
            raise AssertionError("dynamic lookup redirected immutable audio dispatch")

        core.vision = FakeVision()
        fake.judge = poisoned  # type: ignore[method-assign]
        original = FakeVision.judge
        FakeVision.judge = poisoned  # type: ignore[assignment]
        try:
            result = asyncio.run(
                core.perceive_audio_judged(
                    {"audio": (canonical(candidate),)},
                    t=0.25,
                    goal_ids=("audio",),
                )
            )
        finally:
            FakeVision.judge = original  # type: ignore[assignment]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].verdict.evidence, "captured")

    def test_compiled_fake_freezes_audio_token_helper_and_script_state(self) -> None:
        candidate = unit(0)
        token = token_for(candidate)
        fake = FakeVision(
            script={
                token: Verdict(True, 0.99, evidence="captured"),
                "redirected": Verdict(False, 0.99, evidence="redirected"),
            }
        )
        core = Percept.create(vision=fake, discover_plugins=False)
        self.assertEqual(apply_watch_plan(core, audio_plan()).status, "applied")
        binding = core._audio_judge_bindings["audio"]
        payload = audio_frame_from_units((canonical(candidate),), capability=CAP)
        original_helper = audio_input_module.audio_script_token
        helper_calls = 0

        def poisoned_helper(frame: AudioFrame) -> str:
            nonlocal helper_calls
            helper_calls += 1
            return "redirected"

        audio_input_module.audio_script_token = poisoned_helper
        fake._script[token].satisfied = False
        fake._script[token].evidence = "mutated-live-script"
        fake._script = {"redirected": Verdict(False, 1.0, "replacement")}
        try:
            first = asyncio.run(
                binding.dispatch("sound", None, audio=payload)
            )
            first.evidence = "consumer-mutated"
            first.usage["poison"] = True
            second = asyncio.run(
                binding.dispatch("sound", None, audio=payload)
            )
        finally:
            audio_input_module.audio_script_token = original_helper

        self.assertEqual(helper_calls, 0)
        self.assertTrue(second.satisfied)
        self.assertEqual(second.evidence, "captured")
        self.assertEqual(second.usage, {})

    def test_complete_releases_audio_binding(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        self.assertEqual(apply_watch_plan(core, audio_plan()).status, "applied")
        self.assertIn("audio", core._audio_judge_bindings)
        core.goals.apply([GoalOp(op="complete", goal_id="audio")])
        self.assertNotIn("audio", core._audio_judge_bindings)

    def test_class_mutation_before_prepare_is_rejected(self) -> None:
        original = FakeVision.judge

        async def poisoned(*args: Any, **kwargs: Any) -> Verdict:
            return Verdict(True, 1.0)

        FakeVision.judge = poisoned  # type: ignore[assignment]
        try:
            core = Percept.create(vision=FakeVision(), discover_plugins=False)
            receipt = apply_watch_plan(core, audio_plan())
        finally:
            FakeVision.judge = original  # type: ignore[assignment]
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(core.goals.all(), [])

    def test_cached_reapply_retains_captured_adapter_after_core_swap(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        plan = audio_plan()
        first = apply_watch_plan(core, plan)
        captured = core._audio_judge_bindings["audio"]
        core.vision = FakeVision()
        second = apply_watch_plan(core, plan)
        self.assertEqual(second.status, first.status)
        self.assertIs(core._audio_judge_bindings["audio"], captured)

    def test_binding_is_frozen_and_identity_specific(self) -> None:
        first = Percept.create(vision=FakeVision(), discover_plugins=False)
        second = Percept.create(vision=FakeVision(), discover_plugins=False)
        apply_watch_plan(first, audio_plan())
        apply_watch_plan(second, audio_plan())
        one = first._audio_judge_bindings["audio"]
        two = second._audio_judge_bindings["audio"]
        self.assertIsNot(one.adapter, two.adapter)
        self.assertIsNot(one.binding_token, two.binding_token)
        with self.assertRaises((AttributeError, TypeError)):
            one.adapter = two.adapter


class AudioOnlyRuntimeTests(unittest.TestCase):
    def core_for(self, script: dict[str, Verdict], plan: WatchPlan | None = None) -> Percept:
        core = Percept.create(
            vision=FakeVision(script=script), discover_plugins=False
        )
        self.assertEqual(apply_watch_plan(core, plan or audio_plan()).status, "applied")
        return core

    def test_audio_only_fire_retains_exact_judged_evidence(self) -> None:
        candidate = unit(0)
        core = self.core_for({token_for(candidate): Verdict(True, 0.99, evidence="yes")})
        incidents = asyncio.run(collect(core, [candidate], judge_dispatch="inline"))
        self.assertEqual(len(incidents), 1)
        snippet = incidents[0].snippet
        self.assertEqual(snippet.kind, "audio")
        self.assertEqual((snippet.t_start, snippet.t_end), (0.0, 0.25))
        self.assertEqual(len(snippet.frames), 1)
        self.assertEqual(snippet.frames[0].source_sequence, 0)

    def test_aggregate_provider_cap_trims_before_admission_and_keeps_newest_unit(self) -> None:
        quiet = unit(0, start=0.0, energy=0.0)
        wake = unit(1, start=0.25, energy=1.0)
        core = self.core_for({token_for(wake): Verdict(True, 0.99)})
        original = core._audio_judge_bindings["audio"]
        bounded = replace(
            original,
            capability=AudioInputCapability(
                modes=frozenset({"audio_only", "fused"}),
                max_input_bytes=100,
                max_provider_bytes=47,
            ),
        )
        core._audio_judge_bindings["audio"] = bounded
        core._audio_judge_binding_tokens["audio"] = bounded.binding_token
        health = []
        stats = RuntimeStats()

        incidents = asyncio.run(
            collect(
                core,
                [quiet, wake],
                judge_dispatch="inline",
                stats=stats,
                on_health=health.append,
            )
        )

        self.assertEqual(len(incidents), 1)
        self.assertEqual([item.seq for item in incidents[0].snippet.frames], [1])
        capped = [item for item in health if item.kind == "audio_evidence_capped"]
        self.assertEqual(len(capped), 1)
        self.assertEqual(capped[0].data["dropped_units"], 1)
        self.assertEqual((stats.judge_calls, stats.perceive_dispatches), (1, 1))

    def test_missing_source_and_unsupported_options_fail_before_mutation(self) -> None:
        core = self.core_for({})
        before_revision = core.goals.revision
        before_state = dict(core.goals.get("audio").state)

        async def missing() -> None:
            async for _ in run(core, source=None, audio_source=None):
                pass

        with self.assertRaisesRegex(ValueError, "requires audio_source"):
            asyncio.run(missing())
        self.assertEqual(core.audio_windows, {})
        self.assertEqual(core.goals.revision, before_revision)
        self.assertEqual(core.goals.get("audio").state, before_state)

        async def unsupported() -> None:
            async for _ in run(
                core,
                source=None,
                audio_source=[],
                persistence=object(),
            ):
                pass

        with self.assertRaisesRegex(ValueError, "does not support"):
            asyncio.run(unsupported())
        self.assertEqual(core.audio_windows, {})

    def test_malformed_carrier_does_not_advance_then_valid_fires(self) -> None:
        bad = unit(99)
        object.__setattr__(bad.chunk, "sample_rate", True)
        good = unit(0)
        core = self.core_for({token_for(good): Verdict(True, 0.99)})
        health = []
        incidents = asyncio.run(
            collect(
                core,
                [bad, good],
                judge_dispatch="inline",
                on_health=health.append,
            )
        )
        self.assertEqual(len(incidents), 1)
        self.assertTrue(any(item.kind == "audio_unit_rejected" for item in health))
        self.assertEqual(incidents[0].snippet.frames[0].source_sequence, 0)

    def test_malformed_legacy_shape_and_wrong_lane_do_not_poison_next_carrier(self) -> None:
        good = unit(0)
        wrong_lane = SourceEnvelope(
            source_id="video",
            modality="video",
            epoch=0,
            sequence=0,
            observed_time=1.0,
            source_time=1.0,
            format=SourceFormat(media_type="application/x-percept-opaque"),
            payload=object(),
        )
        core = self.core_for({token_for(good): Verdict(True, 0.99)})
        health = []
        incidents = asyncio.run(
            collect(
                core,
                [("bad",), wrong_lane, good],
                judge_dispatch="inline",
                on_health=health.append,
            )
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].snippet.frames[0].source_sequence, 0)
        self.assertEqual(
            sum(item.kind == "audio_unit_rejected" for item in health), 2
        )

    def test_explicit_wall_clock_does_not_collapse_source_clock(self) -> None:
        candidate = unit(0, start=2.0, event_t=50.25)
        core = self.core_for({token_for(candidate): Verdict(True, 0.99)})
        incidents = asyncio.run(
            collect(
                core,
                [candidate],
                judge_dispatch="inline",
                timebase="wall",
                wall_clock=lambda: 50.25,
            )
        )
        self.assertEqual(incidents[0].ts, 50.25)
        retained = incidents[0].snippet.frames[0]
        self.assertEqual((retained.t_start, retained.t_end), (2.0, 2.25))
        self.assertEqual(retained.observed_time, 50.25)
        self.assertEqual(retained.source_time, 2.25)
        retained = incidents[0].snippet.frames[0]
        self.assertEqual((retained.source_time, retained.t_end), (2.25, 2.25))

    def _suppression_test(self, keyword: str) -> None:
        first = unit(0, start=0.0, samples=(10000,))
        skipped = unit(1, start=0.25)
        resumed = unit(2, start=0.5, samples=(32767,))
        core = self.core_for(
            {
                token_for(first): Verdict(False, 0.99),
                token_for(resumed): Verdict(True, 0.99),
            }
        )
        states = iter((False, True, False))
        incidents = asyncio.run(
            collect(
                core,
                [first, skipped, resumed],
                judge_dispatch="inline",
                **{keyword: lambda: next(states)},
            )
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual([item.seq for item in incidents[0].snippet.frames], [2])

    def test_pause_resume_is_a_continuity_boundary(self) -> None:
        self._suppression_test("paused")

    def test_mute_resume_is_a_continuity_boundary(self) -> None:
        self._suppression_test("muted")

    # --- AUDIO-SUPPRESSED-NO-RECEIPT -------------------------------------------------------
    # A unit offered while muted()/paused() used to be `continue`d with no counter and no health
    # event, while source_coverage still reported excluded: 0 — the unit simply vanished from every
    # ledger the runtime publishes. These pin that every offered unit now has exactly one term.

    def _suppression_ledger(self, keyword: str) -> tuple[list, dict]:
        first = unit(0, start=0.0, samples=(10000,))
        skipped_a = unit(1, start=0.25)
        skipped_b = unit(2, start=0.5)
        resumed = unit(3, start=0.75, samples=(32767,))
        core = self.core_for(
            {
                token_for(first): Verdict(False, 0.99),
                token_for(resumed): Verdict(True, 0.99),
            }
        )
        states = iter((False, True, True, False))
        health: list = []
        asyncio.run(
            collect(
                core,
                [first, skipped_a, skipped_b, resumed],
                judge_dispatch="inline",
                on_health=health.append,
                **{keyword: lambda: next(states)},
            )
        )
        ended = [event for event in health if event.kind == "runtime_ended"][-1]
        return health, ended.data

    def test_suppressed_units_are_counted_and_the_ingress_ledger_reconciles(self) -> None:
        for keyword in ("muted", "paused"):
            with self.subTest(keyword=keyword):
                health, data = self._suppression_ledger(keyword)
                self.assertEqual(data["audio_chunks"], 2)
                self.assertEqual(data["audio_suppressed"], 2)
                self.assertEqual(data["audio_rejected"], 0)
                self.assertEqual(data["audio_offered"], 4)
                # The invariant, stated exactly: nothing offered may go unaccounted for.
                self.assertEqual(
                    data["audio_offered"],
                    data["audio_chunks"] + data["audio_suppressed"] + data["audio_rejected"],
                )
                # ... and the modality's own coverage agrees rather than reporting excluded: 0.
                coverage = data["source_coverage"]["audio"]
                self.assertEqual(coverage["excluded"], 2)
                self.assertEqual(coverage["observed"] + coverage["excluded"], 4)

    def test_suppression_window_edges_are_receipted_with_reason_and_size(self) -> None:
        for keyword in ("muted", "paused"):
            with self.subTest(keyword=keyword):
                health, _ = self._suppression_ledger(keyword)
                started = [e for e in health if e.kind == "audio_suppression_started"]
                ended = [e for e in health if e.kind == "audio_suppression_ended"]
                self.assertEqual(len(started), 1)
                self.assertEqual(len(ended), 1)
                self.assertEqual(started[0].data["reason"], keyword)
                # The window's exact size, reported once, rather than one event per swallowed chunk.
                self.assertEqual(ended[0].data["suppressed_chunks"], 2)

    def test_malformed_carrier_is_excluded_in_coverage_not_only_receipted(self) -> None:
        # The refusal decision is taken on a preview probe, which mutates no coverage — so this used
        # to receipt an `audio_unit_rejected` event while leaving excluded at 0.
        good = unit(0, start=0.0, samples=(32767,))
        core = self.core_for({token_for(good): Verdict(False, 0.99)})
        health: list = []
        asyncio.run(
            collect(
                core,
                [good, ("not", "an", "ingress")],
                judge_dispatch="inline",
                on_health=health.append,
            )
        )
        data = [event for event in health if event.kind == "runtime_ended"][-1].data
        self.assertEqual(
            [e.kind for e in health].count("audio_unit_rejected"), 1
        )
        self.assertEqual(data["audio_rejected"], 1)
        self.assertEqual(data["audio_chunks"], 1)
        self.assertEqual(data["audio_offered"], 2)
        coverage = data["source_coverage"]["audio"]
        self.assertEqual(coverage["excluded"], 1)
        self.assertEqual(coverage["observed"] + coverage["excluded"], 2)

    # --- AUDIO-GATE-DEAF -------------------------------------------------------------------
    # A gate that never wakes across a whole run must SAY SO. Without this, "the room was quiet" and
    # "my gate could not hear" are byte-for-byte identical to the caller.

    def _run_with_gate(self, units: list, **kwargs: Any) -> list:
        health: list = []
        asyncio.run(
            self._drain(
                self.core_for({}),
                units,
                on_health=health.append,
                **kwargs,
            )
        )
        return health

    @staticmethod
    async def _drain(core: Percept, source: Any, **kwargs: Any) -> list:
        return [
            incident
            async for incident in run(
                core,
                source=None,
                audio_source=source,
                audio_settle_s=0.0,
                judge_dispatch="inline",
                **kwargs,
            )
        ]

    def test_gate_that_never_wakes_emits_zero_wake_coverage(self) -> None:
        # Audible-but-quiet material: every chunk carries real signal, none of it clears the bar.
        # This is the corpus's own shape — its p99 chunk energy was 0.0097 and its loudest 0.0277.
        health = self._run_with_gate(QUIET_UNITS(4))
        events = [e for e in health if e.kind == "zero_wake_coverage"]
        self.assertEqual(len(events), 1, [e.kind for e in health])
        data = events[0].data
        self.assertEqual(data["detector"], "acoustic-onset")
        self.assertEqual(data["wakes"], 0)
        self.assertEqual(data["chunks_observed"], 4)
        self.assertTrue(data["at_end"])
        # The gate's own setting is reported beside what the signal actually did — the pair that
        # makes the two readings separable from outside the process.
        self.assertGreater(data["wake_threshold"], 0.0)
        self.assertEqual(data["rearm_threshold"], data["wake_threshold"] / 4.0)
        self.assertGreater(data["energy_max"], 0.0)
        self.assertLess(data["energy_max"], data["wake_threshold"])
        self.assertAlmostEqual(
            data["loudest_fraction_of_threshold"],
            data["energy_max"] / data["wake_threshold"],
        )
        # It is emitted BEFORE runtime_ended, the way the video path orders its zero-admission twin.
        kinds = [e.kind for e in health]
        self.assertLess(kinds.index("zero_wake_coverage"), kinds.index("runtime_ended"))

    def test_zero_wake_coverage_reports_wakes_in_runtime_ended_too(self) -> None:
        health = self._run_with_gate(QUIET_UNITS(3))
        data = [e for e in health if e.kind == "runtime_ended"][-1].data
        self.assertEqual(data["audio_wakes"], 0)

    def test_a_gate_that_does_wake_never_reports_deafness(self) -> None:
        loud = [
            unit(0, start=0.0, samples=(0,), energy=0.0),
            unit(1, start=0.25, samples=(32767,)),
        ]
        health = self._run_with_gate(loud)
        self.assertNotIn("zero_wake_coverage", [e.kind for e in health])
        data = [e for e in health if e.kind == "runtime_ended"][-1].data
        self.assertGreaterEqual(data["audio_wakes"], 1)

    def test_no_gate_configured_never_reports_deafness(self) -> None:
        # audio_gate=None is a documented fixed-cadence mode: there is no gate that could be deaf,
        # so a run with zero wakes is not a fact about anything and must stay silent.
        core = self.core_for({})
        core.goals.get("audio").state["cadence"] = {
            "min_judge_interval": 10.0,
            "gate_on": "fixed",
        }
        health: list = []
        asyncio.run(
            self._drain(
                core,
                QUIET_UNITS(3),
                audio_gate=None,
                on_health=health.append,
            )
        )
        self.assertNotIn("zero_wake_coverage", [e.kind for e in health])
        self.assertEqual(
            [e for e in health if e.kind == "runtime_ended"][-1].data["audio_wakes"], 0
        )

    def test_quiet_media_units_advance_deadline_without_judge(self) -> None:
        plan = audio_plan(
            rearm="deadline",
            explicit={
                "Goal.state.rearm": "deadline",
                "Goal.state.deadline_after_s": 1.0,
            },
        )
        first = unit(0, samples=(0,), energy=0.0)
        second = unit(1, start=0.25, samples=(0, 0, 0, 0), energy=0.0)
        core = self.core_for({}, plan)
        incidents = asyncio.run(
            collect(core, [first, second], judge_dispatch="inline")
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].concern_id, "audio")
        self.assertEqual(core.goals.get("audio").state["deadline_clock_domain"], "audio_event")

    def test_async_path_is_bounded_and_flushes_on_normal_eos(self) -> None:
        candidate = unit(0)
        core = self.core_for({token_for(candidate): Verdict(True, 0.99)})
        stats = RuntimeStats()
        incidents = asyncio.run(
            collect(core, [candidate], judge_dispatch="async", stats=stats)
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(stats.judge_calls, 1)

    def test_absolute_deadline_blocks_late_audio_cognition_commit(self) -> None:
        candidate = unit(0)
        core = self.core_for(
            {token_for(candidate): Verdict(True, 0.99)}
        )
        stats = RuntimeStats()
        health = []

        async def drive() -> list:
            started = asyncio.Event()
            release = asyncio.Event()
            original = core.perceive_audio_judged

            async def delayed(*args: Any, **kwargs: Any):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                return await original(*args, **kwargs)

            core.perceive_audio_judged = delayed  # type: ignore[method-assign]

            async def source():
                yield candidate
                await started.wait()
                while stats.judge_deadline_expired == 0:
                    await asyncio.sleep(0.001)
                self.assertNotIn(("audio", None), core.gate._armed)
                self.assertEqual(len(core.action_log), 0)
                release.set()

            return await collect(
                core,
                source(),
                judge_dispatch="async",
                judge_decision_deadline_s=0.01,
                judge_stall_after_s=0.2,
                stats=stats,
                on_health=health.append,
            )

        incidents = asyncio.run(drive())
        self.assertEqual(incidents, [])
        self.assertNotIn(("audio", None), core.gate._armed)
        self.assertEqual(len(core.action_log), 0)
        self.assertEqual(stats.judge_deadline_expired, 1)
        self.assertTrue(
            any(item.kind == "judge_deadline_expired" for item in health)
        )

    def test_trace_joins_source_audio_judgment_and_incident(self) -> None:
        candidate = unit(0)
        core = self.core_for({token_for(candidate): Verdict(True, 0.99)})
        recorder = TraceRecorder(
            RunTraceManifest.create(
                source="camera",
                driver="audio-only",
                goals=list(core.goals.active()),
                policy=TracePolicy(level="full"),
            )
        )
        incidents = asyncio.run(
            collect(
                core,
                [candidate],
                judge_dispatch="inline",
                on_trace=recorder.tracer(),
            )
        )
        recorder.close()
        source = [row for row in recorder.records if row.stage == STAGE_SOURCE_AUDIO]
        judgments = [
            row for row in recorder.records if row.stage == STAGE_JUDGMENT_EVENT
        ]
        created = [
            row for row in recorder.records if row.stage == STAGE_INCIDENT_CREATED
        ]
        self.assertEqual((len(source), len(judgments), len(created)), (1, 1, 1))
        traced_payload = source[0].data["provider_payloads"]["audio"]
        snippet = incidents[0].snippet
        self.assertEqual(traced_payload["mime"], snippet.mime)
        self.assertEqual(traced_payload["bytes"], len(snippet.media))
        self.assertEqual(
            traced_payload["sha256"],
            hashlib.sha256(
                snippet.mime.lower().encode("ascii") + b"\0" + snippet.media
            ).hexdigest(),
        )
        self.assertEqual(
            (traced_payload["t_start"], traced_payload["t_end"]),
            (snippet.t_start, snippet.t_end),
        )
        self.assertEqual(judgments[0].data["evidence_seq"], source[0].seq)
        self.assertEqual(
            created[0].data["judgment_event_id"],
            judgments[0].data["judgment_event_id"],
        )
        self.assertEqual(
            incidents[0].judgment_event_id,
            judgments[0].data["judgment_event_id"],
        )

    def test_cancellation_resistant_completion_after_gap_cannot_commit(self) -> None:
        first = unit(0, start=0.0)
        after_gap = unit(2, start=1.0)
        core = self.core_for({token_for(first): Verdict(True, 0.99)})

        async def drive() -> list:
            started = asyncio.Event()
            release = asyncio.Event()
            original = core.perceive_audio_judged

            async def delayed(*args: Any, **kwargs: Any):
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # Simulate a provider transport that suppresses task cancellation.
                    await release.wait()
                return await original(*args, **kwargs)

            core.perceive_audio_judged = delayed  # type: ignore[method-assign]

            async def source():
                yield first
                await started.wait()
                yield after_gap
                release.set()

            return await collect(core, source(), judge_dispatch="async")

        incidents = asyncio.run(drive())
        self.assertEqual(len(incidents), 1)
        self.assertEqual([row.seq for row in incidents[0].snippet.frames], [2])

    def test_completed_result_before_gap_is_delivered_at_completion_linearization(self) -> None:
        first = unit(0, start=0.0)
        after_gap = unit(2, start=1.0)
        core = self.core_for({token_for(first): Verdict(True, 0.99)})
        stats = RuntimeStats()

        async def drive() -> list:
            completed = asyncio.Event()
            original = core.perceive_audio_judged

            async def observed(*args: Any, **kwargs: Any):
                result = await original(*args, **kwargs)
                completed.set()
                return result

            core.perceive_audio_judged = observed  # type: ignore[method-assign]

            async def source():
                yield first
                await completed.wait()
                await asyncio.sleep(0)
                yield after_gap

            return await collect(
                core, source(), judge_dispatch="async", stats=stats
            )

        self.assertEqual(len(asyncio.run(drive())), 1)
        self.assertEqual(stats.judge_source_discontinuity_dropped, 0)

    def test_completed_result_before_goal_revision_is_delivered_at_completion(self) -> None:
        first = unit(0)
        core = self.core_for({token_for(first): Verdict(True, 0.99)})
        stats = RuntimeStats()

        async def drive() -> list:
            completed = asyncio.Event()
            original = core.perceive_audio_judged

            async def observed(*args: Any, **kwargs: Any):
                result = await original(*args, **kwargs)
                completed.set()
                return result

            core.perceive_audio_judged = observed  # type: ignore[method-assign]

            async def source():
                yield first
                await completed.wait()
                await asyncio.sleep(0)
                core.apply_goal_ops(
                    [
                        GoalOp(
                            op="update",
                            goal_id="audio",
                            fields={"say": "updated"},
                        )
                    ]
                )

            return await collect(
                core, source(), judge_dispatch="async", stats=stats
            )

        self.assertEqual(len(asyncio.run(drive())), 1)
        self.assertEqual(stats.judge_revision_dropped, 0)

    def test_cancellation_does_not_flush_or_deliver_inflight_result(self) -> None:
        candidate = unit(0)
        core = self.core_for({token_for(candidate): Verdict(True, 0.99)})

        async def drive() -> list:
            started = asyncio.Event()
            release = asyncio.Event()
            original = core.perceive_audio_judged

            async def delayed(*args: Any, **kwargs: Any):
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return await original(*args, **kwargs)

            core.perceive_audio_judged = delayed  # type: ignore[method-assign]

            async def source():
                yield candidate
                await asyncio.Event().wait()

            delivered = []
            task = asyncio.create_task(
                collect(
                    core,
                    source(),
                    judge_dispatch="async",
                    on_incident=delivered.append,
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            for _ in range(4):
                await asyncio.sleep(0)
            self.assertNotIn(("audio", None), core.gate._armed)
            self.assertEqual(len(core.action_log), 0)
            core.perceive_audio_judged = original  # type: ignore[method-assign]
            subsequent = await collect(
                core, [candidate], judge_dispatch="inline"
            )
            return delivered, subsequent

        delivered, subsequent = asyncio.run(drive())
        self.assertEqual(delivered, [])
        self.assertEqual(len(subsequent), 1)

    def test_multi_watch_evidence_and_schedule_are_isolated(self) -> None:
        candidate = unit(0)
        token = token_for(candidate)
        core = Percept.create(
            vision=FakeVision(script={token: Verdict(True, 0.99)}),
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            [
                audio_plan("day-one", schedule={"days": [1]}),
                audio_plan("day-two", schedule={"days": [2]}),
            ],
        )
        self.assertEqual(receipt.status, "applied")
        captured = []
        original = core.perceive_audio_judged

        async def inspect_mapping(mapping: Any, *args: Any, **kwargs: Any):
            captured.append(mapping)
            return await original(mapping, *args, **kwargs)

        core.perceive_audio_judged = inspect_mapping  # type: ignore[method-assign]
        incidents = asyncio.run(
            collect(
                core,
                [candidate],
                judge_dispatch="inline",
                clock=lambda: CivilTime(weekday=1, minute_of_day=600),
            )
        )
        self.assertEqual([item.concern_id for item in incidents], ["day-one"])
        self.assertEqual(tuple(captured[0]), ("day-one",))

    def test_multi_watch_jobs_freeze_separate_evidence_tuples(self) -> None:
        candidate = unit(0)
        core = Percept.create(
            vision=FakeVision(
                script={token_for(candidate): Verdict(True, 0.99)}
            ),
            discover_plugins=False,
        )
        self.assertEqual(
            apply_watch_plan(
                core, [audio_plan("first"), audio_plan("second")]
            ).status,
            "applied",
        )
        captured = []
        original = core.perceive_audio_judged

        async def inspect_mapping(mapping: Any, *args: Any, **kwargs: Any):
            captured.append(mapping)
            return await original(mapping, *args, **kwargs)

        core.perceive_audio_judged = inspect_mapping  # type: ignore[method-assign]
        incidents = asyncio.run(
            collect(core, [candidate], judge_dispatch="inline")
        )
        self.assertEqual(
            [item.concern_id for item in incidents], ["first", "second"]
        )
        mapping = captured[0]
        self.assertEqual(tuple(mapping), ("first", "second"))
        self.assertIsNot(mapping["first"], mapping["second"])
        self.assertEqual(mapping["first"], mapping["second"])


if __name__ == "__main__":
    unittest.main()
