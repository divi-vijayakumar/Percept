"""Carrier-independent audio-unit ordering and shared edge-energy parity."""

from __future__ import annotations

import json
import asyncio
import queue
import struct
import wave
from pathlib import Path

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.agent import _concat_audio
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.events import AudioBoundary, AudioUnit, AudioWindow
from percept.runtime import run
from percept.runtime.audio import (
    _put_latest,
    normalize_audio_envelope,
    normalize_audio_unit,
    open_audio_source,
    pcm_energy,
)
from percept.runtime.envelope import SourceEnvelope, SourceFormat, audio_unit_envelope

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "percept-edge"
        / "test"
        / "fixtures"
        / "audio_stream.golden.json"
    ).read_text()
)


def _pcm(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _unit(row: dict) -> AudioUnit:
    boundaries = tuple(AudioBoundary(item["kind"], item["t"]) for item in row.get("boundaries", ()))
    return AudioUnit(
        seq=row["seq"],
        t_start=row["t_start"],
        t_end=row["t_end"],
        chunk=AudioFrame(
            pcm=b"\x00\x00",
            sample_rate=row.get("sample_rate", 16_000),
            channels=row.get("channels", 1),
        ),
        sample_rate=row.get("sample_rate"),
        channels=row.get("channels"),
        direction=row.get("direction", "unknown"),
        participant_id=row.get("participant_id"),
        boundaries=boundaries,
    )


@pytest.mark.parametrize("vector", FIXTURE["energy_vectors"], ids=lambda row: row["name"])
def test_shared_pcm_energy_vectors(vector: dict) -> None:
    assert pcm_energy(_pcm(vector["samples"])) == pytest.approx(vector["expected"], abs=1e-12)
    if chunks := vector.get("chunks"):
        weighted = sum(pcm_energy(_pcm(chunk)) * len(chunk) for chunk in chunks) / sum(
            len(chunk) for chunk in chunks
        )
        assert weighted == pytest.approx(vector["expected"], abs=1e-12)


@pytest.mark.parametrize("vector", FIXTURE["timeline_vectors"], ids=lambda row: row["name"])
def test_shared_ordered_timeline_vectors(vector: dict) -> None:
    timeline = AudioWindow(seconds=10)
    results = [timeline.push_unit(_unit(row)).as_dict() for row in vector["units"]]
    for actual, expected in zip(results, vector["expected"]):
        for key, value in expected.items():
            if isinstance(value, float):
                assert actual[key] == pytest.approx(value, abs=1e-12)
            else:
                assert actual[key] == value
    assert [unit.seq for unit in timeline.units()] == vector["order"]
    if "directions" in vector:
        assert [unit.direction for unit in timeline.units()] == vector["directions"]
    if "participants" in vector:
        assert [unit.participant_id for unit in timeline.units()] == vector["participants"]
    if "boundary_kinds" in vector:
        assert [boundary.kind for boundary in timeline.boundaries()] == vector["boundary_kinds"]


def test_explicit_window_returns_interval_intersections() -> None:
    timeline = AudioWindow(seconds=10)
    for seq, start in enumerate((0.0, 0.1, 0.2)):
        timeline.push_unit(
            AudioUnit(seq, start, start + 0.1, AudioFrame(pcm=b"\x00\x00"))
        )
    assert [unit.seq for unit in timeline.window(0.05, 0.15)] == [0, 1]


def test_exact_100ms_clip_uses_half_open_lower_boundary() -> None:
    timeline = AudioWindow(seconds=10)
    first = AudioFrame(pcm=b"\x00\x00" * 1600, sample_rate=16_000)
    second = AudioFrame(pcm=b"\x01\x00" * 1600, sample_rate=16_000)
    timeline.push(0.1, first)
    timeline.push(0.2, second)
    assert timeline.clip(0.1) == [second]


def test_public_window_push_normalizes_typed_frame_duration_and_energy() -> None:
    timeline = AudioWindow(seconds=10)
    frame = AudioFrame(pcm=b"\x00\x20" * 160, sample_rate=16_000)
    result = timeline.push(1.0, frame)
    unit = timeline.units()[0]
    assert result.accepted
    assert (unit.t_start, unit.t_end) == pytest.approx((0.99, 1.0))
    assert unit.energy == pytest.approx(0.0625)
    assert (unit.sample_rate, unit.channels, unit.sample_format) == (16_000, 1, "s16le")


def test_buffer_is_bounded_under_deterministic_overload() -> None:
    timeline = AudioWindow(seconds=1000, max_units=8, max_bytes=16)
    evicted = 0
    for seq in range(1000):
        result = timeline.push_unit(
            AudioUnit(seq, float(seq), seq + 0.1, AudioFrame(pcm=b"\x00\x00"))
        )
        evicted += result.evicted
    assert len(timeline) == 8
    assert timeline.nbytes == 16
    assert evicted == 992
    assert timeline.units()[0].seq == 992


def test_capture_queue_drop_oldest_is_bounded_and_counted() -> None:
    q: queue.Queue = queue.Queue(maxsize=2)
    dropped = [0]
    for value in range(20):
        _put_latest(q, value, dropped)
    assert [q.get_nowait(), q.get_nowait()] == [18, 19]
    assert dropped == [18]


@pytest.mark.parametrize(
    "unit",
    [
        AudioUnit(-1, 0.0, 0.1, b""),
        AudioUnit(0, float("nan"), 0.1, b""),
        AudioUnit(0, 0.2, 0.1, b""),
        AudioUnit(0, 0.0, 0.1, b"", energy=1.1),
        AudioUnit(0, 0.0, 0.1, b"", sample_rate=0),
        AudioUnit(0, 0.0, 0.1, b"", channels=0),
    ],
)
def test_invalid_units_fail_closed(unit: AudioUnit) -> None:
    with pytest.raises(ValueError):
        AudioWindow().push_unit(unit)


def test_odd_s16_pcm_and_oversized_units_fail_before_window_mutation() -> None:
    with pytest.raises(ValueError, match="even"):
        pcm_energy(b"\x00")
    timeline = AudioWindow(max_bytes=2)
    with pytest.raises(ValueError, match="byte cap"):
        timeline.push_unit(AudioUnit(0, 0.0, 0.1, AudioFrame(pcm=b"\x00\x00\x00\x00")))
    assert timeline.units() == []
    assert timeline.nbytes == 0
    encoded = AudioFrame(pcm=b"", encoded=b"RIFF", mime="audio/wav")
    with pytest.raises(ValueError, match="byte cap"):
        timeline.push_unit(AudioUnit(0, 0.0, 0.1, encoded))


def test_declared_raw_pcm_alignment_and_mutable_buffer_copy() -> None:
    timeline = AudioWindow(seconds=10)
    mutable = bytearray(b"\x01\x00\x02\x00")
    timeline.push_unit(
        AudioUnit(
            0,
            0.0,
            0.1,
            mutable,
            sample_rate=16_000,
            channels=1,
            sample_format="s16le",
        )
    )
    mutable[0] = 9
    retained = timeline.units()[0]
    assert retained.chunk == b"\x01\x00\x02\x00"
    assert isinstance(retained.chunk, bytes)

    with pytest.raises(ValueError, match="aligned"):
        timeline.push_unit(
            AudioUnit(
                1,
                0.1,
                0.2,
                bytearray(b"\x00\x00\x00"),
                sample_rate=16_000,
                channels=1,
                sample_format="s16le",
            )
        )


def test_typed_frame_mutable_pcm_is_detached_on_ingress() -> None:
    mutable = bytearray(b"\x01\x00")
    frame = AudioFrame(pcm=mutable, sample_rate=16_000)
    timeline = AudioWindow(seconds=10)
    timeline.push_unit(AudioUnit(0, 0.0, 0.1, frame))
    mutable[0] = 7
    retained = timeline.units()[0].chunk
    assert retained.pcm == b"\x01\x00"
    assert isinstance(retained.pcm, bytes)


def test_duplicate_history_survives_count_eviction() -> None:
    timeline = AudioWindow(seconds=100, max_units=2, max_bytes=100)
    for seq in range(3):
        timeline.push_unit(AudioUnit(seq, seq * 0.1, (seq + 1) * 0.1, b"x"))
    assert [unit.seq for unit in timeline.units()] == [1, 2]
    result = timeline.push_unit(AudioUnit(0, 0.0, 0.1, b"x"))
    assert not result.accepted
    assert result.duplicate
    assert [unit.seq for unit in timeline.units()] == [1, 2]


def test_windows_and_boundaries_use_half_open_end() -> None:
    timeline = AudioWindow(seconds=10)
    timeline.push_unit(
        AudioUnit(
            0,
            0.0,
            0.1,
            b"x",
            boundaries=(
                AudioBoundary("onset", 0.0),
                AudioBoundary("endpoint", 0.1),
            ),
        )
    )
    assert [boundary.kind for boundary in timeline.boundaries(0.0, 0.1)] == ["onset"]
    assert timeline.window(0.1, 0.2) == []


def test_audio_unit_preserves_legacy_tuple_protocol() -> None:
    unit = AudioUnit(2, 0.0, 0.1, b"x", event_t=7.0, energy=0.25)
    assert tuple(unit) == (7.0, b"x", 0.25)
    assert len(unit) == 3
    assert unit[0] == 7.0
    assert unit[-1] == 0.25


def test_audio_envelope_normalization_preserves_exact_unit_metadata_and_owned_bytes() -> None:
    mutable = bytearray(b"\x01\x00")
    submitted = AudioUnit(
        7,
        2.0,
        2.25,
        AudioFrame(pcm=mutable, sample_rate=16_000),
        event_t=50.0,
        energy=0.25,
        direction="input",
        participant_id="speaker",
        boundaries=(AudioBoundary("onset", 2.0),),
    )
    envelope = audio_unit_envelope(
        submitted,
        source_id="audio:push",
        sequence=0,
        observed_time=50.0,
    )
    mutable[0] = 9
    normalized = normalize_audio_envelope(envelope, event_t=2.25, unit=submitted)

    assert normalized.seq == 7
    assert (normalized.t_start, normalized.t_end, normalized.event_t) == (2.0, 2.25, 2.25)
    assert normalized.chunk.pcm == b"\x01\x00"
    assert normalized.energy == 0.25
    assert normalized.direction == "input"
    assert normalized.participant_id == "speaker"
    assert normalized.boundaries == (AudioBoundary("onset", 2.0),)


def test_plain_audio_envelope_derives_duration_and_format_for_ordered_window() -> None:
    frame = AudioFrame(pcm=b"\x00\x20" * 160, sample_rate=16_000, channels=1)
    envelope = SourceEnvelope(
        source_id="audio:capture",
        modality="audio",
        epoch=3,
        sequence=4,
        observed_time=100.0,
        source_time=3.0,
        format=SourceFormat(
            media_type="audio/pcm",
            sample_rate=16_000,
            channels=1,
            sample_format="s16le",
        ),
        payload=frame,
        signal=None,
    )
    unit = normalize_audio_envelope(envelope, event_t=3.0)

    assert unit.seq == 4
    assert (unit.t_start, unit.t_end) == pytest.approx((2.99, 3.0))
    assert (unit.sample_rate, unit.channels, unit.sample_format) == (16_000, 1, "s16le")
    assert unit.energy == pytest.approx(0.0625)


def test_legacy_push_normalizes_to_typed_unit_with_known_format() -> None:
    frame = AudioFrame(pcm=b"\x00\x20" * 160, sample_rate=16_000, channels=1)
    unit = normalize_audio_unit((3.0, frame, None), seq=7)
    assert unit.seq == 7
    assert unit.t_end == 3.0
    assert unit.t_start == pytest.approx(2.99)
    assert unit.sample_rate == 16_000
    assert unit.channels == 1
    assert unit.energy == pytest.approx(0.0625)


def test_percept_push_audio_uses_true_duration_and_exposes_legacy_gap() -> None:
    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    frame = AudioFrame(pcm=b"\x00\x20" * 160, sample_rate=16_000, channels=1)
    first = core.push_audio(frame, t=3.0)
    second = core.push_audio(frame, t=3.5)
    units = core._audio_window("camera").units()
    assert first.accepted and second.accepted
    assert units[0].t_start == pytest.approx(2.99)
    assert units[0].t_end == 3.0
    assert units[0].sample_rate == 16_000
    assert units[0].channels == 1
    assert units[0].sample_format == "s16le"
    assert units[0].energy == pytest.approx(0.0625)
    assert (units[1].t_start, units[1].t_end) == pytest.approx((3.49, 3.5))
    assert second.gap_s == pytest.approx(0.49)


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            AudioFrame(
                pcm=struct.pack("<2f", 0.5, -0.5),
                sample_rate=16_000,
                sample_format="f32le",
            ),
            0.25,
        ),
        (
            AudioFrame(
                pcm=bytes((128, 192, 64)),
                sample_rate=16_000,
                sample_format="u8",
            ),
            (0.0 + 0.25 + 0.25) / 3,
        ),
    ],
)
def test_push_audio_derives_energy_for_each_raw_pcm_format(
    frame: AudioFrame, expected: float
) -> None:
    core = Percept.create(vision=FakeVision(), discover_plugins=False)
    core.push_audio(frame, t=1.0)
    assert core._audio_window("camera").units()[0].energy == pytest.approx(expected)


def test_mixed_audio_formats_are_never_concatenated_or_mislabeled() -> None:
    mono = AudioFrame(pcm=b"\x00\x00", sample_rate=16_000, channels=1)
    stereo = AudioFrame(pcm=b"\x00\x00\x00\x00", sample_rate=16_000, channels=2)
    assert _concat_audio([mono, stereo]) is None
    assert _concat_audio([("audio/wav", b"a"), ("audio/ogg", b"b")]) is None
    assert _concat_audio([b"a", ("audio/wav", b"b")]) is None


def test_zero_capture_queue_size_is_rejected_instead_of_becoming_unbounded() -> None:
    async def consume_one():
        return await anext(open_audio_source("unused.wav", maxsize=0))

    with pytest.raises(ValueError, match="unbounded"):
        asyncio.run(consume_one())


def _runtime_core() -> Percept:
    core = Percept.create(
        vision=FakeVision(
            script={"quiet": Verdict(satisfied=False, confidence=0.95, evidence="quiet")}
        ),
        discover_plugins=False,
    )
    core.add_goal(Goal(id="sound", condition="a sound occurs", say="heard"))
    return core


def _run_audio_units(units, **kwargs):
    core = _runtime_core()
    still = gray_frame(8, 8, 0)
    utterances = []
    health = []

    async def audio():
        for unit in units:
            yield unit
            await asyncio.sleep(0.001)

    async def video():
        for seq in range(40):
            yield (seq * 0.1, still, "quiet")
            await asyncio.sleep(0.001)

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=video(),
                audio_source=audio(),
                audio_settle_s=0.0,
                audio_window_s=10.0,
                on_utterance=kwargs.pop("on_utterance", None),
                on_health=health.append,
                allow_scripted_brain=True,
                judge_dispatch="inline",
                **kwargs,
            )
        ]

    asyncio.run(collect())
    return core, utterances, health


def test_explicit_audio_unit_sequence_is_ordered_by_audio_window_not_source_tracker() -> None:
    units = [
        AudioUnit(1, 0.1, 0.2, b"newer", energy=0.0),
        AudioUnit(0, 0.0, 0.1, b"older", energy=0.0),
    ]
    core, _, health = _run_audio_units(units)

    assert [(unit.seq, unit.chunk) for unit in core._audio_window("camera").units()] == [
        (0, b"older"),
        (1, b"newer"),
    ]
    assert "source_reorder" not in [event.kind for event in health]
    ended = next(event for event in health if event.kind == "runtime_ended")
    assert ended.data["source_coverage"]["audio"] == {
        "observed": 2,
        "excluded": 0,
        "discontinuities": 0,
        "missing_sequences": 0,
    }


@pytest.mark.parametrize("discontinuity", ["gap", "format"])
def test_explicit_audio_gap_or_format_change_resets_dsp_but_retains_ordered_evidence(
    discontinuity: str,
) -> None:
    first = AudioUnit(
        0,
        0.0,
        0.1,
        AudioFrame(pcm=b"\x01\x00", sample_rate=16_000),
        energy=0.001,
    )
    if discontinuity == "gap":
        second = AudioUnit(
            2,
            0.2,
            0.3,
            AudioFrame(pcm=b"\x00\x00", sample_rate=16_000),
            energy=0.0,
        )
    else:
        second = AudioUnit(
            1,
            0.1,
            0.2,
            AudioFrame(
                pcm=b"\x00\x00\x00\x00",
                sample_rate=16_000,
                sample_format="f32le",
            ),
            energy=0.0,
        )
    core, _, health = _run_audio_units(
        [first, second],
        on_utterance=lambda *_: None,
        utterance_hangover_frames=1,
        utterance_min_speech_frames=1,
    )

    assert [unit.seq for unit in core._audio_window("camera").units()] == [
        first.seq,
        second.seq,
    ]
    assert [boundary.kind for boundary in core._audio_window("camera").boundaries()] == [
        "vad_start"
    ]
    assert "source_gap" not in [event.kind for event in health]
    assert "source_format_change" not in [event.kind for event in health]
    assert "audio_discontinuity" in [event.kind for event in health]
    assert core.source_continuity_generation("camera") == 0


def test_actual_source_envelope_reorder_is_excluded_before_live_audio_dsp() -> None:
    fmt = SourceFormat(media_type="application/x-percept-opaque")
    newer = SourceEnvelope(
        source_id="audio:wire",
        modality="audio",
        epoch=0,
        sequence=1,
        observed_time=100.0,
        source_time=0.2,
        format=fmt,
        payload=b"quiet",
        signal=0.0,
    )
    reordered = SourceEnvelope(
        source_id="audio:wire",
        modality="audio",
        epoch=0,
        sequence=0,
        observed_time=101.0,
        source_time=0.1,
        format=fmt,
        payload=b"loud",
        signal=0.2,
    )
    core, _, health = _run_audio_units([newer, reordered])

    assert [(unit.seq, unit.chunk) for unit in core._audio_window("camera").units()] == [
        (1, b"quiet")
    ]
    assert core._audio_window("camera").boundaries() == []
    assert "source_reorder" in [event.kind for event in health]
    ended = next(event for event in health if event.kind == "runtime_ended")
    assert ended.data["source_coverage"]["audio"] == {
        "observed": 1,
        "excluded": 1,
        "discontinuities": 1,
        "missing_sequences": 1,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        AudioUnit(0, 0.2, 0.1, b"bad-interval", energy=0.0),
        AudioUnit(
            0,
            0.0,
            0.1,
            AudioFrame(pcm=b"\x00\x00", sample_rate=16_000),
            sample_rate=8_000,
            channels=1,
            sample_format="s16le",
            energy=0.0,
        ),
    ],
    ids=["interval", "format"],
)
def test_invalid_explicit_seq_zero_does_not_poison_corrected_retry_or_coverage(
    invalid: AudioUnit,
) -> None:
    corrected = AudioUnit(0, 0.0, 0.1, b"corrected", energy=0.0)
    following = AudioUnit(1, 0.1, 0.2, b"following", energy=0.0)
    core, _, health = _run_audio_units([invalid, corrected, following])

    assert [(unit.seq, unit.chunk) for unit in core._audio_window("camera").units()] == [
        (0, b"corrected"),
        (1, b"following"),
    ]
    ended = next(event for event in health if event.kind == "runtime_ended")
    assert ended.data["source_coverage"]["audio"] == {
        "observed": 2,
        "excluded": 1,
        "discontinuities": 0,
        "missing_sequences": 0,
    }


def test_overlapping_explicit_unit_does_not_poison_corrected_retry_or_coverage() -> None:
    first = AudioUnit(0, 0.0, 0.1, b"first", energy=0.0)
    overlapping = AudioUnit(1, 0.05, 0.15, b"overlap", energy=0.0)
    corrected = AudioUnit(1, 0.1, 0.2, b"corrected", energy=0.0)
    core, _, health = _run_audio_units([first, overlapping, corrected])

    assert [(unit.seq, unit.chunk) for unit in core._audio_window("camera").units()] == [
        (0, b"first"),
        (1, b"corrected"),
    ]
    ended = next(event for event in health if event.kind == "runtime_ended")
    assert ended.data["source_coverage"]["audio"] == {
        "observed": 2,
        "excluded": 1,
        "discontinuities": 0,
        "missing_sequences": 0,
    }


def test_capture_metadata_survives_source_envelope_to_audio_unit(tmp_path: Path) -> None:
    path = tmp_path / "metadata.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x01\x00" * 80)

    async def capture_one():
        stream = open_audio_source(
            str(path),
            chunk_s=0.01,
            direction="output",
            participant_id="assistant",
        )
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    envelope = asyncio.run(capture_one())
    unit = normalize_audio_envelope(envelope, event_t=envelope.source_time)
    assert envelope.audio_direction == "output"
    assert envelope.participant_id == "assistant"
    assert unit.direction == "output"
    assert unit.participant_id == "assistant"


@pytest.mark.parametrize("case", ["duplicate", "overlap", "reorder"])
def test_duplicate_overlap_and_reorder_never_replay_through_live_onset(case: str) -> None:
    def quiet(seq, start, end):
        return AudioUnit(seq, start, end, b"q", energy=0.0)

    def loud(seq, start, end):
        return AudioUnit(seq, start, end, b"l", energy=0.2)

    if case == "duplicate":
        units = [
            quiet(0, 0.0, 0.1),
            loud(1, 0.1, 0.2),
            quiet(1, 0.1, 0.2),
            loud(2, 0.2, 0.3),
        ]
    elif case == "overlap":
        units = [
            quiet(0, 0.0, 0.1),
            loud(1, 0.1, 0.2),
            quiet(2, 0.15, 0.25),
            loud(2, 0.2, 0.3),
        ]
    else:
        units = [
            quiet(1, 0.0, 0.1),
            loud(2, 0.1, 0.2),
            quiet(0, 0.0, 0.0),
            loud(3, 0.2, 0.3),
        ]
    core, _, _ = _run_audio_units(units)
    assert [boundary.kind for boundary in core._audio_window("camera").boundaries()] == ["onset"]


@pytest.mark.parametrize("discontinuity", ["gap", "format"])
def test_source_gap_and_format_change_reset_window_and_in_progress_vad(
    discontinuity: str,
) -> None:
    first_frame = AudioFrame(
        pcm=b"\x01\x00",
        sample_rate=16_000,
        t_start=0.0,
        t_end=0.1,
    )
    first = SourceEnvelope(
        source_id="audio:capture",
        modality="audio",
        epoch=0,
        sequence=0,
        observed_time=100.0,
        source_time=0.1,
        format=SourceFormat(
            media_type="audio/pcm",
            sample_rate=16_000,
            channels=1,
            sample_format="s16le",
        ),
        payload=first_frame,
        signal=0.001,
    )
    if discontinuity == "gap":
        second_frame = AudioFrame(
            pcm=b"\x00\x00",
            sample_rate=16_000,
            t_start=0.2,
            t_end=0.3,
        )
        second_format = first.format
        second_sequence = 2
    else:
        second_frame = AudioFrame(
            pcm=b"\x00\x00\x00\x00",
            sample_rate=16_000,
            sample_format="f32le",
            t_start=0.1,
            t_end=0.2,
        )
        second_format = SourceFormat(
            media_type="audio/pcm",
            sample_rate=16_000,
            channels=1,
            sample_format="f32le",
        )
        second_sequence = 1
    second = SourceEnvelope(
        source_id="audio:capture",
        modality="audio",
        epoch=0,
        sequence=second_sequence,
        observed_time=101.0,
        source_time=second_frame.t_end,
        format=second_format,
        payload=second_frame,
        signal=0.0,
    )
    utterances = []
    core, _, health = _run_audio_units(
        [first, second],
        on_utterance=lambda clip, at: utterances.append((clip, at)),
        utterance_hangover_frames=1,
        utterance_min_speech_frames=1,
    )
    assert utterances == []
    # Source continuity is authoritative: the old boundary and media are discarded before the first
    # unit in the new continuity span enters the ordered audio window.
    assert core._audio_window("camera").boundaries() == []
    assert [unit.seq for unit in core._audio_window("camera").units()] == [second.sequence]
    expected = "source_gap" if discontinuity == "gap" else "source_format_change"
    assert expected in [event.kind for event in health]
    assert core.source_continuity_generation("camera") == 1
    ended = next(event for event in health if event.kind == "runtime_ended")
    assert ended.data["source_coverage"]["audio"] == {
        "observed": 2,
        "excluded": 0,
        "discontinuities": 1,
        "missing_sequences": 1 if discontinuity == "gap" else 0,
    }


def test_audio_reconnect_resets_wall_anchor_before_source_clock_restarts() -> None:
    core = _runtime_core()
    still = gray_frame(8, 8, 0)
    health = []
    fmt = SourceFormat(
        media_type="audio/pcm",
        sample_rate=16_000,
        channels=1,
        sample_format="s16le",
    )

    async def audio():
        for epoch, source_time, observed_time in ((0, 10.0, 100.0), (1, 0.1, 101.0)):
            yield SourceEnvelope(
                source_id="audio:reconnect",
                modality="audio",
                epoch=epoch,
                sequence=0,
                observed_time=observed_time,
                source_time=source_time,
                format=fmt,
                payload=AudioFrame(
                    pcm=b"\x01\x00" * 16,
                    sample_rate=16_000,
                    t_start=max(0.0, source_time - 0.001),
                    t_end=source_time,
                ),
                signal=0.001,
            )
            await asyncio.sleep(0.001)

    async def video():
        for seq in range(30):
            yield (seq * 0.1, still, "quiet")
            await asyncio.sleep(0.001)

    wall_times = iter((100.0, 101.0))

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=video(),
                audio_source=audio(),
                timebase="media",
                audio_window_s=10.0,
                on_utterance=lambda *_: None,
                stale_audio_s=1.0,
                wall_clock=lambda: next(wall_times, 101.0),
                on_health=health.append,
                allow_scripted_brain=True,
                judge_dispatch="inline",
            )
        ]

    asyncio.run(collect())
    kinds = [event.kind for event in health]
    assert "source_reconnect" in kinds
    assert "audio_stale" not in kinds
    assert core.source_continuity_generation("camera") == 1


def test_public_amplitude_thresholds_preserve_pre_mean_square_sensitivity() -> None:
    units = [
        AudioUnit(seq, seq * 0.1, (seq + 1) * 0.1, b"x", energy=energy)
        for seq, energy in enumerate((0.0, 0.0004, 0.0004, 0.0))
    ]
    utterances = []
    core, _, _ = _run_audio_units(
        units,
        audio_wake={"high": 0.02, "low": 0.01},
        on_utterance=lambda clip, at: utterances.append((clip, at)),
        utterance_threshold=0.01,
        utterance_hangover_frames=1,
        utterance_min_speech_frames=2,
    )
    assert len(utterances) == 1
    assert [boundary.kind for boundary in core._audio_window("camera").boundaries()] == [
        "onset",
        "vad_start",
        "endpoint",
    ]


def test_worker_retains_onset_vad_start_and_endpoint_boundaries() -> None:
    core = _runtime_core()
    still = gray_frame(8, 8, 0)
    utterances = []

    async def audio():
        for seq, energy in enumerate((0.0, 0.2, 0.2, 0.2, 0.0, 0.0)):
            yield AudioUnit(
                seq=seq,
                t_start=seq * 0.1,
                t_end=(seq + 1) * 0.1,
                chunk=AudioFrame(pcm=b"\x00\x00" * 16, sample_rate=16_000),
                energy=energy,
                sample_rate=16_000,
                channels=1,
                direction="input",
                participant_id="user",
            )
            await asyncio.sleep(0.001)

    async def video():
        for seq in range(40):
            yield (seq * 0.1, still, "quiet")
            await asyncio.sleep(0.001)

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=video(),
                audio_source=audio(),
                audio_settle_s=0.0,
                audio_window_s=10.0,
                on_utterance=lambda clip, at: utterances.append((clip, at)),
                utterance_hangover_frames=2,
                utterance_min_speech_frames=2,
                allow_scripted_brain=True,
                judge_dispatch="inline",
            )
        ]

    asyncio.run(collect())
    assert [boundary.kind for boundary in core._audio_window("camera").boundaries()] == [
        "onset",
        "vad_start",
        "endpoint",
    ]
    assert len(utterances) == 1
    assert core._audio_window("camera").units()[1].direction == "input"
    assert core._audio_window("camera").units()[1].participant_id == "user"
