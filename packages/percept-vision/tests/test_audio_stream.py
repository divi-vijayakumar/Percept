"""Carrier-independent audio-unit ordering and shared edge-energy parity."""

from __future__ import annotations

import json
import asyncio
import queue
import struct
from pathlib import Path

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.events import AudioBoundary, AudioUnit, AudioWindow
from percept.runtime import run
from percept.runtime.audio import _put_latest, normalize_audio_unit, pcm_energy

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


def test_legacy_push_normalizes_to_typed_unit_with_known_format() -> None:
    frame = AudioFrame(pcm=b"\x00\x20" * 160, sample_rate=16_000, channels=1)
    unit = normalize_audio_unit((3.0, frame, None), seq=7)
    assert unit.seq == 7
    assert unit.t_end == 3.0
    assert unit.t_start == pytest.approx(2.99)
    assert unit.sample_rate == 16_000
    assert unit.channels == 1
    assert unit.energy == pytest.approx(0.0625)


def test_worker_retains_onset_vad_start_and_endpoint_boundaries() -> None:
    core = Percept.create(
        vision=FakeVision(
            script={"quiet": Verdict(satisfied=False, confidence=0.95, evidence="quiet")}
        ),
        discover_plugins=False,
    )
    core.add_goal(Goal(id="sound", condition="a sound occurs", say="heard"))
    still = gray_frame(8, 8, 0)
    utterances = []

    async def audio():
        for seq, energy in enumerate((0.0, 0.2, 0.2, 0.2, 0.0, 0.0)):
            yield AudioUnit(
                seq=seq,
                t_start=float(seq),
                t_end=float(seq) + 0.1,
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
