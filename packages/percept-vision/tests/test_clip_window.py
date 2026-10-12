"""Per-condition event evidence windows: Condition.clip_s / clip_frames, pre-roll limits,
evidence_short reporting and replay of conditions recorded before the fields existed."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

import percept
from percept import monitor as rt
from percept import trace as tr
from percept.clips import PreRoll
from percept.contracts import Condition, Frame
from percept.delivery import MemoryStore
from percept.judges import FakeJudge
from percept.replay import decode_condition, encode_condition
from percept.sources import PushSource
from test_monitor import At, Never, Y, answers, collect, image, run


async def push_steady(src: PushSource, n: int, step: float = 0.5) -> None:
    """``n`` frames ``step`` seconds apart on the wall clock, ending now; the picture barely
    changes, so no scene reset interrupts a long run."""
    t0 = time.time() - (n - 1) * step
    for i in range(n):
        src.push(image(i % 3), t_wall=t0 + i * step)
        await asyncio.sleep(0.02)


def event(**kw) -> Condition:
    return Condition("fall", "a person falls", kind="event", **kw)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("kw", [
    {"kind": "state", "clip_s": 2.0},
    {"kind": "state", "clip_frames": 2},
    {"kind": "event", "clip_s": 0.0},
    {"kind": "event", "clip_s": -1.0},
    {"kind": "event", "clip_s": math.inf},
    {"kind": "event", "clip_s": math.nan},
    {"kind": "event", "clip_s": True},
    {"kind": "event", "clip_frames": 0},
    {"kind": "event", "clip_frames": 17},
    {"kind": "event", "clip_frames": 2.0},
])
def test_invalid_evidence_windows_are_refused(kw):
    with pytest.raises(ValueError):
        Condition("c", "something", **kw)


def test_the_window_defaults_to_four_seconds_and_four_frames():
    assert event().clip_window == (rt.EVENT_CLIP_S, rt.CLIP_FRAMES) == (4.0, 4)
    assert event(clip_s=2, clip_frames=16).clip_window == (2.0, 16)


def test_a_window_longer_than_the_pre_roll_is_refused_at_watch_and_at_add():
    async def main():
        m = percept.Monitor(FakeJudge({}), [event(trigger=Never(), clip_s=9.0)], store=MemoryStore())
        with pytest.raises(ValueError, match=r"clip_s=9 is longer than the pre-roll of stream 'cam' \(8 s\)"):
            m.watch(PushSource("cam"))
        assert "cam" not in m._streams
        m.watch(PushSource("long", preroll=PreRoll(seconds=12.0)))      # a source with a long enough pre-roll
        m.remove("fall")
        m.watch(PushSource("cam"))
        with pytest.raises(ValueError, match="clip_s=8.5 is longer than the pre-roll of stream 'cam'"):
            m.add(event(trigger=Never(), clip_s=8.5))
        m.add(event(trigger=Never(), clip_s=8.0))                       # exactly the pre-roll fits
        await m.aclose()

    run(main())


# --------------------------------------------------------------------------- which frames are judged


def _evidence(cond: Condition, n_frames: int, step: float = 0.5, t_trigger_back: float | None = None):
    """The frames ``_Stream.evidence`` picks for ``cond`` from a pre-roll holding ``n_frames``
    frames ``step`` seconds apart, the newest now; plus the look's recorded clip coverage."""
    async def main():
        m = percept.Monitor(FakeJudge({}), [cond], store=MemoryStore())
        src = PushSource("cam")
        m.watch(src)
        stream = m._streams["cam"]
        now = m._wall()
        frames = [Frame("cam", 0, i, float(i), now - (n_frames - 1 - i) * step, image(i)) for i in range(n_frames)]
        for f in frames:
            stream.ring.add(f)
        stream.epoch_map[0] = stream.epoch
        stream.latest, stream.latest_mono = frames[-1], m._mono()
        anchor = None if t_trigger_back is None else frames[-1].t_wall - t_trigger_back
        w = rt._Work(rt.EVENT, cond.id, "scene", m._mono(), t_trigger=anchor)
        ev = stream.evidence(w)
        await m.aclose()
        return frames, ev[0], w.clip

    return run(main())


def test_a_short_window_judges_its_frames_within_clip_s_of_the_trigger():
    frames, picked, clip = _evidence(event(clip_s=2.0, clip_frames=2), 17)
    newest = frames[-1].t_wall
    assert len(picked) == 2 and picked[-1] is frames[-1]
    assert all(newest - 2.0 <= f.t_wall <= newest for f in picked)
    assert picked[0].t_wall == newest - 2.0
    assert not rt._short(clip)


def test_a_long_window_judges_its_frames_spread_over_it():
    frames, picked, clip = _evidence(event(clip_s=8.0, clip_frames=6), 17)
    assert len(picked) == 6 and picked[0] is frames[0] and picked[-1] is frames[-1]
    assert [f.seq for f in picked] == [0, 3, 6, 10, 13, 16]
    assert clip == (8.0, 8.0, 6, 6) and not rt._short(clip)


def test_the_default_window_is_unchanged():
    frames, picked, clip = _evidence(event(), 17)
    newest = frames[-1].t_wall
    held = tuple(f for f in frames if f.t_wall >= newest - 4.0)
    assert picked == rt._spread(held, 4) and [f.seq for f in picked] == [8, 11, 13, 16]
    _, explicit, _ = _evidence(event(clip_s=4.0, clip_frames=4), 17)
    assert [f.seq for f in explicit] == [f.seq for f in picked]


def test_a_window_the_pre_roll_does_not_yet_hold_is_judged_and_marked_short():
    frames, picked, clip = _evidence(event(clip_s=4.0, clip_frames=4), 3)      # one second held
    assert [f.seq for f in picked] == [0, 1, 2]
    assert clip == (1.0, 4.0, 3, 4) and rt._short(clip)
    _, _, clip = _evidence(event(clip_s=4.0, clip_frames=2), 8)                 # 3.5 of 4 s: under 90%
    assert clip[0] == 3.5 and rt._short(clip)
    _, _, clip = _evidence(event(clip_s=4.0, clip_frames=2), 9)                 # all of it
    assert clip[0] == 4.0 and not rt._short(clip)


# --------------------------------------------------------------------------- end to end


def test_an_event_with_a_long_window_is_verified_on_the_same_frames():
    fall = event(trigger=At(17), clip_s=8.0, clip_frames=6)
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [fall], store=MemoryStore(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_steady(src, 20)
            src.close()
            return await collect(watch), m.coverage("cam")

    got, cov = run(main())
    detect = [s for s in judge.sent if s[3].purpose == "detect"]
    verify = [s for s in judge.sent if s[3].purpose == "verify"]
    assert len(detect) == 1 and len(verify) == 1
    assert len(detect[0][1]) == 6
    assert verify[0][1] == detect[0][1] and verify[0][3].evidence_t == detect[0][3].evidence_t
    assert detect[0][3].evidence_t[1] - detect[0][3].evidence_t[0] == pytest.approx(8.0)
    assert [d.decision.type for d in got] == ["met"]
    assert cov["evidence_short"] == {}


@pytest.mark.parametrize("trigger_at, short", [(1, True), (12, False)])
def test_evidence_short_is_counted_apart_from_fates(tmp_path, trigger_at, short):
    fall = event(trigger=At(trigger_at))
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [fall], store=MemoryStore(), max_requests_per_min=1000,
                                   trace=tmp_path / "t.jsonl") as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_steady(src, trigger_at + 1)
            src.close()
            await collect(watch)
            return m.coverage("cam"), m.stats()

    cov, stats = run(main())
    assert cov["evidence_short"] == ({"fall": 1} if short else {})
    assert "evidence_short" not in cov["fates"].get("fall", {})
    assert cov["fates"]["fall"].get("judged", 0) >= 1
    assert "evidence_short" not in stats.get("fates", {})
    detect = [r for r in tr.read(tmp_path / "t.jsonl") if r["kind"] == "request" and r["purpose"] == "detect"]
    assert len(detect) == 1 and detect[0]["evidence_short"] is short and detect[0]["clip_s"] == 4.0
    assert detect[0]["evidence_s"] == (0.5 if short else 4.0)


# --------------------------------------------------------------------------- replay


def test_conditions_recorded_without_a_window_decode_to_the_defaults():
    old = {"id": "fall", "text": "a person falls", "kind": "event", "judge": None, "zones": [], "exclude_zones": [],
           "not_when": [], "min_confidence": 0.6, "confirm": 1, "confirm_window_s": 20.0, "verify": "auto",
           "fast_path_confidence": None, "candidate_deadline_s": 30.0, "on_unverified": "uncertain",
           "rearm_s": 120.0, "recheck_every_s": 60.0, "clear_after_no": 3, "max_open_s": None, "cooldown_s": 0.0,
           "emit_cleared": False, "version": 1}
    cond = decode_condition(old)
    assert cond == event() and cond.clip_s is None and cond.clip_frames is None
    assert encode_condition(cond) == old                     # unchanged records for unchanged conditions
    windowed = event(clip_s=6.0, clip_frames=8)
    data = encode_condition(windowed)
    assert (data["clip_s"], data["clip_frames"]) == (6.0, 8) and decode_condition(data) == windowed
