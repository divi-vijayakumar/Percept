"""SDK-0 spike: what is lost when evidence is supplied through ``Monitor.poke(..., evidence=...)``.

Each gap has a characterization test (current behavior, passing) and, where the gap is real, a
test of the smallest desired behavior marked ``xfail(strict=True)``: the suite stays green, and
an xfail turns into a failure (XPASS) the moment the gap is closed, so the marker must then be
removed. Offline: a scripted ``FakeJudge``, a ``PushSource`` and the wall clock.

The stream in every test holds five 96x128 textured grey frames, ``STEP`` seconds apart, the
newest pushed at about "now"; poked evidence is a solid red image, so a test can tell which
pixels the judge, the verification or the keyframe used.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pytest
from PIL import Image

import percept
from percept.contracts import Condition
from percept.delivery import MemoryStore
from percept.judges import FakeJudge, policy
from percept.sources import PushSource
from test_monitor import Never, Y, image, run

STEP = 0.5          # seconds between stream frames
N_FRAMES = 5        # stream frames at t0 + 0.0, 0.5, 1.0, 1.5, 2.0
BETWEEN = 1.25      # an observed_at offset that falls between two stream frames
ON_FRAME = 1.0      # an observed_at offset equal to a stream frame's t_wall (frame index 2)


def red(h: int = 96, w: int = 128) -> np.ndarray:
    """A solid red BGR image: nothing like the grey stream frames."""
    out = np.zeros((h, w, 3), np.uint8)
    out[..., 2] = 255
    return out


def is_red(bgr: np.ndarray) -> bool:
    return bool(bgr[..., 2].mean() > 200 and bgr[..., 0].mean() < 60 and bgr[..., 1].mean() < 60)


def jpeg_rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def event(**kw: Any) -> Condition:
    return Condition("fell", "a package fell", kind="event", trigger=Never(), **kw)


@dataclass
class Session:
    t0: float                       # wall time of the oldest stream frame
    observed_at: Optional[float]
    poked_wall: float               # wall time just before the poke
    judge: FakeJudge
    got: list = field(default_factory=list)
    health: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    def requests(self, purpose: Optional[str] = None) -> list:
        """The provider requests sent, as (prompt, jpegs, max_output_tokens, JudgeRequest)."""
        return [s for s in self.judge.sent if purpose is None or s[3].purpose == purpose]

    @property
    def decisions(self) -> list:
        return [d.decision for d in self.got]


async def _until(cond: Callable[[], bool], tries: int = 800) -> None:
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def _session(cond: Condition, *, evidence: Any = None, offset: Optional[float] = BETWEEN,
                   source: Optional[PushSource] = None, frame: Callable[[int], np.ndarray] = image,
                   done: Optional[Callable[[Session], bool]] = None) -> Session:
    """Push the stream, poke once, wait until ``done`` (default: one decision), then close."""
    judge = FakeJudge(lambda r: {i.condition.id: Y for i in r.items})
    async with percept.Monitor(judge, [cond], store=MemoryStore(), max_requests_per_min=1000) as m:
        src = source or PushSource("cam")
        watch = m.watch(src, fps=None)
        t0 = time.time() - (N_FRAMES - 1) * STEP
        for i in range(N_FRAMES):
            src.push(frame(i), t_wall=t0 + i * STEP)
            await asyncio.sleep(0.02)
        s = Session(t0, None if offset is None else t0 + offset, time.time(), judge)

        async def listen() -> None:
            async for h in m.health():
                s.health.append(h)

        async def deliver() -> None:
            async for d in watch:
                s.got.append(d)

        listener, collector = asyncio.create_task(listen()), asyncio.create_task(deliver())
        m.poke("cam", cond.id, reason="sdk-0", observed_at=s.observed_at, evidence=evidence)
        await _until(lambda: (done or (lambda x: bool(x.got)))(s))
        s.coverage = m.coverage("cam")
        src.close()
        await asyncio.wait_for(collector, 10)
        listener.cancel()
    return s


# --------------------------------------------------------------------------- 1. evidence time


def test_gap1_multi_image_poke_collapses_to_one_instant_and_the_prompt_says_0_s():
    s = run(_session(event(verify="off"), evidence=(red(), red(), red())))
    prompt, _, _, req = s.requests("detect")[0]
    assert len(req.images) == 3
    assert req.evidence_t == (s.observed_at, s.observed_at)            # one instant for three images
    assert "There are 3 images in time order" in prompt.user
    assert "They cover 0.0 s" in prompt.user                           # told to the judge as zero duration
    assert s.decisions[0].t_observed == s.observed_at


def test_gap1_without_observed_at_the_evidence_time_is_the_dispatch_wall_clock():
    s = run(_session(event(verify="off"), evidence=(red(), red()), offset=None))
    req = s.requests("detect")[0][3]
    assert req.evidence_t[0] == req.evidence_t[1] >= s.poked_wall     # when it was sent, not when it was seen


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 1: poked images carry no per-image time; a multi-image "
                                       "poke is described to the judge as covering 0.0 s")
def test_gap1_desired_a_multi_image_poke_is_not_described_as_zero_duration():
    s = run(_session(event(verify="off"), evidence=(red(), red(), red())))
    prompt = s.requests("detect")[0][0]
    assert "They cover 0.0 s" not in prompt.user


# --------------------------------------------------------------------------- 2. evidence refs


def test_gap2_a_decision_from_poked_evidence_has_no_evidence_refs():
    s = run(_session(event(verify="off"), evidence=(red(), red())))
    d = s.decisions[0]
    assert d.type == "met" and d.request_ids
    assert d.evidence == ()                                            # nothing says what was judged


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 2: poked evidence leaves Decision.evidence empty")
def test_gap2_desired_a_decision_from_poked_evidence_references_what_was_judged():
    s = run(_session(event(verify="off"), evidence=(red(), red())))
    d = s.decisions[0]
    assert d.evidence and all(e.t_wall == s.observed_at for e in d.evidence)


# --------------------------------------------------------------------------- 3. keyframe


def distinct(i: int) -> np.ndarray:
    """Stream frames whose textures differ enough to tell which one a JPEG came from."""
    return image(40 * i)


def test_gap3_the_keyframe_of_a_poked_decision_is_a_stream_frame_not_the_poked_image():
    s = run(_session(event(verify="off"), evidence=red(), offset=ON_FRAME, frame=distinct))
    kf = jpeg_rgb(s.decisions[0].keyframe).astype(np.int16)
    assert not is_red(kf[..., ::-1].astype(np.uint8))
    diffs = [float(np.abs(kf - distinct(i).astype(np.int16)).mean()) for i in range(N_FRAMES)]
    assert int(np.argmin(diffs)) == 2 and diffs[2] < 10                # the pre-roll frame at t_observed


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 3: the keyframe comes from the stream pre-roll, "
                                       "not from the poked evidence that was judged")
def test_gap3_desired_the_keyframe_of_a_poked_decision_is_the_poked_image():
    s = run(_session(event(verify="off"), evidence=red(), offset=ON_FRAME, frame=distinct))
    assert is_red(jpeg_rgb(s.decisions[0].keyframe)[..., ::-1])


# --------------------------------------------------------------------------- 4. verification


def _verify_resolved(s: Session) -> bool:
    return bool(s.got) and any(h.kind == "verify_deferred" for h in s.health)


def test_gap4_verify_auto_on_poked_evidence_finds_no_original_and_times_out():
    s = run(_session(event(candidate_deadline_s=1.0), evidence=(red(), red())), timeout=30)
    assert [r[3].purpose for r in s.requests()] == ["detect"]           # the verification never runs
    assert [(d.type, d.cause) for d in s.decisions] == [("uncertain", "timeout")]
    assert [h.detail for h in s.health if h.kind == "verify_deferred"] == ["the original evidence left the pre-roll"]
    assert s.coverage["evidence_unavailable"] == {"fell": 1}


def test_gap4_verify_original_when_observed_at_is_a_stream_frame_time_re_judges_that_stream_frame():
    s = run(_session(event(verify="original", candidate_deadline_s=5.0), evidence=red(), offset=ON_FRAME,
                     done=lambda x: bool(x.got)), timeout=30)
    detect, verify = (r[3] for r in s.requests())
    assert detect.purpose == "detect" and is_red(detect.images[0])
    assert verify.purpose == "verify" and verify.evidence_t == detect.evidence_t
    assert np.array_equal(verify.images[0], image(2))                  # a different picture is "verified"
    assert [d.type for d in s.decisions] == ["met"]


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 4: an event verifying 'original' on poked evidence "
                                       "never re-judges the poked images: it ends uncertain/timeout")
def test_gap4_desired_verify_auto_re_judges_the_poked_images():
    s = run(_session(event(candidate_deadline_s=1.0), evidence=(red(), red())), timeout=30)
    verifies = s.requests("verify")
    assert verifies and len(verifies[0][3].images) == 2 and all(is_red(i) for i in verifies[0][3].images)
    assert [d.type for d in s.decisions] == ["met"]


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 4: when observed_at equals a pre-roll frame's time, the "
                                       "verification judges that stream frame instead of the poked image")
def test_gap4_desired_verify_original_never_substitutes_a_stream_frame_for_poked_evidence():
    s = run(_session(event(verify="original", candidate_deadline_s=5.0), evidence=red(), offset=ON_FRAME),
            timeout=30)
    verifies = s.requests("verify")
    assert verifies and is_red(verifies[0][3].images[0])


# --------------------------------------------------------------------------- 5. resolution


def test_gap5_built_in_encoding_caps_the_long_side_at_640_whatever_the_evidence():
    s = run(_session(event(verify="off"), evidence=red(960, 1280)))
    _, jpegs, _, req = s.requests("detect")[0]
    assert req.images[0].shape == (960, 1280, 3)                       # the request holds the native crop ...
    assert jpeg_rgb(jpegs[0]).shape[:2] == (480, 640)                  # ... the judge is sent 640 px


def test_gap5_a_source_opened_at_1280_is_still_sent_at_640_and_no_judge_factory_takes_max_side():
    big = lambda i: image(i, h=960, w=1280)                            # noqa: E731
    s = run(_session(Condition("door", "the door is open", trigger=Never()), source=PushSource("cam", max_side=1280),
                     frame=big, offset=None))
    _, jpegs, _, req = s.requests("detect")[0]
    assert req.images[0].shape == (960, 1280, 3)
    assert jpeg_rgb(jpegs[0]).shape[:2] == (480, 640)
    assert "max_side" in inspect.signature(policy.call).parameters     # the knob exists internally ...
    for factory in (percept.judges.claude, percept.judges.gemini, percept.judges.openai_compatible, FakeJudge):
        assert "max_side" not in inspect.signature(factory).parameters  # ... and no wrapper exposes it


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 5: the judge re-encodes at max_side=640 regardless of the "
                                       "source's max_side or the size of poked evidence")
def test_gap5_desired_a_source_opened_at_1280_reaches_the_judge_at_1280():
    big = lambda i: image(i, h=960, w=1280)                            # noqa: E731
    s = run(_session(Condition("door", "the door is open", trigger=Never()), source=PushSource("cam", max_side=1280),
                     frame=big, offset=None))
    assert jpeg_rgb(s.requests("detect")[0][1][0]).shape[:2] == (960, 1280)


# --------------------------------------------------------------------------- 6. zones


CORNER = (((0.0, 0.0), (0.25, 0.0), (0.25, 0.25), (0.0, 0.25)),)


def test_gap6_poked_evidence_is_not_cropped_to_zones_while_stream_frames_are():
    poked = run(_session(event(verify="off", zones=CORNER), evidence=red()))
    assert poked.requests("detect")[0][3].images[0].shape == (96, 128, 3)          # judged as given
    stream = run(_session(event(verify="off", zones=CORNER), offset=None))
    assert stream.requests("detect")[0][3].images[0].shape[:2] < (96, 128)         # cropped to the zone box


def test_gap6_exclude_zones_are_not_masked_on_stream_frames_either():
    """Documented behavior, not a gap (docs/conditions.md:29): exclude zones only narrow what
    change triggers watch; the judge's crop is a rectangle and may still show them."""
    s = run(_session(event(verify="off", exclude_zones=CORNER), offset=None))
    judged = s.requests("detect")[0][3].images[-1]
    assert np.array_equal(judged, image(N_FRAMES - 1))                 # excluded pixels reach the judge


# --------------------------------------------------------------------------- 7. after the anchor


# The window is "clip_s seconds before the trigger, the newest frame always included"
# (contracts.py, Condition; monitor.py, _Stream.evidence). An anchor at t0 + 1.0 with clip_s=0.9
# holds the whole window (no evidence_short); frames at t0 + 1.5 and t0 + 2.0 are newer than it.
STALE = dict(offset=ON_FRAME)


def stale_event() -> Condition:
    return event(verify="off", clip_s=0.9, clip_frames=3)


def test_gap7_a_stale_anchor_also_judges_every_frame_between_the_anchor_and_the_newest():
    s = run(_session(stale_event(), **STALE))
    assert s.coverage["evidence_short"] == {}                          # the whole window was held
    req = s.requests("detect")[0][3]
    newest = s.t0 + (N_FRAMES - 1) * STEP
    assert req.evidence_t == (s.t0 + STEP, newest)
    d = s.decisions[0]
    after = [e for e in d.evidence[:-1] if e.t_wall > s.observed_at]
    assert [e.seq for e in after] == [3]                               # not just the newest: held has no upper bound
    assert d.t_observed == newest                                      # the evidence end, as documented


def test_gap7_no_condition_field_bounds_a_window_after_the_anchor():
    # Plans refuse judgment.evidence.post_seconds ("evidence ends at the trigger"): test_plans.py.
    assert [f for f in Condition.__dataclass_fields__ if f.startswith("clip")] == ["clip_s", "clip_frames"]


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 7: frames between the anchor and the newest frame are "
                                       "judged; only the newest may be later than the anchor")
def test_gap7_desired_only_the_newest_judged_frame_is_later_than_the_anchor():
    s = run(_session(stale_event(), **STALE))
    assert all(e.t_wall <= s.observed_at for e in s.decisions[0].evidence[:-1])


@pytest.mark.xfail(strict=True, reason="SDK-0 gap 7: proposed contract change, needs a docs decision: "
                                       "t_observed is the evidence end today (decision-semantics.md, "
                                       "delivery.md), not observed_at")
def test_gap7_proposed_t_observed_follows_observed_at():
    s = run(_session(stale_event(), **STALE))
    assert abs(s.decisions[0].t_observed - s.observed_at) <= STEP
