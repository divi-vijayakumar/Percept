"""Tracker baselines: a person detector, a multi-object tracker and a zone-entry rule.

* ``detector_track_rule``: person detector → ByteTrack → "a track's box centre enters the
  condition's zone" → one event per track, then the baselines' cooldown per condition. No judge.
* ``detector_track_decide``: the same tracker entries poke a real ``percept.Monitor`` (subject
  ``track <id>``) whose judge is a deterministic rule judge: ``yes`` for a condition and subject
  the tracker poked, ``no`` otherwise, free. Percept's occurrence, verify and rearm semantics then
  decide what is delivered. Paced in real time like the Percept arms.

Only entry conditions are expressible: a condition whose id is ``enter`` (or whose dataset
``detector`` field says ``enter``). Every other condition is reported unsupported.

Components and licences (see benchmarks/README.md):

* Detector: torchvision ``ssdlite320_mobilenet_v3_large`` with its COCO_V1 weights (the
  ``person`` class, score 0.5 or more). torchvision's code is BSD-3. The weights carry no licence
  of their own: torchvision states that pre-trained models "may have their own licenses or terms
  and conditions derived from the dataset used for training" (COCO: annotations CC BY 4.0, images
  under Flickr terms) and leaves the user to determine permission for their use.
* Tracker: ``benchmarks/tracker.py``, written for this harness from the ByteTrack paper's
  association method (no code from ByteTrack, SORT or DeepSORT), with scipy (BSD-3) for the
  assignment.

torch, torchvision and scipy come from ``benchmarks/requirements-detector.txt``; they are
imported only when a tracker arm runs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from percept import Condition

from benchmarks import arms as A
from benchmarks.cases import VideoCase

SCORE = 0.5             # a person detection starts or confirms a track at this score or more
LOW_SCORE = 0.1         # ByteTrack's second stage: lower-score boxes only extend existing tracks
TRACK_BUFFER_S = 2.0    # a track unseen this long is dropped; the person returning gets a new id
BORDER = 0.08           # a track first seen inside a zone this close to its border may be entering
INWARD = 0.03           # ... and is an entry once its centre has moved this much further inside
ENTER = "enter"
ENTER_IDS = (ENTER, "person_enters_scene_through_structure")    # local datasets' label, MEVA's activity
INSTALL = "the tracker arms need torch, torchvision and scipy: pip install -r benchmarks/requirements-detector.txt"
_FULL_FRAME = (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),)


def require() -> None:
    """Raise a clear SystemExit when the tracker arms' extra dependencies are missing."""
    try:
        import scipy.optimize  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"{INSTALL} ({e})") from None


def supported(case: VideoCase) -> dict:
    """{condition id: condition spec} of the conditions a tracker rule can express: entry only (a
    door opening, or anything else, is unsupported and left out of scoring)."""
    return {c.id: c for c in case.conditions if c.id in ENTER_IDS}


# --------------------------------------------------------------------------- the rule


def _contains(poly, x: float, y: float) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        (xi, yi), (xj, yj) = poly[i], poly[i - 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
    return inside


def _border_distance(poly, x: float, y: float) -> float:
    best = float("inf")
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i - 1], poly[i]
        dx, dy = x1 - x0, y1 - y0
        u = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)))
        best = min(best, float(np.hypot(x - (x0 + u * dx), y - (y0 + u * dy))))
    return best


class EnterRule:
    """Zone entry per track, in normalised image coordinates. A track enters when its box centre
    goes from outside every zone to inside one, or when it is first seen inside a zone within
    ``border`` of the zone's edge and then moves at least ``inward`` further from that edge. A
    track first seen deeper inside (someone already standing there) never enters unless it leaves
    and comes back. Each track enters at most once."""

    def __init__(self, zones=(), border: float = BORDER, inward: float = INWARD) -> None:
        self.zones = tuple(zones) or _FULL_FRAME
        self.border, self.inward = border, inward
        self.state: dict[int, dict] = {}

    def _where(self, x: float, y: float) -> Optional[float]:
        """Distance to the edge of the zone the point is in (the deepest), None when outside all."""
        d = [_border_distance(z, x, y) for z in self.zones if _contains(z, x, y)]
        return max(d) if d else None

    def step(self, track_id: int, centre: tuple) -> bool:
        d = self._where(*centre)
        s = self.state.get(track_id)
        if s is None:
            self.state[track_id] = {"inside": d is not None, "fired": False,
                                    "edge": d if d is not None and d <= self.border else None}
            return False
        if s["fired"]:
            return False
        fire = (not s["inside"] and d is not None) or (s["edge"] is not None and d is not None
                                                         and d - s["edge"] >= self.inward)
        s["inside"] = d is not None
        if d is None:
            s["edge"] = None
        if fire:
            s["fired"] = True
        return fire


# --------------------------------------------------------------------------- the detector


_MODEL: dict = {}
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(1, thread_name_prefix="person-detector")    # one model, one inference at a time


def _device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model(device: str):
    with _LOCK:
        if device not in _MODEL:
            import torch
            from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
            weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(device)
            _MODEL[device] = (model, weights.meta["categories"].index("person"), torch)
        return _MODEL[device]


def weights_record() -> dict:
    """The detector weights a run used: torchvision's URL and the sha256 of the cached file."""
    import hashlib
    import os
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
    url = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.url
    path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if os.path.exists(path) else None
    import torchvision
    return {"url": url, "sha256": digest, "torch": torch.__version__, "torchvision": torchvision.__version__}


def detect(image: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Person boxes in a BGR frame as (x0, y0, x1, y1, score), coordinates normalised to [0, 1],
    score ``LOW_SCORE`` or more."""
    model, person, torch = _model(device)
    h, w = image.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1).float().div_(255).to(device)
    with torch.inference_mode():
        out = model([x])[0]
    keep = (out["labels"] == person) & (out["scores"] >= LOW_SCORE)
    boxes, scores = out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()
    boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
    return np.hstack([boxes, scores[:, None]]).astype(float) if len(scores) else np.zeros((0, 5))


class TrackEntries:
    """Detector, tracker and one ``EnterRule`` per supported condition for one case. ``step`` is
    one frame: the (condition id, track id) entries it produced. ``cpu_s`` is the thread CPU time
    of detection and tracking (decoding excluded)."""

    def __init__(self, zones: dict, conditions, fps: float, device: str = "cpu") -> None:
        from benchmarks.tracker import Tracker
        self.device = _device(device)
        self.tracker = Tracker(SCORE, LOW_SCORE, SCORE, track_buffer=max(1, round(TRACK_BUFFER_S * fps)))
        self.rules = {cid: EnterRule(zones.get(cid, ())) for cid in conditions}
        self.cpu_s, self.frames, self.max_tracks = 0.0, 0, 0

    def _detect(self, image: np.ndarray) -> np.ndarray:
        c0 = time.thread_time()
        boxes = detect(image, self.device)
        self.cpu_s += time.thread_time() - c0
        return boxes

    async def step(self, t: float, image: np.ndarray) -> list:
        boxes = await asyncio.get_running_loop().run_in_executor(_POOL, self._detect, image)
        c0 = time.thread_time()
        tracks = self.tracker.update(boxes, t)
        entries = []
        for tr in tracks:
            b = tr.history[-1][1]
            centre = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            entries += [(cid, tr.track_id) for cid, rule in self.rules.items() if rule.step(tr.track_id, centre)]
        self.frames += 1
        self.max_tracks = max(self.max_tracks, len(tracks))
        self.cpu_s += time.thread_time() - c0
        return entries

    def compute(self, wall_s: float, case: VideoCase) -> dict:
        minutes = case.duration / 60 or 1.0
        return {"device": self.device, "detector": "ssdlite320_mobilenet_v3_large/COCO_V1", "frames": self.frames,
                "cpu_s": round(self.cpu_s, 3), "wall_s": round(wall_s, 3),
                "cpu_s_per_clip_minute": round(self.cpu_s / minutes, 3), "max_tracks": self.max_tracks}


# --------------------------------------------------------------------------- the arms


async def run_track_rule(case: VideoCase, timelines, *, zones: dict, cooldown_s: float = A.COOLDOWN_S,
                         fps: float = 5.0, pace: Optional[float] = None, device: str = "cpu") -> A.ArmResult:
    """Every tracker entry is a decision at the frame it fired on, unless the condition's cooldown
    still runs. No judge calls, no cost."""
    timelines.set(case.case_id, 0.0, 1.0)
    result = A.ArmResult("detector_track_rule", case.case_id, cooldown_s=cooldown_s)
    conds = supported(case)
    result.unsupported = [c.id for c in case.conditions if c.id not in conds]
    entries, last = TrackEntries(zones, conds, fps, device), {}
    feed, complete, w0 = A._feed(case, fps, timelines), False, time.perf_counter()
    async for t, image in A.frames_of(feed, pace):
        for cid, _track in (await entries.step(t, image) if conds else []):
            result.fires.append((cid, t))
            if cid not in last or t - last[cid] >= cooldown_s:
                last[cid] = t
                result.preds.append(A.Pred(cid, t, t, t, 1.0, (t,)))
    else:
        complete = True
    result.compute = entries.compute(time.perf_counter() - w0, case)
    return A._finish(result, feed, complete)


class _NoTrigger:
    """A trigger that never fires: the tracker's pokes are the only looks."""

    def step(self, frame, conditions, state, now):
        return {}, state


RULE_JUDGE_REASON = "tracker entry"


def rule_judge(poked: set):
    """A deterministic, free judge: ``yes`` (0.9) for a (condition, subject) the tracker poked,
    ``no`` (0.9) for anything else (e.g. a look the runtime started on its own)."""
    from percept.judges import FakeJudge

    def script(request):
        return {i.condition.id: (("yes", 0.9, RULE_JUDGE_REASON) if (i.condition.id, i.subject) in poked
                                 else ("no", 0.9, "no tracker entry")) for i in request.items}
    return FakeJudge(script, name="track-rule", price=None)


RULE_BUDGET = 600       # attempts a minute: the rule judge is free, so the budget never drops a poke


async def run_track_decide(case: VideoCase, judge, timelines, trace_path: Path, *, zones: dict,
                           rearm_s: float = A.COOLDOWN_S, speed: float = 1.0, fps: float = 5.0,
                           device: str = "cpu", stats: Optional[dict] = None) -> A.ArmResult:
    """The tracker's entries poke a real ``Monitor`` paced in real time; its decisions are the arm's.
    ``judge`` is the run's (only its spend-cap flag is used: this arm makes no provider call)."""
    conds = supported(case)
    unsupported = [c.id for c in case.conditions if c.id not in conds]
    if not conds:       # nothing to watch: decode for the frame digest, decide nothing
        r = await run_track_rule(case, timelines, zones=zones, fps=fps, device=device)
        r.arm, r.unsupported, r.fires = "detector_track_decide", unsupported, []
        return r
    conditions = [Condition(c.id, c.text, kind="event", trigger=_NoTrigger(), zones=tuple(zones.get(c.id, ())),
                            rearm_s=rearm_s) for c in conds.values()]
    poked: set = set()
    entries, fires, w0 = TrackEntries(zones, conds, fps, device), [], time.perf_counter()
    ready = False

    async def on_frame(monitor, t, image, wall):
        nonlocal ready
        if not ready:
            await monitor.ready(case.case_id)
            ready = True
        for cid, track in await entries.step(t, image):
            subject = f"track {track}"
            poked.add((cid, subject))
            fires.append((cid, t))
            monitor.poke(case.case_id, cid, subject=subject, observed_at=wall, reason=RULE_JUDGE_REASON)

    result = await A.run_percept(case, judge, timelines, trace_path, speed=speed, fps=fps,
                                 max_requests_per_min=RULE_BUDGET, arm="detector_track_decide", stats=stats,
                                 conditions=conditions, monitor_judge=rule_judge(poked), on_frame=on_frame)
    result.unsupported, result.fires = unsupported, fires
    result.compute = entries.compute(time.perf_counter() - w0, case)
    return result
