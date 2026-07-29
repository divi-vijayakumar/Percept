"""Stage-1 ADMISSION validation — SV-1.4, SV-1.5, SV-1.7.

These are *exactness* tests: every assertion is an exact equality against a value that
the documented tier-0 contract fixes deterministically.  Nothing here needs a
pre-registered quality bar, and nothing here calls a provider or the network.

    SV-1.4  Dwell correctness .............. hysteresis, debounce, expiry split/merge
    SV-1.5  Dwell under a monotonous signal . does the interval hold, or does it flap?
    SV-1.7  Per-Watch attribution ........... N Watches, one source, zero cross-dispatch

Plus a headline characterization: an empirical state-transition table that answers
"is tier-0 admission a binary, or does it have a dwell state?".

Run it::

    /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv1_admission.py

It prints a PASS/FAIL table and writes ``results/sv1_admission.json``.  It is also
importable: :func:`run_all` returns the same structure it serializes.
"""

from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from percept import Goal, Percept
from percept.adapters import Box, Verdict
from percept.backends.fake import FakeVision
from percept.harness.tier0 import Frame, gray_frame, motion_score
from percept.interest import (
    INTEREST_ENTERED,
    INTEREST_EXITED,
    INTEREST_UPDATED,
    InterestIntervalConfig,
    InterestIntervalEngine,
    InterestSample,
)
from percept.runtime import (
    InterestSubscription,
    SourceEnvelope,
    SourceFormat,
    run,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JSON_PATH = os.path.join(RESULTS_DIR, "sv1_admission.json")

# The tier-0 gate default (``Settings.edge.motion_gate``).  Asserted at runtime, not assumed.
DEFAULT_MOTION_GATE = 0.1


# ---------------------------------------------------------------------------
# tiny result recorder
# ---------------------------------------------------------------------------


@dataclass
class Check:
    case: str
    assertion: str
    expected: Any
    actual: Any
    ok: bool


@dataclass
class TestResult:
    test_id: str
    name: str
    checks: list[Check] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self, case: str, assertion: str, expected: Any, actual: Any) -> bool:
        ok = _eq(expected, actual)
        self.checks.append(Check(case, assertion, _jsonable(expected), _jsonable(actual), ok))
        return ok

    @property
    def verdict(self) -> str:
        return "PASS" if all(c.ok for c in self.checks) and self.checks else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "verdict": self.verdict,
            "checks_total": len(self.checks),
            "checks_failed": sum(1 for c in self.checks if not c.ok),
            "checks": [
                {
                    "case": c.case,
                    "assertion": c.assertion,
                    "expected": c.expected,
                    "actual": c.actual,
                    "ok": c.ok,
                }
                for c in self.checks
            ],
            "measurements": _jsonable(self.measurements),
            "notes": list(self.notes),
        }


def _eq(a: Any, b: Any) -> bool:
    """Exact equality.  Floats compare bit-exactly on purpose — no tolerance anywhere."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(v) for v in items]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# engine helpers
# ---------------------------------------------------------------------------


def _engine(
    config: InterestIntervalConfig,
    *,
    source: str = "sv1:source",
    epoch: str = "epoch-0",
) -> InterestIntervalEngine:
    return InterestIntervalEngine(
        source=source,
        stream_epoch=epoch,
        admission_id="sv1.admission",
        detector_id="sv1.detector",
        config=config,
    )


_SEQ = [0]


def _sample(score: float, *, at: float, source_at: float | None = None) -> InterestSample:
    """One validated interest sample on a shared source/observer clock."""
    _SEQ[0] += 1
    return InterestSample(
        source_at=at if source_at is None else source_at,
        observed_at=at,
        score=score,
        reason="sv1_scripted",
        evidence_id=f"ev-{_SEQ[0]:06d}",
    )


def _drive(
    engine: InterestIntervalEngine,
    scores: Sequence[float],
    *,
    start: float = 0.0,
    dt: float = 1.0,
) -> list[Any]:
    """Feed a scripted score sequence at a fixed observed-clock cadence."""
    events: list[Any] = []
    for i, score in enumerate(scores):
        events.extend(engine.observe(_sample(score, at=start + i * dt)))
    return events


def _kinds(events: Sequence[Any]) -> list[str]:
    return [e.kind for e in events]


# ---------------------------------------------------------------------------
# an INDEPENDENT reference model of the documented contract
# ---------------------------------------------------------------------------


class ReferenceDwell:
    """A from-the-docstring reimplementation of the hysteresis/debounce state machine.

    Written against ``InterestIntervalConfig``'s documented semantics only (enter on
    ``score >= enter_threshold`` after ``enter_debounce`` consecutive samples; exit on
    ``score <= exit_threshold`` after ``exit_debounce`` consecutive samples; anything
    strictly between the two thresholds holds whatever state you are in).  Used as a
    differential oracle so SV-1.4 is not just the engine agreeing with itself.
    """

    def __init__(self, config: InterestIntervalConfig) -> None:
        self.cfg = config
        self.open = False
        self.entering = 0
        self.exiting = 0

    def step(self, score: float) -> tuple[list[str], str]:
        if not self.open:
            if score >= self.cfg.enter_threshold:
                self.entering += 1
                if self.entering >= self.cfg.enter_debounce:
                    self.open = True
                    self.entering = 0
                    self.exiting = 0
                    return [INTEREST_ENTERED], "admit"
                return [], "accumulate"
            self.entering = 0
            return [], "suppress"
        if score <= self.cfg.exit_threshold:
            self.exiting += 1
            if self.exiting >= self.cfg.exit_debounce:
                self.open = False
                self.exiting = 0
                return [INTEREST_EXITED], "exit"
            return [], "exiting"
        self.exiting = 0
        return [INTEREST_UPDATED], "dwell"


# ---------------------------------------------------------------------------
# SV-1.4 — dwell correctness (exactness, zero tolerance)
# ---------------------------------------------------------------------------

ENTER = 0.5           # binary-exact
EXIT = 0.25           # binary-exact
DEAD = 0.375          # strictly inside the hysteresis band, binary-exact
DEEP = 0.125          # well below the exit threshold, binary-exact
JUST_UNDER_ENTER = math.nextafter(ENTER, 0.0)
JUST_OVER_EXIT = math.nextafter(EXIT, 1.0)


def sv1_4_dwell_correctness() -> TestResult:
    r = TestResult("SV-1.4", "Dwell correctness — hysteresis, debounce, expiry (exact)")

    # -- A. hysteresis is enforced at construction -------------------------------
    for bad_exit, label in ((ENTER, "exit == enter"), (0.75, "exit > enter")):
        raised = None
        try:
            InterestIntervalConfig(enter_threshold=ENTER, exit_threshold=bad_exit)
        except Exception as exc:  # noqa: BLE001 - we are asserting the exact type
            raised = type(exc).__name__
        r.check("A.config-guard", f"{label} raises ValueError", "ValueError", raised)

    cfg11 = InterestIntervalConfig(
        enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=1, exit_debounce=1
    )

    # -- B. enter boundary is inclusive, exactly ---------------------------------
    r.check(
        "B.enter-boundary",
        "score == enter_threshold admits",
        [INTEREST_ENTERED],
        _kinds(_drive(_engine(cfg11), [ENTER])),
    )
    r.check(
        "B.enter-boundary",
        "score == nextafter(enter, 0) does NOT admit",
        [],
        _kinds(_drive(_engine(cfg11), [JUST_UNDER_ENTER])),
    )
    r.check(
        "B.enter-boundary",
        "score inside the hysteresis band does NOT admit from closed",
        [],
        _kinds(_drive(_engine(cfg11), [DEAD, DEAD, DEAD, DEAD])),
    )

    # -- C. exit boundary is inclusive, exactly ----------------------------------
    r.check(
        "C.exit-boundary",
        "open + score == exit_threshold exits",
        [INTEREST_ENTERED, INTEREST_EXITED],
        _kinds(_drive(_engine(cfg11), [ENTER, EXIT])),
    )
    r.check(
        "C.exit-boundary",
        "open + score == nextafter(exit, 1) stays open (updates)",
        [INTEREST_ENTERED, INTEREST_UPDATED],
        _kinds(_drive(_engine(cfg11), [ENTER, JUST_OVER_EXIT])),
    )

    # -- D. ZERO flap in the hysteresis band -------------------------------------
    # 40 samples oscillating across the ENTER threshold while the interval is open.
    # A single-threshold comparator at ``enter`` would toggle on every sample.
    flap_scores = [ENTER] + [DEAD, ENTER] * 20
    flap_events = _drive(_engine(cfg11), flap_scores)
    naive_transitions = sum(
        1
        for a, b in zip(flap_scores, flap_scores[1:])
        if (a >= ENTER) != (b >= ENTER)
    )
    r.check("D.zero-flap", "exits emitted while oscillating in the band", 0,
            sum(1 for e in flap_events if e.kind == INTEREST_EXITED))
    r.check("D.zero-flap", "re-enters emitted while oscillating in the band", 1,
            sum(1 for e in flap_events if e.kind == INTEREST_ENTERED))
    r.check("D.zero-flap", "one stable interval id", 1,
            len({e.interval_id for e in flap_events}))
    r.check("D.zero-flap", "updates emitted", len(flap_scores) - 1,
            sum(1 for e in flap_events if e.kind == INTEREST_UPDATED))
    r.measurements["zero_flap"] = {
        "samples": len(flap_scores),
        "single_threshold_comparator_transitions": naive_transitions,
        "hysteresis_transitions": 1,
    }

    # closed-side flap: oscillating inside the band from closed must never admit
    closed_flap = _drive(_engine(cfg11), [DEEP] + [DEAD, JUST_UNDER_ENTER] * 20)
    r.check("D.zero-flap", "no admission from closed while oscillating below enter", [],
            _kinds(closed_flap))

    # -- E. enter_debounce is exact ----------------------------------------------
    for n in (2, 3, 5):
        cfg = InterestIntervalConfig(
            enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=n, exit_debounce=1
        )
        r.check("E.enter-debounce", f"n-1={n - 1} consecutive above-enter samples emit nothing", [],
                _kinds(_drive(_engine(cfg), [ENTER] * (n - 1))))
        r.check("E.enter-debounce", f"n={n} consecutive above-enter samples emit exactly one enter",
                [INTEREST_ENTERED], _kinds(_drive(_engine(cfg), [ENTER] * n)))
        # a band sample resets the accumulator: n-1 + band + n-1 must still not admit
        r.check(
            "E.enter-debounce",
            f"n={n}: a hysteresis-band sample RESETS the enter accumulator",
            [],
            _kinds(_drive(_engine(cfg), [ENTER] * (n - 1) + [DEAD] + [ENTER] * (n - 1))),
        )
        r.check(
            "E.enter-debounce",
            f"n={n}: a below-exit sample RESETS the enter accumulator",
            [],
            _kinds(_drive(_engine(cfg), [ENTER] * (n - 1) + [DEEP] + [ENTER] * (n - 1))),
        )

    # -- F. exit_debounce is exact -----------------------------------------------
    for n in (2, 3, 5):
        cfg = InterestIntervalConfig(
            enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=1, exit_debounce=n
        )
        eng = _engine(cfg)
        ev = _drive(eng, [ENTER] + [EXIT] * (n - 1))
        r.check("F.exit-debounce", f"n-1={n - 1} consecutive below-exit samples emit no exit",
                [INTEREST_ENTERED], _kinds(ev))
        r.check("F.exit-debounce", f"n-1={n - 1}: interval is still open", True, eng.active)
        r.check("F.exit-debounce", f"n={n} consecutive below-exit samples emit exactly one exit",
                [INTEREST_ENTERED, INTEREST_EXITED],
                _kinds(_drive(_engine(cfg), [ENTER] + [EXIT] * n)))
        r.check(
            "F.exit-debounce",
            f"n={n}: a hysteresis-band sample RESETS the exit accumulator",
            [INTEREST_ENTERED] + [INTEREST_UPDATED],
            _kinds(_drive(_engine(cfg), [ENTER] + [EXIT] * (n - 1) + [DEAD] + [EXIT] * (n - 1))),
        )

    # -- G. expiry splits vs merges, exactly at the boundary ----------------------
    EXPIRY = 2.0
    cfg_exp = InterestIntervalConfig(
        enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=1,
        exit_debounce=1, expiry_s=EXPIRY,
    )
    last_at = 4.0
    deadline = last_at + EXPIRY  # 6.0, binary-exact

    def _expiry_case(next_at: float) -> tuple[list[str], int]:
        eng = _engine(cfg_exp)
        first = eng.observe(_sample(ENTER, at=last_at))
        rest = eng.observe(_sample(ENTER, at=next_at))
        ids = {e.interval_id for e in first + rest}
        return _kinds(rest), len(ids)

    kinds, ids = _expiry_case(math.nextafter(deadline, 0.0))
    r.check("G.expiry", "gap == nextafter(expiry, 0) MERGES (update, same interval)",
            [INTEREST_UPDATED], kinds)
    r.check("G.expiry", "merge keeps exactly one interval id", 1, ids)

    kinds, ids = _expiry_case(deadline)
    r.check("G.expiry", "gap == expiry_s exactly SPLITS (expired exit + new enter)",
            [INTEREST_EXITED, INTEREST_ENTERED], kinds)
    r.check("G.expiry", "split mints a second interval id", 2, ids)

    kinds, ids = _expiry_case(math.nextafter(deadline, 10.0))
    r.check("G.expiry", "gap == nextafter(expiry, +inf) SPLITS", [INTEREST_EXITED, INTEREST_ENTERED],
            kinds)

    eng = _engine(cfg_exp)
    eng.observe(_sample(ENTER, at=last_at))
    split = eng.observe(_sample(ENTER, at=deadline))
    r.check("G.expiry", "the split exit reason is 'expired'", "expired", split[0].reason)

    # advance() (observer clock only) fires expiry at, and not before, the deadline
    eng = _engine(cfg_exp)
    eng.observe(_sample(ENTER, at=last_at))
    r.check("G.expiry", "advance() just under the deadline emits nothing", [],
            _kinds(eng.advance(observed_at=math.nextafter(deadline, 0.0))))
    r.check("G.expiry", "advance() at the deadline emits exactly one expired exit",
            [INTEREST_EXITED], _kinds(eng.advance(observed_at=deadline)))

    # a partial enter debounce cannot bridge a gap that would have expired an interval
    cfg_exp3 = InterestIntervalConfig(
        enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=2,
        exit_debounce=1, expiry_s=EXPIRY,
    )
    eng = _engine(cfg_exp3)
    eng.observe(_sample(ENTER, at=last_at))  # partial debounce (1 of 2)
    bridged = eng.observe(_sample(ENTER, at=deadline))  # gap == expiry
    r.check("G.expiry", "an expiry gap discards a partial enter debounce", [], _kinds(bridged))

    # -- H. differential vs the independent reference model ----------------------
    # A seeded deterministic score walk over four configurations; the engine's whole
    # event-kind stream must equal the reference model's, sample for sample.
    walk = _score_walk(2000)
    diff_configs = [
        (1, 1), (1, 3), (3, 1), (4, 4),
    ]
    total_events = 0
    for enter_db, exit_db in diff_configs:
        cfg = InterestIntervalConfig(
            enter_threshold=ENTER, exit_threshold=EXIT,
            enter_debounce=enter_db, exit_debounce=exit_db,
        )
        eng = _engine(cfg)
        ref = ReferenceDwell(cfg)
        engine_stream: list[str] = []
        ref_stream: list[str] = []
        for i, score in enumerate(walk):
            engine_stream.extend(_kinds(eng.observe(_sample(score, at=float(i)))))
            ref_stream.extend(ref.step(score)[0])
        total_events += len(engine_stream)
        r.check(
            "H.differential",
            f"enter_debounce={enter_db} exit_debounce={exit_db}: "
            f"{len(walk)}-sample stream matches the reference model exactly",
            ref_stream,
            engine_stream,
        )
        r.check(
            "H.differential",
            f"enter_debounce={enter_db} exit_debounce={exit_db}: open-state agrees at the end",
            ref.open,
            eng.active,
        )
    r.measurements["differential"] = {
        "samples_per_config": len(walk),
        "configs": [{"enter_debounce": a, "exit_debounce": b} for a, b in diff_configs],
        "events_compared": total_events,
    }
    return r


def _score_walk(n: int) -> list[float]:
    """A deterministic, seed-fixed score walk that straddles both thresholds.

    Pure integer LCG so the sequence is byte-identical on every platform and Python build.
    """
    state = 0x5EED
    out: list[float] = []
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        # 65 discrete levels in [0, 1] — 1/64 steps are binary-exact, so no float drift.
        out.append((state % 65) / 64.0)
    return out


# ---------------------------------------------------------------------------
# SV-1.5 — dwell under a MONOTONOUS signal
# ---------------------------------------------------------------------------

MONOTONOUS_N = 300


def sv1_5_monotonous_dwell() -> TestResult:
    r = TestResult("SV-1.5", "Dwell under a monotonous signal — does the interval hold?")

    cfg = InterestIntervalConfig(
        enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=1, exit_debounce=1
    )

    # -- M1: constant score exactly AT the enter threshold ------------------------
    ev = _drive(_engine(cfg), [ENTER] * MONOTONOUS_N)
    counts = collections.Counter(_kinds(ev))
    r.check("M1.constant-at-enter", "exactly one enter", 1, counts[INTEREST_ENTERED])
    r.check("M1.constant-at-enter", "zero exits", 0, counts[INTEREST_EXITED])
    r.check("M1.constant-at-enter", "one update per subsequent sample", MONOTONOUS_N - 1,
            counts[INTEREST_UPDATED])
    r.check("M1.constant-at-enter", "one interval id for the whole positive", 1,
            len({e.interval_id for e in ev}))

    # -- M2: constant score inside the hysteresis band after onset ----------------
    # The operationally important case: the signal decays to a plateau that is below
    # the enter threshold but above the exit threshold.
    ev = _drive(_engine(cfg), [ENTER] + [DEAD] * (MONOTONOUS_N - 1))
    counts = collections.Counter(_kinds(ev))
    r.check("M2.band-plateau", "exactly one enter", 1, counts[INTEREST_ENTERED])
    r.check("M2.band-plateau", "zero exits", 0, counts[INTEREST_EXITED])
    r.check("M2.band-plateau", "interval holds for the whole plateau", MONOTONOUS_N - 1,
            counts[INTEREST_UPDATED])

    # -- M3: constant score with expiry armed and cadence inside the deadline -----
    cfg_exp = InterestIntervalConfig(
        enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=1,
        exit_debounce=1, expiry_s=2.0,
    )
    ev = _drive(_engine(cfg_exp), [ENTER] * MONOTONOUS_N, dt=1.0)
    counts = collections.Counter(_kinds(ev))
    r.check("M3.expiry-armed", "expiry does not split a continuously sampled positive", 1,
            counts[INTEREST_ENTERED])
    r.check("M3.expiry-armed", "zero exits", 0, counts[INTEREST_EXITED])

    # -- M4..M7: the same question on the REAL live percept.run() path ------------
    live = asyncio.run(_live_monotonous_cases())
    r.measurements["live"] = live

    m4 = live["M4_constant_motion_ordinary_watch"]
    r.check("M4.live-constant-motion", "exactly one enter over the whole positive", 1,
            m4["entered"])
    r.check("M4.live-constant-motion", "zero mid-positive exits", 0, m4["mid_exits"])
    r.check("M4.live-constant-motion", "one update per subsequent admitted unit",
            m4["frames"] - 2, m4["updated"])
    r.check("M4.live-constant-motion", "one stable interval id", 1, m4["interval_ids"])

    m5 = live["M5_static_after_onset_ordinary_watch"]
    r.check(
        "M5.live-static-scene",
        f"source units the interval stays open across (of {m5['frames'] - 1} post-onset units)",
        m5["frames"] - 1,
        m5["updated"] + 1,
    )
    r.check("M5.live-static-scene", "zero mid-positive exits", 0, m5["mid_exits"])

    m6 = live["M6_static_after_onset_coasting_watch"]
    r.check("M6.live-coast-static", "exactly one enter over the whole positive", 1,
            m6["entered"])
    r.check("M6.live-coast-static", "zero mid-positive exits", 0, m6["mid_exits"])

    m7 = live["M7_oscillating_coasting_watch"]
    r.check("M7.live-coast-oscillating", "exactly one enter over the whole positive", 1,
            m7["entered"])
    r.check("M7.live-coast-oscillating", "zero mid-positive exits", 0, m7["mid_exits"])
    r.check("M7.live-coast-oscillating", "one stable interval id", 1, m7["interval_ids"])

    r.notes.append(
        "M1-M3 exercise the InterestIntervalEngine directly; M4-M7 exercise percept.run()."
    )
    return r


# ---------------------------------------------------------------------------
# live-path scaffolding (fully offline: scripted vision, in-memory envelopes)
# ---------------------------------------------------------------------------

OPAQUE = SourceFormat(media_type="application/x-percept-opaque")
FRAME_W, FRAME_H = 8, 8


def _uniform(level: int) -> Frame:
    return gray_frame(FRAME_W, FRAME_H, level)


def _split_frame(left: int, right: int) -> Frame:
    pixels: list[int] = []
    for _ in range(FRAME_H):
        for x in range(FRAME_W):
            pixels.append(left if x < FRAME_W // 2 else right)
    return Frame(w=FRAME_W, h=FRAME_H, pixels=tuple(pixels))


def _envelope(seq: int, frame: Frame, *, payload: str = "clear", dt: float = 0.25) -> SourceEnvelope:
    return SourceEnvelope(
        source_id="video:sv1",
        modality="video",
        epoch=0,
        sequence=seq,
        source_time=10.0 + seq * dt,
        observed_time=100.0 + seq * dt,
        format=OPAQUE,
        payload=payload,
        signal=frame,
    )


def _agent(goals: Sequence[Goal], *, script: dict[str, Verdict] | None = None) -> Any:
    vision = FakeVision(
        script=script
        or {"clear": Verdict(satisfied=False, confidence=0.99, evidence="scripted: not satisfied")}
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    for goal in goals:
        core.add_goal(goal)
    return core


async def _run_live(
    frames: Sequence[Frame],
    goals: Sequence[Goal],
    *,
    payloads: Sequence[str] | None = None,
    script: dict[str, Verdict] | None = None,
    session: str = "sv1",
) -> tuple[list[Any], list[Any]]:
    """Drive the real ``percept.run`` worker over an in-memory envelope list."""
    core = _agent(goals, script=script)
    subscription = InterestSubscription()
    events: list[Any] = []
    incidents: list[Any] = []

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    source = [
        _envelope(i, frame, payload=(payloads[i] if payloads else "clear"))
        for i, frame in enumerate(frames)
    ]
    async for incident in run(
        core,
        source=source,
        interest_subscription=subscription,
        interest_session_id=session,
        judge_dispatch="inline",
    ):
        incidents.append(incident)
    await consumer
    return events, incidents


def _summarize(events: Sequence[Any], watch_id: str, frames: int) -> dict[str, Any]:
    mine = [e for e in events if e.watch_id == watch_id]
    counts = collections.Counter(e.kind for e in mine)
    exits = [e for e in mine if e.kind == INTEREST_EXITED]
    return {
        "frames": frames,
        "entered": counts[INTEREST_ENTERED],
        "updated": counts[INTEREST_UPDATED],
        "exited": counts[INTEREST_EXITED],
        # A terminal end_of_stream exit is the contract closing the book, not a flap.
        "mid_exits": sum(1 for e in exits if e.reason != "end_of_stream"),
        "exit_reasons": dict(collections.Counter(e.reason for e in exits)),
        "admission_causes": dict(collections.Counter(e.admission_cause for e in mine)),
        "interval_ids": len({e.interval_id for e in mine}),
        "re_enters": max(0, counts[INTEREST_ENTERED] - 1),
    }


async def _live_monotonous_cases() -> dict[str, Any]:
    out: dict[str, Any] = {}
    ordinary = [Goal(id="w-mono", condition="the thing is happening", say="fired")]
    # A brief-window prior is what arms the worker's fovea coast for a Watch.
    coasting = [
        Goal(
            id="w-mono",
            condition="the thing is happening",
            say="fired",
            state={"sense": {"brief_window_s": 0.5}},
        )
    ]

    # Constant, above-gate motion for 32 units (a fan, traffic, pacing).
    step = [_uniform(v) for v in ([0, 60] * 16)]
    events, _ = await _run_live(step, ordinary, session="m4")
    out["M4_constant_motion_ordinary_watch"] = _summarize(events, "w-mono", len(step))
    out["M4_constant_motion_ordinary_watch"]["per_unit_motion"] = motion_score(
        _uniform(0), _uniform(60)
    )

    # Onset then a genuinely static positive (someone standing still, a paused screen).
    static = [_uniform(0), _uniform(255)] + [_uniform(255)] * 30
    events, _ = await _run_live(static, ordinary, session="m5")
    out["M5_static_after_onset_ordinary_watch"] = _summarize(events, "w-mono", len(static))

    # The same static positive, but on a Watch whose coast route is armed.
    events, _ = await _run_live(static, coasting, session="m6")
    out["M6_static_after_onset_coasting_watch"] = _summarize(events, "w-mono", len(static))

    # A coasting Watch on a monotonous positive whose motion oscillates across the
    #      gate (two still units, then a small move — the shape of almost any real scene).
    oscillating = [_uniform(v) for v in ([0, 0, 40] * 20)]
    events, _ = await _run_live(oscillating, coasting, session="m7")
    out["M7_oscillating_coasting_watch"] = _summarize(events, "w-mono", len(oscillating))
    return out


# ---------------------------------------------------------------------------
# SV-1.7 — per-Watch attribution
# ---------------------------------------------------------------------------

LEFT_BOX = Box(x=0.0, y=0.0, w=0.5, h=1.0)
RIGHT_BOX = Box(x=0.5, y=0.0, w=0.5, h=1.0)
HI = 200  # keeps the full-frame mean strictly below 1.0 even when BOTH halves move


def sv1_7_per_watch_attribution() -> TestResult:
    r = TestResult("SV-1.7", "Per-Watch attribution — N Watches, one source, zero cross-dispatch")

    from percept.config import load_settings

    gate_default = load_settings().edge.motion_gate
    r.check("0.fixture", "tier-0 gate default is the documented 0.1", DEFAULT_MOTION_GATE,
            gate_default)

    levels = [
        (0, 0),      # 0: cold start, no predecessor
        (HI, 0),     # 1: LEFT moves
        (HI, 0),     # 2: still
        (HI, HI),    # 3: RIGHT moves
        (0, HI),     # 4: LEFT moves
        (0, HI),     # 5: still
        (HI, 0),     # 6: BOTH move
        (HI, 0),     # 7: still
    ]
    frames = [_split_frame(a, b) for a, b in levels]

    watches = {
        "w-left": {"region": LEFT_BOX, "bias": 0.0},
        "w-right": {"region": RIGHT_BOX, "bias": 0.0},
        "w-any": {"region": None, "bias": 0.0},
        # The distractor: threshold pinned to 1.0, which this fixture can never reach.
        "w-distractor": {"region": None, "bias": 0.95},
    }
    goals = [
        Goal(
            id=wid,
            condition=f"{wid} condition",
            say=f"{wid} fired",
            state=(
                {k: v for k, v in (
                    ("region_hint", spec["region"]),
                    ("motion_bias", spec["bias"] or None),
                ) if v is not None}
            ),
        )
        for wid, spec in watches.items()
    ]

    # Independently recompute what each Watch MUST admit, from the frames alone.
    expected_scores: dict[str, list[float]] = {}
    expected_admits: dict[str, list[bool]] = {}
    for wid, spec in watches.items():
        threshold = min(1.0, gate_default + spec["bias"])
        scores = [0.0]
        for i in range(1, len(frames)):
            scores.append(motion_score(frames[i - 1], frames[i], region=spec["region"]))
        expected_scores[wid] = scores
        expected_admits[wid] = [s >= threshold for s in scores]

    max_full = max(expected_scores["w-any"])
    r.check("0.fixture", "the fixture never reaches the distractor's 1.0 threshold", True,
            max_full < 1.0)

    expected_lifecycle = _expected_lifecycle(expected_admits, expected_scores)

    events, incidents = asyncio.run(
        _run_live(
            frames,
            goals,
            payloads=["hit"] * len(frames),
            script={"hit": Verdict(satisfied=True, confidence=0.95, evidence="scripted: satisfied")},
            session="sv17",
        )
    )

    # The exact stream: who, what, on which source unit, why, and carrying whose score.
    actual_lifecycle = [
        (e.watch_id, e.kind, e.event_sequence, e.reason, e.score) for e in events
    ]
    r.check("A.attribution", "the whole lifecycle stream matches the independently derived one",
            expected_lifecycle, actual_lifecycle)

    admitting = {wid for wid, flags in expected_admits.items() if any(flags)}
    r.check("A.attribution", "only Watches that admitted appear in the lifecycle",
            sorted(admitting), sorted({e.watch_id for e in events}))

    thresh_mismatches = []
    for e in events:
        want = min(1.0, gate_default + watches[e.watch_id]["bias"])
        if e.threshold != want:
            thresh_mismatches.append((e.watch_id, want, e.threshold))
    r.check("A.attribution", "each event carries its OWN Watch's threshold", [],
            sorted(set(thresh_mismatches)))

    # -- B. no interval identity is ever shared between Watches -------------------
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for e in events:
        owners[e.interval_id].add(e.watch_id)
    shared = sorted(iid for iid, ws in owners.items() if len(ws) != 1)
    r.check("B.no-cross-dispatch", "no interval id is shared across Watches", [], shared)

    # -- C. the distractor never dispatches --------------------------------------
    r.check("C.distractor", "distractor emits zero lifecycle events", 0,
            sum(1 for e in events if e.watch_id == "w-distractor"))
    r.check("C.distractor", "distractor never admits in the derived expectation", False,
            any(expected_admits["w-distractor"]))
    concern_ids = sorted({getattr(i, "concern_id", None) for i in incidents})
    r.check("C.distractor", "distractor produces zero incidents", False,
            "w-distractor" in concern_ids)
    r.check("C.distractor", "every incident belongs to a Watch that actually admitted", [],
            sorted(c for c in concern_ids if c not in admitting))

    # -- D. the public callback contract -----------------------------------------
    raised = asyncio.run(_on_interest_callback_rejected(goals[:1]))
    r.check("D.api-contract", "run(on_interest=...) raises InterestCallbackUnsupportedError",
            "InterestCallbackUnsupportedError", raised)

    r.measurements["per_unit_scores"] = {
        wid: [round(s, 6) for s in scores] for wid, scores in expected_scores.items()
    }
    r.measurements["thresholds"] = {
        wid: min(1.0, gate_default + spec["bias"]) for wid, spec in watches.items()
    }
    r.measurements["incidents"] = sorted(
        (getattr(i, "concern_id", None) or "?") for i in incidents
    )
    r.measurements["lifecycle"] = [list(row) for row in actual_lifecycle]
    return r


def _expected_lifecycle(
    admits: dict[str, list[bool]],
    scores: dict[str, list[float]],
) -> list[tuple[str, str, int, str, float]]:
    """Derive the exact live lifecycle stream from a per-Watch admission matrix.

    The tracker walks Watches in sorted id order within each source unit, opens on the
    first admitted unit, updates on every subsequent admitted unit, closes with
    ``signal_lost`` on the first non-admitted unit, and closes anything still open with
    ``end_of_stream``.  An exit reports the LAST ADMITTED unit's score — the closing unit
    is the transition boundary, not evidence (``percept/interest.py:1281-1300``).
    """
    n = len(next(iter(admits.values())))
    open_now: dict[str, bool] = {wid: False for wid in admits}
    last_admitted: dict[str, int] = {wid: 0 for wid in admits}
    out: list[tuple[str, str, int, str, float]] = []
    for i in range(n):
        for wid in sorted(admits):
            if not admits[wid][i]:
                if open_now[wid]:
                    open_now[wid] = False
                    out.append(
                        (wid, INTEREST_EXITED, i, "signal_lost", scores[wid][last_admitted[wid]])
                    )
                continue
            kind = INTEREST_UPDATED if open_now[wid] else INTEREST_ENTERED
            open_now[wid] = True
            last_admitted[wid] = i
            out.append((wid, kind, i, "tier0_motion", scores[wid][i]))
    for wid in sorted(admits):
        if open_now[wid]:
            out.append(
                (wid, INTEREST_EXITED, n - 1, "end_of_stream", scores[wid][last_admitted[wid]])
            )
    return out


async def _on_interest_callback_rejected(goals: Sequence[Goal]) -> str:
    core = _agent(goals)
    try:
        async for _ in run(
            core,
            source=[_envelope(0, _uniform(0))],
            on_interest=lambda event: None,
        ):
            pass
    except Exception as exc:  # noqa: BLE001 - the exact type is the assertion
        return type(exc).__name__
    return "<no exception>"


# ---------------------------------------------------------------------------
# characterization — the empirical state-transition table
# ---------------------------------------------------------------------------

TABLE_CONFIG = InterestIntervalConfig(
    enter_threshold=ENTER, exit_threshold=EXIT, enter_debounce=2, exit_debounce=2
)

TABLE_SCRIPT: list[tuple[float, str]] = [
    (DEEP, "below exit_threshold, closed"),
    (DEAD, "inside the hysteresis band, closed"),
    (ENTER, "at enter_threshold (1 of 2)"),
    (DEAD, "band sample resets the accumulator"),
    (ENTER, "at enter_threshold (1 of 2, again)"),
    (ENTER, "at enter_threshold (2 of 2)"),
    (ENTER, "still above enter_threshold"),
    (DEAD, "inside the hysteresis band, OPEN"),
    (EXIT, "at exit_threshold (1 of 2)"),
    (DEAD, "band sample resets the exit accumulator"),
    (EXIT, "at exit_threshold (1 of 2, again)"),
    (EXIT, "at exit_threshold (2 of 2)"),
    (DEEP, "below exit_threshold, closed again"),
]


def state_transition_table() -> tuple[list[dict[str, Any]], TestResult]:
    """Empirically derive the suppress / accumulate / admit+dwell / exiting mapping.

    This is the direct answer to "is tier-0 admission binary?".  Each row is one sample;
    the emitted events and the engine's public ``active``/``interval_id`` are observed,
    and the derived state label is cross-checked against the independent reference model.
    """
    r = TestResult("CHAR", "State-transition table (suppress / accumulate / admit+dwell)")
    engine = _engine(TABLE_CONFIG)
    ref = ReferenceDwell(TABLE_CONFIG)
    rows: list[dict[str, Any]] = []
    for i, (score, meaning) in enumerate(TABLE_SCRIPT):
        events = engine.observe(_sample(score, at=float(i)))
        ref_events, ref_state = ref.step(score)
        rows.append(
            {
                "i": i,
                "score": score,
                "input": meaning,
                "events": _kinds(events),
                "state": ref_state,
                "open_after": engine.active,
                "interval_open": engine.interval_id is not None,
            }
        )
        r.check("table", f"row {i} ({meaning}): emitted events", ref_events, _kinds(events))
        r.check("table", f"row {i} ({meaning}): open state", ref.open, engine.active)

    observed_states = {row["state"] for row in rows}
    r.check(
        "table",
        "all four documented states are observable",
        sorted(["suppress", "accumulate", "admit", "dwell", "exiting", "exit"]),
        sorted(observed_states),
    )
    # 4 suppress + 2 accumulate + 2 exiting samples emit nothing; the state machine is
    # therefore NOT a binary in/out — 8 of 13 samples change internal state silently.
    r.check("table", "silent (event-free) samples in a 13-sample script", 8,
            sum(1 for row in rows if not row["events"]))
    r.check("table", "event-emitting samples in the same script", 5,
            sum(1 for row in rows if row["events"]))
    return rows, r


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all() -> dict[str, Any]:
    table, table_result = state_transition_table()
    results = [
        sv1_4_dwell_correctness(),
        sv1_5_monotonous_dwell(),
        sv1_7_per_watch_attribution(),
        table_result,
    ]
    import percept

    return {
        "suite": "sv1_admission",
        "stage": "Stage-1 ADMISSION",
        "runtime": {
            "percept_version": getattr(percept, "__version__", "unknown"),
            "percept_path": os.path.dirname(os.path.abspath(percept.__file__)),
            "python": sys.version.split()[0],
            "provider_calls": 0,
            "network": False,
        },
        "tests": [r.as_dict() for r in results],
        "state_transition_table": table,
        "verdicts": {r.test_id: r.verdict for r in results},
    }


def _print_report(report: dict[str, Any]) -> int:
    print()
    print("=" * 78)
    print("  Stage-1 ADMISSION validation — SV-1.4 / SV-1.5 / SV-1.7")
    print(f"  percept {report['runtime']['percept_version']}  ·  "
          f"python {report['runtime']['python']}  ·  offline, deterministic")
    print("=" * 78)

    print()
    print("  STATE-TRANSITION TABLE  (enter=0.500 exit=0.250, enter_debounce=2, exit_debounce=2)")
    print(f"  {'#':>2}  {'score':>6}  {'input':<42} {'state':<11} events")
    print("  " + "-" * 92)
    for row in report["state_transition_table"]:
        events = ", ".join(e.split(".")[-1] for e in row["events"]) or "—"
        print(f"  {row['i']:>2}  {row['score']:>6.3f}  {row['input']:<42} "
              f"{row['state']:<11} {events}")

    print()
    print(f"  {'TEST':<7} {'VERDICT':<8} {'CHECKS':<12} NAME")
    print("  " + "-" * 92)
    failed_total = 0
    for test in report["tests"]:
        failed_total += test["checks_failed"]
        checks = f"{test['checks_total'] - test['checks_failed']}/{test['checks_total']}"
        print(f"  {test['test_id']:<7} {test['verdict']:<8} {checks:<12} {test['name']}")

    for test in report["tests"]:
        failures = [c for c in test["checks"] if not c["ok"]]
        if not failures:
            continue
        print()
        print(f"  FAILURES — {test['test_id']}")
        for c in failures:
            print(f"    [{c['case']}] {c['assertion']}")
            print(f"        expected: {_short(c['expected'])}")
            print(f"        actual:   {_short(c['actual'])}")

    live = next(
        (t["measurements"].get("live") for t in report["tests"] if t["test_id"] == "SV-1.5"),
        None,
    )
    if live:
        print()
        print("  MONOTONOUS DWELL — measured on the live percept.run() path")
        print(f"  {'case':<40} {'units':>6} {'ent':>5} {'upd':>5} {'exit':>5} "
              f"{'re-ent':>7}  exit reasons")
        print("  " + "-" * 108)
        for name, m in live.items():
            print(f"  {name:<40} {m['frames']:>6} {m['entered']:>5} {m['updated']:>5} "
                  f"{m['exited']:>5} {m['re_enters']:>7}  {m['exit_reasons']}")

    print()
    overall = "PASS" if failed_total == 0 else "FAIL"
    print(f"  OVERALL: {overall}  ({failed_total} failed check(s))")
    print("=" * 78)
    print()
    return 0 if failed_total == 0 else 1


def _short(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, default=repr)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    report = run_all()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False, default=repr)
        handle.write("\n")
    code = _print_report(report)
    print(f"  machine-readable results -> {JSON_PATH}")
    print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
