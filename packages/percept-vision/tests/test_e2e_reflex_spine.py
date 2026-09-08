"""E2E-1 — the reflex spine: "tell me when the door is left open", end to end.

The canonical rising-edge watch, driven through the SHIPPED entrypoint (:func:`percept.runtime.run`)
over synthesized frames. A still scene, a door opening, the door staying open, closing, re-opening.
Nothing is downloaded and nothing below the public entrypoint is mocked: the only doubles are the
scripted judge (:class:`percept.backends.fake.FakeVision`) and the synthetic clip
(:func:`percept.harness.stream_gen.parametric_motion_clip`).

The scenario, in one table (4 fps, 40 frames, ``t`` in source seconds)::

    frames  0..7    t 0.00 .. 1.75   still, door shut      <- the still prefix: it must cost NOTHING
    frame   8       t 2.00           the door swings open  <- occurrence 1
    frames  9..15   t 2.25 .. 3.75   still, door open      <- staying open must not re-fire
    frame   16      t 4.00           the door swings shut
    frames 17..23   t 4.25 .. 5.75   still, door shut
    frame   24      t 6.00           the door swings open  <- occurrence 2
    frames 25..39   t 6.25 .. 9.75   still, door open

What this file holds
--------------------
1. **One incident per opening.** Two openings produce exactly two incidents. The controls that make
   that a measurement rather than a coincidence sit beside it: three openings produce three, and a
   door that opens once and never shuts produces ONE incident out of fourteen tier-0 escalations.
   The count tracks rising edges, not motion.
1a. **Which mechanism actually holds that count** (``WhichMechanismOwnsTheDuplicate``). Three
   suppressors are stacked, and on the shipped physical-scene path the rising-edge latch the bar is
   named after is not the one doing the work — the episode fold suppresses 13 of 14 ticks before the
   gate is consulted at all. Peeling them off one at a time is the only way the headline number
   means anything about a mechanism.
2. **The still prefix is free.** Zero judge calls before the first opening — with a coast-bound arm
   of the same clip proving the zero is the gate's doing and not the harness never reaching a judge.
3. **Tier-0 economy**, read off the shared :mod:`percept.eval.stage_attribution` funnel rather than
   recounted here, with an all-motion arm that falls below the same bar.
4. **Timeliness** from the SOURCE occurrence to the durable event, with a sustained-duration
   (``hold_s``) arm that moves the lag and a longer one that breaks the bar.
5. **The cross-cutting invariants for every stage traversed** (specification section 3): one episode
   per occurrence, one entity id across both, the cache serving the repeat, and no future leakage.

What this file does NOT establish
---------------------------------
* **No latency claim.** The judge is scripted, so every wall-clock figure here is a floor on the
  runtime's own overhead, never a measurement of a system with a provider in it. The wall-clock
  assertions are limited to publishability (uncensored, all percentiles present) — the property the
  shared helper exists to enforce — and the timeliness bar is asserted in SOURCE time, which is the
  only clock a deterministic replay can speak about.
* **No model quality claim.** ``FakeVision`` looks a frame up in a dict.

Pins and characterizations, and what a fixer does with each
-----------------------------------------------------------
* ``test_a_cache_served_verdict_is_not_an_actual_judge_call`` — **FIXED** (was a strict xfail;
  FIND-REFLEX-01). The trace used to count a cache serve as a judge call. It now publishes
  ``provider_calls`` in explicit provider-call units, derived from logical starts minus every verdict
  carrying the shared zero-spend serve marker. A cache HIT — the gate still classifies, but zero
  vision calls are spent — is therefore not spend. Measured on this scenario: the cache cuts
  provider calls from 6 to 3 while
  ``judge_completed`` reads 6 either way, so the funnel reported twice the spend that happened and
  ``estimated_saved_calls`` read 34 when the truth is 37. The serve now carries the shared zero-spend
  marker (``percept.tracing.JUDGE_SERVED_KEY``, read through ``judge_call_was_served``) and folds
  into the subtracted counter, so this asserts the fixed behaviour.
* ``test_on_a_short_occurrence_the_merge_window_alone_holds_the_headline_count`` — **REWRITTEN**
  (was a characterization; FIND-REFLEX-05). It used to assert that a delivery-merged incident leaves
  no mark on the trace: ``gate_outcomes["suppressed"] == 0`` and two ``gate.classified`` records
  reading ``silent: rising_edge``. That is a silent action carrying a FIRE reason — the trace derives
  the action from the absent ``FireEvent`` and pairs it with the last gate outcome — so
  ``TraceSummary.gate_outcomes`` under-counted the gate's fires and over-counted its silents by
  exactly the merge count, and the funnel charged the GATE stage for a drop the DELIVERY layer made.
  The merge now announces a ``merged_duplicate`` suppression (a deliberate addition to the closed
  ``GATE_REASONS`` vocabulary, which ADR-006 open question 5 anticipates for exactly this seam), so
  the assertions read the fixed behaviour and the test opts into ``record_health`` — the bridge that
  puts a health event on the trace at all.
* ``test_an_open_episode_partitions_the_cache_slots_and_that_costs_calls`` — **characterization, not
  a pin.** It asserts today's behaviour: the context fingerprint is ``None`` with no episode open
  and the episode's hash while one is, and the frame-hash layer holds one entry per
  ``(goal, entity, condition, fingerprint)`` slot, so a verdict formed on one side of an episode
  boundary can never serve a tick on the other. Keying on the fingerprint is stated intent
  (``percept/cache/framehash.py:190-197``, ``percept/agent.py:7064``); the PRICE is stated nowhere.
  A fixer who changes that must REWRITE this assertion, not delete a marker.

  **This replaced a wrong mechanism, and the correction is the point.** The first reading of one
  measurement — six judged ticks, zero cache hits on the shipped profile — was written up as "an
  open scene episode disables the perception cache on the physical-scene path, so enabling it there
  buys nothing". Measured across three clips it is not: on the same profile a door that opens and
  stays open serves **12 of 14** judged ticks from cache, and a churning scene serves 37 of 39. The
  scenario in this file is simply the worst case for it — two judged ticks per occurrence and the
  door state flipping across every episode boundary, so all three of its available savings land on a
  slot transition. A fix written to "the episode text changes the fingerprint every tick" would have
  been aimed at a mechanism that does not exist.
"""

from __future__ import annotations

import asyncio
import inspect
import unittest
import unittest.mock
from dataclasses import dataclass
from typing import Any, Sequence

from percept import Goal, Percept
from percept.adapters import Box, Observation, Verdict
from percept.backends.fake import FakeVision
from percept.behavior import INCIDENT_MERGE_WINDOW_S
from percept.cache.framehash import dhash, hamming
from percept.config import CacheSettings, Settings
from percept.gate import Gate
from percept.eval.stage_attribution import StageAttribution, attribute
from percept.harness.stream_gen import parametric_motion_clip
from percept.runtime import RuntimeStats, run
from percept.source_profiles_default import PHYSICAL_SCENE
from percept.tracing import (
    STAGE_EDGE_DECISION,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_STARTED,
    BaselineAssumption,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    TraceRecord,
    TraceSummary,
)

# --------------------------------------------------------------------------------------------
# The scene
# --------------------------------------------------------------------------------------------

#: The runtime's own tier-0 downsample geometry (``percept.runtime.source.GATE_W/GATE_H``), so the
#: gate frames are the shape a real camera path would hand it.
GATE_W, GATE_H = 48, 36
#: 4 fps source. Every ``t`` in the docstring table is ``index * FRAME_DT``.
FRAME_DT = 0.25
#: A door-sized patch of the 1728-pixel gate frame — comfortably above the shipped ``block_max``
#: admission bar, so an opening is never a near-threshold coin flip.
MOTION_PIXELS = 400

N_FRAMES = 40
FIRST_OPEN_I = 8
CLOSES_I = 16
SECOND_OPEN_I = 24

#: The two source occurrences, in source seconds. These are the ground truth every timeliness and
#: recall assertion below is measured against.
OPENINGS_T: tuple[float, ...] = (FIRST_OPEN_I * FRAME_DT, SECOND_OPEN_I * FRAME_DT)

#: The two judge-side frames. They are nested numeric grids rather than string tokens for one
#: reason: the perception cache hashes what it is given (``percept.cache.framehash.dhash``) and a
#: bare string is un-hashable, so a string-token scenario can never exercise the cache at all.
#: SHUT is flat (every horizontal gradient is False -> hash 0); OPEN is a strict ramp (every
#: gradient is True -> all 64 bits set). ``TheCacheServesTheRepeat`` asserts that separation.
SHUT_FRAME: tuple[tuple[float, ...], ...] = tuple(tuple(0.0 for _ in range(9)) for _ in range(8))
OPEN_FRAME: tuple[tuple[float, ...], ...] = tuple(
    tuple(float(x * 10) for x in range(9)) for _ in range(8)
)

DOOR_IS_OPEN = Verdict(satisfied=True, confidence=0.95, evidence="the door is open")
DOOR_IS_SHUT = Verdict(satisfied=False, confidence=0.95, evidence="the doorway is shut")

#: The eye is expected to filter most of a mostly-still scene. THIS NUMBER IS THE TEST'S OWN and is
#: not sourced from a profile: no shipped :class:`~percept.sourceprofile.SourceProfile` declares an
#: admission-economy floor. The only profile-declared admission bar is ``parameters['coverage_floor']``
#: — a MINIMUM admission count (the honest-deafness direction), the opposite of a ceiling on the
#: admitted fraction — and ``PHYSICAL_SCENE`` declares neither. Treated here as a smoke bar with an
#: arm that falls below it, never as a calibrated release gate.
ADMISSION_ECONOMY_SMOKE_FLOOR = 0.5

#: The promise window a fire must land inside, measured from the source occurrence
#: (``PRODUCT_SPEC`` reflex p95). Asserted in SOURCE seconds — see the module docstring.
REFLEX_WINDOW_S = 2.0


def _clip(
    *,
    n: int,
    swings: Sequence[int],
    open_spans: Sequence[tuple[int, int]],
) -> list[tuple[float, Any, Any]]:
    """One synthetic watch stream as ``(t, gate_frame, judge_frame)`` triples.

    ``swings`` are the frame indices where the door physically moves — the gate frames the tier-0
    eye sees. ``open_spans`` are half-open index ranges over which the door IS open — the judge
    frames the scripted model reads. The two are deliberately independent: motion is what buys a
    model call, and the model is what says what the scene means.
    """
    timeline = parametric_motion_clip(
        w=GATE_W, h=GATE_H, n=n, motion_frames=swings, motion_pixels=MOTION_PIXELS, dt=FRAME_DT
    )
    return [
        (
            frame.t,
            frame.frame,
            OPEN_FRAME if any(lo <= i < hi for lo, hi in open_spans) else SHUT_FRAME,
        )
        for i, frame in enumerate(timeline)
    ]


#: The scenario in the module docstring.
SPINE = _clip(
    n=N_FRAMES,
    swings=(FIRST_OPEN_I, CLOSES_I, SECOND_OPEN_I),
    open_spans=((FIRST_OPEN_I, CLOSES_I), (SECOND_OPEN_I, N_FRAMES)),
)

#: The other half of "left open": the door opens once at t=2.0 and never shuts, while the scene keeps
#: moving. ONE occurrence, fourteen tier-0 escalations, fourteen judged ticks — which is what makes
#: it the arm that can say anything about the cache. ``SPINE`` judges only two ticks per occurrence,
#: so it has almost no repeat to serve.
LEFT_OPEN = _clip(n=N_FRAMES, swings=(8, 12, 16, 20, 28, 32, 36), open_spans=((8, N_FRAMES),))


# --------------------------------------------------------------------------------------------
# Driving the shipped runtime
# --------------------------------------------------------------------------------------------


class _CountingVision(FakeVision):
    """``FakeVision`` that counts the calls that actually reached the provider seam.

    The only way to tell a paid model call from a verdict the cache served: every counter inside the
    runtime is upstream of the cache.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.provider_calls = 0

    async def judge(self, condition: str, frame: Any, **kw: Any) -> Verdict:  # type: ignore[override]
        self.provider_calls += 1
        return await super().judge(condition, frame, **kw)


@dataclass(frozen=True)
class _Run:
    """Everything one drive of the runtime produced, so an assertion can name its stage."""

    incidents: list
    stats: RuntimeStats
    records: tuple[TraceRecord, ...]
    summary: TraceSummary
    attribution: StageAttribution
    core: Percept
    provider_calls: int
    health: list

    def judged_before(self, t: float) -> int:
        """Judge calls the trace shows for source evidence strictly earlier than ``t``."""
        return sum(
            1
            for r in self.records
            if r.stage == STAGE_JUDGE_STARTED and r.source_ts is not None and r.source_ts < t
        )

    def observed_before(self, t: float) -> int:
        return sum(
            1
            for r in self.records
            if r.stage == STAGE_EDGE_DECISION and r.source_ts is not None and r.source_ts < t
        )


def _drive(
    source: Sequence[tuple[float, Any, Any]],
    *,
    state: dict | None = None,
    source_profile: Any = PHYSICAL_SCENE,
    cache: bool = False,
    detects: bool = False,
    evidence_window_s: float = 0.5,
    record_health: bool = False,
) -> _Run:
    """Run one watch over ``source`` through :func:`percept.runtime.run` and record everything.

    ``evidence_window_s`` is declared on the manifest so ``admitted_evidence_union_s`` measures a
    real UNION of overlapping windows rather than the ``0.0`` an undeclared window correctly yields.

    ``record_health`` additionally routes the runtime's health channel into the recorder
    (``TraceRecorder.health_sink``), which is the bridge that turns a ``gate_suppressed`` event into
    a ``gate.suppressed`` trace stage. OFF by default so every other assertion in this file measures
    the same record stream it always did; the one test that reads suppression from the funnel opts in.
    """
    detect_script = (
        {
            str(OPEN_FRAME): [Observation(label="door", box=Box(0.1, 0.1, 0.3, 0.6), confidence=0.9)],
            str(SHUT_FRAME): [Observation(label="door", box=Box(0.1, 0.1, 0.3, 0.6), confidence=0.9)],
        }
        if detects
        else None
    )
    vision = _CountingVision(
        script={str(OPEN_FRAME): DOOR_IS_OPEN, str(SHUT_FRAME): DOOR_IS_SHUT},
        detect_script=detect_script,
    )
    settings = Settings(
        # ttl_s is deliberately far longer than any wall-clock this run can take, so the cache arm
        # never depends on how fast the machine replayed the clip.
        cache=CacheSettings(enabled=cache, ttl_s=3600.0, audit_rate=0.0)
    )
    kwargs = {"source_profile": source_profile} if source_profile is not None else {}
    core = Percept.create(
        vision=vision, settings=settings, discover_plugins=False, **kwargs
    )
    core.add_goal(
        Goal(
            id="door",
            condition="the door is open",
            say="the door is open",
            state=dict(state or {}),
        )
    )

    manifest = RunTraceManifest.create(
        source="camera",
        driver="synthetic reflex-spine clip",
        goals=list(core.goals.active()),
        policy=TracePolicy(level="full"),
        evidence_window_s=evidence_window_s,
        # The counterfactual every economy figure here is measured against, declared explicitly:
        # an ungated sampler that judges every observed frame.
        baseline=BaselineAssumption.every_observation(len(source)),
    )
    recorder = TraceRecorder(manifest)
    stats = RuntimeStats()
    health: list = []
    health_to_trace = recorder.health_sink() if record_health else None

    def on_health(event: Any) -> None:
        health.append(event)
        if health_to_trace is not None:
            health_to_trace(event)

    async def _collect() -> list:
        return [
            incident
            async for incident in run(
                core,
                source=list(source),
                stats=stats,
                judge_dispatch="inline",
                on_trace=recorder.tracer(),
                on_health=on_health,
            )
        ]

    incidents = asyncio.run(_collect())
    recorder.close()
    records = recorder.records
    return _Run(
        incidents=incidents,
        stats=stats,
        records=records,
        summary=recorder.summary(),
        attribution=attribute(records, manifest=manifest),
        core=core,
        provider_calls=vision.provider_calls,
        health=health,
    )


def _fire_times(runres: _Run) -> list[float]:
    return [float(incident.ts) for incident in runres.incidents]


def _future_leaks(records: Sequence[TraceRecord]) -> list[tuple[float, str, float]]:
    """Records that precede an ``incident.created`` in ingest order while carrying LATER source time.

    A decision at ``t`` may only rest on evidence available at or before ``t``. The trace is written
    in the order the runtime actually decided things, so a record from the future sitting upstream
    of a fire is exactly the leak this predicate is looking for.
    """
    leaks: list[tuple[float, str, float]] = []
    for i, record in enumerate(records):
        if record.stage != STAGE_INCIDENT_CREATED or record.source_ts is None:
            continue
        for earlier in records[:i]:
            if earlier.source_ts is not None and earlier.source_ts > record.source_ts:
                leaks.append((float(record.source_ts), earlier.stage, float(earlier.source_ts)))
    return leaks


# --------------------------------------------------------------------------------------------
# 1. Exactly one incident per opening
# --------------------------------------------------------------------------------------------


class OneIncidentPerOpening(unittest.TestCase):
    """Two openings -> two incidents, never three. And the count tracks EDGES, not motion."""

    def test_two_openings_produce_exactly_two_incidents(self) -> None:
        result = _drive(SPINE)
        self.assertEqual(len(result.incidents), 2, _explain(result))
        self.assertEqual(_fire_times(result), list(OPENINGS_T))
        # Every fire is the gate's rising edge, and the gate agrees with the delivered count.
        self.assertEqual({i.reason for i in result.incidents}, {"rising_edge"})
        self.assertEqual(result.attribution.funnel.gate_outcomes["fire"], 2)
        self.assertEqual(result.summary.incidents, 2)

    def test_the_door_staying_open_does_not_re_fire(self) -> None:
        """The seven still frames while the door stands open must add no third incident.

        They are also, deliberately, not free of judge calls: the frame where the door settles back
        to still is itself a change, so it IS admitted and IS judged — and answers "open" — and the
        gate suppresses it. The duplicate is stopped at the gate, not at admission, and that is the
        distinction this asserts.
        """
        result = _drive(SPINE)
        judged_during_first_occurrence = [
            r.source_ts
            for r in result.records
            if r.stage == STAGE_JUDGE_STARTED
            and r.source_ts is not None
            and OPENINGS_T[0] <= r.source_ts < CLOSES_I * FRAME_DT
        ]
        self.assertEqual(judged_during_first_occurrence, [2.0, 2.25])
        self.assertEqual(len([t for t in _fire_times(result) if t < CLOSES_I * FRAME_DT]), 1)
        # A judged frame that did not fire left a `silent` gate outcome, not a second incident.
        self.assertEqual(result.attribution.funnel.gate_outcomes["silent"], 4)

    def test_a_third_opening_produces_a_third_incident(self) -> None:
        """MUTATION CONTROL for the assertion above: the answer is not the constant 2."""
        three = _clip(
            n=56,
            swings=(8, 16, 24, 32, 40),
            open_spans=((8, 16), (24, 32), (40, 56)),
        )
        result = _drive(three)
        self.assertEqual(len(result.incidents), 3)
        self.assertEqual(_fire_times(result), [2.0, 6.0, 10.0])

    def test_a_door_that_never_shuts_fires_once_despite_fourteen_escalations(self) -> None:
        """MUTATION CONTROL: the count tracks rising EDGES, not tier-0 admissions.

        Same length of clip, the door opens once at t=2.0 and stays open while the scene keeps
        moving. Tier 0 escalates fourteen times; exactly one incident is delivered.
        """
        result = _drive(LEFT_OPEN)
        self.assertEqual(result.stats.escalations, 14)
        self.assertEqual(len(result.incidents), 1, _explain(result))
        self.assertEqual(_fire_times(result), [2.0])

    def test_a_never_rearming_watch_reports_only_the_first_opening(self) -> None:
        """MUTATION CONTROL: the count is the goal PROGRAM's answer, not the harness's.

        Same clip, same eye, same judge — one word of goal state (``rearm="never"``, the
        "tell me the first time" watch) and the second opening is deliberately not reported.
        """
        once = _drive(SPINE, state={"rearm": "never"})
        self.assertEqual(_fire_times(once), [OPENINGS_T[0]])
        self.assertEqual(once.attribution.funnel.gate_outcomes["fire"], 1)


class WhichMechanismOwnsTheDuplicate(unittest.TestCase):
    """"One incident per opening" is held by THREE stacked suppressors. Naming the wrong one is easy.

    The rising-edge latch (``percept/gate.py``, one armed bit per key) is what the E2E-1 bar is
    written about. It is not, on the shipped physical-scene path, the mechanism doing the work:

    1. **Episode-owned idempotency** (``percept/behavior.py``: "the episode already reported and
       nothing changed — do not consult the gate at all"). On ``LEFT_OPEN`` this suppresses 13 of the
       14 judged ticks, and :meth:`Gate.decide` is reached exactly ONCE in the whole run.
    2. **The rising-edge latch** — what is left once the episode fold is not in play.
    3. **The delivery merge** (:data:`percept.behavior.INCIDENT_MERGE_WINDOW_S`, 0.7 s), which drops
       a re-notification of a key that already delivered inside the window.

    Each of the three alone is enough to hold the count on some scenario in this file, which is
    exactly why the headline bar cannot be read as evidence about any one of them. The mutants below
    take the layers off one at a time and measure which is load-bearing where.
    """

    @staticmethod
    def _latch_disabled():
        """Re-arm the gate key after every decision — the rising-edge latch, deleted."""
        original = Gate.decide

        def rearmed(self: Gate, key: Any, *a: Any, **kw: Any) -> Any:
            outcome = original(self, key, *a, **kw)
            state = self._state(key)
            state.armed = True
            state.last_fire = None
            return outcome

        return unittest.mock.patch.object(Gate, "decide", rearmed)

    @staticmethod
    def _counting_gate(sink: list) -> Any:
        original = Gate.decide

        def counted(self: Gate, key: Any, *a: Any, **kw: Any) -> Any:
            sink.append(key)
            return original(self, key, *a, **kw)

        return unittest.mock.patch.object(Gate, "decide", counted)

    def test_on_the_shipped_path_the_episode_suppresses_before_the_gate_is_consulted(self) -> None:
        """The gate cannot be what holds the count when it is asked once in fourteen ticks."""
        consulted: list = []
        with self._counting_gate(consulted):
            shipped = _drive(LEFT_OPEN, source_profile=PHYSICAL_SCENE)
        self.assertEqual(shipped.summary.judge_started, 14)
        self.assertEqual(len(consulted), 1)
        self.assertEqual(shipped.core.metrics.snapshot()["gate.idempotent_suppressed"], 13)
        self.assertEqual(_fire_times(shipped), [OPENINGS_T[0]])

        # ...so deleting the latch entirely changes nothing on this path.
        with self._latch_disabled():
            latchless = _drive(LEFT_OPEN, source_profile=PHYSICAL_SCENE)
        self.assertEqual(_fire_times(latchless), [OPENINGS_T[0]])

    def test_without_the_episode_fold_the_latch_is_what_holds_the_count(self) -> None:
        """Drop the profile and the gate is asked on every tick — and deleting the latch shows."""
        consulted: list = []
        with self._counting_gate(consulted):
            bare = _drive(LEFT_OPEN, source_profile=None)
        self.assertEqual(len(consulted), 14)
        self.assertEqual(_fire_times(bare), [OPENINGS_T[0]])

        with self._latch_disabled():
            latchless = _drive(LEFT_OPEN, source_profile=None)
        self.assertEqual(_fire_times(latchless), [2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0])
        # Seven MORE duplicates existed and the merge window absorbed those — the third layer,
        # visible only once the two above it are gone.
        self.assertEqual(latchless.core.metrics.snapshot()["gate.merged_duplicate"], 7)

    def test_on_a_short_occurrence_the_merge_window_alone_holds_the_headline_count(self) -> None:
        """``SPINE`` is judged twice per occurrence, 0.25 s apart — inside the 0.7 s merge window.

        With the episode fold gone AND the latch deleted, the headline "2 openings -> 2 incidents"
        still passes, held by nothing but the merge. The bar is true; it is not evidence about the
        rising edge on this clip.
        """
        judged = sorted(
            r.source_ts
            for r in _drive(SPINE, source_profile=None).records
            if r.stage == STAGE_JUDGE_STARTED and r.source_ts is not None
        )
        self.assertEqual({round(b - a, 6) for a, b in zip(judged, judged[1:])}, {0.25, 1.75})
        self.assertLess(FRAME_DT, INCIDENT_MERGE_WINDOW_S)

        with self._latch_disabled():
            latchless = _drive(SPINE, source_profile=None, record_health=True)
        self.assertEqual(_fire_times(latchless), list(OPENINGS_T))  # STILL two
        self.assertEqual(latchless.core.metrics.snapshot()["gate.merged_duplicate"], 2)

        # ...and the merge is ON THE TRACE, attributed to the layer that made it. REWRITTEN with the
        # fix for finding reflex-05: the merge used to leave nothing behind but the in-process
        # counter above, while the trace derived the record's action from the absent FireEvent and
        # paired it with the gate's own reason — so a .ptrace read `silent: rising_edge` twice and
        # TraceSummary.gate_outcomes charged the GATE for a drop the DELIVERY layer made. The merge
        # now announces a `merged_duplicate` suppression (a new code in the closed GATE_REASONS
        # vocabulary — ADR-006 open question 5 names this seam and names the price), so
        # `gate.suppressed` counts it and the classified record says which layer withheld it.
        self.assertEqual(latchless.attribution.funnel.gate_outcomes["suppressed"], 2)
        reasons = [
            r.data.get("reason")
            for r in latchless.records
            if r.stage == STAGE_GATE_CLASSIFIED and r.data.get("action") == "silent"
        ]
        self.assertEqual(reasons.count("silent: merged_duplicate"), 2)
        self.assertEqual(reasons.count("silent: rising_edge"), 0, "never the gate's own reason")
        # The health channel carries the same fact, which is what an operator actually reads.
        self.assertEqual(
            [
                ev.data.get("reason")
                for ev in latchless.health
                if ev.kind == "gate_suppressed"
            ],
            ["merged_duplicate", "merged_duplicate"],
        )


# --------------------------------------------------------------------------------------------
# 2. The still prefix costs nothing
# --------------------------------------------------------------------------------------------


class StillFramesCostZeroJudgeCalls(unittest.TestCase):
    def test_the_still_prefix_spends_zero_judge_calls(self) -> None:
        result = _drive(SPINE)
        first_opening = OPENINGS_T[0]
        # The eye SAW all eight prefix frames — this is not a run that skipped them...
        self.assertEqual(result.observed_before(first_opening), FIRST_OPEN_I)
        # ...and admitted none of them, so nothing reached a model.
        self.assertEqual(result.judged_before(first_opening), 0)
        self.assertEqual(result.provider_calls, 6)  # all six are at or after the first opening
        prefix_escalations = [
            r
            for r in result.records
            if r.stage == STAGE_EDGE_DECISION
            and r.source_ts is not None
            and r.source_ts < first_opening
            and r.data.get("escalate")
        ]
        self.assertEqual(prefix_escalations, [])

    def test_a_coast_bound_watch_spends_judge_calls_on_the_same_still_prefix(self) -> None:
        """MUTATION CONTROL: the zero above is the tier-0 gate's doing, not the harness's.

        Bind a fixed sampling floor onto the same goal over the same frames and the still prefix
        starts costing model calls. The zero can fail; on the reflex spine it does not.
        """
        coasting = _drive(SPINE, state={"sense": {"base_fps": 2.0, "max_fps": 4.0}})
        self.assertGreater(coasting.stats.coast_admits, 0)
        self.assertGreater(coasting.judged_before(OPENINGS_T[0]), 0)
        # Nearly four times the model calls for the same two incidents...
        self.assertGreater(coasting.provider_calls, 3 * _drive(SPINE).provider_calls)
        # ...so the outcome is unchanged and it is the COST that moved. That is the whole of tier 0,
        # and it also says the duplicate suppression survives heavy oversampling of an open door.
        self.assertEqual(_fire_times(coasting), list(OPENINGS_T))


# --------------------------------------------------------------------------------------------
# 3. Tier-0 economy
# --------------------------------------------------------------------------------------------


class Tier0Economy(unittest.TestCase):
    def test_admission_economy_is_read_from_the_shared_funnel(self) -> None:
        result = _drive(SPINE)
        funnel = result.attribution.funnel
        self.assertEqual(funnel.counts["observed"], N_FRAMES)
        self.assertEqual(funnel.counts["admitted"], 6)
        # The helper's number and the summary's are the same number, not two derivations of it.
        self.assertAlmostEqual(
            funnel.admission_economy,
            1.0 - result.summary.admitted / result.summary.observed,
        )
        self.assertAlmostEqual(funnel.admission_economy, 0.85)
        self.assertGreaterEqual(funnel.admission_economy, ADMISSION_ECONOMY_SMOKE_FLOOR)

    def test_the_funnel_spine_is_monotone_on_this_single_goal_watch(self) -> None:
        """``seen >= gated >= judged >= fired``.

        Stated for THIS watch, which arms exactly one goal. ``admitted`` counts units of evidence and
        ``judge_started`` counts ``(evidence x goal)``, so the relation is a property of the
        single-goal shape here and is not claimed to generalise.
        """
        result = _drive(SPINE)
        counts = result.attribution.funnel.counts
        self.assertEqual(len(result.core.goals.active()), 1)
        self.assertGreaterEqual(counts["observed"], counts["admitted"])
        self.assertGreaterEqual(counts["admitted"], counts["judge_started"])
        self.assertGreaterEqual(counts["judge_started"], counts["judge_completed"])
        self.assertGreaterEqual(counts["judge_completed"], counts["fired"])

    def test_admitted_seconds_are_a_union_of_windows_not_count_times_interval(self) -> None:
        """Six admissions x a 0.5s declared window is 3.0s of source. The truth is 2.25s.

        The three pairs of admissions overlap (each door swing admits the swing frame and the frame
        where the scene settles, 0.25s apart, under a 0.5s window), so the multiplication
        over-reports by a third. The funnel reports the union.
        """
        result = _drive(SPINE, evidence_window_s=0.5)
        funnel = result.attribution.funnel
        self.assertEqual(funnel.counts["admitted"], 6)
        self.assertAlmostEqual(funnel.admitted_evidence_union_s, 2.25)
        self.assertNotAlmostEqual(funnel.admitted_evidence_union_s, 6 * 0.5)

    def test_an_all_motion_clip_falls_below_the_same_bar(self) -> None:
        """MUTATION CONTROL: the economy bar can fail, so passing it means something."""
        churning = _clip(
            n=N_FRAMES, swings=tuple(range(0, N_FRAMES, 2)), open_spans=((0, N_FRAMES),)
        )
        result = _drive(churning)
        self.assertLess(result.attribution.funnel.admission_economy, ADMISSION_ECONOMY_SMOKE_FLOOR)

    def test_the_run_declares_the_baseline_its_savings_is_measured_against(self) -> None:
        """A savings figure with no stated counterfactual is not a measurement."""
        result = _drive(SPINE)
        self.assertTrue(result.summary.baseline.declared)
        self.assertEqual(result.summary.estimated_baseline_calls, N_FRAMES)


# --------------------------------------------------------------------------------------------
# 4. Timeliness
# --------------------------------------------------------------------------------------------


class TheFireIsTimely(unittest.TestCase):
    def test_each_fire_lands_inside_the_reflex_window_of_its_source_occurrence(self) -> None:
        result = _drive(SPINE)
        lags = [fire - occurrence for fire, occurrence in zip(_fire_times(result), OPENINGS_T)]
        self.assertEqual(len(lags), len(OPENINGS_T))
        for lag in lags:
            self.assertGreaterEqual(lag, 0.0)  # never before the evidence that caused it
            self.assertLessEqual(lag, REFLEX_WINDOW_S)
        # On a rising-edge watch with no dwell the pipeline commits on the FIRST frame of the
        # occurrence, so the lag is exactly zero. That is the property; the arms below are what make
        # the bar above able to fail.
        self.assertEqual(lags, [0.0, 0.0])

    def test_a_sustained_duration_watch_moves_the_lag_by_exactly_its_dwell(self) -> None:
        """"Left open" is a dwell. A half-second hold moves both fires half a second later."""
        left_open = _drive(SPINE, state=_sustained(hold_s=0.5))
        lags = [fire - occurrence for fire, occurrence in zip(_fire_times(left_open), OPENINGS_T)]
        self.assertEqual(lags, [0.5, 0.5])
        for lag in lags:
            self.assertLessEqual(lag, REFLEX_WINDOW_S)

    def test_a_dwell_longer_than_the_window_breaks_the_timeliness_bar(self) -> None:
        """MUTATION CONTROL: the reflex bar is not hollow — this configuration fails it.

        A three-second hold on the same clip fires once, three seconds after the occurrence it is
        reporting, and misses the second occurrence entirely. Both halves of "timely recall" — the
        lag AND the recall — go red together.
        """
        slow = _drive(SPINE, state=_sustained(hold_s=3.0))
        fires = _fire_times(slow)
        self.assertEqual(len(fires), 1)  # recall: one of two occurrences reported
        self.assertGreater(fires[0] - OPENINGS_T[1], REFLEX_WINDOW_S)

    def test_the_wall_clock_segments_are_uncensored_and_may_be_quoted(self) -> None:
        """Publishability, not a latency number.

        No deadline was declared and every judge span closed, so the helper must be willing to quote
        each segment. What it would quote is the runtime's own overhead against a dict lookup — see
        the module docstring — so no percentile VALUE is asserted here.
        """
        result = _drive(SPINE)
        attribution = result.attribution
        self.assertTrue(attribution.publishable, attribution.describe())
        self.assertEqual(attribution.unpublishable_segments, ())
        self.assertEqual(attribution.unmeasured_segments, ())
        self.assertFalse(attribution.distribution("judge").is_censored)
        self.assertFalse(attribution.truncation["judge"].is_truncated)
        # Two spans reached the delivery segment: one per fire. The last stage really ran.
        self.assertEqual(attribution.truncation["delivery"].closes, 2)
        self.assertIsNotNone(attribution.slowest_segment())


def _sustained(*, hold_s: float) -> dict:
    """The goal state a sustained-duration ("left open for N seconds") watch requires.

    ``percept.duration_routes`` refuses a worker-driven ``hold_s`` watch that does not also declare a
    fixed judge cadence — a dwell timer sampled by a motion gate would credit time it never observed
    — so the cadence is part of the shape, not decoration.
    """
    return {
        "hold_s": hold_s,
        "cadence": {"min_judge_interval": FRAME_DT, "gate_on": "fixed"},
        "sense": {"base_fps": 1.0 / FRAME_DT, "max_fps": 1.0 / FRAME_DT},
    }


# --------------------------------------------------------------------------------------------
# 5. The cross-cutting invariants for every stage traversed
# --------------------------------------------------------------------------------------------


class OneEpisodePerOccurrence(unittest.TestCase):
    """A run that fires correctly but mints two episodes has not passed."""

    def test_two_occurrences_produce_exactly_two_scene_episodes(self) -> None:
        result = _drive(SPINE)
        log = result.core.episodes["door"]
        self.assertEqual(log.count(), 2)
        episodes = list(log._episodes)
        self.assertEqual([e.started_at for e in episodes], list(OPENINGS_T))
        # The first occurrence CLOSED when the door shut; the second is still open at end of stream.
        self.assertEqual([e.status for e in episodes], ["closed", "open"])
        self.assertEqual(episodes[0].t_end, CLOSES_I * FRAME_DT)
        # One subject for the whole run: the scene. Occurrence count is the episode's job; counting
        # subjects is not (percept/tests/test_camera_scene_episode_fold.py states that forfeit).
        self.assertEqual({e.subject_key for e in episodes}, {"scene"})

    def test_the_still_frames_inside_an_occurrence_do_not_open_a_third_episode(self) -> None:
        result = _drive(SPINE)
        episodes = list(result.core.episodes["door"]._episodes)
        self.assertEqual(len(episodes), len(OPENINGS_T))
        self.assertEqual(episodes[0].observations, 2)  # the swing frame and the settle frame

    def test_a_watch_with_no_source_profile_mints_no_episode_at_all(self) -> None:
        """MUTATION CONTROL, and a boundary worth stating out loud.

        The two-episode result above is a property of the physical-scene profile's identity
        strategy. Drop the profile — which is what a bare ``Percept.create(vision=...)`` does — and
        the fold declines for want of a content key, the episode map stays EMPTY, and the run still
        delivers the same two incidents. Episode idempotency is therefore not something the reflex
        spine gets for free; it is something the profile buys.
        """
        bare = _drive(SPINE, source_profile=None)
        self.assertEqual(bare.core.episodes, {})
        self.assertEqual(_fire_times(bare), list(OPENINGS_T))


class OneContinuousIdentity(unittest.TestCase):
    def test_one_entity_id_carries_both_occurrences(self) -> None:
        result = _drive(SPINE, detects=True)
        self.assertEqual(len(result.incidents), 2)
        entity_ids = {incident.entity_id for incident in result.incidents}
        self.assertEqual(len(entity_ids), 1)
        self.assertIsNotNone(next(iter(entity_ids)))
        # ...and the graph holds exactly that one entity, so the id was not minted per occurrence.
        self.assertEqual(len(result.core.graph._entities), 1)
        self.assertEqual(set(result.core.graph._entities), entity_ids)

    def test_the_gate_keeps_one_key_and_leaves_no_orphan_state(self) -> None:
        result = _drive(SPINE, detects=True)
        keys = list(result.core.gate._keys)
        self.assertEqual(len(keys), 1)
        goal_id, entity_id = keys[0]
        self.assertEqual(goal_id, "door")
        self.assertEqual(entity_id, result.incidents[0].entity_id)

    def test_without_a_detector_there_is_no_entity_to_carry(self) -> None:
        """The boundary, stated rather than papered over.

        With no detector wired the incidents carry ``entity_id=None`` and the graph is empty. That
        is correct — nothing was individuated — but it means the identity invariant above is bought
        by the detector, not by the reflex spine.
        """
        result = _drive(SPINE, detects=False)
        self.assertEqual({i.entity_id for i in result.incidents}, {None})
        self.assertEqual(len(result.core.graph._entities), 0)


class NoFutureLeakage(unittest.TestCase):
    def test_no_decision_rests_on_evidence_that_had_not_arrived(self) -> None:
        result = _drive(SPINE)
        self.assertEqual(_future_leaks(result.records), [])

    def test_each_incidents_evidence_window_ends_at_its_own_fire(self) -> None:
        result = _drive(SPINE)
        for incident in result.incidents:
            snippet = incident.snippet
            self.assertLessEqual(snippet.t_end, incident.ts)
            self.assertLessEqual(snippet.t_start, snippet.t_end)

    def test_the_leak_predicate_catches_a_planted_leak(self) -> None:
        """MUTATION CONTROL for the predicate itself: an empty result must mean something.

        Splice one edge decision from t=5.25 in ahead of the fire at t=2.0 and the same predicate
        that returned nothing above returns that record.
        """
        records = list(_drive(SPINE).records)
        first_fire = next(i for i, r in enumerate(records) if r.stage == STAGE_INCIDENT_CREATED)
        from_the_future = next(
            r for r in records if r.stage == STAGE_EDGE_DECISION and (r.source_ts or 0.0) > 5.0
        )
        records.insert(first_fire, from_the_future)
        leaks = _future_leaks(records)
        self.assertTrue(leaks)
        self.assertEqual(leaks[0][0], OPENINGS_T[0])
        self.assertGreater(leaks[0][2], OPENINGS_T[0])


class TheCacheServesTheRepeat(unittest.TestCase):
    """The perception cache must serve a repeat and must never serve across a state change."""

    def test_the_two_door_states_are_maximally_distinguishable_to_the_cache(self) -> None:
        """The premise the two tests below rest on, asserted rather than assumed."""
        shut, opened = dhash(SHUT_FRAME), dhash(OPEN_FRAME)
        self.assertIsNotNone(shut)
        self.assertIsNotNone(opened)
        self.assertEqual(hamming(shut, opened), 64)
        self.assertGreater(hamming(shut, opened), CacheSettings().tau_bits)

    def test_the_cache_halves_the_provider_calls_without_changing_the_outcome(self) -> None:
        cold = _drive(SPINE, source_profile=None, cache=False)
        warm = _drive(SPINE, source_profile=None, cache=True)
        self.assertEqual(cold.provider_calls, 6)
        self.assertEqual(warm.provider_calls, 3)
        self.assertEqual(warm.core.metrics.snapshot()["cache.hit"], 3)
        # The saving is free: same incidents, same times, same verdicts.
        self.assertEqual(_fire_times(warm), _fire_times(cold))
        self.assertEqual(_fire_times(warm), list(OPENINGS_T))

    def test_a_state_change_is_never_served_from_the_stale_slot(self) -> None:
        """The fence. Three of six judged frames are repeats and are served; the three that changed
        state are re-perceived, so the door shutting is never answered from an open-door verdict."""
        warm = _drive(SPINE, source_profile=None, cache=True)
        verdicts = [
            r.data.get("satisfied")
            for r in warm.records
            if r.stage == "judge.finished" and r.source_ts is not None
        ]
        self.assertEqual(verdicts, [True, True, False, False, True, True])
        self.assertEqual(warm.provider_calls, 3)  # exactly the three state changes

    def test_the_shipped_profile_serves_the_repeat_on_a_door_left_open(self) -> None:
        """The cross-cutting "the cache serves the repeat" invariant, on the SHIPPED camera path.

        This is the arm that can actually say it. A door that opens once and stays open is judged
        fourteen times; twelve of those are the same open doorway again, and the cache serves all
        twelve without the physical-scene profile being dropped. One incident either way.
        """
        warm = _drive(LEFT_OPEN, source_profile=PHYSICAL_SCENE, cache=True)
        cold = _drive(LEFT_OPEN, source_profile=PHYSICAL_SCENE, cache=False)
        self.assertEqual(warm.summary.judge_started, 14)
        self.assertEqual(cold.provider_calls, 14)
        self.assertEqual(warm.provider_calls, 2)
        self.assertEqual(warm.core.metrics.snapshot()["cache.hit"], 12)
        self.assertEqual(_fire_times(warm), _fire_times(cold))
        self.assertEqual(_fire_times(warm), [OPENINGS_T[0]])

    def test_an_open_episode_partitions_the_cache_slots_and_that_costs_calls(self) -> None:
        """CHARACTERIZATION of today's behaviour — see the module docstring, not a pin.

        The context fingerprint is ``None`` while no episode is open and the episode's own hash
        while one is, and :class:`percept.cache.framehash.FrameHashLayer` holds ONE entry per
        ``(goal, entity, condition, fingerprint)`` slot. The tick that opens an episode is judged
        before the fold runs, so its verdict lands in the ``None`` slot and every later tick of that
        episode looks in a different one. Keying on the fingerprint is stated intent — a verdict
        formed under one episode state must not be served under another. The PRICE is what is
        characterized here, and it is scenario-shaped, not per-episode:

        * ``LEFT_OPEN`` — one long occurrence: 1 extra call, and 12 of 14 ticks still served.
        * ``SPINE`` — two short occurrences with the state flipping across each boundary: 3 extra
          calls out of 3 available, so the cache serves NOTHING. Each return to the ``None`` slot
          finds it holding the other door state, stored a whole occurrence ago.

        The second shape read alone says "an open episode disables the cache". The first says it
        costs one call. Only both together say what actually happens.
        """
        bare_open, shipped_open = (
            _drive(LEFT_OPEN, source_profile=p, cache=True) for p in (None, PHYSICAL_SCENE)
        )
        self.assertEqual((bare_open.provider_calls, shipped_open.provider_calls), (1, 2))
        self.assertEqual(shipped_open.core.episodes["door"].count(), 1)

        bare_spine, shipped_spine = (
            _drive(SPINE, source_profile=p, cache=True) for p in (None, PHYSICAL_SCENE)
        )
        self.assertEqual((bare_spine.provider_calls, shipped_spine.provider_calls), (3, 6))
        self.assertEqual(shipped_spine.core.episodes["door"].count(), 2)
        # Three extra calls for two episodes: the cost is NOT one per episode. It is every slot
        # transition, and this clip has three of them.
        self.assertGreater(
            shipped_spine.provider_calls - bare_spine.provider_calls,
            shipped_spine.core.episodes["door"].count(),
        )
        self.assertIsNone(shipped_spine.core.metrics.snapshot().get("cache.hit"))
        self.assertEqual(shipped_spine.provider_calls, shipped_spine.summary.judge_started)


# --------------------------------------------------------------------------------------------
# 6. The bar that was a pin (FIND-REFLEX-01, fixed)
# --------------------------------------------------------------------------------------------


def test_a_cache_served_verdict_is_not_an_actual_judge_call() -> None:
    """The funnel's spend is what reached the PROVIDER, not what reached a verdict.

    Was a strict xfail: `judge.started`/`judge.finished` are emitted for a cache serve too (the tick
    is real — the gate still classifies the served verdict), and nothing marked it, so
    the old derived call count included the runtime's own cache hits as model spend by an error that
    GROWS with the hit rate the cache exists to raise. The serve now carries the shared zero-spend
    marker and the summary publishes the resulting provider-call count directly.
    """
    warm = _drive(SPINE, source_profile=None, cache=True)
    # Ground truth, measured at the provider seam. TheCacheServesTheRepeat pins these separately, so
    # if the cache itself ever breaks that class goes red rather than this bar quietly changing
    # meaning.
    assert warm.provider_calls == 3
    assert warm.core.metrics.snapshot()["cache.hit"] == 3

    summary = warm.summary
    assert summary.provider_calls == warm.provider_calls, (
        f"the funnel reports {summary.provider_calls} provider calls; "
        f"{warm.provider_calls} reached the provider"
    )
    assert summary.provider_calls_unit == "provider_calls"
    assert summary.estimated_saved_calls == N_FRAMES - warm.provider_calls
    # The tick is still IN the funnel — only the spend is subtracted back out. And the cache's own
    # share of that subtraction is legible rather than folded anonymously into "direct".
    assert summary.judge_completed == 6
    assert summary.judge_cache_hits == 3


def test_the_cold_run_marks_nothing_and_its_arithmetic_is_unchanged() -> None:
    """The control: with the cache OFF every verdict cost a call, nothing is marked, and the
    identity is the one it always was. So the subtraction above is a property of the serve, not of
    the counter having been made to read low."""
    cold = _drive(SPINE, source_profile=None, cache=False)
    summary = cold.summary
    assert summary.judge_direct_hits == 0
    assert summary.judge_cache_hits == 0
    assert summary.provider_calls == cold.provider_calls
    assert summary.provider_calls_unit == "provider_calls"


def _explain(result: _Run) -> str:
    """Stage attribution as the failure message, so a red assertion names its stage."""
    return "\n" + result.attribution.describe()


if __name__ == "__main__":  # pragma: no cover - unittest entrypoint
    unittest.main()


# --------------------------------------------------------------------------------------------
# 6b. The THIRD zero-spend path (FIND-REFLEX-06, fixed)
# --------------------------------------------------------------------------------------------


def test_a_locally_completed_fused_bundle_is_not_an_actual_judge_call() -> None:
    """A fused bundle that never reached a provider is not model spend.

    FIND-REFLEX-01 marked two zero-spend paths — a perception-cache serve and a confident
    ``DirectJudge`` substitution. There is a third: an INCOMPLETE fused bundle reaches a verdict
    through :meth:`percept.agent.Percept.complete_fused_locally`, which builds the verdict from the
    bundle's own coverage uncertainty and calls nothing. That path omitted
    ``_ProviderCallCompletion.provider_spent=False`` — the sibling local completion twenty lines
    above it declares it — so it defaulted to ``True`` and every locally-completed bundle inflated
    ``provider_calls`` and entered the provider-latency population as a 0.0 s sample.

    The declaration is what is asserted here, not an inference from empty ``usage`` or a zero
    latency: those are side effects a future completion could legitimately have while really having
    spent a call. ``served_from`` is its own vocabulary member (``fused_local``) rather than reusing
    ``cache``, so the three reasons a tick cost nothing stay tellable apart.
    """
    import percept.agent as _agent
    from percept.tracing import JUDGE_SERVED_FROM, judge_call_was_served

    assert "fused_local" in JUDGE_SERVED_FROM, "the third zero-spend reason needs its own name"

    # The declaration itself: the local completion must say it did not spend.
    src = inspect.getsource(_agent.Percept.complete_fused_locally)
    assert "provider_spent=False" in src, (
        "complete_fused_locally builds a verdict with no provider call, so its completion must "
        "DECLARE provider_spent=False — defaulting to True is what counted it as spend"
    )

    # …and the marker the readers consult must carry that reason, distinctly.
    marker = {"served": True, "served_from": "fused_local"}
    assert judge_call_was_served(marker), "the shared predicate must recognise the third reason"
    assert not judge_call_was_served({}), "a tick that really spent adds no keys"
