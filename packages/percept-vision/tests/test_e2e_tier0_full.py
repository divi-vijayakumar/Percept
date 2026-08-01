"""Tier 0 in full: video interest, audio energy, the coast floor and the saccade, COMPOSED.

Tier 0 is not one mechanism. It is four admission routes that share one frame loop and one
partition, and every economic claim percept makes rests on all four behaving together. This file
drives the shipped runtime (:func:`percept.runtime.run`) over synthesized grid frames and
synthesized PCM — nothing is downloaded, nothing is mocked below the public entrypoint — and holds
the whole of it.

What is held here
-----------------
1. **The partition invariant.** ``escalations == signal_admits + fixed_admits + coast_admits +
   audio_wakes``, on a single run where all FOUR buckets are non-zero, AND on the frames where two
   causes bought the same frame — which is the only place the identity can actually break. The
   identity is worthless when three terms are zero, so the non-zero-ness is asserted first.
2. **Localized admission.** A 6x6 event on a 48x36 still background admits under ``block_max``
   (the SHIPPED default statistic) and does not admit under ``mean``. Same frames, same runtime,
   one setting apart — the ``mean`` arm is the control that proves the ``block_max`` arm can fail.
3. **Audio-only wake.** An event with no visual signature at all admits on energy alone, and the
   energy that admitted it is ONE definition end-to-end: the ``audio_wake`` knob is amplitude, the
   detector bar is its square, and the measured chunk energy is mean-square. The wake fixture sits
   deliberately in ``[high**2, high)`` so a runtime that crossed the two unsquared would NOT wake.
4. **The fovea.** A brief-window goal coasts on a blank screen at exactly its declared interval,
   and a motion spike raises the processing rate for exactly ``hold_s`` before reverting.
5. **Deafness is loud.** A gate that never wakes reports ``zero_wake_coverage`` carrying
   ``loudest_fraction_of_threshold`` — the number that separates "my bar is mis-set" from "the room
   was silent". A gate that DOES wake reports nothing.
6. **Economy.** Calls per source-hour against an ungated 1 fps sampler, read off the shared
   :mod:`percept.eval.stage_attribution` funnel rather than re-counted here.

Pins, and what a fixer does with each
-------------------------------------
* ``test_edge_decision_record_reproduces_its_own_decision`` — **FIXED** (was a strict xfail;
  FIND-TIERZERO-01). The per-frame ``edge.decision`` record paired the whole-frame MEAN gauge with
  the ``block_max`` cutoff that actually applied, so on the shipped default an admitted frame read
  ``motion=0.0208``, ``threshold=0.2``, ``escalate=True`` and a reader could not derive the decision
  from the record that exists to explain it. The admitting statistic (``admitting_score`` /
  ``admitting_detector``) and the closest-miss (``loudest_score``) now ride ALONGSIDE the gauges,
  which are deliberately config-stable and were not changed. The same run's live-interest record
  was already correct and is still asserted beside it as the control.
* ``test_judge_calls_counts_every_call_that_reached_a_provider`` — **FIXED** (was a strict xfail;
  FIND-TIERZERO-03). ``judge_calls`` is the numerator of every calls-per-source-hour figure and its
  own docstring says it counts each ``(frame x goal)`` that reached a provider; it was incremented
  once per admitted FRAME. Measured at the provider seam: three goals on one admitted frame invoked
  ``vision.judge()`` three times and were reported as one.
* ``test_the_funnel_is_monotone_from_admitted_to_judged`` — **strict xfail**. ``admitted`` is a
  per-unit-of-evidence count and ``judge_started`` is a per-``(evidence x goal)`` count, so on any
  multi-goal watch the funnel's ``seen >= gated >= judged`` spine inverts and ``FunnelStep.dropped``
  goes negative — the value its own docstring calls a signal that the trace lost records. This is
  the same defect the load lane filed as FIND-LOADSTAGE-02, reached from the tier-0 side; the two
  pins flip together and a fixer should delete both markers in one change.
* ``test_localized_event_arms_the_saccade`` — **FIXED** (was a characterization of the opposite
  behaviour; FIND-TIERZERO-02). The fovea's saccade was armed from the mean gauge and therefore
  could not see the localized events the shipped ``block_max`` gate exists to catch: the localized
  run's fovea cadence was byte-identical to a stream on which nothing ever happened. The saccade now
  reads the max of the gauge and that Watch's own gate score, so it sees what the gate sees.

Two corrections to the integration spec's tier-0-in-full row
------------------------------------------------------------
**"Deafness is loud ... a scene with nothing happening does not [report]."** The runtime does not
make that distinction, deliberately: ``_AcousticGateWitness.due`` fires whenever a configured gate
was fed audio and never woke, and its adjacent comment states the rationale — "a doorbell watch in
an empty house correctly never wakes, and alarming on that would contradict tier-0's whole economic
claim". It is ``loudest_fraction_of_threshold`` — near 1.0 for a mis-set bar, exactly 0.0 for
digital silence — that separates the two readings. Both cases are asserted below, against the
runtime's real contract (FIND-TIERZERO-06).

**"Partition invariant ... already pinned by test_inc2_coast_admission.py::PartitionInvariantTests."**
It is not (FIND-TIERZERO-04). That test asserts a THREE-term identity — ``escalations ==
signal_admits + coast_admits + audio_wakes`` — against a four-bucket partition, and its fixture
supplies no audio source and no fixed-cadence goal, so two of its three terms are structurally
zero. Measured: a mutant that drops EVERY admitted frame into ``fixed_admits`` as well breaks the
real partition outright and leaves that test green. The four-term identity, and the coincident-cause
frames where it can actually break, are held here instead.
"""

from __future__ import annotations

import asyncio
import struct
import unittest
from typing import Any

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.config import EdgeSettings, Settings
from percept.contracts import AudioFrame
from percept.edge.tier0 import Frame, gray_frame, motion_score, salience_stats
from percept.eval.stage_attribution import attribute, stage_funnel
from percept.runtime import (
    InterestSubscription,
    RuntimeStats,
    SourceEnvelope,
    SourceFormat,
    run,
)
from percept.runtime.audio import normalize_audio_unit, pcm_energy
from percept.tracing import (
    STAGE_EDGE_DECISION,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)

# ---------------------------------------------------------------------------
# The synthetic eye — 48x36 is the shipped tier-0 gate raster (sourceprofile.py cites it), and the
# default 8x6 block grid lays exactly 6x6-pixel blocks on it. A 6x6 event aligned to one block is
# therefore the smallest event `block_max` can score at 1.0 while the whole-frame mean reads 1/48th
# of that. Both numbers are asserted, not assumed — see SaliencePartitionTests.
# ---------------------------------------------------------------------------

GATE_W, GATE_H = 48, 36
BLOCK = 6  # 48/8 == 36/6 == 6 px per block on the default grid

STILL = gray_frame(GATE_W, GATE_H, 0)
FLIP = gray_frame(GATE_W, GATE_H, 255)

OPAQUE = SourceFormat(media_type="application/x-percept-opaque")

#: The scripted vision token every frame in this file carries. Tier 0 decides WHETHER a judge runs;
#: what the judge answers belongs to the reflex-spine test, so one unsatisfied verdict is enough.
TOKEN = "frame"
SCRIPT = {TOKEN: Verdict(satisfied=False, confidence=0.9, evidence="nothing has changed")}


def localized(fill: int = 255, *, block_x: int = 1, block_y: int = 1) -> Frame:
    """A still frame with exactly ONE 6x6 block lit — a hand, a cup, a cursor."""
    pixels = [0] * (GATE_W * GATE_H)
    for y in range(block_y * BLOCK, (block_y + 1) * BLOCK):
        for x in range(block_x * BLOCK, (block_x + 1) * BLOCK):
            pixels[y * GATE_W + x] = fill
    return Frame(w=GATE_W, h=GATE_H, pixels=tuple(pixels))


EVENT = localized()


# ---------------------------------------------------------------------------
# The synthetic ear. Amplitude in, PCM out, and the ENERGY is derived by the shipped ingress
# (`normalize_audio_unit` -> `_pcm_frame_energy`) rather than by this file — which is the only way
# an assertion about "one definition end-to-end" can mean anything.
# ---------------------------------------------------------------------------

AUDIO_RATE = 8  # Hz; two samples per chunk => a 0.25 s chunk, the documented mic cadence.
AUDIO_SAMPLES = 2
AUDIO_CHUNK_S = AUDIO_SAMPLES / AUDIO_RATE


def pcm_chunk(amplitude: float, start: float) -> AudioFrame:
    """One constant-amplitude s16le chunk. ``amplitude`` is a peak/RMS value in ``[-1, 1]``."""
    value = max(-32768, min(32767, int(round(amplitude * 32768))))
    raw = struct.pack("<" + "h" * AUDIO_SAMPLES, *([value] * AUDIO_SAMPLES))
    return AudioFrame(
        pcm=raw,
        sample_rate=AUDIO_RATE,
        channels=1,
        sample_format="s16le",
        t_start=start,
        t_end=start + AUDIO_CHUNK_S,
    )


def audio_units(amplitudes: list[float]) -> list[Any]:
    """A contiguous run of chunks, each carrying the energy the SHIPPED ingress derives from its PCM."""
    units = []
    for i, amplitude in enumerate(amplitudes):
        chunk = pcm_chunk(amplitude, start=i * AUDIO_CHUNK_S)
        units.append(normalize_audio_unit((chunk.t_end, chunk, None), seq=i))
    return units


# The amplitude knob `run(audio_wake=...)` takes, and the two fixtures that bracket it.
#
# WAKE_AMPLITUDE is amplitude; the detector's bar is its SQUARE (0.09). ABOVE_ENERGY_AMPLITUDE is
# chosen so its mean-square energy (~0.0999) lands inside [0.09, 0.3) — above the squared bar and
# BELOW the raw amplitude knob. A runtime that compared mean-square energy against the unsquared
# knob would therefore stay asleep on it, which is what makes this fixture a units test and not a
# threshold test.
WAKE_AMPLITUDE = 0.3
WAKE_ENERGY = WAKE_AMPLITUDE**2
ABOVE_ENERGY_AMPLITUDE = 0.31623  # energy ~0.09999: in [0.09, 0.3)
NEAR_MISS_AMPLITUDE = 0.29  # energy 0.0841: real signal that just misses a mis-set bar
QUIET_AMPLITUDE = 0.05  # energy 0.0025


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def build_core(
    *,
    salience_stat: str = "block_max",
    goals: tuple[Goal, ...] = (),
) -> Percept:
    core = Percept.create(vision=FakeVision(script=dict(SCRIPT)), discover_plugins=False)
    core.settings = Settings(edge=EdgeSettings(salience_stat=salience_stat))
    for goal in goals:
        core.add_goal(goal)
    return core


def watch_goal(goal_id: str = "watch") -> Goal:
    """A plain judged Watch — the tuned-signal (VID) route."""
    return Goal(id=goal_id, condition="a cup appears on the desk", say="a cup appeared")


def brief_goal(goal_id: str = "brief", window_s: float = 1.0) -> Goal:
    """A brief-action Watch — compiles the fovea coast + saccade (base 2/window, max 3x base)."""
    return Goal(
        id=goal_id,
        condition="a hand briefly waves",
        say="a hand waved",
        state={"sense": {"brief_window_s": window_s}},
    )


def fixed_goal(goal_id: str = "sweep", interval_s: float = 1.3) -> Goal:
    """A fixed-cadence Watch — the route that is never relabelled as motion salience."""
    return Goal(
        id=goal_id,
        condition="the page changed",
        say="the page changed",
        state={"cadence": {"gate_on": "fixed", "min_judge_interval": interval_s}},
    )


def video(frames: list[Frame], *, fps: float = 10.0, start: float = 0.0) -> list[tuple]:
    """A deterministic ``(t, frame, token)`` stream on the stream's own clock."""
    step = 1.0 / fps
    return [(round(start + i * step, 4), frame, TOKEN) for i, frame in enumerate(frames)]


def envelopes(frames: list[Frame], *, fps: float = 10.0) -> list[SourceEnvelope]:
    """The same stream as typed envelopes — required by the live-interest lifecycle path."""
    step = 1.0 / fps
    return [
        SourceEnvelope(
            source_id="camera",
            modality="video",
            epoch=0,
            sequence=i,
            source_time=round(i * step, 4),
            observed_time=round(100.0 + i * step, 4),
            format=OPAQUE,
            payload=TOKEN,
            signal=frame,
        )
        for i, frame in enumerate(frames)
    ]


class Run:
    """One completed ``percept.runtime.run`` and everything it emitted."""

    def __init__(self, stats, health, incidents, records) -> None:
        self.stats = stats
        self.health = health
        self.incidents = incidents
        self.records = records

    # -- health helpers ----------------------------------------------------
    def kinds(self) -> list[str]:
        return [event.kind for event in self.health]

    def of_kind(self, kind: str) -> list:
        return [event for event in self.health if event.kind == kind]

    def escalations(self) -> list[tuple[float, str]]:
        """``(stream ts, cause)`` for every admitted frame, in order."""
        return [(event.ts, event.data["cause"]) for event in self.of_kind("escalation")]

    def causes_at(self, cause: str) -> list[float]:
        return [ts for ts, this in self.escalations() if this == cause]

    def ended(self) -> dict:
        ended = self.of_kind("runtime_ended")
        assert len(ended) == 1, self.kinds()
        return ended[0].data

    # -- funnel ------------------------------------------------------------
    def funnel(self):
        """The shared stage-attribution funnel — never a count re-derived in this file."""
        return stage_funnel(self.records)

    def attribution(self):
        return attribute(self.records)

    def edge_decisions(self) -> list:
        return [row for row in self.records if row.stage == STAGE_EDGE_DECISION]

    def fovea_timeline(self) -> list[float]:
        """Stream timestamps of every frame the fovea coast processed, from the trace.

        NOT from the ``escalation`` health event: that reports ONE cause per frame, and a frame the
        coast and the motion gate both took is reported as ``motion``. Reconstructing the fovea's
        cadence from the cause labels therefore drops exactly the frames where the two coincide,
        which is where a saccade assertion lives. ``edge.decision`` carries ``coast_admit`` per
        frame and is the authoritative record.
        """
        return [
            float(row.trace_id.split("@")[1])
            for row in self.edge_decisions()
            if row.data["coast_admit"]
        ]


def drive(
    core: Percept,
    source: list,
    *,
    audio_source: list | None = None,
    trace: bool = False,
    **kwargs: Any,
) -> Run:
    """Run the shipped worker to exhaustion and collect stats, health, incidents and trace."""
    stats = RuntimeStats()
    health: list = []
    recorder = None
    if trace:
        recorder = TraceRecorder(
            RunTraceManifest.create(
                source="camera",
                driver="percept.run",
                goals=list(core.goals.active()),
                policy=TracePolicy(level="full"),
            )
        )
        kwargs["on_trace"] = recorder.tracer()
    if audio_source is not None:
        kwargs.setdefault("audio_source", audio_source)
        kwargs.setdefault("audio_settle_s", 0.0)

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=source,
                stats=stats,
                on_health=health.append,
                judge_dispatch="inline",
                **kwargs,
            )
        ]

    incidents = asyncio.run(collect())
    records: list = []
    if recorder is not None:
        recorder.close()
        records = list(recorder.records)
    return Run(stats, health, incidents, records)


def assert_partition(case: unittest.TestCase, stats: RuntimeStats, *, where: str) -> None:
    """The tier-0 admission partition, asserted on EVERY run this file makes.

    ``RuntimeStats`` declares each admitted frame lands in exactly one bucket. Holding it once, on
    one fixture, would prove nothing about the composition — so it is re-checked on every arm.
    """
    case.assertEqual(
        stats.escalations,
        stats.signal_admits + stats.fixed_admits + stats.coast_admits + stats.audio_wakes,
        f"admission partition broken in {where}: escalations={stats.escalations} "
        f"signal={stats.signal_admits} fixed={stats.fixed_admits} "
        f"coast={stats.coast_admits} audio={stats.audio_wakes}",
    )


# ---------------------------------------------------------------------------
# 1. The salience the whole file rests on — measured, not assumed.
# ---------------------------------------------------------------------------


class SaliencePartitionTests(unittest.TestCase):
    """The fixture's own numbers, against the shipped cutoffs. If these move, everything below is
    describing a different scene and would silently stop testing what it claims to."""

    def test_a_single_lit_block_is_invisible_to_the_mean_and_maximal_to_block_max(self) -> None:
        stats = salience_stats(STILL, EVENT)
        edge = EdgeSettings()
        # One 6x6 block of 1728 px: the mean dilutes it to 1/48, block_max keeps it whole.
        self.assertAlmostEqual(stats["mean"], (BLOCK * BLOCK) / (GATE_W * GATE_H))
        self.assertEqual(stats["block_max"], 1.0)
        # ...and that is the whole point: it is under the mean cutoff and over the block_max one.
        self.assertLess(stats["mean"], edge.motion_gate)
        self.assertGreaterEqual(stats["block_max"], edge.motion_gate_block_max)
        # The one-pass diagnostic and the gate's own scorer must agree bit for bit.
        self.assertEqual(stats["mean"], motion_score(STILL, EVENT, stat="mean"))
        self.assertEqual(stats["block_max"], motion_score(STILL, EVENT, stat="block_max"))

    def test_a_whole_frame_change_is_maximal_to_both(self) -> None:
        stats = salience_stats(STILL, FLIP)
        self.assertEqual(stats["mean"], 1.0)
        self.assertEqual(stats["block_max"], 1.0)

    def test_the_shipped_default_statistic_is_the_localizing_one(self) -> None:
        # Everything below is written against the SHIPPED posture, not a test-only one.
        self.assertEqual(EdgeSettings().salience_stat, "block_max")


# ---------------------------------------------------------------------------
# 2. Localized admission — the VID half of tier 0.
# ---------------------------------------------------------------------------

#: Four still frames, then the localized event, then it persists. 10 fps.
LOCALIZED_STREAM = [STILL] * 4 + [EVENT] * 4


class LocalizedAdmissionTests(unittest.TestCase):
    """A 6x6 event on a 48x36 still background admits under ``block_max`` and NOT under ``mean``."""

    def _arm(self, stat: str) -> Run:
        return drive(
            build_core(salience_stat=stat, goals=(watch_goal(),)),
            video(LOCALIZED_STREAM, fps=4.0),
            trace=True,
        )

    def test_block_max_admits_the_localized_event_and_mean_does_not(self) -> None:
        localized_arm = self._arm("block_max")
        mean_arm = self._arm("mean")

        # The whole claim, in two numbers off the same frames.
        self.assertEqual(localized_arm.stats.signal_admits, 1)
        self.assertEqual(mean_arm.stats.escalations, 0)

        # ...and the judge spend follows the admission, which is what makes it a product outcome
        # rather than a gate statistic.
        self.assertEqual(localized_arm.stats.judge_calls, 1)
        self.assertEqual(mean_arm.stats.judge_calls, 0)

        assert_partition(self, localized_arm.stats, where="localized/block_max")
        assert_partition(self, mean_arm.stats, where="localized/mean")

        # The funnel says the same thing through the shared helper, not through a second count.
        self.assertEqual(localized_arm.funnel().counts["admitted"], 1)
        self.assertEqual(mean_arm.funnel().counts["admitted"], 0)
        self.assertEqual(mean_arm.funnel().ratio("admission"), 0.0)

    def test_a_mean_gate_that_admits_nothing_says_so_and_names_the_bar_it_used(self) -> None:
        # Deafness has a video twin, and the `mean` arm above is exactly the scene that trips it:
        # a real event happened and the eye bought nothing. It must not be silent about that.
        mean_arm = self._arm("mean")
        reports = mean_arm.of_kind("zero_admission_coverage")
        self.assertEqual(len(reports), 1, mean_arm.kinds())
        data = reports[0].data
        self.assertEqual(data["admitted"], 0)
        self.assertEqual(data["frames_seen"], len(LOCALIZED_STREAM))
        # The statistic AND the cutoff, so a reader can retune from their own signal.
        self.assertEqual(data["stat"], "mean")
        self.assertEqual(data["threshold"], EdgeSettings().motion_gate)
        # A run that DID admit must stay silent — otherwise the report detects nothing.
        self.assertNotIn("zero_admission_coverage", self._arm("block_max").kinds())


# ---------------------------------------------------------------------------
# 3. CROSS-CUTTING — the admission record must describe the statistic that ADMITTED.
# ---------------------------------------------------------------------------


class AdmissionRecordTests(unittest.TestCase):
    """Two recorders watch the same admitted frame. One names the admitting statistic; one does not.

    This is the concern §3 of the spec calls "the admission record must describe the statistic that
    actually admitted (not the mean gauge)", carried through the live runtime rather than asserted
    on a hand-built decision.
    """

    def _localized_run(self) -> tuple[Run, list]:
        """One run with BOTH the live-interest lifecycle and a full trace attached."""
        core = build_core(goals=(watch_goal(),))
        subscription = InterestSubscription()
        events: list = []
        recorder = TraceRecorder(
            RunTraceManifest.create(
                source="camera",
                driver="percept.run",
                goals=list(core.goals.active()),
                policy=TracePolicy(level="full"),
            )
        )
        stats = RuntimeStats()
        health: list = []

        async def main():
            async def consume():
                async for event in subscription:
                    events.append(event)

            consumer = asyncio.create_task(consume())
            incidents = [
                incident
                async for incident in run(
                    core,
                    source=envelopes(LOCALIZED_STREAM, fps=4.0),
                    stats=stats,
                    on_health=health.append,
                    on_trace=recorder.tracer(),
                    interest_subscription=subscription,
                    interest_session_id="e2e-tier0-full",
                    judge_dispatch="inline",
                )
            ]
            await consumer
            return incidents

        incidents = asyncio.run(main())
        recorder.close()
        return Run(stats, health, incidents, list(recorder.records)), events

    def test_the_live_interest_record_carries_the_block_max_score_not_the_mean_gauge(self) -> None:
        completed, events = self._localized_run()
        self.assertEqual(completed.stats.signal_admits, 1)
        entered = [event for event in events if event.kind == "interest.entered"]
        self.assertEqual(len(entered), 1, [event.kind for event in events])
        record = entered[0]
        # It names the statistic...
        self.assertEqual(record.detector_id, "motion_score.block_max")
        # ...carries the score that statistic produced (1.0), not the mean gauge (~0.0208)...
        self.assertEqual(record.score, 1.0)
        self.assertNotAlmostEqual(record.score, salience_stats(STILL, EVENT)["mean"])
        # ...against the cutoff that statistic is calibrated for.
        self.assertEqual(record.threshold, EdgeSettings().motion_gate_block_max)
        self.assertGreaterEqual(record.score, record.threshold)

    @staticmethod
    def reported_saliences(data: dict) -> dict:
        """Every numeric SALIENCE the record carries — the candidate scores a reader could compare.

        Booleans are excluded explicitly: ``escalate`` is a ``bool``, ``bool`` is a subclass of
        ``int``, and ``True >= 0.2`` — so a naive numeric sweep would find the record 'consistent'
        by reading its own verdict back as a score. That is the shape of hollow gate this pin exists
        to avoid.
        """
        return {
            key: float(value)
            for key, value in data.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key != "threshold"
        }

    def test_edge_decision_record_reproduces_its_own_decision(self) -> None:
        """FIND-TIERZERO-01, fixed (was a strict xfail).

        The record used to pair the whole-frame MEAN gauge with the ``block_max`` cutoff that
        actually applied: an admitted localized frame read ``motion=0.0208, threshold=0.2,
        escalate=True``, so the one record whose purpose is to explain an admission could not, and
        named no field for the statistic that did the admitting. The mean gauge is deliberately
        config-stable and was NOT changed; the admitting statistic now rides alongside it.
        """
        completed, _events = self._localized_run()
        admitted = [row for row in completed.edge_decisions() if row.data["escalate"]]
        self.assertEqual(len(admitted), 1)
        data = admitted[0].data
        # A single-goal run reports exactly one cutoff, so a reader has everything they need.
        scores = self.reported_saliences(data)
        self.assertTrue(
            any(score >= data["threshold"] for score in scores.values()),
            "edge.decision reports no salience that clears the cutoff it reports, on a frame it "
            f"reports as escalated: saliences={scores!r} threshold={data['threshold']!r}",
        )
        # The mean gauge is UNCHANGED — this is an addition, not a substitution, so the gauge stays
        # comparable across configurations exactly as MotionGate.evaluate documents.
        self.assertAlmostEqual(data["motion"], round(salience_stats(STILL, EVENT)["mean"], 4))
        # ...and the reader can now name the knob that governs the decision.
        self.assertEqual(data["admitting_score"], 1.0)
        self.assertEqual(data["admitting_detector"], "motion_score.block_max")
        self.assertGreaterEqual(data["admitting_score"], data["threshold"])

    def test_a_silent_frame_reports_how_close_it_came(self) -> None:
        """The other half of the record's stated purpose — "why DIDN'T it fire". A frame the gate
        refused carries no admitting score (nothing admitted it) and does carry the loudest score
        any armed Watch measured, which is the number that separates a quiet scene from a mis-set
        bar. Without it a silent frame's only salience was a mean gauge on the wrong scale."""
        completed, _events = self._localized_run()
        silent = [row for row in completed.edge_decisions() if not row.data["escalate"]]
        self.assertTrue(silent)
        for row in silent:
            self.assertIsNone(row.data["admitting_score"])
            self.assertIsNone(row.data["admitting_detector"])
            self.assertLess(row.data["loudest_score"], row.data["threshold"])

    def test_the_same_record_is_self_consistent_when_the_gauge_is_the_statistic(self) -> None:
        # The control for the pin above: under `mean` the reported gauge IS the admitting
        # statistic, and the record reproduces its own decision exactly. So the pin is about the
        # pairing, not about tracing being broken in general.
        completed = drive(
            build_core(salience_stat="mean", goals=(watch_goal(),)),
            video([STILL] * 4 + [FLIP] * 4, fps=4.0),
            trace=True,
        )
        admitted = [row for row in completed.edge_decisions() if row.data["escalate"]]
        self.assertEqual(len(admitted), 1)
        data = admitted[0].data
        self.assertGreaterEqual(data["motion"], data["threshold"])
        # ...and the same fix-shape-agnostic reading the pin above uses passes here, which is what
        # proves that reading is not simply always false.
        scores = self.reported_saliences(data)
        self.assertTrue(any(score >= data["threshold"] for score in scores.values()))


# ---------------------------------------------------------------------------
# 4. The audio half — an event with no visual signature at all.
# ---------------------------------------------------------------------------

#: Four seconds of a completely still camera at 10 fps. Nothing here can admit a frame visually.
AUDIO_ONLY_VIDEO = video([STILL] * 40, fps=10.0)


class AudioWakeTests(unittest.TestCase):
    """An audio-invisible event admits on energy alone — and the energy is ONE definition."""

    def _run(self, amplitudes: list[float], **kwargs: Any) -> Run:
        return drive(
            build_core(goals=(watch_goal(),)),
            AUDIO_ONLY_VIDEO,
            audio_source=audio_units(amplitudes),
            audio_wake={"high": WAKE_AMPLITUDE, "low": WAKE_AMPLITUDE / 2},
            **kwargs,
        )

    def test_an_audio_invisible_event_admits_on_energy_alone(self) -> None:
        amplitudes = [QUIET_AMPLITUDE] * 16
        amplitudes[6] = ABOVE_ENERGY_AMPLITUDE  # a chunk ending at t=1.75
        completed = self._run(amplitudes)

        self.assertEqual(completed.stats.audio_wakes, 1)
        # Nothing else could have bought that frame: the camera never moved.
        self.assertEqual(completed.stats.signal_admits, 0)
        self.assertEqual(completed.stats.coast_admits, 0)
        self.assertEqual(completed.stats.fixed_admits, 0)
        self.assertEqual(completed.stats.escalations, 1)
        self.assertEqual(completed.stats.judge_calls, 1)
        assert_partition(self, completed.stats, where="audio-only wake")

        # The admitted frame is the first video frame at or after the chunk, and its escalation is
        # attributed to audio — never relabelled as motion.
        self.assertEqual(completed.causes_at("audio"), [1.8])
        self.assertEqual(completed.ended()["audio_wakes"], 1)

    def test_a_silent_camera_and_a_quiet_room_buy_nothing_at_all(self) -> None:
        completed = self._run([QUIET_AMPLITUDE] * 16)
        self.assertEqual(completed.stats.escalations, 0)
        self.assertEqual(completed.stats.judge_calls, 0)
        assert_partition(self, completed.stats, where="audio-only quiet")

    # -- CROSS-CUTTING: amplitude vs mean-square, never crossed unsquared ---------------

    def test_the_wake_bar_is_the_amplitude_knob_squared_exactly_once(self) -> None:
        # The knob callers type is amplitude; every detector downstream is mean-square energy. The
        # gate's own report is where the conversion becomes checkable from outside the process.
        completed = self._run([QUIET_AMPLITUDE] * 8)
        report = completed.of_kind("zero_wake_coverage")
        self.assertEqual(len(report), 1, completed.kinds())
        data = report[0].data
        self.assertAlmostEqual(data["wake_threshold"], WAKE_ENERGY)
        self.assertAlmostEqual(data["rearm_threshold"], (WAKE_AMPLITUDE / 2) ** 2)
        # ...and the energies it measured are mean-square too, derived by the shipped ingress from
        # the PCM this file synthesized. Both sides of the comparison, one domain.
        expected = pcm_energy(pcm_chunk(QUIET_AMPLITUDE, 0.0).pcm)
        self.assertAlmostEqual(data["energy_max"], expected)
        self.assertAlmostEqual(data["energy_mean"], expected)
        self.assertAlmostEqual(expected, QUIET_AMPLITUDE**2, places=5)

    def test_a_chunk_between_the_squared_bar_and_the_raw_knob_still_wakes(self) -> None:
        # THE units test. This chunk's mean-square energy sits inside [high**2, high): it clears
        # the real (squared) bar and is far under the raw amplitude knob. A runtime that compared
        # energy against the unsquared knob anywhere on this path would stay asleep on it.
        energy = pcm_energy(pcm_chunk(ABOVE_ENERGY_AMPLITUDE, 0.0).pcm)
        self.assertGreaterEqual(energy, WAKE_ENERGY)
        self.assertLess(energy, WAKE_AMPLITUDE)

        amplitudes = [QUIET_AMPLITUDE] * 16
        amplitudes[6] = ABOVE_ENERGY_AMPLITUDE
        self.assertEqual(self._run(amplitudes).stats.audio_wakes, 1)

        # ...and the same fixture one notch quieter, whose energy is under the squared bar, does not.
        below = [QUIET_AMPLITUDE] * 16
        below[6] = NEAR_MISS_AMPLITUDE
        self.assertLess(pcm_energy(pcm_chunk(NEAR_MISS_AMPLITUDE, 0.0).pcm), WAKE_ENERGY)
        self.assertEqual(self._run(below).stats.audio_wakes, 0)


# ---------------------------------------------------------------------------
# 5. The fovea — the coast floor and the saccade.
# ---------------------------------------------------------------------------

BRIEF_WINDOW_S = 1.0
#: What `percept.sensorplan` derives from that prior, and `percept.sampling.SensePolicy` runs on.
COAST_FPS = 2.0 / BRIEF_WINDOW_S  # >= 2 judged frames inside the window
COAST_INTERVAL_S = 1.0 / COAST_FPS
SACCADE_FPS = COAST_FPS * 3.0  # _BRIEF_MAX_HEADROOM
SACCADE_INTERVAL_S = 1.0 / SACCADE_FPS
HOLD_S = 0.5  # SensePolicy default


class FoveaTests(unittest.TestCase):
    """The coast admits on a blank screen at the DECLARED interval; the saccade densifies within
    ``hold_s`` and then reverts. Both read off the run's own escalation timeline."""

    def _blank_then(self, spike: Frame | None, *, at: int = 10, n: int = 26) -> Run:
        frames = [STILL if (spike is None or i < at) else spike for i in range(n)]
        return drive(
            build_core(goals=(brief_goal(window_s=BRIEF_WINDOW_S),)),
            video(frames, fps=10.0),
            trace=True,
        )

    @staticmethod
    def gaps(timeline: list[float]) -> list[float]:
        return [round(b - a, 4) for a, b in zip(timeline, timeline[1:])]

    def test_coast_admits_on_a_blank_screen_at_the_declared_interval(self) -> None:
        completed = self._blank_then(None, n=26)  # 2.5 s of nothing at all, at 10 fps
        coast = completed.fovea_timeline()
        # Every admit is a coast admit — the motion gate never escalated on a still frame.
        self.assertEqual(completed.stats.signal_admits, 0)
        self.assertEqual(completed.stats.coast_admits, len(coast))
        assert_partition(self, completed.stats, where="blank-screen coast")

        # The declared interval, exactly — not "roughly", and not count x span.
        self.assertGreaterEqual(len(coast), 4)
        self.assertEqual(set(self.gaps(coast)), {COAST_INTERVAL_S})
        # ...which is the brief window's Nyquist: >= 2 judged frames per window, as promised.
        self.assertLessEqual(COAST_INTERVAL_S * 2, BRIEF_WINDOW_S + 1e-9)

    def test_a_motion_spike_raises_the_rate_for_hold_s_and_then_reverts(self) -> None:
        completed = self._blank_then(FLIP, at=10, n=26)
        spike_at = next(ts for ts, cause in completed.escalations() if cause == "motion")
        self.assertEqual(spike_at, 1.0)

        coast = completed.fovea_timeline()
        inside = [ts for ts in coast if spike_at < ts <= spike_at + HOLD_S]
        after = [ts for ts in coast if ts > spike_at + HOLD_S]

        # WITHIN hold_s the fovea processes at the escalated rate...
        self.assertEqual(inside, [1.2, 1.4])
        for a, b in zip([spike_at] + inside, inside):
            self.assertLess(round(b - a, 4), COAST_INTERVAL_S)
            self.assertGreaterEqual(round(b - a, 4), SACCADE_INTERVAL_S)
        # ...and reverts to the coast floor once the hold expires.
        self.assertEqual(self.gaps(after), [COAST_INTERVAL_S])
        self.assertEqual(round(after[0] - inside[-1], 4), COAST_INTERVAL_S)

        # The control that proves the assertion above can fail: with NO spike the same stream
        # produces no sub-interval gap anywhere.
        self.assertEqual(set(self.gaps(self._blank_then(None, n=26).fovea_timeline())),
                         {COAST_INTERVAL_S})

    def test_localized_event_arms_the_saccade(self) -> None:
        """FIND-TIERZERO-02, fixed. This was a CHARACTERIZATION of the opposite behaviour.

        The saccade used to be armed from ``decision.signal.motion``, the whole-frame MEAN gauge,
        against a threshold (``_BRIEF_ESCALATE_THRESHOLD`` = 0.05) calibrated on that scale, while
        the SHIPPED gate admits on ``block_max``. So a localized event — precisely the shape a
        brief-action prior exists for, and precisely what ``block_max`` was introduced to catch —
        cleared the gate at 1.0 and left the fovea at its coast floor, because its mean gauge reads
        0.0208. The localized run's fovea cadence was BYTE-IDENTICAL to a stream on which nothing
        ever happened: the coverage the brief prior sells was delivered only for whole-frame motion.

        The saccade now sees what the gate sees (the max of the gauge and this Watch's own gate
        score), so both runs densify.
        """
        localized_run = self._blank_then(EVENT, at=10, n=26)
        flip_run = self._blank_then(FLIP, at=10, n=26)
        blank_run = self._blank_then(None, n=26)

        # Both admit the event through the tuned signal...
        self.assertEqual(localized_run.stats.signal_admits, 1)
        self.assertEqual(flip_run.stats.signal_admits, 1)

        # ...and BOTH now densify the fovea. The localized run is no longer indistinguishable from
        # a stream on which nothing ever happened.
        self.assertNotEqual(localized_run.fovea_timeline(), blank_run.fovea_timeline())
        self.assertIn(
            0.2,  # the 6 fps escalated beat, as a 10 fps capture can express it
            self.gaps(localized_run.fovea_timeline()),
            "the saccade must engage for a localized event — that is what block_max admits",
        )
        self.assertIn(0.2, self.gaps(flip_run.fovea_timeline()))
        self.assertNotEqual(flip_run.fovea_timeline(), blank_run.fovea_timeline())
        # The two events are now indistinguishable TO THE FOVEA, which is the point: a hand waving
        # in one corner buys the same densified look as the whole scene changing.
        self.assertEqual(localized_run.fovea_timeline(), flip_run.fovea_timeline())

        # The localized event is a whole-frame non-event, so this is a property of the SACCADE
        # seeing the gate's statistic, not of the frames having become globally noisy: the mean
        # gauge on the admitting frame is still far under the saccade floor.
        self.assertLess(salience_stats(STILL, EVENT)["mean"], 0.05)

        # The blank control still coasts at exactly its declared interval — nothing here armed a
        # saccade that should not have armed.
        self.assertEqual(set(self.gaps(blank_run.fovea_timeline())), {COAST_INTERVAL_S})


# ---------------------------------------------------------------------------
# 6. Deafness is loud.
# ---------------------------------------------------------------------------


class DeafnessTests(unittest.TestCase):
    """A gate that never wakes must SAY SO, and must say enough to tell the two readings apart.

    SPEC CORRECTION. That row reads "a gate that never wakes emits ``zero_wake_coverage``...;
    a scene with nothing happening does **not**". The runtime deliberately emits it for BOTH — a
    doorbell watch in an empty house never waking is correct behaviour and alarming on it would
    contradict tier-0's economic claim (``percept/runtime/worker.py``, the AUDIO-GATE-DEAF block).
    What separates them is ``loudest_fraction_of_threshold``: near 1.0 means real audio kept just
    missing a mis-set bar; exactly 0.0 means there was nothing to hear. Both are asserted here.
    """

    def _run(self, amplitudes: list[float]) -> Run:
        return drive(
            build_core(goals=(watch_goal(),)),
            AUDIO_ONLY_VIDEO,
            audio_source=audio_units(amplitudes),
            audio_wake={"high": WAKE_AMPLITUDE, "low": WAKE_AMPLITUDE / 2},
        )

    def _report(self, completed: Run) -> dict:
        events = completed.of_kind("zero_wake_coverage")
        self.assertEqual(len(events), 1, completed.kinds())
        return events[0].data

    def test_a_mis_set_bar_reports_deafness_with_a_fraction_that_says_retune(self) -> None:
        # Real, continuous signal that never quite clears the bar — the AUDIO-GATE-DEAF shape.
        data = self._report(self._run([NEAR_MISS_AMPLITUDE] * 8))
        self.assertEqual(data["detector"], "acoustic-onset")
        self.assertEqual(data["wakes"], 0)
        self.assertEqual(data["chunks_observed"], 8)
        self.assertTrue(data["at_end"])
        self.assertGreater(data["energy_max"], 0.0)
        self.assertLess(data["energy_max"], data["wake_threshold"])
        # THE number: the loudest thing heard, as a fraction of the bar it had to clear.
        fraction = data["loudest_fraction_of_threshold"]
        self.assertAlmostEqual(fraction, data["energy_max"] / data["wake_threshold"])
        self.assertGreater(fraction, 0.9, "a near-miss must read as a mis-set bar, not as silence")

    def test_digital_silence_reports_the_same_event_with_a_fraction_of_zero(self) -> None:
        data = self._report(self._run([0.0] * 8))
        self.assertEqual(data["wakes"], 0)
        self.assertEqual(data["energy_max"], 0.0)
        # The runtime DOES report here (contra the spec row) — and the number is what carries the
        # reading. Zero is "there was nothing to hear", not "your gate is broken".
        self.assertEqual(data["loudest_fraction_of_threshold"], 0.0)

    def test_a_single_near_miss_in_a_quiet_room_still_reads_as_a_mis_set_bar(self) -> None:
        """The fraction must come from the LOUDEST chunk, not from the run's average.

        A mis-set bar in the field is almost never a continuous tone: it is a quiet room with one
        doorbell in it. If the fraction were derived from ``energy_mean`` this run would read 0.12 —
        "there was nothing to hear" — and the operator would never retune. Mutation-tested: making
        the witness publish its mean as ``energy_max`` leaves every other assertion in this file
        green and kills only this one.
        """
        amplitudes = [0.02] * 8
        amplitudes[3] = NEAR_MISS_AMPLITUDE  # ONE loud chunk in an otherwise near-silent run
        data = self._report(self._run(amplitudes))

        self.assertEqual(data["wakes"], 0)
        self.assertAlmostEqual(data["energy_max"], NEAR_MISS_AMPLITUDE**2, places=4)
        # The mean is an order of magnitude below the max — that gap is the whole test.
        self.assertLess(data["energy_mean"] * 5, data["energy_max"])
        fraction = data["loudest_fraction_of_threshold"]
        self.assertAlmostEqual(fraction, data["energy_max"] / data["wake_threshold"])
        self.assertGreater(fraction, 0.9, "one near-miss must read as a mis-set bar")
        self.assertLess(
            data["energy_mean"] / data["wake_threshold"],
            0.2,
            "...and it would read as silence if the fraction came from the mean",
        )
        # The rearm counter agrees: exactly one chunk was even loud enough to re-arm the gate.
        self.assertEqual(data["chunks_at_or_above_rearm"], 1)

    def test_the_two_readings_are_separable_from_outside_the_process(self) -> None:
        deaf = self._report(self._run([NEAR_MISS_AMPLITUDE] * 8))
        silent = self._report(self._report_silent())
        # Strict order, and a floor under the gap: "retune me" must never be within reach of
        # "the room was empty", whatever the two absolute numbers happen to be.
        self.assertEqual(silent["loudest_fraction_of_threshold"], 0.0)
        self.assertGreater(
            deaf["loudest_fraction_of_threshold"] - silent["loudest_fraction_of_threshold"],
            0.5,
        )

    def _report_silent(self) -> Run:
        return self._run([0.0] * 8)

    def test_a_gate_that_wakes_never_reports_deafness(self) -> None:
        amplitudes = [QUIET_AMPLITUDE] * 16
        amplitudes[6] = ABOVE_ENERGY_AMPLITUDE
        completed = self._run(amplitudes)
        self.assertEqual(completed.stats.audio_wakes, 1)
        self.assertNotIn("zero_wake_coverage", completed.kinds())

    def test_the_report_lands_before_the_run_ends(self) -> None:
        completed = self._run([NEAR_MISS_AMPLITUDE] * 8)
        kinds = completed.kinds()
        self.assertLess(kinds.index("zero_wake_coverage"), kinds.index("runtime_ended"))


# ---------------------------------------------------------------------------
# 7. All four routes at once — the partition, and the composition.
# ---------------------------------------------------------------------------


def composed_run() -> Run:
    """One stream, one worker, all four admission routes live.

    * ``sweep``   — a fixed 1.3 s cadence               -> ``fixed_admits``
    * ``brief``   — a 1.0 s brief-action prior          -> ``coast_admits`` (+ the saccade)
    * ``watch``   — a plain judged Watch on block_max   -> ``signal_admits``
    * one loud PCM chunk at t=1.75 on a still camera    -> ``audio_wakes``
    """
    frames = [EVENT if i >= 20 else STILL for i in range(40)]  # 4 s at 10 fps, event at t=2.0
    amplitudes = [0.0] * 16
    amplitudes[6] = ABOVE_ENERGY_AMPLITUDE  # a chunk ending at t=1.75
    core = build_core(goals=(watch_goal(), brief_goal(window_s=BRIEF_WINDOW_S), fixed_goal()))
    return drive(
        core,
        video(frames, fps=10.0),
        audio_source=audio_units(amplitudes),
        audio_wake={"high": WAKE_AMPLITUDE, "low": WAKE_AMPLITUDE / 2},
        trace=True,
    )


def coincident_run() -> Run:
    """One frame that TWO admission routes both bought — where a partition can actually break.

    The camera changes at ``t=1.8`` and the loud PCM chunk ends at ``t=1.75``, so the frame at
    ``t=1.8`` is admitted by the motion gate AND by the audio wake. A four-bucket partition that
    counts causes rather than frames double-counts exactly here and nowhere else: a run where every
    admit has a single cause holds the identity no matter how the buckets are written, which is why
    a composed-but-non-coincident fixture is not a test of it.
    """
    frames = [EVENT if i >= 18 else STILL for i in range(40)]
    amplitudes = [0.0] * 16
    amplitudes[6] = ABOVE_ENERGY_AMPLITUDE  # a chunk ending at t=1.75
    return drive(
        build_core(goals=(watch_goal(),)),
        video(frames, fps=10.0),
        audio_source=audio_units(amplitudes),
        audio_wake={"high": WAKE_AMPLITUDE, "low": WAKE_AMPLITUDE / 2},
    )


class CoincidentCauseTests(unittest.TestCase):
    """The partition on frames two routes bought at once — the only frames that can break it.

    SPEC CORRECTION. The row cites ``test_inc2_coast_admission.py::PartitionInvariantTests`` as
    already pinning this. That test asserts a THREE-term identity against a four-bucket partition
    (``fixed_admits`` is absent from the equation) and drives a fixture with no audio source and no
    fixed-cadence goal, so two of its three terms are structurally zero. Mutation-tested: dropping
    every admitted frame into ``fixed_admits`` as well breaks the real partition and leaves it
    green. See FIND-TIERZERO-04.
    """

    def setUp(self) -> None:
        self.run = coincident_run()

    def test_one_frame_bought_twice_is_still_one_escalation_in_exactly_one_bucket(self) -> None:
        stats = self.run.stats
        # Both routes wanted this frame — the health event says so, naming the pair.
        self.assertEqual(self.run.escalations(), [(1.8, "both")])
        # ...and it was bought ONCE. The tuned signal takes the bucket; audio is not also charged.
        self.assertEqual(stats.escalations, 1)
        self.assertEqual(stats.signal_admits, 1)
        self.assertEqual(stats.audio_wakes, 0)
        assert_partition(self, stats, where="audio and motion on one frame")
        # One frame, one judge call — the economy claim reads the frame count, not the cause count.
        self.assertEqual(stats.judge_calls, 1)

    def test_a_coast_tick_under_a_gate_admit_is_charged_once(self) -> None:
        # The composed run's t=2.0 frame is due on the fovea coast AND clears the motion gate; its
        # t=0.0 frame is due on the coast AND on the fixed cadence. Neither may be charged to the
        # coast bucket as well, or the identity over-counts.
        composed = composed_run()
        rows = composed.edge_decisions()
        due = [row for row in rows if row.data["coast_admit"]]
        shared = [row for row in due if row.data["escalate"]]
        alone = [row for row in due if not row.data["escalate"]]
        self.assertGreaterEqual(len(shared), 2, "the fixture must contain real coincidences")
        self.assertIn((2.0, "motion"), composed.escalations())
        self.assertIn(2.0, composed.fovea_timeline(), "the fovea was due at t=2.0")
        # The coast is charged for the ticks it bought ALONE, never for the ones it shared.
        self.assertEqual(composed.stats.coast_admits, len(alone))
        self.assertLess(composed.stats.coast_admits, len(due))
        assert_partition(self, composed.stats, where="coast under a gate admit")


class CompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = composed_run()

    def test_all_four_admission_routes_fire_on_one_stream(self) -> None:
        stats = self.run.stats
        # The identity is worthless if three terms are zero, so this comes FIRST.
        self.assertGreater(stats.signal_admits, 0)
        self.assertGreater(stats.fixed_admits, 0)
        self.assertGreater(stats.coast_admits, 0)
        self.assertGreater(stats.audio_wakes, 0)
        # ...and every cause is named in the timeline, never collapsed into "motion".
        self.assertEqual(
            {cause for _ts, cause in self.run.escalations()},
            {"motion", "fixed", "coast", "audio"},
        )

    def test_the_admission_partition_holds_across_all_four(self) -> None:
        assert_partition(self, self.run.stats, where="composed run")

    def test_the_coverage_summary_carries_the_same_partition(self) -> None:
        # The stats object is in-process; the health payload is what a deployment actually sees.
        # They must agree, or an operator's dashboard is auditing a different run.
        data = self.run.ended()
        self.assertEqual(
            data["admitted"],
            data["signal_admits"] + data["fixed_admits"] + data["coast_admits"] + data["audio_wakes"],
        )
        for field, value in (
            ("signal_admits", self.run.stats.signal_admits),
            ("fixed_admits", self.run.stats.fixed_admits),
            ("coast_admits", self.run.stats.coast_admits),
            ("audio_wakes", self.run.stats.audio_wakes),
            ("admitted", self.run.stats.escalations),
        ):
            self.assertEqual(data[field], value, field)

    def test_the_funnel_agrees_with_the_runtime_counters(self) -> None:
        funnel = self.run.funnel()
        self.assertEqual(funnel.counts["observed"], self.run.stats.frames)
        self.assertEqual(funnel.counts["admitted"], self.run.stats.escalations)
        # The admission step — the one the economy row reads — is in one unit at both ends.
        self.assertLessEqual(funnel.counts["admitted"], funnel.counts["observed"])
        self.assertEqual(funnel.step("admission").dropped, 40 - self.run.stats.escalations)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "PIN (defect, measured): the funnel's `seen >= gated >= judged >= fired` spine inverts "
            "on any multi-goal watch. `admitted` counts UNITS OF EVIDENCE (source.frame, "
            "deduplicated by trace_id) while `judge_started` counts (evidence x goal), so three "
            "goals admitted on one frame read admitted=1 -> judge_started=3, a dispatch ratio of "
            "300%, and FunnelStep.dropped = -2 — the value that field's own docstring calls "
            "something 'a well-formed trace cannot do' and 'a real signal' of record loss. Every "
            "bar written on the dispatch step therefore misreads a normal multi-goal run. Fix by "
            "making the two points share a unit (or by declaring the unit on each point); do NOT "
            "clamp `dropped`."
        ),
    )
    def test_the_funnel_is_monotone_from_admitted_to_judged(self) -> None:
        funnel = self.run.funnel()
        self.assertGreaterEqual(
            funnel.counts["admitted"],
            funnel.counts["judge_started"],
            f"funnel inverted: {funnel.describe()}",
        )

    def test_the_composed_run_is_deterministic(self) -> None:
        # Two independent pumps (video, audio) share one worker. If the interleaving were not
        # deterministic, every number above would be a coin flip and this whole file would be noise.
        again = composed_run()
        self.assertEqual(again.escalations(), self.run.escalations())


# ---------------------------------------------------------------------------
# 8. CROSS-CUTTING — identity of measurement: a count is invariant to frame rate.
# ---------------------------------------------------------------------------


class FrameRateInvarianceTests(unittest.TestCase):
    """The same four seconds of source, captured four ways, buys the same number of judgments.

    Section 3 of the integration spec makes this a property of every stage, not a stage of its own,
    and tier 0 is where it bites hardest: ``escalations`` is the numerator of the economy claim and
    the denominator of the recall one, so a count that scales with capture rate would make both
    figures a property of the camera rather than of the scene.
    """

    RATES = (2.0, 4.0, 10.0, 25.0)

    @staticmethod
    def four_seconds(fps: float) -> list[Frame]:
        """Still for 2 s, then the localized event for 2 s — the same SOURCE at every rate."""
        n = int(4 * fps)
        return [EVENT if i >= n // 2 else STILL for i in range(n)]

    def test_one_event_costs_one_judgment_at_every_capture_rate(self) -> None:
        seen = {}
        for fps in self.RATES:
            completed = drive(
                build_core(goals=(watch_goal(),)),
                video(self.four_seconds(fps), fps=fps),
            )
            assert_partition(self, completed.stats, where=f"invariance/{fps}fps")
            seen[fps] = (
                completed.stats.frames,
                completed.stats.signal_admits,
                completed.stats.judge_calls,
                completed.causes_at("motion"),
            )
        # Frames scale with the rate — that is the control that proves the streams really differ.
        self.assertEqual([seen[fps][0] for fps in self.RATES], [8, 16, 40, 100])
        # The admissions do not, and they land at the same source instant.
        for fps in self.RATES:
            self.assertEqual(seen[fps][1:], (1, 1, [2.0]), f"{fps} fps: {seen[fps]!r}")

    def test_one_audio_onset_costs_one_judgment_at_every_capture_rate(self) -> None:
        # The ear's cadence is fixed; only the video capture rate moves. A wake must still be a
        # one-shot — the admitted frame is whichever arrives first after the onset, never all of them.
        amplitudes = [QUIET_AMPLITUDE] * 16
        amplitudes[6] = ABOVE_ENERGY_AMPLITUDE
        for fps in self.RATES:
            completed = drive(
                build_core(goals=(watch_goal(),)),
                video([STILL] * int(4 * fps), fps=fps),
                audio_source=audio_units(amplitudes),
                audio_wake={"high": WAKE_AMPLITUDE, "low": WAKE_AMPLITUDE / 2},
            )
            self.assertEqual(completed.stats.audio_wakes, 1, f"{fps} fps")
            self.assertEqual(completed.stats.escalations, 1, f"{fps} fps")
            assert_partition(self, completed.stats, where=f"invariance/audio/{fps}fps")
            # The admitted frame is at or after the onset (chunk 6 ends at t=1.75), within one frame.
            (at,) = completed.causes_at("audio")
            self.assertGreaterEqual(at, 1.75)
            self.assertLess(at - 1.75, 1.0 / fps + 1e-9)


# ---------------------------------------------------------------------------
# 9. Economy — against an ungated 1 fps sampler.
# ---------------------------------------------------------------------------

#: The ungated baseline the economy row names: one frame a second, judged unconditionally.
UNGATED_FPS = 1.0
CALLS_PER_HOUR_UNGATED = UNGATED_FPS * 3600.0


def calls_per_source_hour(judge_calls: int, span_s: float) -> float:
    return judge_calls / span_s * 3600.0


class EconomyTests(unittest.TestCase):
    """Calls per source-hour, and the configuration in which the claim does NOT hold.

    Reported, not just barred: the absolute figure is a gate INPUT (see the spec's §6 boundary),
    and the only thing this suite can independently justify is the comparison against the ungated
    sampler on the same stream.
    """

    #: 60 s of a still camera at 4 fps with one localized event — the still-scene economics tier 0
    #: exists for.
    STILL_SCENE = [EVENT if 120 <= i < 124 else STILL for i in range(240)]

    def test_a_motion_gated_watch_is_far_cheaper_than_an_ungated_sampler(self) -> None:
        completed = drive(
            build_core(goals=(watch_goal(),)),
            video(self.STILL_SCENE, fps=4.0),
            trace=True,
        )
        span = 239 / 4.0
        gated = calls_per_source_hour(completed.stats.judge_calls, span)

        self.assertLess(gated, CALLS_PER_HOUR_UNGATED)
        # Two escalations (the event arriving, and leaving) out of 240 frames.
        self.assertEqual(completed.stats.judge_calls, 2)
        self.assertLess(gated, CALLS_PER_HOUR_UNGATED / 10.0)

        # ...and the same fact off the shared funnel, as the fraction of observed units that never
        # reached a model.
        funnel = completed.funnel()
        self.assertEqual(funnel.counts["observed"], 240)
        self.assertGreater(funnel.admission_economy, 0.98)
        assert_partition(self, completed.stats, where="economy/still scene")

    def test_the_baseline_is_a_real_comparison_and_not_a_tautology(self) -> None:
        # The control: a goal on a fixed cadence FASTER than the baseline sampler spends MORE than
        # it. So the bar above is a property of the gate, not of the arithmetic.
        completed = drive(
            build_core(goals=(fixed_goal(interval_s=0.25),)),
            video(self.STILL_SCENE, fps=4.0),
        )
        span = 239 / 4.0
        self.assertGreater(
            calls_per_source_hour(completed.stats.judge_calls, span), CALLS_PER_HOUR_UNGATED
        )

    def test_judge_calls_counts_every_call_that_reached_a_provider(self) -> None:
        """FIND-TIERZERO-03, fixed (was a strict xfail).

        ``RuntimeStats.judge_calls`` — the numerator of every calls-per-source-hour figure, and what
        the ``runtime_ended`` health payload publishes — documents itself as per ``(frame x goal)``
        that reached a provider. It was incremented once per admitted FRAME, so three goals on one
        admitted frame invoked ``vision.judge()`` three times and were reported as ONE: a deployment
        watching N goals under-reported its model spend, and its economy gate read green, by up to
        N times. ``perceive_dispatches`` and the trace's ``judge.started`` rows already carried the
        right count, which is what made the disagreement measurable.
        """
        class CountingVision(FakeVision):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.invocations = 0

            def judge(self, *args: Any, **kwargs: Any):
                self.invocations += 1
                return super().judge(*args, **kwargs)

        vision = CountingVision(script=dict(SCRIPT))
        core = Percept.create(vision=vision, discover_plugins=False)
        core.settings = Settings(edge=EdgeSettings(salience_stat="mean"))
        for goal_id in ("a", "b", "c"):
            core.add_goal(Goal(id=goal_id, condition=f"{goal_id} happens", say=goal_id))
        completed = drive(core, video([STILL, STILL, FLIP, FLIP], fps=2.0))

        self.assertEqual(completed.stats.escalations, 1)
        self.assertEqual(vision.invocations, 3)  # what the provider was actually asked
        self.assertEqual(
            completed.stats.judge_calls,
            vision.invocations,
            "judge_calls is the numerator of calls/source-hour and must count provider calls",
        )

    def test_a_brief_window_coast_costs_more_than_the_ungated_baseline(self) -> None:
        """MEASURED, and it is a real result rather than a failure.

        A 1.0 s brief-action prior compiles a 2 fps coast floor. That floor is DENSER than the 1 fps
        sampler the economy row compares against, so on this configuration tier 0 is more expensive
        than its own baseline — before any event happens. The economy claim is a property of the
        admission posture, not of tier 0 as such, and the spec row states no floor that could
        distinguish them. See FIND-TIERZERO-05.
        """
        completed = drive(
            build_core(goals=(brief_goal(window_s=BRIEF_WINDOW_S),)),
            video([STILL] * 240, fps=4.0),
        )
        span = 239 / 4.0
        coasting = calls_per_source_hour(completed.stats.judge_calls, span)
        self.assertEqual(completed.stats.signal_admits, 0)
        self.assertGreater(completed.stats.coast_admits, 0)
        self.assertGreater(coasting, CALLS_PER_HOUR_UNGATED)
        assert_partition(self, completed.stats, where="economy/brief coast")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
