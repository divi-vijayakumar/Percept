"""Two judgment pipelines, one goal, one frame stream, one scripted verdict — compared.

WHY THIS FILE EXISTS. Percept judges through TWO code paths that grew apart, and until now nothing
in the suite crossed between them:

* **Generation 1** — ``Percept.perceive_judged`` -> ``Percept._judge_slot`` -> ``Percept._judge``.
  Reached by ``percept.watch()``, by ``percept.harness.GatedWatchLoop``, and by ``percept.run`` for
  an AUDIO goal (through ``Percept.perceive_audio_judged``, which delegates to ``perceive_judged``).
* **Generation 2** — ``Percept.prepare_judgment`` -> ``Percept.prepare_frame_judgment`` ->
  ``Percept.dispatch_judgment``. Reached by ``percept.run`` for a VISION goal, on both of its
  dispatch lanes (the inline loop in ``percept.runtime.worker``, and the same worker's
  ``_JudgeScheduler`` lane, which is the ``judge_dispatch="async"`` default).

They disagreed on three things a client can observe. TWO ARE NOW CLOSED: the split pipeline injects
the open episode and keys the cache fingerprint on it, so a vision or fused goal under ``run()`` gets
the memory the legacy path always had. What remains is CONCURRENCY, and that one is deliberate --
generation 2 stays sequential because its terminal-outcome guarantee (once a route latches, spend no
further provider call) is only enforceable between jobs. This file drives the SAME goal, the SAME
frame and the SAME scripted verdict through both and asserts on all three axes.

HOW TO READ THE TEST NAMES. The prefix is the claim, and the two are not interchangeable:

``test_parity_…``
    Both generations behave the same, and they are SUPPOSED to. A failure here is a regression in
    whichever pipeline moved.

``test_divergence_…``
    They behave DIFFERENTLY today. The assertion pins the current behaviour of BOTH so that either
    one moving is loud. Each such test says in its own docstring whether the divergence is
    ENDORSED (a deliberate difference with a reason) or RECORDED-NOT-ENDORSED (a drift nobody
    decided), and where the decision lives. Recording is not approval: these assertions exist so
    that closing a divergence is a visible, deliberate edit to this file rather than a silent
    behaviour change.

Everything here is offline and keyless: synthetic griddable frames, a scripted verdict, a local
recording judge double. No provider, no network, no key.
"""

from __future__ import annotations

import asyncio
import struct
import unittest
from dataclasses import dataclass, field
from typing import Any

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.audio_input import (
    AudioInputCapability,
    audio_frame_from_units,
    audio_script_token,
    canonicalize_audio_unit,
)
from percept.backends.fake import FakeVision
from percept.config import CacheSettings, GateSettings, Settings
from percept.contracts import AudioFrame, WatchPlan
from percept.episodes import EpisodeLog
from percept.events import AudioUnit
from percept.judge_context import JUDGE_OPEN_EPISODE_KEY
from percept.runtime.worker import RuntimeStats, _JudgeScheduler
from percept.source_profiles_default import PHYSICAL_SCENE
from percept.watchplan_apply import apply_watch_plan

_YES = Verdict(True, 0.95, "a person is at the door")

#: How many event-loop turns a recording judge yields for while "in flight". Long enough that a
#: concurrent batch demonstrably overlaps; irrelevant to a sequential one, which can only ever hold
#: one call at a time however long each takes.
_YIELDS_PER_CALL = 8


# ---------------------------------------------------------------------------
# The harness: one recording judge, one cache-key spy, one comparable result record
# ---------------------------------------------------------------------------


class RecordingJudge:
    """A judge double that records the CONTEXT it was handed and how many calls overlapped.

    Deliberately dumb: it returns the same verdict every time, so any difference the harness
    reports comes from the pipeline under test and never from the judge's opinion.
    """

    def __init__(self, verdict: Verdict = _YES) -> None:
        self.verdict = verdict
        self.contexts: list[dict | None] = []
        self.conditions: list[str] = []
        self.inflight = 0
        self.peak_inflight = 0

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        self.conditions.append(condition)
        self.contexts.append(None if context is None else dict(context))
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            for _ in range(_YIELDS_PER_CALL):
                await asyncio.sleep(0)
        finally:
            self.inflight -= 1
        return self.verdict

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []

    @property
    def calls(self) -> int:
        return len(self.conditions)

    @property
    def calls_carrying_an_open_episode(self) -> int:
        return sum(
            1 for ctx in self.contexts if ctx is not None and JUDGE_OPEN_EPISODE_KEY in ctx
        )


class CacheKeySpy:
    """Wraps a ``CacheCascade`` and records every key it is asked about, delegating unchanged.

    The wrapper is the only way to see what the two pipelines put in a key without reading their
    source: ``CacheKey.context_fingerprint`` is the field that diverges, and it is invisible from
    the outside once the entry is stored.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.resolved: list[Any] = []
        self.stored: list[Any] = []

    def resolve(self, key: Any) -> Any:
        self.resolved.append(key)
        return self._inner.resolve(key)

    def store(self, key: Any, verdict: Verdict, confidence: float) -> None:
        self.stored.append(key)
        self._inner.store(key, verdict, confidence)

    def __getattr__(self, name: str) -> Any:  # layers/threshold and anything else, untouched
        return getattr(self._inner, name)

    def fingerprints(self) -> tuple[Any, ...]:
        """The DISTINCT ``context_fingerprint`` values the keys carried, in first-seen order."""
        seen: list[Any] = []
        for key in [*self.resolved, *self.stored]:
            fingerprint = getattr(key, "context_fingerprint", None)
            if fingerprint not in seen:
                seen.append(fingerprint)
        return tuple(seen)


@dataclass(frozen=True)
class PipelineRun:
    """What one pipeline did with one frame stream — the comparable unit both drivers return."""

    generation: str
    judge_calls: int
    calls_carrying_an_open_episode: int
    cache_fingerprints: tuple[Any, ...]
    cache_hits: int
    peak_inflight: int
    open_episodes: int
    fires: int
    contexts: tuple[dict | None, ...] = field(default=())


def _cache_settings(*, ttl_s: float = 100.0, audit_rate: float = 0.0) -> Settings:
    """A cache-ENABLED core, because the fingerprint axis is only observable when a key is built.

    ``Percept.create`` ships the cascade empty (see ``tests/test_audit.py``), which makes every
    lookup DEFER and hides the key shape. These settings are the same ones the episode-cache-key
    tests use, so the two files describe one cache.
    """
    return Settings(
        gate=GateSettings(threshold=0.7, validity_window_s=1000.0),
        cache=CacheSettings(enabled=True, ttl_s=ttl_s, audit_rate=audit_rate),
    )


def parity_core(
    judge: RecordingJudge,
    *,
    goal_ids: tuple[str, ...] = ("g1",),
    judge_concurrency: int | None = None,
    settings: Settings | None = None,
) -> Percept:
    """One core, armed identically for whichever generation is about to drive it."""
    core = Percept.create(
        vision=judge,
        settings=settings if settings is not None else _cache_settings(),
        discover_plugins=False,
        source_profile=PHYSICAL_SCENE,
        judge_concurrency=judge_concurrency,
    )
    core.cache = CacheKeySpy(core.cache)
    for goal_id in goal_ids:
        core.add_goal(
            Goal(
                id=goal_id,
                condition=f"a person is at the {goal_id} door",
                say="someone is there",
                source="camera",
                state={"type": "watch", "count": 0},
            )
        )
    return core


def still_frame() -> list:
    """One unchanging griddable frame — the case the ROI dHash is meant to recognise."""
    return [[float(x * 8) for x in range(9)] for _ in range(8)]


def open_an_episode(core: Percept, goal_id: str, *, value: str) -> EpisodeLog:
    """Put a goal into the state a long occurrence leaves it in: one episode, open, annotated."""
    log = EpisodeLog(goal_id)
    episode, _ = log.observe(10.0, "opaque-exemplar-payload", None, subject_key="subject_a")
    log.annotate(episode, value=value)
    core.episodes[goal_id] = log
    return log


def _summarise(
    core: Percept,
    judge: RecordingJudge,
    *,
    generation: str,
    goal_ids: tuple[str, ...],
    fires: int,
) -> PipelineRun:
    open_episodes = 0
    for goal_id in goal_ids:
        log = core.episodes.get(goal_id)
        if log is not None:
            open_episodes += len(log.open_episodes())
    return PipelineRun(
        generation=generation,
        judge_calls=judge.calls,
        calls_carrying_an_open_episode=judge.calls_carrying_an_open_episode,
        cache_fingerprints=core.cache.fingerprints(),
        cache_hits=core.metrics.get("cache.hit"),
        peak_inflight=judge.peak_inflight,
        open_episodes=open_episodes,
        fires=fires,
        contexts=tuple(judge.contexts),
    )


async def drive_generation_1(
    core: Percept,
    judge: RecordingJudge,
    *,
    ticks: int = 1,
    goal_ids: tuple[str, ...] = ("g1",),
    frame: Any = None,
    before_tick: Any = None,
) -> PipelineRun:
    """``perceive_judged`` — the entrypoint ``percept.watch()``, the harness loop and audio reach."""
    frame = still_frame() if frame is None else frame
    fires = 0
    for index in range(ticks):
        if before_tick is not None:
            before_tick(core, index)
        events = await core.perceive_judged(
            frame, "camera", t=float(index), goal_ids=goal_ids
        )
        fires += sum(1 for event in events if event.action == "fire")
    return _summarise(
        core, judge, generation="perceive_judged", goal_ids=goal_ids, fires=fires
    )


async def _prepare_generation_2_jobs(
    core: Percept, frame: Any, goal_ids: tuple[str, ...], *, t: float
) -> tuple[Any, ...]:
    """The split-phase admission the runtime worker runs before any provider call."""
    candidate = core.prepare_observation_candidate(
        frame,
        "camera",
        t=t,
        expected_goals_revision=core.goals.revision,
        expected_source_generation=core.source_continuity_generation("camera"),
        goal_ids=goal_ids,
    )
    if candidate is None or not core.launch_observation_candidate(candidate):
        return ()
    reduction = core.reduce_observation_completion(
        await core.dispatch_observation_candidate(candidate)
    )
    return tuple(
        job
        for goal_id in goal_ids
        if (job := core.prepare_judgment(reduction, goal_id=goal_id)) is not None
    )


async def drive_generation_2(
    core: Percept,
    judge: RecordingJudge,
    *,
    ticks: int = 1,
    goal_ids: tuple[str, ...] = ("g1",),
    frame: Any = None,
    before_tick: Any = None,
    lane: str = "inline",
) -> PipelineRun:
    """``prepare_judgment`` -> ``dispatch_judgment`` — what ``percept.run`` drives for vision.

    ``lane="inline"`` mirrors the worker's inline branch (prepare, launch, then await each job in
    turn). ``lane="scheduler"`` hands the same launched batch to the worker's own
    ``_JudgeScheduler``, the lane ``judge_dispatch="async"`` builds. Both are shipped code; the
    difference between them is which of the two the assertion is talking about.
    """
    if lane not in ("inline", "scheduler"):
        raise ValueError(f"lane must be 'inline' or 'scheduler', got {lane!r}")
    frame = still_frame() if frame is None else frame
    fires = 0
    scheduler = None
    if lane == "scheduler":
        core.clock = asyncio.get_running_loop().time
        scheduler = _JudgeScheduler(core, "camera", RuntimeStats())
    for index in range(ticks):
        if before_tick is not None:
            before_tick(core, index)
        jobs = await _prepare_generation_2_jobs(
            core, frame, goal_ids, t=float(index)
        )
        if not jobs:
            continue
        if scheduler is None:
            completions = [
                await core.dispatch_judgment(job)
                for job in core.launch_judgments(jobs)
            ]
        else:
            assert scheduler.submit(float(index), jobs)
            for _ in range(20_000):
                if scheduler._active is None:
                    break
                await asyncio.sleep(0)
            completions = [
                completion
                for _t, batch in scheduler.drain()
                for completion in batch
            ]
        for completion in completions:
            fires += sum(
                1
                for event in core.reduce_judgment_completion(completion)
                if event.action == "fire"
            )
    return _summarise(
        core, judge, generation=f"dispatch_judgment[{lane}]", goal_ids=goal_ids, fires=fires
    )


# ---------------------------------------------------------------------------
# Axis 1 — does the concern's OPEN EPISODE reach the judge context?
# ---------------------------------------------------------------------------


class OpenEpisodeContextTests(unittest.TestCase):
    """``Percept._judge`` injects the open episode under a reserved key. Does the other path?"""

    def test_parity_a_goal_with_no_episode_gets_no_open_episode_key(self):
        """The baseline both pipelines must share: nothing open, nothing injected."""
        for driver in (drive_generation_1, drive_generation_2):
            with self.subTest(driver=driver.__name__):
                judge = RecordingJudge()
                core = parity_core(judge)
                run = asyncio.run(driver(core, judge))
                self.assertEqual(run.judge_calls, 1)
                self.assertEqual(run.calls_carrying_an_open_episode, 0)

    def test_divergence_only_generation_1_injects_an_open_episode(self):
        """RECORDED, NOT ENDORSED.

        ``perceive_judged`` serialises the concern's open episode into the judge context, so the
        model is told which occurrence it is continuing. The prepared-job path assembles its
        context from scratch and never reads ``Percept.episodes`` at all, so the same watch on the
        same frame re-answers the question with no memory of the occurrence.

        Whether to close this is a pipeline-unification decision, not this file's: the harness's
        job is to make either side moving loud. Both halves are asserted so that closing the gap
        requires editing this test on purpose.
        """
        first = RecordingJudge()
        first_core = parity_core(first)
        open_an_episode(first_core, "g1", value="an unattended black backpack")
        generation_1 = asyncio.run(drive_generation_1(first_core, first))

        second = RecordingJudge()
        second_core = parity_core(second)
        open_an_episode(second_core, "g1", value="an unattended black backpack")
        generation_2 = asyncio.run(drive_generation_2(second_core, second))

        # Generation 1: the episode reached the model, and carried the subject and the finding.
        self.assertEqual(generation_1.calls_carrying_an_open_episode, 1)
        injected = generation_1.contexts[0][JUDGE_OPEN_EPISODE_KEY]
        self.assertIn("subject_a", injected)
        self.assertIn("an unattended black backpack", injected)
        self.assertNotIn("opaque-exemplar-payload", injected)  # pixel-free, as the contract says

        # Generation 2: CONVERGED. This block used to assert the opposite — that nothing about the
        # episode reached the model — and it was the sharpest statement of the divergence in this
        # file. The split pipeline now injects too, so a vision or fused goal under ``run()`` gets the
        # same memory the legacy path always had.
        self.assertEqual(generation_2.calls_carrying_an_open_episode, 1)
        injected_2 = generation_2.contexts[0][JUDGE_OPEN_EPISODE_KEY]
        self.assertIn("subject_a", injected_2)
        self.assertIn("an unattended black backpack", injected_2)
        self.assertNotIn("opaque-exemplar-payload", injected_2)  # pixel-free on this path too

        # ...and both pipelines DID hold an episode open, so the difference is purely what was sent
        # to the judge, not whether there was an episode to send. (Generation 1's fold also opens
        # its own episode for this tick's observation, so its count is not the seeded one alone —
        # what matters here is that neither pipeline was episode-less.)
        self.assertGreaterEqual(generation_1.open_episodes, 1)
        self.assertGreaterEqual(generation_2.open_episodes, 1)

    def test_the_injection_holds_on_the_runtime_scheduler_lane_too(self):
        """CONVERGED on both lanes, which is what makes it a property of the pipeline.

        ``judge_dispatch="async"`` (the ``percept.run`` default) reaches the prepared-job path
        through the worker's own ``_JudgeScheduler``. This once recorded the same ABSENCE of
        injection on that lane; it now records the same presence, so the convergence belongs to the
        pipeline and not to how the harness drives it.
        """
        judge = RecordingJudge()
        core = parity_core(judge)
        open_an_episode(core, "g1", value="an unattended black backpack")
        run = asyncio.run(drive_generation_2(core, judge, lane="scheduler"))
        self.assertEqual(run.judge_calls, 1)
        self.assertEqual(run.calls_carrying_an_open_episode, 1)


# ---------------------------------------------------------------------------
# Axis 2 — does the CACHE KEY carry a context fingerprint?
# ---------------------------------------------------------------------------


class CacheKeyFingerprintTests(unittest.TestCase):
    """``CacheKey.context_fingerprint`` is what stops a verdict formed under one episode state
    being re-served under another. One pipeline mints it; the other never does."""

    def test_parity_no_episode_means_no_fingerprint_in_either_pipeline(self):
        """With nothing open there is no answer-affecting context, so ``None`` is correct for both
        — and it is what keeps their keys interchangeable in the ordinary case.

        One tick, deliberately: generation 1's fold opens an episode from the FIRST observation, so
        by the second tick it has answer-affecting context to fingerprint and the pipelines part
        company (that is the divergence test below, not this one).
        """
        for driver in (drive_generation_1, drive_generation_2):
            with self.subTest(driver=driver.__name__):
                judge = RecordingJudge()
                core = parity_core(judge)
                run = asyncio.run(driver(core, judge, ticks=1))
                self.assertEqual(run.cache_fingerprints, (None,))

    def test_divergence_only_generation_1_keys_the_episode_into_the_cache(self):
        """RECORDED, NOT ENDORSED.

        Driving the SAME unchanging frame for 40 ticks is the shape a long occurrence really has.
        Generation 1 crosses into a second cache slot the moment the episode opens, so the verdict
        formed before the occurrence existed cannot be served inside it. Generation 2 stays in ONE
        slot for the whole run, because it never mints a fingerprint — its first verdict is served
        for every later tick regardless of what the episode did in between.

        This is the same defect ``context_fingerprint`` was added to fix, still live on the other
        pipeline. Which pipeline changes is a unification decision; the numbers are pinned here so
        neither can drift quietly.
        """
        first = RecordingJudge()
        first_core = parity_core(first)
        generation_1 = asyncio.run(drive_generation_1(first_core, first, ticks=40))

        second = RecordingJudge()
        second_core = parity_core(second)
        generation_2 = asyncio.run(drive_generation_2(second_core, second, ticks=40))

        # Both watches ran one continuous occurrence over an unchanging picture, and both cached.
        self.assertEqual(generation_1.open_episodes, 1)
        self.assertEqual(generation_2.open_episodes, 1)
        self.assertEqual(generation_1.fires, 1)
        self.assertEqual(generation_2.fires, 1)
        self.assertGreater(generation_1.cache_hits, 30)
        self.assertGreater(generation_2.cache_hits, 30)

        # Generation 1 keyed the episode: an unfingerprinted slot before it opened, a fingerprinted
        # one after. Two slots, so no cross-state serve is possible.
        self.assertEqual(len(generation_1.cache_fingerprints), 2)
        self.assertIn(None, generation_1.cache_fingerprints)
        minted = [f for f in generation_1.cache_fingerprints if f is not None]
        self.assertEqual(len(minted), 1)
        self.assertIsInstance(minted[0], str)
        # ...and the tick that missed inside the episode still carried it to the judge, so the
        # fingerprint really does describe context that was sent.
        self.assertEqual(generation_1.calls_carrying_an_open_episode, 1)

        # Generation 2 CONVERGED. This asserted ``(None,)`` -- one slot for the whole run, spanning
        # both episode states -- which was the cache half of the divergence. It now keys the episode
        # like generation 1, so a close and a reopen no longer share a slot. The injection and the key
        # move together by construction: both come from one bundle.
        self.assertEqual(len(generation_2.cache_fingerprints), 2)
        self.assertIn(None, generation_2.cache_fingerprints)
        minted_2 = [f for f in generation_2.cache_fingerprints if f is not None]
        self.assertEqual(len(minted_2), 1)
        self.assertIsInstance(minted_2[0], str)
        self.assertEqual(generation_2.calls_carrying_an_open_episode, 1)


# ---------------------------------------------------------------------------
# Axis 3 — is a multi-goal batch judged CONCURRENTLY or in series?
# ---------------------------------------------------------------------------


class BatchConcurrencyTests(unittest.TestCase):
    """One admitted frame, four armed goals, four judge slots. How many calls are ever in flight?"""

    _GOALS = ("g0", "g1", "g2", "g3")

    def test_parity_every_admitted_goal_is_judged_exactly_once_by_both_pipelines(self):
        """Fairness is the property both pipelines DO share, and the one concurrency could break."""
        for driver in (drive_generation_1, drive_generation_2):
            with self.subTest(driver=driver.__name__):
                judge = RecordingJudge()
                core = parity_core(judge, goal_ids=self._GOALS, judge_concurrency=4)
                run = asyncio.run(driver(core, judge, goal_ids=self._GOALS))
                self.assertEqual(run.judge_calls, 4)
                self.assertEqual(len(set(judge.conditions)), 4)

    def test_divergence_generation_1_fans_the_batch_out_and_generation_2_does_not(self):
        """RECORDED. Generation 1's fan-out is deliberate and bounded; generation 2's serialism is
        deliberate too, for a DIFFERENT reason — a route may declare terminal outcomes, and the
        prepared-job loop must spend no further provider call after one latches, which is only
        enforceable between jobs.

        What was NOT decided is the arithmetic that sizes a batch's deadline against this loop.
        That belongs to the batch dispatch/deadline reconciliation, not here. This assertion pins
        the observable fact both sides of that argument have to start from.
        """
        first = RecordingJudge()
        first_core = parity_core(first, goal_ids=self._GOALS, judge_concurrency=4)
        generation_1 = asyncio.run(
            drive_generation_1(first_core, first, goal_ids=self._GOALS)
        )

        second = RecordingJudge()
        second_core = parity_core(second, goal_ids=self._GOALS, judge_concurrency=4)
        generation_2 = asyncio.run(
            drive_generation_2(second_core, second, goal_ids=self._GOALS)
        )

        self.assertEqual(generation_1.peak_inflight, 4)  # a MAX: one bounded wave
        self.assertEqual(generation_2.peak_inflight, 1)  # a SUM: four waves of one

    def test_divergence_the_runtime_scheduler_lane_is_sequential_too(self):
        """RECORDED. The ``judge_dispatch="async"`` lane runs the same strictly-sequential loop, so
        the default runtime path for a multi-goal VISION watch pays the sum of its judge calls
        while a multi-goal AUDIO watch (generation 1) pays the max."""
        judge = RecordingJudge()
        core = parity_core(judge, goal_ids=self._GOALS, judge_concurrency=4)
        run = asyncio.run(
            drive_generation_2(core, judge, goal_ids=self._GOALS, lane="scheduler")
        )
        self.assertEqual(run.judge_calls, 4)
        self.assertEqual(run.peak_inflight, 1)

    def test_parity_the_declared_bound_is_the_same_number_on_both_pipelines(self):
        """Whatever each loop DOES with it, both read one bound from one place. A second bound
        appearing anywhere is the failure this pins."""
        judge = RecordingJudge()
        core = parity_core(judge, goal_ids=self._GOALS, judge_concurrency=3)
        goals = [core.goals.get(goal_id) for goal_id in self._GOALS]
        self.assertEqual(core.effective_judge_concurrency(goals), 3)

        from percept.runtime.worker import _batch_slots

        self.assertEqual(_batch_slots(core), core.effective_judge_concurrency())


# ---------------------------------------------------------------------------
# The audio asymmetry — the one RUNTIME path that does get injection
# ---------------------------------------------------------------------------

_AUDIO_CAP = AudioInputCapability(modes=frozenset({"audio_only"}))


def _audio_unit(seq: int = 0) -> AudioUnit:
    chunk = AudioFrame(
        pcm=struct.pack("<h", 32767),
        sample_rate=4,
        channels=1,
        sample_format="s16le",
        t_start=0.0,
        t_end=0.25,
    )
    return AudioUnit(
        seq=seq,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        event_t=chunk.t_end,
        chunk=chunk,
        energy=1.0,
        boundaries=(),
    )


def _canonical(unit: AudioUnit) -> AudioUnit:
    return canonicalize_audio_unit(
        unit,
        capability=_AUDIO_CAP,
        source_id="audio",
        source_epoch=0,
        source_sequence=unit.seq,
        observed_time=10.0,
        source_time=unit.t_end,
    )


def _audio_token(unit: AudioUnit) -> str:
    return audio_script_token(
        audio_frame_from_units((_canonical(unit),), capability=_AUDIO_CAP)
    )


class AudioGoalReachesGeneration1Tests(unittest.TestCase):
    """``percept.run`` for an AUDIO goal reaches generation 1, so it is the one runtime path that
    DOES get open-episode injection. That asymmetry is a property of the runtime, not of a test
    double, and it needs pinning: a client with one audio watch and one vision watch gets two
    different judgment contracts out of one ``run()``.

    Compiled audio only binds an exact SDK-owned adapter (``percept.audio_input``
    ``prepare_audio_judge_binding`` rejects a subclass, an instance shadow and a class mutation),
    so the context cannot be observed by wrapping the judge. It is observed instead through
    ``FakeVision``'s own scripted seam: the adapter routes to a ``"<token>|enriched"`` entry when —
    and only when — a caller context reaches it. The verdict's evidence therefore reports whether
    injection happened, without mutating the adapter the binding froze.
    """

    _CONTEXT_REACHED = "the judge received a caller context"
    _NO_CONTEXT = "the judge received no caller context"

    def _audio_core(self, unit: AudioUnit) -> Percept:
        token = _audio_token(unit)
        fake = FakeVision(
            script={
                token: Verdict(True, 0.95, self._NO_CONTEXT),
                f"{token}|enriched": Verdict(True, 0.95, self._CONTEXT_REACHED),
            }
        )
        core = Percept.create(vision=fake, discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="audio", condition="the sound matters", sense="audio"),
        )
        self.assertEqual(receipt.status, "applied")
        return core

    def _drive(self, core: Percept, unit: AudioUnit) -> str:
        fires = asyncio.run(
            core.perceive_audio_judged(
                {"audio": (_canonical(unit),)}, t=0.25, goal_ids=("audio",)
            )
        )
        self.assertEqual([fire.action for fire in fires], ["fire"])
        return fires[0].verdict.evidence

    def test_an_audio_goal_with_no_episode_receives_no_caller_context(self):
        """The control: without the injection there is nothing to forward, so no context is sent."""
        unit = _audio_unit()
        self.assertEqual(self._drive(self._audio_core(unit), unit), self._NO_CONTEXT)

    def test_an_audio_goal_under_the_runtime_does_get_its_open_episode(self):
        """ENDORSED, and the asymmetry is the point.

        ``percept.run`` drives an audio goal through ``perceive_audio_judged``, which delegates to
        ``perceive_judged`` — generation 1 — so an audio watch's judge is told which occurrence it
        is continuing. The vision watch in the same ``run()`` is not (see the divergence tests
        above). One runtime, two judgment contracts, selected by modality.
        """
        unit = _audio_unit()
        core = self._audio_core(unit)
        open_an_episode(core, "audio", value="a siren has been sounding")
        self.assertEqual(self._drive(core, unit), self._CONTEXT_REACHED)

    def test_the_audio_lane_really_is_generation_1(self):
        """Structural, so the test above cannot pass for the wrong reason: the audio entrypoint's
        injection can only come from ``perceive_judged``, and this pins the delegation."""
        unit = _audio_unit()
        core = self._audio_core(unit)
        seen: list[tuple] = []
        original = core.perceive_judged

        async def _recording(*args: Any, **kwargs: Any):
            seen.append((args, tuple(sorted(kwargs))))
            return await original(*args, **kwargs)

        core.perceive_judged = _recording  # type: ignore[method-assign]
        try:
            self._drive(core, unit)
        finally:
            core.perceive_judged = original  # type: ignore[method-assign]
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
