"""Restart mid-flight — does one occurrence stay one occurrence across a kill and a restore?

This is an end-to-end capability file, not a mechanism file: every assertion drives the real
``percept.runtime.run`` worker over a real source, with a real persistence store, and kills the
process at a material transition named in the product specification's mandatory-fault list (worker
kill during a state/outbox transaction phase; power loss after accepted non-firing inputs; a kill
mid-dwell; a kill inside a refractory window). Nothing here needs a download: the frames are
``percept.harness.tier0.gray_frame`` and the verdicts are the scripted offline backend, so the whole
file is deterministic.

The four transitions, and what each one is supposed to prove:

===============================  =============================================================
kill point                        the continuity property it tests
===============================  =============================================================
after a fire has been delivered   the same occurrence must not be reported a second time
inside a refractory window        the refractory memory must still suppress an early re-fire
mid-dwell                         the dwell must restart honestly, crediting no pre-restart time
with episode memory in use        the amnesia must be LOUD, and recovery inside the objective
===============================  =============================================================

**What holds.** The persistence half is sound. ``Percept.snapshot`` carries the gate's armed bit,
its last-fire stamp, the entity graph, the goals and the action log; ``Percept.restore`` puts them
back; and the dwell timer is dropped on purpose, by two independent belts, so a half-completed dwell
never fires early after a restart. Recovery to the first correct post-restart decision is inside the
30 s objective, and the measurement moves when the scenario does.

**What the fix for findings restart-01/02/03 established.** A restart still resets episode memory
(session-only, ADR-006) and the first post-restart observation still folds as a NEW episode. What no
longer follows is the re-arm: ``Gate.rearm`` used to set ``armed = True`` **and** clear
``last_fire``, so ``BehaviorExecutor.fire_judged`` honouring that fold destroyed the restored
disarmed latch and the restored refractory stamp before the gate was ever consulted, and one
continuing occurrence was reported twice on the same entity id. The invariant now held in the gate
is that a re-arm asserts an edge may fire again and never rewrites the record of what already fired:
``last_fire`` survives every re-arm, and a key read back from a snapshot refuses to be re-armed
while its edge is spent, until a decision in THIS session gives the re-arm something to stand on.
The bars below are ordinary assertions again.

**What the fix for finding restart-04 established.** The amnesia is now announced for this goal
shape too (``TheEpisodeResetIsAnnounced``). ``Percept.restore`` queued ``episode_memory_reset`` only
for goals declaring ``type='count'`` or ``rearm='each_event'``, so the one shape whose lost episode
memory changes what a subscriber receives — a plain watch with a bound subject — restored in total
silence. The predicate now also covers any concern that SELECTS an identity policy, which is the
same resolution the subject-bound episode folds themselves gate on. It stays CONDITIONAL: a concern
that declared nothing could never have built an episode and still says nothing.

Every bar here is paired with the passing control that isolates it — the identical stream run
without the restart, or the identical restart run without the episode subject — so a failure is
attributed to the restart rather than merely observed near one.
"""

from __future__ import annotations

import asyncio
import unittest
import warnings
from dataclasses import replace

from percept import Goal, Percept
from percept.adapters import Box, Observation, Verdict
from percept.backends.fake import FakeVision
from percept.behavior_routes import UndrivenBehaviorRouteWarning
from percept.config import Settings
from percept.eval.stage_attribution import attribute, stage_funnel
from percept.gate import FIRE, SILENT, Gate
from percept.harness.tier0 import gray_frame
from percept.persistence import InMemoryPersistence
from percept.runtime import run
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

#: The qualification targets this file measures against (product specification, operational service
#: levels). They are targets, not claims about the code — which is exactly why the failing bars below
#: are recorded rather than relaxed.
RECOVERY_OBJECTIVE_S = 30.0

YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted: the door is open")
NO = Verdict(satisfied=False, confidence=0.95, evidence="scripted: the door is shut")
STILL = gray_frame(8, 8, 0)

#: A fixed-cadence sense, so a motionless scene still admits on a declared interval. Without it a
#: still gray frame never reaches a judge and there is no occurrence to be continuous about.
FIXED_5 = {"min_judge_interval": 5.0, "gate_on": "fixed"}

#: One scripted detection, so the entity graph has a track to carry an id across the restart.
DETECTIONS = {"open": [Observation(label="door", box=Box(0.1, 0.1, 0.4, 0.4), confidence=0.9)]}


def _core(*, refractory_s: float = 0.0, detect: dict | None = None) -> Percept:
    settings = replace(Settings(), gate=replace(Settings().gate, refractory_s=refractory_s))
    return Percept.create(
        vision=FakeVision(script={"open": YES, "shut": NO}, detect_script=detect),
        settings=settings,
        discover_plugins=False,
    )


def _goal(**state) -> Goal:
    """A door watch. ``state`` overrides let one recipe cover every scoping this file needs."""
    base = {"type": "watch", "cadence": dict(FIXED_5)}
    base.update(state)
    return Goal(id="door", condition="the door is open", say="the door was left open", state=base)


def _stream(*seconds: float, token: str = "open") -> list[tuple]:
    """``(t, frame, judge_token)`` source items on the stream timeline."""
    return [(float(t), STILL, token) for t in seconds]


class _Session:
    """One process lifetime: the incidents it delivered, the health it published, its trace."""

    def __init__(self, incidents, health, recorder) -> None:
        self.incidents = list(incidents)
        self.health = list(health)
        self.recorder = recorder

    @property
    def fire_times(self) -> list[float]:
        return [inc.ts for inc in self.incidents]

    @property
    def health_kinds(self) -> list[str]:
        return [ev.kind for ev in self.health]

    def attribution(self):
        """Funnel + per-segment latency, from the shared helper — never parsed here by hand."""
        return attribute(self.recorder.records, manifest=self.recorder.manifest)


async def _session(
    core: Percept,
    source,
    *,
    store=None,
    restore: bool = False,
    sink=None,
    run_id: str = "run",
    **kw,
) -> _Session:
    """Drive one process lifetime to completion and return everything it emitted."""
    recorder = TraceRecorder(
        RunTraceManifest.create(run_id=run_id, driver="run", policy=TracePolicy(level="full"))
    )
    health: list = []
    health_to_trace = recorder.health_sink()

    def on_health(event):
        health.append(event)
        health_to_trace(event)

    incidents = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndrivenBehaviorRouteWarning)
        async for incident in run(
            core,
            source=source,
            persistence=store,
            restore=restore,
            on_incident=sink,
            on_health=on_health,
            on_trace=recorder.tracer(),
            judge_dispatch="inline",
            **kw,
        ):
            incidents.append(incident)
    recorder.close()
    return _Session(incidents, health, recorder)


def _drive(*args, **kw) -> _Session:
    return asyncio.run(_session(*args, **kw))


# ---------------------------------------------------------------------------
# 1. Kill after a delivered fire: one occurrence must stay one incident.
# ---------------------------------------------------------------------------


class OneOccurrenceIsOneIncidentAcrossARestart(unittest.TestCase):
    """The headline bar. A door opens and stays open. The worker is killed after the incident has
    been delivered and committed, and restarted against the continuation of the same stream.

    The occurrence never ended — the condition was confidently true on both sides of the kill, on
    the same subject, at the same cadence. The only difference between the control and the restart
    scenario is the process boundary.
    """

    #: Four observations of one continuing occurrence. The restart splits it after the second.
    STREAM = _stream(0, 5, 10, 15)
    KILL_AFTER = 2

    def test_the_uninterrupted_stream_reports_the_occurrence_exactly_once(self) -> None:
        """The control. Without a restart the episode fold suppresses every re-statement, so the
        gate is not even consulted after the first fire — this is the behaviour the restart must
        reproduce, and the reason the duplicate below is attributable to the restart alone."""
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        session = _drive(core, self.STREAM)

        self.assertEqual(session.fire_times, [0.0])
        funnel = session.attribution().funnel
        self.assertEqual(funnel.counts["observed"], 4, "every frame reached the always-on decision")
        self.assertEqual(funnel.counts["judge_completed"], 4, "all four were judged")
        self.assertEqual(funnel.counts["fired"], 1, "…and exactly one of them became an event")

    def test_the_snapshot_records_the_edge_as_spent_and_restore_puts_it_back(self) -> None:
        """The persistence half is sound, and this is the assertion that says so. It matters
        because it localises the failure below: the durable state is correct when it is read back,
        so nothing that follows can be blamed on the snapshot."""
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        first = _drive(core, self.STREAM[: self.KILL_AFTER], store=store, run_id="before")
        self.assertEqual(first.fire_times, [0.0])

        record = store.load("percept")["gate"]["keys"][0]
        self.assertIs(record["armed"], False, "the snapshot says this edge has already fired")
        self.assertEqual(record["last_fire"], 0.0, "…and remembers when")

        restored = _core()
        restored.add_goal(_goal(identity_policy="scene"))
        restored.restore(store.load("percept"))
        key = tuple(record["key"])
        self.assertIs(restored.gate._keys[key].armed, False, "restore installs the spent edge")
        self.assertEqual(restored.gate._keys[key].last_fire, 0.0)

    def test_the_restored_edge_and_the_reset_episode_map_disagree_after_a_restore(self) -> None:
        """The diagnosis, stated as structure rather than as an outcome. After a restore the gate
        says "this occurrence has already been reported" and the episode map says "I have never
        seen anything". Two durable-state stores describing the same occurrence, restored to
        contradictory answers — and the episode map is the one the fire path consults first."""
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        _drive(core, self.STREAM[: self.KILL_AFTER], store=store)

        restored = _core()
        restored.add_goal(_goal(identity_policy="scene"))
        restored.restore(store.load("percept"))
        self.assertEqual(
            [state.armed for state in restored.gate._keys.values()], [False],
            "the gate remembers the fire",
        )
        self.assertEqual(restored.episodes, {}, "…and the episode map does not (session-only)")

    def test_the_same_stream_split_by_a_restart_still_reports_it_once(self) -> None:
        """THE BAR (finding restart-01). The first post-restart observation still folds as a new
        episode against the emptied episode map, and ``BehaviorExecutor.fire_judged`` still honours
        ``should_rearm`` by calling ``Gate.rearm`` — but the gate now refuses to re-arm an edge it
        read back from a snapshot and has not yet decided on, so the spent edge ``restore`` installed
        survives to answer the observation and the subscriber is told about the door once.
        """
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(_goal(identity_policy="scene"))
        first = _drive(before, self.STREAM[: self.KILL_AFTER], store=store, run_id="before")

        after = _core()
        after.add_goal(_goal(identity_policy="scene"))
        second = _drive(
            after, self.STREAM[self.KILL_AFTER:], store=store, restore=True, run_id="after"
        )

        combined = stage_funnel(tuple(first.recorder.records) + tuple(second.recorder.records))
        self.assertEqual(
            combined.counts["fired"], 1,
            f"one occurrence, one event — got {first.fire_times + second.fire_times}",
        )

    def test_an_entity_scoped_watch_does_not_re_report_its_tracked_subject(self) -> None:
        """The same bar with an identity the subscriber can read. The entity graph IS restored, so
        the duplicate this used to produce carried the SAME entity id — not two subjects being
        confused, one subject being announced twice. One subject, one announcement."""
        store = InMemoryPersistence()
        before = _core(detect=DETECTIONS)
        before.add_goal(_goal(identity_policy="track_iou_label"))
        first = _drive(before, self.STREAM[: self.KILL_AFTER], store=store)

        after = _core(detect=DETECTIONS)
        after.add_goal(_goal(identity_policy="track_iou_label"))
        second = _drive(after, self.STREAM[self.KILL_AFTER:], store=store, restore=True)

        subjects = [inc.entity_id for inc in first.incidents + second.incidents]
        self.assertEqual(len(set(subjects)), 1, "one tracked subject throughout")
        self.assertEqual(
            len(first.incidents + second.incidents), 1,
            f"…and one event for it — got {len(subjects)} on entity {subjects[0]!r}",
        )

    def test_a_second_occurrence_in_one_session_still_gets_its_own_incident(self) -> None:
        """The mutation that stops the two bars above from reading as "re-arming was switched off".
        No restart, identical scoping, and a PROVEN boundary in the middle — the door is confidently
        shut at t=5 — so the re-open at t=10 is a second occurrence and is reported as one."""
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        session = _drive(core, [(0.0, STILL, "open"), (5.0, STILL, "shut"), (10.0, STILL, "open")])
        self.assertEqual(session.fire_times, [0.0, 10.0])

    def test_the_restored_edge_re_opens_once_this_session_watches_it_go_false(self) -> None:
        """...and the same mutation ACROSS the restart, which is what makes the refusal evidence-
        based rather than a permanent latch. The restarted session observes the door shut at t=10 —
        its own proof that the reported occurrence ended, which no amount of missing episode memory
        can manufacture — and the re-open at t=15 fires normally."""
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(_goal(identity_policy="scene"))
        first = _drive(before, self.STREAM[: self.KILL_AFTER], store=store, run_id="before")
        self.assertEqual(first.fire_times, [0.0])

        after = _core()
        after.add_goal(_goal(identity_policy="scene"))
        second = _drive(
            after,
            [(10.0, STILL, "shut"), (15.0, STILL, "open")],
            store=store, restore=True, run_id="after",
        )
        self.assertEqual(second.fire_times, [15.0], "an observed boundary lifts the refusal")


# ---------------------------------------------------------------------------
# 2. Kill inside a refractory window: the second suppression belt.
# ---------------------------------------------------------------------------


class RefractoryMemoryAcrossARestart(unittest.TestCase):
    """``Gate.decide``'s refractory suppresses a fire landing within ``refractory_s`` of this key's
    last real fire, and its docstring is explicit that this must hold even across a confident-false
    blip that re-armed the edge mid-occurrence. The gate snapshot carries ``last_fire`` for exactly
    that reason ("a restart mid-refractory-window would silently re-open it to fire early").

    The stamp survives the round trip, and (findings restart-02/03) it now survives the first
    observation after it: ``Gate.rearm`` no longer clears ``last_fire``, because an episode boundary
    a caller inferred from a confident-false blip is precisely the case the refractory was bought to
    absorb. Only a real fire moves the stamp.
    """

    REFRACTORY_S = 3600.0

    def test_the_last_fire_stamp_survives_the_snapshot_round_trip(self) -> None:
        store = InMemoryPersistence()
        core = _core(refractory_s=self.REFRACTORY_S)
        core.add_goal(_goal(identity_policy="scene"))
        _drive(core, _stream(0, 5), store=store)

        gate = Gate(refractory_s=self.REFRACTORY_S)
        gate.restore(store.load("percept")["gate"])
        self.assertEqual([st.last_fire for st in gate._keys.values()], [0.0])

        # And the restored stamp is operative. Isolate it from the armed latch by re-arming the
        # edge first with a confident-false — the blip the refractory exists to survive — then the
        # due fire 10 s later, well inside the 3600 s window, is suppressed with the typed reason.
        key = next(iter(gate._keys))
        self.assertEqual(gate.decide(key, NO, now=7.0).reason, "not_satisfied")
        outcome = gate.decide(key, YES, now=10.0)
        self.assertEqual((outcome.action, outcome.reason), (SILENT, "refractory"))

        # The mutation: without the restored stamp the identical sequence FIRES, so the assertion
        # above is checking the stamp rather than the latch.
        blank = Gate(refractory_s=self.REFRACTORY_S)
        blank.decide(key, NO, now=7.0)
        self.assertEqual(blank.decide(key, YES, now=10.0).action, FIRE)

    def test_a_subject_free_watch_honours_the_refractory_window(self) -> None:
        """The control for the two bars below, and the proof the knob is wired at all: with no
        episode subject bound, a confident-false blip re-arms the edge and the refractory memory
        still suppresses the re-fire, publishing the typed suppression."""
        core = _core(refractory_s=self.REFRACTORY_S)
        core.add_goal(_goal())  # no identity_policy -> no subject -> gate-only fold
        session = _drive(core, [(0.0, STILL, "open"), (5.0, STILL, "shut"), (10.0, STILL, "open")])

        self.assertEqual(session.fire_times, [0.0])
        reasons = [ev.data.get("reason") for ev in session.health if ev.kind == "gate_suppressed"]
        self.assertIn("refractory", reasons, "the suppression is published, not silent")

    def test_a_bound_episode_subject_does_not_erase_the_refractory_window(self) -> None:
        """THE ROOT MECHANISM (finding restart-03), with no restart in it at all. Same stream, same
        refractory, one difference: a subject is bound, so the confident-false blip closes an episode
        and the next true observation folds as a NEW one and asks for a re-arm.

        This is the exact scenario ``Gate.decide``'s refractory paragraph names as the reason the
        feature exists — the judge flickering to "no" mid-occurrence. ``rearm`` clearing ``last_fire``
        made the knob inert here and only here, i.e. on every configuration with a subject bound,
        which is every configuration the tuning was for.
        """
        core = _core(refractory_s=self.REFRACTORY_S)
        core.add_goal(_goal(identity_policy="scene"))
        session = _drive(core, [(0.0, STILL, "open"), (5.0, STILL, "shut"), (10.0, STILL, "open")])
        self.assertEqual(session.fire_times, [0.0], "the blip must not re-open the window")
        # ...and it is SUPPRESSED, not merely absent — the same published typed reason the
        # subject-free control asserts, which is what tells an operator the knob did something.
        reasons = [ev.data.get("reason") for ev in session.health if ev.kind == "gate_suppressed"]
        self.assertIn("refractory", reasons, "the suppression is published, not silent")

    def test_a_restart_inside_the_refractory_window_does_not_re_fire_early(self) -> None:
        """THE BAR (finding restart-02). Killed 5 s into a 3600 s refractory window and restored, the
        next observation used to re-fire — 10 s after the fire the window was supposed to protect.

        The stamp is in the snapshot and ``restore`` installs it; what destroyed it was the first
        fold's re-arm, one layer up. The gate's own restore docstring calls losing exactly one of its
        two suppression belts "strictly worse than losing both", and this reached BOTH from the
        outside after a correct restore — so the assertion below is paired with a direct read of the
        post-restart keyspace showing both belts intact rather than one of them merely unused.
        """
        store = InMemoryPersistence()
        before = _core(refractory_s=self.REFRACTORY_S)
        before.add_goal(_goal(identity_policy="scene"))
        first = _drive(before, _stream(0, 5), store=store)
        self.assertEqual(first.fire_times, [0.0])

        after = _core(refractory_s=self.REFRACTORY_S)
        after.add_goal(_goal(identity_policy="scene"))
        second = _drive(after, _stream(10, 15), store=store, restore=True)
        self.assertEqual(second.fire_times, [], "nothing may fire inside the restored window")

        states = list(after.gate._keys.values())
        self.assertEqual([st.armed for st in states], [False], "belt 1: the latch is still spent")
        self.assertEqual([st.last_fire for st in states], [0.0], "belt 2: the window still knows when")


# ---------------------------------------------------------------------------
# 3. Kill mid-dwell: a restarted dwell must credit no pre-restart time.
# ---------------------------------------------------------------------------


class ADwellRestartsHonestly(unittest.TestCase):
    """A 10 s sustained-duration watch is killed 5 s into its dwell. The honest answer is that the
    dwell starts over: its start is a stamp on a timeline the new process cannot rebase, so
    resuming it would fire early or late with no way to tell which.

    Two independent belts enforce that, and both are asserted — a single belt would leave the
    property resting on whichever one a future change happens to keep.
    """

    HOLD_S = 10.0

    def _dwell_goal(self, hold_s: float | None = None) -> Goal:
        return _goal(hold_s=self.HOLD_S if hold_s is None else hold_s)

    def test_an_uninterrupted_dwell_fires_the_moment_it_completes(self) -> None:
        """The positive control, and the value the failing case would produce: uninterrupted, this
        watch fires at t=10 — so t=10 is a time this harness can and does emit. An assertion that
        the restarted dwell does NOT fire at t=10 is therefore discriminating rather than vacuous."""
        core = _core()
        core.add_goal(self._dwell_goal())
        session = _drive(core, _stream(0, 5, 10, 15))
        self.assertEqual(session.fire_times, [10.0])

    def test_a_dwell_killed_half_way_starts_over_rather_than_completing_early(self) -> None:
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(self._dwell_goal())
        first = _drive(before, _stream(0, 5), store=store)
        self.assertEqual(first.fire_times, [], "5 s of a 10 s dwell fires nothing")

        after = _core()
        after.add_goal(self._dwell_goal())
        second = _drive(after, _stream(10, 15, 20, 25), store=store, restore=True)

        self.assertNotIn(
            10.0, second.fire_times,
            "crediting the 5 s served before the kill would have completed the dwell here",
        )
        self.assertEqual(
            second.fire_times, [20.0],
            "the dwell restarts at the first post-restart observation and runs its full length",
        )

    def test_the_snapshot_carries_the_dwell_timer_for_inspection_and_restore_refuses_it(self) -> None:
        """Belt one, with the mutation that proves the belt is load-bearing: put the pre-restart
        ``dwell_since`` back by hand and the identical decision flips from SILENT to FIRE. The
        assertion above is therefore checking something that can change."""
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(self._dwell_goal())
        _drive(core, _stream(0, 5), store=store)

        snapshot = store.load("percept")["gate"]
        self.assertEqual(
            snapshot["keys"][0]["dwell_since"], 0.0,
            "the payload records the timer, so an operator can see what was dropped",
        )

        honest = Gate()
        honest.restore(snapshot)
        key = tuple(snapshot["keys"][0]["key"])
        self.assertIsNone(honest._keys[key].dwell_since, "restore refuses to resume it")
        self.assertEqual(honest.decide(key, YES, hold_s=self.HOLD_S, now=10.0).action, SILENT)

        mutant = Gate()
        mutant.restore(snapshot)
        mutant._keys[key].dwell_since = 0.0  # the mutation: credit the pre-restart time
        mutant._keys[key].dwell_last_supported = 5.0
        self.assertEqual(
            mutant.decide(key, YES, hold_s=self.HOLD_S, now=10.0).action, FIRE,
            "a resumed timer fires here — which is what the honest path must not do",
        )

    def test_the_restart_discontinuity_clears_pending_dwell_as_well(self) -> None:
        """Belt two: the worker treats the restart as a source discontinuity, and
        ``Percept.advance_source_continuity`` invalidates unproven duration support for every
        sustained-duration watch on that source. Asserted separately because it holds even if a
        future change teaches the gate snapshot to rebase a dwell."""
        key = ("door", None)
        core = _core()
        core.add_goal(self._dwell_goal())
        core.gate._state(key).dwell_since = 0.0
        core.gate._state(key).dwell_last_supported = 5.0
        self.assertEqual(core.gate._dwell_since, {key: 0.0})

        core.advance_source_continuity("camera")
        self.assertEqual(core.gate._dwell_since, {}, "no unproven duration support survives")

        # The mutation: a watch with no declared dwell has nothing to invalidate, so the sweep is
        # selective rather than a blanket reset of the keyspace.
        other = _core()
        other.add_goal(_goal())  # no hold_s
        other.gate._state(key).dwell_since = 0.0
        other.advance_source_continuity("camera")
        self.assertEqual(other.gate._dwell_since, {key: 0.0})


# ---------------------------------------------------------------------------
# 4. The episode reset must be loud.
# ---------------------------------------------------------------------------


class TheEpisodeResetIsAnnounced(unittest.TestCase):
    """Episode memory is session-only, so a restart forgets it. That is a decision; being silent
    about it is not. ``Percept.restore`` queues one ``episode_memory_reset`` health event and the
    worker drains it into ``on_health`` at startup."""

    def test_a_counting_goal_announces_the_reset(self) -> None:
        store = InMemoryPersistence()
        before = _core()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndrivenBehaviorRouteWarning)
            before.add_goal(Goal(id="cups", condition="a cup on the desk", say="counted",
                                 state={"type": "count", "count": 0}))
        before.save_state(store)

        after = _core()
        after.restore(store.load("percept"))
        queued = after._restore_health_log
        self.assertEqual([kind for kind, _ in queued], ["episode_memory_reset"])
        self.assertEqual(queued[0][1]["goal_ids"], ["cups"])

    def test_a_restart_with_no_episode_memory_in_use_stays_quiet(self) -> None:
        """The mutation of the loudness rule: it must be conditional. A goal that never used the
        episode map has nothing to announce, and announcing it anyway would train an operator to
        ignore the event."""
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
        before.save_state(store)

        after = _core()
        after.restore(store.load("percept"))
        self.assertEqual(after._restore_health_log, [])

    def test_a_watch_whose_episode_memory_was_used_announces_the_reset_too(self) -> None:
        """THE BAR. It used to fail: the announcement was conditioned on a goal declaring
        ``type='count'`` or ``rearm='each_event'``. A plain watch with a bound subject uses the
        episode map just as much — it is what suppresses every re-statement of a live occurrence —
        and losing it is what produces the duplicate incident this file opens with. That goal shape
        restored in silence, so the one health event designed to explain the amnesia was absent
        exactly where the amnesia changes what the subscriber receives (finding restart-04).

        ``Percept.restore``'s predicate now also covers any concern that SELECTS an identity policy,
        the same resolution the subject-bound episode folds gate on. It is one announcement covering
        both shapes that lost their memory silently: this one, and the claim slot of finding
        claimconv-02 (``test_e2e_claim_conversation.TheClaimTimelineAcrossARestart``).
        """
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(_goal(identity_policy="scene"))
        first = _drive(before, _stream(0, 5), store=store)
        self.assertEqual(first.fire_times, [0.0], "the occurrence was reported…")
        self.assertNotEqual(before.episodes, {}, "…and the episode map is what remembers it")

        after = _core()
        after.add_goal(_goal(identity_policy="scene"))
        second = _drive(after, _stream(10, 15), store=store, restore=True)
        self.assertIn("episode_memory_reset", second.health_kinds)


# ---------------------------------------------------------------------------
# 5. Recovery time.
# ---------------------------------------------------------------------------


class RecoveryIsInsideTheObjective(unittest.TestCase):
    """Qualified worker recovery is targeted at 30 s. Measured as wall-clock around an in-process
    ``snapshot``/``restore`` that number is four orders of magnitude inside the objective and no
    scenario this harness can build would breach it — a bar that cannot fail is not evidence, so it
    is not asserted here.

    What IS asserted is recovery on the STREAM clock: from the first observation the restarted
    worker accepts to the first correct decision it publishes. That is the interval during which a
    subscriber is not being told about a condition that is already true, it is a property of the
    restored temporal state rather than of this machine, and it moves — the companion test drives
    it past the objective.
    """

    def _recovery_s(self, hold_s: float, resume_at: float, horizon) -> float | None:
        store = InMemoryPersistence()
        before = _core()
        before.add_goal(_goal(hold_s=hold_s))
        _drive(before, _stream(0, 5), store=store)

        after = _core()
        after.add_goal(_goal(hold_s=hold_s))
        session = _drive(after, _stream(*horizon), store=store, restore=True)
        if not session.incidents:
            return None
        self.assertTrue(session.attribution().publishable, "no censored segment in the sample")
        return session.fire_times[0] - resume_at

    def test_recovery_to_the_first_correct_decision_is_inside_the_objective(self) -> None:
        recovery = self._recovery_s(10.0, 10.0, range(10, 40, 5))
        self.assertIsNotNone(recovery, "the restarted worker must reach a decision at all")
        self.assertLessEqual(recovery, RECOVERY_OBJECTIVE_S)
        self.assertEqual(recovery, 10.0, "recovery is the dwell the restart honestly restarted")

    def test_the_recovery_measurement_can_breach_the_objective(self) -> None:
        """The mutation: widen the sustained duration past the objective and the same measurement
        reports a breach. Without this the assertion above is a number nobody has seen move."""
        recovery = self._recovery_s(45.0, 10.0, range(10, 90, 5))
        self.assertGreater(recovery, RECOVERY_OBJECTIVE_S)


# ---------------------------------------------------------------------------
# 6. Committed state must not be lost.
# ---------------------------------------------------------------------------


class CommittedStateSurvivesTheRestart(unittest.TestCase):
    """The zero-loss half of the objective. Goals, the gate's edge state, the entity graph, the
    action log and the delivery ledger — the outbox's memory of what has already been delivered —
    all come back. The ledger used to survive only when an autosave tick happened to land
    (FIND-RESTART-05); the exit path now commits both halves, as its own comment always promised."""

    def test_goals_gate_entities_and_the_action_log_all_come_back(self) -> None:
        store = InMemoryPersistence()
        before = _core(detect=DETECTIONS)
        before.add_goal(_goal(identity_policy="track_iou_label"))
        _drive(before, _stream(0, 5), store=store)

        snapshot = store.load("percept")
        self.assertEqual([item["goal"]["id"] for item in snapshot["goals"]], ["door"])
        self.assertTrue(snapshot["gate"]["keys"], "the edge state is committed")
        self.assertEqual([entity["id"] for entity in snapshot["entities"]], ["e1"])

        after = _core(detect=DETECTIONS)
        after.restore(snapshot)
        self.assertEqual([goal.id for goal in after.goals.all()], ["door"])
        self.assertEqual([entity["id"] for entity in after.graph.snapshot()], ["e1"])

    def test_the_entity_identity_survives_the_restart_rather_than_being_minted_afresh(self) -> None:
        """A restored id is only worth something if it is the SAME track, so the assertion is on
        ``first_seen`` — a re-minted ``e1`` would carry the post-restart time. The mutation drops
        the entities from the snapshot and shows the measurement move."""
        store = InMemoryPersistence()
        before = _core(detect=DETECTIONS)
        before.add_goal(_goal(identity_policy="track_iou_label"))
        _drive(before, _stream(0, 5), store=store)

        after = _core(detect=DETECTIONS)
        after.add_goal(_goal(identity_policy="track_iou_label"))
        after.restore(store.load("percept"))
        _drive(after, _stream(10, 15))
        survivors = after.graph.snapshot()
        self.assertEqual([entity["id"] for entity in survivors], ["e1"])
        self.assertEqual(survivors[0]["first_seen"], 0.0, "the track predates the restart")

        amnesiac = _core(detect=DETECTIONS)
        amnesiac.add_goal(_goal(identity_policy="track_iou_label"))
        amnesiac.restore({**store.load("percept"), "entities": []})
        _drive(amnesiac, _stream(10, 15))
        self.assertEqual(
            [entity["first_seen"] for entity in amnesiac.graph.snapshot()], [10.0],
            "without the restored graph the id is a fresh track wearing the same name",
        )

    def test_the_autosave_loop_commits_the_delivery_ledger(self) -> None:
        """The ledger machinery itself works: when an autosave tick lands, the delivered event id
        is written under its own key and a restart would dedupe a redelivery of it."""
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        delivered: list = []
        _drive(
            core, _stream(*range(0, 40, 5)), store=store, sink=delivered.append,
            autosave_every_s=0.001,
        )
        self.assertEqual(len(delivered), 1)
        self.assertEqual(
            [row[0] for row in store.load("percept_delivery_ledger")], [delivered[0].id]
        )

    def test_the_delivery_ledger_is_committed_on_the_way_out(self) -> None:
        """THE BAR (FIND-RESTART-05, fixed — it was an expected failure).

        ``run``'s exit path saved the core snapshot and stopped; the delivery ledger was written
        only from inside the autosave loop. Its own comment promises "a clean shutdown loses nothing
        since the last autosave", and everything the outbox learned since the last tick was exactly
        what was lost — with ``autosave_every_s`` unset (the DEFAULT) that is the entire ledger.

        The consequence was at-least-once delivery losing its idempotency key across precisely the
        restart it was built for: a redelivery after a clean stop reached the subscriber again.
        """
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(_goal(identity_policy="scene"))
        delivered: list = []
        _drive(core, _stream(0, 5), store=store, sink=delivered.append)

        self.assertEqual(len(delivered), 1, "one event was delivered and committed")
        self.assertIsNotNone(store.load("percept"), "the core snapshot was written on exit")
        self.assertEqual(
            [row[0] for row in store.load("percept_delivery_ledger") or []], [delivered[0].id],
            "the outbox's memory of that delivery must be committed too",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
