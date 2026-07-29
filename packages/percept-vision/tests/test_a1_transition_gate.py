"""A1: the transition sense fires through the GATE, not around it.

The defect this pins
--------------------
``BehaviorExecutor.transition_decide`` used to decide entirely on two bits it kept in its own
per-concern context and return a ``Fired`` directly. It was the only sense that never called
:class:`percept.gate.Gate`, which meant:

* it re-armed its own latch on EVERY confident, non-contradictory ``from`` tick, so a single judge
  flicker back to state A mid-episode re-opened the edge and the same ongoing action re-notified for
  as long as it held;
* it took no ``t`` at all, so the gate's time-based ``refractory_s`` guard could not apply;
* it bypassed the 0.7s delivery merge that lives inside ``fire_judged``;
* and its fires carried ``reason=None``, hard-coded, because no gate decision existed to report.

The fix keeps the ORDERING WITNESS ("has A been seen?") in the executor's isolated context — the gate
has no vocabulary for it, nor for the contradictory tick — and hands the RE-FIRE GUARD to the gate,
translating "A was re-established" into the confident-false the gate re-arms on.

Every count below is REPORTED, never targeted: the assertions are on the mechanism (which gate reason
fired, which seam dropped a delivery), and no shipped default is changed anywhere in this module.
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.behavior import INCIDENT_MERGE_WINDOW_S, BehaviorExecutor
from percept.gate import GATE_REASONS, FIRE, Gate
from percept.incidents import make_incident

YES = Verdict(True, 0.95, "confident yes")
NO = Verdict(False, 0.95, "confident no")
UNSURE = Verdict(True, 0.4, "not confident")

FROM = "the region shows state A"
TO = "the region shows state B"


def _goal(goal_id: str = "t", **state) -> Goal:
    base = {"type": "transition", "from": FROM, "to": TO, "count": 0}
    base.update(state)
    return Goal(id=goal_id, condition="A becomes B", say="changed", state=base)


def _ex(**gate_kwargs) -> tuple[Gate, BehaviorExecutor]:
    gate = Gate(**gate_kwargs)
    return gate, BehaviorExecutor(gate)


class StrictScriptVision:
    """A judge double that REFUSES anything the contract did not promise it.

    The Vision contract is ``judge(condition, frame, ...) -> Verdict`` for a condition this scenario
    actually scripted. Anything else — an unscripted condition, an unscripted frame, a positional
    misuse — is a bug in the caller or in the scenario, and a permissive double would hide it behind a
    plausible default verdict. So it raises.
    """

    def __init__(self, script: dict) -> None:
        self._script = script

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        key = (condition, str(frame))
        if key not in self._script:
            raise AssertionError(
                f"judge called with unscripted {key!r}; the double promises a verdict only for "
                f"conditions/frames this scenario declared, never a stand-in default"
            )
        sat, conf = self._script[key]
        return Verdict(satisfied=sat, confidence=conf, evidence=f"{condition}@{frame}")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _run_agent(frames, *, identity_policy=None, goal_id="t"):
    """Drive the LIVE agent path. ``frames`` is a list of ``(token, t, from_pair, to_pair)``."""
    script = {}
    for tok, _t, fv, tv in frames:
        script[(FROM, tok)] = fv
        script[(TO, tok)] = tv
    state = {}
    if identity_policy is not None:
        state["identity_policy"] = identity_policy
    agent = Percept.create(vision=StrictScriptVision(script), discover_plugins=False)
    agent.add_goal(_goal(goal_id, **state))
    fires = []

    async def go():
        for tok, t, _fv, _tv in frames:
            fires.extend(
                ev for ev in await agent.perceive_judged(tok, t=t) if ev.action == FIRE
            )

    asyncio.run(go())
    return fires, agent


# ==================================================================================================
# ACCEPTANCE 1 + 3 — the repro, and the delivery merge it proves is reachable.
# ==================================================================================================


class HeldActionDoesNotRenotifyAtJudgeCadence(unittest.TestCase):
    """The measured defect shape: one ongoing action, a judge that flickers back to A between
    confident reads of B, and a fresh notification on every tick that follows a flicker."""

    #: The flicker script, at a 0.2s judge cadence. ``to`` is confidently true whenever the judge is
    #: asked about the ongoing action; ``from`` flickers confidently true on alternating ticks (a tick
    #: confident in BOTH is contradictory and is deliberately not this scenario — the defect needs a
    #: flicker the executor accepts as an ordering edge, which is precisely what makes it insidious).
    FLICKER = [
        (0.0, YES, NO),      # A witnessed — arms
        (0.2, NO, YES),      # B — the ONE genuine A->B edge
        (0.4, YES, UNSURE),  # judge flickers back to A (uncertain about B) — the re-arm
        (0.6, NO, YES),      # B again — the action never stopped
        (0.8, YES, UNSURE),  # flicker
        (1.0, NO, YES),      # B again
    ]

    def _drive(self, executor, goal, script):
        return [
            executor.transition_decide(goal, fr, to, focus_id="scene", now=t)
            for t, fr, to in script
        ]

    def test_repro_default_config_withholds_the_flicker_re_notification(self):
        """Pre-fix this delivers THREE notifications for one continuous action: the executor re-armed
        on each flicker and returned a ``Fired`` straight past the delivery seam. Nothing is
        configured here — the merge window is the shipped default, and no number below is a target:
        the assertions are that fewer notifications survive than edges were decided, and that the
        withholding is attributable BY NAME to the seam this path used to bypass."""
        _, ex = _ex()
        ticks = self._drive(ex, _goal(), self.FLICKER)
        delivered = [tick for tick in ticks if tick.fired is not None]
        merged = [tick for tick in ticks if tick.outcome == "merged_duplicate"]

        self.assertTrue(merged, "the delivery seam must be reached and must withhold at least once")
        self.assertEqual(
            len(delivered) + len(merged), 3,
            "the gate decided three rising edges — the flicker really did re-arm it each time; the "
            "fix is not that fewer edges are decided but that fewer are NOTIFIED",
        )
        self.assertEqual(
            [tick.outcome for tick in ticks],
            ["armed", "fire", "rearmed", "merged_duplicate", "rearmed", "fire"],
            "a re-notification inside the "
            f"{INCIDENT_MERGE_WINDOW_S}s window is a re-report of the occurrence already delivered; "
            "one landing outside it is a distinct occurrence and is still delivered",
        )

    def test_repro_through_the_public_watch_api(self):
        """The same flicker driven through ``perceive_judged`` — an API whose signature is identical
        before and after this fix, so this reproduces the defect as a DIFFERENCE IN DELIVERED
        NOTIFICATIONS rather than as a missing keyword argument.

        Pre-fix: three FireEvents reach the subscriber for one continuous action, 0.4s apart, each
        one a fresh 'it happened' for something that never stopped happening.
        """
        frames = [
            (f"x{i}", t, (fr.satisfied, fr.confidence), (to.satisfied, to.confidence))
            for i, (t, fr, to) in enumerate(self.FLICKER)
        ]
        fires, _ = _run_agent(frames)

        gaps = [round(b - a, 6) for a, b in zip(
            [f[1] for f in frames if f[0] in {"x1", "x3", "x5"}][:-1],
            [f[1] for f in frames if f[0] in {"x1", "x3", "x5"}][1:],
        )]
        self.assertEqual(gaps, [0.4, 0.4], "the scenario's edges really are a judge-cadence apart")
        self.assertLess(
            len(fires), 3,
            "one continuous action must not re-notify once per judge tick; every notification "
            "withheld here was withheld by the delivery seam this path used to return around",
        )
        # Whatever survives carries the gate's typed reason — the deliveries are gate decisions now.
        self.assertTrue(all(fire.reason == "rising_edge" for fire in fires))

    def test_merge_window_is_load_bearing_not_incidental(self):
        """Turning the delivery window OFF restores every re-notification — proof the merge (and not
        some other accident of the rewrite) is what withheld them, and that the edges really did
        reach the gate."""
        gate = Gate()
        ex = BehaviorExecutor(gate, merge_window_s=0.0)  # 0 disables the merge by contract
        ticks = self._drive(ex, _goal(), self.FLICKER)
        self.assertEqual(len([t for t in ticks if t.fired is not None]), 3)

    def test_at_the_measured_cadence_the_operator_refractory_is_now_reachable(self):
        """The re-fires measured in production sat ~1.0-1.3s apart — WIDER than the delivery window,
        so the merge alone cannot absorb them. The gate's documented ``refractory_s`` knob is the
        guard for exactly that ("a judge briefly flickered to 'no' mid-episode"), and routing this
        sense through the gate is what finally lets it apply. No shipped default is changed: the
        knob is supplied by this scenario, and the un-configured half is reported, not asserted at.
        """
        cadence = [
            (0.0, YES, NO),
            (1.0, NO, YES),      # the one genuine edge
            (2.0, YES, UNSURE),  # flicker
            (3.0, NO, YES),      # re-notification 1.0s after the last, outside the merge window
            (4.0, YES, UNSURE),
            (5.0, NO, YES),
        ]
        _, plain = _ex()
        unguarded = len([t for t in self._drive(plain, _goal(), cadence) if t.fired is not None])

        _, guarded_ex = _ex(refractory_s=2.5)
        guarded_ticks = self._drive(guarded_ex, _goal(), cadence)
        guarded = len([t for t in guarded_ticks if t.fired is not None])

        self.assertLess(
            guarded, unguarded,
            "with a refractory configured the gate must withhold the flicker-driven re-fires; "
            "before this fix the transition sense never reached the gate, so the knob was inert",
        )
        self.assertIn(
            "suppressed_refractory", [t.outcome for t in guarded_ticks],
            "the suppression must be attributable to the gate's refractory guard by name",
        )


# ==================================================================================================
# ACCEPTANCE 2 — a populated GateReason on the wire.
# ==================================================================================================


class FiresCarryTheGateReason(unittest.TestCase):
    def test_executor_fire_carries_a_real_reason_code(self):
        _, ex = _ex()
        goal = _goal()
        ex.transition_decide(goal, YES, NO, now=0.0)
        tick = ex.transition_decide(goal, NO, YES, now=1.0)

        self.assertIsNotNone(tick.fired)
        self.assertIsNotNone(tick.fired.reason, "reason was hard-coded None on this path")
        self.assertIn(tick.fired.reason, GATE_REASONS)
        self.assertEqual(tick.fired.reason, "rising_edge")

    def test_reason_survives_to_the_delivered_incident(self):
        """It is not enough for the executor to know the reason — the agent dropped it when building
        the FireEvent, so every transition Incident reached a subscriber with an empty ``reason``
        beside a populated judged one."""
        fires, _ = _run_agent([
            ("k0", 1.0, (True, 0.95), (False, 0.95)),
            ("k1", 2.0, (False, 0.95), (True, 0.95)),
        ])
        self.assertEqual(len(fires), 1)
        self.assertEqual(fires[0].reason, "rising_edge")
        self.assertEqual(make_incident(fires[0], ts=2.0).reason, "rising_edge")

    def test_the_frozen_judgment_event_records_the_same_reason(self):
        fires, _ = _run_agent([
            ("j0", 1.0, (True, 0.95), (False, 0.95)),
            ("j1", 2.0, (False, 0.95), (True, 0.95)),
        ])
        judgment = fires[0].judgment
        self.assertIsNotNone(judgment, "the live transition path threads timing, so it has a record")
        self.assertEqual(judgment.reason, "rising_edge")
        self.assertEqual(judgment.evidence_origin, "transition")


# ==================================================================================================
# ACCEPTANCE 4 — true positives are preserved.
# ==================================================================================================


class TruePositivesArePreserved(unittest.TestCase):
    def test_two_genuine_transitions_across_a_re_established_from_fire_exactly_twice(self):
        """``Gate.decide`` re-arms ONLY on a confident-false. A confident, non-contradictory ``from``
        tick IS that falling edge — without translating it, the first fire would latch the key
        permanently and the second genuine transition would be silently lost."""
        _, ex = _ex()
        goal = _goal()
        script = [
            (0.0, YES, NO),   # A
            (1.0, NO, YES),   # -> B : fire 1
            (2.0, YES, NO),   # back to A, confidently: the falling edge
            (3.0, NO, YES),   # -> B : fire 2
        ]
        fired = [
            ex.transition_decide(goal, fr, to, now=t).fired is not None
            for t, fr, to in script
        ]
        self.assertEqual(fired, [False, True, False, True])

    def test_two_genuine_transitions_on_the_live_agent_path(self):
        fires, agent = _run_agent([
            ("g0", 1.0, (True, 0.95), (False, 0.95)),
            ("g1", 2.0, (False, 0.95), (True, 0.95)),   # fire 1
            ("g2", 3.0, (True, 0.95), (False, 0.95)),   # re-established 'from'
            ("g3", 4.0, (False, 0.95), (True, 0.95)),   # fire 2
        ])
        self.assertEqual(len(fires), 2)
        self.assertEqual(agent.goals.get("t").state["count"], 2)

    def test_sustained_to_chatter_after_a_fire_adds_zero_and_the_gate_says_why(self):
        """The complement of the test above: B holding is not a new edge. Every chatter tick must
        still REACH the gate (that is what replaced the ctx re-fire bit), and the gate must answer
        ``already_fired`` every time."""
        _, ex = _ex()
        goal = _goal()
        ex.transition_decide(goal, YES, NO, now=0.0)
        first = ex.transition_decide(goal, NO, YES, now=1.0)
        self.assertIsNotNone(first.fired)

        chatter = [
            ex.transition_decide(goal, NO, YES, now=float(t))
            for t in range(2, 12)
        ]
        self.assertEqual([tick.fired for tick in chatter], [None] * 10)
        self.assertEqual(
            [tick.outcome for tick in chatter], ["suppressed_latched"] * 10,
            "a held B must be suppressed BY THE GATE's spent edge, observably, on every tick",
        )

    def test_sustained_to_chatter_on_the_live_agent_path_adds_zero(self):
        """The agent-path twin of the test above, on an API whose signature did not change — so it
        pins the DELIVERED count, not the executor's internals."""
        frames = [("h0", 1.0, (True, 0.95), (False, 0.95))]
        frames += [(f"h{i}", float(i + 1), (False, 0.95), (True, 0.95)) for i in range(1, 21)]
        fires, agent = _run_agent(frames)
        self.assertEqual(len(fires), 1)
        self.assertEqual(agent.goals.get("t").state["count"], 1)

    def test_joining_mid_stream_already_in_B_still_never_fires(self):
        """The ordering witness is the half the gate cannot express, so it must survive the rewrite:
        a watch that never saw A must never fire, however long B holds."""
        _, ex = _ex()
        goal = _goal()
        ticks = [ex.transition_decide(goal, NO, YES, now=float(t)) for t in range(8)]
        self.assertEqual([tick.fired for tick in ticks], [None] * 8)

    def test_a_contradictory_tick_still_changes_no_latch(self):
        """A tick confident in BOTH states cannot establish ordering. It must reach neither gate
        branch, so a spent edge stays spent and an unarmed one stays unarmed."""
        gate, ex = _ex()
        goal = _goal()
        both = ex.transition_decide(goal, YES, YES, now=0.0)
        self.assertIsNone(both.fired)
        self.assertEqual(gate._armed, {}, "a contradictory tick must not touch the gate at all")

        ex.transition_decide(goal, YES, NO, now=1.0)
        self.assertIsNotNone(ex.transition_decide(goal, NO, YES, now=2.0).fired)
        after = ex.transition_decide(goal, YES, YES, now=3.0)
        self.assertIsNone(after.fired)
        self.assertEqual(after.outcome, "suppressed_latched")
        self.assertIs(
            gate._armed[("t", "scene")], False,
            "a contradiction must not re-arm the spent edge",
        )

    def test_an_unsure_to_never_fires_and_never_reaches_the_gate(self):
        gate, ex = _ex()
        goal = _goal()
        ex.transition_decide(goal, YES, NO, now=0.0)
        tick = ex.transition_decide(goal, NO, UNSURE, now=1.0)
        self.assertIsNone(tick.fired)
        self.assertIs(
            gate._armed[("t", "scene")], True,
            "ambiguity must never be routed to the gate as a decision — the edge stays live and the "
            "gate never answers ASK on a path that has never delivered one",
        )


# ==================================================================================================
# ACCEPTANCE 5 — key discipline under entity churn, per identity policy.
# ==================================================================================================


class KeyDisciplineUnderChurn(unittest.TestCase):
    """New gate keys default to ARMED, so if the latch key moved with the in-focus entity a stream
    that mints a fresh entity id per tick would re-create the defect through fresh keys. It cannot,
    for two independent reasons, and both are pinned here."""

    def test_undeclared_and_scene_policies_pin_one_key_per_goal(self):
        """Reason one: under the default (and explicitly declared) scene policy the latch key is the
        literal stable ``"scene"``, so entity churn cannot mint a key at all."""
        for policy in (None, "scene", "content_key", "speaker_label", "not_a_real_resolver"):
            with self.subTest(policy=policy):
                frames = [("c0", 1.0, (True, 0.95), (False, 0.95))]
                frames += [
                    (f"c{i}", float(i + 1), (False, 0.95), (True, 0.95))
                    for i in range(1, 25)
                ]
                fires, agent = _run_agent(frames, identity_policy=policy)
                self.assertEqual(len(fires), 1)
                transition_keys = [
                    k for k in agent.gate._armed if isinstance(k, tuple) and k[0] == "t"
                ]
                self.assertEqual(
                    transition_keys, [("t", "scene")],
                    f"policy {policy!r} must latch on the stable scene scope, one key per goal",
                )

    def test_a_freshly_minted_entity_key_cannot_fire_on_its_first_tick(self):
        """Reason two, which is what protects a DECLARED per-entity policy: a fresh key is armed, but
        arming is not enough — the ordering witness is per latch key too, so an entity id that was
        never observed in state A under its OWN key has nothing to complete. A stream that re-mints
        an id every tick therefore never fires, and mints no gate key either."""
        gate, ex = _ex()
        goal = _goal(identity_policy="track_iou_label")
        ticks = [
            ex.transition_decide(goal, NO, YES, focus_id=f"entity-{i}", now=float(i))
            for i in range(500)
        ]
        self.assertEqual([tick.fired for tick in ticks], [None] * 500)
        self.assertEqual(
            gate._armed, {},
            "a never-witnessed entity must not even mint a gate key, or 58k mints an hour would "
            "become 58k armed edges",
        )

    def test_per_entity_latches_stay_isolated_and_each_fires_once(self):
        gate, ex = _ex()
        goal = _goal(identity_policy="track_iou_label")

        # e1 witnesses A; e2 completes B without ever witnessing A -> only e1 is completable.
        ex.transition_decide(goal, YES, NO, focus_id="e1", now=0.0)
        self.assertIsNone(ex.transition_decide(goal, NO, YES, focus_id="e2", now=1.0).fired)
        self.assertIsNotNone(ex.transition_decide(goal, NO, YES, focus_id="e1", now=2.0).fired)
        # ...and e1's edge is then spent for e1 alone.
        self.assertIsNone(ex.transition_decide(goal, NO, YES, focus_id="e1", now=3.0).fired)
        self.assertEqual(set(gate._armed), {("t", "e1")})

    def test_transition_keys_are_evictable_and_an_entity_reset_reaches_them(self):
        """The latch keys are deliberately NOT ``no_evict``: pinning one per entity forever is the
        unbounded growth the gate's bounds exist to stop. ``Gate.reset_entity`` — which the runtime
        wires to entity-graph eviction — must therefore reach a transition latch, and the delivery
        merge must catch the re-fire a re-minted key would otherwise produce."""
        gate, ex = _ex()
        goal = _goal(identity_policy="track_iou_label")
        ex.transition_decide(goal, YES, NO, focus_id="e1", now=0.0)
        self.assertIsNotNone(ex.transition_decide(goal, NO, YES, focus_id="e1", now=0.1).fired)
        self.assertEqual(set(gate._armed), {("t", "e1")})

        gate.reset_entity("e1")  # the entity dies; its edge goes with it
        self.assertEqual(gate._armed, {})

        # The id is re-minted inside the delivery window. The ordering witness in ctx survived, so a
        # confident B forms a candidate and the fresh key is armed -> the GATE decides FIRE. Only the
        # delivery merge stands between that and a duplicate notification for one occurrence.
        churned = ex.transition_decide(goal, NO, YES, focus_id="e1", now=0.5)
        self.assertIsNone(churned.fired)
        self.assertEqual(churned.outcome, "merged_duplicate")

        # Outside the window it is a genuinely separate occurrence and is delivered.
        gate.reset_entity("e1")
        later = ex.transition_decide(goal, NO, YES, focus_id="e1", now=0.5 + INCIDENT_MERGE_WINDOW_S)
        self.assertIsNotNone(later.fired)

    def test_a_per_entity_policy_with_no_subject_in_focus_spends_nothing(self):
        """The agent skips the tick entirely when a per-entity policy has nothing in focus, so the
        shared scene key can never be armed or fired on behalf of the entities it isolates."""
        frames = [
            ("z0", 1.0, (True, 0.95), (False, 0.95)),
            ("z1", 2.0, (False, 0.95), (True, 0.95)),
        ]
        # StrictScriptVision raises if judged at all; with no focus the route must not reach it.
        script = {}
        agent = Percept.create(vision=StrictScriptVision(script), discover_plugins=False)
        agent.add_goal(_goal("t", identity_policy="track_iou_label"))

        async def go():
            out = []
            for tok, t, _f, _to in frames:
                out.extend(await agent.perceive_judged(tok, t=t))
            return out

        self.assertEqual(asyncio.run(go()), [])
        self.assertEqual(agent.gate._armed, {})


# ==================================================================================================
# The seam is threaded for dwell, and the clockless caller still works.
# ==================================================================================================


class DwellAndTheClocklessCaller(unittest.TestCase):
    def test_a_threaded_hold_s_defers_the_fire_through_the_gate(self):
        """``hold_s`` is admitted by the shipped duration contract only on an ordinary frame Watch, so
        the production transition route passes ``None``. The seam is threaded regardless, so the
        dwell path is reachable rather than structurally dead."""
        _, ex = _ex()
        goal = _goal()
        ex.transition_decide(goal, YES, NO, now=0.0, hold_s=2.0)
        pending = ex.transition_decide(goal, NO, YES, now=1.0, hold_s=2.0)
        self.assertIsNone(pending.fired)
        self.assertEqual(pending.outcome, "dwell_pending")

        still_pending = ex.transition_decide(goal, NO, YES, now=2.0, hold_s=2.0)
        self.assertIsNone(still_pending.fired)

        satisfied = ex.transition_decide(goal, NO, YES, now=3.5, hold_s=2.0)
        self.assertIsNotNone(satisfied.fired)
        self.assertEqual(satisfied.fired.reason, "dwell_satisfied")

    def test_a_caller_that_threads_no_clock_still_gets_the_rising_edge(self):
        """A bare executor caller (and every pre-existing positional call site) passes no ``now``.
        That is the clockless gate: rising edge only, no refractory, no dwell, no merge."""
        _, ex = _ex()
        goal = _goal()
        self.assertIsNone(ex.fire_transition(goal, YES, NO))
        first = ex.fire_transition(goal, NO, YES)
        self.assertIsNotNone(first)
        self.assertEqual(first.reason, "rising_edge")
        self.assertIsNone(ex.fire_transition(goal, NO, YES))
        ex.fire_transition(goal, YES, NO)
        self.assertIsNotNone(ex.fire_transition(goal, NO, YES))


if __name__ == "__main__":
    unittest.main()
