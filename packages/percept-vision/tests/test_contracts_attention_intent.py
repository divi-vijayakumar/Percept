"""PCPT-026 — AttentionIntent + GoalOrigin provenance + the session goal-change/list seams.

Stdlib + scripted backends only (no keys, no network). Three pillars:

  1. The requested->effective ATTENTION clamp: a conductor (an untrusted language author) cannot
     self-grant ``preempt="interrupt"``; only a trusted :class:`GoalOrigin` (or an explicit policy
     allowlist) keeps it. ttl/verbatim/preferred_channel map through untouched.
  2. Durable GoalOrigin PROVENANCE: stamped by the SDK entry point (structured arm -> STRUCTURED_TRUSTED;
     conductor GoalOps / arm_from_language -> LANGUAGE_AUTHORED), OVERWRITING a caller-supplied value,
     and surviving snapshot/restore + save_state/load_state.
  3. The session seams the voice template needs: ``on_goal_change`` fires at the AUTHORITATIVE registry
     mutation point (so it catches the arm_from_language path too), and ``list_goals`` bounds the read.
"""
from __future__ import annotations

import asyncio
import json
import unittest

import percept
from percept.adapters import Goal, GoalOrigin
from percept.backends.fake import FakeVision, ScriptedConductor
from percept.contracts import AttentionIntent, Consequence, Incident, Verdict
from percept.incidents import (
    DEFAULT_ATTENTION_POLICY,
    AttentionPolicy,
    attention_intent_for,
    make_incident,
    parse_requested_delivery,
)
from percept.persistence import InMemoryPersistence, SqlitePersistence
from percept.runtime.session import RuntimeSession


def _incident(goal_id: str = "g1", *, message: str = "alert", ts: float = 1.0) -> Incident:
    return Incident(
        id=f"inc-{goal_id}",
        concern_id=goal_id,
        verdict=Verdict(satisfied=True, confidence=0.9),
        ts=ts,
        message=message,
    )


def _goal(goal_id: str = "g1", *, delivery: dict | None = None, state_extra: dict | None = None,
          origin: GoalOrigin = GoalOrigin.LANGUAGE_AUTHORED) -> Goal:
    state: dict = {"type": "watch", "count": 0}
    if delivery is not None:
        state["delivery"] = delivery
    if state_extra:
        state.update(state_extra)
    return Goal(id=goal_id, condition="a person is here", say="alert", state=state, origin=origin)


def _core(**kw):
    vision = FakeVision(script={"seen": Verdict(satisfied=True, confidence=0.95, evidence="a person")})
    return percept.Percept.create(vision=vision, discover_plugins=False, **kw)


# ---------------------------------------------------------------------------
# AttentionIntent DTO + GoalOrigin enum
# ---------------------------------------------------------------------------
class AttentionIntentContractTests(unittest.TestCase):
    def test_frozen_and_default_channel(self):
        ai = AttentionIntent(message="m", goal_id="g", incident_id="i", ts=1.0,
                             preempt="queue", ttl_s=None, verbatim=False)
        self.assertIsNone(ai.preferred_channel)  # a hint; defaults to "app decides"
        with self.assertRaises(Exception):
            ai.preempt = "interrupt"  # frozen membrane

    def test_goal_origin_wire_values(self):
        self.assertEqual(GoalOrigin.STRUCTURED_TRUSTED.value, "structured_trusted")
        self.assertEqual(GoalOrigin.LANGUAGE_AUTHORED.value, "language_authored")
        self.assertEqual(GoalOrigin.RESTORED.value, "restored")
        self.assertEqual(GoalOrigin.SYSTEM_TEMPLATE.value, "system_template")

    def test_default_goal_origin_is_least_privileged(self):
        # a freshly-constructed Goal is never trusted-by-default
        self.assertEqual(Goal(id="g", condition="c", say="s").origin, GoalOrigin.LANGUAGE_AUTHORED)

    def test_attention_intent_in_json_schema(self):
        from percept.contracts import jsonschema as js
        defs = js.generate_all()["$defs"]
        self.assertIn("AttentionIntent", defs)
        self.assertEqual(defs["AttentionIntent"]["properties"]["preempt"],
                         {"enum": ["interrupt", "queue", "drop"]})
        # Goal.origin renders as the enum's wire values
        self.assertEqual(set(defs["Goal"]["properties"]["origin"]["enum"]),
                         {"structured_trusted", "language_authored", "restored", "system_template"})


# ---------------------------------------------------------------------------
# parse_requested_delivery — the single legacy-dict interpreter
# ---------------------------------------------------------------------------
class ParseRequestedDeliveryTests(unittest.TestCase):
    def test_missing_preempt_defaults_to_queue(self):
        r = parse_requested_delivery(None)
        self.assertEqual(r.preempt, "queue")  # legacy: absent == hold behind the turn
        self.assertIsNone(r.ttl_s)
        self.assertFalse(r.verbatim)
        self.assertIsNone(r.preferred_channel)

    def test_explicit_values_preserved(self):
        r = parse_requested_delivery({"preempt": "interrupt", "ttl_s": 5, "verbatim": True,
                                      "preferred_channel": "speech"})
        self.assertEqual(r.preempt, "interrupt")
        self.assertEqual(r.ttl_s, 5.0)  # coerced to float, matching the legacy delivery-dict semantics
        self.assertIsInstance(r.ttl_s, float)
        self.assertTrue(r.verbatim)
        self.assertEqual(r.preferred_channel, "speech")

    def test_drop_preserved(self):
        self.assertEqual(parse_requested_delivery({"preempt": "drop"}).preempt, "drop")

    def test_garbage_preempt_collapses_to_queue(self):
        self.assertEqual(parse_requested_delivery({"preempt": "yolo"}).preempt, "queue")

    def test_non_dict_is_all_defaults(self):
        self.assertEqual(parse_requested_delivery("nope").preempt, "queue")


# ---------------------------------------------------------------------------
# AttentionPolicy.clamp — the requested->effective matrix
# ---------------------------------------------------------------------------
class ClampMatrixTests(unittest.TestCase):
    def _effective(self, *, requested_preempt: str, origin: GoalOrigin,
                   policy: AttentionPolicy | None = None, consequence=None, goal_id="g1") -> str:
        pol = policy or DEFAULT_ATTENTION_POLICY
        req = parse_requested_delivery({"preempt": requested_preempt})
        return pol.clamp(requested=req, origin=origin, consequence=consequence,
                         incident=_incident(goal_id)).preempt

    def test_language_authored_interrupt_is_clamped(self):
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.LANGUAGE_AUTHORED), "queue")

    def test_structured_trusted_keeps_interrupt(self):
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.STRUCTURED_TRUSTED), "interrupt")

    def test_system_template_keeps_interrupt(self):
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.SYSTEM_TEMPLATE), "interrupt")

    def test_restored_interrupt_is_clamped(self):
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.RESTORED), "queue")

    def test_allowlisted_consequence_class_keeps_interrupt(self):
        pol = AttentionPolicy(allow_interrupt_consequences=frozenset({"severe"}))
        self.assertEqual(
            self._effective(requested_preempt="interrupt", origin=GoalOrigin.LANGUAGE_AUTHORED,
                            policy=pol, consequence=Consequence(miss_cost="severe")),
            "interrupt",
        )
        # a mild consequence under the same policy is still clamped
        self.assertEqual(
            self._effective(requested_preempt="interrupt", origin=GoalOrigin.LANGUAGE_AUTHORED,
                            policy=pol, consequence=Consequence(miss_cost="mild")),
            "queue",
        )

    def test_explicit_goal_allowlist_keeps_interrupt(self):
        pol = AttentionPolicy(allow_interrupt_goal_ids=frozenset({"g1"}))
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.LANGUAGE_AUTHORED, policy=pol,
                                          goal_id="g1"), "interrupt")
        self.assertEqual(self._effective(requested_preempt="interrupt",
                                          origin=GoalOrigin.LANGUAGE_AUTHORED, policy=pol,
                                          goal_id="other"), "queue")

    def test_queue_and_drop_never_elevated(self):
        # the policy only ever REDUCES authority — a trusted origin does not upgrade a queue/drop request
        self.assertEqual(self._effective(requested_preempt="queue",
                                          origin=GoalOrigin.STRUCTURED_TRUSTED), "queue")
        self.assertEqual(self._effective(requested_preempt="drop",
                                          origin=GoalOrigin.STRUCTURED_TRUSTED), "drop")

    def test_ttl_verbatim_channel_pass_through(self):
        req = parse_requested_delivery({"preempt": "queue", "ttl_s": 3, "verbatim": True,
                                        "preferred_channel": "webhook"})
        ai = DEFAULT_ATTENTION_POLICY.clamp(requested=req, origin=GoalOrigin.LANGUAGE_AUTHORED,
                                            consequence=None, incident=_incident("g1"))
        self.assertEqual(ai.ttl_s, 3.0)
        self.assertTrue(ai.verbatim)
        self.assertEqual(ai.preferred_channel, "webhook")

    def test_intent_carries_incident_provenance(self):
        inc = _incident("g7", message="the kettle boiled", ts=12.5)
        ai = DEFAULT_ATTENTION_POLICY.clamp(
            requested=parse_requested_delivery(None), origin=GoalOrigin.STRUCTURED_TRUSTED,
            consequence=None, incident=inc)
        self.assertEqual(ai.message, "the kettle boiled")
        self.assertEqual(ai.goal_id, "g7")
        self.assertEqual(ai.incident_id, "inc-g7")
        self.assertEqual(ai.ts, 12.5)


# ---------------------------------------------------------------------------
# attention_intent_for — the end-to-end goal+incident mapping
# ---------------------------------------------------------------------------
class AttentionIntentForTests(unittest.TestCase):
    def test_language_authored_goal_interrupt_clamped(self):
        goal = _goal(delivery={"preempt": "interrupt"}, origin=GoalOrigin.LANGUAGE_AUTHORED)
        ai = attention_intent_for(_incident("g1"), goal, DEFAULT_ATTENTION_POLICY)
        self.assertEqual(ai.preempt, "queue")

    def test_structured_goal_keeps_interrupt(self):
        goal = _goal(delivery={"preempt": "interrupt"}, origin=GoalOrigin.STRUCTURED_TRUSTED)
        ai = attention_intent_for(_incident("g1"), goal, DEFAULT_ATTENTION_POLICY)
        self.assertEqual(ai.preempt, "interrupt")

    def test_consequence_derived_from_goal_state(self):
        # a severe-miss goal + a consequence-allowlist policy keeps interrupt even language-authored
        goal = _goal(delivery={"preempt": "interrupt"}, state_extra={"miss_cost": "severe"},
                     origin=GoalOrigin.LANGUAGE_AUTHORED)
        pol = AttentionPolicy(allow_interrupt_consequences=frozenset({"severe"}))
        self.assertEqual(attention_intent_for(_incident("g1"), goal, pol).preempt, "interrupt")

    def test_default_policy_when_none(self):
        goal = _goal(delivery={"preempt": "interrupt"}, origin=GoalOrigin.LANGUAGE_AUTHORED)
        self.assertEqual(attention_intent_for(_incident("g1"), goal).preempt, "queue")

    def test_end_to_end_from_fire(self):
        # a fire -> make_incident -> attention_intent_for, the path the template walks
        class _Fire:
            goal_id = "g1"
            text = "someone is at the door"
            verdict = Verdict(satisfied=True, confidence=0.9)
        inc = make_incident(_Fire(), ts=2.0)
        goal = _goal(delivery={"preempt": "interrupt", "verbatim": True},
                     origin=GoalOrigin.STRUCTURED_TRUSTED)
        ai = attention_intent_for(inc, goal)
        self.assertEqual(ai.preempt, "interrupt")
        self.assertTrue(ai.verbatim)
        self.assertEqual(ai.message, "someone is at the door")


# ---------------------------------------------------------------------------
# GoalOrigin provenance — stamped by the API path, never caller-claimed
# ---------------------------------------------------------------------------
class ProvenanceStampingTests(unittest.TestCase):
    def test_add_goal_stamps_structured_trusted_overwriting_caller(self):
        core = _core()
        # caller LIES, claiming trust it should not get; the structured path overwrites it anyway
        core.add_goal(_goal("g1", origin=GoalOrigin.LANGUAGE_AUTHORED))
        self.assertEqual(core.goals.get("g1").origin, GoalOrigin.STRUCTURED_TRUSTED)

    def test_apply_goal_ops_create_stamps_language_authored_overwriting_caller(self):
        from percept.goals import GoalOp
        core = _core()
        # caller tries to claim STRUCTURED_TRUSTED through a conductor op — overwritten to LANGUAGE_AUTHORED
        core.apply_goal_ops([GoalOp(op="create", goal=_goal("g2", origin=GoalOrigin.STRUCTURED_TRUSTED))])
        self.assertEqual(core.goals.get("g2").origin, GoalOrigin.LANGUAGE_AUTHORED)

    def test_update_op_cannot_set_origin(self):
        from percept.goals import GoalOp
        core = _core()
        core.add_goal(_goal("g3"))  # STRUCTURED_TRUSTED
        # Origin is executable authority and is never part of the safe presentation-only update.
        with self.assertRaisesRegex(ValueError, "only the non-executable 'say'"):
            core.apply_goal_ops([GoalOp(op="update", goal_id="g3",
                                        fields={"say": "new", "origin": GoalOrigin.RESTORED})])
        g = core.goals.get("g3")
        self.assertNotEqual(g.say, "new")
        self.assertEqual(g.origin, GoalOrigin.STRUCTURED_TRUSTED)  # origin is immutable via ops

    def test_arm_from_language_stamps_language_authored(self):
        async def go():
            goal = _goal("kettle", origin=GoalOrigin.STRUCTURED_TRUSTED)  # the script's goal LIES too
            conductor = ScriptedConductor(script={"kettle": goal})
            core = _core(conductor=conductor, allow_scripted_brain=True)
            await core.arm_from_language("tell me when the kettle boils")
            g = core.goals.get("kettle")
            self.assertIsNotNone(g)
            self.assertEqual(g.origin, GoalOrigin.LANGUAGE_AUTHORED)
        asyncio.run(go())


# ---------------------------------------------------------------------------
# Provenance survives persistence
# ---------------------------------------------------------------------------
class ProvenanceDurabilityTests(unittest.TestCase):
    def test_origin_survives_snapshot_restore(self):
        core = _core()
        core.add_goal(_goal("g1"))  # -> STRUCTURED_TRUSTED
        snap = core.snapshot()
        # snapshot is JSON-able: origin persisted as its wire string, not a raw enum object
        blob = json.dumps(snap)
        self.assertIn("structured_trusted", blob)
        restored = _core()
        restored.restore(json.loads(blob))
        self.assertEqual(restored.goals.get("g1").origin, GoalOrigin.STRUCTURED_TRUSTED)

    def test_origin_survives_save_load_state(self):
        for store in (InMemoryPersistence(), SqlitePersistence(":memory:")):
            with self.subTest(store=type(store).__name__):
                a = _core()
                a.add_goal(_goal("g1"))
                a.save_state(store)
                b = _core()
                self.assertTrue(b.load_state(store))
                self.assertEqual(b.goals.get("g1").origin, GoalOrigin.STRUCTURED_TRUSTED)

    def test_language_authored_origin_survives(self):
        from percept.goals import GoalOp
        core = _core()
        core.apply_goal_ops([GoalOp(op="create", goal=_goal("g2"))])  # -> LANGUAGE_AUTHORED
        restored = _core()
        restored.restore(json.loads(json.dumps(core.snapshot())))
        self.assertEqual(restored.goals.get("g2").origin, GoalOrigin.LANGUAGE_AUTHORED)

    def test_pre_provenance_snapshot_restores_as_restored(self):
        # a snapshot from before origin existed carries no `origin` key -> neutral RESTORED
        core = _core()
        legacy_snap = {
            "version": 2,
            "goals": [{"goal": {"id": "old", "condition": "c", "say": "s"}, "status": "active"}],
        }
        core.restore(legacy_snap)
        self.assertEqual(core.goals.get("old").origin, GoalOrigin.RESTORED)


# ---------------------------------------------------------------------------
# Session seam: on_goal_change fires for EVERY registry mutation path
# ---------------------------------------------------------------------------
class OnGoalChangeTests(unittest.TestCase):
    def test_fires_on_structured_arm(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            hits = []
            sess.on_goal_change(lambda: hits.append(1))
            await sess.arm(_goal("g1"))
            self.assertEqual(len(hits), 1)
        asyncio.run(go())

    def test_fires_on_arm_from_language(self):
        async def go():
            conductor = ScriptedConductor(script={"kettle": _goal("kettle")})
            core = _core(conductor=conductor, allow_scripted_brain=True)
            sess = RuntimeSession(core)
            hits = []
            sess.on_goal_change(lambda: hits.append(1))
            await core.arm_from_language("tell me when the kettle boils")
            self.assertEqual(len(hits), 1)  # the rich command path DOES reach the callback
        asyncio.run(go())

    def test_fires_on_disarm(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1"))
            hits = []
            sess.on_goal_change(lambda: hits.append(1))
            await sess.disarm("g1")
            self.assertEqual(len(hits), 1)
        asyncio.run(go())

    def test_fires_on_restore(self):
        async def go():
            core = _core()
            core.add_goal(_goal("g1"))
            snap = core.snapshot()
            sess = RuntimeSession(core)
            hits = []
            sess.on_goal_change(lambda: hits.append(1))
            core.restore(snap)
            self.assertEqual(len(hits), 1)
        asyncio.run(go())

    def test_same_id_replacement_is_rejected_without_change_notification(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1", state_extra={"v": 1}))
            hits = []
            sess.on_goal_change(lambda: hits.append(1))
            with self.assertRaisesRegex(ValueError, "same-id replacement"):
                await sess.arm(_goal("g1", state_extra={"v": 2}))
            self.assertEqual(hits, [])
        asyncio.run(go())

    def test_unsubscribe_stops_callbacks(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            hits = []
            unsub = sess.on_goal_change(lambda: hits.append(1))
            await sess.arm(_goal("g1"))
            unsub()
            await sess.arm(_goal("g2"))
            self.assertEqual(len(hits), 1)  # only the first arm counted
        asyncio.run(go())

    def test_isolated_one_raising_subscriber_does_not_break_others(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            good = []

            def boom():
                raise RuntimeError("bad consumer")

            sess.on_goal_change(boom)
            sess.on_goal_change(lambda: good.append(1))
            await sess.arm(_goal("g1"))  # must not raise; the good subscriber still fires
            self.assertEqual(len(good), 1)
        asyncio.run(go())


# ---------------------------------------------------------------------------
# Session seam: list_goals is a bounded public read
# ---------------------------------------------------------------------------
class ListGoalsTests(unittest.TestCase):
    def test_lists_active_goals_bounded_fields(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1"))
            rows = sess.list_goals()
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0].keys()), {"id", "condition", "status", "origin", "delivery"})
            self.assertEqual(rows[0]["id"], "g1")
            self.assertEqual(rows[0]["condition"], "a person is here")
            self.assertEqual(rows[0]["status"], "active")
            # PCPT-029 enrichment: durable origin (wire string, structured via session.arm) + the parsed
            # requested delivery (defaults when the goal declares none).
            self.assertEqual(rows[0]["origin"], "structured_trusted")
            self.assertEqual(rows[0]["delivery"], {"preempt": "queue", "ttl_s": None, "verbatim": False})
        asyncio.run(go())

    def test_goal_view_resolves_one_goal_with_authored_delivery(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1", delivery={"preempt": "interrupt", "ttl_s": 12.0,
                                                 "verbatim": True}))
            view = sess.goal_view("g1")
            self.assertEqual(view["id"], "g1")
            self.assertEqual(view["origin"], "structured_trusted")
            self.assertEqual(view["delivery"], {"preempt": "interrupt", "ttl_s": 12.0, "verbatim": True})
            self.assertIsNone(sess.goal_view("nope"))   # unknown id → None, no Goal leak
        asyncio.run(go())

    def test_excludes_disarmed_goals(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1"))
            await sess.arm(_goal("g2"))
            await sess.disarm("g1")
            ids = {r["id"] for r in sess.list_goals()}
            self.assertEqual(ids, {"g2"})  # cancelled goal not shown
        asyncio.run(go())

    def test_returns_plain_dicts_no_core_leak(self):
        async def go():
            core = _core()
            sess = RuntimeSession(core)
            await sess.arm(_goal("g1"))
            row = sess.list_goals()[0]
            self.assertIsInstance(row, dict)
            self.assertNotIn("state", row)  # no goal internals leak out
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
