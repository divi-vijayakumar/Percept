"""The plan conductor (L1) — a goal's PRIMITIVE is authored, once, at submission.

``Goal.state["type"]`` used to be a constant: unless a caller named a primitive by hand, every goal
got the default plain Watch. :mod:`percept.conductor` makes it a decision taken from the goal's words.

Everything here is offline. The provider seam is injected (``PlanConductor(call=...)``) or replaced
entirely by a frozen artifact, so the suite makes no provider call — which is also the point being
tested: an LLM in the authoring path must not be able to make a run irreproducible.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from percept import Percept
from percept.behavior_routes import BehaviorRoute, goal_behavior_route
from percept.conductor import (
    AUTHORING_NAMESPACE,
    CONDUCTOR_MODEL,
    PROMPT_VERSION,
    AuthoredPrimitive,
    AuthoringCache,
    AuthoringCacheMiss,
    AuthoringError,
    PlanConductor,
    authoring_key,
    goal_kind_of,
    selection_accuracy,
)
from percept.tracing import content_hash
from percept.watchplan_apply import apply_watch_plan

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / "eval" / "primitive-authoring" / "corpus-100.frozen.json"

# One scripted reply per goal shape — what haiku is asked to produce, exercised without haiku.
RHYTHM = '{"behavior": {"type": "rhythm"}, "rhythm": {"mode": "discrete", "channel": "limb-vertical"}}'
RHYTHM_TARGET = '{"behavior": {"type": "rhythm", "accumulate": {"fire_when": "n == 20"}}, "rhythm": {}}'
TRANSITION = (
    '{"behavior": {"type": "transition", "from": "the onion is unpeeled", '
    '"to": "the onion is peeled"}}'
)
WATCH = '{"behavior": {}}'
COUNT = '{"behavior": {"type": "count"}}'
STATE = '{"behavior": {"rearm": "each_event"}}'


class _ScriptedCall:
    """The provider seam, scripted. Counts calls so "authored once, at submission" is checkable."""

    def __init__(self, replies: dict[str, str], default: str = WATCH) -> None:
        self.replies = replies
        self.default = default
        self.calls: list[str] = []

    async def __call__(self, system: str, user: str) -> str:
        self.calls.append(user)
        return self.replies.get(user, self.default)


def _author(text: str, reply: str, **kwargs) -> AuthoredPrimitive:
    conductor = PlanConductor(call=_ScriptedCall({text: reply}), **kwargs)
    return asyncio.run(conductor.author(text))


class PrimitiveSelectionTests(unittest.TestCase):
    """The conductor picks a primitive, and the pick lands in Goal.state.*."""

    def test_rhythm_for_a_rep_counting_goal(self):
        primitive = _author("a person completes a squat", RHYTHM)
        self.assertEqual(primitive.behavior_type, "rhythm")
        self.assertEqual(primitive.explicit["Goal.state.type"], "rhythm")
        self.assertEqual(primitive.explicit["Goal.state.rhythm.channel"], "limb-vertical")
        self.assertEqual(goal_kind_of(primitive), "count")

    def test_rhythm_target_becomes_the_count_milestone(self):
        # "tell me when I've done 20" -> the EXISTING fire_when predicate, which is what
        # percept.rhythm treats as the goal's one Incident (never one per rep).
        primitive = _author("tell me when I have done 20 push-ups", RHYTHM_TARGET)
        self.assertEqual(primitive.explicit["Goal.state.fire_when"], "n == 20")

    def test_transition_carries_both_named_states(self):
        primitive = _author("Peel onion.", TRANSITION)
        self.assertEqual(primitive.behavior_type, "transition")
        self.assertEqual(primitive.explicit["Goal.state.from"], "the onion is unpeeled")
        self.assertEqual(primitive.explicit["Goal.state.to"], "the onion is peeled")
        self.assertEqual(goal_kind_of(primitive), "transition")

    def test_a_holding_activity_stays_a_plain_watch(self):
        primitive = _author("Holding a book", WATCH)
        self.assertEqual(primitive.behavior_type, "watch")
        self.assertEqual(dict(primitive.explicit), {})  # nothing authored == the safe default
        self.assertEqual(goal_kind_of(primitive), "duration")

    def test_a_state_monitor_rearms_on_every_change(self):
        primitive = _author("Monitor the lid whenever it changes between open and closed", STATE)
        self.assertEqual(primitive.explicit["Goal.state.rearm"], "each_event")
        self.assertEqual(goal_kind_of(primitive), "state")

    def test_distinct_occurrence_counting_is_the_count_primitive(self):
        primitive = _author("Count how many children are visible", COUNT)
        self.assertEqual(primitive.behavior_type, "count")
        self.assertEqual(goal_kind_of(primitive), "count")


class AuthoringFirewallTests(unittest.TestCase):
    """The reply is untrusted; it goes through the shipped coercion firewalls."""

    def test_an_unknown_route_is_dropped_to_the_safe_watch(self):
        primitive = _author("x", '{"behavior": {"type": "telepathy"}}')
        self.assertEqual(primitive.behavior_type, "watch")
        self.assertEqual(dict(primitive.explicit), {})

    def test_an_unsafe_fire_when_never_reaches_goal_state(self):
        primitive = _author(
            "x", '{"behavior": {"accumulate": {"fire_when": "__import__(\'os\').system(\'x\')"}}}'
        )
        self.assertNotIn("Goal.state.fire_when", primitive.explicit)

    def test_a_transition_missing_a_state_is_not_authored_as_a_transition(self):
        primitive = _author("x", '{"behavior": {"type": "transition", "from": "a"}}')
        self.assertEqual(primitive.behavior_type, "watch")

    def test_a_garbage_rhythm_prior_is_dropped_but_the_primitive_survives(self):
        primitive = _author("x", '{"behavior": {"type": "rhythm"}, "rhythm": {"channel": 17}}')
        self.assertEqual(primitive.behavior_type, "rhythm")
        self.assertEqual(
            [p for p in primitive.explicit if p.startswith("Goal.state.rhythm.")], []
        )

    def test_a_rhythm_prior_is_not_attached_to_a_non_rhythm_primitive(self):
        primitive = _author("x", '{"behavior": {}, "rhythm": {"channel": "limb-vertical"}}')
        self.assertEqual(dict(primitive.explicit), {})

    def test_an_unreadable_reply_degrades_to_a_recorded_fallback(self):
        primitive = _author("x", "I'm sorry, I can't do that.")
        self.assertEqual(primitive.behavior_type, "watch")
        self.assertEqual(primitive.origin, "fallback")  # never an invisible degradation

    def test_a_fenced_reply_is_still_parsed(self):
        primitive = _author("x", f"```json\n{TRANSITION}\n```")
        self.assertEqual(primitive.behavior_type, "transition")


class AuthoringNamespaceTests(unittest.TestCase):
    """The conductor writes Goal.state.* and nothing else — the mirror of the optimizer guardrail."""

    def test_every_authored_path_is_goal_state(self):
        for reply in (RHYTHM, RHYTHM_TARGET, TRANSITION, COUNT, STATE):
            primitive = _author("x", reply)
            with self.subTest(reply=reply[:40]):
                self.assertTrue(
                    all(path.startswith(AUTHORING_NAMESPACE) for path in primitive.explicit),
                    primitive.explicit,
                )

    def test_a_foreign_path_is_refused_rather_than_carried(self):
        with self.assertRaises(AuthoringError):
            AuthoredPrimitive.from_record(
                {
                    "goal_text": "x",
                    "behavior_type": "watch",
                    "explicit": {"Settings.edge.motion_gate": 0.9},
                }
            )

    def test_the_optimizer_guardrail_on_goal_state_is_untouched(self):
        # Authoring Goal.state.* and OPTIMIZING it are different acts. Adding the first must not
        # have relaxed the second: a scored recall number still cannot rewrite what a goal means.
        from percept.optimizer import GUARDRAIL_FORBIDDEN_PREFIXES

        self.assertTrue(
            any(AUTHORING_NAMESPACE.startswith(p) or p == AUTHORING_NAMESPACE
                for p in GUARDRAIL_FORBIDDEN_PREFIXES),
            GUARDRAIL_FORBIDDEN_PREFIXES,
        )


class AuthoredPlanIsHashableTests(unittest.TestCase):
    """The authored plan stays a normal, hashable, armable WatchPlan."""

    def test_the_authored_fields_ride_in_the_watchplan_explicit_block(self):
        primitive = _author("a person completes a squat", RHYTHM)
        plan = primitive.to_watch_plan(id="reps")
        self.assertEqual(plan.condition, "a person completes a squat")  # verbatim, never compiled
        self.assertEqual(plan.to_dict()["explicit"], dict(primitive.explicit))

    def test_authoring_changes_the_plan_hash_and_re_authoring_does_not(self):
        rhythm_plan = _author("g", RHYTHM).to_watch_plan(id="g")
        watch_plan = _author("g", WATCH).to_watch_plan(id="g")
        again = _author("g", RHYTHM).to_watch_plan(id="g")
        self.assertNotEqual(content_hash(rhythm_plan.to_dict()), content_hash(watch_plan.to_dict()))
        self.assertEqual(content_hash(rhythm_plan.to_dict()), content_hash(again.to_dict()))

    def test_an_authored_rhythm_plan_arms_as_a_rhythm_goal(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        primitive = _author("a person completes a squat", RHYTHM)
        receipt = apply_watch_plan(core, primitive.to_watch_plan(id="reps"))
        self.assertEqual(receipt.status, "applied")
        self.assertIs(goal_behavior_route(core.goals.get("reps")), BehaviorRoute.EXTERNAL_RHYTHM)

    def test_an_authored_transition_plan_arms_as_a_transition_goal(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        primitive = _author("Peel onion.", TRANSITION)
        receipt = apply_watch_plan(core, primitive.to_watch_plan(id="peel"))
        self.assertEqual(receipt.status, "applied")
        self.assertIs(goal_behavior_route(core.goals.get("peel")), BehaviorRoute.FRAME_TRANSITION)


class AuthoringIsPerGoalOnceTests(unittest.TestCase):
    """Authoring happens at submission, once per goal — never on the reflex path."""

    def test_a_repeated_goal_text_spends_exactly_one_call(self):
        call = _ScriptedCall({"a person completes a squat": RHYTHM})
        conductor = PlanConductor(call=call, cache=AuthoringCache())

        async def _twice():
            first = await conductor.author("a person completes a squat")
            second = await conductor.author("a person completes a squat")
            return first, second

        first, second = asyncio.run(_twice())
        self.assertEqual(len(call.calls), 1)
        self.assertEqual(first.explicit, second.explicit)
        self.assertEqual(second.origin, "cache")

    def test_the_key_is_content_addressed_by_prompt_model_and_text(self):
        self.assertEqual(authoring_key("a"), authoring_key("a"))
        self.assertNotEqual(authoring_key("a"), authoring_key("b"))
        self.assertNotEqual(authoring_key("a"), authoring_key("a", model="other-model"))


class FrozenArtifactTests(unittest.TestCase):
    """The pin: a frozen artifact replays without a provider and refuses to author live."""

    def _frozen(self) -> AuthoringCache:
        cache = AuthoringCache()
        cache.put(_author("a person completes a squat", RHYTHM))
        return AuthoringCache(cache.to_dict()["entries"], frozen=True)

    def test_a_frozen_hit_replays_with_no_provider_call(self):
        call = _ScriptedCall({})
        conductor = PlanConductor(call=call, cache=self._frozen())
        primitive = asyncio.run(conductor.author("a person completes a squat"))
        self.assertEqual(call.calls, [])  # the network was never reached
        self.assertEqual(primitive.behavior_type, "rhythm")
        self.assertEqual(primitive.origin, "cache")

    def test_a_frozen_miss_refuses_rather_than_authoring_live(self):
        call = _ScriptedCall({})
        conductor = PlanConductor(call=call, cache=self._frozen())
        with self.assertRaises(AuthoringCacheMiss):
            asyncio.run(conductor.author("something not pinned"))
        self.assertEqual(call.calls, [])  # a miss must not silently fall through to a call

    def test_a_frozen_artifact_is_read_only(self):
        with self.assertRaises(AuthoringError):
            self._frozen().put(_author("x", WATCH))

    def test_round_trips_through_a_file(self):
        import tempfile

        cache = AuthoringCache()
        cache.put(_author("a person completes a squat", RHYTHM))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.json"
            cache.save(path)
            reloaded = AuthoringCache.load(path)
            self.assertTrue(reloaded.frozen)  # loading a pin yields a pin
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(
                reloaded.get(authoring_key("a person completes a squat")).behavior_type, "rhythm"
            )


class ScoringTests(unittest.TestCase):
    """The selection-accuracy metric itself, so the reported number is auditable."""

    def test_per_kind_accuracy_and_confusion(self):
        pairs = [
            (_author("a", RHYTHM), "count"),
            (_author("b", COUNT), "count"),
            (_author("c", TRANSITION), "transition"),
            (_author("d", WATCH), "duration"),
            (_author("e", STATE), "state"),
            (_author("f", WATCH), "transition"),  # one miss
        ]
        score = selection_accuracy(pairs)
        self.assertEqual((score["hit"], score["total"]), (5, 6))
        self.assertEqual(score["per_kind"]["count"], {"hit": 2, "total": 2, "accuracy": 1.0})
        self.assertEqual(score["confusion"]["transition"], {"transition": 1, "duration": 1})

    def test_a_default_watch_scores_duration_and_an_each_event_watch_scores_state(self):
        self.assertEqual(goal_kind_of(_author("x", WATCH)), "duration")
        self.assertEqual(goal_kind_of(_author("x", STATE)), "state")


@unittest.skipUnless(FROZEN.exists(), "the frozen corpus authoring artifact is not present")
class FrozenCorpusReplayTests(unittest.TestCase):
    """Replay the recorded corpus measurement offline and re-derive its scorecard.

    This is what makes the reported number checkable without a key: the artifact holds every authored
    decision, and the score is recomputed here from the SAME scoring code the runner used.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FROZEN.read_text())

    def test_the_artifact_names_the_model_and_prompt_it_was_produced_with(self):
        self.assertEqual(self.payload["model"], CONDUCTOR_MODEL)
        self.assertEqual(self.payload["prompt_version"], PROMPT_VERSION)
        self.assertEqual(self.payload["corpus"]["n_cases"], 100)

    def test_replaying_the_pin_reproduces_the_recorded_scorecard(self):
        cache = AuthoringCache(self.payload["entries"], frozen=True)
        call = _ScriptedCall({})
        conductor = PlanConductor(call=call, cache=cache)
        cases = self.payload["cases"]
        replayed = asyncio.run(
            conductor.author_all([case["goal_text"] for case in cases])
        )
        self.assertEqual(call.calls, [])  # a pinned re-run reaches no provider
        score = selection_accuracy(zip(replayed, (case["goal_kind"] for case in cases)))
        self.assertEqual(score["hit"], self.payload["scorecard"]["hit"])
        self.assertEqual(score["per_kind"], self.payload["scorecard"]["per_kind"])

    def test_it_beats_the_fixed_single_watchplan_baseline_by_a_wide_margin(self):
        # The baseline every measured run used: one hand-armed watch for all 100 goals, which can
        # only ever be right on the 4 goals whose kind IS a plain watch shape.
        self.assertGreaterEqual(self.payload["scorecard"]["hit"], 80)

    def test_every_rep_counting_goal_selected_the_rhythm_primitive(self):
        rep_goals = [c for c in self.payload["cases"] if "completes a" in c["goal_text"]]
        self.assertGreaterEqual(len(rep_goals), 20)
        self.assertEqual({c["authored_type"] for c in rep_goals}, {"rhythm"})

    def test_every_pinned_decision_is_an_armable_goal_state_plan(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        for i, record in enumerate(self.payload["entries"].values()):
            primitive = AuthoredPrimitive.from_record(record)
            plan = primitive.to_watch_plan(id=f"pinned-{i}")
            self.assertEqual(plan.to_dict()["explicit"], dict(primitive.explicit))
            if primitive.behavior_type in ("watch", "transition", "rhythm"):
                receipt = apply_watch_plan(core, plan)
                self.assertEqual(receipt.status, "applied", primitive.goal_text)


class NoVendorImportTests(unittest.TestCase):
    def test_importing_the_conductor_does_not_import_the_vendor_sdk(self):
        import subprocess
        import sys

        code = (
            "import sys, percept.conductor as c; "
            "assert 'anthropic' not in sys.modules; print(c.CONDUCTOR_MODEL)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(CONDUCTOR_MODEL, out.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
