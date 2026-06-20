"""L1: consent — act is flag-gated, scoped, logged; blocked attempts raise + are audited."""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import ConsentSettings, Settings
from percept.consent import ConsentDenied


def _act_agent(allow=True, actions=("open_path",)):
    settings = Settings(
        consent=ConsentSettings(allow_computer_use=allow, allowed_actions=frozenset(actions))
    )
    a = Percept.create(settings=settings, discover_plugins=False)
    a.add_goal(Goal(id="g", condition="open the file", say="opening", modality="act"))
    return a


class TestConsentGate(unittest.TestCase):
    def test_act_allowed_when_flag_and_scope_and_modality(self):
        a = _act_agent(allow=True, actions=("open_path",))
        rec = a.act("g", "open_path", {"path": "/tmp/x"}, t=1.0)
        self.assertEqual(rec.kind, "act")
        self.assertTrue(rec.allowed)
        self.assertEqual(a.metrics.get("act.allowed"), 1)
        self.assertEqual(len(a.action_log.entries("act")), 1)

    def test_act_blocked_without_flag(self):
        a = _act_agent(allow=False, actions=("open_path",))
        with self.assertRaises(ConsentDenied):
            a.act("g", "open_path", t=1.0)
        # blocked attempts are still audited (nothing acts silently)
        self.assertEqual(len(a.action_log.entries("act_blocked")), 1)
        self.assertEqual(a.metrics.get("act.blocked"), 1)

    def test_act_blocked_outside_action_scope(self):
        a = _act_agent(allow=True, actions=("open_path",))
        with self.assertRaises(ConsentDenied):
            a.act("g", "delete_everything", t=1.0)

    def test_act_blocked_when_goal_not_act_modality(self):
        a = _act_agent(allow=True, actions=("open_path",))
        a.add_goal(Goal(id="speak_goal", condition="x", say="y", modality="speak"))
        with self.assertRaises(ConsentDenied):
            a.act("speak_goal", "open_path", t=1.0)

    def test_default_profile_denies_everything(self):
        a = Percept.create(discover_plugins=False)  # no consent settings -> off
        a.add_goal(Goal(id="g", condition="x", say="y", modality="act"))
        with self.assertRaises(ConsentDenied):
            a.act("g", "open_path", t=1.0)


class TestActionLogRecordsFires(unittest.TestCase):
    def test_fires_are_logged(self):
        async def run():
            v = FakeVision(
                script={"coffee": Verdict(True, 0.95)},
                detect_script={"coffee": [Observation("cup", Box(0.4, 0.4, 0.2, 0.2), 0.9)]},
            )
            a = Percept.create(vision=v, discover_plugins=False)
            a.add_goal(Goal(id="g1", condition="coffee is present", say="careful"))
            await a.perceive_judged("coffee", t=1.0)  # fires on the rising edge
            self.assertEqual(len(a.action_log.entries("fire")), 1)
            self.assertEqual(a.action_log.entries("fire")[0].goal_id, "g1")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
