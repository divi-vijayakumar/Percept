"""A4 action+verification & A5 per-source consent (L1).

World-changing actions under the safety contract are consent-gated via act(), and for
a high-false-fire-cost concern it instantiates a VERIFICATION concern (reusing the deadline/absence
machinery: confirm the effect by a deadline, else fire "couldn't confirm").
Consent is per-source: allowed_sources scopes acting to specific source_ids.
"""

import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.consent import ConsentDenied, ConsentProfile
from percept.gate import FIRE

PERMIT = ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"press_stop"}))


def _agent(consent=PERMIT):
    a = Percept.create(vision=FakeVision(script={}), discover_plugins=False)
    a.consent = consent
    return a


def _act_goal(a, gid="stove", false_fire="irreversible", source="camera"):
    g = Goal(id=gid, condition="the stove is on", say="...", modality="act", source=source)
    if false_fire is not None:
        g.state["false_fire_cost"] = false_fire  # irreversible -> requires verification
    a.add_goal(g)
    return g


class TestActuateVerification(unittest.TestCase):
    def test_high_false_fire_action_instantiates_verification_concern(self):
        a = _agent()
        _act_goal(a)
        rec = a.actuate("stove", "press_stop", expect="the stove is off", verify_within=10.0, t=0.0)
        self.assertTrue(rec.allowed)
        vid = rec.detail["verification_goal_id"]
        self.assertEqual(vid, "stove::verify::press_stop")
        v = a.goals.get(vid)
        self.assertIsNotNone(v)
        self.assertEqual(v.state["rearm"], "deadline")  # reuses A1 absence machinery
        self.assertEqual(v.state["verifies"], "stove")
        self.assertEqual(v.condition, "the stove is off")
        self.assertEqual(len(a.action_log.entries("verify_created")), 1)

    def test_mild_action_creates_no_verification(self):
        a = _agent()
        _act_goal(a, gid="lamp", false_fire=None)  # mild/mild -> no verification required
        rec = a.actuate("lamp", "press_stop", t=0.0)
        self.assertNotIn("verification_goal_id", rec.detail)
        self.assertIsNone(a.goals.get("lamp::verify::press_stop"))

    def test_actuate_blocked_without_consent_creates_no_verification(self):
        a = _agent(consent=ConsentProfile())  # computer-use OFF
        _act_goal(a)
        with self.assertRaises(ConsentDenied):
            a.actuate("stove", "press_stop", t=0.0)
        self.assertIsNone(a.goals.get("stove::verify::press_stop"))  # nothing armed on a blocked act


class TestVerificationComposesWithDeadlines(unittest.TestCase):
    def test_unconfirmed_effect_fires_couldnt_confirm(self):
        a = _agent()
        _act_goal(a)
        a.actuate("stove", "press_stop", verify_within=10.0, t=0.0)
        self.assertEqual(a.check_deadlines(t=5.0), [])  # within the window -> waiting
        ev = a.check_deadlines(t=11.0)  # effect never confirmed -> fire
        self.assertEqual([e.action for e in ev], [FIRE])

    def test_confirmation_cancels_the_verification(self):
        a = _agent()
        _act_goal(a)
        vid = a.actuate("stove", "press_stop", verify_within=10.0, t=0.0).detail["verification_goal_id"]
        a.last_satisfied[vid] = 3.0  # the effect was confirmed at t=3
        self.assertEqual(a.check_deadlines(t=11.0), [])  # confirmed in time -> silent


class TestPerSourceConsent(unittest.TestCase):
    def test_acting_scoped_to_consented_source(self):
        a = _agent(
            ConsentProfile(
                allow_computer_use=True,
                allowed_actions=frozenset({"press_stop"}),
                allowed_sources=frozenset({"cam-1"}),
            )
        )
        _act_goal(a, gid="g1", false_fire=None, source="cam-1")
        _act_goal(a, gid="g2", false_fire=None, source="cam-2")
        self.assertTrue(a.act("g1", "press_stop", t=0.0).allowed)  # consented source
        with self.assertRaises(ConsentDenied):
            a.act("g2", "press_stop", t=0.0)  # un-consented source -> denied (no cross-node bind)

    def test_empty_allowed_sources_is_backward_compatible(self):
        a = _agent()  # PERMIT has no allowed_sources -> no source restriction
        _act_goal(a, gid="g", false_fire=None, source="any-source")
        self.assertTrue(a.act("g", "press_stop", t=0.0).allowed)


if __name__ == "__main__":
    unittest.main()
