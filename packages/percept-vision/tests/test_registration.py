"""The stable detector and concern registration API.

Deterministic stdlib tests (no model, no keys):
  * register_detector façade — the ergonomic twin of register(); manifest derived or supplied.
  * detector teardown        — unregister() / clear_detectors() drop a skill from selection.
  * register_concern         — a registered Concern is returned / fetchable by id.
  * concern -> WatchSpec      — a registered concern compiles into a WatchCondition (anchor + detectors).
  * concern teardown          — unregister_concern() / clear_concerns() drop it from compilation.
"""

from __future__ import annotations

import unittest

from percept.adapters import Concern, Goal, Subject, WatchSpec

from percept.harness.concerns import (
    clear_concerns,
    compile_registered_watchspec,
    concern_to_condition,
    detectors_for_concern,
    get_concern,
    register_concern,
    registered_concern_ids,
    registered_concerns,
    unregister_concern,
)
from percept.harness.detectors import (
    DetectorManifest,
    MotionPeriodicity,
    clear_detectors,
    make_detector,
    manifest_of,
    register,
    register_detector,
    registered,
    unregister,
)


def _concern(cid="c1", condition="milk is boiling over", **extra) -> Concern:
    c = Concern(
        id=cid,
        identity=Subject(kind="entity", ref=condition),
        condition=condition,
        say="heads up",
    )
    for k, v in extra.items():
        setattr(c, k, v)
    return c


class TestRegisterDetectorFacade(unittest.TestCase):
    """register_detector(name, detector) sits beside register(impl, manifest); both interoperate."""

    def tearDown(self):
        unregister("test-skill")

    def test_facade_registers_and_makes_detector(self):
        m = register_detector(
            "test-skill",
            MotionPeriodicity,
            senses="a test signal",
            when_to_use="in a test",
            output="a test boundary",
        )
        self.assertIn("test-skill", registered())
        self.assertIsInstance(make_detector("test-skill"), MotionPeriodicity)
        # the derived manifest is what the brain reads to compose it
        self.assertEqual(m.name, "test-skill")
        self.assertEqual(manifest_of("test-skill").when_to_use, "in a test")

    def test_facade_minimal_call_derives_empty_manifest(self):
        register_detector("test-skill", MotionPeriodicity)
        m = manifest_of("test-skill")
        self.assertEqual(m.name, "test-skill")
        self.assertEqual((m.senses, m.when_to_use, m.output), ("", "", ""))

    def test_facade_accepts_explicit_manifest(self):
        man = DetectorManifest(name="test-skill", senses="s", when_to_use="w", output="o")
        register_detector("ignored-name", MotionPeriodicity, manifest=man)
        # the manifest's own name is authoritative when supplied
        self.assertIn("test-skill", registered())
        self.assertEqual(manifest_of("test-skill").senses, "s")

    def test_facade_and_register_produce_identical_entries(self):
        register("test-skill", MotionPeriodicity,
                 DetectorManifest(name="test-skill", senses="s", when_to_use="w", output="o"))
        d1 = make_detector("test-skill")
        unregister("test-skill")
        register_detector("test-skill", MotionPeriodicity, senses="s", when_to_use="w", output="o")
        d2 = make_detector("test-skill")
        self.assertEqual(type(d1), type(d2))


class TestDetectorTeardown(unittest.TestCase):
    """unregister() / clear_detectors() are the teardown half — built-ins restore on re-import."""

    def tearDown(self):
        # restore the registry to its imported state for the rest of the suite
        import importlib

        import percept.harness.detectors as det

        importlib.reload(det)

    def test_unregister_drops_a_skill(self):
        register_detector("ephemeral", MotionPeriodicity)
        self.assertIn("ephemeral", registered())
        self.assertTrue(unregister("ephemeral"))
        self.assertNotIn("ephemeral", registered())
        self.assertFalse(unregister("ephemeral"))  # idempotent: gone -> False

    def test_clear_detectors_empties_registry(self):
        self.assertTrue(registered())  # built-ins present
        clear_detectors()
        self.assertEqual(registered(), [])


class TestConcernRegistry(unittest.TestCase):
    def setUp(self):
        clear_concerns()

    def tearDown(self):
        clear_concerns()

    def test_register_and_fetch(self):
        c = _concern("baby")
        self.assertIs(register_concern(c), c)  # returns the concern
        self.assertEqual(registered_concern_ids(), ["baby"])
        self.assertIs(get_concern("baby"), c)
        self.assertEqual([x.id for x in registered_concerns()], ["baby"])

    def test_register_accepts_a_goal_via_compat_shim(self):
        register_concern(Goal(id="g1", condition="door open", say="hey"))
        c = get_concern("g1")
        self.assertIsInstance(c, Concern)
        self.assertEqual(c.condition, "door open")

    def test_reregister_same_id_replaces(self):
        register_concern(_concern("dup", condition="first"))
        register_concern(_concern("dup", condition="second"))
        self.assertEqual(registered_concern_ids(), ["dup"])
        self.assertEqual(get_concern("dup").condition, "second")

    def test_get_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_concern("nope")


class TestConcernSelectedIntoWatchSpec(unittest.TestCase):
    """The lifecycle payoff: a registered concern compiles DOWN into a WatchSpec."""

    def setUp(self):
        clear_concerns()

    def tearDown(self):
        clear_concerns()

    def test_registered_concern_compiles_into_watchspec(self):
        register_concern(_concern("milk", condition="milk is boiling over"))
        spec = compile_registered_watchspec(threshold=0.4)
        self.assertIsInstance(spec, WatchSpec)
        self.assertEqual(len(spec.conditions), 1)
        cond = spec.conditions[0]
        self.assertEqual(cond.goal_id, "milk")
        self.assertEqual(cond.anchor_text, "milk is boiling over")
        self.assertEqual(cond.threshold, 0.4)

    def test_concern_detectors_select_into_condition(self):
        # a concern annotated with a detector override selects it (validated against the registry)
        c = _concern("reps", condition="count squats",
                     detectors=["motion-periodicity", "bogus"])
        self.assertEqual(detectors_for_concern(c), ["motion-periodicity"])  # firewall drops bogus
        cond = concern_to_condition(c, threshold=0.2)
        self.assertEqual(cond.detectors, ["motion-periodicity"])

    def test_capability_handshake_filters_concern_detectors(self):
        c = _concern("reps", condition="count squats",
                     detectors=["motion-periodicity", "acoustic-onset"])
        cond = concern_to_condition(c, threshold=0.2, edge_capabilities={"motion-periodicity"})
        self.assertEqual(cond.detectors, ["motion-periodicity"])  # acoustic dropped: edge can't run it

    def test_un_annotated_concern_selects_no_detectors(self):
        # byte-for-byte parity with the goal default: no annotation -> no detectors
        self.assertEqual(detectors_for_concern(_concern("plain")), [])

    def test_compile_orders_by_id_and_takes_explicit_subset(self):
        register_concern(_concern("b"))
        register_concern(_concern("a"))
        spec = compile_registered_watchspec(threshold=0.5)
        self.assertEqual([c.goal_id for c in spec.conditions], ["a", "b"])  # id-sorted
        # explicit subset bypasses the registry
        sub = compile_registered_watchspec(threshold=0.5, concerns=[_concern("z")])
        self.assertEqual([c.goal_id for c in sub.conditions], ["z"])


class TestConcernTeardown(unittest.TestCase):
    def setUp(self):
        clear_concerns()

    def tearDown(self):
        clear_concerns()

    def test_unregister_concern_drops_from_compilation(self):
        register_concern(_concern("keep"))
        register_concern(_concern("drop"))
        self.assertTrue(unregister_concern("drop"))
        spec = compile_registered_watchspec(threshold=0.3)
        self.assertEqual([c.goal_id for c in spec.conditions], ["keep"])
        self.assertFalse(unregister_concern("drop"))  # idempotent

    def test_clear_concerns_empties_registry(self):
        register_concern(_concern("x"))
        clear_concerns()
        self.assertEqual(registered_concern_ids(), [])
        self.assertEqual(compile_registered_watchspec(threshold=0.3).conditions, [])


if __name__ == "__main__":
    unittest.main()
