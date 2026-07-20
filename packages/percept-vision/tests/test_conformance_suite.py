"""L1: the PUBLISHED Vision-adapter conformance suite.

Two directions, both load-bearing:

1. **It passes what conforms** — the in-core double and every shipping vendor adapter's SHAPE.
2. **It fails what does not** — one deliberately broken adapter per invariant. A conformance suite
   that only ever passes is theatre; each check below is proved to actually bite.
"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from percept import conformance
from percept.backends.fake import FakeVision
from percept.contracts import Box, Goal, Observation, Verdict


class _Conforming:
    """A minimal, fully conformant third-party adapter (the copy-paste skeleton in the docs)."""

    async def judge(
        self,
        condition: str,
        frame,
        *,
        roi=None,
        exemplar=None,
        corrections=None,
        frames=None,
        audio=None,
        context=None,
    ) -> Verdict:
        return Verdict(
            satisfied=False,
            confidence=0.5,
            evidence="stub",
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    async def detect(self, frame, goals) -> list:
        return [Observation(label="thing", box=Box(0.1, 0.1, 0.2, 0.2), confidence=0.9)]

    async def detect_events(self, frames, goals) -> list:
        return []


def make_conforming() -> _Conforming:
    """A zero-arg factory — one of the target shapes ``load_target`` must accept."""
    return _Conforming()


#: An already-constructed adapter instance — the other accepted target shape.
CONFORMING_INSTANCE = _Conforming()


def _run(adapter, **kw):
    kw.setdefault("check_injection", False)
    return conformance.check_vision_adapter(adapter, **kw)


class TestPassesConformingAdapters(unittest.TestCase):
    def test_minimal_third_party_adapter_is_conformant(self):
        report = _run(_Conforming())
        self.assertTrue(report.ok, report.format())
        # every advisory check passes too for the reference skeleton
        self.assertEqual([r.name for r in report.failures], [])

    def test_in_core_double_is_conformant(self):
        report = _run(FakeVision())
        self.assertTrue(report.ok, report.format())

    def test_shipping_vendor_adapters_pass_the_static_checks(self):
        # Vendor SDKs/keys are absent on the L1 lane, so only the SHAPE is checkable here — but the
        # shape is exactly what a third party copies, so a drift would be a published-contract break.
        from percept import registry

        for name in ("claude", "gemini", "openrouter"):
            with self.subTest(backend=name):
                adapter = registry.make_vision(name)
                report = conformance.check_vision_adapter(
                    adapter, live_calls=False, check_injection=False
                )
                self.assertTrue(report.ok, report.format())
                self.assertFalse(report.complete)  # honest: required checks were skipped

    def test_report_is_partial_not_conformant_when_required_checks_skip(self):
        report = conformance.check_vision_adapter(
            _Conforming(), live_calls=False, check_injection=False
        )
        self.assertTrue(report.ok)
        self.assertFalse(report.complete)
        self.assertIn("PARTIALLY CHECKED", report.format())
        # every call-based check is accounted for, never silently absent
        for name, _sev in conformance.CALL_CHECKS + conformance.INJECTION_CHECKS:
            self.assertEqual(report.get(name).status, conformance.SKIP, name)


class TestCatchesBrokenAdapters(unittest.TestCase):
    def _fails(self, adapter, check_name):
        report = _run(adapter)
        result = report.get(check_name)
        self.assertIsNotNone(result, f"{check_name} missing from report:\n{report.format()}")
        self.assertEqual(result.status, conformance.FAIL, report.format())
        return report, result

    def test_missing_method_fails_and_short_circuits(self):
        class NoDetect(_Conforming):
            detect = None

        report, _ = self._fails(NoDetect(), "vision.methods_present")
        self.assertFalse(report.ok)
        self.assertEqual(report.get("vision.runtime_protocol").status, conformance.FAIL)
        # the call checks are SKIPPED (not silently missing) once the method set is wrong
        self.assertEqual(
            report.get("judge.minimal_call_returns_verdict").status, conformance.SKIP
        )

    def test_sync_judge_fails(self):
        class SyncJudge(_Conforming):
            def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                      frames=None, audio=None, context=None):
                return Verdict(satisfied=False, confidence=0.1)

        _, result = self._fails(SyncJudge(), "judge.is_coroutine_function")
        self.assertIn("coroutine", result.detail)

    def test_judge_missing_escalation_kwarg_fails(self):
        class NoContext:
            async def judge(self, condition, frame, *, roi=None, exemplar=None,
                            corrections=None, frames=None, audio=None):
                return Verdict(satisfied=False, confidence=0.1)

            async def detect(self, frame, goals):
                return []

            async def detect_events(self, frames, goals):
                return []

        _, result = self._fails(NoContext(), "judge.accepts_optional_escalations")
        self.assertIn("context", result.detail)

    def test_judge_kwarg_with_non_none_default_fails(self):
        class BadDefault(_Conforming):
            async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                            frames=(), audio=None, context=None):
                return Verdict(satisfied=False, confidence=0.1)

        _, result = self._fails(BadDefault(), "judge.accepts_optional_escalations")
        self.assertIn("frames", result.detail)

    def test_var_keyword_judge_is_accepted(self):
        class Kwargs(_Conforming):
            async def judge(self, condition, frame, **kwargs):
                return Verdict(satisfied=True, confidence=0.6, usage={"cost_usd": 0.0})

        report = _run(Kwargs())
        self.assertTrue(report.ok, report.format())

    def test_non_verdict_return_fails(self):
        class DuckTyped(_Conforming):
            async def judge(self, condition, frame, **kwargs):
                class Fake:
                    satisfied = True
                    confidence = 0.9

                return Fake()

        _, result = self._fails(DuckTyped(), "judge.minimal_call_returns_verdict")
        self.assertIn("Verdict", result.detail)

    def test_unnormalised_confidence_fails(self):
        class Logit(_Conforming):
            async def judge(self, condition, frame, **kwargs):
                return Verdict(satisfied=True, confidence=7.4)

        _, result = self._fails(Logit(), "judge.minimal_call_returns_verdict")
        self.assertIn("[0.0, 1.0]", result.detail)

    def test_non_bool_satisfied_fails(self):
        class IntSatisfied(_Conforming):
            async def judge(self, condition, frame, **kwargs):
                return Verdict(satisfied=1, confidence=0.5)  # type: ignore[arg-type]

        _, result = self._fails(IntSatisfied(), "judge.minimal_call_returns_verdict")
        self.assertIn("bool", result.detail)

    def test_raising_on_an_escalation_fails_that_check_only(self):
        class NoRoi(_Conforming):
            async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                            frames=None, audio=None, context=None):
                if roi is not None:
                    raise NotImplementedError("no crop support")
                return Verdict(satisfied=False, confidence=0.5)

        report, result = self._fails(NoRoi(), "judge.escalation_roi")
        self.assertIn("NotImplementedError", result.detail)
        self.assertFalse(report.ok)  # roi is REQUIRED
        self.assertEqual(report.get("judge.minimal_call_returns_verdict").status, conformance.PASS)

    def test_raising_on_an_optional_modality_is_advisory_only(self):
        class NoAudio(_Conforming):
            async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                            frames=None, audio=None, context=None):
                if audio is not None:
                    raise TypeError("this backend has no ears")
                return Verdict(satisfied=False, confidence=0.5)

        report = _run(NoAudio())
        self.assertEqual(report.get("judge.escalation_audio").status, conformance.FAIL)
        self.assertEqual(report.get("judge.escalation_audio").severity, conformance.ADVISORY)
        self.assertTrue(report.ok, "an absent MODALITY must not make an adapter non-conformant")

    def test_pixel_boxes_from_detect_fail(self):
        class PixelBoxes(_Conforming):
            async def detect(self, frame, goals):
                return [Observation(label="person", box=Box(120, 40, 300, 220), confidence=0.8)]

        _, result = self._fails(PixelBoxes(), "detect.returns_normalised_observations")
        self.assertIn("NORMALISED", result.detail)

    def test_detect_events_returning_non_list_fails(self):
        class BadEvents(_Conforming):
            async def detect_events(self, frames, goals):
                return None

        self._fails(BadEvents(), "detect_events.returns_list")

    def test_missing_usage_is_advisory_only(self):
        report = _run(FakeVision())
        usage = report.get("judge.reports_usage")
        self.assertEqual(usage.status, conformance.FAIL)
        self.assertEqual(usage.severity, conformance.ADVISORY)
        self.assertTrue(report.ok)


class TestFailClosedInjection(unittest.TestCase):
    """The seam's other half: your adapter is the one that runs, or the core refuses to run."""

    def test_injected_instance_survives_and_live_core_is_fail_closed(self):
        report = conformance.check_vision_adapter(_Conforming(), check_injection=True)
        self.assertEqual(report.get("injection.instance_survives_create").status, conformance.PASS)
        self.assertEqual(report.get("injection.live_core_is_fail_closed").status, conformance.PASS)
        self.assertEqual(report.get("injection.not_a_scripted_double").status, conformance.PASS)
        self.assertTrue(report.ok, report.format())

    def test_in_core_double_is_flagged_as_scripted_but_only_advisory(self):
        report = conformance.check_vision_adapter(FakeVision(), check_injection=True)
        scripted = report.get("injection.not_a_scripted_double")
        self.assertEqual(scripted.status, conformance.FAIL)
        self.assertEqual(scripted.severity, conformance.ADVISORY)
        self.assertTrue(report.ok)

    def test_live_core_with_no_vision_really_raises_mode_error(self):
        # The property the conformance check asserts, asserted directly — so a regression in the
        # CORE (not the suite) is attributable.
        from percept.agent import ModeError, Percept

        with self.assertRaises(ModeError):
            Percept.create(mode="live", discover_plugins=False)


class TestCliAndLoader(unittest.TestCase):
    def test_load_target_accepts_class_and_instance_and_factory(self):
        self.assertIsInstance(
            conformance.load_target("percept.backends.fake:FakeVision"), FakeVision
        )
        # a zero-arg factory function
        self.assertIsInstance(
            conformance.load_target("tests.test_conformance_suite:make_conforming"), _Conforming
        )
        # an already-constructed instance is returned as-is
        self.assertIs(
            conformance.load_target("tests.test_conformance_suite:CONFORMING_INSTANCE"),
            CONFORMING_INSTANCE,
        )

    def test_load_target_rejects_a_bad_spec(self):
        with self.assertRaises(ValueError):
            conformance.load_target("percept.backends.fake.FakeVision")

    def test_cli_exit_zero_for_a_conformant_adapter(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = conformance.main(["percept.backends.fake:FakeVision"])
        self.assertEqual(code, 0)
        self.assertIn("CONFORMANT", out.getvalue())

    def test_cli_exit_two_for_an_unloadable_target(self):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = conformance.main(["no.such.module:Nope"])
        self.assertEqual(code, 2)
        self.assertIn("could not load", err.getvalue())

    def test_cli_static_mode_makes_no_calls(self):
        calls = []

        class Counting(_Conforming):
            async def judge(self, condition, frame, **kwargs):  # pragma: no cover — must not run
                calls.append(condition)
                return Verdict(satisfied=False, confidence=0.0)

        import percept.conformance as mod

        mod._cli_probe_adapter = Counting()  # type: ignore[attr-defined]
        try:
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                code = mod.main(["--static", "percept.conformance:_cli_probe_adapter"])
            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
            self.assertIn("PARTIALLY CHECKED", out.getvalue())
        finally:
            del mod._cli_probe_adapter  # type: ignore[attr-defined]


class TestSuiteMetadata(unittest.TestCase):
    def test_versions_are_semver_strings(self):
        for v in (conformance.CONFORMANCE_VERSION, conformance.CONTRACT_VERSION):
            self.assertRegex(v, r"^\d+\.\d+\.\d+$")

    def test_documented_escalations_match_the_frozen_judge_signature(self):
        # The suite must not drift from the seam it publishes: every kwarg it demands is really on
        # Vision.judge, and every optional kwarg on Vision.judge is really demanded.
        import inspect

        from percept.contracts import Vision

        sig = inspect.signature(Vision.judge)
        optional = tuple(
            n
            for n, p in sig.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
        )
        self.assertEqual(set(optional), set(conformance.JUDGE_OPTIONAL_KWARGS))
        self.assertEqual(set(conformance.ESCALATION_SEVERITY), set(conformance.JUDGE_OPTIONAL_KWARGS))

    def test_goal_probe_is_a_real_goal(self):
        g = Goal(id="conformance", condition=conformance.PROBE_CONDITION, say="x")
        self.assertTrue(g.condition)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
