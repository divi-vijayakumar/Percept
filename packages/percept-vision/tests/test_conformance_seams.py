"""L1: conformance WIDENED past the Vision seam.

the implementation settled which extension seams survive; this suite covers the two that are not the cloud
judge, in the SAME reporting shape:

* ``check_detector`` — the bring-your-own ADMISSION detector (``register_detector`` + a
  ``DetectorManifest``), the seam a client plugs an OCR / phash model into for screen content.
* ``check_tier0_adapter`` — the ``structural_adapter`` / ``semantic_adapter`` callables ``run()``
  now accepts.

Same two directions as the Vision suite, both load-bearing: it passes what conforms (every shipping
detector), and it FAILS what does not — one deliberately broken plug-in per invariant, because a
conformance suite that only ever passes is theatre. The standing rule for this suite is:
the honest scope disclaimer is on EVERY report, for EVERY role.
"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from percept import conformance
from percept.harness import detectors as det
from percept.harness.tier0 import Frame, gray_frame


# --- fixtures: a conformant detector, then one break per invariant --------------------------------


class GoodDetector:
    """A minimal, fully conformant admission detector — the copy-paste skeleton.

    Stands in for a client's OCR "the document changed" detector: it emits a boundary whenever the
    signal crosses 0.5 upward. Pure, deterministic, resettable."""

    signal = "change"  # deliberately NOT a routable channel in the broken variant below
    name = "good-detector"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev = 0.0

    def push(self, t: float, v: float) -> list:
        out = [det.Boundary(self.name, "threshold", t, v)] if v >= 0.5 > self._prev else []
        self._prev = v
        return out


class Routable(GoodDetector):
    signal = "motion"


MANIFEST = det.DetectorManifest(
    name="good-detector",
    senses="a cheap 0..1 change score per frame",
    when_to_use="admission on a stream whose pixels barely move (a document/screen)",
    output="a 'threshold' boundary when the score crosses 0.5",
    category=det.CATEGORY_CHANGE,
)


def _check(factory=Routable, **kw):
    kw.setdefault("manifest", MANIFEST)
    return conformance.check_detector(factory, **kw)


def _result(report, name):
    r = report.get(name)
    assert r is not None, f"{name} missing from report:\n{report.format()}"
    return r


class TestDetectorPassesWhatConforms(unittest.TestCase):
    def test_reference_detector_is_conformant(self):
        report = _check()
        self.assertTrue(report.ok, report.format())
        self.assertTrue(report.complete, report.format())
        self.assertEqual([r.name for r in report.failures], [], report.format())
        self.assertEqual(report.role, conformance.ROLE_DETECTOR)

    def test_every_shipping_scalar_detector_is_conformant(self):
        for name in ("motion-periodicity", "acoustic-onset"):
            with self.subTest(name):
                factory = det._REGISTRY[name][0]
                report = conformance.check_detector(factory, name=name)
                self.assertTrue(report.ok, report.format())
                self.assertTrue(report.complete, report.format())

    def test_frame_backed_detector_skips_the_push_checks_instead_of_faking_a_frame(self):
        report = conformance.check_detector(det.PoseChannels, name="pose-channels")
        self.assertTrue(report.ok, report.format())
        self.assertFalse(report.complete)  # honest partial, never a silent pass
        skipped = {r.name for r in report.skipped}
        self.assertEqual(skipped, {n for n, _ in conformance.DETECTOR_CALL_CHECKS})
        self.assertIn("PARTIALLY CHECKED", report.format())

    def test_caller_supplied_samples_unlock_a_frame_backed_detector(self):
        frames = [gray_frame(2, 2, 7), gray_frame(2, 2, 9)]

        class FrameDetector(GoodDetector):
            signal = "frame"

            def push(self, t, frame):
                return [det.Boundary(self.name, "sample", t, float(frame.pixels[0]))]

        report = conformance.check_detector(
            FrameDetector, samples=[(0.0, frames[0]), (0.1, frames[1])], manifest=MANIFEST
        )
        self.assertTrue(report.ok, report.format())
        self.assertTrue(report.complete, report.format())

    def test_static_only_run_skips_every_push_check(self):
        report = _check(live_calls=False)
        self.assertTrue(report.ok, report.format())
        self.assertFalse(report.complete)
        self.assertEqual(
            {r.name for r in report.skipped}, {n for n, _ in conformance.DETECTOR_CALL_CHECKS}
        )


class TestDetectorFailsWhatDoesNot(unittest.TestCase):
    def assertFails(self, report, name):
        r = _result(report, name)
        self.assertEqual(r.status, conformance.FAIL, report.format())
        return r

    def test_unroutable_signal_channel_fails(self):
        # `signal = "change"` is not one of motion/energy/frame; DetectorBank would silently route it
        # to MOTION. That silence is exactly the production failure this suite prevents.
        report = _check(GoodDetector)
        self.assertFails(report, "detector.signal_channel_is_routable")
        self.assertFalse(report.ok)

    def test_missing_push_fails_and_skips_the_call_checks(self):
        class NoPush:
            signal = "motion"

            def reset(self):
                pass

        report = _check(NoPush)
        self.assertFails(report, "detector.push_and_reset_present")
        self.assertFalse(report.ok)
        for name, _sev in conformance.DETECTOR_CALL_CHECKS:
            self.assertEqual(_result(report, name).status, conformance.SKIP)

    def test_async_push_fails(self):
        class AsyncPush(Routable):
            async def push(self, t, v):  # the gate is synchronous — this silently emits nothing
                return []

        self.assertFails(_check(AsyncPush), "detector.push_and_reset_present")

    def test_factory_needing_arguments_fails(self):
        class NeedsArgs:
            signal = "motion"

            def __init__(self, cfg):
                self.cfg = cfg

            def reset(self):
                pass

            def push(self, t, v):
                return []

        report = _check(NeedsArgs)
        self.assertFails(report, "detector.factory_constructs")

    def test_non_list_push_return_fails(self):
        class ScalarReturn(Routable):
            def push(self, t, v):
                return det.Boundary(self.name, "threshold", t, v)  # not a list

        self.assertFails(_check(ScalarReturn), "detector.push_returns_boundaries")

    def test_non_boundary_element_fails(self):
        class DictBoundary(Routable):
            def push(self, t, v):
                return [{"detector": self.name, "kind": "threshold", "t": t, "value": v}]

        self.assertFails(_check(DictBoundary), "detector.push_returns_boundaries")

    def test_non_finite_timestamp_fails(self):
        class NanT(Routable):
            def push(self, t, v):
                return [det.Boundary(self.name, "threshold", float("nan"), v)]

        self.assertFails(_check(NanT), "detector.push_returns_boundaries")

    def test_unattributed_boundary_fails(self):
        class Anonymous(Routable):
            def push(self, t, v):
                return [det.Boundary("", "threshold", t, v)]

        self.assertFails(_check(Anonymous), "detector.boundaries_are_attributed")

    def test_boundaries_split_across_two_names_fail(self):
        class TwoFaced(Routable):
            def push(self, t, v):
                return [det.Boundary(self.name if v > 0.5 else "other", "threshold", t, v)]

        self.assertFails(_check(TwoFaced), "detector.boundaries_are_attributed")

    def test_a_name_the_registry_does_not_know_is_advisory_only(self):
        # functionally harmless (the runtime maps boundaries by goal_id) but it makes every trace row
        # name something the client's plan never mentions — reported, never blocking
        class Impostor(Routable):
            def push(self, t, v):
                return [det.Boundary("some-other-detector", "threshold", t, v)]

        report = _check(Impostor)
        r = _result(report, "detector.attribution_matches_registry_name")
        self.assertEqual(r.status, conformance.FAIL)
        self.assertEqual(r.severity, conformance.ADVISORY)
        self.assertTrue(report.ok, report.format())

    def test_detector_that_dies_on_a_flat_signal_fails(self):
        class DividesByRange(Routable):
            def __init__(self):
                super().__init__()
                self._lo, self._hi = 1.0, 0.0

            def push(self, t, v):
                self._lo, self._hi = min(self._lo, v), max(self._hi, v)
                _ = (v - self._lo) / (self._hi - self._lo)  # ZeroDivisionError on a flat stream
                return []

        self.assertFails(_check(DividesByRange), "detector.tolerates_degenerate_signal")

    def test_nondeterministic_detector_fails(self):
        counter = {"n": 0}

        class Drifting(Routable):
            def push(self, t, v):
                counter["n"] += 1  # process-global state: run 2 disagrees with run 1
                return [det.Boundary(self.name, "threshold", t, float(counter["n"]))]

        self.assertFails(_check(Drifting), "detector.deterministic_on_fixed_input")

    def test_reset_that_does_not_reset_fails(self):
        class StickyReset(Routable):
            def __init__(self):
                self._fired = False

            def reset(self):
                pass  # forgets to clear _fired

            def push(self, t, v):
                if self._fired:
                    return []
                self._fired = True
                return [det.Boundary(self.name, "threshold", t, v)]

        report = _check(StickyReset)
        self.assertFails(report, "detector.reset_restores_initial_state")

    def test_manifest_of_the_wrong_type_fails(self):
        report = _check(Routable, manifest={"name": "good-detector"})
        self.assertFails(report, "detector.manifest_well_formed")

    def test_empty_manifest_fields_are_advisory_only(self):
        report = conformance.check_detector(Routable, name="bare")
        for name in ("detector.manifest_is_descriptive", "detector.manifest_declares_category"):
            r = _result(report, name)
            self.assertEqual(r.status, conformance.FAIL)
            self.assertEqual(r.severity, conformance.ADVISORY)
        self.assertTrue(report.ok, report.format())  # advisory never blocks


class TestDetectorSelectability(unittest.TestCase):
    """The selectability check drives the REAL path: register -> firewall -> compile_watchspec."""

    def test_selectability_passes_and_leaves_the_registry_untouched(self):
        before = det.registered()
        report = _check()
        self.assertEqual(_result(report, "detector.selectable_by_name_into_watchspec").status,
                         conformance.PASS)
        self.assertEqual(det.registered(), before)
        self.assertNotIn("good-detector", det.registered())

    def test_a_pre_existing_registration_is_restored_not_clobbered(self):
        det.register("good-detector", Routable, MANIFEST)
        try:
            _check()
            self.assertIn("good-detector", det.registered())
            self.assertIs(det.manifest_of("good-detector"), MANIFEST)
        finally:
            det.unregister("good-detector")

    def test_the_check_really_reaches_compile_watchspec(self):
        # Sanity on the bar itself: a name the firewall drops must not be reported selectable.
        report = _check()
        self.assertTrue(report.ok, report.format())
        self.assertEqual(det.coerce_authored_detectors(["good-detector"]), [])  # gone again


# --- the tier-0 adapter role ----------------------------------------------------------------------


def _mean_luma(frame) -> float:
    return sum(frame.pixels) / (255.0 * len(frame.pixels))


class TestTier0AdapterRole(unittest.TestCase):
    def test_a_bounded_adapter_is_conformant_in_both_roles(self):
        for role in conformance.TIER0_ROLES:
            with self.subTest(role):
                report = conformance.check_tier0_adapter(_mean_luma, role=role)
                self.assertTrue(report.ok, report.format())
                self.assertTrue(report.complete, report.format())
                self.assertEqual(report.role, role)

    def test_run_really_exposes_both_adapter_parameters(self):
        # the drift guard: this check is only meaningful if it would notice the seam disappearing
        import inspect

        from percept.runtime import run

        params = inspect.signature(run).parameters
        for role in conformance.TIER0_ROLES:
            self.assertIn(f"{role}_adapter", params)

    def test_unnormalised_score_fails(self):
        report = conformance.check_tier0_adapter(lambda f: 42.0, role="structural")
        self.assertEqual(
            _result(report, "adapter.returns_bounded_score").status, conformance.FAIL
        )
        self.assertFalse(report.ok)

    def test_nan_score_fails(self):
        report = conformance.check_tier0_adapter(lambda f: float("nan"))
        self.assertEqual(
            _result(report, "adapter.returns_bounded_score").status, conformance.FAIL
        )

    def test_non_numeric_score_fails(self):
        report = conformance.check_tier0_adapter(lambda f: "0.5")
        self.assertEqual(
            _result(report, "adapter.returns_bounded_score").status, conformance.FAIL
        )

    def test_adapter_that_dies_on_a_degenerate_frame_fails(self):
        def brittle(frame) -> float:
            lo, hi = min(frame.pixels), max(frame.pixels)
            return (frame.pixels[0] - lo) / (hi - lo)  # uniform frame -> ZeroDivisionError

        report = conformance.check_tier0_adapter(brittle, role="semantic")
        self.assertEqual(
            _result(report, "adapter.tolerates_degenerate_frame").status, conformance.FAIL
        )
        self.assertFalse(report.ok)

    def test_an_adapter_that_raises_inside_the_gate_fails_loudly(self):
        seen = {"n": 0}

        def dies_on_the_fourth_frame(frame) -> float:
            seen["n"] += 1
            if seen["n"] > 3:
                raise RuntimeError("model session expired")
            return 0.4

        report = conformance.check_tier0_adapter(dies_on_the_fourth_frame)
        self.assertEqual(
            _result(report, "adapter.does_not_crash_the_gate").status, conformance.FAIL
        )
        self.assertEqual(
            _result(report, "adapter.score_reaches_the_tier0_signal").status, conformance.SKIP
        )

    def test_non_callable_target_short_circuits(self):
        report = conformance.check_tier0_adapter(0.7)
        self.assertEqual(_result(report, "adapter.is_callable").status, conformance.FAIL)
        for name, _sev in conformance.TIER0_CALL_CHECKS:
            self.assertEqual(_result(report, name).status, conformance.SKIP)

    def test_wrong_arity_fails(self):
        report = conformance.check_tier0_adapter(lambda frame, model: 0.5)
        self.assertEqual(_result(report, "adapter.is_callable").status, conformance.FAIL)

    def test_stochastic_adapter_is_advisory_only(self):
        vals = iter([0.1, 0.2] * 50)

        report = conformance.check_tier0_adapter(lambda f: next(vals))
        r = _result(report, "adapter.is_deterministic")
        self.assertEqual(r.status, conformance.FAIL)
        self.assertEqual(r.severity, conformance.ADVISORY)

    def test_the_score_must_be_the_score_the_signal_reports(self):
        # a real MotionGate carries the adapter; the reported Tier0Signal channel IS its output
        report = conformance.check_tier0_adapter(_mean_luma, role="structural")
        self.assertEqual(
            _result(report, "adapter.score_reaches_the_tier0_signal").status, conformance.PASS
        )

    def test_caller_frames_suppress_the_probe_note(self):
        frames = [Frame(w=2, h=2, pixels=(0, 10, 20, 30)), gray_frame(2, 2, 200)]
        report = conformance.check_tier0_adapter(_mean_luma, frames=frames)
        self.assertFalse(report.frame_was_probe)
        self.assertNotIn("no frames= were supplied", report.format())
        self.assertTrue(report.ok, report.format())

    def test_unknown_role_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            conformance.check_tier0_adapter(_mean_luma, role="motion")


# --- the shared reporting shape + the honest scope disclaimer -------------------------------------


class TestOneReportingShape(unittest.TestCase):
    def test_every_role_prints_the_scope_disclaimer(self):
        reports = [
            _check(),
            conformance.check_tier0_adapter(_mean_luma, role="structural"),
            conformance.check_tier0_adapter(_mean_luma, role="semantic"),
        ]
        for report in reports:
            with self.subTest(report.role):
                text = report.format()
                self.assertIn("NOT a measure of the model", text)
                self.assertIn("percept eval", text)
                self.assertIn(conformance.ROLE_TITLES[report.role], text)

    def test_the_vision_role_keeps_the_disclaimer_too(self):
        from percept.backends.fake import FakeVision

        report = conformance.check_vision_adapter(FakeVision(), check_injection=False)
        self.assertEqual(report.role, conformance.ROLE_VISION)
        self.assertIn("NOT a measure of the model", report.format())

    def test_new_roles_reuse_the_existing_result_types(self):
        report = _check()
        self.assertIsInstance(report, conformance.ConformanceReport)
        for r in report.results:
            self.assertIsInstance(r, conformance.CheckResult)
            self.assertIn(r.severity, (conformance.REQUIRED, conformance.ADVISORY))
            self.assertIn(r.status, (conformance.PASS, conformance.FAIL, conformance.SKIP))

    def test_declared_check_names_match_what_the_suite_emits(self):
        names = [r.name for r in _check().results]
        declared = [n for n, _ in conformance.DETECTOR_STATIC_CHECKS + conformance.DETECTOR_CALL_CHECKS]
        self.assertEqual(names, declared)

        names = [r.name for r in conformance.check_tier0_adapter(_mean_luma).results]
        declared = [n for n, _ in conformance.TIER0_STATIC_CHECKS + conformance.TIER0_CALL_CHECKS]
        self.assertEqual(names, declared)

    def test_suite_version_is_semver(self):
        parts = conformance.CONFORMANCE_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.isdigit() for p in parts))


# --- the ONE CLI ----------------------------------------------------------------------------------


class TestCliCoversTheNewRoles(unittest.TestCase):
    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = conformance.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_detector_role_exit_zero(self):
        code, out, _ = self._main(
            ["percept.harness.detectors:MotionPeriodicity", "--role", "detector"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("admission detector conformance", out)
        self.assertIn("NOT a measure of the model", out)

    def test_detector_role_exit_one_for_a_broken_detector(self):
        code, out, _ = self._main(
            ["tests.test_conformance_seams:GoodDetector", "--role", "detector"]
        )
        self.assertEqual(code, 1, out)
        self.assertIn("detector.signal_channel_is_routable", out)

    def test_detector_name_override(self):
        code, out, _ = self._main(
            ["tests.test_conformance_seams:Routable", "--role", "detector", "--name", "ocr-doc"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("selectable as 'ocr-doc'", out)
        self.assertNotIn("ocr-doc", det.registered())

    def test_tier0_role_exit_zero_and_is_not_instantiated_by_load_target(self):
        code, out, _ = self._main(
            ["tests.test_conformance_seams:_mean_luma", "--role", "semantic"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("tier-0 semantic_adapter conformance", out)

    def test_static_mode_makes_no_calls_into_a_detector(self):
        code, out, _ = self._main(
            ["tests.test_conformance_seams:Exploding", "--role", "detector", "--static"]
        )
        self.assertEqual(code, 0, out)
        self.assertIn("PARTIALLY CHECKED", out)

    def test_unloadable_target_exits_two(self):
        code, _out, err = self._main(["tests.test_conformance_seams:nope", "--role", "detector"])
        self.assertEqual(code, 2)
        self.assertIn("could not load", err)

    def test_load_target_can_return_the_factory_itself(self):
        obj = conformance.load_target(
            "tests.test_conformance_seams:Routable", instantiate=False
        )
        self.assertIs(obj, Routable)
        self.assertIsInstance(
            conformance.load_target("tests.test_conformance_seams:Routable"), Routable
        )


class Exploding(Routable):
    """A detector that raises the moment it is pushed — proves ``--static`` never calls it."""

    def push(self, t, v):
        raise RuntimeError("this detector must not be called in static mode")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
