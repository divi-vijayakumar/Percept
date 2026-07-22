"""``percept.build_info()`` — wheel provenance is importable and ties into RunTraceManifest.

In a source checkout that never went through the packaging build step (this test suite's normal
environment), every field gracefully falls back to ``"unreleased"`` instead of raising or
fabricating a value. A real build (see ``_build_backend.py``) regenerates ``percept/_build_info.py``
with real values; that end-to-end packaging path is exercised manually (``python -m build``), not
here — this module only checks the *contract* of the accessor and its plumbing into the trace
manifest.
"""

from __future__ import annotations

import unittest

import percept
from percept._build import BuildInfo, build_info
from percept.tracing import RunTraceManifest


class BuildInfoTests(unittest.TestCase):
    def test_build_info_importable_and_non_empty(self):
        info = build_info()
        self.assertIsInstance(info, BuildInfo)
        for field_name in ("package_version", "git_commit", "contract_version", "build_timestamp"):
            value = getattr(info, field_name)
            self.assertIsInstance(value, str)
            self.assertTrue(value, f"{field_name} must not be empty")

    def test_unreleased_fallback_in_a_bare_source_checkout(self):
        # This suite runs against the source tree, not a built wheel, so the checked-in stub
        # (percept/_build_info.py) applies: every field is the literal "unreleased" sentinel,
        # never a stale or guessed value.
        info = build_info()
        self.assertEqual(info.package_version, "unreleased")
        self.assertEqual(info.git_commit, "unreleased")
        self.assertEqual(info.contract_version, "unreleased")
        self.assertEqual(info.build_timestamp, "unreleased")

    def test_build_info_never_raises(self):
        # Public contract: no exception, ever — a bad/missing stamp degrades to the fallback.
        try:
            build_info()
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"build_info() raised {exc!r}")

    def test_exported_from_top_level_package(self):
        self.assertIs(percept.build_info, build_info)
        self.assertIs(percept.BuildInfo, BuildInfo)
        self.assertIn("build_info", percept.__all__)
        self.assertIn("BuildInfo", percept.__all__)

    def test_to_dict_round_trips_the_four_fields(self):
        info = build_info()
        payload = info.to_dict()
        self.assertEqual(
            set(payload),
            {"package_version", "git_commit", "contract_version", "build_timestamp"},
        )

    def test_run_trace_manifest_carries_build_provenance(self):
        # A returned trace ties back to the exact build that produced it: RunTraceManifest.create()
        # stamps build_commit/build_timestamp from percept.build_info(), additive on top of the
        # already-recorded percept_version/contract_version.
        manifest = RunTraceManifest.create()
        info = build_info()
        self.assertEqual(manifest.build_commit, info.git_commit)
        self.assertEqual(manifest.build_timestamp, info.build_timestamp)

    def test_run_trace_manifest_round_trips_build_fields(self):
        manifest = RunTraceManifest.create()
        payload = manifest.to_dict()
        self.assertIn("build_commit", payload)
        self.assertIn("build_timestamp", payload)
        restored = RunTraceManifest.from_dict(payload)
        self.assertEqual(restored.build_commit, manifest.build_commit)
        self.assertEqual(restored.build_timestamp, manifest.build_timestamp)

    def test_run_trace_manifest_from_dict_defaults_when_absent(self):
        # Older/foreign trace payloads without the new fields must still load (additive field).
        manifest = RunTraceManifest.create()
        payload = manifest.to_dict()
        del payload["build_commit"]
        del payload["build_timestamp"]
        restored = RunTraceManifest.from_dict(payload)
        self.assertIsNone(restored.build_commit)
        self.assertIsNone(restored.build_timestamp)


if __name__ == "__main__":
    unittest.main()
