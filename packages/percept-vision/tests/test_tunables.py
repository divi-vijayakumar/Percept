"""C0: percept.tunables — registry integrity, generated-artifact parity, and Settings completeness.

Three things this file pins:

1. **Registry integrity** — :data:`percept.tunables.REGISTRY` builds with no duplicate paths and every
   spec's ``owner`` is one of :data:`percept.tunables.OWNERS` (already enforced by
   :meth:`TunableSpec.__post_init__` / :func:`_build_registry`; re-asserted here as a locked contract).
2. **Parity** — the checked-in ``tunables.json`` artifact is BYTE-IDENTICAL to what
   ``percept.tunables.generate()`` produces right now. A registry edit that forgets to regenerate the
   artifact fails this test, the same "generated file drifts from its generator" class of bug a
   schema/fixture parity test catches elsewhere in this suite.
3. **Completeness** — every field on :class:`percept.config.Settings` (the ADR-008 Context section's
   first-named configuration surface) is either a ``Settings.``-prefixed path in the registry, or
   explicitly excluded with a reason recorded in :data:`_EXCLUDED_SETTINGS_FIELDS` below. Today the
   exclusion set is empty: the salvaged registry already has exact 1:1 coverage of the ``Settings`` tree
   (verified by walking the live dataclass, not by re-typing the field list by hand — a renamed/added
   Settings field fails this test until the registry (or this exclusion list, with a reason) is updated).
"""

from __future__ import annotations

import json
import unittest
from dataclasses import fields, is_dataclass

from percept import tunables
from percept.config import Settings

#: Settings fields deliberately NOT represented as a ``Settings.*`` tunable path, each with a reason.
#: Empty today — every Settings field maps 1:1 to a registry path. Add an entry here (never silently
#: skip a field) if a future Settings field is genuinely not owner-classifiable at C0's boundary.
_EXCLUDED_SETTINGS_FIELDS: dict[str, str] = {}


def _settings_leaf_paths(instance: object, prefix: str = "Settings") -> set[str]:
    """Walk a (possibly nested) frozen dataclass instance to its leaf field paths, dotted."""
    out: set[str] = set()
    for f in fields(instance):
        value = getattr(instance, f.name)
        path = f"{prefix}.{f.name}"
        if is_dataclass(value) and not isinstance(value, type):
            out |= _settings_leaf_paths(value, path)
        else:
            out.add(path)
    return out


class TestRegistryIntegrity(unittest.TestCase):
    def test_no_duplicate_paths(self):
        paths = [spec.path for spec in tunables.REGISTRY]
        self.assertEqual(len(paths), len(set(paths)), "duplicate TunableSpec.path in REGISTRY")

    def test_every_owner_is_declared(self):
        for spec in tunables.REGISTRY:
            self.assertIn(spec.owner, tunables.OWNERS, spec.path)

    def test_registry_nonempty(self):
        self.assertGreater(len(tunables.REGISTRY), 0)

    def test_gate_owned_fields_never_safe_to_optimize(self):
        # Re-asserts ADR-008 Invariants ("One Gate... no configuration object may carry gate semantics")
        # at the registry level, not just inside TunableSpec.__post_init__.
        for spec in tunables.REGISTRY:
            if spec.owner == "Gate":
                self.assertFalse(spec.safe_to_optimize, f"{spec.path}: Gate-owned field marked optimizable")

    def test_sorted_by_path(self):
        paths = [spec.path for spec in tunables.REGISTRY]
        self.assertEqual(paths, sorted(paths), "REGISTRY must be sorted by path (see _build_registry)")


class TestArtifactParity(unittest.TestCase):
    """The checked-in tunables.json must be exactly what generate() produces right now."""

    def test_artifact_exists(self):
        self.assertTrue(tunables.ARTIFACT_PATH.exists(), f"missing {tunables.ARTIFACT_PATH}; run "
                         "`python -m percept.tunables`")

    def test_artifact_byte_matches_generate(self):
        on_disk = tunables.ARTIFACT_PATH.read_text(encoding="utf-8")
        regenerated = tunables.generate()
        self.assertEqual(
            on_disk, regenerated,
            "tunables.json is stale — run `python -m percept.tunables` and commit the result",
        )

    def test_artifact_is_well_formed(self):
        payload = json.loads(tunables.ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], tunables.SCHEMA_VERSION)
        self.assertIsInstance(payload["tunables"], list)
        self.assertEqual(len(payload["tunables"]), len(tunables.REGISTRY))
        paths = [t["path"] for t in payload["tunables"]]
        self.assertEqual(paths, sorted(paths))


class TestSettingsCompleteness(unittest.TestCase):
    """Every Settings field is a tunable path or explicitly excluded-with-reason."""

    def test_every_settings_field_is_covered_or_excluded(self):
        settings_paths = _settings_leaf_paths(Settings())
        registry_settings_paths = {
            spec.path for spec in tunables.REGISTRY if spec.path.startswith("Settings.")
        }
        covered = registry_settings_paths | set(_EXCLUDED_SETTINGS_FIELDS)
        missing = settings_paths - covered
        self.assertEqual(
            missing, set(),
            f"Settings field(s) neither a registry path nor excluded-with-reason: {sorted(missing)}",
        )

    def test_no_stale_settings_entries(self):
        # A registry path or exclusion naming a Settings field that no longer exists is equally a bug —
        # it means the inventory (or the exclusion list) drifted from the live dataclass tree.
        settings_paths = _settings_leaf_paths(Settings())
        registry_settings_paths = {
            spec.path for spec in tunables.REGISTRY if spec.path.startswith("Settings.")
        }
        stale_registry = registry_settings_paths - settings_paths
        stale_excluded = set(_EXCLUDED_SETTINGS_FIELDS) - settings_paths
        self.assertEqual(stale_registry, set(), f"registry paths naming a nonexistent Settings field: "
                          f"{sorted(stale_registry)}")
        self.assertEqual(stale_excluded, set(), f"excluded paths naming a nonexistent Settings field: "
                          f"{sorted(stale_excluded)}")

    def test_exclusion_reasons_are_nonempty(self):
        for path, reason in _EXCLUDED_SETTINGS_FIELDS.items():
            self.assertTrue(reason.strip(), f"{path}: exclusion reason must not be blank")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
