"""The env loader's constraints must agree with the registry that documents them.

``percept/tunables.py`` is the single declaration of what every ``PERCEPT_*`` variable means: its
``value_type``, its ``default``, its ``choices``, its ``bounds``. ``docs/configuration/tunables.md``
is GENERATED from that registry, so those bounds and vocabularies are a published promise.

``percept/config.py`` cannot import the registry to read them — ``percept.tunables`` imports
``percept.behavior_routes`` and ``percept.config`` is imported by ``percept/__init__.py``, so the
import would close a cycle. The values are therefore duplicated at the ``_env`` call sites, exactly
as ``percept.resolver._MISS_COST_ORDER`` duplicates ``percept.domain_pack``'s copy, and exactly as
that duplication is defended: **by a test that reads both and requires them to agree.**

This is that test. It spies on the ``_env`` calls one ``load_settings()`` makes, and for every
``Settings.*`` row in the registry that declares an ``env_var`` it requires:

* config reads that variable at all (a registry row nothing loads is a promise nobody keeps);
* config passes the registry's ``choices``;
* config passes the registry's ``bounds``;
* config passes the registry's ``default`` as the fallback.

The reverse direction is checked too: a variable ``load_settings`` reads with no registry row would
be an undocumented knob.
"""

from __future__ import annotations

import os
import unittest

from percept import config as config_mod
from percept import tunables


def _spy_on_env_calls() -> dict[str, dict]:
    """Every ``_env`` call one clean ``load_settings()`` makes, by variable name."""
    recorded: dict[str, dict] = {}
    real_env = config_mod._env

    def spy(name, default, cast, *, choices=None, bounds=None, sink=None):
        recorded[name] = {"default": default, "choices": choices, "bounds": bounds}
        return real_env(name, default, cast, choices=choices, bounds=bounds, sink=sink)

    config_mod._env = spy
    try:
        config_mod.load_settings()
    finally:
        config_mod._env = real_env
    return recorded


#: The registry rows ``load_settings`` is responsible for: the ``Settings.*`` tree. The other
#: ``env_var`` rows name variables read elsewhere — ``PERCEPT_MODE`` by :func:`resolve_mode`,
#: ``backend.*`` / ``PoseOpenness.*`` by the backends themselves — and are out of this loader's scope.
def _settings_env_specs() -> dict[str, tunables.TunableSpec]:
    return {
        spec.env_var: spec
        for spec in tunables.REGISTRY
        if spec.env_var and spec.path.startswith("Settings.")
    }


class EnvConstraintDriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = dict(os.environ)
        for key in [k for k in os.environ if k.startswith("PERCEPT_") and k != "PERCEPT_MODE"]:
            del os.environ[key]
        self.recorded = _spy_on_env_calls()
        self.specs = _settings_env_specs()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_the_two_sets_of_variables_are_the_same_set(self):
        """Neither a registry row nothing loads, nor a loaded knob nothing documents."""
        self.assertEqual(
            sorted(self.recorded),
            sorted(self.specs),
            "percept/config.py and the Settings.* half of percept/tunables.py disagree about "
            "which PERCEPT_* variables exist",
        )

    def test_every_declared_vocabulary_is_passed_to_the_loader(self):
        """``choices`` is what stops an out-of-vocabulary value being laundered downstream."""
        for name, spec in sorted(self.specs.items()):
            with self.subTest(env_var=name):
                self.assertEqual(
                    self.recorded[name]["choices"],
                    spec.choices,
                    f"{name}: config.py passes choices="
                    f"{self.recorded[name]['choices']!r}, registry declares {spec.choices!r}",
                )

    def test_every_declared_bound_is_passed_to_the_loader(self):
        """``bounds`` is a published promise: docs/configuration/tunables.md prints it."""
        for name, spec in sorted(self.specs.items()):
            with self.subTest(env_var=name):
                self.assertEqual(
                    self.recorded[name]["bounds"],
                    spec.bounds,
                    f"{name}: config.py passes bounds="
                    f"{self.recorded[name]['bounds']!r}, registry declares {spec.bounds!r}",
                )

    def test_every_declared_default_is_the_loader_fallback(self):
        """The value substituted on a rejection must be the one the registry advertises."""
        for name, spec in sorted(self.specs.items()):
            with self.subTest(env_var=name):
                self.assertEqual(
                    self.recorded[name]["default"],
                    spec.default,
                    f"{name}: config.py falls back to {self.recorded[name]['default']!r}, "
                    f"registry declares default={spec.default!r}",
                )

    def test_at_least_one_row_of_each_kind_exists(self):
        """Guard against the three tests above passing vacuously on an empty intersection."""
        self.assertGreater(len(self.specs), 30)
        self.assertTrue(any(s.choices for s in self.specs.values()))
        self.assertTrue(any(s.bounds for s in self.specs.values()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
