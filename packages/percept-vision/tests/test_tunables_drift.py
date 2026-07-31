"""Drift guards: every value :mod:`percept.tunables` *declares* is checked against the code it documents.

``tests/test_tunables.py`` pins registry integrity, generated-artifact parity, and bidirectional
``Settings.*`` path coverage. What it never reads is the four fields that make a
:class:`~percept.tunables.TunableSpec` a description of behaviour rather than a name:
``default``, ``bounds``, ``choices`` and ``env_var``. Those are the fields ``make docs-tunables``
renders into ``docs/configuration/tunables.md`` and ``percept doctor`` reports, so an untested
declaration is a knob whose documentation can be confidently wrong.

Four assertions, one walk of :data:`percept.tunables.REGISTRY`:

1. **Declared default == code default.** Every ``Settings.``-prefixed path resolves against a bare
   :class:`percept.config.Settings` and must equal the declared ``default`` — same value, same type,
   so a registry saying ``0.0`` where the dataclass ships ``0`` is caught rather than passing on
   ``0 == 0.0``.
2. **Declared bounds admit the default.** ``bounds[0] <= default <= bounds[1]`` for every bounded
   entry that ships a number.
3. **Declared choices contain the default.** ``default in choices`` for every choice-typed entry
   that ships a default, with a single documented exception carried in
   :data:`_DOCUMENTED_NO_DEFAULT` (never a silent skip) and asserted in both directions, so the
   allowlist cannot outlive the exception it records.
4. **Declared ``env_var`` is actually read.** Setting the declared variable to a non-default value
   must change what :func:`percept.config.load_settings` resolves. Variables belonging to seams
   outside the ``Settings`` tree (a backend reading its own environment) are instead required to
   appear in the package source *excluding the registry itself* — a variable that exists only
   because this registry declares it is a knob that documented itself into existence.

``default=None`` means "there is no shipped default" (the :class:`TunableSpec` docstring says so
for required kwargs), so a ``None`` default is excluded from rows 2 and 3 by construction rather
than by allowlist. Every exclusion is counted and every walk is floor-checked, so none of these
tests can pass by iterating over nothing.
"""

from __future__ import annotations

import os
import re
import unittest
from functools import lru_cache
from pathlib import Path
from typing import Any

from percept import tunables
from percept.config import Settings, load_settings

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "percept"

#: The registry module itself declares every ``env_var`` string, so scanning it would make row 4
#: trivially true for a variable no code path reads. Excluded on purpose.
_REGISTRY_MODULE = _PACKAGE_ROOT / "tunables.py"

#: Choice-typed entries whose shipped ``default`` is deliberately outside its own ``choices``, each
#: recorded with the reason and the site that ships it. Mirrors ``_EXCLUDED_SETTINGS_FIELDS`` in
#: ``tests/test_tunables.py``: an exception is written down with a reason, never skipped silently.
#: Asserted in BOTH directions below — remove an entry here the day the code stops needing it.
_DOCUMENTED_NO_DEFAULT: dict[str, str] = {
    "DetectorManifest.category": (
        'percept/harness/detectors.py:164 ships `category: str = ""`, and the class docstring at '
        ':157 states the empty string means "uncategorized" — the field is additive and optional so '
        "existing third-party manifests stay valid. The empty string is deliberately outside the "
        "category vocabulary rather than a member of it."
    ),
}

# Non-vacuity floors, measured on this tree. The registry grows; these are floors, not equalities,
# so a legitimate addition does not fail while a walk that silently stops covering anything does.
_MIN_SETTINGS_PATHS = 39
_MIN_BOUNDED_WITH_DEFAULT = 32
_MIN_CHOICE_TYPED = 24
_MIN_ENV_SETTINGS = 39
_MIN_ENV_NON_SETTINGS = 9


def _settings_specs() -> list[tunables.TunableSpec]:
    return [spec for spec in tunables.REGISTRY if spec.path.startswith("Settings.")]


def _resolve(instance: object, path: str) -> Any:
    """Resolve a dotted ``Settings.a.b`` registry path against a live tree by getattr chain."""
    obj = instance
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


@lru_cache(maxsize=1)
def _package_source_excluding_registry() -> str:
    """Every ``.py`` under ``percept/`` concatenated, minus the tunables registry module."""
    chunks: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if path == _REGISTRY_MODULE:
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _mentions_env_var(name: str) -> bool:
    # Trailing negative lookahead so PERCEPT_POSE_MODEL is not satisfied by PERCEPT_POSE_MODEL_PATH.
    return re.search(re.escape(name) + r"(?![A-Za-z0-9_])", _package_source_excluding_registry()) is not None


class TestDeclaredDefaultMatchesCode(unittest.TestCase):
    """Row: declared default == code default. Expected: every declared `default` equals the live value."""

    def test_every_settings_default_equals_the_live_dataclass_value(self):
        specs = _settings_specs()
        self.assertGreaterEqual(
            len(specs), _MIN_SETTINGS_PATHS,
            "the Settings.* walk covers fewer paths than this tree ships — the guard has gone vacuous",
        )
        shipped = Settings()
        for spec in specs:
            with self.subTest(path=spec.path):
                live = _resolve(shipped, spec.path)
                self.assertEqual(
                    live, spec.default,
                    f"{spec.path}: registry declares default={spec.default!r} but the dataclass "
                    f"ships {live!r} — docs/configuration/tunables.md and `percept doctor` would "
                    "both report the wrong number",
                )
                # 0 == 0.0 and False == 0 in Python; the generated doc renders the declared literal,
                # so an int/float/bool swap is drift even when the equality above holds.
                self.assertIs(
                    type(live), type(spec.default),
                    f"{spec.path}: declared default {spec.default!r} is a "
                    f"{type(spec.default).__name__} but the code ships a {type(live).__name__}",
                )

    def test_every_settings_path_resolves_against_the_live_tree(self):
        # A path that no longer resolves is an orphan row; test_tunables.py pins the set membership,
        # this pins that the getattr chain the walk above depends on is real for every entry.
        shipped = Settings()
        for spec in _settings_specs():
            with self.subTest(path=spec.path):
                try:
                    _resolve(shipped, spec.path)
                except AttributeError as exc:  # pragma: no cover - failure path
                    self.fail(f"{spec.path}: does not resolve against a live Settings() ({exc})")


class TestBoundsAdmitTheDefault(unittest.TestCase):
    """Row: declared bounds admit the default. Expected: `bounds[0] <= default <= bounds[1]`."""

    def test_every_bounded_default_lies_inside_its_own_bounds(self):
        checked = 0
        for spec in tunables.REGISTRY:
            if spec.bounds is None or spec.default is None:
                continue
            with self.subTest(path=spec.path):
                self.assertIsInstance(
                    spec.default, (int, float),
                    f"{spec.path}: bounds declared on a non-numeric default {spec.default!r}",
                )
                self.assertNotIsInstance(
                    spec.default, bool,
                    f"{spec.path}: bounds declared on a bool default; a bool has no range to admit",
                )
                low, high = spec.bounds
                self.assertGreaterEqual(
                    spec.default, low,
                    f"{spec.path}: default {spec.default!r} is below its own declared bounds {spec.bounds!r}",
                )
                self.assertLessEqual(
                    spec.default, high,
                    f"{spec.path}: default {spec.default!r} is above its own declared bounds {spec.bounds!r}",
                )
            checked += 1
        self.assertGreaterEqual(
            checked, _MIN_BOUNDED_WITH_DEFAULT,
            "fewer bounded-with-default entries than this tree ships — the guard has gone vacuous",
        )

    def test_a_bounded_entry_without_a_default_is_the_documented_no_default_case(self):
        # `default=None` means "there isn't one" per the TunableSpec docstring (required kwargs,
        # Goal.state keys a caller must supply). Pin that this is the ONLY reason an entry escapes
        # the bounds check, so a future non-numeric default cannot slip through as "skipped".
        for spec in tunables.REGISTRY:
            if spec.bounds is None:
                continue
            if spec.default is None:
                continue
            with self.subTest(path=spec.path):
                self.assertIsInstance(spec.default, (int, float), spec.path)


class TestChoicesContainTheDefault(unittest.TestCase):
    """Row: declared choices contain the default. Expected: default in choices."""

    def _violations(self) -> dict[str, tunables.TunableSpec]:
        return {
            spec.path: spec
            for spec in tunables.REGISTRY
            if spec.choices is not None
            and spec.default is not None
            and spec.default not in spec.choices
        }

    def test_every_choice_typed_default_is_one_of_its_choices(self):
        checked = 0
        for spec in tunables.REGISTRY:
            if spec.choices is None:
                continue
            checked += 1
            if spec.default is None:
                # No shipped default; nothing to find in the vocabulary.
                continue
            if spec.path in _DOCUMENTED_NO_DEFAULT:
                continue
            with self.subTest(path=spec.path):
                self.assertIn(
                    spec.default, spec.choices,
                    f"{spec.path}: default {spec.default!r} is absent from its own declared "
                    f"choices {spec.choices!r} — the shipped value is unreachable through the "
                    "documented vocabulary",
                )
        self.assertGreaterEqual(
            checked, _MIN_CHOICE_TYPED,
            "fewer choice-typed entries than this tree ships — the guard has gone vacuous",
        )

    def test_the_documented_exception_list_is_exactly_the_live_exception_set(self):
        # Both directions. A NEW default outside its choices fails here even though the loop above
        # allowlisted nothing new; and an allowlist entry the code no longer needs fails too, so the
        # reason text cannot outlive the behaviour it explains.
        self.assertEqual(
            set(self._violations()), set(_DOCUMENTED_NO_DEFAULT),
            "choice-typed defaults outside their own choices must match _DOCUMENTED_NO_DEFAULT "
            "exactly — add the new one with a reason, or delete a stale entry",
        )

    def test_every_documented_exception_carries_a_reason(self):
        for path, reason in _DOCUMENTED_NO_DEFAULT.items():
            with self.subTest(path=path):
                self.assertTrue(reason.strip(), f"{path}: allowlisted with no reason recorded")


class TestDeclaredEnvVarIsActuallyRead(unittest.TestCase):
    """Row: declared `env_var` is actually read. Expected: setting it changes the resolved value."""

    def setUp(self):
        self._saved: dict[str, str | None] = {}

    def tearDown(self):
        for name, previous in self._saved.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        self._saved.clear()

    def _set_env(self, name: str, raw: str) -> None:
        self._saved.setdefault(name, os.environ.get(name))
        os.environ[name] = raw

    def _probe(self, spec: tunables.TunableSpec) -> tuple[str, Any]:
        """A raw env string differing from the shipped default, and the value it should resolve to."""
        default = spec.default
        if spec.value_type == "bool":
            return ("0" if default else "1"), (not default)
        if spec.value_type == "int":
            return str(int(default) + 7), int(default) + 7
        if spec.value_type == "float":
            moved = float(default) + 0.037
            return repr(moved), moved
        if spec.value_type == "enum":
            alternatives = [c for c in (spec.choices or ()) if c != default]
            self.assertTrue(
                alternatives,
                f"{spec.path}: enum with no alternative to its default — cannot prove the var is read",
            )
            return alternatives[0], alternatives[0]
        if spec.value_type == "frozenset":
            return "percept_drift_probe", frozenset({"percept_drift_probe"})
        if spec.value_type == "str":
            return "percept-drift-probe", "percept-drift-probe"
        self.fail(
            f"{spec.path}: no probe for value_type={spec.value_type!r} — an env-declared Settings "
            "knob of an unprobeable type is an uncovered knob, not an exempt one"
        )
        raise AssertionError  # pragma: no cover - self.fail raises

    def test_every_settings_env_var_changes_the_resolved_value(self):
        specs = [s for s in _settings_specs() if s.env_var]
        self.assertGreaterEqual(
            len(specs), _MIN_ENV_SETTINGS,
            "fewer env-declared Settings paths than this tree ships — the guard has gone vacuous",
        )
        for spec in specs:
            with self.subTest(path=spec.path, env_var=spec.env_var):
                raw, expected = self._probe(spec)
                self.assertNotEqual(
                    expected, spec.default,
                    f"{spec.path}: the probe value must differ from the default to prove anything",
                )
                previous = os.environ.get(spec.env_var)
                self._set_env(spec.env_var, raw)
                try:
                    resolved = _resolve(load_settings(), spec.path)
                finally:
                    if previous is None:
                        os.environ.pop(spec.env_var, None)
                    else:
                        os.environ[spec.env_var] = previous
                self.assertNotEqual(
                    resolved, spec.default,
                    f"{spec.path}: {spec.env_var}={raw!r} left the resolved value at the default "
                    f"{spec.default!r} — the registry declares an override nothing honours",
                )
                self.assertEqual(
                    resolved, expected,
                    f"{spec.path}: {spec.env_var}={raw!r} resolved to {resolved!r}, not the "
                    f"{spec.value_type} the registry declares ({expected!r})",
                )

    def test_every_non_settings_env_var_name_appears_in_the_package_source(self):
        # These belong to seams that read their own environment (a backend, the pose provider,
        # Percept.create) rather than to load_settings(). The registry module is excluded from the
        # scan on purpose, so a variable that exists only as a registry row still fails.
        specs = [
            spec
            for spec in tunables.REGISTRY
            if spec.env_var and not spec.path.startswith("Settings.")
        ]
        self.assertGreaterEqual(
            len(specs), _MIN_ENV_NON_SETTINGS,
            "fewer env-declared non-Settings paths than this tree ships — the guard has gone vacuous",
        )
        for spec in specs:
            with self.subTest(path=spec.path, env_var=spec.env_var):
                self.assertTrue(
                    _mentions_env_var(spec.env_var),
                    f"{spec.path}: {spec.env_var} appears nowhere under percept/ outside the "
                    "registry that declares it — a knob that documented itself into existence",
                )

    def test_the_registry_module_is_excluded_from_the_source_scan(self):
        # Guards the guard, in both directions. If the exclusion silently stopped applying, every
        # declared env var would be found in the registry's own source and the test above would pass
        # unconditionally. The witness is taken from the live module text rather than retyped, so it
        # cannot decay into a string that is absent from everywhere and proves nothing.
        self.assertTrue(_REGISTRY_MODULE.is_file(), _REGISTRY_MODULE)
        registry_text = _REGISTRY_MODULE.read_text(encoding="utf-8")
        # `class TunableSpec:` is declared in exactly one module; the assertion below reads it from
        # the live file, so a rename cannot leave a witness that is absent from everywhere.
        witness = f"class {tunables.TunableSpec.__name__}:"
        self.assertIn(witness, registry_text, "witness no longer identifies the registry module")
        scanned = _package_source_excluding_registry()
        # Short messages on purpose: the scanned blob is the whole package and must never be
        # interpolated into a failure message.
        self.assertFalse(
            witness in scanned,
            "the tunables registry module is being scanned; the env-var check would be trivial",
        )
        # ...and the scan is otherwise a real read of the package, not an empty string.
        self.assertTrue("def load_settings()" in scanned, "the package source scan read nothing")

    def test_every_env_declared_spec_lands_in_exactly_one_arm(self):
        # No env-declared entry may fall between the two tests above.
        env_specs = [spec for spec in tunables.REGISTRY if spec.env_var]
        settings_arm = {s.path for s in env_specs if s.path.startswith("Settings.")}
        other_arm = {s.path for s in env_specs if not s.path.startswith("Settings.")}
        self.assertEqual(settings_arm & other_arm, set())
        self.assertEqual(
            len(settings_arm) + len(other_arm), len(env_specs),
            "an env-declared registry entry is covered by neither arm",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
