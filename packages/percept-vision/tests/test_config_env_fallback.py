"""The env overlay: what an operator types, and what the settings tree actually becomes.

Everything a deployment tunes arrives as a ``PERCEPT_*`` string, and ``percept/config.py`` turns
that string into a typed field through exactly one funnel — ``_env`` (:65-72) plus the per-field
``cast``. This module pins that funnel's edges, because every one of them is a place where an
operator's stated intent can quietly become something else:

* an **unset** var must leave the declared default standing;
* an **empty-string** var must not be read as an instruction;
* a **garbage** value must not be adopted, and must not be swallowed in silence;
* a **bool** var must document which spellings mean true, because the ones outside that set do
  not merely fail to turn a knob on — they turn a defaulted-on knob off;
* an **out-of-vocabulary** enum must be refused where it is typed, not laundered into a different
  mode three layers downstream.

Each of those is now enforced by one ordered pipeline inside ``_env``, and every refusal is both
announced (``warnings`` + ``logging``) and recorded on ``Settings.env_rejections``, so "the operator
typo'd this" is distinguishable from "nobody ever configured it" by a caller, not only by a human
watching a console.
"""

from __future__ import annotations

import logging
import os
import unittest
import warnings

from percept import Goal
from percept.config import (
    ModeError,
    Settings,
    _as_bool,
    as_dict,
    load_settings,
    missing_credential,
)
from percept.harness.tier0 import _coerce_salience_stat, compile_watchspec

# The salience vocabulary, restated here rather than imported, so this file fails if the private
# set in tier0 is widened without anyone revisiting the load-time contract.
SALIENCE_VOCAB = frozenset({"mean", "block_max", "p95"})

# The four spellings `_as_bool` (:75) accepts, and a sample of what an operator plausibly types
# instead. Every token in the second group resolves to False.
TRUE_TOKENS = ("1", "true", "yes", "on")
TRUE_VARIANTS = ("1", "true", "TRUE", "True", "yes", "YES", "on", "ON", " on ", "\ttrue\n", "Yes ")
FALSE_TOKENS = (
    "enabled",
    "ENABLED",
    "y",
    "t",
    "2",
    "-1",
    "1.0",
    "0",
    "false",
    "off",
    "no",
    "",
    "   ",
    "true!",
    "yes please",
    "on,off",
)


class _EnvCase(unittest.TestCase):
    """Snapshot the whole environment; restore it byte-for-byte afterwards.

    Six other suites run in the same process and read these same names, so a leaked
    ``PERCEPT_*`` value is not a local failure — it is somebody else's mysterious one.
    """

    def setUp(self) -> None:
        self._env_snapshot = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def clear_percept_env(self) -> None:
        """Drop every ``PERCEPT_*`` override except the mode the suite fixture sets."""
        for key in [k for k in os.environ if k.startswith("PERCEPT_") and k != "PERCEPT_MODE"]:
            del os.environ[key]

    def capture_diagnostics(self):
        """Run ``load_settings()`` capturing BOTH standard diagnostic channels.

        Returns ``(settings, warnings_list, log_records)``. Checking warnings alone would be an
        unfair test — a cast failure could legitimately be reported through ``logging`` instead —
        so this listens on the root logger at DEBUG as well. If both come back empty, no diagnostic
        was emitted anywhere a caller could observe it.
        """
        records: list[logging.LogRecord] = []

        class _Collector(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        root = logging.getLogger()
        handler = _Collector()
        previous_level = root.level
        previous_disable = logging.root.manager.disable
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        logging.disable(logging.NOTSET)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                settings = load_settings()
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
            logging.disable(previous_disable)
        return settings, list(caught), records


# ---------------------------------------------------------------------------
# Unset and empty-string vars
# ---------------------------------------------------------------------------


class UnsetAndEmptyVarTests(_EnvCase):
    def test_no_env_overrides_reproduces_the_declared_defaults_exactly(self):
        """With nothing set, the loaded tree equals the dataclass defaults field for field."""
        self.clear_percept_env()
        self.assertEqual(as_dict(load_settings()), as_dict(Settings()))

    def test_an_unset_var_leaves_its_own_field_at_the_shipped_default(self):
        """Spot-check the fields this module goes on to abuse, so later failures are unambiguous."""
        self.clear_percept_env()
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.threshold, 0.7)
        self.assertEqual(loaded.entity.max_entities, 64)
        self.assertAlmostEqual(loaded.edge.motion_gate, 0.10)
        self.assertEqual(loaded.edge.salience_stat, "block_max")
        self.assertIs(loaded.gate.stale_ask, False)
        self.assertIs(loaded.edge.coast_enabled, True)

    def test_an_empty_string_var_falls_back_to_the_default_for_float_and_int_casts(self):
        """``float("")`` / ``int("")`` raise, so the numeric casts land back on the default.

        This is the covered half of the boundary: an exported-but-empty var (``PERCEPT_X=`` in a
        shell profile or compose file) must not be read as an instruction.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_THRESHOLD"] = ""
        os.environ["PERCEPT_GATE_VALIDITY_S"] = ""
        os.environ["PERCEPT_MOTION_GATE"] = ""
        os.environ["PERCEPT_MAX_ENTITIES"] = ""
        os.environ["PERCEPT_FRAME_WINDOW"] = ""
        os.environ["PERCEPT_CACHE_TAU_BITS"] = ""
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.threshold, 0.7)
        self.assertAlmostEqual(loaded.gate.validity_window_s, 90.0)
        self.assertAlmostEqual(loaded.edge.motion_gate, 0.10)
        self.assertEqual(loaded.entity.max_entities, 64)
        self.assertEqual(loaded.events.window, 12)
        self.assertEqual(loaded.cache.tau_bits, 6)

    def test_an_empty_string_var_is_not_an_instruction_for_a_string_cast(self):
        """``str("")`` cannot raise, so the empty term has to sit ABOVE the cast to catch this.

        ``PERCEPT_SALIENCE_STAT=`` used to produce the empty string — outside the salience
        vocabulary, and therefore coerced to ``mean`` downstream. Now it is refused where it is
        typed and the declared default stands.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_SALIENCE_STAT"] = ""
        os.environ["PERCEPT_ACTUATION"] = ""
        loaded = load_settings()
        self.assertEqual(loaded.edge.salience_stat, "block_max")
        self.assertEqual(loaded.consent.actuation, "none")
        self.assertIn(loaded.edge.salience_stat, SALIENCE_VOCAB)
        self.assertEqual(_coerce_salience_stat(None, loaded.edge.salience_stat), "block_max")
        self.assertEqual(
            {r.var: r.reason for r in loaded.env_rejections},
            {"PERCEPT_SALIENCE_STAT": "empty", "PERCEPT_ACTUATION": "empty"},
        )

    def test_an_empty_backend_name_falls_back_to_the_unset_default(self):
        """The third victim of the same term, and the one with a credential consequence.

        ``ModelSettings.vision`` defaults to ``None`` precisely so a LIVE run with no backend
        selected raises rather than silently faking (agent.py's ``_pick`` tests ``sel is not
        None``). An empty override used to make it ``''`` — not ``None``, so ``_pick`` accepted it
        as a selection, and not a known backend, so ``missing_credential`` had no keys to demand
        and the LIVE run proceeded credential-unchecked.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_VISION_BACKEND"] = ""
        os.environ["PERCEPT_ACT_ALLOWED_ACTIONS"] = ""
        loaded = load_settings()
        self.assertIsNone(loaded.models.vision)
        self.assertIsNone(missing_credential(loaded.models.vision))  # nothing to check: unset
        self.assertEqual(loaded.consent.allowed_actions, frozenset())
        self.assertEqual(
            sorted(r.var for r in loaded.env_rejections),
            ["PERCEPT_ACT_ALLOWED_ACTIONS", "PERCEPT_VISION_BACKEND"],
        )

    def test_an_empty_string_var_falls_back_to_the_default_for_the_bool_cast(self):
        """The boundary's stated expectation, applied to the one bool whose default is True."""
        self.clear_percept_env()
        os.environ["PERCEPT_COAST_ENABLED"] = ""
        self.assertIs(load_settings().edge.coast_enabled, True)

    def test_an_empty_bool_var_leaves_a_defaulted_on_knob_on_and_says_so(self):
        """``coast_enabled`` is the only bool in the tree that defaults True, so it is the only one
        where "empty means False" was observable — and it is exactly the knob that keeps a
        tier-0-illegible source (a document on screen) from going wholly unwatched.
        """
        self.clear_percept_env()
        self.assertIs(load_settings().edge.coast_enabled, True)  # baseline: shipped default
        os.environ["PERCEPT_COAST_ENABLED"] = ""
        settings, caught, records = self.capture_diagnostics()
        self.assertIs(settings.edge.coast_enabled, True)
        (rejection,) = settings.env_rejections
        self.assertEqual(
            (rejection.var, rejection.raw, rejection.reason, rejection.substituted),
            ("PERCEPT_COAST_ENABLED", "", "empty", True),
        )
        self.assertTrue(any("PERCEPT_COAST_ENABLED" in str(w.message) for w in caught))
        self.assertTrue(any("PERCEPT_COAST_ENABLED" in r.getMessage() for r in records))


# ---------------------------------------------------------------------------
# _as_bool's accepted vocabulary
# ---------------------------------------------------------------------------


class AsBoolVocabularyTests(_EnvCase):
    def test_as_bool_is_true_for_exactly_the_four_accepted_tokens(self):
        for raw in TRUE_VARIANTS:
            with self.subTest(raw=raw):
                self.assertIs(_as_bool(raw), True)
        # The accepted set is case- and whitespace-insensitive but is not extensible: the four
        # canonical tokens are the whole of it.
        for token in TRUE_TOKENS:
            with self.subTest(token=token):
                self.assertIs(_as_bool(token.upper()), True)

    def test_as_bool_is_false_for_every_other_spelling(self):
        for raw in FALSE_TOKENS:
            with self.subTest(raw=raw):
                self.assertIs(_as_bool(raw), False)

    def test_enabled_reaches_load_settings_as_off_with_no_diagnostic(self):
        """``PERCEPT_GATE_STALE_ASK=enabled`` is documented to mean False.

        Driven all the way through ``load_settings`` rather than asserted on ``_as_bool``, because
        the operator-visible artifact is the settings field, not the helper. The paired ``on``
        case proves the knob is reachable at all, so the False arm is not passing vacuously.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_STALE_ASK"] = "enabled"
        settings, caught, records = self.capture_diagnostics()
        self.assertIs(settings.gate.stale_ask, False)
        self.assertIs(settings.gate.stale_ask, Settings().gate.stale_ask)  # indistinguishable
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual([r.getMessage() for r in records], [])

        os.environ["PERCEPT_GATE_STALE_ASK"] = "on"
        self.assertIs(load_settings().gate.stale_ask, True)

    def test_an_unaccepted_spelling_turns_a_defaulted_on_knob_off(self):
        """The asymmetry worth knowing: an unrecognized token is not inert, it means False.

        For a knob defaulting False (``stale_ask``) a typo is merely ignored. For one defaulting
        True (``coast_enabled``) the same typo flips the deployment.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_COAST_ENABLED"] = "enabled"
        self.assertIs(load_settings().edge.coast_enabled, False)
        os.environ["PERCEPT_COAST_ENABLED"] = "yes"
        self.assertIs(load_settings().edge.coast_enabled, True)

    def test_every_bool_field_in_the_tree_reads_through_the_same_vocabulary(self):
        """No bool knob has its own private spelling rules."""
        self.clear_percept_env()
        bool_vars = {
            "PERCEPT_GATE_STALE_ASK": lambda s: s.gate.stale_ask,
            "PERCEPT_GATE_REQUIRE_FALSE_FIRST": lambda s: s.gate.require_false_first,
            "PERCEPT_CACHE_ENABLED": lambda s: s.cache.enabled,
            "PERCEPT_ALLOW_COMPUTER_USE": lambda s: s.consent.allow_computer_use,
            "PERCEPT_EGO_COMPENSATE": lambda s: s.edge.ego_compensate,
            "PERCEPT_COAST_ENABLED": lambda s: s.edge.coast_enabled,
        }
        for name, read in bool_vars.items():
            for raw, expected in (("on", True), ("enabled", False), ("1", True), ("t", False)):
                with self.subTest(var=name, raw=raw):
                    os.environ[name] = raw
                    self.assertIs(read(load_settings()), expected)
            del os.environ[name]


# ---------------------------------------------------------------------------
# A bad cast for each type
# ---------------------------------------------------------------------------


class BadCastFallbackTests(_EnvCase):
    def test_a_garbage_float_falls_back_to_the_shipped_default(self):
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_THRESHOLD"] = "high"
        os.environ["PERCEPT_MOTION_GATE"] = "high"
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.threshold, 0.7)
        self.assertAlmostEqual(loaded.edge.motion_gate, 0.10)

    def test_a_garbage_int_falls_back_to_the_shipped_default(self):
        self.clear_percept_env()
        os.environ["PERCEPT_MAX_ENTITIES"] = "lots"
        os.environ["PERCEPT_FRAME_WINDOW"] = "12.5"  # int("12.5") raises too
        loaded = load_settings()
        self.assertEqual(loaded.entity.max_entities, 64)
        self.assertEqual(loaded.events.window, 12)

    def test_the_fallback_is_the_default_not_a_zero_or_a_none(self):
        """A cast failure must not degrade into a falsy value that reads as "no bound"."""
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_KEY_TTL_S"] = "forever"
        os.environ["PERCEPT_PREROLL_MAX_BYTES"] = "128MB"
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.key_ttl_s, 3600.0)
        self.assertEqual(loaded.events.preroll_max_bytes, 134_217_728)


class BadCastDiagnosticTests(_EnvCase):
    def test_a_bad_cast_for_each_type_emits_a_diagnostic(self):
        """A rejected override must say so on some channel a caller can observe.

        An operator who exports ``PERCEPT_GATE_THRESHOLD=high`` believes they raised the
        confident-yes cutoff. The run proceeds at 0.7 and nothing anywhere records that their
        instruction was discarded.
        """
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_THRESHOLD"] = "high"  # float cast
        os.environ["PERCEPT_MAX_ENTITIES"] = "lots"  # int cast
        settings, caught, records = self.capture_diagnostics()
        # The fallback half holds (pinned green above); only the diagnostic is missing.
        self.assertAlmostEqual(settings.gate.threshold, 0.7)
        self.assertEqual(settings.entity.max_entities, 64)
        self.assertTrue(
            caught or records,
            "two overrides were discarded and neither warnings nor logging said anything",
        )
        self.assertEqual(
            [(r.var, r.reason) for r in settings.env_rejections],
            [("PERCEPT_GATE_THRESHOLD", "uncast"), ("PERCEPT_MAX_ENTITIES", "uncast")],
        )

    def test_a_discarded_override_is_distinguishable_from_an_unset_one(self):
        """The durable half: the rejection rides on the tree, not only on a console line.

        Every tuned field still reads exactly as it would with nothing set — that is the whole
        point of falling back — so ``env_rejections`` is the ONLY thing that can tell a caller,
        ``/health`` or a receipt that two overrides were thrown away.
        """
        self.clear_percept_env()
        baseline = as_dict(load_settings())
        self.assertEqual(baseline["env_rejections"], [])
        os.environ["PERCEPT_GATE_THRESHOLD"] = "high"
        os.environ["PERCEPT_MAX_ENTITIES"] = "lots"
        settings, caught, records = self.capture_diagnostics()

        rendered = as_dict(settings)
        self.assertNotEqual(rendered, baseline)
        self.assertEqual(
            rendered["env_rejections"],
            [
                {
                    "var": "PERCEPT_GATE_THRESHOLD",
                    "raw": "high",
                    "reason": "uncast",
                    "substituted": 0.7,
                },
                {
                    "var": "PERCEPT_MAX_ENTITIES",
                    "raw": "lots",
                    "reason": "uncast",
                    "substituted": 64,
                },
            ],
        )
        # Strip the ledger and the two trees are identical again — the fields themselves are
        # genuinely back on their defaults, which is why the ledger has to exist.
        rendered.pop("env_rejections")
        baseline.pop("env_rejections")
        self.assertEqual(rendered, baseline)
        self.assertEqual(len(caught), 2)
        self.assertEqual(len([r for r in records if r.name == "percept.config"]), 2)


# ---------------------------------------------------------------------------
# Declared bounds
# ---------------------------------------------------------------------------


class DeclaredBoundsTests(_EnvCase):
    """A castable value that is nonetheless outside the field's published range.

    ``docs/configuration/tunables.md`` is GENERATED from ``percept/tunables.py``, so every one of
    these ranges is a promise the shipped public doc already makes. Enforcing them at the loader is
    what makes the promise true.
    """

    def test_a_threshold_above_one_cannot_produce_a_gate_that_never_fires(self):
        """``PERCEPT_GATE_THRESHOLD=5.0`` casts cleanly and used to load verbatim — after which
        ``classify`` returns 'unknown' for every verdict and the watch can never fire, silently."""
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_THRESHOLD"] = "5.0"
        settings, caught, records = self.capture_diagnostics()
        self.assertAlmostEqual(settings.gate.threshold, 0.7)
        (rejection,) = settings.env_rejections
        self.assertEqual(
            (rejection.var, rejection.raw, rejection.reason, rejection.substituted),
            ("PERCEPT_GATE_THRESHOLD", "5.0", "out_of_bounds", 0.7),
        )
        self.assertTrue(any("PERCEPT_GATE_THRESHOLD" in str(w.message) for w in caught))
        self.assertTrue(any("PERCEPT_GATE_THRESHOLD" in r.getMessage() for r in records))

    def test_negative_counts_and_rates_are_refused(self):
        """A negative eviction bound and a negative audit rate are not weaker settings, they are
        nonsense the rest of the tree has no defined behaviour for."""
        self.clear_percept_env()
        os.environ["PERCEPT_MAX_ENTITIES"] = "-3"
        os.environ["PERCEPT_CACHE_AUDIT_RATE"] = "-1"
        loaded = load_settings()
        self.assertEqual(loaded.entity.max_entities, 64)
        self.assertAlmostEqual(loaded.cache.audit_rate, 0.05)
        self.assertEqual(
            [(r.var, r.reason) for r in loaded.env_rejections],
            [
                ("PERCEPT_MAX_ENTITIES", "out_of_bounds"),
                ("PERCEPT_CACHE_AUDIT_RATE", "out_of_bounds"),
            ],
        )

    def test_the_bounds_are_inclusive_at_both_ends(self):
        """The endpoints the registry declares are legal values, not the first refused ones."""
        self.clear_percept_env()
        for raw in ("0.0", "1.0"):
            with self.subTest(raw=raw):
                os.environ["PERCEPT_GATE_THRESHOLD"] = raw
                loaded = load_settings()
                self.assertAlmostEqual(loaded.gate.threshold, float(raw))
                self.assertEqual(loaded.env_rejections, ())
        os.environ["PERCEPT_MAX_ENTITIES"] = "1"
        self.assertEqual(load_settings().entity.max_entities, 1)

    def test_an_unbounded_field_still_accepts_its_documented_off_switch(self):
        """``key_ttl_s``/``max_keys``/``refractory_s`` document 0 (or negative) as "off", and the
        registry declares no bound where that is the case — so the arm must not invent one."""
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_KEY_TTL_S"] = "-1"
        os.environ["PERCEPT_GATE_REFRACTORY_S"] = "-1"
        os.environ["PERCEPT_GATE_MAX_KEYS"] = "0"
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.key_ttl_s, -1.0)
        self.assertAlmostEqual(loaded.gate.refractory_s, -1.0)
        self.assertEqual(loaded.gate.max_keys, 0)
        self.assertEqual(loaded.env_rejections, ())

    def test_a_value_inside_the_bounds_is_adopted_unchanged(self):
        """Control: the arm is a filter, not a clamp — an in-range override still lands."""
        self.clear_percept_env()
        os.environ["PERCEPT_GATE_THRESHOLD"] = "0.85"
        os.environ["PERCEPT_MOTION_GATE"] = "0.42"
        loaded = load_settings()
        self.assertAlmostEqual(loaded.gate.threshold, 0.85)
        self.assertAlmostEqual(loaded.edge.motion_gate, 0.42)
        self.assertEqual(loaded.env_rejections, ())


# ---------------------------------------------------------------------------
# An out-of-vocabulary salience_stat
# ---------------------------------------------------------------------------


def _watch_goal(gid: str = "kitchen") -> Goal:
    return Goal(id=gid, condition="someone is at the counter", say=f"saw {gid}")


def _compile_with(settings) -> object:
    """Compile one condition the way the runtime worker does (worker.py:4952-4966)."""
    spec = compile_watchspec(
        [_watch_goal()],
        threshold=settings.edge.motion_gate,
        salience_stat=settings.edge.salience_stat,
        ego_compensate=settings.edge.ego_compensate,
        stat_thresholds={
            "mean": settings.edge.motion_gate,
            "block_max": settings.edge.motion_gate_block_max,
            "p95": settings.edge.motion_gate_p95,
        },
    )
    return spec.conditions[0]


class SalienceStatVocabularyTests(_EnvCase):
    def test_an_invalid_salience_stat_is_rejected_at_load(self):
        """Either raise, or leave a value the rest of the system can act on. Not a third thing."""
        self.clear_percept_env()
        os.environ["PERCEPT_SALIENCE_STAT"] = "nonsense"
        try:
            loaded = load_settings()
        except (ValueError, ModeError):
            return  # refused loudly where it was typed — the expectation is met
        self.assertIn(
            loaded.edge.salience_stat,
            SALIENCE_VOCAB,
            "an out-of-vocabulary salience mode was accepted verbatim at load",
        )

    def test_an_invalid_salience_stat_never_reaches_the_compiled_condition(self):
        """The consequence the row names, asserted as no longer reachable from the environment.

        ``_coerce_salience_stat`` (tier0.py) is a firewall for an untrusted PER-GOAL hint. Its
        ``default`` argument is trusted, so when the deployment default was itself invalid the
        firewall's last line handed back ``mean`` — not the shipped ``block_max`` — for every goal
        at once, taking the compiled cutoff from 0.20 down to 0.10 with it. The firewall still
        behaves that way in isolation; what changed is that the loader no longer hands it a value
        outside the vocabulary.
        """
        self.clear_percept_env()
        shipped = _compile_with(load_settings())
        self.assertEqual(shipped.salience_stat, "block_max")
        self.assertAlmostEqual(shipped.threshold, 0.20)

        os.environ["PERCEPT_SALIENCE_STAT"] = "nonsense"
        settings, caught, records = self.capture_diagnostics()
        self.assertEqual(settings.edge.salience_stat, "block_max")
        (rejection,) = settings.env_rejections
        self.assertEqual(
            (rejection.var, rejection.raw, rejection.reason, rejection.substituted),
            ("PERCEPT_SALIENCE_STAT", "nonsense", "not_in_choices", "block_max"),
        )
        self.assertTrue(any("PERCEPT_SALIENCE_STAT" in str(w.message) for w in caught))
        self.assertTrue(any("PERCEPT_SALIENCE_STAT" in r.getMessage() for r in records))

        # The firewall's fallback is unchanged in isolation — it simply never sees this input now.
        self.assertEqual(_coerce_salience_stat(None, "nonsense"), "mean")
        compiled = _compile_with(settings)
        self.assertEqual(compiled.salience_stat, "block_max")
        self.assertAlmostEqual(compiled.threshold, 0.20)
        self.assertEqual(compiled.salience_stat, shipped.salience_stat)

    def test_a_valid_salience_stat_still_reaches_the_compiled_condition(self):
        """Control: the coercion path is not simply always returning mean."""
        self.clear_percept_env()
        for stat, cutoff in (("mean", 0.10), ("block_max", 0.20), ("p95", 0.15)):
            with self.subTest(stat=stat):
                os.environ["PERCEPT_SALIENCE_STAT"] = stat
                settings = load_settings()
                self.assertEqual(settings.edge.salience_stat, stat)
                compiled = _compile_with(settings)
                self.assertEqual(compiled.salience_stat, stat)
                self.assertAlmostEqual(compiled.threshold, cutoff)

    def test_a_per_goal_hint_outside_the_vocabulary_still_falls_back_to_a_valid_default(self):
        """The firewall does work when the DEPLOYMENT default is sound — the hint is the untrusted
        input it was written for."""
        self.clear_percept_env()
        self.assertEqual(_coerce_salience_stat("nonsense", "block_max"), "block_max")
        self.assertEqual(_coerce_salience_stat(None, "block_max"), "block_max")
        self.assertEqual(_coerce_salience_stat("p95", "block_max"), "p95")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
