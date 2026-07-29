"""— ADR-004 §1 must not lie about what ``watch()`` resolves.

ADR-004 (`.adr/ADR-004-delivery-flow-replay.md`) shipped a "NOT guaranteed" bullet claiming
``profile`` is *not resolved yet* and that ``_settings_from_profile`` is a minimal pass-through. That
described an older ``watch.py`` before profile resolution: as shipped, ``_build_core`` resolves a real
``WatchProfile`` through
``resolve_watch_profile`` and applies ``settings``/``sense_policy``/``plan_mode`` at construction. An
integrator trusting the old text and passing a ``WatchProfile`` got silently changed gate semantics.

These tests are a DRIFT GUARD, not prose review. They pin, against the real modules:

1. the retracted false claim is gone and the corrected statement is present;
2. every ``watch.py`` line citation in the ADR points at the code it says it does;
3. the corrected claim is behaviourally true — a ``WatchProfile`` passed to ``_build_core`` really
   does change the gate's edge semantics, and ``percept=`` really does silently win over ``profile=``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from percept import Percept
from percept.config import Settings, load_settings
from percept.watch import _build_core, _settings_from_profile
from percept.watchprofile import CadenceProfile, GateProfile, WatchProfile

_PKG = Path(__file__).resolve().parents[1]  # packages/percept-vision
_REPO = _PKG.parents[1]  # repo root
_ADR = _REPO / ".adr" / "ADR-004-delivery-flow-replay.md"
_WATCH = _PKG / "percept" / "watch.py"


def _adr_text() -> str:
    return _ADR.read_text(encoding="utf-8")


def _watch_lines() -> list[str]:
    # 1-indexed access: pad index 0 so lines[n] is source line n.
    return [""] + _WATCH.read_text(encoding="utf-8").splitlines()


def _profile(**overrides) -> WatchProfile:
    kwargs = dict(
        name="x5-fixture",
        environment="fixed_camera",
        goal_type="stative",
        state="start_anchored",
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20},
        plan_mode="window",
        cadence=CadenceProfile(base_fps=2.0, max_fps=20.0, escalate_threshold=0.3, hold_s=1.0),
        gate=GateProfile(require_false_first=True, refractory_s=2.5),
    )
    kwargs.update(overrides)
    return WatchProfile(**kwargs)


class AdrDoesNotDenyProfileResolutionTests(unittest.TestCase):
    """The specific contract lie X5 removes must not come back."""

    def test_adr_exists(self):
        self.assertTrue(_ADR.is_file(), f"ADR-004 not found at {_ADR}")

    def test_adr_does_not_claim_profile_is_unresolved(self):
        text = _adr_text()
        for lie in (
            "`profile` (a future `WatchProfile`) is not resolved yet",
            "is not resolved yet",
            "deliberately minimal pass-through",
        ):
            self.assertNotIn(
                lie, text,
                f"ADR-004 still contains the retracted pre-resolution claim {lie!r}; watch.py resolves "
                "a WatchProfile for real (see percept/watch.py:_build_core).",
            )

    def test_adr_states_the_profile_is_resolved_and_cites_the_code(self):
        text = _adr_text()
        self.assertIn("resolve_watch_profile", text)
        self.assertIn("`watch.py:77-90`", text)  # _build_core
        self.assertIn("`watch.py:58-74`", text)  # _settings_from_profile
        self.assertIn("`watch.py:47`", text)  # the import that makes resolution real

    def test_adr_warns_that_percept_kwarg_overrides_profile(self):
        text = _adr_text()
        self.assertIn("`watch.py:191`", text)
        self.assertRegex(text, r"`percept=`[^\n]*wins over[^\n]*`profile=`")


class AdrWatchPyCitationsResolveTests(unittest.TestCase):
    """Every ``watch.py:N`` / ``watch.py:N-M`` citation in ADR-004 must point at the right code."""

    def test_cited_ranges_are_inside_the_file(self):
        lines = _watch_lines()
        last = len(lines) - 1
        cites = re.findall(r"watch\.py:\s*(\d+)(?:-(\d+))?", _adr_text())
        self.assertTrue(cites, "expected ADR-004 to cite watch.py by line")
        for start, end in cites:
            lo, hi = int(start), int(end or start)
            self.assertLessEqual(hi, last, f"watch.py:{lo}-{hi} runs past EOF ({last} lines)")
            self.assertLessEqual(lo, hi)

    def test_watch_version_citation(self):
        self.assertIn("`watch.py:55`", _adr_text())
        self.assertTrue(_watch_lines()[55].startswith("WATCH_VERSION"))

    def test_build_core_citation_spans_build_core(self):
        lines = _watch_lines()
        self.assertTrue(lines[77].startswith("def _build_core("))
        self.assertIn("return Percept.create(settings=_settings_from_profile(profile))", lines[90])
        self.assertIn("resolve_watch_profile(profile)", "\n".join(lines[83:90]))

    def test_settings_from_profile_citation_spans_that_function(self):
        lines = _watch_lines()
        self.assertTrue(lines[58].startswith("def _settings_from_profile("))
        self.assertIn('getattr(profile, "settings", None)', lines[74])

    def test_percept_kwarg_precedence_citation(self):
        self.assertIn(
            "core = percept if percept is not None else _build_core(profile)", _watch_lines()[191]
        )

    def test_as_watch_event_and_aiter_citations(self):
        lines = _watch_lines()
        self.assertTrue(lines[93].startswith("def _as_watch_event("))
        self.assertTrue(lines[104].startswith("async def _aiter("))
        self.assertIn("yield item", lines[108])

    def test_yield_loop_citation(self):
        lines = _watch_lines()
        self.assertIn("now = core.clock()", lines[194])
        self.assertIn("for ev in fired:", lines[196])
        self.assertIn("yield _as_watch_event(ev, t=now)", lines[197])


class ProfileReallyChangesGateSemanticsTests(unittest.TestCase):
    """The corrected ADR statement, proven against the code rather than asserted in prose."""

    def test_watchprofile_changes_the_gate_edge_knobs(self):
        default_core = _build_core(None)
        profiled = _build_core(_profile(gate=GateProfile(require_false_first=True, refractory_s=2.5)))
        # The default core must not already carry these — otherwise the test proves nothing.
        self.assertFalse(default_core.gate.require_false_first)
        self.assertIn(default_core.gate.refractory_s, (None, 0, 0.0))
        self.assertTrue(profiled.gate.require_false_first)
        self.assertEqual(profiled.gate.refractory_s, 2.5)

    def test_watchprofile_changes_cadence_and_plan_mode(self):
        profile = _profile()
        core = _build_core(profile)
        self.assertEqual(core.sense_policy, profile.cadence.to_sense_policy())
        self.assertEqual(core.plan_mode, "window")

    def test_profile_none_carries_no_override(self):
        core = _build_core(None)
        self.assertIsNone(core.sense_policy)
        self.assertIsNone(core.plan_mode)
        self.assertIsNone(_settings_from_profile(None))

    def test_bare_settings_and_settings_attr_are_pass_through_only(self):
        settings = load_settings()
        self.assertIs(_settings_from_profile(settings), settings)

        class _Opaque:
            def __init__(self, s: Settings) -> None:
                self.settings = s

        self.assertIs(_settings_from_profile(_Opaque(settings)), settings)
        # ...and neither shape carries a cadence/plan_mode override.
        core = _build_core(_Opaque(settings))
        self.assertIsNone(core.sense_policy)
        self.assertIsNone(core.plan_mode)

    def test_watchprofile_does_not_touch_gate_threshold(self):
        # ADR-004 says gate.threshold is NOT a profile-settable knob — it stays at load_settings().
        base = load_settings()
        core = _build_core(_profile())
        self.assertEqual(core.gate.threshold, base.gate.threshold)

    def test_supplied_percept_wins_over_profile(self):
        # ADR-004's remaining profile warning: `percept=` silently ignores `profile=`.
        supplied = Percept.create(mode="offline", discover_plugins=False)
        self.assertFalse(supplied.gate.require_false_first)
        # watch() uses the supplied core as-is; _build_core is never consulted.
        self.assertIsNone(supplied.sense_policy)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
