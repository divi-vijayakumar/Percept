"""the implementation's architectural rule, enforced mechanically: "the runtime reads its behavior from a profile;
the harness writes profiles. No knob is hardcoded in firing logic."

A grep-style guard over the profile-resolution seam (``percept/agent.py``, ``percept/watch.py``,
``percept/profile_wiring.py``): none of them may contain a bare salience-stat literal
(``"block_max"`` / ``"p95"``) or a bare non-zero threshold-shaped float (``0.NN``) OUTSIDE the one
sanctioned resolution table in ``profile_wiring.py`` (which necessarily names the ``WatchProfile``
vocabulary to bind it onto ``EdgeSettings`` — that IS the wiring, not a hardcoded knob used to decide a
firing outcome). Every real knob value must instead be read off a ``Settings``/``WatchProfile`` object.

This is deliberately scoped to the files the implementation owns — it is the regression guard for THIS wiring, not a
codebase-wide sweep (a broader sweep over ``gate.py`` / ``harness/tier0.py`` remains future work;
those files are not built or reviewed here).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_PERCEPT_DIR = Path(__file__).resolve().parent.parent / "percept"

# Firing-logic files under this guard: every threshold/stat must come from configuration.
# off a Settings/WatchProfile object, not typed in as a number/string.
_FIRING_LOGIC_FILES = ("agent.py", "watch.py")

# The one sanctioned resolution module: it MUST name the salience_stat vocabulary to bind it onto
# EdgeSettings fields (that mapping IS the wiring), but it still must not contain a bare threshold float
# -- every threshold comes from ``profile.stat_thresholds`` / the base ``Settings``, never a number typed
# directly into this file.
_RESOLUTION_FILE = "profile_wiring.py"

# A bare decimal literal, shape-matched against the ONE range every salience/gate probability-style knob
# in this codebase is validated to live in: strictly between 0 and 1 (see
# ``WatchProfile.__post_init__``'s ``stat_thresholds`` check, and ``EdgeSettings``/``GateSettings``'s
# ``motion_gate*`` / ``threshold`` docstrings). This deliberately does NOT flag 0.0/1.0 (the codebase's
# boolean-ish sentinels, e.g. confidence-as-0.0-or-1.0) or values >= 1 (fps/seconds constants like
# ``perception_cooldown_s`` are a different knob family, outside this guard's scope) -- narrow enough to
# have zero false positives against the pre-resolution baseline, wide enough to catch a hardcoded salience/gate
# threshold like "0.20" or "0.7" landing directly in firing logic instead of coming off Settings/Profile.
_THRESHOLD_FLOAT_RE = re.compile(r"(?<![\w.])0\.\d+(?![\w.])")

_SALIENCE_STAT_LITERALS = ('"block_max"', "'block_max'", '"p95"', "'p95'")


def _threshold_shaped_float_literals(text: str) -> list[str]:
    return [m for m in _THRESHOLD_FLOAT_RE.findall(text) if 0.0 < float(m) < 1.0]


class NoHardcodedKnobsInFiringLogicTests(unittest.TestCase):
    def test_firing_logic_files_exist(self):
        # sanity: prove the guard is actually scanning something, not silently vacuous.
        for name in _FIRING_LOGIC_FILES + (_RESOLUTION_FILE,):
            self.assertTrue((_PERCEPT_DIR / name).is_file(), f"expected percept/{name} to exist")

    def test_no_salience_stat_literal_in_firing_logic(self):
        for name in _FIRING_LOGIC_FILES:
            text = (_PERCEPT_DIR / name).read_text()
            for literal in _SALIENCE_STAT_LITERALS:
                self.assertNotIn(
                    literal, text,
                    f"percept/{name} hardcodes salience-stat literal {literal} -- firing logic must read "
                    f"salience_stat off Settings/WatchProfile, never branch on the literal directly",
                )

    def test_no_bare_threshold_float_in_firing_logic(self):
        for name in _FIRING_LOGIC_FILES:
            text = (_PERCEPT_DIR / name).read_text()
            hits = _threshold_shaped_float_literals(text)
            self.assertEqual(
                hits, [],
                f"percept/{name} contains bare numeric literal(s) {hits} shaped like a hardcoded "
                f"threshold/cadence knob -- read it off Settings/WatchProfile instead",
            )

    def test_resolution_module_uses_no_bare_threshold_float(self):
        # profile_wiring.py is allowed to NAME the salience_stat vocabulary (that binding is the wiring
        # itself) but must never hardcode a threshold number -- every value it produces must trace back to
        # ``profile.stat_thresholds`` or a ``base`` Settings field, not a number typed into this file.
        text = (_PERCEPT_DIR / _RESOLUTION_FILE).read_text()
        hits = _threshold_shaped_float_literals(text)
        self.assertEqual(
            hits, [],
            f"percept/{_RESOLUTION_FILE} contains bare numeric literal(s) {hits} -- resolved values must "
            f"come from the WatchProfile/base Settings, never a literal in the resolver",
        )

    def test_resolution_module_does_name_the_salience_stat_vocabulary(self):
        # the inverse check: profile_wiring.py SHOULD reference the vocabulary (that's its job) -- if this
        # ever goes false, the resolution table itself was gutted/moved and the wiring is no longer real.
        text = (_PERCEPT_DIR / _RESOLUTION_FILE).read_text()
        for literal in _SALIENCE_STAT_LITERALS[::2]:  # one quote style is enough to prove presence
            if literal in text:
                return
        self.fail(f"percept/{_RESOLUTION_FILE} no longer names the salience_stat vocabulary at all")


if __name__ == "__main__":
    unittest.main()
