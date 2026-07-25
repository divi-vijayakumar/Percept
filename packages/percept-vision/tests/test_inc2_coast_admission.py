"""INC2 — signal-only admission via the strict_admission ExecutionPolicy.

The nine checks the card declares, one class per concern:

1. strict_admission@1 resolves SCREEN_DOCUMENT with NO coast bindings emitted, the coast_enabled
   SettingOrigin still present in provenance, and an applied_with_warnings receipt carrying
   coast_required_overridden.
2. Omitting the policy (or naming balanced@1) is byte-for-byte today's bindings.
3. A policy writing SourceProfile.parameters.coast_interval_s raises ConfigError (the ownership gate,
   pinned so a future edit cannot quietly let an ExecutionPolicy reach a SourceProfile-owned field).
4. Blank-screen fixture: default admits on the coast (judge calls spent), strict admits ZERO on the
   same static content (zero coast admits, zero judge calls) while a real content change still admits
   via the tuned/motion signal.
5. The admission partition invariant escalations == signal_admits + coast_admits + audio_wakes.
6. The coverage summary (runtime_ended payload) carries the split.
7. An UNTUNED profile under strict warns (coast_disabled), never errors.
8. A brief-action goal's fovea coast (a goal-authored sense, not the SourceProfile coast) is
   UNAFFECTED by coast_enabled=False.
9. A WatchPlan-explicit Goal.state.sense.base_fps still wins precedence over the profile coast.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.config import EdgeSettings, Settings
from percept.contracts import WatchPlan
from percept.edge.tier0 import gray_frame
from percept.execution_policy import ExecutionPolicy, get_execution_policy
from percept.resolver import ConfigError, ConfigResolver
from percept.runtime import RuntimeStats, run
from percept.source_profiles_default import SCREEN_DOCUMENT
from percept.sourceprofile import SourceProfile

_COAST_KEYS = (
    "Goal.state.sense.base_fps",
    "Goal.state.sense.max_fps",
    "Goal.state.cadence.min_judge_interval",
    "Goal.state.cadence.gate_on",
)


def _plan() -> WatchPlan:
    return WatchPlan(condition="a red banner appears on the page")


def _strict() -> ExecutionPolicy:
    return get_execution_policy("strict_admission@1")


# --- 1. strict resolution: no coast bindings, coast_enabled origin present, loud warning ----------
class StrictResolutionTests(unittest.TestCase):
    def test_no_coast_bindings_but_coast_enabled_origin_lands_and_receipt_warns(self):
        r = ConfigResolver()
        res = r.resolve(_plan(), source_profile=SCREEN_DOCUMENT, execution_policy=_strict())
        for key in _COAST_KEYS:
            self.assertNotIn(key, res.effective, f"{key} must not be bound under strict_admission")
        # the gating READ itself is provenance-visible: the coast_enabled row is still resolved.
        self.assertIn("Settings.edge.coast_enabled", res.effective)
        self.assertIs(res.effective["Settings.edge.coast_enabled"], False)
        origin = res.provenance["Settings.edge.coast_enabled"]
        self.assertEqual(origin.owner_layer, "ExecutionPolicy")
        self.assertEqual(origin.source_id, "strict_admission")
        receipt = r.last_receipt
        self.assertEqual(receipt.status, "applied_with_warnings")
        codes = {w.code for w in receipt.warnings}
        self.assertIn("coast_required_overridden", codes)
        warn = next(w for w in receipt.warnings if w.code == "coast_required_overridden")
        self.assertEqual(warn.path, "SourceProfile.parameters.coast_interval_s")
        self.assertIsNone(warn.goal_id)  # plan-level, not per-goal


# --- 2. omitted / balanced == byte-for-byte today's bindings -------------------------------------
class BalancedIsByteForByteTests(unittest.TestCase):
    def test_balanced_and_omitted_match_and_bind_the_coast(self):
        baseline = ConfigResolver().resolve(_plan(), source_profile=SCREEN_DOCUMENT)
        balanced = ConfigResolver().resolve(
            _plan(), source_profile=SCREEN_DOCUMENT, execution_policy=get_execution_policy("balanced@1")
        )
        self.assertEqual(balanced.effective, baseline.effective)
        # and the coast IS bound on this default path (the thing strict removes)
        self.assertEqual(baseline.effective["Goal.state.sense.base_fps"], 0.5)
        self.assertEqual(baseline.effective["Settings.edge.coast_enabled"], True)


# --- 3. ownership gate: an ExecutionPolicy cannot reach a SourceProfile-owned field ---------------
class OwnershipGateTests(unittest.TestCase):
    def test_policy_writing_coast_interval_s_raises(self):
        bad = ExecutionPolicy(id="bad_coast", version="1",
                              values={"SourceProfile.parameters.coast_interval_s": 5.0})
        with self.assertRaises(ConfigError):
            ConfigResolver().resolve(_plan(), source_profile=SCREEN_DOCUMENT, execution_policy=bad)


# --- 4. blank-screen fixture: default coasts, strict is signal-only -------------------------------
W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

# All-STILL over 5s: the motion gate never escalates, so ONLY a coast can spend a judge here.
BLANK = [(round(0.5 * i, 2), STILL, "blank") for i in range(11)]  # t = 0.0 .. 5.0 at 2 fps
# A real content change: STILL -> MOVED clears the motion gate -> a tuned-signal admit.
CHANGED = [(0.0, STILL, "blank"), (1.0, MOVED, "banner"), (2.0, STILL, "blank")]
# One continuous synthetic stream for the mixed partition test.  Concatenating CHANGED at t=0 after
# BLANK ends at t=5 would now correctly be rejected as an undeclared timestamp reset.
MIXED = BLANK + [(t + 6.0, frame, token) for t, frame, token in CHANGED]


def _doc_vision() -> FakeVision:
    return FakeVision(script={
        "blank": Verdict(satisfied=False, confidence=0.9, evidence="nothing changed"),
        "banner": Verdict(satisfied=True, confidence=0.95, evidence="a red banner is shown"),
    })


def _coast_sense(execution_policy) -> dict:
    """The Goal.state.sense the resolver actually produces for SCREEN_DOCUMENT under this policy —
    so the worker run is driven by the REAL resolved bindings, not a hand-built stand-in."""
    res = ConfigResolver().resolve(_plan(), source_profile=SCREEN_DOCUMENT, execution_policy=execution_policy)
    sense = {}
    if "Goal.state.sense.base_fps" in res.effective:
        sense["base_fps"] = res.effective["Goal.state.sense.base_fps"]
    if "Goal.state.sense.max_fps" in res.effective:
        sense["max_fps"] = res.effective["Goal.state.sense.max_fps"]
    return sense


def _doc_core(execution_policy) -> Percept:
    core = Percept.create(vision=_doc_vision(), discover_plugins=False)
    sense = _coast_sense(execution_policy)
    state = {"sense": sense} if sense else {}
    core.add_goal(Goal(id="doc", condition="a red banner appears on the page", say="banner", state=state))
    return core


def _run(core, source, stats):
    return asyncio.run(_collect(core, source, stats=stats))


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class BlankScreenAdmissionTests(unittest.TestCase):
    def test_default_coasts_but_strict_is_signal_only(self):
        # DEFAULT (coast bound): the coast forces judges on the static screen.
        default_stats = RuntimeStats()
        _run(_doc_core(None), BLANK, default_stats)
        self.assertGreater(default_stats.coast_admits, 0, "default must coast-admit on static content")
        self.assertGreater(default_stats.judge_calls, 0, "default spends judge calls on the coast")

        # STRICT (coast unbound): the same static content admits NOTHING and spends NO judge call.
        strict_stats = RuntimeStats()
        _run(_doc_core(_strict()), BLANK, strict_stats)
        self.assertEqual(strict_stats.coast_admits, 0, "strict must not coast-admit")
        self.assertEqual(strict_stats.judge_calls, 0, "strict spends zero judge calls on static content")

        # STRICT still admits a REAL content change via the tuned/motion signal.
        changed_stats = RuntimeStats()
        _run(_doc_core(_strict()), CHANGED, changed_stats)
        self.assertGreater(changed_stats.signal_admits, 0, "strict still admits a real change via the signal")
        self.assertEqual(changed_stats.coast_admits, 0)

    # expose the before/after the card asks the report to name
    def numbers(self):  # pragma: no cover - helper for the human report, not an assertion
        d, s = RuntimeStats(), RuntimeStats()
        _run(_doc_core(None), BLANK, d)
        _run(_doc_core(_strict()), BLANK, s)
        return d, s


# --- 5. the admission partition invariant --------------------------------------------------------
class PartitionInvariantTests(unittest.TestCase):
    def test_escalations_equal_signal_plus_coast_plus_audio(self):
        # A MIXED run: some coast admits (static ticks) AND a motion escalation (the change).
        stats = RuntimeStats()
        _run(_doc_core(None), MIXED, stats)
        self.assertEqual(
            stats.escalations,
            stats.signal_admits + stats.coast_admits + stats.audio_wakes,
        )
        self.assertGreater(stats.coast_admits, 0)
        self.assertGreater(stats.signal_admits, 0)


# --- 6. the coverage summary carries the split ---------------------------------------------------
class CoverageSummaryTests(unittest.TestCase):
    def test_runtime_ended_payload_carries_the_admission_split(self):
        events = []
        asyncio.run(_collect(_doc_core(None), BLANK, on_health=events.append))
        ended = [e for e in events if e.kind == "runtime_ended"]
        self.assertEqual(len(ended), 1)
        data = ended[0].data
        for key in ("signal_admits", "coast_admits", "audio_wakes"):
            self.assertIn(key, data)
        self.assertEqual(
            data["admitted"], data["signal_admits"] + data["coast_admits"] + data["audio_wakes"]
        )
        self.assertGreater(data["coast_admits"], 0)


# --- 7. an UNTUNED profile under strict warns, never errors ---------------------------------------
_UNTUNED = SourceProfile(
    name="untuned_coaster",
    modalities=frozenset({"video"}),
    temporal_unit="frame",
    admission_strategy="coast_only",
    evidence_strategy="screenshot",
    identity_strategy="content_key",
    parameters={"change_signal": "UNTUNED", "coast_interval_s": 2.0},  # NO coast_required
    stability="experimental",
)


class UntunedProfileStrictTests(unittest.TestCase):
    def test_untuned_profile_under_strict_warns_without_error(self):
        r = ConfigResolver()
        res = r.resolve(_plan(), source_profile=_UNTUNED, execution_policy=_strict())  # must not raise
        self.assertNotIn("Goal.state.sense.base_fps", res.effective)
        self.assertNotIn("Goal.state.cadence.min_judge_interval", res.effective)
        receipt = r.last_receipt
        self.assertEqual(receipt.status, "applied_with_warnings")
        codes = {w.code for w in receipt.warnings}
        # no coast_required on this profile -> the softer coast_disabled code, not the escalated one
        self.assertIn("coast_disabled", codes)
        self.assertNotIn("coast_required_overridden", codes)


# --- 8. the brief-action fovea coast is a goal-authored sense, unaffected by coast_enabled --------
QUIET = [(round(0.25 * i, 2), STILL, "blank") for i in range(9)]  # t = 0.0 .. 2.0 at 4 fps, all still


def _brief_core(coast_enabled: bool) -> Percept:
    core = Percept.create(vision=_doc_vision(), discover_plugins=False)
    # coast_enabled lives in settings; the brief coast must ignore it (it is the goal's own fovea).
    core.settings = Settings(edge=EdgeSettings(coast_enabled=coast_enabled))
    core.add_goal(Goal(id="brief", condition="a red banner appears on the page", say="banner",
                       state={"sense": {"brief_window_s": 0.5}}))
    return core


class BriefCoastUnaffectedTests(unittest.TestCase):
    def test_brief_goal_coast_still_admits_under_coast_disabled(self):
        off = RuntimeStats()
        _run(_brief_core(coast_enabled=False), QUIET, off)
        on = RuntimeStats()
        _run(_brief_core(coast_enabled=True), QUIET, on)
        # the fovea coast fires the same either way — Settings.edge.coast_enabled never touches it.
        self.assertGreater(off.coast_admits, 0, "brief fovea coast must fire even under coast_enabled=False")
        self.assertEqual(off.coast_admits, on.coast_admits)


# --- 9. a WatchPlan-explicit sense floor wins precedence over the profile coast -------------------
class ExplicitPrecedenceTests(unittest.TestCase):
    def test_explicit_base_fps_wins_over_profile_coast(self):
        plan = WatchPlan(condition="a red banner appears on the page",
                         explicit={"Goal.state.sense.base_fps": 8.0})
        # default: explicit beats the profile's 0.5 coast binding
        res = ConfigResolver().resolve(plan, source_profile=SCREEN_DOCUMENT)
        self.assertEqual(res.effective["Goal.state.sense.base_fps"], 8.0)
        # strict: the profile coast is gone, the explicit floor still stands
        res_strict = ConfigResolver().resolve(plan, source_profile=SCREEN_DOCUMENT, execution_policy=_strict())
        self.assertEqual(res_strict.effective["Goal.state.sense.base_fps"], 8.0)


if __name__ == "__main__":
    unittest.main()
