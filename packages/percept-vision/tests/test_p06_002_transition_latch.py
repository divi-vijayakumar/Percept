"""P06-002 — INDEPENDENT acceptance suite: transition-latch identity keying.

Incident criterion (the intended POST-FIX behaviour these tests pin):

    A transition goal (rising edge from-state A -> to-state B) fires EXACTLY ONCE
    when A is observed then B is observed, REGARDLESS of which entity is "in focus"
    on each frame -- because the transition latch is keyed by the goal's DECLARED
    identity policy (defaulting to a stable "scene"), not by the volatile in-focus
    entity id. A goal that DECLARES a track/appearance identity policy gets
    independent per-entity latches instead. Transition judge calls are METERED
    (they appear in the funnel + emit judge.usage) and FAIL-CLOSED (a raising or
    malformed judge degrades the tick without mutating the latch or crashing).

These tests encode the intended behaviour, written solely from the behavioural
spec + the frozen public seams. On the CURRENT base several of them FAIL -- that
is correct: they catch the bug. The failure map is documented in the suite's
final report. They are written to run GREEN once the fix's public symbols
(``percept.identity_policy.SCENE_IDENTITY`` / ``transition_latch_key`` and the
additive ``transition.decision`` trace fields) exist.

Harness idiom only (NOT fix logic) borrowed from tests/test_transition.py and
tests/test_judge_dispatch_transition.py: how to build a Percept with a scripted
vision, feed frames, drive focus, and read incidents/traces.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Box, Goal, Verdict
from percept.agent import Percept
from percept.gate import FIRE
from percept.observability import funnel_key
from percept.tracing import (
    STAGE_JUDGE_USAGE,
    STAGE_JUDGMENT_EVENT,
    STAGE_TRANSITION_DECISION,
)

# The new public seam the fix introduces. Guard the import so THIS module still
# COLLECTS on the current base (letting the pure-behaviour negatives run there);
# the tests that exercise the seam assert it is present and fail cleanly, with a
# descriptive message, when it is not.
try:
    from percept.identity_policy import SCENE_IDENTITY, transition_latch_key

    _LATCH_SEAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on the pre-fix base
    SCENE_IDENTITY = "scene"
    transition_latch_key = None  # type: ignore[assignment]
    _LATCH_SEAM_AVAILABLE = False


# Generic condition text (no client/domain terminology).
FROM = "the region shows state A"
TO = "the region shows state B"

YES = (True, 0.9)   # confident satisfied
NO = (False, 0.9)   # confident not-satisfied
GOAL_ID = "g"
CAMERA = "camera"


def _tid(t: float) -> str:
    """The per-frame trace id perceive_judged derives (``source@now``)."""
    return f"{CAMERA}@{t}"


class _Focus:
    """Minimal stand-in for the entity graph's in-focus ``Entity`` (id / box / asserted()).

    The agent reads ``focus.id`` (-> the latch key on the buggy base), ``focus.box``
    and ``focus.asserted()``; nothing else. Driving focus this way gives each frame a
    deterministic in-focus entity id (or ``None``) without reverse-engineering the
    tracker's id assignment."""

    def __init__(self, fid: object) -> None:
        self.id = fid
        self.box = Box(0.1, 0.1, 0.2, 0.2)

    def asserted(self) -> dict:
        return {}


def _focus_for(fid: object) -> "_Focus | None":
    return None if fid is None else _Focus(fid)


class ScriptVision:
    """Scripts ``(condition, frame_token) -> (satisfied, confidence)``.

    ``detect`` yields nothing, so the entity graph stays empty and the test controls
    focus directly (via a patched ``graph.in_focus``). Verdicts optionally carry a
    ``usage`` dict so judge.usage metering is observable. ``raise_on`` / ``malformed_on``
    drive the fail-closed cases."""

    def __init__(self, script, *, usage=None, evidence=None,
                 raise_on=None, malformed_on=None) -> None:
        self._script = script
        self._usage = usage or {}
        self._evidence = evidence or {}
        self._raise_on = set(raise_on or ())
        self._malformed_on = set(malformed_on or ())

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        await asyncio.sleep(0)  # yield so a monotonic clock advances -> real latency
        tok = str(frame)
        if tok in self._raise_on:
            raise RuntimeError(f"judge failed hard on {tok!r}")
        if tok in self._malformed_on:
            return "not-a-verdict"  # a malformed, non-Verdict return
        sat, conf = self._script.get((condition, tok), NO)
        ev = self._evidence.get((condition, tok), f"{condition}@{tok}")
        return Verdict(satisfied=sat, confidence=conf, evidence=ev, usage=dict(self._usage))

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _run(frames, *, identity_policy=None, vision=None):
    """Drive the AGENT path over ``frames`` (each a dict with tok/from/to/focus[/t/*_ev]).

    Returns ``(fires, agent, traces)``. Focus is set PER FRAME by patching
    ``graph.in_focus``; each frame gets a distinct ``t`` (hence a distinct trace id)."""
    for i, f in enumerate(frames):
        f.setdefault("t", float(i + 1))
        _FRAME_T[f["tok"]] = f["t"]  # let a test recover a frame's trace id by token

    if vision is None:
        script, evidence = {}, {}
        for f in frames:
            script[(FROM, f["tok"])] = f.get("from", NO)
            script[(TO, f["tok"])] = f.get("to", NO)
            if "from_ev" in f:
                evidence[(FROM, f["tok"])] = f["from_ev"]
            if "to_ev" in f:
                evidence[(TO, f["tok"])] = f["to_ev"]
        vision = ScriptVision(script, evidence=evidence)

    traces: list = []
    a = Percept.create(vision=vision, discover_plugins=False, on_trace=traces.append)
    state = {"type": "transition", "from": FROM, "to": TO, "count": 0}
    if identity_policy is not None:
        state["identity_policy"] = identity_policy
    a.add_goal(Goal(id=GOAL_ID, condition="A becomes B", say="changed", state=state))

    fires: list = []

    async def go():
        for f in frames:
            a.graph.in_focus = (lambda _fid=f.get("focus"): _focus_for(_fid))  # per-frame focus
            evs = await a.perceive_judged(f["tok"], t=f["t"])
            fires.extend(e for e in evs if e.action == FIRE)

    asyncio.run(go())
    return fires, a, traces


# frames set t = index+1; this map lets a per-token lookup recover a frame's trace
# id in tests that need one. Populated by _run.
_FRAME_T: dict = {}


# ==================================================================================
# 1-3: a SCENE (default) transition fires exactly once across a focus change.
# ==================================================================================


class SceneLatchSurvivesFocusChange(unittest.TestCase):
    """Criterion: the default-scene latch is INDEPENDENT of the volatile in-focus
    entity, so a from->to edge fires exactly once however focus moves between the
    arming frame and the completing frame."""

    def test_1_arm_focus_entity_complete_focus_none(self):
        """#1 arm with focus=e1, complete with focus=None -> exactly ONE fire.

        Repro: on the buggy base the latch is keyed by ``focus.id``, so arming under
        ``e1`` and completing under ``scene`` (focus None) lands on a DIFFERENT latch
        and never fires."""
        fires, a, _ = _run([
            {"tok": "a1", "from": YES, "to": NO, "focus": "e1"},   # arm under e1
            {"tok": "a2", "from": NO, "to": YES, "focus": None},   # complete, focus gone
        ])
        self.assertEqual(len(fires), 1)
        self.assertEqual(a.goals.get(GOAL_ID).state["count"], 1)

    def test_2_arm_focus_none_complete_focus_entity(self):
        """#2 arm with focus=None, complete with focus=e1 -> exactly ONE fire."""
        fires, _, _ = _run([
            {"tok": "b1", "from": YES, "to": NO, "focus": None},   # arm under scene
            {"tok": "b2", "from": NO, "to": YES, "focus": "e1"},   # complete, e1 now in focus
        ])
        self.assertEqual(len(fires), 1)

    def test_3_arm_e1_complete_e2_largest_box_swap(self):
        """#3 arm focus=e1, complete focus=e2 (largest-box swap) -> exactly ONE fire."""
        fires, _, _ = _run([
            {"tok": "c1", "from": YES, "to": NO, "focus": "e1"},
            {"tok": "c2", "from": NO, "to": YES, "focus": "e2"},
        ])
        self.assertEqual(len(fires), 1)


# ==================================================================================
# 4-5: one-shot / re-arm hold across focus churn on the scene latch.
# ==================================================================================


class SceneLatchOneShotAndRearm(unittest.TestCase):
    def test_4_one_shot_under_focus_churn(self):
        """#4 after firing, repeated to=yes ticks under varying focus are
        SUPPRESSED (the scene latch is spent): exactly one fire total, and each
        churn tick reports outcome=='suppressed_latched'."""
        frames = [
            {"tok": "d0", "from": YES, "to": NO, "focus": "a"},   # arm
            {"tok": "d1", "from": NO, "to": YES, "focus": "a"},   # fire
            {"tok": "d2", "from": NO, "to": YES, "focus": "b"},   # churn -> suppressed
            {"tok": "d3", "from": NO, "to": YES, "focus": None},  # churn -> suppressed
            {"tok": "d4", "from": NO, "to": YES, "focus": "c"},   # churn -> suppressed
        ]
        fires, _, traces = _run(frames)
        self.assertEqual(len(fires), 1)
        for tok in ("d2", "d3", "d4"):
            row = _one_decision(traces, tok)
            self.assertEqual(
                row.data.get("outcome"), "suppressed_latched",
                f"churn tick {tok} should be suppressed by the spent scene latch",
            )

    def test_5_rearm_under_focus_churn(self):
        """#5 fire, then a later from=yes under a DIFFERENT focus re-arms the scene
        latch, and a following to=yes fires again -> exactly TWO fires total."""
        frames = [
            {"tok": "e0", "from": YES, "to": NO, "focus": "a"},   # arm
            {"tok": "e1", "from": NO, "to": YES, "focus": "a"},   # fire 1
            {"tok": "e2", "from": YES, "to": NO, "focus": "b"},   # re-arm (different focus)
            {"tok": "e3", "from": NO, "to": YES, "focus": "c"},   # fire 2 (different focus)
        ]
        fires, a, _ = _run(frames)
        self.assertEqual(len(fires), 2)
        self.assertEqual(a.goals.get(GOAL_ID).state["count"], 2)


# ==================================================================================
# 6: negatives stay quiet (these are expected to PASS even on the pre-fix base).
# ==================================================================================


class NegativesStayQuiet(unittest.TestCase):
    def test_6a_already_in_B_never_saw_A_with_focus_churn(self):
        """#6a joined mid-stream already in B (to=yes, from never yes) under focus
        churn -> ZERO fires. The 'must have seen A first' rule holds regardless of
        focus."""
        frames = [
            {"tok": f"f{i}", "from": NO, "to": YES, "focus": foc}
            for i, foc in enumerate(["a", "b", None, "c", "a"])
        ]
        fires, _, _ = _run(frames)
        self.assertEqual(fires, [])

    def test_6b_steady_no_edge_run(self):
        """#6b a steady run with no edge (from=no, to=no) under focus churn -> ZERO
        fires."""
        frames = [
            {"tok": f"g{i}", "from": NO, "to": NO, "focus": foc}
            for i, foc in enumerate([None, "a", "b", None, "c"])
        ]
        fires, _, _ = _run(frames)
        self.assertEqual(fires, [])


# ==================================================================================
# 7: opt-in per-entity isolation via a DECLARED track policy.
# ==================================================================================


class DeclaredTrackPolicyIsolation(unittest.TestCase):
    """Criterion: a goal that DECLARES ``identity_policy='track_iou_label'`` gets
    INDEPENDENT per-entity latches -- and when focus is absent on completion the
    key falls back to scene (``latch_key_source == 'fallback_scene'``)."""

    def test_7_per_entity_isolation_and_fallback(self):
        # Arm under e1; a to=yes under e2 must NOT fire (e2 is not armed); a to=yes
        # under e1 then fires.
        frames = [
            {"tok": "h0", "from": YES, "to": NO, "focus": "e1"},  # arm e1
            {"tok": "h1", "from": NO, "to": YES, "focus": "e2"},  # e2 not armed -> no fire
            {"tok": "h2", "from": NO, "to": YES, "focus": "e1"},  # e1 armed -> fire
        ]
        fires, _, traces = _run(frames, identity_policy="track_iou_label")
        self.assertEqual(len(fires), 1)

        # The declared-policy ticks key by the focus id, and the trace says so.
        for tok, ent in (("h0", "e1"), ("h1", "e2"), ("h2", "e1")):
            row = _one_decision(traces, tok)
            self.assertEqual(row.data.get("identity_policy"), "track_iou_label")
            self.assertEqual(row.data.get("latch_key"), ent)
            self.assertEqual(row.data.get("latch_key_source"), "declared")
            self.assertEqual(row.data.get("focus_entity_id"), ent)

        # Fallback: declared track policy but focus ABSENT on completion -> no fire,
        # and the key falls back to scene.
        fb_frames = [
            {"tok": "h3", "from": YES, "to": NO, "focus": "e1"},   # arm e1
            {"tok": "h4", "from": NO, "to": YES, "focus": None},   # focus gone -> fallback scene
        ]
        fb_fires, _, fb_traces = _run(fb_frames, identity_policy="track_iou_label")
        self.assertEqual(fb_fires, [], "a fallback-scene completion must not fire the e1 latch")
        fb_row = _one_decision(fb_traces, "h4")
        self.assertEqual(fb_row.data.get("latch_key"), "scene")
        self.assertEqual(fb_row.data.get("latch_key_source"), "fallback_scene")

        # And the pure seam agrees on the fallback mapping.
        self.assertTrue(_LATCH_SEAM_AVAILABLE, _SEAM_MSG)
        self.assertEqual(transition_latch_key("track_iou_label", None), ("scene", "fallback_scene"))


# ==================================================================================
# 8: metering + fail-closed + provenance.
# ==================================================================================

_USAGE = {"input_tokens": 12, "output_tokens": 4, "cost_usd": 0.002}


class MeteringFailClosedProvenance(unittest.TestCase):
    def test_8a_metered_two_judge_calls_and_usage_rows(self):
        """#8a a transition tick spends TWO judge calls that are METERED: two
        judge.usage rows under the frame's trace id, and nonzero recorded latency.
        Plus a firing tick's judgment.event joins to its source.frame."""
        # Metering: a single arm tick spends the from+to judge pair.
        script = {(FROM, "m0"): YES, (TO, "m0"): NO}
        vis = ScriptVision(script, usage=_USAGE)
        _, a, traces = _run([{"tok": "m0", "from": YES, "to": NO, "focus": None, "t": 1.0}],
                            vision=vis)

        usage_rows = [ev for ev in traces
                      if ev.stage == STAGE_JUDGE_USAGE and ev.trace_id == _tid(1.0)]
        self.assertEqual(len(usage_rows), 2,
                         "both transition judge calls must emit judge.usage under the frame trace id")
        self.assertEqual(a.metrics.get("judge.latency.count"), 2,
                         "both transition judge calls must record latency in the funnel")
        self.assertGreater(a.metrics.get("judge.latency.total_s"), 0.0)
        self.assertEqual(a.metrics.get(funnel_key("judged", concern=GOAL_ID)), 2)

        # Provenance: a firing transition tick emits a judgment.event carrying a
        # judgment_event_id, joinable to this frame's source.frame row.
        pf, _, ptraces = _run([
            {"tok": "p0", "from": YES, "to": NO, "focus": None},   # arm
            {"tok": "p1", "from": NO, "to": YES, "focus": None},   # fire
        ])
        self.assertEqual(len(pf), 1)
        fire_tid = _tid(2.0)  # second frame
        je = [ev for ev in ptraces
              if ev.stage == STAGE_JUDGMENT_EVENT and ev.trace_id == fire_tid]
        self.assertEqual(len(je), 1, "a transition fire must emit exactly one judgment.event")
        self.assertTrue(je[0].data.get("judgment_event_id"),
                        "the judgment.event must carry a judgment_event_id join key")
        src = [ev for ev in ptraces if ev.stage == "source.frame" and ev.trace_id == fire_tid]
        self.assertEqual(len(src), 1,
                         "judgment_event_id -> judgment.event -> source.frame join must resolve")

    def test_8b_miss_decision_row_carries_identity_and_confidence_fields(self):
        """#8b a MISS tick's transition.decision row carries the additive identity +
        confidence/evidence fields."""
        vis = ScriptVision(
            {(FROM, "n0"): (False, 0.88), (TO, "n0"): (False, 0.91)},
            evidence={(FROM, "n0"): "from-ev-xyz", (TO, "n0"): "to-ev-xyz"},
        )
        _, _, traces = _run([{"tok": "n0", "from": (False, 0.88), "to": (False, 0.91),
                              "focus": None, "t": 1.0}], vision=vis)
        row = _one_decision(traces, "n0")
        d = row.data
        for field in ("identity_policy", "latch_key", "latch_key_source", "focus_entity_id",
                      "from_confidence", "to_confidence", "from_evidence", "to_evidence"):
            self.assertIn(field, d, f"transition.decision must carry {field!r}")
        # Default (undeclared) policy -> scene, sourced as the default.
        self.assertEqual(d.get("latch_key"), "scene")
        self.assertEqual(d.get("latch_key_source"), "default_scene")
        self.assertIsNone(d.get("focus_entity_id"))
        self.assertEqual(d.get("from_confidence"), 0.88)
        self.assertEqual(d.get("to_confidence"), 0.91)
        self.assertEqual(d.get("from_evidence"), "from-ev-xyz")
        self.assertEqual(d.get("to_evidence"), "to-ev-xyz")

    def test_8c_fail_closed_on_raising_judge(self):
        """#8c a judge that RAISES degrades the tick: no exception escapes, the latch
        is not mutated, outcome=='judge_degraded', and a subsequent healthy tick still
        fires."""
        script = {(FROM, "r0"): YES, (TO, "r0"): NO,
                  (FROM, "r2"): NO, (TO, "r2"): YES}
        vis = ScriptVision(script, raise_on={"r1"})
        fires, _, traces = _run([
            {"tok": "r0", "from": YES, "to": NO, "focus": None},   # arm
            {"tok": "r1", "focus": None},                          # judge raises -> degrade
            {"tok": "r2", "from": NO, "to": YES, "focus": None},   # healthy -> fires
        ], vision=vis)
        self.assertEqual(len(fires), 1, "the healthy tick after a degrade must still fire")
        degraded = _one_decision(traces, "r1")
        self.assertEqual(degraded.data.get("outcome"), "judge_degraded")

    def test_8c_fail_closed_on_malformed_verdict(self):
        """#8c (variant) a judge that returns a malformed NON-Verdict degrades the tick
        identically: no crash, latch intact, outcome=='judge_degraded', later fire OK."""
        script = {(FROM, "s0"): YES, (TO, "s0"): NO,
                  (FROM, "s2"): NO, (TO, "s2"): YES}
        vis = ScriptVision(script, malformed_on={"s1"})
        fires, _, traces = _run([
            {"tok": "s0", "from": YES, "to": NO, "focus": None},
            {"tok": "s1", "focus": None},                          # malformed return -> degrade
            {"tok": "s2", "from": NO, "to": YES, "focus": None},
        ], vision=vis)
        self.assertEqual(len(fires), 1)
        degraded = _one_decision(traces, "s1")
        self.assertEqual(degraded.data.get("outcome"), "judge_degraded")


# ==================================================================================
# Direct unit coverage of the frozen seam: transition_latch_key (every mapping row).
# ==================================================================================

_SEAM_MSG = (
    "percept.identity_policy.SCENE_IDENTITY / transition_latch_key is missing -- the "
    "P06-002 fix seam is not present on this build."
)


class TransitionLatchKeyMapping(unittest.TestCase):
    """Every row of the frozen ``transition_latch_key(declared_policy, focus_id)``
    contract. Fails cleanly (descriptive message) on a build without the seam."""

    def setUp(self):
        self.assertTrue(_LATCH_SEAM_AVAILABLE, _SEAM_MSG)

    def test_scene_identity_constant(self):
        self.assertEqual(SCENE_IDENTITY, "scene")

    def test_none_defaults_to_scene(self):
        self.assertEqual(transition_latch_key(None, "e1"), ("scene", "default_scene"))
        self.assertEqual(transition_latch_key(None, None), ("scene", "default_scene"))

    def test_declared_scene(self):
        self.assertEqual(transition_latch_key("scene", "e1"), ("scene", "declared"))
        self.assertEqual(transition_latch_key("scene", None), ("scene", "declared"))

    def test_track_policy_with_focus_present_is_per_entity(self):
        self.assertEqual(transition_latch_key("track_iou_label", "e1"), ("e1", "declared"))
        self.assertEqual(
            transition_latch_key("appearance_recent_set", "e9"), ("e9", "declared")
        )

    def test_track_policy_with_focus_absent_falls_back_to_scene(self):
        self.assertEqual(
            transition_latch_key("track_iou_label", None), ("scene", "fallback_scene")
        )
        self.assertEqual(
            transition_latch_key("appearance_recent_set", None), ("scene", "fallback_scene")
        )

    def test_non_track_policies_fall_back_to_scene(self):
        for policy in ("content_key", "speaker_label", "something_custom"):
            self.assertEqual(
                transition_latch_key(policy, "e1"), ("scene", "fallback_scene"),
                f"{policy!r} is neither scene nor a track policy -> fallback_scene",
            )


# ----------------------------------------------------------------------------------
# Small helper: fetch the single transition.decision row for a frame token. Uses the
# per-token trace-id map _run populated, so a test names the frame, not the timestamp.
# ----------------------------------------------------------------------------------


def _one_decision(traces, tok):
    tid = _tid(_FRAME_T[tok])
    rows = [ev for ev in traces
            if ev.stage == STAGE_TRANSITION_DECISION and ev.trace_id == tid]
    if len(rows) != 1:
        raise AssertionError(
            f"expected exactly one transition.decision row for {tok!r} (trace {tid}), "
            f"got {len(rows)}"
        )
    return rows[0]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
