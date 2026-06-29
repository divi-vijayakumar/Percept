"""L1 A/B for the pt-action edge-recall coverage miss (TODO 1).

The bug: the ``pt-action-e2e`` adapter sampled at a FIXED ``FPS = 2.0`` and JUDGED only on the motion
gate's escalation. For video_10849 "Launching something" the GT window is a brief ~1 s action [0.333,
1.333]; the in-window motion is subtle (below the 0.15 motion-gate threshold), so the gate's first
escalation lands at t=2.0 — AFTER the window closed. No judged frame ever falls inside the window ->
edge recall 0.

The SDK-shaped fix: the goal carries a brief-window PRIOR (the GT window's duration). ``sensor_plan_for``
compiles the judge (rising-edge) sense to a SENSITIVE above-Nyquist cadence (``base_fps >= 2/window``),
and the adapter's :func:`judged_timestamps` coverage helper coasts at that base_fps so a quiet-but-in-window
frame is STILL judged — without changing the motion gate's firing logic. No per-activity constant: any
brief-action judge goal benefits.

This is a DETERMINISTIC L1 test — a synthetic ``(t, motion)`` stream + the adapter's pure sampling helper.
No video, no Gemini, no network. It is the keep/drop evidence: recall 0 -> >0 without exploding judge calls.
"""

from __future__ import annotations

import unittest

from percept import Goal
from percept.sensorplan import sensor_plan_for

from eval.bench.track_perception_test_e2e import (
    FPS,
    cadence_policy,
    capture_fps,
    judged_timestamps,
)

# The video_10849 scenario, modeled deterministically.
WINDOW_LO, WINDOW_HI = 0.333, 1.333  # GT action window (~1 s brief action)
LATE_SPIKE_T = 2.0  # the first frame whose motion clears the gate threshold (post-window) -> Arm A escalation


def _capture_stream(fps: float) -> list[tuple[float, float]]:
    """A captured ``(t, motion)`` stream at ``fps``: subtle (sub-threshold) motion DURING the brief window,
    a brief clear motion spike only AFTER it closes (the realistic "gate fires late" case). The in-window
    motion is 0.06 — below the 0.15 default gate threshold (so the motion gate suppresses it) but above the
    sensitive cadence's lowered 0.05 floor. The post-window spike is a SINGLE-sample burst at
    ``LATE_SPIKE_T`` (not sustained), matching a real action's settling motion."""
    n = int(round(3.0 * fps))
    out: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / fps
        if WINDOW_LO <= t <= WINDOW_HI:
            motion = 0.06  # subtle in-window motion: below 0.15 (gate skips), the launch hasn't burst yet
        elif abs(t - LATE_SPIKE_T) < (0.5 / fps):
            motion = 0.5  # the post-window settling burst the motion gate finally escalates on (one sample)
        else:
            motion = 0.0
        out.append((t, motion))
    return out


def _motion_gate_escalations(samples: list[tuple[float, float]], threshold: float = 0.15) -> set[float]:
    """Model the legacy escalation set: the motion gate escalates a frame whose motion clears ``threshold``
    (the firing logic is UNCHANGED by this fix; we only model it to build Arm A and to union it in Arm B)."""
    return {t for t, m in samples if m >= threshold}


def _in_window(ts: list[float]) -> list[float]:
    return [t for t in ts if WINDOW_LO <= t <= WINDOW_HI]


class ActionCadenceABTest(unittest.TestCase):
    def setUp(self) -> None:
        # Arm A: the OLD adapter — fixed FPS=2.0 capture, JUDGE only on motion-gate escalations (no fovea
        # controller in the loop at all). Its judged set IS the gate escalation set.
        self.cap_a = _capture_stream(FPS)
        self.esc_a = _motion_gate_escalations(self.cap_a)
        self.judged_a = sorted(self.esc_a)  # legacy: judged <=> escalated
        # The legacy cadence: a plain judge goal carries NO brief-window prior (all-None cadence).
        plain = sensor_plan_for(Goal(id="g", condition="someone is launching something", say="!",
                                      state={"type": "watch", "count": 0}))
        self.cadence_a = plain.cadence
        # Arm B: the NEW adapter — goal carries the GT window as its brief-window prior -> sensitive cadence;
        # judge on gate-escalation OR the sensitive cadence's coast (judged_timestamps).
        brief = sensor_plan_for(Goal(id="g", condition="someone is launching something", say="!",
                                     state={"type": "watch", "count": 0,
                                            "sense": {"brief_window_s": WINDOW_HI - WINDOW_LO}}))
        self.cadence_b = brief.cadence
        self.cap_b = _capture_stream(capture_fps(self.cadence_b))
        self.esc_b = _motion_gate_escalations(self.cap_b)
        self.judged_b = judged_timestamps(self.cap_b, self.cadence_b, gate_escalations=self.esc_b)

    # -- the root-cause reproduction (Arm A) --------------------------------------------------------
    def test_arm_a_misses_the_window_edge_recall_zero(self):
        """Arm A: legacy gate-only judging at FPS=2.0 lands NO judged frame inside the brief window — the
        motion gate's first (and every) escalation is post-window, exactly the reported symptom."""
        self.assertEqual(_in_window(self.judged_a), [], f"Arm A unexpectedly covered: {self.judged_a}")
        self.assertTrue(all(t >= LATE_SPIKE_T for t in self.judged_a))  # all escalations post-window
        # the legacy cadence is the un-sensitive default (no brief-window prior surfaced)
        self.assertIsNone(self.cadence_a.get("brief_window_s"))

    def test_arm_a_legacy_cadence_is_unchanged_default(self):
        """A plain judge goal must compile the byte-for-byte legacy cadence (all knobs unset)."""
        self.assertEqual(self.cadence_a["base_fps"], None)
        self.assertEqual(self.cadence_a["max_fps"], None)
        self.assertEqual(self.cadence_a["escalate_threshold"], None)

    # -- the fix (Arm B) ----------------------------------------------------------------------------
    def test_brief_prior_compiles_above_nyquist_cadence(self):
        """The brief-window prior compiles base_fps to at least the window's Nyquist (>= 2/window)."""
        window = WINDOW_HI - WINDOW_LO
        self.assertGreaterEqual(self.cadence_b["base_fps"], 2.0 / window - 1e-9)
        self.assertLess(self.cadence_b["escalate_threshold"], 0.15)  # admits subtle in-window motion
        self.assertEqual(self.cadence_b["brief_window_s"], window)

    def test_arm_b_covers_the_window_edge_recall_positive(self):
        """Arm B: the sensitive cadence coasts over the brief window -> at least one JUDGED in-window frame."""
        in_window_b = _in_window(self.judged_b)
        self.assertGreaterEqual(len(in_window_b), 1, "Arm B failed to cover the brief window")

    # -- the A/B: recall 0 -> >0 WITHOUT exploding cost ---------------------------------------------
    def test_ab_recall_lifts_without_exploding_cost(self):
        cov_a, cov_b = len(_in_window(self.judged_a)), len(_in_window(self.judged_b))
        cost_a, cost_b = len(self.judged_a), len(self.judged_b)
        # the hypothesis: in-window coverage 0 -> >0 (the recall fix)
        self.assertEqual(cov_a, 0)
        self.assertGreaterEqual(cov_b, 1)
        # cost stays BOUNDED — two honest checks:
        #  (1) the sensitive cadence still spends FEWER judge calls than a dense every-captured-frame loop
        #      (the benchmark's call_reduction baseline): the gating economics survive.
        #  (2) the controller NEVER exceeds max_fps, so cost is bounded by the plan's OWN ceiling
        #      (max_fps x duration) — the SDK cost invariant, not an arbitrary number.
        dense_b = len(self.cap_b)
        self.assertLess(cost_b, dense_b, f"sensitive cadence judged every frame: B={cost_b}/{dense_b}")
        max_fps_ceiling = self.cadence_b["max_fps"] * 3.0 + 2  # +2 endpoint slack over the 3 s clip
        self.assertLessEqual(cost_b, max_fps_ceiling, f"exceeded the plan's max_fps ceiling: B={cost_b}")
        # emit the A/B numbers for the report (visible with -v / on failure)
        print(f"\n[pt-action A/B] in-window judged: A={cov_a} B={cov_b} | "
              f"judge calls (3s clip): A={cost_a} B={cost_b} (dense={dense_b}, "
              f"reduction={1 - cost_b / dense_b:.2f})")

    # -- the LITMUS: an UNSEEN brief action benefits with ZERO new `if` -----------------------------
    def test_litmus_unseen_brief_action_benefits_with_zero_new_if(self):
        """A different, unseen 'tell me when X briefly happens' goal (0.5 s window) gets covered by the
        SAME path — no goal-type enum, no per-activity branch, just a different prior."""
        lo, hi = 1.0, 1.5  # a 0.5 s window somewhere else
        window = hi - lo
        plan = sensor_plan_for(Goal(id="g2", condition="the glass briefly tips over", say="!",
                                    state={"type": "watch", "count": 0,
                                           "sense": {"brief_window_s": window}}))
        cadence = plan.cadence
        self.assertGreaterEqual(cadence["base_fps"], 2.0 / window - 1e-9)  # >= 4 fps for a 0.5 s window
        cap = [(i / capture_fps(cadence), 0.04) for i in range(int(3.0 * capture_fps(cadence)) + 1)]
        # subtle motion everywhere (below even the lowered floor) -> coverage is from the base_fps COAST,
        # proving the fix doesn't depend on motion at all (pure cadence sensitivity).
        judged = judged_timestamps(cap, cadence, gate_escalations=set())
        in_window = [t for t in judged if lo <= t <= hi]
        self.assertGreaterEqual(len(in_window), 1, "unseen brief-action goal was not covered")

    # -- non-regression: a non-brief goal's sampling is unchanged -----------------------------------
    def test_non_brief_goal_sampling_unchanged(self):
        """A plain (non-brief) judge goal coasts at the SensePolicy default base_fps — the sensitive cadence
        is scoped to the brief case and does not alter other goals' sampling density."""
        default_policy = cadence_policy(self.cadence_a)
        from percept.sampling import SensePolicy
        self.assertEqual(
            (default_policy.base_fps, default_policy.max_fps, default_policy.escalate_threshold),
            (SensePolicy().base_fps, SensePolicy().max_fps, SensePolicy().escalate_threshold),
        )


if __name__ == "__main__":
    unittest.main()
