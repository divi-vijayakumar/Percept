"""The rhythm sense — the core perceptual primitive that counts a repeating cycle (rep counting et al.).

L1 discipline: the tracker is exercised on SYNTHETIC scalar signals (no pose, no video, no model), so the
counting logic is asserted exactly and the tests stay fast/deterministic. The pose *measurement* (the eye)
is exercised separately with FAKE COCO keypoints. Together they prove "the eye measures, the brain decides"
without asserting any backend intelligence.
"""

from __future__ import annotations

import math
import unittest

from percept import Goal, Percept
from percept.behavior import coerce_authored_rhythm, program_of
from percept.rhythm import RhythmTracker


def _wave(n_cycles, T=12, amp=1.0, fps=6.0, jitter=0.0, decay=1.0, channel="c"):
    """A clean (optionally jittered / decaying) oscillation: ``n_cycles`` periods of ``T`` samples."""
    out = []
    for i in range(n_cycles * T):
        a = amp * (decay ** (i / T))
        v = a * math.sin(2 * math.pi * i / T)
        if jitter:
            v += jitter * math.sin(2 * math.pi * i / 3.0)  # fast sub-amplitude wobble
        out.append((i / fps, {channel: v}))
    return out


def _wave_hz(freq_hz, duration_s, fps, channel="c"):
    """A pure oscillation of a given physical FREQUENCY + duration, sampled at ``fps`` — so the SAME signal
    can be fed at different rates to test rate-agnosticism."""
    n = int(duration_s * fps)
    return [(i / fps, {channel: math.sin(2 * math.pi * freq_hz * i / fps)}) for i in range(n)]


def _run(samples, **prior):
    tr = RhythmTracker(**prior)
    last_count_idx = None
    for i, (t, ch) in enumerate(samples):
        if tr.observe(t, ch) is not None:
            last_count_idx = i
    return tr, last_count_idx


class RhythmTrackerSignalTests(unittest.TestCase):
    """Causal, baseline-relative counting on synthetic signals — the heart of the rhythm sense."""

    def test_clean_counts_exactly(self):
        for n in (1, 5, 16):  # n=1 is the "count to 1 jumping jack" cold-start case
            tr, _ = _run(_wave(n))
            self.assertEqual(tr.count, n, f"clean {n}-cycle wave")

    def test_rejects_sub_amplitude_jitter(self):
        tr, _ = _run(_wave(16, jitter=0.15))
        self.assertEqual(tr.count, 16)

    def test_robust_to_decaying_amplitude(self):
        # The non-causal-normalization failure case: reps shrink (fatigue); baseline-relative still counts.
        self.assertEqual(_run(_wave(16, decay=0.92))[0].count, 16)
        self.assertEqual(_run(_wave(16, decay=0.85))[0].count, 16)

    def test_robust_to_variable_cadence(self):
        fast = _wave(8, T=8)
        slow = [(t + fast[-1][0] + 1 / 6, ch) for (t, ch) in _wave(8, T=18)]
        self.assertEqual(_run(fast + slow)[0].count, 16)

    def test_pause_injects_no_phantom_rep(self):
        s = _wave(8) + [((96 + j) / 6.0, {"c": 0.0}) for j in range(40)] + [(t + 30, ch) for (t, ch) in _wave(8)]
        self.assertEqual(_run(s)[0].count, 16)

    def test_count_is_causal_not_end_of_stream(self):
        # The count must increment when a rep COMPLETES, not retrospectively at the end (count-out-loud).
        tr, last = _run(_wave(3, T=12))
        self.assertEqual(tr.count, 3)
        self.assertIsNotNone(last)
        self.assertLess(last, 3 * 12 - 4, "last rep counted well before the final sample")

    def test_count_is_rate_agnostic_across_fps(self):
        # The same 0.5 Hz / 10-rep signal yields the SAME count at 6/15/30/60 fps — the time-based debounce
        # makes the counter independent of the sampling rate (the whole point of the retina/fovea split).
        for freq, dur, reps in [(0.5, 20, 10), (0.2, 25, 5)]:
            counts = {fps: _run(_wave_hz(freq, dur, fps))[0].count for fps in (6, 15, 30, 60)}
            self.assertEqual(set(counts.values()), {reps}, f"{freq}Hz counts differ by fps: {counts}")

    def test_fast_reps_recovered_only_at_high_capture_rate(self):
        # A 2 Hz / 10-rep signal is ALIASED at 6 fps (~3 samples/rep) and undercounts badly; capturing at
        # 30 fps recovers it. No counter can fix the 6 fps case — the reps aren't in the data (Nyquist).
        slow = _run(_wave_hz(2.0, 5, 6))[0].count
        fast = _run(_wave_hz(2.0, 5, 30))[0].count
        self.assertLessEqual(slow, 2, f"6fps should alias a 2Hz rep, got {slow}")
        self.assertGreaterEqual(fast, 8, f"30fps should recover ~10, got {fast}")

    def test_prominence_rejects_aborted_low_reps(self):
        # "peak not achieved?" — once big reps establish the amplitude, a much shallower bump (continuous
        # with the same baseline, so no DC-step artifact) is a sub-prominence aborted turn and must NOT
        # count. Three triangle reps to 1.0, then a 0.15 wobble -> 3.
        vals = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.075, 0.15, 0.075, 0]
        tr, _ = _run([(i / 6.0, {"c": v}) for i, v in enumerate(vals)])
        self.assertEqual(tr.count, 3)

    def test_selects_rep_channel_over_noise_and_reports_span(self):
        base = _wave(10, channel="rep")
        for i, (_t, ch) in enumerate(base):
            ch["noise"] = 0.05 * math.sin(2 * math.pi * i / 2.3)  # small, fast, off-cadence
        tr, _ = _run(base)
        self.assertEqual(tr.count, 10)
        self.assertEqual(tr.selected, "rep")
        rep = tr.reps[0]
        self.assertLess(rep.start_t, rep.peak_t)
        self.assertLess(rep.peak_t, rep.stop_t)  # a first-class start -> peak -> stop SPAN

    def test_prior_channel_biases_selection_but_cannot_force_a_miscount(self):
        # A prior naming a DEAD channel is ignored (a channel with zero reps is never selected).
        base = _wave(6, channel="rep")
        for _t, ch in base:
            ch["dead"] = 0.0
        tr, _ = _run(base, prior_channel="dead")
        self.assertEqual(tr.count, 6)
        self.assertEqual(tr.selected, "rep")


class RhythmProgramTests(unittest.TestCase):
    def test_program_of_compiles_rhythm_sense(self):
        g = Goal(id="r", condition="a completed squat repetition", say="rep", state={"type": "rhythm"})
        prog = program_of(g)
        self.assertEqual(prog.sense, "rhythm")
        self.assertEqual(prog.accumulate.kind, "count_distinct")

    def test_program_of_carries_rhythm_fire_when(self):
        g = Goal(id="r", condition="rep", say="ten", state={"type": "rhythm", "fire_when": "n == 10"})
        self.assertEqual(program_of(g).accumulate.fire_when, "n == 10")

    def test_rhythm_prior_firewall_keeps_valid_drops_garbage(self):
        ok = coerce_authored_rhythm({"channel": "limb-vertical", "cadence_s": 2.0, "whole_body": True})
        self.assertEqual(ok["rhythm"]["channel"], "limb-vertical")
        self.assertEqual(ok["rhythm"]["cadence_s"], 2.0)
        self.assertIs(ok["rhythm"]["whole_body"], True)
        self.assertEqual(coerce_authored_rhythm({"channel": 123, "cadence_s": -5}), {})  # bad types/values
        self.assertEqual(coerce_authored_rhythm("nope"), {})
        self.assertEqual(coerce_authored_rhythm({"channel": "x" * 100}), {})  # over-long hint dropped


class RhythmAgentTests(unittest.TestCase):
    """The rhythm sense through the full core: perceive_rhythm -> fire_episode -> gate."""

    def _core(self, **state):
        core = Percept.create(discover_plugins=False)
        core.add_goal(Goal(id="reps", condition="a completed jumping jack repetition", say="rep",
                           state={"type": "rhythm", **state}))
        return core

    def test_perceive_rhythm_counts_complete_reps(self):
        core = self._core()
        fires = 0
        for i in range(16 * 12):
            if core.perceive_rhythm("reps", t=i / 6.0, channels={"limb-vertical": math.sin(2 * math.pi * i / 12)}):
                fires += 1
        self.assertEqual(core.edge_counts["reps"], 16)
        self.assertEqual(core.goals.get("reps").state["count"], 16)
        self.assertEqual(fires, 16)

    def test_fire_when_gates_user_fire_but_not_the_count(self):
        core = self._core(fire_when="n == 10")
        fires = sum(
            bool(core.perceive_rhythm("reps", t=i / 6.0, channels={"c": math.sin(2 * math.pi * i / 12)}))
            for i in range(16 * 12)
        )
        self.assertEqual(core.edge_counts["reps"], 16)  # all reps counted
        self.assertEqual(fires, 1)  # only the 10th raises a user-facing fire

    def test_legacy_perceive_boundary_still_blind_increments(self):
        # Backward compatibility: the old boundary path is an untouched +1 per call.
        core = Percept.create(discover_plugins=False)
        core.add_goal(Goal(id="g", condition="rep", say="x", state={"type": "watch"}))
        for k in range(3):
            core.perceive_boundary("g", t=float(k))
        self.assertEqual(core.edge_counts["g"], 3)


if __name__ == "__main__":
    unittest.main()
