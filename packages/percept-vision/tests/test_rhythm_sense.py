"""The rhythm sense — the core perceptual primitive that counts a repeating cycle (rep counting et al.).

L1 discipline: the tracker is exercised on SYNTHETIC scalar signals (no pose, no video, no model), so the
counting logic is asserted exactly and the tests stay fast/deterministic. The pose *measurement* (the eye)
is exercised separately with FAKE COCO keypoints. Together they prove "the eye measures, the brain decides"
without asserting any backend intelligence.
"""

from __future__ import annotations

import math
import random
import unittest
from collections import deque


from percept import Goal, Percept
from percept.behavior import coerce_authored_rhythm, program_of
from percept.harness.detectors import DetectorManifest, register, unregister
from percept.harness.pose import FrameNotDecodable
from percept.measurement import ChannelScore
from percept.rhythm import (
    MeasurementRecord,
    Rep,
    RhythmDrive,
    RhythmTracker,
    _ChannelState,
    has_evidence,
    select_channel,
)


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


# --- helpers for the boundary suites below ----------------------------------------------------------

def _fold(pairs, **knobs):
    """Fold ``(t, value)`` pairs into ONE channel detector; return ``(state, [confirmed reps])``.

    One channel, so the selection layer is out of the picture and the counting guards are asserted
    on their own."""
    st = _ChannelState(name="c", **knobs)
    reps = []
    for t, v in pairs:
        rep = st.observe(t, v)
        if rep is not None:
            reps.append(rep)
    return st, reps


def _at(values, dt=1 / 6.0):
    """Stamp a bare value list onto a fixed time step (6 fps by default, the cached-bench rate)."""
    return [(i * dt, float(v)) for i, v in enumerate(values)]


def _fabricate_channel(name, n_reps, amp, intervals=()):
    """A channel detector with a hand-authored rep tally / amplitude / interval set.

    The legacy selector reads exactly those three things, so authoring them directly isolates
    *selection* from *detection* — the alternative (synthesising a signal that happens to yield 40
    reps on one channel and 2 on another) proves less and is far more brittle."""
    st = _ChannelState(name=name, amp=amp)
    st.intervals = list(intervals)
    st.reps = [
        Rep(n=i + 1, channel=name, start_t=float(i), peak_t=i + 0.3, stop_t=i + 0.6, amplitude=amp)
        for i in range(n_reps)
    ]
    return st


def _tones(duration_s, fps, specs):
    """``[(t, {channel: value})]`` — several pure tones of different frequency on one clock, so the
    SAME physical signal can be re-sampled at another rate and the two answers compared."""
    return [
        (i / fps, {ch: math.sin(2 * math.pi * freq * i / fps) for ch, freq in specs})
        for i in range(int(duration_s * fps))
    ]


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

    def test_expected_rom_gate_rejects_small_reps_relative_guard_misses(self):
        # The fidget case: big reps to 1.0 (prom 1.0) then small reps to 0.4 (prom 0.4). The RELATIVE guard
        # keeps the 0.4 reps (0.4 > 0.2×amp), but the ABSOLUTE expected-ROM gate (min_amp = 0.5×1.0) rejects
        # them. This is the movement-too-small-to-be-a-rep case the relative guard can't see.
        big = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0]
        small = [0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0]
        samples = [(i / 6.0, {"c": v}) for i, v in enumerate(big + small)]
        ungated, _ = _run(samples, prior_channel="c")
        gated, _ = _run(samples, prior_channel="c", expected_rom=1.0)  # min_amp = 0.5
        self.assertGreater(ungated.count, gated.count)
        self.assertEqual(gated.count, 3)  # only the full-ROM reps survive

    def test_expected_rom_off_by_default_is_backward_compatible(self):
        big = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0]
        small = [0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0]
        samples = [(i / 6.0, {"c": v}) for i, v in enumerate(big + small)]
        tr, _ = _run(samples, prior_channel="c")  # no expected_rom → no absolute gate
        self.assertGreaterEqual(tr.count, 5)  # both big and small counted

    def test_expected_rom_binds_only_the_prior_channel(self):
        # expected_rom is in the PRIOR channel's native units, so it must NOT gate a different channel.
        big = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0]
        small = [0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0]
        samples = [(i / 6.0, {"other": v}) for i, v in enumerate(big + small)]
        tr, _ = _run(samples, prior_channel="c", expected_rom=1.0)  # prior names absent "c"; "other" ungated
        self.assertGreaterEqual(tr.count, 5)

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


class RhythmRateInvarianceTests(unittest.TestCase):
    """Build 2 — the causal fixed-time (period-adaptive) low-pass keeps the count from COLLAPSING as fps
    rises on NOISY signals (the rate-invariance fix), while staying exactly lossless on clean signals.
    Synthetic + deterministic (seeded white noise), so the contract is asserted without pose/video."""

    @staticmethod
    def _noisy_count(freq, n_reps, fps, sigma, smooth_s, seed=1234):
        rng = random.Random(seed)
        tr = RhythmTracker(smooth_s=smooth_s)
        for i in range(int(n_reps / freq * fps)):
            t = i / fps
            tr.observe(t, {"c": math.sin(2 * math.pi * freq * t) + sigma * rng.gauss(0, 1)})
        return tr.count

    def test_smoothing_prevents_high_fps_collapse_on_noise(self):
        # The pathology + the fix, side by side at 60 fps: legacy (smooth_s=0) collapses; the default holds.
        gt = 12
        legacy = self._noisy_count(0.5, gt, 60.0, 0.30, smooth_s=0.0)
        smoothed = self._noisy_count(0.5, gt, 60.0, 0.30, smooth_s=0.15)
        self.assertLessEqual(legacy, gt // 2, f"legacy counter should collapse at high fps, got {legacy}")
        self.assertGreaterEqual(smoothed, int(0.8 * gt), f"smoothed counter should hold recall, got {smoothed}")

    def test_smoothed_count_does_not_collapse_across_fps(self):
        # Above the Nyquist floor the noisy count holds >=80% recall at every rate (no denser-is-worse).
        gt = 12
        counts = {fps: self._noisy_count(0.5, gt, fps, 0.30, smooth_s=0.15) for fps in (6, 12, 30, 60, 160)}
        for fps, c in counts.items():
            self.assertGreaterEqual(c, int(0.8 * gt), f"recall collapsed at {fps}fps: {counts}")

    def test_smoothing_is_lossless_on_clean_signal(self):
        # The default smoothing must not change a clean count at any fps (no regression on noise-free signals).
        for fps in (6, 30, 160):
            self.assertEqual(self._noisy_count(0.5, 12, fps, 0.0, smooth_s=0.15), 12, f"clean changed @ {fps}fps")


class CanonicalSelectorTests(unittest.TestCase):
    """Build 3 — the canonical channel selector + its hysteresis policy. The selector picks the rep-owning
    channel on the canonical grid (rate-invariant) with a prior/margin-over-dwell guard; it is OPT-IN, so
    the default tracker is the unchanged legacy medoid."""

    @staticmethod
    def _score(ch, score):
        return ChannelScore(ch, score=score, support=1.0, amplitude=1.0, regularity=1.0, stability=1.0, period_s=1.0)

    def test_policy_trusts_viable_prior_but_ignores_dead_prior(self):
        scores = {"hip": self._score("hip", 10.0), "torso": self._score("torso", 5.0)}
        self.assertEqual(select_channel(scores, prior="torso"), "torso")  # prior wins despite lower score
        dead = {"hip": self._score("hip", 10.0), "dead": self._score("dead", 0.0)}
        self.assertEqual(select_channel(dead, prior="dead"), "hip")  # a zero-score prior is not viable

    def test_policy_margin_holds_incumbent_until_decisive(self):
        close = {"a": self._score("a", 1.1), "b": self._score("b", 1.0)}
        self.assertEqual(select_channel(close, current="b", margin=0.25), "b")  # 10% lead < 25% margin -> hold
        decisive = {"a": self._score("a", 1.6), "b": self._score("b", 1.0)}
        self.assertEqual(select_channel(decisive, current="b", margin=0.25), "a")  # 60% lead -> switch

    def test_policy_empty_returns_current(self):
        self.assertEqual(select_channel({}, current="x"), "x")
        self.assertIsNone(select_channel({"d": self._score("d", 0.0)}))

    def test_canonical_tracker_picks_rep_channel_over_noise(self):
        rng = random.Random(3)
        tr = RhythmTracker(canonical_select=True, select_band_s=(1.0, 5.0))
        for i in range(int(24 * 30)):  # 24 s @ 30 fps, 0.5 Hz -> ~12 reps
            t = i / 30.0
            tr.observe(t, {"rep": math.sin(2 * math.pi * 0.5 * t), "noise": 0.05 * rng.gauss(0, 1)})
        self.assertEqual(tr.selected, "rep")
        self.assertGreaterEqual(tr.count, 10)

    def test_canonical_select_is_off_by_default(self):
        self.assertFalse(RhythmTracker().canonical_select)


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

    def test_smooth_s_firewall_bounds_the_optin_knob(self):
        # smooth_s is an opt-in rate-invariance knob: accepted in (0, 1.0], clamped, else dropped (default off).
        self.assertEqual(coerce_authored_rhythm({"smooth_s": 0.15})["rhythm"]["smooth_s"], 0.15)
        self.assertNotIn("smooth_s", coerce_authored_rhythm({"smooth_s": 0}).get("rhythm", {}))  # 0 = off
        self.assertNotIn("smooth_s", coerce_authored_rhythm({"smooth_s": -1}).get("rhythm", {}))
        self.assertNotIn("smooth_s", coerce_authored_rhythm({"smooth_s": 5.0}).get("rhythm", {}))  # over max
        self.assertNotIn("smooth_s", coerce_authored_rhythm({"smooth_s": "x"}).get("rhythm", {}))

    def test_canonical_select_firewall_is_strict_bool(self):
        self.assertIs(coerce_authored_rhythm({"canonical_select": True})["rhythm"]["canonical_select"], True)
        self.assertNotIn("canonical_select", coerce_authored_rhythm({"canonical_select": "yes"}).get("rhythm", {}))
        self.assertNotIn("canonical_select", coerce_authored_rhythm({"canonical_select": 1}).get("rhythm", {}))

    def test_expected_rom_firewall_channel_bound_and_unit_marked(self):
        ok = coerce_authored_rhythm({"channel": "limb-vertical", "expected_rom": 0.18})["rhythm"]
        self.assertEqual(ok["expected_rom"], 0.18)
        self.assertEqual(ok["expected_rom_space"], "channel_native")  # marked, so units can't be confused later
        # dropped WITHOUT a channel — expected_rom has no units without a channel to be native to
        self.assertNotIn("expected_rom", coerce_authored_rhythm({"expected_rom": 0.18}).get("rhythm", {}))
        # dropped for a non-channel-native space — the forward guard against frame-normalized numbers
        self.assertNotIn("expected_rom", coerce_authored_rhythm(
            {"channel": "c", "expected_rom": 0.18, "expected_rom_space": "frame_normalized"})["rhythm"])
        self.assertNotIn("expected_rom", coerce_authored_rhythm({"channel": "c", "expected_rom": -1})["rhythm"])


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


class CausalRepDetectorBoundaryTests(unittest.TestCase):
    """The per-channel causal detector at its edges.

    Three things the signal-level suite above cannot reach: the exact-equality behaviour of each
    admission guard (all four are ``>=``, so the value ON the bar counts), the six tunables no other
    test gives a value to, and the two ordinary inputs — a peak held flat, an outsized opening rep —
    that silently cost a caller real repetitions.
    """

    # --- streams with no countable turn ------------------------------------------------------------
    def test_a_stream_too_short_or_too_flat_to_turn_never_counts(self):
        self.assertEqual(_fold(_at([0.5]))[1], [])           # one sample: only the seed, no first diff
        self.assertEqual(_fold(_at([0.0, 1.0]))[1], [])      # two samples: one diff, no reversal
        self.assertEqual(_fold(_at([0.3] * 20))[1], [])      # every step flat -> every step a no-op
        ramp = [i * 0.1 for i in range(30)]
        self.assertEqual(_fold(_at(ramp))[1], [])            # monotonic: the direction never flips

    def test_a_clip_ending_mid_ascent_drops_the_incomplete_rep(self):
        # The count is registered at the CLOSING reversal, so a rep whose descent never arrives is
        # not counted — the clip ending on the way up adds nothing, it does not round up.
        complete = [0.0, 0.5, 1.0, 0.5] * 3 + [0.0]
        self.assertEqual(len(_fold(_at(complete))[1]), 3)
        self.assertEqual(len(_fold(_at(complete + [0.3, 0.6, 0.9]))[1]), 3)

    # --- the three prominence gates, each ON its bar ------------------------------------------------
    def test_prominence_exactly_at_the_cold_start_floor_is_admitted(self):
        # Cold start: amp is 0, so the relative bar collapses to prom_frac x floor and `floor` itself
        # binds. It is a `>=`, so a first rep that rises EXACTLY the floor counts.
        self.assertEqual(len(_fold(_at([0.0, 0.02, 0.04, 0.02, 0.0]))[1]), 1)
        self.assertEqual(len(_fold(_at([0.0, 0.02, 0.04 - 1e-9, 0.02, 0.0]))[1]), 0)

    def test_prominence_exactly_at_the_relative_amplitude_bar_is_admitted(self):
        # The window must be FULL for the relative gate to be armed at all (an unarmed gate is the
        # warm-up state, asserted separately below), so it is primed here to isolate the `>=`.
        knobs = dict(floor=0.0, prom_frac=0.5, amp=1.0,      # bar = 0.5 x max(amp, floor) = 0.5
                     _recent_amps=deque([1.0, 1.0, 1.0]))
        self.assertEqual(len(_fold(_at([0.0, 0.25, 0.5, 0.25, 0.0]), **knobs)[1]), 1)
        knobs["_recent_amps"] = deque([1.0, 1.0, 1.0])       # a fresh window for the second fold
        self.assertEqual(len(_fold(_at([0.0, 0.25, 0.5 - 1e-9, 0.25, 0.0]), **knobs)[1]), 0)

    def test_the_relative_bar_is_not_applied_until_the_amplitude_window_is_full(self):
        # The warm-up rule that makes the count order-independent: with fewer than `amp_window`
        # accepted reps there is no trustworthy amplitude to be a fraction OF, so only the fixed
        # `floor` binds. The same sub-bar excursion is admitted cold and rejected once the window fills.
        tiny = _at([0.0, 0.2, 0.4, 0.2, 0.0])   # prominence 0.4, under the 0.5 x 1.0 relative bar
        self.assertEqual(len(_fold(tiny, floor=0.0, prom_frac=0.5, amp=1.0)[1]), 1)   # unarmed: admitted
        self.assertEqual(
            len(_fold(tiny, floor=0.0, prom_frac=0.5, amp=1.0,
                      _recent_amps=deque([1.0, 1.0, 1.0]))[1]), 0)                    # armed: rejected
        self.assertEqual(_ChannelState(name="c").amp_window, 3)

    def test_prominence_exactly_at_the_absolute_range_of_motion_bar_is_admitted(self):
        # min_amp is the authored expected-range gate in the channel's native units — the one guard
        # a fidget cannot satisfy by being merely proportionate to itself.
        knobs = dict(floor=0.0, prom_frac=0.0, min_amp=0.25)
        self.assertEqual(len(_fold(_at([0.0, 0.125, 0.25, 0.125, 0.0]), **knobs)[1]), 1)
        self.assertEqual(len(_fold(_at([0.0, 0.125, 0.25 - 1e-9, 0.125, 0.0]), **knobs)[1]), 0)

    def test_a_reversal_confirms_at_exactly_the_direction_persistence_window(self):
        # The turn confirms when the opposite direction has persisted `>=` min_run_s SECONDS. A step
        # landing exactly on the window confirms; the same stream one nanosecond short confirms nothing.
        pairs = [(i * 0.25, v) for i, v in enumerate([0.0, 0.5, 1.0, 0.5, 0.0])]
        _state, reps = _fold(pairs, min_run_s=0.25, floor=0.0)
        self.assertEqual([r.stop_t for r in reps], [1.0])
        self.assertEqual(_fold(pairs, min_run_s=0.25 + 1e-9, floor=0.0)[1], [])

    # --- the six tunables with no other coverage ----------------------------------------------------
    def test_the_debounce_windows_fall_back_to_their_cold_start_values_with_no_interval_yet(self):
        cold = _ChannelState(name="c", smooth_s=0.2)
        self.assertIsNone(cold._median_interval())      # empty intervals is the cold-start state
        self.assertEqual(cold._min_run_s(), cold.min_run_s)
        self.assertEqual(cold._smooth_window(), 0.2)
        self.assertEqual(_ChannelState(name="c")._smooth_window(), 0.0)  # smoothing is opt-in

    def test_a_learned_period_only_ever_shortens_the_persistence_window(self):
        fast = _ChannelState(name="c", intervals=[1.0], period_cap_frac=0.05)
        self.assertAlmostEqual(fast._min_run_s(), 0.05)   # 0.05 x 1.0 s beats the 0.15 s cold start
        slow = _ChannelState(name="c", intervals=[1.0], period_cap_frac=1.0)
        self.assertAlmostEqual(slow._min_run_s(), 0.15)   # a big cap never GROWS the window

    def test_the_smoothing_window_tracks_the_learned_period_and_is_capped(self):
        tracked = _ChannelState(name="c", smooth_s=0.2, smooth_period_frac=0.18, intervals=[1.0])
        self.assertAlmostEqual(tracked._smooth_window(), 0.18)   # the cold-start 0.2 is superseded
        capped = _ChannelState(name="c", smooth_s=0.2, smooth_period_frac=0.5, smooth_cap_s=0.3,
                               intervals=[1.0])
        self.assertAlmostEqual(capped._smooth_window(), 0.3)

    def test_a_negative_measured_interval_inverts_the_persistence_guard(self):
        # The window is min(fixed, period_cap_frac x median interval) with NO sign check, so a single
        # negative inter-rep gap turns the jitter guard into a negative number — after which every
        # tentative direction flip confirms on its very first sample.
        self.assertAlmostEqual(_ChannelState(name="c", intervals=[-1.0])._min_run_s(), -0.4)

    def test_the_causal_smoother_keeps_a_sample_sitting_exactly_on_the_window_edge(self):
        single = _ChannelState(name="c", smooth_s=1.0)
        self.assertIsNone(single.observe(0.0, 0.7))
        self.assertEqual(single._prev, 0.7)      # a window holding one sample is that sample, not 0
        on_edge = _ChannelState(name="c", smooth_s=1.0)
        on_edge.observe(0.0, 0.0)
        on_edge.observe(1.0, 1.0)                # exactly smooth_s old: retained (the cutoff is `<`)
        self.assertEqual(on_edge._prev, 0.5)
        past_edge = _ChannelState(name="c", smooth_s=1.0)
        past_edge.observe(0.0, 0.0)
        past_edge.observe(1.0 + 1e-9, 1.0)       # a nanosecond older: evicted, window of one again
        self.assertEqual(past_edge._prev, 1.0)

    def test_the_period_debounce_rejects_a_rep_inside_its_fraction_of_the_learned_period(self):
        pairs = _at([0.0, 0.5, 1.0, 0.5] * 6)
        self.assertEqual(len(_fold(pairs, period_debounce=0.0)[1]), 5)   # off: every reversal counts
        self.assertEqual(len(_fold(pairs, period_debounce=2.0)[1]), 3)   # 2x the median gap swallows

    def test_the_amplitude_estimate_is_the_median_of_the_recent_accepted_prominences(self):
        # Rewritten from `..._moves_at_exactly_the_configured_ema_rate`, which pinned the arithmetic of
        # a single-sample-seeded EMA. That EMA is gone (BUG-A-05-02): the estimate is now the MEDIAN of
        # the last `amp_window` ACCEPTED prominences, which no single outlier can drag.
        pairs = _at([0.0, 0.5, 1.0, 0.5, 0.0, 0.1, 0.2, 0.1, 0.0])
        knobs = dict(floor=0.0, prom_frac=0.0)   # both reps admitted, so only the estimate is under test
        state, reps = _fold(pairs, **knobs)
        self.assertEqual(len(reps), 2)
        self.assertAlmostEqual(reps[0].amplitude, 1.0)
        self.assertAlmostEqual(reps[1].amplitude, 0.2)
        self.assertAlmostEqual(state.amp, 0.5 * (1.0 + 0.2))      # two samples -> the mean of the pair
        self.assertEqual(list(state._recent_amps), [1.0, 0.2])

        # A third rep makes the median a real median: the outlier stops being half the answer.
        three, reps3 = _fold(_at([0.0, 0.5, 1.0, 0.5, 0.0, 0.1, 0.2, 0.1,
                                  0.0, 0.1, 0.2, 0.1, 0.0]), **knobs)
        self.assertEqual([round(r.amplitude, 6) for r in reps3], [1.0, 0.2, 0.2])
        self.assertAlmostEqual(three.amp, 0.2)

    def test_the_amplitude_window_forgets_beyond_its_configured_depth(self):
        # It is a RECENT-amplitude estimate, so it stays fatigue-adaptive: the opening rep drops out of
        # the window once `amp_window` newer ones have been accepted, instead of living forever in an EMA.
        pairs = _at([0.0, 0.5, 1.0, 0.5] + [0.0, 0.1, 0.2, 0.1] * 3 + [0.0])
        state, reps = _fold(pairs, floor=0.0, prom_frac=0.0, amp_window=3)
        self.assertEqual([round(r.amplitude, 6) for r in reps], [1.0, 0.2, 0.2, 0.2])
        self.assertEqual(list(state._recent_amps), [0.2, 0.2, 0.2])   # the 1.0 has aged out
        self.assertAlmostEqual(state.amp, 0.2)

    # --- the two inputs that silently cost reps -----------------------------------------------------
    def test_a_plateau_at_the_peak_reports_only_the_instant_the_top_was_first_reached(self):
        # A flat step returns BEFORE the running-extreme update, so the peak instant is pinned to the
        # first sample at the top and the rest of the plateau is discarded. Holding the top for
        # another second and a half does not move `peak_t` by one frame; it only stretches the
        # reported peak->stop span, which is what a caller reads as "how long the turn took".
        held_for = {}
        for held in (1, 3, 8):
            _state, reps = _fold(_at([0.0, 0.5] + [1.0] * held + [0.5, 0.0, 0.0]))
            self.assertEqual(len(reps), 1, f"a plateau of {held} should still be one rep")
            held_for[held] = reps[0]
        self.assertEqual({round(r.peak_t, 9) for r in held_for.values()}, {round(2 / 6.0, 9)})
        self.assertAlmostEqual(held_for[1].stop_t - held_for[1].peak_t, 2 / 6.0)
        self.assertAlmostEqual(held_for[8].stop_t - held_for[8].peak_t, 9 / 6.0)

    def test_an_outsized_first_rep_no_longer_starves_the_smaller_reps_that_follow(self):
        # Rewritten (BUG-A-05-02). This asserted the defect: `amp` was an EMA SEEDED from the first
        # confirmed rep and updated only by an ACCEPTED one, so one anomalously large opening movement
        # raised the relative bar above every honest rep that followed and nothing could lower it again
        # — the SAME seven movements counted 1 in this order and 6 in the other.
        big = [0.0, 5.0, 10.0, 5.0, 0.0]
        small = [0.0] + [0.5, 1.0, 0.5, 0.0] * 6   # a leading trough, so neither order loses an edge rep
        opened_big, big_first = _fold(_at(big + small))
        opened_small, small_first = _fold(_at(small + big))
        self.assertEqual(len(big_first), 7)
        self.assertEqual(len(small_first), 7)      # order alone no longer moves the count
        self.assertEqual(sorted(round(r.amplitude, 6) for r in big_first),
                         sorted(round(r.amplitude, 6) for r in small_first))
        # ...because the estimate is a median over a short window, so the 10.0 is outvoted either way.
        self.assertAlmostEqual(opened_big.amp, 1.0)
        self.assertAlmostEqual(opened_small.amp, 1.0)

    # --- unvalidated inputs -------------------------------------------------------------------------
    def test_a_non_finite_channel_is_dropped_at_the_seam_and_never_reaches_the_detector(self):
        # Rewritten (BUG-A-05-03). This asserted the defect at the DETECTOR: `nan == prev` and
        # `nan > prev` are both False, so the sign expression classified a NaN as a downward step,
        # minted a phantom reversal and carried the NaN into `_prev`. The guard belongs at the seam the
        # caller's vector crosses (RhythmTracker.observe), not inside the sign expression — so
        # `_ChannelState` stays a pure sign-and-run detector and simply never sees the bad reading.
        tracker = RhythmTracker()
        tracker.observe(0.0, {"c": 0.0})
        tracker.observe(1 / 6.0, {"c": 0.5})
        st = tracker._channels["c"]
        self.assertEqual(st._tent, 1)
        tracker.observe(2 / 6.0, {"c": float("nan")})
        self.assertEqual(st._tent, 1)              # no phantom reversal ...
        self.assertEqual(st._prev, 0.5)            # ... and the state is not poisoned
        self.assertEqual(tracker.measurement.nonfinite, 1)   # the drop is RECORDED, not silent
        self.assertEqual(tracker.measurement.measured, 2)    # 3 frames observed, 2 of them measurable
        self.assertEqual(tracker.measurement.observed, 3)

    def test_a_frame_whose_only_channel_is_non_finite_counts_as_observed_but_not_measured(self):
        tracker = RhythmTracker()
        for i in range(4):
            tracker.observe(i / 6.0, {"c": float("inf")})
        self.assertEqual(tracker._channels, {})              # no detector was ever built for it
        self.assertEqual(tracker.measurement.observed, 4)
        self.assertEqual(tracker.measurement.measured, 0)
        self.assertEqual(tracker.measurement_coverage, 0.0)  # ... and 0.0 is not the same as `None`
        self.assertIsNone(RhythmTracker().measurement_coverage)

    def test_a_single_non_finite_sample_no_longer_costs_a_real_rep(self):
        # Rewritten (BUG-A-05-03): one bad sample out of 120 used to remove a whole repetition from the
        # tally with the caller told nothing. Dropping the reading instead costs the frame, not the rep.
        clean = [(i / 6.0, {"c": math.sin(2 * math.pi * i / 12)}) for i in range(120)]
        poisoned = [(t, dict(ch)) for t, ch in clean]
        poisoned[29][1]["c"] = float("nan")
        self.assertEqual(_run(clean)[0].count, 10)
        tracker, _ = _run(poisoned)
        self.assertEqual(tracker.count, 10)
        self.assertEqual(tracker.measurement.nonfinite, 1)

    def test_a_rewound_timestamp_re_seeds_instead_of_muting_the_counter(self):
        # Rewritten (BUG-A-05-04). This asserted the defect: `t - run_start_t` goes negative, so no
        # direction run could reach its persistence window until wall time climbed back past where it
        # was, and every rep in the rewound span was dropped in silence (9 forward, 4 rewound).
        # A rewinding media timebase is a recoverable source property, so the policy is RESET (the same
        # one MotionGate chose) — the phase state measured against the abandoned timeline is forgotten.
        forward = _at([0.0, 0.5, 1.0, 0.5] * 10)
        self.assertEqual(len(_fold(forward)[1]), 9)
        rewound = [(t - 5.0 if i >= 20 else t, v) for i, (t, v) in enumerate(forward)]
        state, reps = _fold(rewound)
        self.assertEqual(len(reps), 8)               # one rep lost AT the seam, not the whole span
        self.assertEqual(state._time.rewinds, 1)     # and the discontinuity is counted, not swallowed

    def test_the_rewind_keeps_the_reps_already_counted_and_the_amplitude_learned(self):
        forward = _at([0.0, 0.5, 1.0, 0.5] * 10)
        rewound = [(t - 5.0 if i >= 20 else t, v) for i, (t, v) in enumerate(forward)]
        state, _reps = _fold(rewound)
        self.assertGreaterEqual(len(state.reps), 4)  # the reps before the seek still happened
        self.assertAlmostEqual(state.amp, 1.0)       # magnitudes and durations are timeline-independent
        self.assertTrue(state.intervals)

    def test_the_frame_level_record_counts_a_rewind_once_however_many_channels_there_are(self):
        tracker = RhythmTracker()
        for i in range(8):
            tracker.observe(i / 6.0, {"a": float(i), "b": float(-i), "c": 0.0})
        tracker.observe(0.5, {"a": 1.0, "b": -1.0, "c": 0.0})   # a seek backwards
        self.assertEqual(tracker.measurement.rewinds, 1)
        self.assertEqual(tracker.measurement.observed, 9)

    def test_a_non_finite_timestamp_is_refused_outright_rather_than_bypassing_every_window(self):
        # Not a policy choice: every comparison against a NaN is False, so a NaN `t` does not tighten
        # the persistence window, it BYPASSES it — and there is no timeline position it could mean.
        tracker = RhythmTracker()
        tracker.observe(0.0, {"c": 0.0})
        with self.assertRaises(ValueError):
            tracker.observe(float("nan"), {"c": 1.0})


class ChannelSelectionBoundaryTests(unittest.TestCase):
    """Which channel owns the count.

    The shipped default is the legacy medoid over RAW rep tallies; the rate-invariant canonical
    selector that exists to replace it is off by default. Both policies are pinned here, along with
    the hysteresis boundaries (margin, dwell, scoring gate) that decide when a switch is allowed.
    """

    @staticmethod
    def _score(channel, score):
        return ChannelScore(channel, score=score, support=1.0, amplitude=1.0, regularity=1.0,
                            stability=1.0, period_s=1.0)

    def test_the_shipped_selector_changes_its_answer_with_the_capture_rate_alone(self):
        # One 20 s physical signal, three body channels, two capture rates. Nothing in the world
        # changes between the two runs — yet the owning channel and the reported count both move.
        specs = [("a", 0.5), ("b", 1.0), ("c", 2.8)]
        answers = {}
        for fps in (6, 30):
            tracker = RhythmTracker()
            for t, channels in _tones(20.0, fps, specs):
                tracker.observe(t, channels)
            answers[fps] = (tracker.selected, tracker.count)
        self.assertEqual(answers[6], ("b", 20))
        self.assertEqual(answers[30], ("c", 15))
        self.assertFalse(RhythmTracker().canonical_select)  # the rate-invariant fix is OFF by default

    def test_exactly_two_live_channels_are_broken_by_the_weight_of_evidence(self):
        # Rewritten (BUG-A-05-06b). With two channels the median of the tallies IS their mean, so both
        # sit the same distance from it and the first key always ties — which used to leave
        # `regularity` deciding alone, and `regularity()` returns 1.0 below two reps. A 2-rep channel
        # therefore beat a 40-rep one and 2 was the reported count. The tally now breaks the tie first.
        tracker = RhythmTracker()
        tracker._channels = {
            "many": _fabricate_channel("many", 40, 1.0, intervals=[1.0, 3.0, 1.0, 3.0]),
            "few": _fabricate_channel("few", 2, 1.0, intervals=[1.0, 1.0, 1.0, 1.0]),
        }
        self.assertEqual(tracker.selected, "many")
        self.assertEqual(tracker.count, 40)

    def test_regularity_still_decides_when_the_tally_ties_as_well(self):
        # The third key is not dead: with equal tallies the SHAPE of the two rep streams decides.
        tracker = RhythmTracker()
        tracker._channels = {
            "ragged": _fabricate_channel("ragged", 8, 1.0, intervals=[1.0, 3.0, 1.0, 3.0]),
            "even": _fabricate_channel("even", 8, 1.0, intervals=[1.0, 1.0, 1.0, 1.0]),
        }
        self.assertEqual(tracker.selected, "even")

    def test_three_channels_take_the_medoid_of_the_raw_rep_tallies(self):
        def medoid(tallies):
            tracker = RhythmTracker()
            tracker._channels = {n: _fabricate_channel(n, k, 1.0) for n, k in tallies.items()}
            return tracker.selected, tracker.count

        self.assertEqual(medoid({"a": 3, "b": 9, "c": 10}), ("b", 9))
        # The same motion sampled twice as densely doubles two tallies while the third saturates:
        # the consensus channel — and so the reported count — moves, with no change in the world.
        self.assertEqual(medoid({"a": 6, "b": 18, "c": 11}), ("c", 11))

    def test_a_prior_channel_with_one_accidental_rep_does_not_own_the_count(self):
        # Rewritten (BUG-A-05-05). "Viable" used to mean ONE confirmed rep, so a prior naming the wrong
        # body part did not merely bias the answer, it replaced it: `selected == "wrist"`, `count == 1`,
        # against a 20-rep channel. The prior is still a short-circuit — that is deliberate architecture,
        # not a defect — but it now has to clear an evidence floor to fire.
        tracker = RhythmTracker(prior_channel="wrist")
        tracker._channels = {"wrist": _fabricate_channel("wrist", 1, 0.05),
                             "torso": _fabricate_channel("torso", 20, 1.0)}
        self.assertEqual(tracker.selected, "torso")
        self.assertEqual(tracker.count, 20)

    def test_a_prior_channel_with_real_evidence_still_short_circuits_the_selection(self):
        # The mechanism is unchanged where it was right: an EVIDENCED prior wins outright, even against
        # a channel that reps more often. Only the floor underneath it is new.
        tracker = RhythmTracker(prior_channel="wrist")
        tracker._channels = {"wrist": _fabricate_channel("wrist", 6, 1.0, intervals=[1.0] * 5),
                             "torso": _fabricate_channel("torso", 20, 1.0)}
        self.assertEqual(tracker.selected, "wrist")
        self.assertEqual(tracker.count, 6)

    def test_the_evidence_floor_is_two_reps_at_a_real_fraction_of_the_loudest_channel(self):
        self.assertFalse(has_evidence(1, 1.0, 1.0, 0.5))     # one rep has no interval, so no regularity
        self.assertTrue(has_evidence(2, 1.0, 1.0, 0.5))      # `>=`, so two is enough
        self.assertFalse(has_evidence(20, 0.4, 1.0, 0.5))    # ...and it must still be loud enough
        self.assertTrue(has_evidence(20, 0.5, 1.0, 0.5))

    def test_the_floor_never_leaves_a_cold_start_unselectable(self):
        # The cold-start contract is untouched: one departure-and-return is one rep, and the channel
        # carrying it is still selected even though it cannot clear a two-rep floor.
        tracker, _ = _run(_wave(1))
        self.assertEqual(tracker.selected, "c")
        self.assertEqual(tracker.count, 1)

    def test_a_tracker_with_no_channel_reports_a_consistent_zero(self):
        tracker = RhythmTracker()
        self.assertIsNone(tracker.selected)
        self.assertEqual(tracker.count, 0)
        self.assertEqual(tracker.reps, [])
        self.assertEqual(tracker.regularity, 0.0)
        self.assertEqual(tracker.windowed_count(5.0, 10.0), 0)
        self.assertIsNone(select_channel({}))

    def test_one_live_channel_is_selected_whatever_its_amplitude(self):
        tracker = RhythmTracker()
        tracker._channels = {"only": _fabricate_channel("only", 4, 0.0)}
        self.assertEqual(tracker.selected, "only")   # the single-candidate short-circuit
        self.assertEqual(tracker.count, 4)

    def test_an_amplitude_gate_above_one_filters_every_channel_and_falls_back_to_all(self):
        def pick(amp_frac):
            tracker = RhythmTracker(amp_frac=amp_frac)
            tracker._channels = {"a": _fabricate_channel("a", 3, 0.1),
                                 "b": _fabricate_channel("b", 9, 0.1),
                                 "c": _fabricate_channel("c", 10, 2.0)}
            return tracker.selected, tracker.count

        self.assertEqual(pick(0.5), ("c", 10))   # the gate bites: only the loud channel survives
        self.assertEqual(pick(2.0), ("b", 9))    # it filters EVERYTHING, and the `or live` undoes it

    def test_a_channel_exactly_at_the_amplitude_gate_is_kept(self):
        # The tallies are chosen so the gate's `>=` is OBSERVABLE through the answer: with the quiet
        # channel kept the two tie on distance-from-the-median and the larger tally wins; dropped, only
        # the loud one is left. (Before BUG-A-05-06b the tie was broken by regularity, so this pair read
        # ("quiet", 3) / ("loud", 9) instead — same boundary, different consequence.)
        def pick(amp_frac):
            tracker = RhythmTracker(amp_frac=amp_frac)
            tracker._channels = {
                "quiet": _fabricate_channel("quiet", 12, 1.0, intervals=[1.0, 1.0, 1.0]),
                "loud": _fabricate_channel("loud", 9, 2.0, intervals=[1.0, 5.0, 1.0, 5.0]),
            }
            return tracker.selected, tracker.count

        self.assertEqual(pick(0.5), ("quiet", 12))         # 1.0 >= 0.5 x 2.0 -> kept, and it wins
        self.assertEqual(pick(0.5 + 1e-9), ("loud", 9))    # a nanofraction over -> dropped

    def test_an_expected_range_of_motion_with_no_prior_channel_gates_nothing(self):
        # The absolute gate is in the PRIOR channel's native units, so it is bound to that channel
        # only. Authored without one it is accepted, stored, and silently never applied.
        self.assertEqual(RhythmTracker(expected_rom=10.0)._state("c").min_amp, 0.0)
        bound = RhythmTracker(expected_rom=10.0, prior_channel="c")
        self.assertEqual(bound._state("c").min_amp, 5.0)      # amp_gate_frac 0.5 x 10.0
        self.assertEqual(bound._state("other").min_amp, 0.0)

        big = [0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0, 0.5, 1.0, 0.5, 0]
        small = [0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0, 0.2, 0.4, 0.2, 0]
        samples = [(i / 6.0, {"c": v}) for i, v in enumerate(big + small)]
        self.assertEqual(_run(samples, expected_rom=1.0)[0].count, 6)                     # no gate
        self.assertEqual(_run(samples, expected_rom=1.0, prior_channel="c")[0].count, 3)  # gated

    def test_a_challenger_exactly_at_the_margin_holds_the_incumbent(self):
        at_the_bar = {"a": self._score("a", 1.25), "b": self._score("b", 1.0)}
        self.assertEqual(select_channel(at_the_bar, current="b", margin=0.25), "b")   # `<=` holds
        over_the_bar = {"a": self._score("a", 1.25 + 1e-9), "b": self._score("b", 1.0)}
        self.assertEqual(select_channel(over_the_bar, current="b", margin=0.25), "a")
        self.assertEqual(select_channel({"a": self._score("a", 1.0)}), "a")  # one candidate, no hold

    # --- the canonical selector's hysteresis --------------------------------------------------------
    @staticmethod
    def _canonical(dwell):
        """A tracker with two scoreable channels: a loud one and a tenth-amplitude one."""
        tracker = RhythmTracker(canonical_select=True, select_dwell=dwell, select_band_s=(0.5, 5.0))
        tracker._channels = {"quiet": _fabricate_channel("quiet", 5, 0.1),
                             "loud": _fabricate_channel("loud", 12, 1.0)}
        tracker._series = {
            name: deque((i / 30.0, amp * math.sin(2 * math.pi * 0.5 * i / 30.0)) for i in range(360))
            for name, amp in (("quiet", 0.1), ("loud", 1.0))
        }
        return tracker

    def test_the_canonical_selection_commits_immediately_on_a_cold_start(self):
        tracker = self._canonical(dwell=5)
        tracker._reselect(1.0)
        self.assertEqual(tracker._sel_cache, "loud")   # no incumbent short-circuits the whole dwell
        self.assertEqual(tracker._pending_count, 0)

    def test_a_canonical_switch_waits_out_exactly_the_configured_dwell(self):
        tracker = self._canonical(dwell=3)
        tracker._sel_cache = "quiet"                   # an incumbent, so the dwell is live
        for beat in (1.0, 2.0):
            tracker._reselect(beat)
            self.assertEqual(tracker._sel_cache, "quiet")
        self.assertEqual(tracker._pending_count, 2)
        tracker._reselect(3.0)                         # the dwell is a `>=`, so the third one lands
        self.assertEqual(tracker._sel_cache, "loud")
        self.assertEqual(tracker._pending_count, 0)

    def test_a_channel_with_fewer_than_six_raw_samples_is_never_canonically_scored(self):
        def reselect(n_samples):
            tracker = RhythmTracker(canonical_select=True, select_band_s=(1.0, 5.0))
            tracker._channels = {"loud": _fabricate_channel("loud", 12, 1.0)}
            tracker._series = {"loud": deque((i / 2.0, math.sin(2 * math.pi * 0.5 * i / 2.0))
                                             for i in range(n_samples))}
            tracker._reselect(1.0)
            return tracker._sel_cache

        self.assertIsNone(reselect(5))            # below the raw-sample gate: not scored at all
        self.assertEqual(reselect(6), "loud")     # the gate is a `>=`, so six is enough

    def test_selection_falls_back_to_the_legacy_medoid_until_the_first_canonical_answer(self):
        tracker = RhythmTracker(canonical_select=True)
        tracker._channels = {n: _fabricate_channel(n, k, 1.0)
                             for n, k in (("a", 3), ("b", 9), ("c", 10))}
        self.assertIsNone(tracker._sel_cache)
        self.assertEqual(tracker.selected, "b")   # the rate-dependent medoid still owns the cold start


class _SilentEye:
    """A measuring eye that decodes every frame and simply sees nobody — an empty channel vector.

    Deliberately not pose-backed: the drive's arm-time pre-flight only probes detectors carrying a
    provider, so a plain `measure`-capable double reaches the frame loop's own containment path
    without dragging in pixels, numpy, or a pose model."""

    def __init__(self):
        self.frames = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def measure(self, frame):
        self.frames += 1
        return {}


class _OscillatingEye:
    """A working eye: one channel tracing a clean cycle, optionally crashing from a given frame on."""

    def __init__(self, fail_at=None):
        self.frames = 0
        self.resets = 0
        self._fail_at = fail_at

    def reset(self):
        self.resets += 1

    def measure(self, frame):
        if self._fail_at is not None and self.frames >= self._fail_at:
            raise RuntimeError("simulated inference backend crash: CUDA context lost")
        value = math.sin(2 * math.pi * self.frames / 12)
        self.frames += 1
        return {"c": value}


class _FlatEye:
    """An eye that measures a real person every frame, who does not move — a genuine zero count."""

    def reset(self):
        pass

    def measure(self, frame):
        return {"c": 0.5}


class _UndecodableEye:
    """An eye whose transport-to-pixels seam refuses the frame loop's payload outright."""

    def __init__(self):
        self.frames = 0

    def reset(self):
        pass

    def measure(self, frame):
        self.frames += 1
        raise FrameNotDecodable("not an image payload", frame_type=type(frame).__name__)


class _BoundaryOnlyEye:
    """A detector that proposes occurrences but cannot MEASURE — the authoring shape error."""

    def reset(self):
        pass

    def push(self, t, frame):
        return []


class RhythmDriveContainmentBoundaryTests(unittest.TestCase):
    """What the drive does with a frame, at the edges of its two containment layers.

    The layers themselves (arm-time refusal, mid-run drop) are proven against a real pose provider
    elsewhere; what is pinned here is the *shape* of every other exit from `observe` — an eye that
    measured nothing, a payload the eye cannot decode, a missing core seam, a frame of None, an
    empty goal set — and which of those leave a caller able to tell what happened.
    """

    def setUp(self):
        self._armed = 0

    @staticmethod
    def _core():
        return Percept.create(mode="offline", discover_plugins=False)

    def _arm(self, factory, goal_id, core=None, **state):
        """Register a fresh detector name backed by ``factory`` and author one rhythm goal on it.

        Returns ``(goal, made)`` where ``made`` collects every eye the registry constructed — so a
        test can tell a re-used eye from a cold one, which is the whole point of `adopt`."""
        self._armed += 1
        name = f"test-rhythm-eye-{self._armed}"
        made = []

        def _build():
            eye = factory()
            made.append(eye)
            return eye

        register(name, _build, DetectorManifest(
            name, "a synthetic measuring eye", "rhythm drive containment boundaries",
            "per-frame channel measurements",
        ))
        self.addCleanup(unregister, name)
        goal = Goal(id=goal_id, condition="a completed repetition", say="rep",
                    state={"type": "rhythm", "count": 0, "detectors": [name], **state})
        if core is not None:
            core.add_goal(goal)
        return goal, made

    # --- the silent zero ----------------------------------------------------------------------------
    def test_an_eye_that_measures_nothing_leaves_a_measurement_record_and_no_diagnostic(self):
        core = self._core()
        goal, made = self._arm(_SilentEye, "blind", core)
        drive = RhythmDrive(core, [goal])
        self.assertEqual(drive.goal_ids, ("blind",))
        for i in range(30):
            self.assertEqual(drive.observe(i / 6.0, "FRAME"), [])
        self.assertEqual(made[0].frames, 30)                 # the eye really did run, thirty times
        self.assertEqual(drive.count("blind"), 0)
        # Still no tracker and still no DIAGNOSTIC — nothing failed, and a goal that measured nothing
        # was never dropped. What is new is the measurement record beside the zero.
        self.assertNotIn("blind", getattr(core, "rhythm", {}))
        self.assertEqual(drive.arm_failures, {})
        self.assertEqual(drive.pop_mid_run_failures(), {})
        self.assertEqual(drive.measurement("blind"), MeasurementRecord(observed=30, measured=0))

    def test_an_eye_that_measured_nothing_for_a_whole_run_is_reported(self):
        # Rewritten (BUG-A-05-01). This was an xfail demanding the record appear in `arm_failures` /
        # `pop_mid_run_failures`, which is the WRONG channel: both of those mean "this goal was DROPPED
        # and will never be measured again", and a subject who walks out of frame for 300 frames is not
        # that. The bug's own "or an equivalent record" hedge is what landed: a MEASUREMENT coverage.
        core = self._core()
        goal, _made = self._arm(_SilentEye, "blind_run", core)
        drive = RhythmDrive(core, [goal])
        for i in range(300):
            drive.observe(i / 6.0, "FRAME")
        self.assertEqual(drive.count("blind_run"), 0)
        self.assertEqual(drive.arm_failures, {})              # NOT a failure ...
        self.assertEqual(drive.pop_mid_run_failures(), {})    # ... and NOT a drop
        self.assertEqual(drive.measurement_coverage("blind_run"), 0.0)
        self.assertEqual(drive.measurement("blind_run").observed, 300)

    def test_a_measured_run_and_an_unmeasured_one_report_the_same_zero_but_different_coverage(self):
        # The whole point of the record: the third state between "the goal failed" and "it counted 0".
        blind_core = self._core()
        blind_goal, _ = self._arm(_SilentEye, "blind", blind_core)
        blind = RhythmDrive(blind_core, [blind_goal])
        still_core = self._core()
        still_goal, _ = self._arm(lambda: _FlatEye(), "still", still_core)
        still = RhythmDrive(still_core, [still_goal])
        for i in range(60):
            blind.observe(i / 6.0, "FRAME")
            still.observe(i / 6.0, "FRAME")
        self.assertEqual((blind.count("blind"), still.count("still")), (0, 0))   # the same zero ...
        self.assertEqual(blind.measurement_coverage("blind"), 0.0)               # ... told apart
        self.assertEqual(still.measurement_coverage("still"), 1.0)

    def test_coverage_is_none_before_the_first_frame_which_is_not_zero(self):
        core = self._core()
        goal, _ = self._arm(_OscillatingEye, "unstarted", core)
        drive = RhythmDrive(core, [goal])
        self.assertIsNone(drive.measurement_coverage("unstarted"))
        self.assertIsNone(drive.measurement("unstarted"))
        self.assertIsNone(drive.measurement_coverage("no-such-goal"))
        drive.observe(0.0, "FRAME")
        self.assertEqual(drive.measurement_coverage("unstarted"), 1.0)

    def test_a_goal_whose_eye_crashed_keeps_the_frames_it_was_offered_in_its_record(self):
        # The record is a measurement history, so a dropped goal's evidence base survives the drop —
        # the failure channel says WHY it stopped, the record says how much it had seen by then.
        core = self._core()
        goal, _made = self._arm(lambda: _OscillatingEye(fail_at=10), "crashes", core)
        drive = RhythmDrive(core, [goal])
        for i in range(40):
            drive.observe(i / 6.0, "FRAME")
        self.assertIn("crashes", drive.pop_mid_run_failures())
        record = drive.measurement("crashes")
        self.assertEqual((record.observed, record.measured), (11, 10))   # the crashing frame counts too

    # --- the two exception paths --------------------------------------------------------------------
    def test_an_undecodable_payload_ends_the_call_for_every_goal_behind_it(self):
        core = self._core()
        bad, _ = self._arm(_UndecodableEye, "bad", core)
        good, good_made = self._arm(_OscillatingEye, "good", core)
        drive = RhythmDrive(core, [bad, good])
        self.assertEqual(drive.goal_ids, ("bad", "good"))
        with self.assertRaises(FrameNotDecodable):
            drive.observe(0.0, "FRAME")
        # The re-raise is deliberate (the loop's payload is bad, not this goal's provider) — but the
        # blast radius is every goal iterated after it, and nothing is recorded about the skip.
        self.assertEqual(good_made[0].frames, 0)
        self.assertEqual(drive.pop_mid_run_failures(), {})
        self.assertEqual(drive.goal_ids, ("bad", "good"))   # nor is the offending goal dropped

    def test_any_other_eye_failure_drops_only_that_goal(self):
        core = self._core()
        broken, _ = self._arm(lambda: _OscillatingEye(fail_at=0), "broken", core)
        working, working_made = self._arm(_OscillatingEye, "working", core)
        drive = RhythmDrive(core, [broken, working])
        for i in range(120):
            drive.observe(i / 6.0, "FRAME")
        self.assertEqual(drive.goal_ids, ("working",))
        self.assertEqual(working_made[0].frames, 120)
        self.assertEqual(drive.count("working"), 10)
        self.assertIn("broken", drive.pop_mid_run_failures())

    def test_an_eye_failing_late_keeps_the_reps_it_already_counted_and_reports_once(self):
        core = self._core()
        goal, made = self._arm(lambda: _OscillatingEye(fail_at=500), "late", core)
        drive = RhythmDrive(core, [goal])
        for i in range(600):
            drive.observe(i / 6.0, "FRAME")
        self.assertEqual(drive.goal_ids, ())
        self.assertEqual(drive.count("late"), 42)     # the reps before the crash still happened
        failures = drive.pop_mid_run_failures()
        self.assertIn("late", failures)
        self.assertIn("CUDA context lost", failures["late"])
        self.assertEqual(drive.pop_mid_run_failures(), {})   # drained exactly once, not per frame
        self.assertEqual(made[0].frames, 500)                # and never measured again

    def test_one_goal_with_no_measuring_detector_aborts_construction_for_every_goal(self):
        # A boundary-only detector on a rhythm goal is a shape error and is refused loudly — but it
        # is refused from the constructor, so the working goal beside it is never armed either.
        core = self._core()
        working, _ = self._arm(_OscillatingEye, "works", core)
        broken, _ = self._arm(_BoundaryOnlyEye, "shapeless", core)
        with self.assertRaises(ValueError) as caught:
            RhythmDrive(core, [working, broken])
        self.assertIn("shapeless", str(caught.exception))
        self.assertIn("measure(frame)", str(caught.exception))

    # --- the degenerate drives ----------------------------------------------------------------------
    def test_an_empty_goal_list_is_falsy_and_observes_nothing(self):
        drive = RhythmDrive(self._core(), [])
        self.assertEqual(drive.goal_ids, ())
        self.assertFalse(drive)
        self.assertEqual(drive.observe(0.0, "FRAME"), [])
        self.assertEqual(drive.count("anything"), 0)
        self.assertEqual(drive.arm_failures, {})

    def test_a_core_without_the_rhythm_seam_never_reaches_the_eye(self):
        class _CoreWithoutRhythm:
            pass

        goal, made = self._arm(_OscillatingEye, "seamless")
        drive = RhythmDrive(_CoreWithoutRhythm(), [goal])
        self.assertEqual(drive.goal_ids, ("seamless",))   # armed...
        self.assertFalse(drive)                            # ...and not drivable, which is testable
        self.assertEqual(drive.observe(0.0, "FRAME"), [])
        self.assertEqual(made[0].frames, 0)
        self.assertEqual(drive.count("seamless"), 0)

    def test_a_frame_of_none_is_skipped_before_the_eye(self):
        core = self._core()
        goal, made = self._arm(_OscillatingEye, "gap", core)
        drive = RhythmDrive(core, [goal])
        self.assertEqual(drive.observe(0.0, None), [])
        self.assertEqual(made[0].frames, 0)

    # --- lifecycle ----------------------------------------------------------------------------------
    def test_reset_cold_starts_the_eye_but_deliberately_not_the_tracker(self):
        core = self._core()
        goal, made = self._arm(_OscillatingEye, "rst", core)
        drive = RhythmDrive(core, [goal])
        for i in range(120):
            drive.observe(i / 6.0, "FRAME")
        counted = drive.count("rst")
        self.assertEqual(counted, 10)
        drive.reset()
        self.assertEqual(made[0].resets, 1)
        self.assertEqual(drive.count("rst"), counted)   # a pause is not evidence the reps did not happen

    def test_adopting_no_previous_drive_is_a_no_op(self):
        core = self._core()
        goal, made = self._arm(_OscillatingEye, "fresh", core)
        drive = RhythmDrive(core, [goal])
        self.assertIs(drive.adopt(None), drive)
        self.assertEqual(drive.goal_ids, ("fresh",))
        drive.observe(0.0, "FRAME")
        self.assertEqual(made[0].frames, 1)

    def test_adopting_a_previous_drive_carries_the_surviving_eye_across_a_regate(self):
        core = self._core()
        goal, made = self._arm(_OscillatingEye, "regate", core)
        first = RhythmDrive(core, [goal])
        for i in range(60):
            first.observe(i / 6.0, "FRAME")
        second = RhythmDrive(core, [goal])
        self.assertEqual(len(made), 2)          # construction built a fresh, cold eye...
        self.assertEqual(made[1].frames, 0)
        second.adopt(first)
        second.observe(10.0, "FRAME")
        self.assertEqual(made[1].frames, 0)     # ...which adopt discards, so no mid-rep cold start
        self.assertEqual(made[0].frames, 61)

    def test_a_bare_goal_stays_silent_while_a_milestone_goal_fires_once(self):
        core = self._core()
        bare, _ = self._arm(_OscillatingEye, "bare", core)
        bare_drive = RhythmDrive(core, [bare])
        bare_fires = []
        for i in range(120):
            bare_fires += bare_drive.observe(i / 6.0, "FRAME")
        self.assertEqual(bare_fires, [])
        self.assertEqual(bare_drive.count("bare"), 10)   # counted, and deliberately not announced

        milestone_core = self._core()
        milestone, _ = self._arm(_OscillatingEye, "milestone", milestone_core, fire_when="n == 3")
        milestone_drive = RhythmDrive(milestone_core, [milestone])
        fired = []
        for i in range(120):
            fired += milestone_drive.observe(i / 6.0, "FRAME")
        self.assertEqual(len(fired), 1)
        self.assertEqual(milestone_drive.count("milestone"), 10)


if __name__ == "__main__":
    unittest.main()
