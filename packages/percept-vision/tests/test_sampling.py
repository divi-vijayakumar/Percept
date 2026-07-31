"""The fovea — SamplingController (perceive much at capture rate, process per-concern density).

Deterministic: a synthetic ``(t, motion)`` capture stream in, processing decisions out. No model.
"""

from __future__ import annotations

import unittest
import warnings

from percept._streamtime import StreamTimeRewindWarning
from percept.behavior import coerce_authored_sense
from percept.sampling import SamplingController, SensePolicy


def _feed(ctrl, cid, fps, dur, motion_fn):
    """Drive a ``fps``-captured stream of length ``dur`` and return the timestamps it chose to PROCESS."""
    return [
        i / fps
        for i in range(int(dur * fps))
        if ctrl.should_process(cid, i / fps, motion_fn(i / fps))
    ]


def _min_gap(ts):
    return min((b - a for a, b in zip(ts, ts[1:])), default=float("inf"))


class SamplingControllerTests(unittest.TestCase):
    def test_coasts_at_base_fps_when_quiet(self):
        # Capture at 30 fps but, with no motion, process only ~base_fps (4/s) — most frames skipped (cheap).
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15, escalate_threshold=0.15))
        ts = _feed(c, "g", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        self.assertLessEqual(len(ts), 14)  # ≈ 4/s × 3s, never the full 90 captured frames
        self.assertGreaterEqual(len(ts), 9)

    def test_escalates_on_motion_burst_then_relaxes(self):
        # A motion burst in [1.0, 1.5] should densify processing during the burst + hold; quiet stays sparse.
        c = SamplingController(SensePolicy(base_fps=3, max_fps=15, escalate_threshold=0.15, hold_s=0.5))
        ts = _feed(c, "g", fps=30, dur=3.0, motion_fn=lambda t: 1.0 if 1.0 <= t < 1.5 else 0.0)
        quiet = [x for x in ts if x < 1.0]
        burst = [x for x in ts if 1.0 <= x <= 2.0]  # burst + the 0.5s saccade hold
        self.assertGreater(len(burst), 3 * len(quiet) / 3)  # markedly denser than the quiet ~3/s baseline
        self.assertGreaterEqual(len(burst), 8)  # approaching max_fps during the burst+hold

    def test_never_exceeds_max_fps(self):
        # Motion pinned high (always escalated) -> processing capped at max_fps (the cost ceiling).
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15, escalate_threshold=0.15))
        ts = _feed(c, "g", fps=60, dur=2.0, motion_fn=lambda t: 1.0)
        self.assertGreaterEqual(_min_gap(ts), 1.0 / 15 - 1e-6)
        self.assertLessEqual(len(ts), 15 * 2 + 2)

    def test_per_concern_isolation(self):
        c = SamplingController()
        c.configure("slow", SensePolicy(base_fps=2, max_fps=2))
        c.configure("fast", SensePolicy(base_fps=10, max_fps=10))
        slow = _feed(c, "slow", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        fast = _feed(c, "fast", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        self.assertGreater(len(fast), len(slow) * 2)  # independent density per concern

    def test_reset_clears_timing(self):
        c = SamplingController(SensePolicy(base_fps=1, max_fps=1))
        self.assertTrue(c.should_process("g", 0.0, 0.0))
        self.assertFalse(c.should_process("g", 0.1, 0.0))  # within the 1 fps interval
        c.reset("g")
        self.assertTrue(c.should_process("g", 0.2, 0.0))  # post-reset processes immediately

    def test_sense_firewall_keeps_valid_drops_garbage_and_clamps(self):
        ok = coerce_authored_sense(
            {
                "base_fps": 4,
                "max_fps": 15,
                "escalate_threshold": 0.2,
                "brief_window_s": 1.0,
            }
        )
        self.assertEqual(
            ok["sense"],
            {
                "base_fps": 4.0,
                "max_fps": 15.0,
                "escalate_threshold": 0.2,
                "brief_window_s": 1.0,
            },
        )
        # inverted envelope is repaired (max >= base), out-of-range threshold dropped, rates clamped to 60.
        fixed = coerce_authored_sense({"base_fps": 30, "max_fps": 2, "escalate_threshold": 5})
        self.assertGreaterEqual(fixed["sense"]["max_fps"], fixed["sense"]["base_fps"])
        self.assertNotIn("escalate_threshold", fixed["sense"])
        self.assertEqual(coerce_authored_sense({"base_fps": 999})["sense"]["base_fps"], 60.0)
        self.assertEqual(
            coerce_authored_sense({"brief_window_s": 0.001})["sense"]["brief_window_s"],
            2.0 / 60.0,
        )
        self.assertEqual(
            coerce_authored_sense({"brief_window_s": 999})["sense"]["brief_window_s"],
            30.0,
        )
        self.assertEqual(coerce_authored_sense("nope"), {})
        self.assertEqual(coerce_authored_sense({"base_fps": -1}), {})


class SensePolicyBoundaryTests(unittest.TestCase):
    """The boundary table for ``should_process`` (spec row M-25).

    Three comparisons decide every frame, and each one is asymmetric at its edge:
    ``motion >= escalate_threshold`` (arms), ``t < _boost_until`` (half-open hold), and
    ``(t - last) >= interval - 1e-9`` (a one-nanosecond slack, so an exact-rate capture is admitted).
    Everything below pins one of those edges, or one of the envelopes the policy does not validate.
    """

    # -- the three comparisons, at their exact boundary values ------------------------------------

    def test_first_call_admits_because_there_is_no_history(self):
        c = SamplingController(SensePolicy(base_fps=0.5, max_fps=0.5))  # a 2 s interval
        self.assertTrue(c.should_process("g", 123.456, 0.0))  # arbitrary t: `last is None` short-circuits
        self.assertFalse(c.should_process("g", 123.5, 0.0))  # ... and from then on it is metered
        self.assertTrue(c.should_process("other", 123.5, 0.0))  # a fresh concern has its own history

    def test_motion_exactly_at_escalate_threshold_arms_the_saccade(self):
        # `motion >= p.escalate_threshold` — the threshold value itself arms, one ulp below does not.
        armed = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        self.assertTrue(armed.should_process("g", 0.0, 0.15))
        # 0.1 s later the 1 fps base floor would refuse; the armed 10 fps hold admits.
        self.assertTrue(armed.should_process("g", 0.1, 0.0))

        below = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        self.assertTrue(below.should_process("g", 0.0, 0.1499))
        self.assertFalse(below.should_process("g", 0.1, 0.0))

    def test_t_exactly_at_boost_until_is_no_longer_boosted(self):
        # `t < self._boost_until[...]` — half-open, so the hold expires AT boost_until, not after it.
        inside = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        inside.should_process("g", 0.0, 0.2)  # arms the hold out to t == 0.5
        self.assertTrue(inside.should_process("g", 0.49, 0.0))

        edge = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        edge.should_process("g", 0.0, 0.2)
        self.assertFalse(edge.should_process("g", 0.5, 0.0))  # exactly at the edge: back to base_fps

    def test_gap_exactly_equal_to_the_interval_admits_and_the_slack_is_one_nanosecond(self):
        exact = SamplingController(SensePolicy(base_fps=4, max_fps=4))  # interval 0.25 s
        self.assertTrue(exact.should_process("g", 0.0, 0.0))
        self.assertTrue(exact.should_process("g", 0.25, 0.0))  # == interval admits, by design

        under = SamplingController(SensePolicy(base_fps=4, max_fps=4))
        under.should_process("g", 0.0, 0.0)
        self.assertFalse(under.should_process("g", 0.25 - 1e-6, 0.0))

        # The comparison is `>= interval - 1e-9`, not `>= interval`: a hair under the raw interval
        # STILL admits, so an exact-rate capture is never tripped by float error.
        slack = SamplingController(SensePolicy(base_fps=4, max_fps=4))
        slack.should_process("g", 0.0, 0.0)
        self.assertTrue(slack.should_process("g", 0.25 - 1e-10, 0.0))

    # -- envelopes ---------------------------------------------------------------------------------

    def test_base_equals_max_makes_motion_irrelevant(self):
        # No saccade is expressible: the escalated target equals the quiet one, so the two streams
        # must produce byte-identical decisions.
        quiet = SamplingController(
            SensePolicy(base_fps=5, max_fps=5, escalate_threshold=0.15, hold_s=0.5)
        )
        busy = SamplingController(
            SensePolicy(base_fps=5, max_fps=5, escalate_threshold=0.15, hold_s=0.5)
        )
        q = _feed(quiet, "g", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        b = _feed(busy, "g", fps=30, dur=3.0, motion_fn=lambda t: 1.0)
        self.assertEqual(q, b)
        self.assertGreater(len(q), 1)
        self.assertGreaterEqual(_min_gap(q), 1.0 / 5 - 1e-6)

    def test_hold_s_zero_makes_max_fps_unreachable(self):
        # `_boost_until = t + 0.0 == t`, and the test is `t < boost_until`, so not even the frame
        # that spiked is boosted. max_fps becomes dead weight and a burst is resolved at base_fps.
        dead = SamplingController(
            SensePolicy(base_fps=1, max_fps=30, escalate_threshold=0.15, hold_s=0.0)
        )
        held = SamplingController(
            SensePolicy(base_fps=1, max_fps=30, escalate_threshold=0.15, hold_s=0.5)
        )
        d = _feed(dead, "g", fps=30, dur=2.0, motion_fn=lambda t: 1.0)
        h = _feed(held, "g", fps=30, dur=2.0, motion_fn=lambda t: 1.0)
        self.assertEqual(len(d), 2)  # t=0.0 and t=1.0 — the 1 fps floor, straight through the burst
        self.assertEqual(len(h), 60)  # every captured frame — the 30 fps ceiling the knob promises

    def test_nan_motion_is_silently_read_as_quiet(self):
        # `float("nan") >= threshold` is False, so a broken cheap-motion statistic never arms the
        # saccade and never raises: the concern coasts at base_fps with no diagnostic at all.
        nan = SamplingController(
            SensePolicy(base_fps=1, max_fps=30, escalate_threshold=0.15, hold_s=0.5)
        )
        quiet = SamplingController(
            SensePolicy(base_fps=1, max_fps=30, escalate_threshold=0.15, hold_s=0.5)
        )
        n = _feed(nan, "g", fps=30, dur=2.0, motion_fn=lambda t: float("nan"))
        q = _feed(quiet, "g", fps=30, dur=2.0, motion_fn=lambda t: 0.0)
        self.assertEqual(n, q)
        self.assertEqual(len(n), 2)  # the base floor, not the ceiling a real burst would have reached

    def test_should_process_is_a_mutation_not_a_query(self):
        c = SamplingController(SensePolicy(base_fps=4, max_fps=4))
        self.assertTrue(c.should_process("g", 0.0, 0.0))
        self.assertFalse(c.should_process("g", 0.0, 0.0))  # identical arguments, opposite answer

        # And a call that ANSWERS FALSE still changes the future — it armed the saccade on its way
        # out, so a caller that "just checks" a frame silently rewrites the next decision.
        armed = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        self.assertTrue(armed.should_process("g", 0.0, 0.0))
        self.assertFalse(armed.should_process("g", 0.05, 1.0))  # too soon even for the boosted rate
        self.assertTrue(armed.should_process("g", 0.1, 0.0))  # ... but the burst it saw is still held

        never = SamplingController(
            SensePolicy(base_fps=1, max_fps=10, escalate_threshold=0.15, hold_s=0.5)
        )
        never.should_process("g", 0.0, 0.0)
        self.assertFalse(never.should_process("g", 0.1, 0.0))  # without that call: still the base rate

    # -- the findings ------------------------------------------------------------------------------

    def test_base_fps_zero_is_refused_at_construction(self):
        """BUG-A-03-01, fixed. A zero quiet floor is not a slow cadence — ``target <= 0`` used to
        return False ABOVE the ``last is None`` first-call branch, so the concern was never perceived
        on a quiet scene and never complained. It is now rejected where it is built, by the one
        authority, so the silent-blindness state is unreachable."""
        with self.assertRaises(ValueError):
            SensePolicy(base_fps=0)  # max_fps=15, threshold=0.15, hold_s=0.5
        with self.assertRaises(ValueError):
            SensePolicy(base_fps=0, hold_s=0.0)  # the doubly-dead variant: no coast AND no saccade
        with self.assertRaises(ValueError):
            SensePolicy(max_fps=0)
        with self.assertRaises(ValueError):
            SensePolicy(base_fps=-1)

        # The smallest positive floor is still accepted and still perceives — the fix refuses zero,
        # it does not impose a minimum rate.
        slow = SamplingController(SensePolicy(base_fps=1e-6, max_fps=15))
        self.assertTrue(slow.should_process("g", 0.0, 0.0))

    def test_a_policy_the_controller_accepts_must_perceive_a_quiet_scene(self):
        try:
            policy = SensePolicy(base_fps=0)
        except Exception:
            return  # rejected at construction, the way CadenceProfile rejects it — also acceptable
        c = SamplingController(policy)
        seen = sum(1 for i in range(30 * 60) if c.should_process("g", i / 30.0, 0.0))
        self.assertGreater(seen, 0, "60 s of quiet capture produced zero perception and zero error")

    def test_an_inverted_envelope_is_refused_at_construction(self):
        """BUG-A-03-02, fixed. An unvalidated ``base_fps > max_fps`` ran the saccade BACKWARDS —
        quiet frames at 20 fps (ten times the declared ceiling) and motion DE-escalating to 2 fps, the
        burst a saccade exists to resolve being the one it under-sampled. Refused at the source now,
        with the same rule and the same message CadenceProfile has always used."""
        with self.assertRaises(ValueError):
            SensePolicy(base_fps=20, max_fps=2, escalate_threshold=0.15, hold_s=0.5)
        # base == max is the boundary and stays legal: it expresses "no saccade", not an inversion.
        SensePolicy(base_fps=5, max_fps=5)

    def test_processing_is_never_denser_than_max_fps(self):
        # The docstring's promise ("never exceed max_fps") over every envelope the type accepts —
        # quiet or busy, and at the base == max boundary where the two rates coincide.
        for base, mx in ((20, 20), (2, 20), (5, 5), (1, 30)):
            for motion in (0.0, 1.0):
                with self.subTest(base_fps=base, max_fps=mx, motion=motion):
                    c = SamplingController(
                        SensePolicy(base_fps=base, max_fps=mx, escalate_threshold=0.15, hold_s=0.5)
                    )
                    ts = _feed(c, "g", fps=60, dur=1.0, motion_fn=lambda t, m=motion: m)
                    self.assertGreaterEqual(
                        _min_gap(ts), 1.0 / mx - 1e-6,
                        "processed denser than max_fps, the declared cost ceiling",
                    )

    def test_a_rewound_span_is_sampled_at_its_own_cadence_not_dropped(self):
        """BUG-A-03-03, fixed. Was a characterization test asserting the defect: it pinned
        ``rewound == []`` — 5 s of capture, 50 frames, none perceived, no error — because ``t - last``
        went negative and stayed negative for the whole rewound span.

        The guard now reads the rewind as a stream discontinuity and drops the concern's timing
        state, so the new timeline is sampled from its first frame at the ordinary cadence.
        """
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15))
        self.assertTrue(c.should_process("g", 10.0, 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StreamTimeRewindWarning)
            rewound = [t / 10.0 for t in range(50, 100) if c.should_process("g", t / 10.0, 0.0)]
        self.assertGreater(len(rewound), 0, "the rewound span must be perceived, not dropped")
        self.assertEqual(rewound[0], 5.0, "sampling resumes at the first frame of the new timeline")
        # base_fps=4 -> a 0.25 s floor. The span is re-sampled at the ordinary cadence, not re-run
        # at capture rate: 5 s of quiet at 4 fps is ~20 frames, not all 50.
        self.assertLess(len(rewound), 50)
        self.assertGreaterEqual(min(b - a for a, b in zip(rewound, rewound[1:])), 0.25 - 1e-6)

        # A motion burst inside the rewound span is no longer dead either: the saccade arms on the
        # new timeline instead of being outrun by a negative gap.
        c2 = SamplingController(SensePolicy(base_fps=4, max_fps=15))
        c2.should_process("g", 10.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StreamTimeRewindWarning)
            self.assertTrue(c2.should_process("g", 6.0, 1.0))

    def test_a_backwards_clock_does_not_silently_blind_the_concern(self):
        """BUG-A-03-03. The rewound span is perceived, and the discontinuity is reported.

        The fovea absorbs a rewind as a stream discontinuity rather than raising: a reconnect or a
        file seek is recoverable source noise, and raising would convert it into a dead watch. That
        is the same policy :meth:`MotionGate._gate_ts` chose on the same untrusted timeline, and
        deliberately not :mod:`percept.interest`'s raise-on-rewind.
        """
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15))
        c.should_process("g", 10.0, 0.0)
        seen = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(50):
                if c.should_process("g", 5.0 + i / 10.0, 0.0):
                    seen += 1
        self.assertGreater(seen, 0, "5 s of capture after a clock rewind was dropped entirely")
        # Loud, not silent: the brake really was dropped, so it has to be traceable to this moment.
        self.assertEqual([type(w.message) for w in caught], [StreamTimeRewindWarning])

    def test_only_the_rewinding_concern_loses_its_timing_state(self):
        """The guard is per concern because ``should_process`` runs once per concern per frame.

        A single controller-wide guard would report ``rewind`` to whichever concern was asked first
        and ``duplicate`` to every other one, so only that first concern would recover.
        """
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for concern in ("a", "b"):
                c.should_process(concern, 10.0, 0.0)
            rewound = [c.should_process(concern, 6.0, 0.0) for concern in ("a", "b")]
        self.assertEqual(rewound, [True, True], "both concerns must recover, not just the first")
        self.assertEqual([type(w.message) for w in caught], [StreamTimeRewindWarning] * 2)

    def test_a_non_finite_timestamp_is_refused_rather_than_poisoning_the_concern(self):
        """A NaN is not a timeline position. Left alone it is written into ``_last_processed`` and
        every later ``(t - last) >= interval`` is False, so the concern never perceives again."""
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15))
        c.should_process("g", 10.0, 0.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                c.should_process("g", bad, 0.0)
        self.assertTrue(c.should_process("g", 10.3, 0.0), "the concern still works afterwards")


class OneCadenceAuthorityTests(unittest.TestCase):
    """``SensePolicy`` is the type every consumer runs on, so it is where the envelope is decided.

    The envelope used to be checked in four places and skipped in this one, which is how a base_fps of
    0 (A-03-01) and an inverted pair (A-03-02) reached the fovea. These pin the delegation itself: the
    profile handle and the policy cannot answer differently, because there is only one implementation.
    """

    def test_the_profile_handle_and_the_policy_accept_exactly_the_same_envelopes(self):
        from percept.watchprofile import CadenceProfile, WatchProfileError

        envelopes = [
            (4.0, 15.0, 0.15, 0.5),      # the shipped default
            (5.0, 5.0, 0.0, 0.0),        # boundaries: base == max, threshold 0, no hold
            (1.0, 30.0, 1.0, 10.0),      # threshold at its upper bound
            (0.0, 15.0, 0.15, 0.5),      # A-03-01: a zero quiet floor
            (-1.0, 15.0, 0.15, 0.5),
            (20.0, 2.0, 0.15, 0.5),      # A-03-02: inverted
            (4.0, 0.0, 0.15, 0.5),
            (4.0, 15.0, 1.5, 0.5),       # threshold out of range
            (4.0, 15.0, -0.01, 0.5),
            (4.0, 15.0, 0.15, -0.1),     # negative hold
            (float("nan"), 15.0, 0.15, 0.5),   # non-finite: every `<= 0` guard misses these
            (float("inf"), float("inf"), 0.15, 0.5),
            (4.0, 15.0, float("nan"), 0.5),
            (4.0, 15.0, 0.15, float("nan")),
        ]
        for env in envelopes:
            with self.subTest(envelope=env):
                policy_err = profile_err = None
                try:
                    SensePolicy(*env)
                except ValueError as exc:
                    policy_err = exc
                try:
                    CadenceProfile(*env)
                except WatchProfileError as exc:
                    profile_err = exc
                self.assertEqual(
                    policy_err is None, profile_err is None,
                    f"the two cadence handles disagree about {env}: "
                    f"policy={policy_err!r} profile={profile_err!r}",
                )
                if policy_err is not None:  # same rule -> same message, only the type differs
                    self.assertEqual(str(policy_err), str(profile_err))

    def test_a_bool_is_not_a_rate(self):
        # `bool` subclasses `int`, so an untyped `True` would otherwise land as a 1 fps envelope —
        # a type confusion that reads as a plausible cadence and is never questioned again.
        for kwargs in ({"base_fps": True}, {"max_fps": True}, {"escalate_threshold": True},
                       {"hold_s": True}, {"base_fps": False}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SensePolicy(**kwargs)

    def test_a_non_numeric_rate_is_refused_rather_than_deferred_to_a_comparison(self):
        for kwargs in ({"base_fps": "4"}, {"max_fps": None}, {"escalate_threshold": []},
                       {"hold_s": object()}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SensePolicy(**kwargs)

    def test_a_nan_envelope_cannot_reach_the_controller(self):
        # NaN is the one that slipped through every `<= 0` / `>` guard in the package: `nan <= 0` is
        # False, and `1.0 / nan` is NaN, so `(t - last) >= nan` is False forever — the controller
        # answers True once (`last is None`) and False for the rest of the run, silently.
        from percept.sampling import is_usable_fps

        for kwargs in ({"base_fps": float("nan")}, {"max_fps": float("nan")},
                       {"base_fps": float("inf")}, {"max_fps": float("inf")}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SensePolicy(**kwargs)
        # ... and the predicate a caller must use INSTEAD of a truthiness test, because NaN is truthy.
        self.assertTrue(bool(float("nan")))
        self.assertFalse(is_usable_fps(float("nan")))
        self.assertFalse(is_usable_fps(0))
        self.assertFalse(is_usable_fps(None))
        self.assertFalse(is_usable_fps(True))
        self.assertFalse(is_usable_fps(float("inf")))
        self.assertTrue(is_usable_fps(4))
        self.assertTrue(is_usable_fps(1e-9))

    def test_the_sense_firewall_only_ever_emits_envelopes_the_policy_accepts(self):
        # coerce_authored_sense clamps; SensePolicy rejects. They must not disagree, or an LLM-authored
        # goal that passed the firewall would still crash (or blind) the fovea downstream.
        for raw in ({"base_fps": 30, "max_fps": 2}, {"base_fps": 999}, {"base_fps": 0},
                    {"escalate_threshold": 5}, {"base_fps": -1}, {"max_fps": 0},
                    {"base_fps": float("nan")}, {"base_fps": True}):
            with self.subTest(raw=raw):
                spec = coerce_authored_sense(raw).get("sense", {})
                d = SensePolicy()
                SensePolicy(  # must not raise for anything the firewall let through
                    base_fps=spec.get("base_fps", d.base_fps),
                    max_fps=max(spec.get("max_fps", d.max_fps), spec.get("base_fps", 0.0)),
                    escalate_threshold=spec.get("escalate_threshold", d.escalate_threshold),
                )


if __name__ == "__main__":
    unittest.main()
