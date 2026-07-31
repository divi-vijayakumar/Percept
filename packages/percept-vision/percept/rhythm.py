"""The rhythm sense — a core perceptual primitive: perceive a repeating cycle and count its reps.

A peer to the executor's other senses (``judge``/``transition``/``episode``/``absence`` in
:mod:`percept.behavior`). It answers one irreducible query — *"is there a repeating cycle, and where are
its turns?"* — over a measured scalar stream (a pose channel, an audio envelope, a step cadence). It is
domain-general: reps, strokes, steps, breaths, stirs all read the same way.

Design constraints that shaped this module (from the build brainstorm):

  * **Causal & differential.** A rep is a trajectory ``trough → peak → trough`` — an up-run then a down-run
    of the first difference. The count fires the instant a prominent peak's closing reversal is confirmed,
    using only PAST samples (no σ-normalization over the whole signal — that is non-causal, you could not
    count out loud). Turns are located by **velocity sign** (amplitude-free, fatigue-robust) and accepted by
    a run-length + prominence guard, not by distribution-relative bands.
  * **Counts a single rep.** Cold-start needs no warmup: one departure-and-return above a minimal-deviation
    floor is one rep (the "count to 1 jumping jack" test). Later reps refine the amplitude/period estimates.
  * **The eye measures, the brain decides.** The edge supplies decomposed channel *measurements*; ALL rep
    logic lives here. Detecting a local extremum is perception; deciding a rep is this module.
  * **Self-isolating state.** One :class:`RhythmTracker` per concern; the gate is never touched (the
    executor routes a confirmed rep through the unchanged gate, see :meth:`percept.behavior.fire_rhythm`).

The tracker runs one :class:`_ChannelState` per measured channel and **selects** the channel whose cycles
are largest and most regular as the one that owns the count. ``count`` is always the selected channel's
confirmed-rep tally.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ._numeric import finite
from ._streamtime import StreamTimeGuard
from .measurement import CANON_HZ, ChannelScore, MeasurementSeries, channel_score


def _median(xs) -> float:
    """The median of a non-empty sequence of floats (stdlib-free, so the low-level modules can use it)."""
    ordered = sorted(xs)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


@dataclass
class MeasurementRecord:
    """The evidence base under a count: how many frames were OFFERED, and how many were MEASURED.

    ONE shape, used at all three folds where a rep tally is published, because they all had the same
    hole — a count with no denominator:

      * :class:`RhythmTracker` — per goal-and-subject, the frames folded into that tracker.
      * :class:`RhythmDrive` — per goal, the frames the drive handed the eye (:meth:`RhythmDrive.observe`
        used to ``continue`` past a frame the eye measured nothing on, leaving no trace of any kind, so a
        run where the eye returned ``{}`` on every frame was byte-identical to a run of genuine zero reps).
      * :class:`percept.rhythm_hero.RhythmHero` — per clip, the frames handed to the hero, so
        ``hero_track_coverage`` divides by frames OBSERVED rather than by frames that happened to carry a
        detection (which made numerator and denominator move together and pinned the field at 1.0).

    A zero count now always carries this record beside it, so "nobody was ever measured" and "the athlete
    stood still" stop reading the same. It is a MEASUREMENT record, deliberately NOT a failure channel:
    a subject who walks out of frame for 300 frames has not failed and the goal has not been dropped
    (that is what ``RhythmDrive.arm_failures`` / ``pop_mid_run_failures`` mean, and routing this there
    would make a recoverable absence look like a dead goal).
    """

    observed: int = 0        # frames offered to this fold
    measured: int = 0        # ...of which carried at least one usable measurement
    nonfinite: int = 0       # channel readings dropped for being NaN/inf/not a number (never folded)
    rewinds: int = 0         # frames whose timestamp stepped BACKWARDS (a seek / reconnected timebase)
    unnormalized: int = 0    # frames whose pose normalizer was the clamp, not the body — the channel
    #                          vector is NOT in torso-lengths, so no ABSOLUTE amplitude gate is valid

    @property
    def coverage(self) -> float | None:
        """``measured / observed`` — or ``None`` before the first frame, which is not the same as 0.0."""
        return None if self.observed <= 0 else self.measured / self.observed


@dataclass(frozen=True)
class Rep:
    """One confirmed repetition as a first-class SPAN (the start/peak/stop the system lacked)."""

    n: int  # 1-based index on its channel
    channel: str
    start_t: float  # opening trough (the rep's start)
    peak_t: float  # the peak (direction reversal)
    stop_t: float  # closing reversal after a prominent peak (the causal count instant)
    amplitude: float  # peak prominence — rise above the bounding trough


@dataclass
class _ChannelState:
    """Causal DIFFERENTIAL rep detector for ONE channel — streaming, no forward stats.

    A rep is a trajectory ``trough → peak → trough``, i.e. an up-run then a down-run of the first
    difference. Rather than amplitude bands (which are scale-sensitive and miss fatigued reps), this works
    on the **velocity sign** with two guards:

      * **run-length (in time)** — the confirmed direction flips only after the opposite direction has
        persisted ``min_run_s`` *seconds* (not sample-steps, so the counter is rate-agnostic and the window
        auto-scales to the learned period), so a brief wobble can't mint a turn ("t0→t1 same dir as t1→t2?").
      * **prominence** — a peak is counted only if its rise above the bounding trough clears
        ``prom_frac × recent_amp`` (relative ⇒ fatigue-adaptive), so a shallow aborted turn — a peak that
        was never really achieved — is skipped.

    A rep is registered causally at the closing reversal after a prominent peak; ``amp`` is the MEDIAN of
    the last ``amp_window`` accepted prominences and the median inter-peak interval debounces sub-cadence
    jitter. Amplitude-free turn location + scale-invariant sign make this robust to drift and comparable
    across channels.

    WHY A MEDIAN WINDOW AND NOT AN EMA. ``amp`` used to be an EMA seeded from a SINGLE sample and updated
    only inside the acceptance branch — so one outsized opening movement set a bar
    (``prom_frac × amp``) that no honest rep could clear, and a rejection never decayed it again. Measured:
    a 10.0 opening excursion followed by six 1.0 ones counted **1**; the SAME six 1.0 excursions followed
    by the 10.0 counted **6**. The order of identical movements moved the reported count by 6x. A median
    over a short window of accepted prominences cannot be dragged by one outlier, and the relative gate is
    held OFF until that window is full (``floor``/``min_amp`` alone govern the opening reps), so the first
    rep can never set the bar for the rest of the run. The cost, stated plainly: during warm-up the only
    jitter rejection is the fixed ``floor`` and the direction-persistence window, so a noisy channel can
    admit an extra early excursion — which is the trade for a count that does not depend on rep order.
    """

    name: str
    # --- tunables (swept offline across actions) -------------------------------------------------------
    floor: float = 0.04  # cold-start minimal prominence (reject still-scene jitter before any rep)
    min_amp: float = 0.0  # ABSOLUTE prominence floor from the action prior (expected ROM × gate-frac), in THIS
    #                       channel's NATIVE units — the same units as ``Rep.amplitude``. 0 = off. Distinct from
    #                       ``floor`` (a tiny fixed cold-start jitter guard) and ``prom_frac`` (RELATIVE to the
    #                       recent amplitude): this rejects movement too small to be the target rep — the fidget
    #                       false-positive — and only the prior channel sets it (see RhythmTracker._state).
    #                       DECLINED for any run containing a frame whose normalizer was CLAMPED — see
    #                       ``observe``'s ``floored`` argument; an absolute gate is only meaningful while the
    #                       units it is stated in are a real measurement.
    # Direction-run debounce is in TIME, not sample-steps, so the counter is RATE-AGNOSTIC: a turn confirms
    # once the opposite direction has persisted >= min_run_s seconds. Cold-start ~0.3s (≈ 2 steps @6fps);
    # once a period is learned it scales to period_frac × period (clamped), so a fast 1 Hz rep keeps a short
    # debounce instead of the fixed 0.33s that ate pommelhorse, while a slow rep keeps strong jitter rejection.
    min_run_s: float = 0.15  # direction-persistence window (seconds), fixed & rate-agnostic. Jitter is
    #                          sub-second regardless of rep speed, so a small FIXED window rejects it; it is
    #                          only ever shortened (never grown) for ultra-fast reps via period_cap_frac.
    period_cap_frac: float = 0.4  # debounce never exceeds this × the learned rep period (shrink for fast reps)
    prom_frac: float = 0.2  # a peak counts only if prominence >= this fraction of the recent amplitude
    amp_window: int = 3  # how many recent ACCEPTED prominences the amplitude estimate is the median of.
    #                      Replaces the single-sample-seeded EMA (see the class docstring): an outlier can
    #                      no longer own the bar, and the relative gate stays OFF until this many reps have
    #                      been accepted, so rep #1 cannot set a bar rep #2 must clear. Small on purpose —
    #                      3 keeps the gate armed by the time an aborted turn can show up, and keeps the
    #                      estimate fatigue-adaptive (it tracks the RECENT reps, not the whole run).
    period_debounce: float = 0.45  # reject a rep closer than this fraction of the median rep interval
    smooth_s: float = 0.0  # causal time-window moving-average span in SECONDS (0 = off; also the cold-start
    #                        window before a period is learned). The rate-invariance fix: the differential
    #                        sign ``sign(v - prev)`` is noise-dominated at high fps — per-sample jitter keeps
    #                        flipping the tentative direction, resetting the unbroken-run clock so reversals
    #                        are missed and the count COLLAPSES as fps rises (the "denser-is-worse"
    #                        pathology). A fixed-TIME low-pass (a trailing window, so causal) pins the noise
    #                        bandwidth to seconds, so the smoothed signal — and thus the count — survives at
    #                        any fps above the floor. The validated offline shape used a fixed-seconds
    #                        savgol; this is its streaming, causal equivalent.
    smooth_period_frac: float = 0.18  # once a rep period is learned, the window tracks this fraction of it
    #                                   (like ``min_run_s`` -> ``_min_run_s``): a fast rep gets a shorter
    #                                   low-pass (its turns survive), a slow rep a longer one (more noise
    #                                   killed) — so ONE setting is period-invariant across cadences, not a
    #                                   per-activity constant. No goal-type table; the period is self-measured.
    smooth_cap_s: float = 0.5  # never smooth over more than this (bound the lag / protect slow reps' turns)

    # --- causal streaming state ------------------------------------------------------------------------
    _win: deque = field(default_factory=deque)  # recent (t, raw_v) inside smooth_s, for the causal smoother
    amp: float = 0.0  # recent prominence magnitude (MEDIAN of the last amp_window confirmed reps)
    last_stop_t: float | None = None
    intervals: list[float] = field(default_factory=list)  # inter-rep gaps (for the period debounce)
    reps: list[Rep] = field(default_factory=list)
    _recent_amps: deque = field(default_factory=deque)  # the last amp_window ACCEPTED prominences
    _prev: float | None = None
    _dir: int = 0  # confirmed direction: +1 rising · -1 falling · 0 unknown
    _tent: int = 0  # tentative direction (sign of the last diff)
    _run_start_t: float = 0.0  # when the current tentative-direction run began (for the time debounce)
    _ext_v: float | None = None  # running extreme of the confirmed segment (max if rising, min if falling)
    _ext_t: float = 0.0
    _last_trough_v: float | None = None  # last confirmed trough (a rep's opening)
    _last_trough_t: float = 0.0
    _time: StreamTimeGuard = field(default_factory=StreamTimeGuard)  # the caller's timeline is untrusted
    _floored_run: bool = False  # any frame since the last reversal was normalized by the CLAMP, not the body
    _floored_frame: bool = False  # ...and the current frame's own answer to that question
    min_amp_declines: int = 0  # how many reversals were judged with the absolute gate DECLINED

    def _median_interval(self) -> float | None:
        if not self.intervals:
            return None
        return _median(self.intervals)

    def _effective_amp(self) -> float:
        """The bar the RELATIVE prominence gate is a fraction of: the median of the recent accepted
        prominences (never one outlier), never below the fixed cold-start ``floor``."""
        return max(self.amp, self.floor)

    def _relative_gate_armed(self) -> bool:
        """Whether the relative gate may be applied at all. It is held off until ``amp_window`` reps have
        been accepted: with fewer, the estimate it is a fraction of is one or two samples wide, so applying
        it would let the opening rep set a bar for every rep after it (see the class docstring)."""
        return len(self._recent_amps) >= self.amp_window

    def _record_amplitude(self, prom: float) -> None:
        """Fold one ACCEPTED prominence into the recent-amplitude window, then republish the median."""
        self._recent_amps.append(prom)
        while len(self._recent_amps) > self.amp_window:
            self._recent_amps.popleft()
        self.amp = _median(self._recent_amps)

    def _reseed(self) -> None:
        """Forget every quantity measured against the ABANDONED timeline, keeping the reps already counted.

        A rewound ``t`` is a seek or a reconnected media timebase — a recoverable source property, not a
        programming error — so this is the RESET policy :class:`~percept._streamtime.StreamTimeGuard`
        leaves to its caller (:class:`~percept.harness.tier0.MotionGate` chose the same one, for the same
        reason; :mod:`percept.interest` raises instead, because its intervals are an evidence record).
        Without it ``t - _run_start_t`` stays negative for the whole length of the rewind, so no direction
        run can ever reach its persistence window and every rep in that span is dropped in silence
        (measured: 9 reps forward, 4 across a 5 s rewind). ``last_stop_t`` goes too — it is a position on
        the old timeline, and left in place it makes the period debounce reject the next rep as well.
        Reps, intervals and the amplitude window are DURATIONS and magnitudes: they survive."""
        self._win.clear()
        self._prev = None
        self._dir = 0
        self._tent = 0
        self._ext_v = None
        self._last_trough_v = None
        self.last_stop_t = None
        self._floored_run = self._floored_frame

    def _min_run_s(self) -> float:
        """The direction-persistence window in seconds — a small FIXED window (rate-agnostic), only
        SHORTENED for ultra-fast reps so it never exceeds a fraction of the period (and never GROWN, which
        would let real jitter ticks reset the run clock and starve slow reps of confirmations)."""
        med = self._median_interval()
        if med is None:
            return self.min_run_s
        return min(self.min_run_s, self.period_cap_frac * med)

    def _smooth_window(self) -> float:
        """The causal low-pass span in seconds. Off when ``smooth_s`` is 0. Until a period is learned it is
        the cold-start ``smooth_s``; once the median rep interval is known it tracks
        ``smooth_period_frac × period`` (clamped to ``smooth_cap_s``), so the SAME setting gives a short
        window to a fast rep and a long one to a slow rep — period-invariant by construction, not tuned."""
        if self.smooth_s <= 0.0:
            return 0.0
        med = self._median_interval()
        if med is None:
            return self.smooth_s
        return min(self.smooth_cap_s, self.smooth_period_frac * med)

    def observe(self, t: float, v: float, floored: bool = False) -> Rep | None:
        """Fold one sample. A reversal confirms once the opposite direction has persisted ``min_run_s``
        SECONDS (time, not steps -> rate-agnostic); the running extreme of the confirmed segment is the
        turning point. (A retreat-from-extreme variant over-counts on noisy pose; an unbroken time-run is
        the better real-data trade-off — see git history / the fovea validation.)

        ``v`` is assumed already validated — :meth:`RhythmTracker.observe` is the seam that refuses a
        non-finite reading, and this stays a pure sign-and-run detector on purpose (every comparison here
        is False against a NaN, so a guard *inside* the sign expression would be the wrong place for it).

        ``floored`` says the frame's channel vector was normalized by the pose clamp rather than by the
        body, so it is NOT in torso-lengths. It DECLINES the absolute ``min_amp`` gate for the run it
        lands in and is tallied in :attr:`min_amp_declines`; ``t`` going backwards re-seeds (:meth:`_reseed`)."""
        self._floored_frame = bool(floored)
        if self._floored_frame:
            self._floored_run = True
        if self._time.check(f"rhythm channel {self.name!r} timestamp", t) == "rewind":
            self._reseed()
        w = self._smooth_window()
        if w > 0.0:  # causal fixed-TIME low-pass: replace v with the mean over the trailing window (t-w, t].
            self._win.append((t, v))  # trailing window -> bounded lag, rate-invariant noise bandwidth.
            cutoff = t - w
            while self._win and self._win[0][0] < cutoff:
                self._win.popleft()
            v = sum(val for _, val in self._win) / len(self._win)
        if self._prev is None:  # seed; nothing to count yet
            self._prev = v
            self._ext_v, self._ext_t = v, t
            self._last_trough_v, self._last_trough_t = v, t
            self._run_start_t = t
            return None
        d = 0 if v == self._prev else (1 if v > self._prev else -1)
        self._prev = v
        if d == 0:  # flat step is a continuation
            return None
        if d != self._tent:  # direction changed -> start a fresh tentative run clock
            self._tent = d
            self._run_start_t = t

        # extend the running extreme of the CONFIRMED direction
        if self._dir == 1 and (self._ext_v is None or v > self._ext_v):
            self._ext_v, self._ext_t = v, t
        elif self._dir == -1 and (self._ext_v is None or v < self._ext_v):
            self._ext_v, self._ext_t = v, t

        if self._tent != self._dir and (t - self._run_start_t) >= self._min_run_s():
            return self._confirm_reversal(t, v)
        return None

    def _confirm_reversal(self, t: float, v: float) -> Rep | None:
        rep: Rep | None = None
        if self._dir == 1 and self._last_trough_v is not None:
            # was rising → the running max is a PEAK. Count the rep at the confirmed downturn (low latency;
            # captures the final rep even if the clip ends before the next trough). Prominence vs the trough
            # we departed from (causal).
            peak_v, peak_t = self._ext_v, self._ext_t
            prom = peak_v - self._last_trough_v
            # The ABSOLUTE gate is stated in torso-normalised units, so it is only applicable while the
            # normalizer really was the torso. Declining it (rather than applying it to numbers scaled by
            # the clamp) is what keeps an anti-fidget gate from silently becoming a DISTANCE gate.
            gate = 0.0 if self._floored_run else self.min_amp
            if self.min_amp > 0.0 and self._floored_run:
                self.min_amp_declines += 1
            relative = (not self._relative_gate_armed()) or prom >= self.prom_frac * self._effective_amp()
            if relative and prom >= self.floor and prom >= gate:
                med = self._median_interval()
                debounced = (
                    med is not None and self.last_stop_t is not None
                    and (t - self.last_stop_t) < self.period_debounce * med
                )
                if not debounced:
                    self._record_amplitude(prom)
                    if self.last_stop_t is not None:
                        self.intervals.append(t - self.last_stop_t)
                    self.last_stop_t = t
                    rep = Rep(n=len(self.reps) + 1, channel=self.name,
                              start_t=self._last_trough_t, peak_t=peak_t, stop_t=t, amplitude=prom)
                    self.reps.append(rep)
        elif self._dir == -1:
            self._last_trough_v, self._last_trough_t = self._ext_v, self._ext_t  # trough opens the next rep
        self._dir = self._tent
        self._ext_v, self._ext_t = v, t
        self._floored_run = self._floored_frame  # the next run is judged on its OWN frames' normalizers
        return rep

    # --- channel-selection score: large, regular cycles win ------------------------------------------
    def regularity(self) -> float:
        """1 / (1 + CV of inter-rep intervals) ∈ (0, 1]; 1.0 when perfectly periodic or <2 reps."""
        xs = self.intervals
        if len(xs) < 2:
            return 1.0
        mean = sum(xs) / len(xs)
        if mean <= 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        cv = (var ** 0.5) / mean
        return 1.0 / (1.0 + cv)

    def score(self) -> float:
        """Channel salience for selection: rep evidence × amplitude × regularity."""
        return len(self.reps) * self.amp * self.regularity()


def has_evidence(reps: float, amp: float, max_amp: float, amp_frac: float) -> bool:
    """Whether a channel has enough evidence to be allowed to OWN the count.

    Two conditions, and the first is the one that was missing:

      * ``reps >= 2`` — a single rep has no INTERVAL, and therefore no regularity. ``regularity()``
        returns ``1.0`` below two reps (it is a *default*, not a measurement), which is precisely why a
        one-rep channel used to score so well and win selections it had no evidence for.
      * ``amp >= amp_frac * max_amp`` — the channel oscillates at a real fraction of the loudest one, the
        amplitude gate the medoid already applied.

    ``reps`` is a rep TALLY on the legacy path and a support in PERIODS on the canonical one; both mean
    "how many cycles did I actually see", which is the quantity the floor is about.

    Callers apply this as a filter with a fall-back to the unfiltered set (``... or live``): the floor
    decides WHO wins when there is evidence, it never leaves the selector with nothing. That keeps the
    cold-start contract — one departure-and-return is still one rep, and the single live channel that
    carries it is still selected — while stopping a one-rep channel from beating a twenty-rep one."""
    return reps >= 2 and amp >= amp_frac * max_amp


def select_channel(
    scores: dict[str, ChannelScore],
    prior: str | None = None,
    current: str | None = None,
    margin: float = 0.25,
    amp_frac: float = 0.5,
) -> str | None:
    """Pick the rep-owning channel from canonical :class:`ChannelScore`s, with hysteresis (Build 3).

    The policy that makes "which channel is the rep" stable, not jittery:

      * **Trust an EVIDENCED prior.** If the goal author named a channel and that channel has cleared
        :func:`has_evidence` — at least two observed cycles, at a real fraction of the loudest channel's
        amplitude — use it: the rep shape comes from goal-setting, not runtime guessing (the
        architecture's point). "Viable" used to mean a bare ``score > 0.0``, which a single incidental
        movement satisfies, so a mis-authored prior catching one twitch permanently owned the count
        (measured: ``selected == 'wrist'``, ``count == 1``, against a 20-rep channel).
      * **Otherwise the highest canonical score wins** — but it only DISPLACES the incumbent ``current`` if
        it beats it by ``margin`` (a relative margin), so a sampling wobble can't flip the selection without
        real evidence. The dwell ("...for several reselects") is enforced by the caller (RhythmTracker).

    The prior remains a SHORT-CIRCUIT, not a weighted bias. That is deliberate architecture, not an
    oversight, and turning it into a bias is a product decision with a conductor-side owner — it is parked,
    not pending. What changed here is only the evidence floor underneath it.

    ``scores`` should already be restricted to viable candidates (channels with confirmed reps), preserving
    the invariant that a channel with zero reps is never selected. Returns ``current`` if nothing is viable."""
    scored = {k: s for k, s in scores.items() if s.score > 0.0}
    if not scored:
        return current
    max_amp = max(s.amplitude for s in scored.values())
    evidenced = {k: s for k, s in scored.items()
                 if has_evidence(s.support, s.amplitude, max_amp, amp_frac)}
    viable = evidenced or scored  # the floor never leaves the selector empty — see has_evidence
    if prior and prior in viable:
        return prior
    champion = max(viable.values(), key=lambda s: s.score).channel
    if current and current in viable and champion != current:
        if viable[champion].score <= viable[current].score * (1.0 + margin):
            return current  # challenger not decisive enough -> hold the incumbent
    return champion


@dataclass
class RhythmTracker:
    """Per-concern rhythm sense: run a causal detector per channel, count the dominant one.

    Feed it the eye's per-frame channel measurements via :meth:`observe`. ``count`` is the confirmed-rep
    tally of the currently-selected (most rep-shaped) channel — the value the brain owns. ``selected`` and
    ``reps`` expose which channel won and the start/peak/stop spans for debugging / the span contract.

    A ``prior`` (set at goal-authoring by the conductor) may name a preferred channel and an expected range
    of motion. It is never required — the benchmark runs prior-less — but when it IS set and the channel it
    names has cleared :func:`has_evidence`, it **short-circuits selection outright**: an evidenced prior
    wins, whatever the other channels scored. That is the shipped behaviour and it is deliberate (the rep
    shape comes from goal-setting, not runtime guessing); making the prior a weighted BIAS instead is a
    parked product decision with a conductor-side owner, not a pending fix. The prose here used to say
    "biases selection ... but is never required" while :func:`select_channel` said "trust a viable prior" —
    the code implemented the second, and this is now what both say.

    ``measurement`` is the evidence base beneath the count (see :class:`MeasurementRecord`): how many
    frames were folded, how many carried a usable reading, and how many were dropped for a non-finite
    channel value, a rewound timestamp, or a clamped normalizer. A zero count is only interpretable
    against it.
    """

    prior_channel: str | None = None
    expected_rom: float | None = None  # expected rep prominence on the PRIOR channel, in that channel's native
    #                                    units (see _ChannelState.min_amp). Set by the conductor; None = no gate.
    amp_gate_frac: float = 0.5  # a rep must reach this fraction of expected_rom to be counted
    floor: float = 0.04
    amp_frac: float = 0.5  # a channel must oscillate at >= this fraction of the strongest to be eligible
    smooth_s: float = 0.0  # causal fixed-time smoothing cold-start span (seconds), applied to EVERY channel
    #                        and then scaled to the measured period (see _ChannelState.smooth_s /
    #                        _smooth_window). DEFAULT 0.0 = byte-for-byte legacy. It is an OPT-IN knob, not the
    #                        default, on evidence: it fixes the high-fps COLLAPSE (synthetic 12->0 becomes
    #                        recall>=0.8; real-pose fps12 |err| 4.0->2.5) but REGRESSES the cached fps=6 bench
    #                        (primed OBO 0.50->0.38) — at low fps there is no high-fps noise to remove and the
    #                        lag perturbs counts / couples to the still-fps-dependent selector. Revisit the
    #                        default after Build 3 (rate-invariant selector) enables above-floor operation.
    # --- Build 3: rate-invariant channel selection (opt-in; default off keeps the legacy medoid) ----------
    canonical_select: bool = False  # score channels on the CANONICAL grid (support × robust_amplitude ×
    #                                 regularity × stability, all in periods/seconds) instead of the medoid
    #                                 over fps-dependent rep tallies — so "which channel is the rep" stops
    #                                 flipping with fps (real-pose cross-fps consistency 0/4 -> 2/4; fps=6
    #                                 bench accuracy unchanged 0.50/0.83). Default off = byte-for-byte legacy.
    select_window_s: float = 30.0  # bounded recent history scored per reselect (live-stream memory cap)
    reselect_every_s: float = 1.0  # recompute the selection at most this often (bounds the scoring cost)
    select_hz: float = CANON_HZ  # canonical resample rate every channel is scored on (a shared electorate)
    select_band_s: tuple[float, float] | None = (0.3, 5.0)  # plausible rep-period band (fast pommel .. slow bench)
    select_margin: float = 0.25  # a challenger must beat the incumbent's score by this fraction to switch
    select_dwell: int = 2  # ...and stay ahead for this many consecutive reselects (margin OVER a dwell)
    _channels: dict[str, _ChannelState] = field(default_factory=dict)
    _last_rep: Rep | None = None
    _series: dict[str, deque] = field(default_factory=dict)  # raw (t, v) ring per channel, for canonical scoring
    _sel_cache: str | None = None  # last canonical selection (recomputed on the reselect cadence)
    _last_reselect_t: float | None = None
    _pending: str | None = None  # a challenger awaiting its dwell before it displaces _sel_cache
    _pending_count: int = 0
    measurement: MeasurementRecord = field(default_factory=MeasurementRecord)  # the count's evidence base
    _time: StreamTimeGuard = field(default_factory=StreamTimeGuard)  # frame-level timeline bookkeeping

    def _state(self, channel: str) -> _ChannelState:
        st = self._channels.get(channel)
        if st is None:
            # The amplitude gate is bound to the PRIOR channel ONLY: expected_rom is in that channel's native
            # units, so it is meaningless on any other channel (a horizontal/angular channel has different
            # units). If there is no prior channel, or this is a different channel, the gate stays off.
            min_amp = (self.amp_gate_frac * self.expected_rom
                       if self.expected_rom is not None and channel == self.prior_channel else 0.0)
            st = _ChannelState(name=channel, floor=self.floor, min_amp=min_amp, smooth_s=self.smooth_s)
            self._channels[channel] = st
        return st

    def observe(self, t: float, channels: dict[str, float], *, scale_floored: bool = False) -> Rep | None:
        """Fold one frame's channel vector. Returns the :class:`Rep` if THIS frame confirmed a rep on the
        currently-selected channel (the causal, low-latency count event), else ``None``.

        THIS is the validating seam — the one place a caller's channel vector crosses into the sense — so
        it is where the firewall lives, and the per-channel detector below it stays a pure sign-and-run
        machine. A reading that is not a real finite number is DROPPED for this frame (a NaN otherwise
        classifies as a downward step, because every comparison against it is False: it mints a phantom
        reversal and is then carried into ``_prev``, poisoning the state) and counted in
        ``measurement.nonfinite`` so the drop is visible rather than showing up as a count one lower than
        the athlete performed. The cost of a drop is typically one rep, not the run: the next confirmed
        reversal re-seeds the running extreme.

        ``scale_floored`` is the eye's report that this frame's channel vector was normalized by the pose
        clamp instead of by the body (``percept.harness.pose.body_scale_measured`` /
        ``PersonMeasure.scale_floored``). It is NOT in torso-lengths, so the absolute ``min_amp`` gate —
        which is stated in those units — is declined for the affected run and the decline is recorded, the
        same way an occluded channel's coverage lets a caller decline to trust the pose phase source.
        A caller that never passes it gets the old behaviour, which is the other half of the contract:
        *either the clamp is observable to the consumer, or the absolute gate is not applied when the
        normalizer is not a real measurement.*"""
        self.measurement.observed += 1
        if self._time.check("rhythm frame timestamp", t) == "rewind":
            self.measurement.rewinds += 1
        if scale_floored:
            self.measurement.unnormalized += 1
        clean: dict[str, float] = {}
        for ch, v in channels.items():
            try:
                clean[ch] = finite(f"rhythm channel {ch!r}", v)
            except ValueError:
                self.measurement.nonfinite += 1  # dropped for this frame; never reaches _ChannelState
        if clean:
            self.measurement.measured += 1
        confirmed: dict[str, Rep] = {}
        for ch, v in clean.items():
            rep = self._state(ch).observe(t, v, floored=scale_floored)
            if rep is not None:
                confirmed[ch] = rep
        if self.canonical_select:  # accumulate raw samples + recompute the canonical selection on a cadence
            for ch, v in clean.items():
                dq = self._series.setdefault(ch, deque())
                dq.append((t, v))
                cutoff = t - self.select_window_s
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
            if self._last_reselect_t is None or (t - self._last_reselect_t) >= self.reselect_every_s:
                self._reselect(t)
        sel = self.selected
        if sel is not None and sel in confirmed:
            self._last_rep = confirmed[sel]
            return confirmed[sel]
        return None

    def _reselect(self, t: float) -> None:
        """Recompute the canonical channel selection (Build 3). Scores only channels that have confirmed
        reps (so a dead channel is never selected), on the shared canonical grid, then applies
        :func:`select_channel`'s prior/margin hysteresis with a dwell: a challenger must beat the incumbent
        by ``select_margin`` for ``select_dwell`` consecutive reselects before it displaces the selection."""
        self._last_reselect_t = t
        scores: dict[str, ChannelScore] = {}
        for name, st in self._channels.items():
            dq = self._series.get(name)
            if st.reps and dq is not None and len(dq) >= 6:
                series = MeasurementSeries(channel=name, samples=tuple(dq))
                scores[name] = channel_score(series, hz=self.select_hz, band_s=self.select_band_s)
        if not scores:
            return
        prior = self.prior_channel if (self.prior_channel and self.prior_channel in scores) else None
        champion = select_channel(scores, prior=prior, current=self._sel_cache, margin=self.select_margin,
                                  amp_frac=self.amp_frac)
        if champion == self._sel_cache or champion is None:
            self._pending, self._pending_count = None, 0
            return
        if champion == self._pending:  # same challenger winning again -> advance its dwell
            self._pending_count += 1
        else:
            self._pending, self._pending_count = champion, 1
        if self._sel_cache is None or self._pending_count >= self.select_dwell:
            self._sel_cache = champion  # cold-start commits immediately; a switch waits out the dwell
            self._pending, self._pending_count = None, 0

    @property
    def selected(self) -> str | None:
        """The channel that owns the count — the canonical score (Build 3, opt-in) or the legacy medoid."""
        if self.canonical_select and self._sel_cache is not None:
            return self._sel_cache  # canonical selection (recomputed on the reselect cadence)
        return self._medoid_selected()  # legacy / canonical cold-start before the first reselect

    def _has_evidence(self, reps: int, amp: float, max_amp: float) -> bool:
        """This tracker's evidence floor for owning the count — :func:`has_evidence` at ``amp_frac``."""
        return has_evidence(reps, amp, max_amp, self.amp_frac)

    def _medoid_selected(self) -> str | None:
        """The legacy selector: trust an EVIDENCED prior, else the amplitude-gated medoid.

        With a goal-set ``prior_channel`` (e.g. the conductor knows "bench press" → the arms), trust it —
        that is the architecture's point: the rep shape comes from goal-setting, not runtime guessing. It
        must however have EVIDENCE (:func:`has_evidence`): "has one confirmed rep" was the whole bar, and a
        single incidental movement clears it, so a mis-authored prior did not bias the answer — it replaced
        it (measured: a 1-rep ``wrist`` prior owning the count against a 20-rep ``torso``).

        Unsupervised (the benchmark case), which body part carries "the rep" is genuinely ambiguous — for
        one bench-press routine the arms read 17, the legs 26, the torso 9 for the SAME motion. We pick the
        **amplitude-gated medoid**: drop channels that barely oscillate (tiny-noise) by amplitude, then take
        the consensus channel (count closest to the median of the rest), which rejects both the jitter
        over-counter and the sub-harmonic under-counter.

        The tie-break is ``(distance from the median, -tally, -regularity)``. The tally is in there because
        with exactly TWO live channels the median IS their mean, so both sit the same distance from it and
        the first key always ties — leaving ``regularity`` to decide alone, and ``regularity()`` returns
        ``1.0`` below two reps. A 2-rep channel therefore beat a 40-rep one, and 2 was the reported count.
        Consensus first, then WEIGHT of evidence, then shape."""
        live = [s for s in self._channels.values() if s.reps]
        if not live:
            return None
        max_amp = max(s.amp for s in live)
        # The evidence floor and the amplitude gate are the same predicate; `or live` is the pre-existing
        # fall-back that keeps a cold start selectable (one rep on one channel is still one rep).
        strong = [s for s in live if self._has_evidence(len(s.reps), s.amp, max_amp)] or live
        if self.prior_channel:
            pc = self._channels.get(self.prior_channel)
            if pc is not None and pc in strong:
                return self.prior_channel
        if len(strong) == 1:
            return strong[0].name
        counts = sorted(len(s.reps) for s in strong)
        m = len(counts) // 2
        median = counts[m] if len(counts) % 2 else 0.5 * (counts[m - 1] + counts[m])
        best = min(strong, key=lambda s: (abs(len(s.reps) - median), -len(s.reps), -s.regularity()))
        return best.name

    @property
    def count(self) -> int:
        sel = self.selected
        return len(self._channels[sel].reps) if sel is not None else 0

    @property
    def reps(self) -> list[Rep]:
        sel = self.selected
        return list(self._channels[sel].reps) if sel is not None else []

    @property
    def last_rep(self) -> Rep | None:
        return self._last_rep

    @property
    def regularity(self) -> float:
        """Periodicity of the selected channel — ``1 / (1 + CV)`` of its inter-rep intervals, in (0, 1].
        ``0.0`` with fewer than two reps (no interval evidence yet) so it reads as low confidence, not high."""
        sel = self.selected
        if sel is None:
            return 0.0
        st = self._channels[sel]
        return st.regularity() if len(st.reps) >= 2 else 0.0

    @property
    def measurement_coverage(self) -> float | None:
        """The fraction of folded frames that carried a usable reading — ``None`` before the first frame.

        The denominator under :attr:`count`. A zero count with coverage ``0.0`` means the eye never
        measured anything; a zero count with coverage ``1.0`` means the subject genuinely did not rep."""
        return self.measurement.coverage

    @property
    def min_amp_declined(self) -> bool:
        """Whether the absolute ``min_amp`` gate was ever DECLINED because the pose normalizer was clamped.

        ``True`` means at least one reversal was judged in units that are not torso-lengths, so the
        anti-fidget gate was not applied to it. A consumer that needs the gate's guarantee should treat
        this the way it treats a low ``channel_coverage`` — route to the silhouette fallback rather than
        assert the count."""
        return any(st.min_amp_declines for st in self._channels.values())

    def windowed_count(self, now: float, window_s: float) -> int:
        """Reps confirmed within the last ``window_s`` seconds — the endurance/concern story (mirrors
        :meth:`percept.episodes.EpisodeLog.count`): "how many reps in the last minute", bounded memory."""
        sel = self.selected
        if sel is None:
            return 0
        return sum(1 for r in self._channels[sel].reps if now - r.stop_t <= window_s)


# ======================================================================================================
# The RUNTIME DRIVE — how a ``state["type"] == "rhythm"`` goal reaches ``percept.run()``
# ======================================================================================================
#
# WHAT AN INCIDENT MEANS FOR A RHYTHM GOAL
# ----------------------------------------
# A rhythm watch COUNTS; it does not fire per judgment. The two are different kinds of output and this
# module keeps them apart:
#
#   * the COUNT is **state**. Every confirmed rep updates ``goal.state["count"]`` (and
#     ``core.edge_counts[goal_id]`` / ``core.rhythm[goal_id]``). Reading it is free and it is always
#     current — it is not an event, so it never needs an Incident to be observable.
#   * an INCIDENT is a **count MILESTONE** — the sparse, consequential moment the user asked to be
#     interrupted for ("tell me when I've done 20"). It is authored with the EXISTING
#     ``Goal.state.fire_when`` predicate over ``{n, t}`` (``"n == 20"``), already AST-firewalled by
#     :func:`percept.behavior.compile_fire_when` and already carried into the rhythm
#     :class:`~percept.behavior.BehaviorProgram` by :func:`percept.behavior.program_of`. No new
#     vocabulary, so an authored rhythm plan stays exactly as hashable/pinnable as it was.
#
# **A goal with no authored milestone emits NO incidents at all.** That is a deliberate, load-bearing
# default, not an oversight: the alternative — one incident per confirmed rep — is precisely the
# pathology that caps the judged path (one fire per judgment, so the count can never exceed the
# judgment density). A rep is not by itself worth interrupting a human for; the tally is. A caller that
# genuinely wants a per-rep stream authors ``fire_when: "n >= 1"`` and opts into it explicitly.
#
# The knock-on property that matters for the ceiling: because the count comes from the SIGNAL
# (RhythmTracker's causal reversal detection at native frame rate) and not from a judgment, a rhythm
# goal's count is bounded by the frame rate, not by how many times a judge was asked.

#: The measuring eye a rhythm goal gets when it authored no ``state["detectors"]`` of its own — the
#: decomposed pose-channel basis (the eye MEASURES; this module DECIDES).
RHYTHM_DEFAULT_DETECTORS: tuple[str, ...] = ("pose-channels",)


def rhythm_emits_incidents(state: object) -> bool:
    """Whether a rhythm goal's confirmed reps may become Incidents — i.e. it authored a milestone.

    ``True`` only when the goal carries a non-blank ``Goal.state["fire_when"]``. See this section's
    header for why the un-authored default is silence rather than one incident per rep.
    """

    if not isinstance(state, Mapping):
        return False
    expr = state.get("fire_when")
    return type(expr) is str and bool(expr.strip())


def _pose_preflight_failure(detector: object) -> str | None:
    """If ``detector`` is a pose-provider-backed measuring eye, warm its provider NOW so a missing
    extra or model surfaces at ARM TIME — before the frame loop ever calls ``measure`` — instead of
    on the first frame. Returns the underlying failure reason, or ``None`` when the detector needs no
    pose provider (a `.measure`-capable detector need not be pose-backed at all) or the provider
    warmed cleanly.

    ``percept.harness.detectors.PoseChannels`` sits outside this fix's owned files (only
    ``percept.rhythm``, two named lines of ``percept.runtime.worker``, ``percept.harness.tier0`` and
    ``percept.harness.pose`` are in scope), so this duck-types the shape its own public constructor
    (``provider=``/``provider_name=``) already accepts, rather than adding a public ``ensure_ready()``
    seam to that class. Duck-typed on purpose rather than ``isinstance``-checked: a detector module
    reloaded mid-session (``importlib.reload`` — see ``tests/test_registration.py``) mints a NEW class
    object, and an ``isinstance`` check against an import captured before that reload would silently
    stop matching the very shape it exists to recognise. A natural follow-up, once that file is in
    scope: give ``PoseChannels`` its own ``ensure_ready()`` and replace this probe with a plain method
    call.
    """
    if not hasattr(detector, "_provider") or not hasattr(detector, "_provider_name"):
        return None  # not pose-provider-backed at all — nothing to pre-flight
    try:
        provider = detector._provider  # noqa: SLF001 - see docstring: no detectors.py seam to call instead
        if provider is not None:
            provider.warm()
        else:
            from .harness.pose import ensure_pose_ready

            ensure_pose_ready(detector._provider_name)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - a missing pose extra fails only THIS goal, never the run
        return str(exc)
    return None


def _pose_unavailable_diagnostic(goal_id: str, reason: str) -> str:
    """The full client-facing diagnostic for one rhythm goal: names the GOAL, then the missing extra
    and the frame-driven alternative (:func:`percept.harness.pose.describe_pose_unavailable`)."""

    from .harness.pose import describe_pose_unavailable

    return f"rhythm goal {goal_id!r} cannot run: {describe_pose_unavailable(reason)}"


class RhythmDrive:
    """Drives every ``type="rhythm"`` goal on one source from the frame loop — the missing seam.

    ``percept.run`` and :class:`~percept.harness.tier0.GatedWatchLoop` own frames; this owns what a
    rhythm goal does with one. Per frame and per goal it asks the goal's MEASURING detectors for the
    channel vector and hands it to ``core.perceive_rhythm``, which folds it into that goal's
    :class:`RhythmTracker`. No tier-0 admission condition, no judge call, no vision token: a rhythm
    goal is compiled out of the :class:`~percept.adapters.WatchSpec` on purpose (see
    :func:`percept.behavior_routes.needs_watch_condition`) and counted from the signal instead.

    :meth:`observe` returns only the fire events that pass this goal's milestone policy
    (:func:`rhythm_emits_incidents`), so the default rhythm goal accrues a count and stays silent.

    Two independent containment layers, because one test cannot prove both: a goal whose pose
    provider is missing altogether is refused HERE, at construction (:attr:`arm_failures`) and never
    observed at all; a goal whose provider passes that pre-flight and then raises INSIDE a later
    ``detect()`` call (:mod:`percept.harness.pose`) is caught in :meth:`observe` and dropped from then
    on (:meth:`pop_mid_run_failures`). Both fail that ONE goal loudly — a diagnostic is always
    recorded — never the silent-zero :meth:`observe` documents refusing; and never any OTHER goal on
    this drive or on the source's Watch gate, which does not go through here at all.
    """

    def __init__(self, core: object, goals: Iterable[object]) -> None:
        self._core = core
        self._perceive = getattr(core, "perceive_rhythm", None)
        self._goals: dict[str, tuple[list, bool]] = {}
        #: Goal id -> diagnostic, for a goal refused before it was ever armed (see :attr:`arm_failures`).
        self._arm_failures: dict[str, str] = {}
        #: Goal id -> diagnostic, for a goal whose eye raised mid-run; drained by
        #: :meth:`pop_mid_run_failures` so a caller reports each exactly once.
        self._mid_run_failures: dict[str, str] = {}
        #: Goal id -> :class:`MeasurementRecord`: frames offered vs frames the eye actually measured.
        #: Deliberately NOT a failure channel — see :meth:`measurement_coverage`.
        self._measurement: dict[str, MeasurementRecord] = {}
        for goal in goals:
            goal_id = str(getattr(goal, "id", ""))
            measurers = self._measurers_for(goal)
            reason = None
            for detector in measurers:
                reason = _pose_preflight_failure(detector)
                if reason is not None:
                    break
            if reason is not None:
                self._arm_failures[goal_id] = _pose_unavailable_diagnostic(goal_id, reason)
                continue
            self._goals[goal_id] = (
                measurers,
                rhythm_emits_incidents(getattr(goal, "state", None)),
            )

    @staticmethod
    def _measurers_for(goal: object) -> list:
        """The goal's MEASURING edge skills: its authored detectors, else the pose-channel default.

        Only detectors exposing ``measure(frame) -> {channel: value}`` qualify — a boundary-only
        detector proposes occurrences, which is the OTHER primitive (``count`` -> ``perceive_boundary``).
        A rhythm goal whose authored detectors can none of them measure is a shape error, not a quiet
        no-op: it would run forever counting nothing.
        """

        from .harness.detectors import detectors_for, make_detector

        names = detectors_for(goal) or list(RHYTHM_DEFAULT_DETECTORS)
        measurers = [d for d in (make_detector(name) for name in names) if hasattr(d, "measure")]
        if not measurers:
            raise ValueError(
                f"rhythm goal {str(getattr(goal, 'id', '<unknown>'))!r} selected detectors {names!r}, "
                "none of which MEASURE channels; the rhythm sense needs a detector exposing "
                f"measure(frame) (e.g. {RHYTHM_DEFAULT_DETECTORS[0]!r})"
            )
        return measurers

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(self._goals)

    @property
    def arm_failures(self) -> dict[str, str]:
        """Every goal refused at construction (its pose provider could not be prepared), mapped to
        its diagnostic. These goals were never armed and :meth:`observe` never touches them; this is
        the ONLY record that they exist, so a caller should log/report it once at construction."""
        return dict(self._arm_failures)

    def measurement(self, goal_id: str) -> MeasurementRecord | None:
        """One goal's :class:`MeasurementRecord`, or ``None`` if it has never been observed."""
        return self._measurement.get(goal_id)

    def measurement_coverage(self, goal_id: str) -> float | None:
        """The fraction of observed frames on which this goal's eye actually MEASURED something.

        The third state this drive did not have. ``arm_failures`` and ``pop_mid_run_failures`` both mean
        "this goal was DROPPED and will never be measured again"; a subject who legitimately walks out of
        frame for 300 frames is not that, and neither is a provider that returns nobody on a dark frame.
        Without this, ``count(goal_id) == 0`` was byte-identical whether the eye returned ``{}`` on every
        single frame or the athlete simply stood still — the silent zero this drive's own docstring says
        it refuses to become. ``None`` before the first frame; ``0.0`` means measured nothing, ever."""
        record = self._measurement.get(goal_id)
        return None if record is None else record.coverage

    def pop_mid_run_failures(self) -> dict[str, str]:
        """Diagnostics for goals whose eye raised INSIDE :meth:`observe` since the last call, then
        were dropped. Draining (not just reading) is deliberate: a caller polls this once per frame
        and must see each failed goal exactly once, not once per subsequent frame — the goal itself
        already stopped being measured the instant it failed."""
        out = dict(self._mid_run_failures)
        self._mid_run_failures.clear()
        return out

    def __bool__(self) -> bool:
        return bool(self._goals) and self._perceive is not None

    def adopt(self, previous: "RhythmDrive | None") -> "RhythmDrive":
        """Carry a surviving goal's MEASURERS across a mid-run regate.

        The tracker itself lives on the core (``core.rhythm[goal_id]``) and is untouched by a regate;
        the eye's per-frame continuity state (the tracked person's hip anchor, the carried-forward
        channel values) lives in the detector and would otherwise cold-start mid-rep.
        """

        if previous is not None:
            for goal_id, (measurers, emits) in self._goals.items():
                prior = previous._goals.get(goal_id)
                if prior is not None:
                    self._goals[goal_id] = (prior[0], emits)
        return self

    def reset(self) -> None:
        """Cold-start every measuring eye (a pause/continuity break must not diff across the gap).

        Deliberately does NOT reset the trackers: the reps already counted happened, and a pause is
        not evidence they did not."""

        for measurers, _ in self._goals.values():
            for detector in measurers:
                detector.reset()

    def observe(self, t: float, frame: object) -> list:
        """Fold one frame into every rhythm goal; return the fire events that cleared the milestone.

        ``frame`` is whatever the frame loop PUBLISHED for this tick, handed to the eye verbatim —
        for a file/camera source that is the runtime's ``("image/jpeg", bytes)`` transport pair, the
        same object the judge would upload. This drive deliberately does not normalise it: the loop's
        payload is opaque by design (a judge uploads it, a clip retains it, a sink forwards it), and a
        measuring eye declares its own input. A frame-backed eye crosses transport to pixels through
        :func:`percept.harness.pose.frame_pixels` — one seam, shared with the boundary detectors that
        ``DetectorBank`` feeds off the same loop — and RAISES :class:`~percept.harness.pose.FrameNotDecodable`
        on a payload it cannot decode rather than returning an empty channel vector that would count
        zero forever. That RAISE propagates out of this call UNCHANGED — it is the frame loop's own
        payload that is bad, not this goal's provider, and it is very possibly transient (the next
        frame may decode fine); swallowing it here would be exactly the silent-zero this drive
        refuses to become.

        A DIFFERENT failure is contained here instead: a goal that passed arm-time pre-flight
        (:attr:`arm_failures`, provider ``warm()`` succeeded) whose provider then errors INSIDE a
        later ``detect()`` call — e.g. a backend crash, unlike a one-off bad frame this is likely to
        recur every frame. Per goal, not per call: that measurer's exception is caught, the goal is
        recorded in :meth:`pop_mid_run_failures` and dropped from every future frame — loudly (a
        diagnostic is always produced), never silently downgraded to counting zero — while every
        OTHER rhythm goal on this drive, and the Watch gate this drive never touches, keeps running
        unaffected.
        """
        from .harness.pose import FrameNotDecodable

        out: list = []
        if self._perceive is None or frame is None:
            return out
        for goal_id in list(self._goals):
            measurers, emits = self._goals[goal_id]
            record = self._measurement.setdefault(goal_id, MeasurementRecord())
            record.observed += 1  # BEFORE any exit below, so the denominator can never move with its numerator
            channels: dict[str, float] = {}
            floored = False
            try:
                for detector in measurers:
                    channels.update(detector.measure(frame))
                    # The eye's own report that it normalized by the CLAMP, not by the body. Duck-typed for
                    # the same reason the arm-time pre-flight is (see _pose_preflight_failure): a detector
                    # module reloaded mid-session mints a new class, and a non-pose eye has no such fact.
                    floored = floored or bool(getattr(detector, "_last_floored", False))
            except FrameNotDecodable:
                raise  # the frame loop's payload is bad, not this goal's provider — see docstring
            except Exception as exc:  # noqa: BLE001 - one rhythm goal's EYE must never end the watch for every other goal
                del self._goals[goal_id]
                self._mid_run_failures[goal_id] = _pose_unavailable_diagnostic(goal_id, str(exc))
                continue
            if not channels:
                # The eye saw nobody this frame — no measurement, no phantom cycle. Recorded, not silent:
                # `measurement_coverage` is what tells a zero count apart from an unmeasured one.
                continue
            record.measured += 1
            if floored:
                record.unnormalized += 1
            fired = self._perceive(goal_id, t=t, channels=channels, scale_floored=floored)
            if fired is not None and emits:
                out.append(fired)
        return out

    def count(self, goal_id: str) -> int:
        """The brain-owned confirmed-rep tally for one goal (0 before the first rep).

        Read :meth:`measurement_coverage` beside it: a zero here means "no rep was confirmed", which is
        only the athlete's answer when the eye was measuring at all."""

        tracker = getattr(self._core, "rhythm", {}).get(goal_id)
        return tracker.count if tracker is not None else 0


__all__ = [
    "RHYTHM_DEFAULT_DETECTORS",
    "MeasurementRecord",
    "Rep",
    "RhythmDrive",
    "RhythmTracker",
    "has_evidence",
    "rhythm_emits_incidents",
    "select_channel",
]
