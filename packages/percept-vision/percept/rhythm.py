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

from .measurement import CANON_HZ, ChannelScore, MeasurementSeries, channel_score


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

    A rep is registered causally at the closing reversal after a prominent peak; ``amp`` is an EMA of rep
    prominences and the median inter-peak interval debounces sub-cadence jitter. Amplitude-free turn
    location + scale-invariant sign make this robust to drift and comparable across channels.
    """

    name: str
    # --- tunables (swept offline across actions) -------------------------------------------------------
    floor: float = 0.04  # cold-start minimal prominence (reject still-scene jitter before any rep)
    min_amp: float = 0.0  # ABSOLUTE prominence floor from the action prior (expected ROM × gate-frac), in THIS
    #                       channel's NATIVE units — the same units as ``Rep.amplitude``. 0 = off. Distinct from
    #                       ``floor`` (a tiny fixed cold-start jitter guard) and ``prom_frac`` (RELATIVE to the
    #                       recent amplitude): this rejects movement too small to be the target rep — the fidget
    #                       false-positive — and only the prior channel sets it (see RhythmTracker._state).
    # Direction-run debounce is in TIME, not sample-steps, so the counter is RATE-AGNOSTIC: a turn confirms
    # once the opposite direction has persisted >= min_run_s seconds. Cold-start ~0.3s (≈ 2 steps @6fps);
    # once a period is learned it scales to period_frac × period (clamped), so a fast 1 Hz rep keeps a short
    # debounce instead of the fixed 0.33s that ate pommelhorse, while a slow rep keeps strong jitter rejection.
    min_run_s: float = 0.15  # direction-persistence window (seconds), fixed & rate-agnostic. Jitter is
    #                          sub-second regardless of rep speed, so a small FIXED window rejects it; it is
    #                          only ever shortened (never grown) for ultra-fast reps via period_cap_frac.
    period_cap_frac: float = 0.4  # debounce never exceeds this × the learned rep period (shrink for fast reps)
    prom_frac: float = 0.2  # a peak counts only if prominence >= this fraction of the recent amplitude
    amp_alpha: float = 0.30  # EMA rate for the recent-prominence estimate (from confirmed reps)
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
    amp: float = 0.0  # recent prominence magnitude (causal EMA over confirmed reps)
    last_stop_t: float | None = None
    intervals: list[float] = field(default_factory=list)  # inter-rep gaps (for the period debounce)
    reps: list[Rep] = field(default_factory=list)
    _prev: float | None = None
    _dir: int = 0  # confirmed direction: +1 rising · -1 falling · 0 unknown
    _tent: int = 0  # tentative direction (sign of the last diff)
    _run_start_t: float = 0.0  # when the current tentative-direction run began (for the time debounce)
    _ext_v: float | None = None  # running extreme of the confirmed segment (max if rising, min if falling)
    _ext_t: float = 0.0
    _last_trough_v: float | None = None  # last confirmed trough (a rep's opening)
    _last_trough_t: float = 0.0

    def _median_interval(self) -> float | None:
        if not self.intervals:
            return None
        xs = sorted(self.intervals)
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    def _effective_amp(self) -> float:
        return max(self.amp, self.floor)

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

    def observe(self, t: float, v: float) -> Rep | None:
        """Fold one sample. A reversal confirms once the opposite direction has persisted ``min_run_s``
        SECONDS (time, not steps -> rate-agnostic); the running extreme of the confirmed segment is the
        turning point. (A retreat-from-extreme variant over-counts on noisy pose; an unbroken time-run is
        the better real-data trade-off — see git history / the fovea validation.)"""
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
            if prom >= self.prom_frac * self._effective_amp() and prom >= self.floor and prom >= self.min_amp:
                med = self._median_interval()
                debounced = (
                    med is not None and self.last_stop_t is not None
                    and (t - self.last_stop_t) < self.period_debounce * med
                )
                if not debounced:
                    self.amp = prom if self.amp == 0.0 else self.amp + self.amp_alpha * (prom - self.amp)
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


def select_channel(
    scores: dict[str, ChannelScore],
    prior: str | None = None,
    current: str | None = None,
    margin: float = 0.25,
) -> str | None:
    """Pick the rep-owning channel from canonical :class:`ChannelScore`s, with hysteresis (Build 3).

    The policy that makes "which channel is the rep" stable, not jittery:

      * **Trust a viable prior.** If the goal author named a channel and it has a positive score, use it —
        the rep shape comes from goal-setting, not runtime guessing (the architecture's point).
      * **Otherwise the highest canonical score wins** — but it only DISPLACES the incumbent ``current`` if
        it beats it by ``margin`` (a relative margin), so a sampling wobble can't flip the selection without
        real evidence. The dwell ("...for several reselects") is enforced by the caller (RhythmTracker).

    ``scores`` should already be restricted to viable candidates (channels with confirmed reps), preserving
    the invariant that a channel with zero reps is never selected. Returns ``current`` if nothing is viable."""
    viable = {k: s for k, s in scores.items() if s.score > 0.0}
    if not viable:
        return current
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

    A ``prior`` (set at goal-authoring by the conductor) may name a preferred channel and a cadence band;
    it biases selection and the cold-start floor but is never required — the benchmark runs prior-less.
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

    def observe(self, t: float, channels: dict[str, float]) -> Rep | None:
        """Fold one frame's channel vector. Returns the :class:`Rep` if THIS frame confirmed a rep on the
        currently-selected channel (the causal, low-latency count event), else ``None``."""
        confirmed: dict[str, Rep] = {}
        for ch, v in channels.items():
            rep = self._state(ch).observe(t, float(v))
            if rep is not None:
                confirmed[ch] = rep
        if self.canonical_select:  # accumulate raw samples + recompute the canonical selection on a cadence
            for ch, v in channels.items():
                dq = self._series.setdefault(ch, deque())
                dq.append((t, float(v)))
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
        champion = select_channel(scores, prior=prior, current=self._sel_cache, margin=self.select_margin)
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

    def _medoid_selected(self) -> str | None:
        """The legacy selector: trust a viable prior, else the amplitude-gated medoid.

        With a goal-set ``prior_channel`` (e.g. the conductor knows "bench press" → the arms), trust it —
        that is the architecture's point: the rep shape comes from goal-setting, not runtime guessing.

        Unsupervised (the benchmark case), which body part carries "the rep" is genuinely ambiguous — for
        one bench-press routine the arms read 17, the legs 26, the torso 9 for the SAME motion. We pick the
        **amplitude-gated medoid**: drop channels that barely oscillate (tiny-noise) by amplitude, then take
        the consensus channel (count closest to the median of the rest), which rejects both the jitter
        over-counter and the sub-harmonic under-counter. Tie-break on regularity."""
        live = [s for s in self._channels.values() if s.reps]
        if not live:
            return None
        if self.prior_channel:
            pc = self._channels.get(self.prior_channel)
            if pc is not None and pc.reps:
                return self.prior_channel
        if len(live) == 1:
            return live[0].name
        max_amp = max(s.amp for s in live)
        strong = [s for s in live if s.amp >= self.amp_frac * max_amp] or live
        if len(strong) == 1:
            return strong[0].name
        counts = sorted(len(s.reps) for s in strong)
        m = len(counts) // 2
        median = counts[m] if len(counts) % 2 else 0.5 * (counts[m - 1] + counts[m])
        best = min(strong, key=lambda s: (abs(len(s.reps) - median), -s.regularity()))
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
            channels: dict[str, float] = {}
            try:
                for detector in measurers:
                    channels.update(detector.measure(frame))
            except FrameNotDecodable:
                raise  # the frame loop's payload is bad, not this goal's provider — see docstring
            except Exception as exc:  # noqa: BLE001 - one rhythm goal's EYE must never end the watch for every other goal
                del self._goals[goal_id]
                self._mid_run_failures[goal_id] = _pose_unavailable_diagnostic(goal_id, str(exc))
                continue
            if not channels:
                continue  # the eye saw nobody this frame — no measurement, no phantom cycle
            fired = self._perceive(goal_id, t=t, channels=channels)
            if fired is not None and emits:
                out.append(fired)
        return out

    def count(self, goal_id: str) -> int:
        """The brain-owned confirmed-rep tally for one goal (0 before the first rep)."""

        tracker = getattr(self._core, "rhythm", {}).get(goal_id)
        return tracker.count if tracker is not None else 0


__all__ = [
    "RHYTHM_DEFAULT_DETECTORS",
    "Rep",
    "RhythmDrive",
    "RhythmTracker",
    "rhythm_emits_incidents",
    "select_channel",
]
