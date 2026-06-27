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

from dataclasses import dataclass, field


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

    # --- causal streaming state ------------------------------------------------------------------------
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

    def observe(self, t: float, v: float) -> Rep | None:
        """Fold one sample. A reversal confirms once the opposite direction has persisted ``min_run_s``
        SECONDS (time, not steps -> rate-agnostic); the running extreme of the confirmed segment is the
        turning point. (A retreat-from-extreme variant over-counts on noisy pose; an unbroken time-run is
        the better real-data trade-off — see git history / the fovea validation.)"""
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
    _channels: dict[str, _ChannelState] = field(default_factory=dict)
    _last_rep: Rep | None = None

    def _state(self, channel: str) -> _ChannelState:
        st = self._channels.get(channel)
        if st is None:
            # The amplitude gate is bound to the PRIOR channel ONLY: expected_rom is in that channel's native
            # units, so it is meaningless on any other channel (a horizontal/angular channel has different
            # units). If there is no prior channel, or this is a different channel, the gate stays off.
            min_amp = (self.amp_gate_frac * self.expected_rom
                       if self.expected_rom is not None and channel == self.prior_channel else 0.0)
            st = _ChannelState(name=channel, floor=self.floor, min_amp=min_amp)
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
        sel = self.selected
        if sel is not None and sel in confirmed:
            self._last_rep = confirmed[sel]
            return confirmed[sel]
        return None

    @property
    def selected(self) -> str | None:
        """The channel that owns the count.

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

    def windowed_count(self, now: float, window_s: float) -> int:
        """Reps confirmed within the last ``window_s`` seconds — the endurance/concern story (mirrors
        :meth:`percept.episodes.EpisodeLog.count`): "how many reps in the last minute", bounded memory."""
        sel = self.selected
        if sel is None:
            return 0
        return sum(1 for r in self._channels[sel].reps if now - r.stop_t <= window_s)


__all__ = ["Rep", "RhythmTracker"]
