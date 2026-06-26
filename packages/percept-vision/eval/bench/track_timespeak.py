"""TimeSpeak — sim-core timeliness of the proactive response.

How soon after a cue's true onset does the agent speak, and does it fire on a GUESS (naive scripted
emitter) or wait for confidence to clear (core)? Each onset has a confidence RAMP (ambiguous -> clear).
Contrasts core (rising-edge three-state gate — refuses to guess) vs a no-VLM scripted baseline (fires
the instant the cue looks true).

  latency  : frames from onset to the first fire of the episode (lower = more timely)
  P-PAUC   : proactive AUC — 1 - latency/window if fired in the window, else 0; mean ×100
  guess%   : fraction of fires made while confidence < the gate threshold (firing on an ambiguous read)

Deterministic sim-core regression — the gate/executor are the real code.

This is not a publishable TML-comparable TimeSpeak result. The release track needs a held-out prompt
corpus, speak windows, early/late/guess accounting, and an edge-E2E adapter.
"""
from __future__ import annotations

from statistics import median

from . import sim
from .types import TrackResult

TRACK = "timespeak"
_FMTS = {
    "latency": "{:.1f}",
    "p-pauc": "{:.0f}",
    "guess%": "{:.0f}",
    "naive_latency": "{:.1f}",
    "naive_p-pauc": "{:.0f}",
    "naive_guess%": "{:.0f}",
}
_MEASURED = ["latency", "p-pauc", "guess%"]


def _first_latencies(fires, episodes):
    fire_ts = sorted(t for (a, t) in fires if a == "fire")
    out = []
    for e in episodes:
        first = next((t for t in fire_ts if e.onset <= t < e.onset + e.duration), None)
        out.append(None if first is None else first - e.onset)
    return out


def _pauc(lat, window):
    if not lat:
        return 0.0
    s = sum((1 - x / window) for x in lat if x is not None and 0 <= x <= window)
    return 100.0 * s / len(lat)


def _guess_rate(fires, script, thr):
    fire_ts = [t for (a, t) in fires if a == "fire"]
    if not fire_ts:
        return 0.0
    guesses = sum(1 for t in fire_ts if sim.conf_at(script, t) < thr)
    return 100.0 * guesses / len(fire_ts)


def score(*, live: bool = False, n: int = 8) -> TrackResult:
    dur, gap, ramp, window = 8, 7, 3, 5
    thr = sim.gate_threshold()
    episodes = [sim.Episode(onset=10 + i * (dur + gap), duration=dur) for i in range(n)]
    T = episodes[-1].onset + dur + gap
    script, frames = sim.build_stream(episodes, T, peak_conf=0.95, ramp=ramp)

    core = sim.run_core(script, frames)
    raw = sim.raw_fires(script, frames)
    clat, rlat = _first_latencies(core, episodes), _first_latencies(raw, episodes)
    cl = [x for x in clat if x is not None]
    rl = [x for x in rlat if x is not None]
    cmed = median(cl) if cl else float("nan")
    rmed = median(rl) if rl else float("nan")
    cg, rg = _guess_rate(core, script, thr), _guess_rate(raw, script, thr)

    cells = {
        ("dense-modeled", "naive_latency"): rmed,
        ("core-replay", "latency"): cmed,
        ("dense-modeled", "naive_p-pauc"): _pauc(rlat, window),
        ("core-replay", "p-pauc"): _pauc(clat, window),
        ("dense-modeled", "naive_guess%"): rg,
        ("core-replay", "guess%"): cg,
    }
    headline = {
        "timespeak reaction-latency naive/core (frames)": f"{rmed:.1f} / {cmed:.1f}",
        "timespeak guess% naive/core": f"{rg:.0f}% / {cg:.0f}%  (core refuses to guess)",
    }
    return TrackResult(
        track=TRACK, cells=cells, fmts=_FMTS, headline=headline,
        measured=["core-replay:latency", "core-replay:p-pauc", "core-replay:guess%"],
        modeled=["dense-modeled:naive_latency", "dense-modeled:naive_p-pauc", "dense-modeled:naive_guess%"],
        reference={"latency": {"value": 0.0, "by": "ideal: fire at onset"}},
        calls=0,
        diagnostic=True,
        excluded=["timespeak: deterministic sim regression, not held-out product E2E"],
        notes=(f"sim: {n} episodes x{dur}f, conf-ramp {ramp}f, window {window}f, gate_thr={thr}; "
               f"core waits for confidence (refuse-to-guess), raw fires on the first true frame."),
    )
