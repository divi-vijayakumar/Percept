"""CueSpeak — sim-core proactive interjection quality.

Does the agent speak ONCE on the rising edge of a cue (precision), catch every cue (recall), and NOT
nag (fires-per-episode)? Contrasts the cognition layer (core = rising-edge three-state gate) against a
naive scripted emitter (raw = fire on every true scripted frame). Deterministic E2E sim — the
gate/executor are real and no VLM is called.

This is a cognition regression track, not a publishable TML-comparable CueSpeak result. The release
track needs a held-out prompt corpus, cue windows, expected utterance semantics, a judge, and an
edge-E2E adapter.
"""
from __future__ import annotations

from . import sim
from .types import TrackResult

TRACK = "cuespeak"
_FMTS = {"precision": "{:.2f}", "recall": "{:.2f}", "fires/ep": "{:.2f}"}
_MEASURED = ["precision", "recall", "fires/ep"]


def _score(fires, episodes):
    """Match fire times to episodes -> (precision, recall, fires_per_episode). First fire of a real
    episode = TP; a re-fire within the same episode or a fire with no cue = FP (spam / false fire)."""
    fire_ts = [t for (a, t) in fires if a == "fire"]
    hit: set[int] = set()
    tp = 0
    for t in fire_ts:
        idx = next((i for i, e in enumerate(episodes) if e.onset <= t < e.onset + e.duration), None)
        if idx is not None and idx not in hit:
            hit.add(idx)
            tp += 1
    precision = tp / len(fire_ts) if fire_ts else 0.0
    recall = len(hit) / len(episodes) if episodes else 0.0
    fires_per_ep = len(fire_ts) / len(episodes) if episodes else 0.0
    return precision, recall, fires_per_ep


def score(*, live: bool = False, n: int = 8) -> TrackResult:
    dur, gap = 5, 7
    episodes = [sim.Episode(onset=10 + i * (dur + gap), duration=dur) for i in range(n)]
    T = episodes[-1].onset + dur + gap
    # spurious confident cue-frames between episodes -> tests false-fire robustness
    noise = [(episodes[0].onset + dur + 2, 0.9), (episodes[min(2, n - 1)].onset + dur + 3, 0.92)]
    script, frames = sim.build_stream(episodes, T, peak_conf=0.95, ramp=0, noise_spikes=noise)

    cp, cr, cf = _score(sim.run_core(script, frames), episodes)
    rp, rr, rf = _score(sim.raw_fires(script, frames), episodes)

    cells = {
        ("raw", "precision"): rp, ("core", "precision"): cp, ("e2e", "precision"): cp,
        ("raw", "recall"): rr, ("core", "recall"): cr, ("e2e", "recall"): cr,
        ("raw", "fires/ep"): rf, ("core", "fires/ep"): cf, ("e2e", "fires/ep"): cf,
    }
    headline = {
        "cuespeak precision naive/core": f"{rp:.2f} / {cp:.2f}",
        "cuespeak fires-per-episode naive/core": f"{rf:.1f} / {cf:.1f}  (1.0 = no nag)",
    }
    return TrackResult(
        track=TRACK, cells=cells, fmts=_FMTS, headline=headline,
        measured=_MEASURED, modeled=[],
        reference={"fires/ep": {"value": 1.0, "by": "ideal: 1 interjection/cue"}},
        calls=0,
        notes=(f"sim: {n} episodes x{dur}f + {len(noise)} noise spikes; "
               f"raw=scripted naive emitter (every true frame), core=rising-edge gate. e2e=core (edge is parity-equal)."),
    )
