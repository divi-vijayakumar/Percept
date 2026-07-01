"""CueSpeak — derived proactive interjection quality.

When saved E2E benchmark artifacts exist, this track derives CueSpeak from those runs without a new
video pass or model call: did the E2E system emit a cue/answer inside the labelled event window, and did
it avoid duplicate speech? If no E2E artifacts are available, it falls back to the deterministic sim-core
regression.

The derived-from-artifacts path is still DRAFT-only until the source E2E runs are full, manifest-backed,
and have independent semantic judging where needed.
"""
from __future__ import annotations

from . import sim
from .derived_interaction import cue_metrics, cues_from_latest_e2e, summarize_sources
from .types import TrackResult

TRACK = "cuespeak"
_FMTS = {
    "precision": "{:.2f}",
    "recall": "{:.2f}",
    "fires/ep": "{:.2f}",
    "duplicate_rate": "{:.2f}",
    "unresolved_rate": "{:.2f}",
    "naive_precision": "{:.2f}",
    "naive_recall": "{:.2f}",
    "naive_fires/ep": "{:.2f}",
}
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
    cues, refs, calls = cues_from_latest_e2e()
    if cues:
        m = cue_metrics(cues)
        cells = {
            ("e2e", "precision"): m["precision"],
            ("e2e", "recall"): m["recall"],
            ("e2e", "fires/ep"): m["fires_per_episode"],
            ("e2e", "duplicate_rate"): m["duplicate_rate"],
            ("e2e", "unresolved_rate"): m["unresolved_rate"],
        }
        precision = m["precision"]
        recall = m["recall"]
        fires_ep = m["fires_per_episode"]
        return TrackResult(
            track=TRACK,
            cells=cells,
            fmts=_FMTS,
            headline={
                "cuespeak derived precision/recall": (
                    f"{precision:.2f} / {recall:.2f}" if precision is not None and recall is not None else "n/a"
                ),
                "cuespeak derived fires-per-episode": f"{fires_ep:.2f}" if fires_ep is not None else "n/a",
            },
            measured=[
                "e2e:precision",
                "e2e:recall",
                "e2e:fires/ep",
                "e2e:duplicate_rate",
                "e2e:unresolved_rate",
            ],
            calls=0,
            coverage=[{
                "dataset": "local-e2e-artifacts",
                "expected": len(cues),
                "observed": len(cues),
                "failed": 0,
                "skipped": 0,
                "notes": f"sources={summarize_sources(cues)}; artifacts={refs}",
            }],
            diagnostic=True,
            excluded=["cuespeak: derived from local sampled E2E artifacts, not a full release split"],
            notes=(
                f"derived from latest E2E result artifacts; source_counts={summarize_sources(cues)}; "
                f"source_calls={calls}; artifacts={refs}"
            ),
        )

    dur, gap = 5, 7
    episodes = [sim.Episode(onset=10 + i * (dur + gap), duration=dur) for i in range(n)]
    T = episodes[-1].onset + dur + gap
    # spurious confident cue-frames between episodes -> tests false-fire robustness
    noise = [(episodes[0].onset + dur + 2, 0.9), (episodes[min(2, n - 1)].onset + dur + 3, 0.92)]
    script, frames = sim.build_stream(episodes, T, peak_conf=0.95, ramp=0, noise_spikes=noise)

    cp, cr, cf = _score(sim.run_core(script, frames), episodes)
    rp, rr, rf = _score(sim.raw_fires(script, frames), episodes)

    cells = {
        ("dense-modeled", "naive_precision"): rp,
        ("core-replay", "precision"): cp,
        ("dense-modeled", "naive_recall"): rr,
        ("core-replay", "recall"): cr,
        ("dense-modeled", "naive_fires/ep"): rf,
        ("core-replay", "fires/ep"): cf,
    }
    headline = {
        "cuespeak precision naive/core": f"{rp:.2f} / {cp:.2f}",
        "cuespeak fires-per-episode naive/core": f"{rf:.1f} / {cf:.1f}  (1.0 = no nag)",
    }
    return TrackResult(
        track=TRACK, cells=cells, fmts=_FMTS, headline=headline,
        measured=["core-replay:precision", "core-replay:recall", "core-replay:fires/ep"],
        modeled=["dense-modeled:naive_precision", "dense-modeled:naive_recall", "dense-modeled:naive_fires/ep"],
        reference={"fires/ep": {"value": 1.0, "by": "ideal: 1 interjection/cue"}},
        calls=0,
        diagnostic=True,
        excluded=["cuespeak: deterministic sim regression, not held-out product E2E"],
        notes=(f"sim: {n} episodes x{dur}f + {len(noise)} noise spikes; "
               f"naive=scripted emitter (every true frame), core-replay=rising-edge gate."),
    )
