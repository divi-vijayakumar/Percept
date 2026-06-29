"""TimeSpeak — derived timeliness of the proactive response.

When saved E2E benchmark artifacts exist, this track derives TimeSpeak from those runs without another
video pass: how soon after the labelled event/answer window opened did Percept emit a cue/answer? If no
E2E artifacts exist, it falls back to the deterministic sim-core regression.

  latency  : frames from onset to the first fire of the episode (lower = more timely)
  P-PAUC   : proactive AUC — 1 - latency/window if fired in the window, else 0; mean ×100
  guess%   : fraction of fires made while confidence < the gate threshold (firing on an ambiguous read)

The derived-from-artifacts path is DRAFT-only until the source E2E runs are full and manifest-backed.
"""
from __future__ import annotations

from statistics import median

from . import sim
from .derived_interaction import cues_from_latest_e2e, summarize_sources, time_metrics
from .types import TrackResult

TRACK = "timespeak"
_FMTS = {
    "latency": "{:.1f}",
    "p-pauc": "{:.0f}",
    "guess%": "{:.0f}",
    "timing_accuracy": "{:.2f}",
    "early_fire_rate": "{:.2f}",
    "late_fire_rate": "{:.2f}",
    "naive_latency": "{:.1f}",
    "naive_p-pauc": "{:.0f}",
    "naive_guess%": "{:.0f}",
}
_MEASURED = ["latency", "p-pauc", "guess%"]


def _timing_cues(cues):
    """Keep only cues that carry a real timing window for the lateness/latency derivation.

    Some upstream artifacts (notably proactiveqa-e2e) do not record a per-turn ground-truth
    window, so ``cues_from_latest_e2e`` emits them with a degenerate placeholder window of
    ``(0.0, 0.0)`` (see ``derived_interaction.cues_from_latest_e2e``). Such a window opens and
    closes at t=0, so EVERY emitted answer (e.g. an answer at t=60s) is mechanically scored as
    "late by 60s" even though there is no ground truth onset to be late against. That floods
    ``late_fire_rate`` with a meaningless signal (N=1 PVQA turn -> 11 identical placeholder cues
    -> 0.58 late-fire on its own) and is not a real timeliness measurement.

    We drop only TRULY degenerate windows (``lo == hi``). Cues that have a genuine window keep
    their lateness/latency contribution unchanged, so this never trades precision for latency:
    it removes rows that have no timing ground truth at all.
    """
    return [c for c in cues if c.window[1] > c.window[0]]


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
    omega_s = 2.0
    cues, refs, calls = cues_from_latest_e2e()
    if cues:
        timing_cues = _timing_cues(cues)
        dropped = len(cues) - len(timing_cues)
        m = time_metrics(timing_cues, omega_s=omega_s)
        cells = {
            ("e2e", "timing_accuracy"): m["timing_accuracy"],
            ("e2e", "p-pauc"): m["p_pauc"],
            ("e2e", "latency"): m["latency_p50"],
            ("e2e", "early_fire_rate"): m["early_fire_rate"],
            ("e2e", "late_fire_rate"): m["late_fire_rate"],
        }
        timing = m["timing_accuracy"]
        pauc = m["p_pauc"]
        latency = m["latency_p50"]
        return TrackResult(
            track=TRACK,
            cells=cells,
            fmts=_FMTS,
            headline={
                "timespeak derived timing accuracy / P-PAUC": (
                    f"{timing:.2f} / {pauc:.0f}" if timing is not None and pauc is not None else "n/a"
                ),
                "timespeak derived latency p50": f"{latency:.1f}s" if latency is not None else "n/a",
            },
            measured=[
                "e2e:timing_accuracy",
                "e2e:p-pauc",
                "e2e:latency",
                "e2e:early_fire_rate",
                "e2e:late_fire_rate",
            ],
            calls=0,
            coverage=[{
                "dataset": "local-e2e-artifacts",
                "expected": len(cues),
                "observed": len(timing_cues),
                "failed": 0,
                "skipped": dropped,
                "notes": (
                    f"sources={summarize_sources(cues)}; artifacts={refs}; "
                    f"timing_skipped={dropped} (placeholder/zero-width windows w/o timing ground truth)"
                ),
            }],
            diagnostic=True,
            excluded=["timespeak: derived from local sampled E2E artifacts, not a full release split"],
            notes=(
                f"derived from latest E2E result artifacts; omega_s={omega_s}; "
                f"source_counts={summarize_sources(cues)}; source_calls={calls}; artifacts={refs}; "
                f"timing_cues={len(timing_cues)}/{len(cues)} "
                f"(dropped {dropped} placeholder/zero-width windows w/o timing ground truth)"
            ),
        )

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
