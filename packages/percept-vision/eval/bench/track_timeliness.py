"""TIMELINESS track — how promptly + correctly does a per-frame watcher fire on an action onset?

On Perception Test ACTION localisation, each event is a should-fire onset (an action begins). A
per-frame Gemini watcher judges "the action of <label> is happening" across a window bracketing the
onset, and the FIRST confident-yes frame is its reaction. We adapt ProactiveVideoQA's PAUC into a
P-PAUC turn-score: full credit if the watcher fires within omega of the onset, decaying credit out
to 2*omega, then zero — but a SILENT watcher (never fires) gets a no-response FLOOR (25) rather than
the harsh PVQA zero, because no-response is safer than a confidently wrong fire. We report both:
  * p_pauc       — P-PAUC with the silent floor (no-fire = 25, ASK-style credit for staying quiet)
  * pvqa_native  — the SAME score with PVQA's native silent floor of 0 (no-fire = 0)
  * reaction_p50/p90 — latency lambda = t_hat - onset over the events that actually fired.

This measures a legacy per-frame watcher only. It is a dense diagnostic, not product E2E.
"""

from __future__ import annotations

from .types import TrackResult

TRACK = "timeliness"

FPS = 4.0
PRE_S = 2.0  # window starts onset-2.0 (clamped at 0)
POST_S = 3.0  # window ends onset+3.0
OMEGA = 1.0  # full-credit tolerance (s)
SEM = 6  # concurrent judge calls
FLOOR = 25.0  # silent (no-fire) floor for P-PAUC — no-response safer than a wrong fire


def _turn_score(t_hat, onset, *, floor):
    """ProactiveVideoQA-style PAUC turn-score for a should-fire onset (0..100)."""
    if t_hat is None:
        return floor
    d = abs(t_hat - onset)
    if d <= OMEGA:
        return 100.0
    if d <= 2 * OMEGA:
        return 100.0 - 75.0 * (d - OMEGA) / OMEGA
    return 0.0


def _result(p_pauc, pvqa_native, p50, p90, *, n_fired, total, calls, notes, modeled=None):
    """Assemble the TrackResult — dense-modeled/diagnostic cells only."""
    cells = {
        ("dense-modeled", "p_pauc"): p_pauc,
        ("dense-modeled", "pvqa_native"): pvqa_native,
        ("dense-modeled", "reaction_p50"): p50,
        ("dense-modeled", "reaction_p90"): p90,
    }
    p50s = "n/a" if p50 is None else f"{p50:.1f}"
    p90s = "n/a" if p90 is None else f"{p90:.1f}"
    headline = {
        f"timeliness P-PAUC / PVQA-native (N PT actions)":
            f"{p_pauc:.1f} / {pvqa_native:.1f} (fired {n_fired}/{total})",
        "timeliness reaction λ p50/p90 (s)": f"{p50s} / {p90s}",
    }
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "p_pauc": "{:.1f}",
            "pvqa_native": "{:.1f}",
            "reaction_p50": "{:.1f}",
            "reaction_p90": "{:.1f}",
        },
        headline=headline,
        measured=[] if modeled else ["dense-modeled:p_pauc", "dense-modeled:pvqa_native", "dense-modeled:reaction_p50", "dense-modeled:reaction_p90"],
        modeled=modeled or [],
        calls=calls,
        notes=notes,
        diagnostic=True,
        excluded=["timeliness: legacy dense watcher diagnostic, not product E2E"],
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        # Offline card path: use the golden-v1 PT sample count when annotations are staged, but keep
        # the VLM-dependent timeliness cells modeled until --live measures them.
        from eval.bench import perception_test as pt

        try:
            total = len(pt.golden_action_events(require_media=False)) or 5
        except Exception:
            total = 5
        return _result(
            62.0,
            48.0,
            0.8,
            1.9,
            n_fired=min(4, total),
            total=total,
            calls=0,
            notes="offline modeled; run --live to measure golden-v1 Perception Test samples",
            modeled=["dense-modeled:p_pauc", "dense-modeled:pvqa_native", "dense-modeled:reaction_p50", "dense-modeled:reaction_p90"],
        )

    import asyncio
    import statistics

    import cv2

    from eval.bench import perception_test as pt
    from eval.count_video import decode
    from percept import registry
    from percept.config import load_settings
    from percept.gate import classify

    settings = load_settings()
    thr = settings.gate.threshold
    vision = registry.make_vision("gemini", settings)

    events = pt.sample_events("action", n)

    def _frame(bgr):
        _, buf = cv2.imencode(".jpg", bgr)
        return ("image/jpeg", buf.tobytes())

    async def _watch(ev):
        """-> (t_hat, n_frames_judged) for one event's watch window."""
        _, full = decode(str(pt.fetch_video(ev["video_id"])), FPS)
        onset = ev["onset_s"]
        lo, hi = max(0.0, onset - PRE_S), onset + POST_S
        cond = f"the action of {ev['label'].lower()} is happening"
        window = sorted((t, bgr) for t, bgr in full if lo <= t <= hi)
        sem = asyncio.Semaphore(SEM)

        async def _judge(t, bgr):
            async with sem:
                v = await vision.judge(cond, _frame(bgr))
                return t, classify(v, 1.0, thr, 0.5) == "yes"

        verdicts = await asyncio.gather(*(_judge(t, bgr) for t, bgr in window))
        t_hat = next((t for t, yes in verdicts if yes), None)  # first ascending confident-yes
        return t_hat, len(window)

    async def _go():
        return await asyncio.gather(*(_watch(ev) for ev in events))

    watched = asyncio.run(_go())  # [(t_hat, n_frames), ...] aligned to events

    calls = sum(nf for _t, nf in watched)
    turn = [_turn_score(t_hat, ev["onset_s"], floor=FLOOR) for ev, (t_hat, _nf) in zip(events, watched)]
    native = [_turn_score(t_hat, ev["onset_s"], floor=0.0) for ev, (t_hat, _nf) in zip(events, watched)]
    lambdas = [t_hat - ev["onset_s"] for ev, (t_hat, _nf) in zip(events, watched) if t_hat is not None]

    p_pauc = statistics.mean(turn) if turn else 0.0
    pvqa_native = statistics.mean(native) if native else 0.0
    if lambdas:
        s = sorted(lambdas)
        p50 = statistics.quantiles(s, n=100, method="inclusive")[49] if len(s) > 1 else s[0]
        p90 = statistics.quantiles(s, n=100, method="inclusive")[89] if len(s) > 1 else s[0]
    else:
        p50 = p90 = None

    n_fired = len(lambdas)
    notes = f"{pt.manifest_hash(events)}; fired {n_fired}/{len(events)}"
    return _result(p_pauc, pvqa_native, p50, p90, n_fired=n_fired, total=len(events), calls=calls, notes=notes)
