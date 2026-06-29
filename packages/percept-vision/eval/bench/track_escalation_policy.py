"""Escalation-policy ablation: recall/precision/cost/latency before spending VLM calls.

This track measures the conductor, not the oracle. A VLM fallback only helps if the cheap edge wakes
near the true event; it only stays cheap if it avoids waking everywhere else. We therefore compare:

* motion   = current motion gate at the configured threshold.
* weighted = weighted cheap-evidence policy (the current ``WakePolicy`` path).
* fallback = selective fallback: lower candidate threshold + a fixed cadence floor.

The precision here is an escalation precision proxy: "would this VLM call land inside the annotated
event window?" The final alert precision can be higher after Gemini filters false candidates, but the
cost and latency are set by this conductor layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from percept.adapters import WatchCondition, WatchSpec
from percept.harness.detectors import WakePolicy
from percept.harness.tier0 import MotionGate

from . import perception_test as pt
from .track_edge_recall import _event_window, _frame_series
from .types import TrackResult

TRACK = "escalation"
POLICY_CHANGE_WEIGHT = 2.0
FALLBACK_CANDIDATE_FACTOR = 0.5
FALLBACK_CANDIDATE_CADENCE_S = 1.0
FALLBACK_FIXED_CADENCE_S = 3.0


@dataclass(frozen=True)
class Policy:
    name: str
    threshold: float
    wake_policy: WakePolicy | None = None
    candidate_cadence_s: float | None = None
    fixed_cadence_s: float | None = None


@dataclass
class PolicyStats:
    events: int = 0
    frames: int = 0
    duration_s: float = 0.0
    hits: int = 0
    tp_calls: int = 0
    fp_calls: int = 0
    latencies_s: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latencies_s is None:
            self.latencies_s = []

    @property
    def calls(self) -> int:
        return self.tp_calls + self.fp_calls

    @property
    def recall(self) -> float | None:
        return self.hits / self.events if self.events else None

    @property
    def precision_proxy(self) -> float | None:
        return self.tp_calls / self.calls if self.calls else None

    @property
    def calls_per_min(self) -> float | None:
        return self.calls / (self.duration_s / 60.0) if self.duration_s > 0 else None

    @property
    def reduction(self) -> float | None:
        return 1.0 - (self.calls / self.frames) if self.frames else None

    @property
    def median_latency_s(self) -> float | None:
        return median(self.latencies_s) if self.latencies_s else None


def _spec_for(policy: Policy) -> WatchSpec:
    conditions = [
        WatchCondition(
            goal_id="candidate",
            anchor_text="the target event may be starting",
            threshold=policy.threshold,
            cadence_s=policy.candidate_cadence_s,
            gate_on="motion",
        )
    ]
    if policy.fixed_cadence_s is not None:
        conditions.append(
            WatchCondition(
                goal_id="cadence-floor",
                anchor_text="periodic fallback look",
                threshold=policy.threshold,
                cadence_s=policy.fixed_cadence_s,
                gate_on="fixed",
            )
        )
    return WatchSpec(conditions=conditions)


def _duration_s(series: list[tuple[float, object]]) -> float:
    if not series:
        return 0.0
    if len(series) == 1:
        return series[0][0]
    return series[-1][0] + max(series[-1][0] - series[-2][0], 0.0)


def _score_policies(events: list[dict], policies: dict[str, Policy]) -> dict[str, PolicyStats]:
    stats = {name: PolicyStats(events=len(events)) for name in policies}
    for event in events:
        lo, hi = _event_window(event)
        gates = {
            name: MotionGate(_spec_for(policy), wake_policy=policy.wake_policy)
            for name, policy in policies.items()
        }
        series = _frame_series(event["path"])
        duration_s = _duration_s(series)
        hits = {name: False for name in policies}
        first_latencies: dict[str, float | None] = {name: None for name in policies}
        for s in stats.values():
            s.frames += len(series)
            s.duration_s += duration_s
        for t, frame in series:
            for name, gate in gates.items():
                if not gate.evaluate(frame, ts=t).escalate:
                    continue
                if lo <= t <= hi:
                    stats[name].tp_calls += 1
                    hits[name] = True
                    if first_latencies[name] is None:
                        first_latencies[name] = max(0.0, t - lo)
                else:
                    stats[name].fp_calls += 1
        for name in policies:
            if hits[name]:
                stats[name].hits += 1
                if first_latencies[name] is not None:
                    stats[name].latencies_s.append(first_latencies[name])
    return stats


def _events(*, live: bool, n: int) -> list[dict]:
    if not live:
        return pt.golden_action_events(require_media=True)[:n]
    events = pt.sample_events("action", n)
    for event in events:
        event["path"] = pt.fetch_video(event["video_id"])
    return events


def _result(stats: dict[str, PolicyStats], *, thr: float, n_events: int, notes: str) -> TrackResult:
    cells = {}
    measured = []
    for name in ("motion", "weighted", "fallback"):
        s = stats[name]
        values = {
            f"{name}_recall": s.recall,
            f"{name}_precision_proxy": s.precision_proxy,
            f"{name}_calls_per_min": s.calls_per_min,
            f"{name}_cost_reduction": s.reduction,
            f"{name}_median_latency_s": s.median_latency_s,
        }
        for metric, value in values.items():
            cells[("edge", metric)] = value
            if value is not None:
                measured.append(f"edge:{metric}")

    fb = stats["fallback"]
    def _fmt(v: float | None, spec: str = ".2f") -> str:
        return "n/a" if v is None else format(v, spec)

    latency_s = "n/a" if fb.median_latency_s is None else f"{fb.median_latency_s:.2f}s"
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "motion_recall": "{:.2f}",
            "weighted_recall": "{:.2f}",
            "fallback_recall": "{:.2f}",
            "motion_precision_proxy": "{:.2f}",
            "weighted_precision_proxy": "{:.2f}",
            "fallback_precision_proxy": "{:.2f}",
            "motion_calls_per_min": "{:.1f}",
            "weighted_calls_per_min": "{:.1f}",
            "fallback_calls_per_min": "{:.1f}",
            "motion_cost_reduction": "{:.2f}",
            "weighted_cost_reduction": "{:.2f}",
            "fallback_cost_reduction": "{:.2f}",
            "motion_median_latency_s": "{:.2f}",
            "weighted_median_latency_s": "{:.2f}",
            "fallback_median_latency_s": "{:.2f}",
        },
        headline={
            "escalation recall motion/weighted/fallback":
                f"{_fmt(stats['motion'].recall)} / {_fmt(stats['weighted'].recall)} / {_fmt(fb.recall)}",
            "fallback cost and latency":
                f"{_fmt(fb.calls_per_min, '.1f')} calls/min, {_fmt(fb.reduction)} reduction, "
                f"median {latency_s}",
        },
        measured=measured,
        coverage=[{
            "dataset": "perception-test/action",
            "expected": n_events,
            "observed": n_events,
            "failed": 0,
            "skipped": 0,
            "notes": "edge-only policy ablation; precision is candidate-window proxy",
        }],
        notes=(
            f"{notes}; N={n_events}; motion_thr={thr:.2f}; weighted change x{POLICY_CHANGE_WEIGHT:g}; "
            f"fallback threshold={thr * FALLBACK_CANDIDATE_FACTOR:.2f}, "
            f"candidate cadence={FALLBACK_CANDIDATE_CADENCE_S}s, fixed cadence={FALLBACK_FIXED_CADENCE_S}s. "
            "precision_proxy is escalation precision before VLM filtering."
        ),
    )


def _unavailable(exc: Exception | None = None) -> TrackResult:
    suffix = f": {type(exc).__name__}: {exc}" if exc is not None else ""
    return TrackResult(
        track=TRACK,
        cells={
            ("edge", "motion_recall"): None,
            ("edge", "weighted_recall"): None,
            ("edge", "fallback_recall"): None,
        },
        headline={"escalation policy ablation": "n/a (local golden media unavailable)"},
        excluded=["escalation: local golden media unavailable"],
        notes=f"offline measurement unavailable{suffix}",
    )


def score(*, live: bool, n: int) -> TrackResult:
    from percept.config import load_settings

    thr = load_settings().edge.motion_gate
    try:
        events = _events(live=live, n=n)
        if not events:
            return _unavailable()
        policies = {
            "motion": Policy("motion", threshold=thr),
            "weighted": Policy(
                "weighted",
                threshold=thr,
                wake_policy=WakePolicy(weights={"change": POLICY_CHANGE_WEIGHT}),
            ),
            "fallback": Policy(
                "fallback",
                threshold=thr * FALLBACK_CANDIDATE_FACTOR,
                candidate_cadence_s=FALLBACK_CANDIDATE_CADENCE_S,
                fixed_cadence_s=FALLBACK_FIXED_CADENCE_S,
            ),
        }
        stats = _score_policies(events, policies)
        notes = "golden-v1 Perception Test action sample" if not live else f"manifest={pt.manifest_hash(events)}"
        return _result(stats, thr=thr, n_events=len(events), notes=notes)
    except Exception as exc:
        return _unavailable(exc)
