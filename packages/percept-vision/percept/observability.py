"""Structured JSON logging + a tiny metrics registry + the standard telemetry funnel + /health.

Stdlib only (the core stays zero-dep). The harness/assistant can route these wherever; the core
just emits.

The **standard funnel** is the one shape every deployment reports a watch loop along::

    seen  ->  gated  ->  judged  ->  refused  ->  fired

- ``seen``     every frame the loop observed (the ungated baseline).
- ``gated``    frames the tier-0 gate escalated (spent a cloud call) — ``seen >= gated``.
- ``judged``   judge invocations actually made — ``gated >= judged`` (one per active watch goal·escalation).
- ``refused``  verdicts the three-state gate abstained on (ASK — refused to guess).
- ``fired``    rising-edge fires the gate emitted (the proactive output).

Counters are scoped per ``concern`` (goal id) and ``camera`` (source) so a dashboard can slice the
funnel by either. :class:`Funnel` builds the dotted counter names on top of the existing
:class:`Metrics` (no new storage). Judge **latency** and **token/cost** ride alongside as summable
counters (call count + total) so a mean is one division.

**OpenTelemetry** export is an OPTIONAL extra (``percept-vision[otel]``). When ``opentelemetry`` is
installed, :func:`otel_exporter` mirrors the funnel + latency to OTel counters/histograms in a
standard shape; when it is absent the exporter degrades to a no-op and the in-process
:class:`Metrics` remains the source of truth. The core never imports ``opentelemetry`` at module top
level (invariant 7).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TraceEvent:
    """One step in a per-frame decision trace.

    Counters answer *aggregate* questions ("are we firing?"); a trace answers the *specific* one:
    "why did THIS frame + goal fire / ask / stay silent?". ``trace_id`` correlates the steps of one
    frame; ``stage`` is one of the emitted stages, in the order a firing frame produces them:
    ``edge.decision`` (from the runtime worker's tier-0 :class:`~percept.edge.tier0.MotionGate` —
    ``motion``/``structural``/``semantic``/``threshold``/``escalate``, emitted for EVERY frame so a
    silent run because the edge never escalated is distinguishable from one that judged and stayed
    quiet), ``source.frame``, ``judge.started``, ``judge.finished`` (``satisfied``/``confidence``),
    ``gate.classified`` (from ``perceive_judged`` — ``truth``/``action``/``reason``, a human-readable
    WHY such as ``"silent: already_fired_or_not_rising"``) and ``incident.created`` (from the runtime
    worker). ``data`` carries the decision inputs/outputs for that stage. This is production telemetry
    for the same edge/verdict/gate/reason shape the forensic bench trace used
    (``eval/bench/track_charades_e2e.py``), now on the real runtime firing path. Emitted only when an
    ``on_trace`` callback is wired — default off, so the hot path is unchanged.
    """

    trace_id: str
    stage: str
    t: float | None = None
    goal_id: str | None = None
    data: dict = field(default_factory=dict)


# An ``on_trace`` callback: receives each :class:`TraceEvent`. Wire one on ``Percept`` / ``run(...)``.
Tracer = Callable[[TraceEvent], None]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = "percept") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Emit a structured line: ``log(lg, "fired", goal_id=g, conf=0.9)``."""
    logger.info(msg, extra={"fields": fields})


class Metrics:
    """A minimal in-process counter registry."""

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()

    def incr(self, name: str, n: int = 1) -> None:
        self._counters[name] += n

    def add(self, name: str, amount: float) -> None:
        """Accumulate a float total (latency seconds, token counts, cost)."""
        self._counters[name] += amount  # type: ignore[assignment]

    def get(self, name: str) -> int:
        return self._counters[name]

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)


# ---------------------------------------------------------------------------
# The standard funnel — seen -> gated -> judged -> refused -> fired
# ---------------------------------------------------------------------------

# The five funnel stages, in order. Public so a dashboard can iterate them.
FUNNEL_STAGES = ("seen", "gated", "judged", "refused", "fired")

# Counter-name prefixes (kept here so the harness/agent never hard-code the strings).
_FUNNEL = "funnel"
_LATENCY = "judge.latency"  # judge wall-time, seconds (count + total)
_TOKENS_IN = "judge.tokens.input"
_TOKENS_OUT = "judge.tokens.output"
_COST = "judge.cost.usd"


def funnel_key(
    stage: str, *, concern: str | None = None, camera: str | None = None, reason: str | None = None
) -> str:
    """The dotted counter name for one funnel stage, optionally scoped.

    ``funnel.<stage>`` (the run total), ``funnel.<stage>.concern.<goal_id>`` (per concern),
    ``funnel.<stage>.camera.<source>`` (per camera), and ``funnel.<stage>.reason.<reason>`` (per
    drop/refusal reason, used by :meth:`Funnel.dropped`) are all distinct keys, so a snapshot
    carries the aggregate *and* every slice without extra storage.
    """
    if concern is not None:
        return f"{_FUNNEL}.{stage}.concern.{concern}"
    if camera is not None:
        return f"{_FUNNEL}.{stage}.camera.{camera}"
    if reason is not None:
        return f"{_FUNNEL}.{stage}.reason.{reason}"
    return f"{_FUNNEL}.{stage}"


class Funnel:
    """Emit the standard ``seen→gated→judged→refused→fired`` counters into a :class:`Metrics`.

    Each stage method bumps three keys — the run total, the per-concern slice, and the per-camera
    slice — so the funnel is sliceable by either dimension with one ``incr`` call site. An optional
    :class:`OTelExporter` (or any object with a matching ``record_stage``/``record_judge`` method) is
    mirrored on every call; absent one, the in-process :class:`Metrics` is the only sink.
    """

    def __init__(self, metrics: Metrics, *, otel: "OTelExporter | None" = None) -> None:
        self._m = metrics
        self._otel = otel

    def _stage(self, stage: str, *, concern: str | None, camera: str | None, n: int) -> None:
        self._m.incr(funnel_key(stage), n)
        if concern is not None:
            self._m.incr(funnel_key(stage, concern=concern), n)
        if camera is not None:
            self._m.incr(funnel_key(stage, camera=camera), n)
        if self._otel is not None:
            self._otel.record_stage(stage, concern=concern, camera=camera, n=n)

    def seen(self, *, camera: str | None = None, n: int = 1) -> None:
        """A frame was observed (no concern scope — a frame precedes any goal)."""
        self._stage("seen", concern=None, camera=camera, n=n)

    def gated(self, *, camera: str | None = None, n: int = 1) -> None:
        """The gate escalated this frame (spent a cloud call)."""
        self._stage("gated", concern=None, camera=camera, n=n)

    def judged(self, *, concern: str | None = None, camera: str | None = None, n: int = 1) -> None:
        """A judge call was made for ``concern`` over ``camera``."""
        self._stage("judged", concern=concern, camera=camera, n=n)

    def refused(self, *, concern: str | None = None, camera: str | None = None, n: int = 1) -> None:
        """The gate abstained (ASK — refused to guess) for ``concern``."""
        self._stage("refused", concern=concern, camera=camera, n=n)

    def fired(self, *, concern: str | None = None, camera: str | None = None, n: int = 1) -> None:
        """A rising-edge fire was emitted for ``concern``."""
        self._stage("fired", concern=concern, camera=camera, n=n)

    def dropped(self, *, reason: str, camera: str | None = None, n: int = 1) -> None:
        """A tick was DROPPED before a verdict was ever reached — e.g. the off-loop judge scheduler
        (:class:`~percept.runtime.worker._JudgeScheduler`) superseding a queued candidate under
        overload, or discarding one pinned to a goals revision that moved mid-flight. Distinct from
        :meth:`refused` (a completed judge call the gate abstained ASK on): a drop never reached the
        judge at all. NOT one of the five :data:`FUNNEL_STAGES` (a dashboard iterating that tuple
        won't double-count it) but rides the same ``funnel.*`` counter namespace + OTel mirror, and —
        per ADR-001 §9 — always carries a ``reason`` (never a silent counter bump).
        """
        self._m.incr(funnel_key("dropped"), n)
        if camera is not None:
            self._m.incr(funnel_key("dropped", camera=camera), n)
        self._m.incr(funnel_key("dropped", reason=reason), n)
        if self._otel is not None:
            self._otel.record_stage("dropped", concern=None, camera=camera, n=n, reason=reason)

    def record_judge(
        self,
        *,
        latency_s: float | None = None,
        input_tokens: float | None = None,
        output_tokens: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record one judge call's wall-time and (if the backend reported it) token/cost.

        Latency is always available (the agent times the call). Token/cost are hooks: passed through
        when the backend attaches them to its verdict, left untouched otherwise. Each metric stores a
        ``.count`` and a ``.total`` so a mean is ``total / count``.
        """
        if latency_s is not None:
            self._m.incr(f"{_LATENCY}.count")
            self._m.add(f"{_LATENCY}.total_s", float(latency_s))
        if input_tokens is not None:
            self._m.add(_TOKENS_IN, float(input_tokens))
        if output_tokens is not None:
            self._m.add(_TOKENS_OUT, float(output_tokens))
        if cost_usd is not None:
            self._m.add(_COST, float(cost_usd))
        if self._otel is not None:
            self._otel.record_judge(
                latency_s=latency_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )


def judge_usage(verdict: Any) -> dict[str, float]:
    """Pull optional token/cost off a verdict the backend may have annotated.

    The firing-contract :class:`~percept.adapters.Verdict` carries only ``satisfied/confidence/
    evidence``; a backend that knows its usage can attach a ``usage`` dict (or ``input_tokens`` /
    ``output_tokens`` / ``cost_usd`` attributes) WITHOUT changing the frozen contract. This reads
    whatever is present and returns a kwargs dict for :meth:`Funnel.record_judge`; an un-annotated
    verdict yields ``{}`` (the hook is simply unused).
    """
    out: dict[str, float] = {}
    usage = getattr(verdict, "usage", None)
    # an EMPTY usage dict is "not annotated" (the 1.3 Verdict default), so loose attributes still read
    src = usage if isinstance(usage, dict) and usage else verdict
    for key, attr in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cost_usd", "cost_usd"),
    ):
        val = src.get(attr) if isinstance(src, dict) else getattr(src, attr, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[key] = float(val)
    return out


# ---------------------------------------------------------------------------
# OpenTelemetry export — OPTIONAL extra; degrades to a no-op without the dep
# ---------------------------------------------------------------------------


class OTelExporter:
    """Mirror the funnel + judge latency to OpenTelemetry, or degrade to a no-op.

    Construct via :func:`otel_exporter`. When ``opentelemetry`` is importable it builds a standard
    set of instruments off the global meter:

    - ``percept.funnel.<stage>`` — a Counter per funnel stage, with ``concern`` / ``camera``
      attributes when scoped.
    - ``percept.funnel.dropped`` — a Counter for :meth:`Funnel.dropped` (NOT one of the five
      :data:`FUNNEL_STAGES`, but mirrored alongside them), with a ``reason`` attribute.
    - ``percept.judge.latency`` — a Histogram (seconds).
    - ``percept.judge.tokens`` — a Counter with a ``direction`` (input/output) attribute.
    - ``percept.judge.cost_usd`` — a Counter.

    When the SDK is absent, :attr:`enabled` is ``False`` and every method is a no-op — the in-process
    :class:`Metrics` stays the source of truth and the core remains zero-dep.
    """

    def __init__(self, meter: Any | None) -> None:
        self.enabled = meter is not None
        self._stage_counters: dict[str, Any] = {}
        self._latency = None
        self._tokens = None
        self._cost = None
        if meter is not None:
            for stage in FUNNEL_STAGES:
                self._stage_counters[stage] = meter.create_counter(f"percept.funnel.{stage}")
            # dropped rides the same stage-counter map (looked up by name in record_stage) without
            # being one of the five FUNNEL_STAGES a dashboard iterates.
            self._stage_counters["dropped"] = meter.create_counter("percept.funnel.dropped")
            self._latency = meter.create_histogram(
                "percept.judge.latency", unit="s", description="judge wall-time"
            )
            self._tokens = meter.create_counter("percept.judge.tokens")
            self._cost = meter.create_counter("percept.judge.cost_usd")

    def record_stage(
        self,
        stage: str,
        *,
        concern: str | None,
        camera: str | None,
        n: int,
        reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        counter = self._stage_counters.get(stage)
        if counter is None:  # pragma: no cover - defensive: unknown stage
            return
        attrs: dict[str, str] = {}
        if concern is not None:
            attrs["concern"] = concern
        if camera is not None:
            attrs["camera"] = camera
        if reason is not None:
            attrs["reason"] = reason
        counter.add(n, attrs)

    def record_judge(
        self,
        *,
        latency_s: float | None = None,
        input_tokens: float | None = None,
        output_tokens: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        if latency_s is not None and self._latency is not None:
            self._latency.record(float(latency_s))
        if input_tokens is not None and self._tokens is not None:
            self._tokens.add(float(input_tokens), {"direction": "input"})
        if output_tokens is not None and self._tokens is not None:
            self._tokens.add(float(output_tokens), {"direction": "output"})
        if cost_usd is not None and self._cost is not None:
            self._cost.add(float(cost_usd))


def otel_exporter(meter: Any | None = None) -> OTelExporter:
    """Build an :class:`OTelExporter`, lazy-importing ``opentelemetry`` (the ``[otel]`` extra).

    Pass a ``meter`` explicitly (tests / custom wiring), or leave it ``None`` to fetch the global
    OTel meter. If ``opentelemetry`` is not installed the returned exporter is a no-op
    (``enabled == False``) — the core degrades to the in-process :class:`Metrics` with no error.
    """
    if meter is None:
        try:  # lazy by design (invariant 7) — the [otel] extra gates this import.
            from opentelemetry import metrics as _otel_metrics  # noqa: PLC0415
        except ImportError:  # pragma: no cover - exercised only without the extra
            return OTelExporter(None)
        meter = _otel_metrics.get_meter("percept")
    return OTelExporter(meter)


def health(version: str, backends: dict[str, str], metrics: Metrics | None = None) -> dict:
    """The /health payload — what's wired and how it's doing."""
    return {
        "status": "ok",
        "version": version,
        "backends": backends,
        "metrics": metrics.snapshot() if metrics else {},
    }


__all__ = [
    "get_logger",
    "log",
    "TraceEvent",
    "Tracer",
    "Metrics",
    "Funnel",
    "FUNNEL_STAGES",
    "funnel_key",
    "judge_usage",
    "OTelExporter",
    "otel_exporter",
    "health",
]
