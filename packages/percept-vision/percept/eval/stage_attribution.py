"""Stage attribution — turn a recorded trace into WHICH stage missed a budget.

A pass/fail on the whole pipe is not a diagnosis. When an end-to-end assertion goes red, the two
questions that actually route the fix are *where did the units go* (the funnel) and *where did the
time go* (the per-segment latency). This module answers both from a ``.ptrace``'s records alone, so
an integration assertion can name a stage instead of naming the pipe.

**It composes; it does not re-derive.** Every number here already has exactly one owner elsewhere in
the tree, and this module calls that owner rather than repeating it:

===========================  =================================================================
what                          who owns it
===========================  =================================================================
the funnel counts             :meth:`percept.tracing.TraceSummary.derive` (which recounts from
                              the records through the same fold the live recorder uses)
admitted evidence duration    :func:`percept.tracing.union_seconds`, reached through
                              :attr:`~percept.tracing.TraceSummary.admitted_evidence_union_s` —
                              a UNION of the evidence windows, never ``count x interval``
the stage vocabulary          :data:`percept.tracing.TRACE_STAGES` constants; the judge span's
                              two boundaries are :data:`percept.slo.JUDGE_LATENCY_START_STAGE`
                              and :data:`percept.slo.JUDGE_LATENCY_END_STAGES`, imported rather
                              than re-spelled, so a degraded judgment keeps closing the span
percentiles                   :func:`percept.slo.measure_judge_latency`, which itself uses
                              :func:`percept.harness.metrics.percentile` (nearest-rank)
"may I quote this?"           :attr:`percept.slo.LatencyDistribution.is_publishable` /
                              :attr:`~percept.slo.LatencyDistribution.is_censored`
===========================  =================================================================

**The three pieces of arithmetic this module does own**, because nothing else can:

1. ``downstream / upstream`` for a pair of adjacent funnel points. A zero denominator has NO ratio
   and reads ``None`` — ``0.0`` and ``1.0`` are both a wrong number that a bar would pass.
2. **Keeping the unit-changing expansion out of the ratio.** Admission counts evidence units;
   dispatch counts ``(evidence, goal)`` opportunities declared by each admission record. One frame
   may create N opportunities. That expansion is not attrition, so the dispatch survival ratio is
   ``judge_started / judge_dispatch_opportunities`` — never ``judge_started / admitted`` and never
   a global-goal multiplier. See :data:`FUNNEL_POINT_UNITS`.
3. ``end.emitted_wall_ts - start.emitted_wall_ts`` for one segment, clamped at zero exactly as
   :func:`percept.slo.judge_provider_spans` already clamps it, with the clamped count reported
   rather than swallowed.

**Refusing to publish is a feature.** A latency distribution truncated by a deadline describes the
deadline, not the system: the "measured" median is the deadline's own shadow. :meth:`
StageAttribution.percentile` raises :class:`CensoredLatencyError` instead of returning such a
number, and it refuses on two independent grounds — a deadline the caller declares
(``deadlines=``), and a judge span the trace itself shows never closed.

**Percentiles do not add.** The per-segment figures are separate distributions over separate sample
populations (one ``intake`` sample per frame; one ``judge`` sample per judged goal). They do not sum
to the ``end_to_end`` figure and are not a per-incident decomposition.
:meth:`StageAttribution.slowest_segment` names the segment with the worst tail — a pointer for a
diagnosis, not an accounting identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from percept.slo import (
    JUDGE_LATENCY_END_STAGES,
    JUDGE_LATENCY_START_STAGE,
    LatencyConditions,
    LatencyDistribution,
    measure_judge_latency,
)
from percept.tracing import (
    STAGE_EDGE_DECISION,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    STAGE_SOURCE_FRAME,
    RunTraceManifest,
    TraceRecord,
    TraceSummary,
    judge_call_was_served,
)

__all__ = [
    "StageAttributionError",
    "CensoredLatencyError",
    "UnmeasuredSegmentError",
    "FUNNEL_POINTS",
    "FUNNEL_STEPS",
    "FUNNEL_POINT_UNITS",
    "PER_CONCERN_STEPS",
    "SEGMENTS",
    "SEGMENT_NAMES",
    "REPORTABLE_PERCENTILES",
    "Segment",
    "FunnelStep",
    "StageFunnel",
    "SegmentTruncation",
    "StageAttribution",
    "conditions_from_manifest",
    "stage_funnel",
    "stage_latencies",
    "attribute",
]


# ---------------------------------------------------------------------------
# 0. Refusals
# ---------------------------------------------------------------------------


class StageAttributionError(RuntimeError):
    """Base for every refusal this module makes."""


class CensoredLatencyError(StageAttributionError):
    """A percentile was asked of a distribution that may not be quoted.

    Raised when a deadline truncated the sample (:attr:`percept.slo.LatencyDistribution.is_censored`)
    or when the trace shows spans that opened and never closed, so the sample silently excludes
    exactly the slow calls the percentile is supposed to describe.
    """


class UnmeasuredSegmentError(StageAttributionError):
    """A figure was asked for a segment this trace carries no sample for.

    Not measured is not zero, and it is not a small number either — it has no value at all.
    """


# ---------------------------------------------------------------------------
# 1. The funnel spine
# ---------------------------------------------------------------------------

#: The funnel points, most upstream first. ``admitted`` and
#: ``judge_dispatch_opportunities`` are deliberately adjacent but are NOT one survival step: the
#: former is evidence, the latter is the exact per-goal expansion that evidence created.
FUNNEL_POINTS: tuple[str, ...] = (
    "observed",
    "admitted",
    "judge_dispatch_opportunities",
    "judge_started",
    "judge_completed",
    "fired",
)

#: Unit of each named point. Kept in the object returned to renderers so a table can never display a
#: ratio between two populations merely because their counts happen to sit next to each other.
FUNNEL_POINT_UNITS: Mapping[str, str] = {
    "observed": "evidence_units",
    "admitted": "evidence_units",
    "judge_dispatch_opportunities": "evidence_goal_pairs",
    "judge_started": "evidence_goal_pairs",
    "judge_completed": "evidence_goal_pairs",
    "fired": "goal_outcomes",
}

#: Each adjacent pair of funnel points, and the name of the transition between them. The first three
#: names deliberately match the latency segment of the same name, so a count ratio and a duration
#: line up on one row of a report.
FUNNEL_STEPS: tuple[tuple[str, str, str], ...] = (
    ("admission", "observed", "admitted"),
    ("dispatch", "judge_dispatch_opportunities", "judge_started"),
    ("judge", "judge_started", "judge_completed"),
    ("outcome", "judge_completed", "fired"),
)

#: Compatibility/export marker: dispatch is the per-concern step. It no longer licenses a global
#: multiplication; its upstream point is already measured in ``evidence_goal_pairs``.
PER_CONCERN_STEPS: frozenset[str] = frozenset({"dispatch"})

#: The percentiles a :class:`~percept.slo.LatencyDistribution` actually carries. Asking for anything
#: else would mean re-deriving from raw samples this module deliberately does not keep.
REPORTABLE_PERCENTILES: tuple[float, ...] = (25.0, 50.0, 95.0, 100.0)

_PERCENTILE_FIELDS: dict[float, str] = {
    25.0: "p25_s",
    50.0: "p50_s",
    95.0: "p95_s",
    100.0: "max_s",
}


@dataclass(frozen=True)
class Segment:
    """One measurable span between two trace stages.

    ``start_stage`` is ``None`` for a span anchored at the unit of evidence's ARRIVAL rather than at
    another stage — those read :attr:`percept.tracing.TraceRecord.latency_s` directly, which the
    recorder already defines as "this decision's emission minus when the runtime first saw this
    ``trace_id``".

    ``unclosed_means_truncated`` is set only where an opened-but-never-closed span is evidence of
    truncation rather than ordinary funnel attrition. That is true of the judge span and of nothing
    else: :data:`percept.slo.JUDGE_LATENCY_END_STAGES` closes it on a DEGRADED judgment as well as a
    finished one, precisely so a slow call still lands in the sample — so a ``judge.started`` with
    neither is a call that could not finish. Elsewhere a start with no end is the funnel doing its
    job (an admitted frame that never needed a judge), and the funnel table already reports it.
    """

    name: str
    start_stage: str | None
    end_stages: tuple[str, ...]
    unclosed_means_truncated: bool = False
    spans_whole_pipe: bool = False
    what: str = ""
    #: Drop closes whose call never reached a provider (a cache serve, a direct hit). Set only on
    #: ``judge``. A sub-millisecond serve is a real wait but it is not a PROVIDER latency, and mixing
    #: the two makes the percentile a function of the cache hit rate rather than of the model — the
    #: same reason :func:`percept.slo.judge_provider_spans` drops them. Without this the segment
    #: silently disagreed with ``slo``, which is the module it imports its own boundaries from
    #: precisely so the two can never disagree (FIND-LOADSTAGE-01, alias site).
    excludes_served: bool = False


#: The measurable segments, upstream first. ``judge``'s boundaries are imported from
#: :mod:`percept.slo` rather than re-declared, so the two modules can never disagree about what
#: closes a judge span.
SEGMENTS: tuple[Segment, ...] = (
    Segment(
        "intake",
        None,
        (STAGE_EDGE_DECISION,),
        what="arrival of the unit of evidence -> the always-on admission decision",
    ),
    Segment(
        "admission",
        STAGE_EDGE_DECISION,
        (STAGE_SOURCE_FRAME,),
        what="the admission decision -> the evidence entering the judged path",
    ),
    Segment(
        "dispatch",
        STAGE_SOURCE_FRAME,
        (STAGE_JUDGE_STARTED,),
        what="admitted evidence -> the model call actually starting (queueing lives here)",
    ),
    Segment(
        "judge",
        JUDGE_LATENCY_START_STAGE,
        tuple(JUDGE_LATENCY_END_STAGES),
        unclosed_means_truncated=True,
        excludes_served=True,
        what="the model call itself, degraded outcomes included",
    ),
    Segment(
        "classification",
        STAGE_JUDGE_FINISHED,
        (STAGE_GATE_CLASSIFIED,),
        what="a verdict -> the consequence decision",
    ),
    Segment(
        "delivery",
        STAGE_GATE_CLASSIFIED,
        (STAGE_INCIDENT_CREATED,),
        what="the consequence decision -> the event existing",
    ),
    Segment(
        "end_to_end",
        None,
        (STAGE_INCIDENT_CREATED,),
        spans_whole_pipe=True,
        what="arrival -> the event existing; CONTAINS every segment above it",
    ),
)

#: Segment names in report order.
SEGMENT_NAMES: tuple[str, ...] = tuple(s.name for s in SEGMENTS)

_BY_NAME: dict[str, Segment] = {s.name: s for s in SEGMENTS}

#: Run id stamped on the summary when a caller derives attribution without a manifest. It names the
#: derivation, never a run — a trace read back without its manifest has no run identity to borrow.
_UNIDENTIFIED_RUN = "stage-attribution (no manifest supplied)"

#: What a conditions field reads when the trace cannot state it. Never a plausible-looking guess:
#: a percentile carrying invented conditions is worse than one carrying none.
_UNSTATED = "unstated"


# ---------------------------------------------------------------------------
# 2. The funnel
# ---------------------------------------------------------------------------


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is zero.

    Arithmetic this module owns (see the module docstring). The ``None`` matters: with nothing
    upstream there is no ratio, and both ``0.0`` and ``1.0`` are numbers a bar would happily pass.
    """
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


@dataclass(frozen=True)
class FunnelStep:
    """One transition between adjacent funnel points: the two counts, and the ratio between them.

    Both endpoints are in the units named by ``upstream_unit`` / ``downstream_unit``. Dispatch's
    upstream is already the exact per-admission ``(evidence, goal)`` population, so no multiplier or
    clamp participates in either the ratio or the signed drop. A legacy trace without goal metadata
    carries ``None`` at that point; its dispatch ratio/drop are unknown rather than fabricated.
    """

    name: str
    upstream: str
    downstream: str
    upstream_count: int | None
    downstream_count: int | None
    upstream_unit: str = "units"
    downstream_unit: str = "units"
    # Backward-compatible field. Exact dispatch accounting has already expanded the upstream count,
    # so this is always 1; callers should read the explicit unit fields.
    fan_out: int = 1

    @property
    def upstream_dispatches(self) -> int | None:
        """Compatibility alias for the already-unit-coherent upstream population."""
        return self.upstream_count

    @property
    def ratio(self) -> float | None:
        """``downstream / upstream`` in the downstream unit — ``None`` when nothing reached the
        upstream point."""
        return _ratio(self.downstream_count, self.upstream_dispatches)

    @property
    def dropped(self) -> int | None:
        """Units that entered this step and did not leave it. SIGNED, and never clamped.

        Measured in the explicitly named common unit. Ordinary fan-out is represented by
        ``judge_dispatch_opportunities`` before this step and therefore never registers as a drop.

        A negative value means downstream outnumbered upstream in the same unit and records are
        inconsistent. ``None`` means the legacy trace did not declare its dispatch population.
        """
        if self.upstream_count is None or self.downstream_count is None:
            return None
        return self.upstream_dispatches - self.downstream_count

    def describe(self) -> str:
        pct = "—" if self.ratio is None else f"{self.ratio * 100:.1f}%"
        upstream = "—" if self.upstream_count is None else str(self.upstream_count)
        downstream = "—" if self.downstream_count is None else str(self.downstream_count)
        units = (
            f" [{self.upstream_unit}]"
            if self.upstream_unit == self.downstream_unit
            else f" [{self.upstream_unit} -> {self.downstream_unit}]"
        )
        return (
            f"{self.name}: {self.upstream}={upstream} -> "
            f"{self.downstream}={downstream} ({pct}){units}"
        )


@dataclass(frozen=True)
class StageFunnel:
    """The funnel table: a count at every point, and the ratio across every adjacent pair.

    :attr:`summary` is the :class:`~percept.tracing.TraceSummary` every count came from — carried,
    not copied out, so a caller that needs cost, gate outcomes or the savings identity reads the one
    authoritative object rather than a second-hand field.
    """

    summary: TraceSummary
    counts: Mapping[str, int | None]
    steps: tuple[FunnelStep, ...]
    units: Mapping[str, str]
    #: Distinct named concerns, for description only. Dispatch arithmetic uses each admission's
    #: declared goal ids and never this run-global cardinality.
    concerns: int = 1

    @property
    def admitted_evidence_union_s(self) -> float:
        """Seconds of source the admissions covered — the UNION of their evidence windows.

        Straight from :attr:`percept.tracing.TraceSummary.admitted_evidence_union_s`, i.e. from
        :func:`percept.tracing.union_seconds`. Overlapping windows are counted once; this is never
        ``admitted x nominal interval``.
        """
        return self.summary.admitted_evidence_union_s

    @property
    def gate_outcomes(self) -> Mapping[str, int]:
        """``fire`` / ``ask`` / ``silent`` / ``suppressed`` counts, as the summary derived them."""
        return dict(self.summary.gate_outcomes)

    @property
    def admission_economy(self) -> float | None:
        """``1 - (admitted / observed)`` — the share of observed units that never reached a model.

        ``None`` when nothing was observed. Reported as a fraction; a caller applies its own floor.
        """
        ratio = self.step("admission").ratio
        return None if ratio is None else 1.0 - ratio

    def step(self, name: str) -> FunnelStep:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"unknown funnel step {name!r}; known: {[s.name for s in self.steps]}")

    def ratio(self, name: str) -> float | None:
        """The ratio across one named step — ``None`` when its upstream count is zero."""
        return self.step(name).ratio

    def describe(self) -> str:
        head = " · ".join(
            f"{p}={'—' if self.counts[p] is None else self.counts[p]} [{self.units[p]}]"
            for p in FUNNEL_POINTS
        )
        body = "\n".join(f"  {s.describe()}" for s in self.steps)
        union = f"{self.admitted_evidence_union_s:g}s"
        return (
            f"funnel: {head} (concerns={self.concerns})\n{body}\n"
            f"  admitted evidence (union of windows): {union}"
        )


def _concerns(records: Sequence[TraceRecord]) -> int:
    """Distinct named concerns for description only; never a dispatch multiplier."""
    return len({r.goal_id for r in records if r.goal_id}) or 1


def stage_funnel(
    records: Iterable[TraceRecord],
    *,
    manifest: RunTraceManifest | None = None,
) -> StageFunnel:
    """Derive the funnel table for ``records``.

    Every count comes from :meth:`percept.tracing.TraceSummary.derive`, which folds the records
    through the same aggregate the live recorder uses — including its rule that a run with no
    ``edge.decision`` at all (a direct ``perceive_judged`` driver) counts admissions as
    observations. That rule is inherited on purpose: two modules disagreeing about what ``observed``
    means is exactly the drift this composition exists to prevent.

    ``manifest`` is optional. Without one the summary is derived against a default manifest, which
    declares no savings baseline and a zero nominal evidence window — so no window is invented for
    an admission that did not declare its own.

    The counts are left exactly as the summary derived them. ``admitted`` remains evidence units;
    ``judge_dispatch_opportunities`` is the exact per-admission goal expansion;
    ``judge_started`` remains logical per-goal judgment starts; and ``provider_calls`` remains the
    cost population after zero-spend serves are removed. No point is multiplied by a run-global
    concern count.
    """
    seq: Sequence[TraceRecord] = tuple(records)
    summary = TraceSummary.derive(manifest or RunTraceManifest(run_id=_UNIDENTIFIED_RUN), seq)
    counts = {
        "observed": summary.observed,
        "admitted": summary.admitted,
        "admitted_evidence_units": summary.admitted,
        "judge_dispatch_opportunities": summary.judge_dispatch_opportunities,
        "judge_started": summary.judge_started,
        "judge_completed": summary.judge_completed,
        "provider_calls": summary.provider_calls,
        "fired": summary.incidents,
    }
    units = {
        **FUNNEL_POINT_UNITS,
        "admitted_evidence_units": summary.admitted_unit,
        "provider_calls": summary.provider_calls_unit,
    }
    concerns = _concerns(seq)
    steps = tuple(
        FunnelStep(
            name,
            up,
            down,
            counts[up],
            counts[down],
            upstream_unit=units[up],
            downstream_unit=units[down],
        )
        for name, up, down in FUNNEL_STEPS
    )
    return StageFunnel(
        summary=summary,
        counts=counts,
        steps=steps,
        units=units,
        concerns=concerns,
    )


# ---------------------------------------------------------------------------
# 3. Per-segment latency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentTruncation:
    """The bookkeeping behind one segment's sample — what the percentile could NOT see.

    A distribution is only as honest as the spans that failed to reach it, so every one of these is
    counted rather than dropped quietly.
    """

    segment: str
    starts: int
    closes: int
    unclosed: int
    orphan_closes: int
    clamped: int
    unclosed_is_truncation: bool = False
    #: Closes excluded because the call never reached a provider (a cache serve). Reported
    #: rather than dropped quietly: the sample is smaller than ``starts`` for a benign reason,
    #: and without this the gap is indistinguishable from truncation.
    served_closes: int = 0

    @property
    def is_truncated(self) -> bool:
        """Whether the missing spans mean the sample is truncated rather than merely filtered.

        See :attr:`Segment.unclosed_means_truncated`: only the judge span earns this, because it is
        the only one whose closing stages already include the degraded outcome.
        """
        return self.unclosed_is_truncation and self.unclosed > 0

    def describe(self) -> str:
        bits = [f"{self.segment}: {self.closes} span(s) measured"]
        if self.starts:
            bits.append(f"{self.starts} opened")
        if self.unclosed:
            marker = "NEVER CLOSED" if self.is_truncated else "unclosed (funnel attrition)"
            bits.append(f"{self.unclosed} {marker}")
        if self.orphan_closes:
            bits.append(f"{self.orphan_closes} closed with no start in this trace")
        if self.clamped:
            bits.append(f"{self.clamped} clamped at 0s (clock ran backwards)")
        return " · ".join(bits)


def _segment_spans(
    records: Sequence[TraceRecord], segment: Segment
) -> tuple[list[float], SegmentTruncation]:
    """Samples for one segment, plus what could not be sampled.

    Anchored spans (``start_stage is None``) read the record's own
    :attr:`~percept.tracing.TraceRecord.latency_s`. Paired spans key on ``(trace_id, goal_id)`` the
    way :func:`percept.slo.judge_provider_spans` does, with one addition it does not need: an
    opening record is NOT consumed by the first close, because a single admitted frame legitimately
    opens one span that many goals then close. Falling back to the most recent opening for the same
    ``trace_id`` covers the reverse shape, where the closing stage carries no ``goal_id`` at all.

    A record missing the clock the span needs contributes nothing rather than a zero — the same rule
    :func:`percept.slo.judge_stage_latencies` states.
    """
    if segment.start_stage is None:
        samples = [
            float(r.latency_s)
            for r in records
            if r.stage in segment.end_stages and r.latency_s is not None
        ]
        return samples, SegmentTruncation(
            segment=segment.name,
            starts=0,
            closes=len(samples),
            unclosed=0,
            orphan_closes=0,
            clamped=0,
            unclosed_is_truncation=segment.unclosed_means_truncated,
        )

    anchors: dict[tuple[Any, Any], float] = {}
    latest: dict[Any, tuple[Any, Any]] = {}
    used: set[tuple[Any, Any]] = set()
    samples: list[float] = []
    starts = orphan_closes = clamped = served_closes = 0

    for r in records:
        emitted = r.emitted_wall_ts
        if r.stage == segment.start_stage:
            starts += 1
            if emitted is None:
                continue
            key = (r.trace_id, r.goal_id)
            anchors[key] = float(emitted)
            latest[r.trace_id] = key
            continue
        if r.stage not in segment.end_stages or emitted is None:
            continue
        if segment.excludes_served and judge_call_was_served(getattr(r, "data", None) or {}):
            # Consume the anchor so this does NOT read as an unclosed span. A served call opened and
            # closed correctly; it is excluded from the PROVIDER distribution, which is not the same
            # fact as a call that could not finish, and `unclosed` is the truncation signal.
            key = (r.trace_id, r.goal_id)
            if key not in anchors:
                key = latest.get(r.trace_id, key)
            used.add(key)
            served_closes += 1
            continue
        key = (r.trace_id, r.goal_id)
        if key not in anchors:
            fallback = latest.get(r.trace_id)
            if fallback is None:
                orphan_closes += 1
                continue
            key = fallback
        used.add(key)
        delta = float(emitted) - anchors[key]
        if delta < 0.0:
            clamped += 1
        samples.append(max(0.0, delta))

    return samples, SegmentTruncation(
        segment=segment.name,
        starts=starts,
        closes=len(samples),
        unclosed=max(0, starts - len(used)),
        orphan_closes=orphan_closes,
        clamped=clamped,
        unclosed_is_truncation=segment.unclosed_means_truncated,
        served_closes=served_closes,
    )


def conditions_from_manifest(manifest: RunTraceManifest | None) -> LatencyConditions:
    """The measurement conditions a trace can actually state — the rest reads ``"unstated"``.

    :class:`percept.slo.LatencyConditions` exists because a percentile with no conditions attached is
    a number rather than a fact. A manifest knows the model behind the judge seam, the driver, and
    the build; it does not know a corpus, because one run is not a corpus. Nothing is filled in with
    a plausible guess.
    """
    model = transport = corpus = build = _UNSTATED
    if manifest is not None:
        for component in manifest.components:
            if component.role in ("judge", "vision") and component.model:
                model = component.model
                break
        else:
            for component in manifest.components:
                if component.role in ("judge", "vision"):
                    model = component.name
                    break
        transport = f"trace driver={manifest.driver}" if manifest.driver else _UNSTATED
        corpus = f"one run ({manifest.run_id}) — a run is not a corpus"
        build = manifest.build_commit or manifest.percept_version or _UNSTATED
    return LatencyConditions(
        model=model, transport=transport, frames=None, corpus=corpus, build=build
    )


def stage_latencies(
    records: Iterable[TraceRecord],
    *,
    conditions: LatencyConditions | None = None,
    deadlines: Mapping[str, float] | None = None,
    manifest: RunTraceManifest | None = None,
) -> dict[str, LatencyDistribution]:
    """One :class:`~percept.slo.LatencyDistribution` per measurable segment.

    A segment with no sample is ABSENT from the result rather than present with zeros — see
    :class:`UnmeasuredSegmentError`.

    ``deadlines`` maps a segment name to the deadline in force while the trace was collected. That
    value is handed to :func:`percept.slo.measure_judge_latency` as ``censored_at_s``, which is what
    makes the resulting distribution refuse to be publishable. Declare it whenever a timeout could
    have cut a span short: a distribution whose slow half was killed by a deadline reports the
    deadline, not the system.
    """
    seq: Sequence[TraceRecord] = tuple(records)
    conds = conditions if conditions is not None else conditions_from_manifest(manifest)
    limits = dict(deadlines or {})
    unknown = sorted(set(limits) - set(SEGMENT_NAMES))
    if unknown:
        raise KeyError(f"deadlines names unknown segment(s) {unknown}; known: {list(SEGMENT_NAMES)}")

    out: dict[str, LatencyDistribution] = {}
    for segment in SEGMENTS:
        samples, truncation = _segment_spans(seq, segment)
        if not samples:
            continue
        notes = []
        if truncation.is_truncated:
            notes.append(
                f"{truncation.unclosed} span(s) opened and never closed — this sample excludes "
                "every call that could not finish"
            )
        if truncation.clamped:
            notes.append(f"{truncation.clamped} span(s) clamped at 0s (clock ran backwards)")
        out[segment.name] = measure_judge_latency(
            samples,
            conditions=conds,
            censored_at_s=limits.get(segment.name),
            note="; ".join(notes),
        )
    return out


# ---------------------------------------------------------------------------
# 4. Both halves, and the publishability rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageAttribution:
    """The funnel and the per-segment latency for one trace, plus whether either may be quoted."""

    funnel: StageFunnel
    latencies: Mapping[str, LatencyDistribution]
    truncation: Mapping[str, SegmentTruncation]
    conditions: LatencyConditions

    # -- what was and was not measured ------------------------------------
    @property
    def measured_segments(self) -> tuple[str, ...]:
        return tuple(n for n in SEGMENT_NAMES if n in self.latencies)

    @property
    def unmeasured_segments(self) -> tuple[str, ...]:
        """Segments this trace carries no sample for. Absent, not zero."""
        return tuple(n for n in SEGMENT_NAMES if n not in self.latencies)

    @property
    def unpublishable_segments(self) -> tuple[str, ...]:
        """Measured segments whose percentiles must not be quoted, and why they exist."""
        return tuple(n for n in self.measured_segments if not self.is_publishable(n))

    @property
    def publishable(self) -> bool:
        """Whether every measured segment may be quoted. One censored segment poisons the report:
        the pipe's total is only as trustworthy as its least trustworthy part."""
        return not self.unpublishable_segments

    def is_publishable(self, segment: str) -> bool:
        """Whether ``segment``'s percentiles may be quoted.

        Two independent grounds for refusal, and either alone is enough:

        1. :attr:`percept.slo.LatencyDistribution.is_publishable` is ``False`` — a declared deadline
           truncated the sample, or a percentile is missing.
        2. The trace shows spans that opened and never closed where that means truncation
           (:attr:`SegmentTruncation.is_truncated`). This one needs no declaration from the caller:
           the evidence is in the records.
        """
        dist = self.latencies.get(segment)
        if dist is None:
            raise UnmeasuredSegmentError(
                f"segment {segment!r} was not measured in this trace; "
                f"measured: {list(self.measured_segments)}"
            )
        truncation = self.truncation.get(segment)
        if truncation is not None and truncation.is_truncated:
            return False
        return dist.is_publishable

    # -- reading a number out ---------------------------------------------
    def distribution(self, segment: str) -> LatencyDistribution:
        """The raw distribution, censored or not — for a caller reporting the censored run AS the
        evidence for why a deadline must move. It is retained for exactly that reason; use
        :meth:`percentile` when you intend to quote a figure."""
        dist = self.latencies.get(segment)
        if dist is None:
            raise UnmeasuredSegmentError(
                f"segment {segment!r} was not measured in this trace; "
                f"measured: {list(self.measured_segments)}"
            )
        return dist

    def percentile(self, segment: str, p: float = 95.0) -> float:
        """One quotable percentile for ``segment``, or a refusal.

        Raises :class:`CensoredLatencyError` when the distribution may not be quoted — that refusal
        is the point of this method. Raises :class:`UnmeasuredSegmentError` when the trace carries no
        sample, because "not measured" has no numeric answer.
        """
        dist = self.distribution(segment)
        if float(p) not in _PERCENTILE_FIELDS:
            raise ValueError(
                f"p must be one of {list(REPORTABLE_PERCENTILES)}, got {p!r} — a distribution "
                "carries only those, and deriving another would need samples this object drops"
            )
        if not self.is_publishable(segment):
            raise CensoredLatencyError(
                f"refusing to report p{int(p)} for segment {segment!r}: {dist.describe()}"
            )
        value = getattr(dist, _PERCENTILE_FIELDS[float(p)])
        if value is None:  # pragma: no cover - measure_judge_latency always fills all four
            raise UnmeasuredSegmentError(f"segment {segment!r} has no p{int(p)}")
        return float(value)

    def slowest_segment(self, p: float = 95.0) -> str | None:
        """The segment with the worst tail at ``p``, excluding the whole-pipe span.

        ``None`` when nothing was measured. Raises :class:`CensoredLatencyError` if ANY candidate is
        unpublishable: a censored segment could be the slowest and there is no way to know, so
        naming a winner from the rest would point the fix at the wrong stage — the precise failure
        this module exists to stop.

        The candidates are separate distributions over separate populations. This names the worst
        tail; it is not a decomposition of any single event's latency.
        """
        candidates = [n for n in self.measured_segments if not _BY_NAME[n].spans_whole_pipe]
        if not candidates:
            return None
        blocked = [n for n in candidates if not self.is_publishable(n)]
        if blocked:
            raise CensoredLatencyError(
                f"cannot rank stages: segment(s) {blocked} may not be quoted, and a censored "
                "segment could be the slowest one"
            )
        return max(candidates, key=lambda n: self.percentile(n, p))

    def describe(self) -> str:
        lines = [self.funnel.describe(), "latency by segment:"]
        for name in self.measured_segments:
            flag = "" if self.is_publishable(name) else "  [NOT QUOTABLE]"
            lines.append(f"  {name}: {self.latencies[name].describe()}{flag}")
        for name in self.unmeasured_segments:
            lines.append(f"  {name}: not measured")
        lines.append("span bookkeeping:")
        for name in SEGMENT_NAMES:
            t = self.truncation.get(name)
            if t is not None and (t.starts or t.closes or t.orphan_closes):
                lines.append(f"  {t.describe()}")
        lines.append(f"publishable: {self.publishable}")
        return "\n".join(lines)


def attribute(
    records: Iterable[TraceRecord],
    *,
    manifest: RunTraceManifest | None = None,
    conditions: LatencyConditions | None = None,
    deadlines: Mapping[str, float] | None = None,
) -> StageAttribution:
    """Funnel + per-segment latency + the publishability rule, from one pass over ``records``.

    This is the entry point an end-to-end assertion should use: it lets the assertion say WHICH
    stage missed a budget, and — through :attr:`StageAttribution.publishable` — refuses to let it
    say anything at all from numbers a deadline already decided.
    """
    seq: Sequence[TraceRecord] = tuple(records)
    conds = conditions if conditions is not None else conditions_from_manifest(manifest)
    latencies = stage_latencies(seq, conditions=conds, deadlines=deadlines)
    truncation = {s.name: _segment_spans(seq, s)[1] for s in SEGMENTS}
    return StageAttribution(
        funnel=stage_funnel(seq, manifest=manifest),
        latencies=latencies,
        truncation=truncation,
        conditions=conds,
    )
