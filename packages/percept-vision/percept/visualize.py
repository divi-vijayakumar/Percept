"""The native trace visualizer — WHERE a decision died, on one shared media-time axis.

This is the trust-to-adoption surface. A client who ran percept and got nothing must be
able to see, at a glance and offline, which STAGE of the pipeline the decision died in — not merely
that nothing happened. So every lane here is a stage of the funnel, all drawn against the same
``source_ts`` (media time) axis, and a GAP in a lane is the answer::

    source/media   ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    frames arrived
    health         ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    source stayed healthy
    admission      ................             ...but nothing was ADMITTED  <- it died HERE
    judge calls    (empty)
    gate           (empty)
    incidents      (empty)

Everything is DERIVED FROM THE EXISTING ARTIFACTS, never recomputed:

* :mod:`percept.tracing` — :class:`~percept.tracing.RunTraceManifest`,
  :class:`~percept.tracing.TraceRecord`, :class:`~percept.tracing.TraceSummary`. **The metrics panel
  IS** :class:`~percept.tracing.TraceSummary`: ``doctor`` and ``eval`` already render from it, so the
  visualizer reads the same object rather than re-counting, and the three cannot disagree.
* :mod:`percept.trace_bundle` — a portable ``.ptrace`` directory/archive is accepted wherever a
  local trace JSON is (:func:`load_visualizable`).
* :func:`percept.harness.golden_eval.score` — the drift lane calls the deterministic §5a greedy
  matcher. The matching rule is NOT reimplemented here.

The per-stat counterfactual
---------------------------
"Nothing fired" is a dead end; "here is the knob" is a fix. Every trace that carries the three
salience statistics per frame (``mean`` / ``block_max`` / ``p95`` — what ``percept doctor`` records)
gets a panel showing, for each statistic, its distribution over the run against **its own**
threshold and how many frames it WOULD have admitted. On the real client screen fixture every stat
sat under half its own cutoff, and seeing that at a glance is the whole point. Thresholds are read
from ``manifest.labels["stat_thresholds"]``; when a trace does not carry them the panel says so
rather than inventing a cutoff.

Deliberately NOT rendered (deferred to ADR-006)
-----------------------------------------------
Episode supersession, document semantic identity, speaker attribution and finding settlement are
**not** drawn. None of those relations exist in the contract yet and drawing them would promise
semantics the runtime cannot deliver. The context lane is limited to what the trace ACTUALLY
carries: ``entity_id`` correlation and goal-revision drops. See
:data:`DEFERRED_TO_ADR_006`.

Security posture (non-negotiable)
---------------------------------
The whole trace is UNTRUSTED input — verdict evidence is provider-generated, and a model can emit
markup.

* **No trace-derived text ever reaches JavaScript.** Every panel, including each decision chain, is
  pre-rendered as escaped HTML; the inline script is a fixed constant that only toggles visibility.
  There is no JSON payload to break out of.
* All text is escaped with :func:`esc` (HTML-escaped, control characters stripped, length-capped).
* **Media embedding is opt-in and defaults OFF** (``VizOptions.include_media``). The first client
  workload is accounting screens; "standalone" must not mean "silently bundle the customer's
  accounting video". A source URI is printed as TEXT and never placed in an attribute, so nothing
  is ever auto-fetched.
* No external scripts, fonts, styles or assets — everything is inlined and the page carries a strict
  CSP (``default-src 'none'``, the script pinned by ``sha256``). It opens from ``file://`` with zero
  network requests.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    STAGE_GATE_CLASSIFIED,
    STAGE_GATE_SUPPRESSED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_DROPPED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    STAGE_JUDGE_USAGE,
    STAGE_RUN_ENDED,
    STAGE_RUN_STARTED,
    STAGE_SOURCE_HEALTH,
    STAGE_TRACE_DROPPED,
    RunTraceManifest,
    TraceRecord,
    TraceSummary,
    load_trace,
)

#: Wire/behaviour version of the rendered page. Bump when a lane is removed or re-meant.
VISUALIZER_VERSION = "1"

#: The salience statistics the counterfactual panel reports, in the order doctor prints them.
#: Each has its OWN calibrated cutoff — ``block_max``/``p95`` sit on a higher value scale than the
#: whole-frame ``mean`` and are never compared to the mean's threshold.
SALIENCE_STATS: tuple[str, ...] = ("mean", "block_max", "p95")

#: Concepts this renderer REFUSES to draw until ADR-006 defines their representation. Rendering
#: them would promise relational semantics the runtime does not carry.
DEFERRED_TO_ADR_006: tuple[str, ...] = (
    "episode supersession",
    "document semantic identity",
    "speaker attribution",
    "finding settlement",
)

#: Record-data keys that may carry media bytes. Never rendered unless media is opted in.
_MEDIA_KEYS: frozenset[str] = frozenset(
    {"frame", "frames", "image", "snippet", "clip", "media", "roi_image"}
)

#: Control characters stripped from every rendered string (tab/newline survive as spaces).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class VisualizeError(Exception):
    """A trace could not be loaded or rendered. Typed, never a silent empty page."""


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VizOptions:
    """What the page is allowed to CARRY and how big it may get.

    ``include_media=False`` is the security default, not a convenience one: a visualizer that
    embeds pixels by default is an exfiltration vector wearing a debugging costume.
    """

    include_media: bool = False
    title: str = "percept trace"
    #: Max lane items drawn per lane. Truncation is REPORTED on the page, never silent.
    max_lane_items: int = 4_000
    #: Max decision chains pre-rendered (each is a full panel).
    max_chains: int = 500
    #: Max characters of any single trace-derived string.
    max_text_chars: int = 2_000
    #: Max data: bytes embedded across the whole page (only reachable with ``include_media``).
    max_media_bytes: int = 4_000_000
    max_media_files: int = 32
    #: Hard cap on the produced HTML. Exceeding it raises rather than writing a 400MB file.
    max_html_bytes: int = 48_000_000

    def __post_init__(self) -> None:
        for name in ("max_lane_items", "max_chains", "max_text_chars", "max_html_bytes"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_media_bytes < 0 or self.max_media_files < 0:
            raise ValueError("media bounds must be >= 0")


# ---------------------------------------------------------------------------
# Escaping — every trace-derived string goes through here
# ---------------------------------------------------------------------------


def esc(value: Any, *, limit: int = 2_000) -> str:
    """HTML-escape UNTRUSTED trace text: strip control chars, cap length, escape ``&<>"'``.

    Verdict evidence is model output and a model can emit markup, so this is applied to EVERY
    trace-derived string — keys and values alike — before it reaches the page.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    elif isinstance(value, (int,)):
        text = str(value)
    elif isinstance(value, (list, tuple, dict)):
        try:
            text = json.dumps(value, default=str, sort_keys=True)[: limit * 2]
        except Exception:  # noqa: BLE001 - untrusted input must never break rendering
            text = repr(value)
    else:
        text = str(value)
    text = _CONTROL_RE.sub(" ", text)
    if len(text) > limit:
        text = text[:limit] + f" …[+{len(text) - limit} chars truncated]"
    return html.escape(text, quote=True)


def _num(x: float) -> str:
    """A number formatted for a style attribute. Never derived from trace TEXT."""
    if x != x or x in (float("inf"), float("-inf")):  # NaN / inf guard
        return "0"
    return f"{x:.4f}"


def _fmt(v: Any, spec: str = ".3f") -> str:
    if v is None:
        return "—"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return esc(v, limit=64)


# ---------------------------------------------------------------------------
# The lane model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneItem:
    """One drawn thing on a lane: a span ``[t0, t1]`` in MEDIA time, plus its escaped detail."""

    t0: float
    t1: float
    kind: str
    label: str
    detail: tuple[tuple[str, Any], ...] = ()
    firing_id: str | None = None


@dataclass(frozen=True)
class Lane:
    """One pipeline STAGE. An empty lane is the diagnosis, so it is always drawn."""

    key: str
    title: str
    note: str
    items: tuple[LaneItem, ...] = ()
    truncated: int = 0


@dataclass(frozen=True)
class StatCounterfactual:
    """What ONE salience statistic would have decided against ITS OWN threshold.

    ``ratio`` is ``max_value / threshold``: below 1.0 the loudest frame of the entire run never
    cleared the cutoff, and below 0.5 the statistic is not merely losing, it is nowhere near.
    """

    stat: str
    threshold: float | None
    frames: int
    max_value: float
    mean_value: float
    median_value: float
    would_admit: int
    active: bool = False

    @property
    def ratio(self) -> float | None:
        if not self.threshold:
            return None
        return self.max_value / float(self.threshold)

    @property
    def verdict(self) -> str:
        if self.threshold is None:
            return "threshold not recorded in this trace"
        if self.would_admit > 0:
            return f"would admit {self.would_admit}/{self.frames} frames"
        r = self.ratio
        if r is not None and r < 0.5:
            return f"never close — peak is {r * 100:.0f}% of its own cutoff"
        return "peak never cleared its own cutoff"


@dataclass(frozen=True)
class DecisionChain:
    """tier-0 -> sense -> VLM -> gate, for ONE unit of evidence, with its own counterfactual."""

    firing_id: str
    trace_id: str
    source_ts: float | None
    action: str
    reason: str
    goal_id: str | None
    stages: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    counterfactual: tuple[StatCounterfactual, ...] = ()


@dataclass(frozen=True)
class VizModel:
    """Everything the page draws — built once, from the artifacts, then escaped into HTML."""

    manifest: RunTraceManifest
    summary: TraceSummary
    lanes: tuple[Lane, ...]
    chains: tuple[DecisionChain, ...]
    counterfactual: tuple[StatCounterfactual, ...]
    t0: float
    t1: float
    notes: tuple[str, ...] = ()
    drift: Any = None
    media: tuple[tuple[str, str, str], ...] = ()  # (name, media_type, data_uri)


# ---------------------------------------------------------------------------
# Building the model from the artifacts
# ---------------------------------------------------------------------------


def _time_domain(records: Sequence[TraceRecord]) -> tuple[float, float]:
    ts = [float(r.source_ts) for r in records if r.source_ts is not None]
    if not ts:
        return (0.0, 1.0)
    lo, hi = min(ts), max(ts)
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def _safe_data(rec: TraceRecord, *, include_media: bool) -> tuple[tuple[str, Any], ...]:
    """Record data as ordered ``(key, value)`` pairs, with media keys dropped unless opted in."""
    items = []
    for k in sorted(rec.data):
        if not include_media and k in _MEDIA_KEYS:
            items.append((k, "<media withheld — embedding is opt-in>"))
            continue
        items.append((k, rec.data[k]))
    return tuple(items)


def _stat_counterfactual(
    manifest: RunTraceManifest, records: Sequence[TraceRecord]
) -> tuple[StatCounterfactual, ...]:
    """Per-stat counterfactual over the whole run — the "here is the knob" panel.

    Values are read off the ``edge.decision`` records the run already wrote (``percept doctor``
    records all three statistics per frame from ONE pass over the pixels); thresholds come from
    ``manifest.labels["stat_thresholds"]``. Nothing is recomputed from pixels here — the
    visualizer has no frames and must never pretend to.
    """
    labels = manifest.labels or {}
    raw_thresholds = labels.get("stat_thresholds") or {}
    active_stat = labels.get("salience_stat")
    edge = [r for r in records if r.stage == OBSERVATION_STAGE]
    rows: list[StatCounterfactual] = []
    for stat in SALIENCE_STATS:
        values = [
            float(r.data[stat])
            for r in edge
            if isinstance(r.data.get(stat), (int, float)) and not isinstance(r.data.get(stat), bool)
        ]
        thr: float | None = None
        raw = raw_thresholds.get(stat) if isinstance(raw_thresholds, Mapping) else None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            thr = float(raw)
        if not values and thr is None:
            continue
        would = sum(1 for v in values if thr is not None and v >= thr)
        ordered = sorted(values)
        rows.append(
            StatCounterfactual(
                stat=stat,
                threshold=thr,
                frames=len(values),
                max_value=max(values) if values else 0.0,
                mean_value=(sum(values) / len(values)) if values else 0.0,
                median_value=ordered[len(ordered) // 2] if ordered else 0.0,
                would_admit=would,
                active=(stat == active_stat),
            )
        )
    return tuple(rows)


def _frame_counterfactual(
    manifest: RunTraceManifest, edge_rec: TraceRecord | None
) -> tuple[StatCounterfactual, ...]:
    """The counterfactual for ONE frame: what each stat would have decided about THIS evidence."""
    if edge_rec is None:
        return ()
    labels = manifest.labels or {}
    raw_thresholds = labels.get("stat_thresholds") or {}
    active_stat = labels.get("salience_stat")
    rows: list[StatCounterfactual] = []
    for stat in SALIENCE_STATS:
        v = edge_rec.data.get(stat)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        raw = raw_thresholds.get(stat) if isinstance(raw_thresholds, Mapping) else None
        thr = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        rows.append(
            StatCounterfactual(
                stat=stat, threshold=thr, frames=1, max_value=float(v),
                mean_value=float(v), median_value=float(v),
                would_admit=1 if (thr is not None and float(v) >= thr) else 0,
                active=(stat == active_stat),
            )
        )
    return tuple(rows)


def _health_lane(records: Sequence[TraceRecord], t0: float, t1: float) -> Lane:
    """Degraded intervals, latched until a recovery (degrade-loud — never forgiven silently)."""
    items: list[LaneItem] = []
    open_at: float | None = None
    open_rec: TraceRecord | None = None
    for r in records:
        if r.stage != STAGE_SOURCE_HEALTH:
            continue
        state = str(r.data.get("state", ""))
        ts = float(r.source_ts if r.source_ts is not None else t0)
        if state == "degraded" and open_at is None:
            open_at, open_rec = ts, r
        elif state in ("recovered", "ended") and open_at is not None:
            items.append(LaneItem(open_at, ts, "degraded", str(open_rec.data.get("kind", "degraded")),
                                  _safe_data(open_rec, include_media=False)))
            open_at, open_rec = None, None
        elif state == "info":
            items.append(LaneItem(ts, ts, "info", str(r.data.get("kind", "info")),
                                  _safe_data(r, include_media=False)))
    if open_at is not None and open_rec is not None:
        items.append(LaneItem(open_at, t1, "degraded",
                              f"{open_rec.data.get('kind', 'degraded')} (never recovered)",
                              _safe_data(open_rec, include_media=False)))
    note = (
        "healthy for the whole run — the source was not the reason"
        if not items
        else "degraded intervals are latched until an explicit recovery"
    )
    return Lane("health", "health", note, tuple(items))


def _admission_lane(
    manifest: RunTraceManifest, records: Sequence[TraceRecord], opts: VizOptions
) -> Lane:
    """Every observed frame: admitted vs dropped, with its motion trace and the threshold rule."""
    labels = manifest.labels or {}
    active_stat = labels.get("salience_stat")
    thresholds = labels.get("stat_thresholds") if isinstance(labels.get("stat_thresholds"), Mapping) else {}
    items: list[LaneItem] = []
    truncated = 0
    for r in records:
        if r.stage != OBSERVATION_STAGE:
            continue
        if len(items) >= opts.max_lane_items:
            truncated += 1
            continue
        ts = float(r.source_ts or 0.0)
        admitted = bool(r.data.get("escalate")) or bool(r.data.get("coast_admit"))
        # The rule that decided it, quoted from the record itself — never recomputed.
        thr = r.data.get("threshold")
        if thr is None and isinstance(active_stat, str):
            thr = (thresholds or {}).get(active_stat)
        value = None
        for key in ((active_stat,) if isinstance(active_stat, str) else ()) + ("motion",) + SALIENCE_STATS:
            if isinstance(r.data.get(key), (int, float)) and not isinstance(r.data.get(key), bool):
                value = float(r.data[key])
                break
        label = "admitted" if admitted else "dropped"
        detail = _safe_data(r, include_media=opts.include_media) + (
            ("rule", f"{active_stat or 'salience'} {_fmt(value, '.4f')} "
                     f"{'>=' if admitted else '<'} threshold {_fmt(thr, '.4f')}"),
        )
        items.append(LaneItem(ts, ts, "admitted" if admitted else "dropped", label, detail))
    n_adm = sum(1 for i in items if i.kind == "admitted")
    note = (
        f"{n_adm} admitted / {len(items)} observed"
        if items
        else "no per-frame edge.decision records in this trace (retention level, not a silent loss)"
    )
    if items and n_adm == 0:
        note += " — THE DECISION DIED HERE: tier-0 admitted nothing, so no judge call was possible"
    return Lane("admission", "admission (tier-0)", note, tuple(items), truncated)


def _judge_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    """One bar per judge call; the bar's WIDTH is wall latency, its detail carries tokens + cost."""
    started: dict[str, TraceRecord] = {}
    usage: dict[str, TraceRecord] = {r.trace_id: r for r in records if r.stage == STAGE_JUDGE_USAGE}
    items: list[LaneItem] = []
    truncated = 0
    for r in records:
        if r.stage == STAGE_JUDGE_STARTED:
            started[r.trace_id] = r
            continue
        if r.stage not in (STAGE_JUDGE_FINISHED, STAGE_JUDGE_DEGRADED, STAGE_JUDGE_DROPPED):
            continue
        if len(items) >= opts.max_lane_items:
            truncated += 1
            continue
        begin = started.pop(r.trace_id, None)
        ts = float(r.source_ts or (begin.source_ts if begin is not None else 0.0) or 0.0)
        lat = r.latency_s
        u = usage.get(r.trace_id)
        detail = _safe_data(r, include_media=opts.include_media) + (
            ("latency_s (wall)", _fmt(lat, ".3f")),
        )
        if u is not None:
            detail += tuple(("usage." + k, v) for k, v in _safe_data(u, include_media=False))
        kind = {
            STAGE_JUDGE_FINISHED: "judge",
            STAGE_JUDGE_DEGRADED: "degraded",
            STAGE_JUDGE_DROPPED: "dropped",
        }[r.stage]
        items.append(LaneItem(ts, ts + max(0.0, float(lat or 0.0)), kind, kind, detail))
    for tid, begin in started.items():  # started, never finished — a call that is still in flight
        items.append(LaneItem(float(begin.source_ts or 0.0), float(begin.source_ts or 0.0),
                              "unfinished", "started, never finished",
                              _safe_data(begin, include_media=False)))
    note = "bar width = wall latency of the call" if items else "no judge call was ever made"
    return Lane("judge", "judge calls", note, tuple(items), truncated)


def _gate_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    """FIRE / ASK / SILENT with the reason code, plus typed suppressions."""
    items: list[LaneItem] = []
    truncated = 0
    n = 0
    for r in records:
        if r.stage not in (STAGE_GATE_CLASSIFIED, STAGE_GATE_SUPPRESSED):
            continue
        if len(items) >= opts.max_lane_items:
            truncated += 1
            continue
        ts = float(r.source_ts or 0.0)
        if r.stage == STAGE_GATE_SUPPRESSED:
            action, reason = "suppressed", str(r.data.get("reason", r.data.get("kind", "suppressed")))
        else:
            action = str(r.data.get("action", "?")).lower()
            reason = str(r.data.get("reason", ""))
        firing_id = f"d{n}"
        n += 1
        items.append(LaneItem(
            ts, ts, action, f"{action.upper()}: {reason}" if reason else action.upper(),
            _safe_data(r, include_media=opts.include_media), firing_id,
        ))
    note = "click a decision (or use ← →) for its chain" if items else "the gate never classified anything"
    return Lane("gate", "gate decisions", note, tuple(items), truncated)


def _context_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    """ONLY what the trace actually carries: ``entity_id`` correlation and goal-revision drops.

    Everything richer (episode supersession, document identity, speaker attribution, finding
    settlement) is deferred to ADR-006 and deliberately not drawn — see :data:`DEFERRED_TO_ADR_006`.
    """
    spans: dict[str, list[float]] = {}
    items: list[LaneItem] = []
    for r in records:
        ent = r.data.get("entity_id")
        if isinstance(ent, str) and ent:
            ts = float(r.source_ts or 0.0)
            got = spans.setdefault(ent, [ts, ts])
            got[0], got[1] = min(got[0], ts), max(got[1], ts)
        if r.stage == STAGE_JUDGE_DROPPED and "revision" in str(r.data.get("kind", "")):
            items.append(LaneItem(float(r.source_ts or 0.0), float(r.source_ts or 0.0),
                                  "revision", "goal revision changed — candidate dropped",
                                  _safe_data(r, include_media=False)))
    for ent, (a, b) in sorted(spans.items())[: opts.max_lane_items]:
        items.append(LaneItem(a, b, "entity", f"entity {ent}", (("entity_id", ent),)))
    items.sort(key=lambda i: (i.t0, i.kind))
    note = (
        "entity_id correlation + goal-revision drops only. Relations "
        "(" + ", ".join(DEFERRED_TO_ADR_006) + ") are DEFERRED to ADR-006 and are not drawn."
    )
    return Lane("context", "context", note, tuple(items))


def _cost_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    items: list[LaneItem] = []
    running = 0.0
    any_priced = False
    for r in records:
        if r.stage != STAGE_JUDGE_USAGE:
            continue
        if len(items) >= opts.max_lane_items:
            break
        v = r.data.get("cost_usd")
        priced = isinstance(v, (int, float)) and not isinstance(v, bool)
        if priced:
            running += float(v)
            any_priced = True
        ts = float(r.source_ts or 0.0)
        # An UNPRICED usage record (tokens known, model not in the price table) shows "—", never a
        # cumulative dollar figure that pretends the call was free.
        label = f"${running:.4f} cumulative" if any_priced else "— (unpriced)"
        items.append(LaneItem(ts, ts, "cost", label, _safe_data(r, include_media=False)))
    if not items:
        # NOT "$0 spent": no usage record means the backend reported nothing, which says nothing
        # about what the run spent. Unknown is safe; a zero is a wrong number a cost bar can pass.
        note = "no judge usage was reported — spend unknown (—)"
    elif any_priced:
        note = f"cumulative ${running:.4f}"
    else:
        note = "judge usage reported, but no model price — spend unknown (—)"
    return Lane("cost", "cost", note, tuple(items))


def _source_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    items: list[LaneItem] = []
    truncated = 0
    for r in records:
        if r.stage not in (ADMISSION_STAGE, STAGE_RUN_STARTED, STAGE_RUN_ENDED, STAGE_TRACE_DROPPED):
            continue
        if len(items) >= opts.max_lane_items:
            truncated += 1
            continue
        ts = float(r.source_ts or 0.0)
        kind = {
            ADMISSION_STAGE: "frame", STAGE_RUN_STARTED: "run", STAGE_RUN_ENDED: "run",
            STAGE_TRACE_DROPPED: "lost",
        }[r.stage]
        items.append(LaneItem(ts, ts, kind, r.stage, _safe_data(r, include_media=opts.include_media)))
    note = "frames that reached perceive_judged (a cloud call became possible)"
    return Lane("source", "source / media", note, tuple(items), truncated)


def _incident_lane(records: Sequence[TraceRecord], opts: VizOptions) -> Lane:
    items: list[LaneItem] = []
    for r in records:
        if r.stage != STAGE_INCIDENT_CREATED:
            continue
        if len(items) >= opts.max_lane_items:
            break
        ts = float(r.source_ts or 0.0)
        items.append(LaneItem(ts, ts, "incident", str(r.data.get("incident_id", "incident")),
                              _safe_data(r, include_media=opts.include_media)))
    note = "delivered incidents" if items else "no incident was ever created"
    return Lane("incidents", "incidents", note, tuple(items))


def _chains(
    manifest: RunTraceManifest, records: Sequence[TraceRecord], opts: VizOptions
) -> tuple[DecisionChain, ...]:
    """One click-through chain per gate decision: tier-0 -> sense -> VLM -> gate."""
    by_trace: dict[str, list[TraceRecord]] = {}
    for r in records:
        by_trace.setdefault(r.trace_id, []).append(r)
    chains: list[DecisionChain] = []
    n = 0
    for r in records:
        if r.stage not in (STAGE_GATE_CLASSIFIED, STAGE_GATE_SUPPRESSED):
            continue
        if len(chains) >= opts.max_chains:
            break
        firing_id = f"d{n}"
        n += 1
        family = by_trace.get(r.trace_id, [r])
        edge = next((x for x in family if x.stage == OBSERVATION_STAGE), None)
        stages: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
        order = [
            ("tier-0 (edge.decision)", OBSERVATION_STAGE),
            ("sense (admitted)", ADMISSION_STAGE),
            ("VLM call started", STAGE_JUDGE_STARTED),
            ("VLM verdict", STAGE_JUDGE_FINISHED),
            ("VLM degraded", STAGE_JUDGE_DEGRADED),
            ("VLM usage (tokens/cost)", STAGE_JUDGE_USAGE),
            ("gate", r.stage),
        ]
        for label, stage in order:
            hit = next((x for x in family if x.stage == stage), None)
            if hit is None:
                stages.append((label, (("—", "no record at this stage for this evidence"),)))
                continue
            detail = _safe_data(hit, include_media=opts.include_media)
            lat = hit.latency_s
            if lat is not None:
                detail = detail + (("latency_s (wall)", _fmt(lat, ".3f")),)
            stages.append((label, detail))
        action = (
            "suppressed" if r.stage == STAGE_GATE_SUPPRESSED
            else str(r.data.get("action", "?")).lower()
        )
        chains.append(DecisionChain(
            firing_id=firing_id,
            trace_id=r.trace_id,
            source_ts=r.source_ts,
            action=action,
            reason=str(r.data.get("reason", "")),
            goal_id=r.goal_id,
            stages=tuple(stages),
            counterfactual=_frame_counterfactual(manifest, edge),
        ))
    return tuple(chains)


def build_model(
    manifest: RunTraceManifest,
    records: Sequence[TraceRecord],
    summary: TraceSummary,
    *,
    options: VizOptions | None = None,
    golden: Any = None,
    media: Sequence[tuple[str, str, bytes]] = (),
) -> VizModel:
    """Assemble every lane, chain and panel from the artifacts. Recomputes no metric."""
    opts = options or VizOptions()
    ordered = sorted(records, key=lambda r: (r.source_ts if r.source_ts is not None else 0.0, r.seq))
    t0, t1 = _time_domain(ordered)
    notes: list[str] = []

    lanes = [
        _source_lane(ordered, opts),
        _health_lane(ordered, t0, t1),
        _admission_lane(manifest, ordered, opts),
        _judge_lane(ordered, opts),
        _gate_lane(ordered, opts),
        _incident_lane(ordered, opts),
        _context_lane(ordered, opts),
        _cost_lane(ordered, opts),
    ]

    drift = None
    if golden is not None:
        drift, drift_lane = _drift(ordered, golden, opts)
        lanes.insert(6, drift_lane)

    if summary.records_dropped:
        notes.append(
            f"{summary.records_dropped} records were LOST to a retention bound — this timeline is "
            "incomplete, and the funnel counters above are still exact."
        )
    if not opts.include_media:
        notes.append(
            "Media embedding is OFF (the default). No pixels are in this file, and no source URI "
            "is ever fetched — URIs are printed as text only."
        )

    embedded: list[tuple[str, str, str]] = []
    if opts.include_media and media:
        total = 0
        for name, media_type, data in list(media)[: opts.max_media_files]:
            if total + len(data) > opts.max_media_bytes:
                notes.append(f"media truncated at {opts.max_media_bytes} bytes (size limit)")
                break
            total += len(data)
            uri = "data:" + _safe_media_type(media_type) + ";base64," + base64.b64encode(data).decode("ascii")
            embedded.append((name, media_type, uri))

    return VizModel(
        manifest=manifest, summary=summary, lanes=tuple(lanes),
        chains=_chains(manifest, ordered, opts),
        counterfactual=_stat_counterfactual(manifest, ordered),
        t0=t0, t1=t1, notes=tuple(notes), drift=drift, media=tuple(embedded),
    )


_MEDIA_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]{1,64}/[A-Za-z0-9!#$&^_.+-]{1,64}$")


def _safe_media_type(media_type: str) -> str:
    """A media type is UNTRUSTED too — anything not matching a strict token pair is neutralized."""
    mt = str(media_type or "")
    return mt if _MEDIA_TYPE_RE.match(mt) else "application/octet-stream"


# ---------------------------------------------------------------------------
# The drift lane — ground truth, scored by the §5a matcher (never reimplemented)
# ---------------------------------------------------------------------------


def _predictions_from_trace(records: Sequence[TraceRecord]) -> list[Any]:
    """Build the predicted-event timeline the §5a matcher scores.

    Ground truth is annotated on the MEDIA timeline, and ``source_ts`` is that same timeline (rule 1
    of :mod:`percept.tracing`: an incident is only ever reported at ``source_ts``), so it is the
    stamp handed to the matcher. Predictions are the gate's FIREs.

    The event ``key`` is the record's ``goal_id`` — the matcher is KEY-AWARE (a fire of goal A
    inside goal B's window is a false fire, not a match), so a ground-truth ``episode_id`` must be
    the ``goal_id`` of the goal it labels. A record with no ``goal_id`` falls back to a synthetic
    per-record key, which by construction matches no episode.
    """
    from .contracts import JudgmentEvent, JudgmentTiming  # noqa: PLC0415

    events = []
    seq = 0
    for r in records:
        if r.stage != STAGE_GATE_CLASSIFIED:
            continue
        if str(r.data.get("action", "")).lower() != "fire":
            continue
        ts = float(r.source_ts or 0.0)
        events.append(JudgmentEvent(
            key=r.goal_id or f"{r.trace_id}#{r.seq}", evidence_seq=seq, action="fire",
            timing=JudgmentTiming(ts, ts, ts, ts, ts),
        ))
        seq += 1
    return events


def load_golden_episodes(payload: Any) -> list[Any]:
    """Parse ground-truth episodes from JSON (path, text or list of mappings).

    Each entry needs ``episode_id``, ``event_start_ts``, ``decision_eligible_ts``,
    ``episode_end_ts``, ``annotation_version``, ``annotator_or_fixture_id`` and optionally
    ``allowed_lateness_s`` — exactly the fields
    :class:`~percept.harness.golden_eval.GroundTruthEpisode` requires. Nothing is defaulted that the
    contract makes required.

    ``episode_id`` is the JOIN KEY: it must be the ``goal_id`` of the goal the episode labels,
    because the §5a matcher only lets a fire claim an episode whose key it carries.
    """
    from .contracts import GroundTruth  # noqa: PLC0415
    from .harness.golden_eval import GroundTruthEpisode  # noqa: PLC0415

    if isinstance(payload, (str, pathlib.Path)):
        p = pathlib.Path(payload)
        raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else json.loads(str(payload))
    else:
        raw = payload
    if isinstance(raw, Mapping):
        raw = raw.get("episodes") or []
    episodes = []
    for entry in raw:
        try:
            episodes.append(GroundTruthEpisode(
                episode_id=str(entry["episode_id"]),
                ground_truth=GroundTruth(
                    event_start_ts=float(entry["event_start_ts"]),
                    decision_eligible_ts=float(entry["decision_eligible_ts"]),
                    annotation_version=str(entry["annotation_version"]),
                    annotator_or_fixture_id=str(entry["annotator_or_fixture_id"]),
                ),
                episode_end_ts=float(entry["episode_end_ts"]),
                allowed_lateness_s=float(entry.get("allowed_lateness_s", 0.0)),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualizeError(f"invalid ground-truth episode {entry!r}: {exc}") from exc
    return episodes


def _drift(records: Sequence[TraceRecord], golden: Any, opts: VizOptions) -> tuple[Any, Lane]:
    """Score the run against ground truth with the DETERMINISTIC §5a greedy matcher."""
    from .harness.golden_eval import MatchOutcome, score  # noqa: PLC0415

    episodes = golden if _is_episode_seq(golden) else load_golden_episodes(golden)
    predictions = _predictions_from_trace(records)
    result = score(episodes, predictions)
    by_id = {e.episode_id: e for e in episodes}
    items: list[LaneItem] = []
    for ep_id in result.matched_episode_ids:
        ep = by_id[ep_id]
        lo, hi = ep.window
        items.append(LaneItem(lo, hi, "match", f"match: {ep_id}",
                              (("window", f"[{lo:g}, {hi:g}]"),)))
    for ep_id in result.missed_episode_ids:
        ep = by_id[ep_id]
        lo, hi = ep.window
        items.append(LaneItem(lo, hi, "miss", f"MISS: {ep_id}",
                              (("window", f"[{lo:g}, {hi:g}]"),
                               ("why", "no fire landed inside this episode's matching window"))))
    for ps in result.prediction_scores:
        if ps.outcome is MatchOutcome.MATCH:
            continue
        kind = "duplicate" if ps.outcome is MatchOutcome.DUPLICATE else "false_fire"
        items.append(LaneItem(ps.decision_emitted_ts, ps.decision_emitted_ts, kind, kind,
                              (("matched_episode_id", ps.matched_episode_id),)))
    items.sort(key=lambda i: (i.t0, i.kind))
    note = (
        f"recall {result.recall:.2f} · precision {result.precision:.2f} · "
        f"{result.true_positives} matched, {result.misses} missed, {result.false_fires} false "
        "(percept.harness.golden_eval.score — the deterministic §5a greedy matcher)"
    )
    # 2am-debuggability: the matcher is key-aware, so mislabeled ground truth reads as a total
    # miss. Say so out loud instead of showing a silent recall 0.00.
    fired_keys = {str(p.key) for p in predictions}
    if predictions and episodes and not (fired_keys & set(by_id)):
        note += (
            " · NOTE: no fired goal_id matches any episode_id "
            f"(fired {sorted(fired_keys)!r} vs episodes {sorted(by_id)!r}) — the matcher is "
            "key-aware, so label each episode with the goal_id it belongs to"
        )
    return result, Lane("drift", "ground-truth drift", note, tuple(items[: opts.max_lane_items]))


def _is_episode_seq(obj: Any) -> bool:
    return (
        isinstance(obj, (list, tuple))
        and bool(obj)
        and all(hasattr(o, "episode_id") and hasattr(o, "ground_truth") for o in obj)
    )


# ---------------------------------------------------------------------------
# Rendering — server-side escaped HTML; NO trace data ever reaches JavaScript
# ---------------------------------------------------------------------------

#: The ENTIRE inline script. It is a fixed constant: it selects/reveals pre-rendered panels and
#: nothing else. No trace-derived string is ever interpolated into it, which removes the
#: JSON-in-script injection class completely (see the module docstring).
_SCRIPT = """
(function(){
  var items = Array.prototype.slice.call(document.querySelectorAll('[data-firing]'));
  var panels = {};
  Array.prototype.slice.call(document.querySelectorAll('[data-chain-for]')).forEach(function(p){
    panels[p.getAttribute('data-chain-for')] = p;
  });
  var empty = document.getElementById('chain-empty');
  var cur = -1;
  function select(i){
    if (!items.length) { return; }
    if (i < 0) { i = 0; }
    if (i >= items.length) { i = items.length - 1; }
    cur = i;
    items.forEach(function(el, k){
      var on = (k === i);
      el.className = on ? el.className.replace(/ sel\\b/g, '') + ' sel'
                        : el.className.replace(/ sel\\b/g, '');
      el.setAttribute('aria-selected', on ? 'true' : 'false');
      el.setAttribute('tabindex', on ? '0' : '-1');
    });
    var want = items[i].getAttribute('data-firing');
    Object.keys(panels).forEach(function(k){ panels[k].hidden = (k !== want); });
    if (empty) { empty.hidden = true; }
    items[i].focus();
    if (items[i].scrollIntoView) { items[i].scrollIntoView({block: 'nearest', inline: 'center'}); }
  }
  items.forEach(function(el, k){
    el.addEventListener('click', function(){ select(k); });
    el.addEventListener('keydown', function(e){
      if (e.key === 'Enter' || e.key === ' ') { select(k); e.preventDefault(); }
    });
  });
  document.addEventListener('keydown', function(e){
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) { return; }
    if (e.key === 'ArrowRight' || e.key === 'j') { select(cur + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'k') { select(cur - 1); e.preventDefault(); }
    else if (e.key === 'Home') { select(0); e.preventDefault(); }
    else if (e.key === 'End') { select(items.length - 1); e.preventDefault(); }
  });
})();
"""

_STYLE = """
:root{color-scheme:light dark;--bg:#fbfbfd;--fg:#16181d;--mut:#5d6470;--line:#dfe3ea;--card:#fff;
--fire:#c2410c;--ask:#a16207;--silent:#64748b;--sup:#7c3aed;--adm:#0e7490;--drop:#cbd5e1;
--bad:#b91c1c;--ok:#15803d;--inc:#be123c;}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--fg:#e6e9ef;--mut:#9aa3b2;--line:#242a35;
--card:#151a22;--drop:#2b3341;--silent:#7c8899;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);overflow-x:hidden;
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 64px}
h1{font-size:19px;margin:0 0 2px}h2{font-size:14px;margin:26px 0 8px;letter-spacing:.04em;
text-transform:uppercase;color:var(--mut)}
.sub{color:var(--mut);font-size:12px;margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.kv{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--card)}
.kv b{display:block;font-size:17px;font-weight:600}
.kv span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.scroller{overflow-x:auto;overflow-y:hidden;border:1px solid var(--line);border-radius:10px;
background:var(--card)}
.track{min-width:900px;padding:6px 0}
.lane{display:grid;grid-template-columns:190px 1fr;align-items:center;border-top:1px solid var(--line)}
.lane:first-child{border-top:0}
.lane .name{padding:6px 10px;font-size:12px}
.lane .name em{display:block;color:var(--mut);font-style:normal;font-size:11px}
.strip{position:relative;height:36px;margin:4px 10px 4px 0;border-radius:6px;
background:linear-gradient(90deg,rgba(127,127,127,.10),rgba(127,127,127,.10))}
.strip.empty:after{content:attr(data-empty);position:absolute;left:8px;top:9px;font-size:11px;
color:var(--mut)}
.mark{position:absolute;top:6px;height:24px;min-width:3px;border-radius:3px;background:var(--silent);
border:0;padding:0;cursor:default}
.mark.dropped{background:var(--drop);top:15px;height:6px}
.mark.admitted,.mark.frame{background:var(--adm)}
.mark.judge{background:#2563eb}.mark.degraded{background:var(--bad)}
.mark.fire{background:var(--fire)}.mark.ask{background:var(--ask)}.mark.silent{background:var(--silent)}
.mark.suppressed{background:var(--sup)}.mark.incident{background:var(--inc)}
.mark.match{background:var(--ok);opacity:.55}.mark.miss{background:var(--bad);opacity:.55}
.mark.false_fire,.mark.duplicate{background:var(--bad)}
.mark.entity{background:#0891b2;opacity:.4;top:14px;height:8px}
.mark.cost{background:#a16207}.mark.revision{background:#7c3aed}
.mark.unfinished{background:repeating-linear-gradient(45deg,var(--bad),var(--bad)3px,transparent 3px,transparent 6px)}
button.mark{cursor:pointer}
button.mark.sel{outline:2px solid var(--fg);outline-offset:1px;z-index:2}
button.mark:focus-visible{outline:2px solid var(--fg);outline-offset:1px}
.axis{display:flex;justify-content:space-between;color:var(--mut);font-size:11px;padding:2px 10px 6px 190px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);font-size:12.5px}
th{color:var(--mut);font-weight:600}
td.n,th.n{text-align:right}
tr.active td{font-weight:600}
.tag{display:inline-block;padding:1px 6px;border-radius:999px;font-size:11px;border:1px solid var(--line)}
.bad{color:var(--bad)}.ok{color:var(--ok)}.mut{color:var(--mut)}
.chain h3{font-size:13px;margin:0 0 6px}
.stage{border-left:3px solid var(--line);padding:2px 0 2px 10px;margin:0 0 10px}
.stage b{font-size:12px}
dl{margin:2px 0 0;display:grid;grid-template-columns:minmax(120px,auto) 1fr;gap:1px 10px;font-size:12px}
dt{color:var(--mut)}dd{margin:0;word-break:break-word;overflow-wrap:anywhere}
pre{white-space:pre-wrap;word-break:break-word;margin:0}
footer{color:var(--mut);font-size:11.5px;margin-top:26px;border-top:1px solid var(--line);padding-top:10px}
ul.notes{margin:6px 0;padding-left:18px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
figure{margin:0 0 12px}figcaption{color:var(--mut);font-size:11.5px}
"""


def _b64_sha256(text: str) -> str:
    return base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii")


def _dl(pairs: Iterable[tuple[str, Any]], limit: int) -> str:
    out = ["<dl>"]
    for k, v in pairs:
        out.append(f"<dt>{esc(k, limit=120)}</dt><dd><pre>{esc(v, limit=limit)}</pre></dd>")
    out.append("</dl>")
    return "".join(out)


def _pos(t: float, t0: float, t1: float) -> float:
    span = max(1e-9, t1 - t0)
    return max(0.0, min(100.0, (float(t) - t0) / span * 100.0))


def _render_lane(lane: Lane, t0: float, t1: float) -> str:
    marks: list[str] = []
    for item in lane.items:
        left = _pos(item.t0, t0, t1)
        right = _pos(item.t1, t0, t1)
        width = max(0.35, right - left)
        # The forensic detail rides in the mark's own title: on the admission lane that is the
        # motion trace and the threshold rule that decided this exact frame.
        tip = "\n".join(f"{k}: {v}" for k, v in item.detail[:12])
        title = f"{item.t0:.3f}s · {item.label}" + (f"\n{tip}" if tip else "")
        style = f"left:{_num(left)}%;width:{_num(width)}%"
        cls = f"mark {esc(item.kind, limit=40)}"
        if item.firing_id:
            marks.append(
                f'<button type="button" class="{cls}" style="{style}" '
                f'data-firing="{esc(item.firing_id, limit=40)}" role="tab" aria-selected="false" '
                f'tabindex="-1" title="{esc(title, limit=1200)}">'
                f'<span class="sr">{esc(item.label, limit=120)}</span></button>'
            )
        else:
            marks.append(f'<span class="{cls}" style="{style}" title="{esc(title, limit=1200)}"></span>')
    empty = "" if lane.items else ' empty'
    body = "".join(marks)
    trunc = (
        f' <span class="bad">({lane.truncated} items truncated by the size limit)</span>'
        if lane.truncated else ""
    )
    return (
        f'<div class="lane"><div class="name">{esc(lane.title, limit=80)}'
        f"<em>{esc(lane.note, limit=400)}{trunc}</em></div>"
        f'<div class="strip{empty}" data-empty="nothing on this lane — the decision did not get here">'
        f"{body}</div></div>"
    )


def _render_counterfactual(rows: Sequence[StatCounterfactual], *, per_frame: bool) -> str:
    if not rows:
        return (
            '<p class="mut">This trace does not carry per-frame salience statistics, so the '
            "counterfactual cannot be shown without inventing numbers. Run "
            "<code>percept-doctor</code> (which records all three) to get it.</p>"
        )
    head = (
        "<tr><th>stat</th><th class='n'>value</th><th class='n'>threshold</th>"
        "<th class='n'>value / threshold</th><th>would this frame admit?</th></tr>"
        if per_frame else
        "<tr><th>stat</th><th class='n'>max</th><th class='n'>median</th><th class='n'>mean</th>"
        "<th class='n'>threshold</th><th class='n'>max / threshold</th><th>verdict</th></tr>"
    )
    body = []
    for r in rows:
        cls = ' class="active"' if r.active else ""
        star = " ←&nbsp;active" if r.active else ""
        ratio = "—" if r.ratio is None else f"{r.ratio:.2f}×"
        rcls = "bad" if (r.ratio is not None and r.ratio < 1.0) else "ok"
        if per_frame:
            verdict = (
                "threshold not recorded" if r.threshold is None
                else ("YES — this stat would have admitted" if r.would_admit
                      else "no — under its own cutoff")
            )
            body.append(
                f"<tr{cls}><td>{esc(r.stat, limit=40)}{star}</td>"
                f"<td class='n'>{r.max_value:.4f}</td><td class='n'>{_fmt(r.threshold, '.4f')}</td>"
                f"<td class='n {rcls}'>{ratio}</td><td>{esc(verdict, limit=200)}</td></tr>"
            )
        else:
            body.append(
                f"<tr{cls}><td>{esc(r.stat, limit=40)}{star}</td>"
                f"<td class='n'>{r.max_value:.4f}</td><td class='n'>{r.median_value:.4f}</td>"
                f"<td class='n'>{r.mean_value:.4f}</td>"
                f"<td class='n'>{_fmt(r.threshold, '.4f')}</td>"
                f"<td class='n {rcls}'>{ratio}</td><td>{esc(r.verdict, limit=200)}</td></tr>"
            )
    return f"<table>{head}{''.join(body)}</table>"


def _render_summary(summary: TraceSummary) -> str:
    tiles = [
        ("observed", f"{summary.observed}", summary.observed_unit),
        ("admitted", f"{summary.admitted}", f"union {summary.admitted_evidence_union_s:.2f}s"),
        ("judge started", f"{summary.judge_started}", "calls"),
        ("judge completed", f"{summary.judge_completed}", "calls"),
        ("degraded / dropped", f"{summary.judge_degraded} / {summary.judge_dropped}", "calls"),
        ("incidents", f"{summary.incidents}", "delivered"),
        # "—", never "$0.0000": an unpriced/unreported run must not read as a free one.
        ("cost", summary.cost_text(), summary.tokens_text()),
        ("watch duration", f"{summary.watch_duration_s:.2f}s",
         f"healthy {summary.source_healthy_s:.2f}s"),
        ("latency p50 / p95", f"{_fmt(summary.latency_p50_s)} / {_fmt(summary.latency_p95_s)}", "wall s"),
        ("records", f"{summary.records_retained}", f"{summary.records_dropped} lost"),
    ]
    gate = ", ".join(f"{k}={v}" for k, v in sorted(summary.gate_outcomes.items())) or "—"
    cells = "".join(
        f'<div class="kv"><span>{esc(a, limit=60)}</span><b>{esc(b, limit=60)}</b>'
        f'<span>{esc(c, limit=80)}</span></div>'
        for a, b, c in tiles
    )
    return (
        f'<div class="grid">{cells}</div>'
        f'<p class="sub">gate outcomes: {esc(gate, limit=200)} · savings: '
        f"{esc(summary.savings_text(), limit=400)}</p>"
    )


def _render_chain(chain: DecisionChain) -> str:
    stages = "".join(
        f'<div class="stage"><b>{esc(label, limit=80)}</b>{_dl(pairs, 2000)}</div>'
        for label, pairs in chain.stages
    )
    return (
        f'<div class="chain card" data-chain-for="{esc(chain.firing_id, limit=40)}" hidden>'
        f"<h3>{esc(chain.action.upper(), limit=40)} at "
        f"{esc(_fmt(chain.source_ts), limit=40)}s · goal {esc(chain.goal_id, limit=120)} · "
        f"trace {esc(chain.trace_id, limit=200)}</h3>"
        f'<p class="sub">reason: {esc(chain.reason, limit=400)}</p>'
        f"{stages}"
        "<h3>per-stat counterfactual for THIS frame</h3>"
        f"{_render_counterfactual(chain.counterfactual, per_frame=True)}"
        "</div>"
    )


def _render_manifest(manifest: RunTraceManifest) -> str:
    plan = manifest.plan
    pairs: list[tuple[str, Any]] = [
        ("run_id", manifest.run_id),
        ("source (text only — never fetched)", manifest.source),
        ("driver", manifest.driver),
        ("percept / watch / contract", f"{manifest.percept_version} / {manifest.watch_version} / "
                                       f"{manifest.contract_version}"),
        ("plan_hash", plan.plan_hash),
        ("goals", list(plan.goal_ids)),
        ("plan_mode / revision", f"{plan.plan_mode} / {plan.revision}"),
        ("retention policy", manifest.policy.to_dict()),
        ("baseline", manifest.baseline.description),
        ("gate config", manifest.gate_config or "not captured"),
    ]
    for c in manifest.components:
        pairs.append((f"component[{c.role}]", f"{c.name} {c.version or ''} {c.model or ''}".strip()))
    return _dl(pairs, 600)


def render_html(
    manifest: RunTraceManifest,
    records: Sequence[TraceRecord],
    summary: TraceSummary,
    *,
    options: VizOptions | None = None,
    golden: Any = None,
    media: Sequence[tuple[str, str, bytes]] = (),
) -> str:
    """Render the standalone page. Opens from ``file://`` with ZERO network requests."""
    opts = options or VizOptions()
    model = build_model(manifest, records, summary, options=opts, golden=golden, media=media)

    script_hash = _b64_sha256(_SCRIPT)
    img_src = "data:" if model.media else "'none'"
    csp = (
        "default-src 'none'; "
        "style-src 'unsafe-inline'; "
        f"script-src 'sha256-{script_hash}'; "
        f"img-src {img_src}; "
        "connect-src 'none'; font-src 'none'; frame-src 'none'; object-src 'none'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )

    lanes = "".join(_render_lane(lane, model.t0, model.t1) for lane in model.lanes)
    chains = "".join(_render_chain(c) for c in model.chains)
    chain_empty = (
        '<div class="card" id="chain-empty"><b>No gate decision was ever recorded.</b> '
        "There is no decision chain to open, and that IS the finding: read the lanes above for the "
        "stage where the pipeline stopped, then the counterfactual panel for the knob.</div>"
        if not model.chains else
        '<div class="card" id="chain-empty">Select a gate decision on the lane above '
        "(click, or ← / → / Home / End) to open its decision chain.</div>"
    )
    notes = "".join(f"<li>{esc(n, limit=600)}</li>" for n in model.notes)
    media_block = "".join(
        f'<figure><img src="{uri}" alt="{esc(name, limit=120)}" style="max-width:100%">'
        f"<figcaption>{esc(name, limit=120)} ({esc(media_type, limit=80)})</figcaption></figure>"
        for name, media_type, uri in model.media
    )

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{esc(opts.title, limit=140)} — {esc(manifest.run_id, limit=80)}</title>
<style>{_STYLE}</style></head>
<body><main>
<h1>percept trace — where the decision died</h1>
<p class="sub">run {esc(manifest.run_id, limit=80)} · driver {esc(manifest.driver, limit=60)} ·
visualizer v{VISUALIZER_VERSION} · every number is read from
<code>percept.tracing.TraceSummary</code>, never recomputed here.</p>

<h2>metrics (TraceSummary)</h2>
{_render_summary(model.summary)}

<h2>lanes — one shared media-time (source_ts) axis</h2>
<div class="scroller"><div class="track">{lanes}</div></div>
<div class="axis"><span>{model.t0:.2f}s</span><span>media time (source_ts)</span>
<span>{model.t1:.2f}s</span></div>

<h2>decision chain</h2>
{chain_empty}
{chains}

<h2>per-stat counterfactual — what each statistic would have decided</h2>
<div class="card">
<p class="sub">Each statistic against <b>its own</b> calibrated cutoff — <code>block_max</code> and
<code>p95</code> sit on a higher value scale than the whole-frame <code>mean</code> and are never
compared to the mean's threshold. A peak far below 1.00× is not a close call; it is the knob.</p>
{_render_counterfactual(model.counterfactual, per_frame=False)}
</div>

<h2>run identity</h2>
<div class="card">{_render_manifest(model.manifest)}</div>

{('<h2>media (opt-in)</h2><div class="card">' + media_block + "</div>") if media_block else ""}

<footer>
<ul class="notes">{notes}</ul>
<p>Relations deferred to ADR-006 and deliberately NOT drawn: {esc(", ".join(DEFERRED_TO_ADR_006), limit=300)}.
This page loads no external script, style, font or asset and makes no network request of any kind.
All trace-derived text is escaped as untrusted input.</p>
</footer>
</main>
<script>{_SCRIPT}</script>
</body></html>
"""
    if len(html_text.encode("utf-8")) > opts.max_html_bytes:
        raise VisualizeError(
            f"rendered page is larger than max_html_bytes={opts.max_html_bytes}; "
            "lower max_lane_items / max_chains or turn media off"
        )
    return html_text


# ---------------------------------------------------------------------------
# Loading — a local trace JSON or a portable .ptrace bundle
# ---------------------------------------------------------------------------


def load_visualizable(
    path: Any,
) -> tuple[RunTraceManifest, list[TraceRecord], TraceSummary, list[tuple[str, str, bytes]]]:
    """Load either a trace JSON (:func:`percept.tracing.load_trace`) or a ``.ptrace`` bundle.

    Returns ``(manifest, records, summary, media)``. ``media`` is whatever evidence a portable
    bundle carried; it is only ever EMBEDDED if the caller opts in.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise VisualizeError(f"no such trace: {p}")
    if p.is_dir() or p.suffix in (".ptrace", ".zip") or p.name.endswith(".ptrace.zip"):
        from .trace_bundle import BundleError, TraceBundle  # noqa: PLC0415

        try:
            bundle = TraceBundle.read(p)
        except BundleError as exc:
            raise VisualizeError(str(exc)) from exc
        media = [(e.name, e.media_type, e.data) for e in bundle.evidence]
        return bundle.manifest, list(bundle.records), bundle.summary, media
    try:
        manifest, records, summary = load_trace(p.read_text(encoding="utf-8"))
    except (ValueError, KeyError, TypeError) as exc:
        raise VisualizeError(f"not a percept trace artifact: {p} ({exc})") from exc
    return manifest, records, summary, []


def visualize_trace(
    trace: Any,
    out_path: Any,
    *,
    options: VizOptions | None = None,
    golden: Any = None,
) -> pathlib.Path:
    """Render ``trace`` (a trace JSON path or a ``.ptrace`` bundle) to a standalone HTML file."""
    manifest, records, summary, media = load_visualizable(trace)
    opts = options or VizOptions()
    page = render_html(manifest, records, summary, options=opts, golden=golden, media=media)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


__all__ = [
    "VISUALIZER_VERSION",
    "SALIENCE_STATS",
    "DEFERRED_TO_ADR_006",
    "DecisionChain",
    "Lane",
    "LaneItem",
    "StatCounterfactual",
    "VisualizeError",
    "VizModel",
    "VizOptions",
    "build_model",
    "esc",
    "load_golden_episodes",
    "load_visualizable",
    "render_html",
    "visualize_trace",
]
