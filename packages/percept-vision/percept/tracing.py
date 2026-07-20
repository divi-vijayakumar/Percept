"""The diagnostic data model — ONE definition of observed / admitted / judged / saved.

``percept doctor``, ``percept eval`` and the visualizer must not invent three definitions of the
same funnel. This module is where the definitions live: a :class:`RunTraceManifest` (who ran, with
which plan and which adapters), a stream of :class:`TraceRecord`\\ s (the stable event vocabulary,
with TWO clocks), a :class:`TracePolicy` that bounds what is retained in production, and a
:class:`TraceSummary` whose numbers are *derived* — never re-counted by a caller.

Largely this is **persistence + structure over an existing stream**, not new telemetry. The runtime
already emits :class:`~percept.observability.TraceEvent`\\ s for every frame
(``edge.decision`` with motion/structural/semantic/threshold/escalate/coast_admit at
``runtime/worker.py``, plus ``source.frame`` / ``judge.started`` / ``judge.finished`` /
``judge.degraded`` / ``gate.classified`` / ``incident.created`` from
:meth:`~percept.agent.Percept.perceive_judged`), and ``RuntimeHealthEvent``\\ s for scheduler drops,
gate suppression and capture health. :class:`TraceRecorder` consumes BOTH, unchanged, through
``on_trace=`` / ``on_health=`` — so it works from :func:`percept.run` **and** from a bare
``perceive_judged()`` driver that bypasses the runtime worker entirely (see
:meth:`TraceRecorder.tracer`).

Four rules this module exists to enforce
----------------------------------------

**1. Two clocks, always.** One ``ts`` cannot debug both event truth and runtime latency. Every
record carries ``source_ts`` (the media/event timeline — ``26.2`` means 26.2s into the clip, and it
is the ONLY time an incident is reported at), ``observed_wall_ts`` (when the runtime first saw that
unit of evidence) and ``emitted_wall_ts`` (when this decision was emitted). Processing latency is
``emitted_wall_ts - observed_wall_ts`` — WALL time, never source time. Both are recorded regardless
of source type: for a live camera they may nearly coincide, for a file replay they diverge wildly,
and that divergence is exactly what a 2am debugging session needs to see.

**2. Recording never blocks capture.** :class:`TraceRecorder` is a bounded ring with a synchronous,
allocation-light ingest. When a bound is hit the record is dropped — and the drop is itself
surfaced as a ``trace_dropped`` record inside the retained stream (and on :attr:`TraceRecorder.dropped`
and in :class:`TraceSummary`). Silent loss is the exact bug class this wave exists to kill, so the
drop marker is written by *evicting a retained record*, guaranteeing it is visible even if the
stream is never drained.

**3. Retention is bounded by policy, not by hope.** A per-frame ``edge.decision`` record is right
for a 36s probe and dangerous for a 24/7 deployment. :class:`TracePolicy` names the three levels —
``summary`` (counters only, continuous), ``decisions`` (judge/gate/incident detail, plus a window
of full detail around incidents and health transitions), ``full`` (per-frame, for bounded
diagnostics). **The counters are exact at every level**: retention filters what is KEPT, never what
is COUNTED.

**4. Savings is a call-count identity, not a fabricated number.** The naive "seconds saved x tokens
per call" is dimensionally broken. The only defensible form needs an EXPLICIT declared baseline
(:class:`BaselineAssumption`, carried in the manifest and echoed in the summary)::

    estimated_baseline_calls = declared_baseline_opportunities
    estimated_saved_calls    = max(0, estimated_baseline_calls - actual_judge_calls)
    estimated_saved_tokens   = estimated_saved_calls * mean_input_tokens_per_actual_call

With zero judge calls there is no measured mean, so ``estimated_saved_tokens`` is ``None`` — the
renderer shows ``—``. It is never back-filled from a guess.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

#: Wire-schema version of the trace artifact (manifest + records + summary). Bump when a field is
#: removed or re-meant; additive fields do not bump it.
TRACE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# 1. The stable event vocabulary
# ---------------------------------------------------------------------------

# Emitted by the runtime worker / perceive_judged today (percept.observability.TraceEvent.stage).
STAGE_EDGE_DECISION = "edge.decision"      # EVERY frame: tier-0 motion/structural/semantic/threshold
STAGE_SOURCE_FRAME = "source.frame"        # a frame ADMITTED into perceive_judged
STAGE_JUDGE_STARTED = "judge.started"
STAGE_JUDGE_FINISHED = "judge.finished"
STAGE_JUDGE_DEGRADED = "judge.degraded"
STAGE_GATE_CLASSIFIED = "gate.classified"  # truth/action/reason
STAGE_INCIDENT_CREATED = "incident.created"

# Added by this module (derived from RuntimeHealthEvent, or authored by a driver).
STAGE_RUN_STARTED = "run.started"
STAGE_RUN_ENDED = "run.ended"
STAGE_JUDGE_USAGE = "judge.usage"          # tokens/cost for one actual judge call
STAGE_JUDGE_DROPPED = "judge.dropped"      # superseded / revision-dropped BEFORE any judge call
STAGE_GATE_SUPPRESSED = "gate.suppressed"  # a fire the baseline gate would have delivered, withheld
STAGE_SOURCE_HEALTH = "source.health"      # a capture/scheduler health TRANSITION (degraded/recovered)
STAGE_TRACE_DROPPED = "trace_dropped"      # THIS recorder dropped records (rule 2 — never silent)

#: The closed stage vocabulary. A stage outside this tuple is still recorded (forward-compatible)
#: but is not counted into any funnel field — an unknown stage must never silently move a number.
TRACE_STAGES: tuple[str, ...] = (
    STAGE_RUN_STARTED,
    STAGE_EDGE_DECISION,
    STAGE_SOURCE_FRAME,
    STAGE_JUDGE_STARTED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_USAGE,
    STAGE_JUDGE_DROPPED,
    STAGE_GATE_CLASSIFIED,
    STAGE_GATE_SUPPRESSED,
    STAGE_INCIDENT_CREATED,
    STAGE_SOURCE_HEALTH,
    STAGE_TRACE_DROPPED,
    STAGE_RUN_ENDED,
)

#: The stage that means "the runtime OBSERVED one unit of evidence" (a frame reached the tier-0
#: gate). ``edge.decision`` fires for every frame, escalated or not — that is what makes "why was it
#: silent" answerable.
OBSERVATION_STAGE = STAGE_EDGE_DECISION

#: The stage that means "this unit of evidence was ADMITTED" (it reached ``perceive_judged`` — a
#: cloud call became possible). In the worker this is emitted strictly after the escalate check; in
#: a direct ``perceive_judged()`` driver every call is by definition an admission, so a run with no
#: ``edge.decision`` at all falls back to counting admissions as observations (see
#: :meth:`TraceSummary._observed_fallback`).
ADMISSION_STAGE = STAGE_SOURCE_FRAME

#: Health-event kinds that OPEN an unhealthy interval (degrade-loud: latched until a recovery).
DEGRADED_HEALTH_KINDS: frozenset[str] = frozenset({
    "judge_scheduler_stalled",
    "judge_scheduler_backpressure",
    "capture_plan_unmet",
    "runtime_error",
    "audio_stale",
})

#: Health-event kinds that CLOSE an unhealthy interval.
RECOVERED_HEALTH_KINDS: frozenset[str] = frozenset({"judge_scheduler_recovered"})

#: Health-event kinds that mean a candidate was dropped before any judge call was spent.
JUDGE_DROP_HEALTH_KINDS: frozenset[str] = frozenset({
    "judge_superseded",
    "judge_revision_dropped",
})

#: Gate actions counted in :attr:`TraceSummary.gate_outcomes`. ``fire``/``ask``/``silent`` come from
#: ``gate.classified``; ``suppressed`` is the O2 typed suppression (start_anchored / refractory) and
#: is NOT one of the three gate actions — a suppressed decision is a silent that the baseline gate
#: WOULD have delivered, which is a different fact and gets its own bucket.
GATE_OUTCOMES: tuple[str, ...] = ("fire", "ask", "silent", "suppressed")


# ---------------------------------------------------------------------------
# 2. Identity — plan, adapters, models
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Deterministic JSON (sorted keys, tight separators) — the input to every hash here."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload: Any) -> str:
    """``sha256`` hex digest of :func:`canonical_json` of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short_hash(payload: Any, length: int = 12) -> str:
    """A short, human-quotable prefix of :func:`content_hash` (for logs and UI chips)."""
    return content_hash(payload)[:length]


@dataclass(frozen=True)
class ComponentIdentity:
    """WHICH implementation produced these numbers.

    ``role`` is the seam (``"vision"``, ``"judge"``, ``"gate"``, ``"profile"``); ``name`` is the
    provider/adapter identity (card RB-PROVIDER: the NAME is the identity); ``model`` is
    configuration carried to it (``"claude-sonnet-4-5"``), never the adapter's identity; ``version``
    is the adapter's own version string; ``config_hash`` content-addresses whatever knobs were bound.
    """

    role: str
    name: str
    version: str | None = None
    model: str | None = None
    config_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role, "name": self.name, "version": self.version,
            "model": self.model, "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentIdentity":
        return cls(
            role=str(payload["role"]), name=str(payload["name"]),
            version=payload.get("version"), model=payload.get("model"),
            config_hash=payload.get("config_hash"),
        )


@dataclass(frozen=True)
class PlanIdentity:
    """WHAT was being watched: the goal set, content-addressed.

    ``goal_ids`` is the ordered active set; ``plan_hash`` addresses the full plan payload (conditions
    included) so two runs that quote the same ``plan_hash`` provably watched for the same thing;
    ``revision`` is the live :class:`~percept.goals.GoalRegistry` revision the run started at
    (goal edits move it mid-run, and a judged frame is always pinned to the revision it was admitted
    under).
    """

    goal_ids: tuple[str, ...] = ()
    plan_hash: str | None = None
    revision: int | None = None
    plan_mode: str | None = None

    @classmethod
    def from_goals(cls, goals: Iterable[Any], *, revision: int | None = None,
                   plan_mode: str | None = None) -> "PlanIdentity":
        """Build from anything with ``.id`` / ``.condition`` (a :class:`~percept.adapters.Goal`)."""
        items = list(goals)
        payload = [
            {
                "id": str(getattr(g, "id", "")),
                "condition": str(getattr(g, "condition", "")),
                "say": str(getattr(g, "say", "")),
                "modality": getattr(g, "modality", None),
            }
            for g in items
        ]
        return cls(
            goal_ids=tuple(str(getattr(g, "id", "")) for g in items),
            plan_hash=content_hash(payload),
            revision=revision,
            plan_mode=plan_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_ids": list(self.goal_ids), "plan_hash": self.plan_hash,
            "revision": self.revision, "plan_mode": self.plan_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanIdentity":
        return cls(
            goal_ids=tuple(payload.get("goal_ids") or ()),
            plan_hash=payload.get("plan_hash"),
            revision=payload.get("revision"),
            plan_mode=payload.get("plan_mode"),
        )


@dataclass(frozen=True)
class BaselineAssumption:
    """The EXPLICIT counterfactual savings is measured against — no baseline, no savings number.

    ``opportunities`` is the number of judge calls the declared baseline strategy WOULD have made
    over this same run (e.g. "judge every frame" over 144 frames -> 144). ``kind`` names the strategy
    and ``description`` states it in words, because a savings figure whose assumption is not quoted
    alongside it is marketing, not measurement. ``None`` opportunities = no baseline declared =
    savings is ``None`` everywhere downstream.
    """

    kind: str = "undeclared"
    opportunities: int | None = None
    description: str = "No baseline declared; savings is not computed."

    @classmethod
    def every_observation(cls, observations: int, *, unit: str = "frame") -> "BaselineAssumption":
        """The usual comparison: one judge call per OBSERVED unit (the ungated baseline)."""
        return cls(
            kind="every_observation",
            opportunities=int(observations),
            description=f"Baseline = one judge call per observed {unit} ({int(observations)} {unit}s).",
        )

    @classmethod
    def fixed_interval(cls, duration_s: float, interval_s: float) -> "BaselineAssumption":
        """A polling baseline: one judge call every ``interval_s`` for ``duration_s``."""
        n = 0 if interval_s <= 0 else int(duration_s // interval_s)
        return cls(
            kind="fixed_interval",
            opportunities=n,
            description=f"Baseline = one judge call every {interval_s}s over {duration_s}s ({n} calls).",
        )

    @property
    def declared(self) -> bool:
        return self.opportunities is not None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "opportunities": self.opportunities, "description": self.description}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineAssumption":
        return cls(
            kind=str(payload.get("kind", "undeclared")),
            opportunities=payload.get("opportunities"),
            description=str(payload.get("description", "")),
        )


# ---------------------------------------------------------------------------
# 3. TracePolicy — bounding production
# ---------------------------------------------------------------------------

TraceLevel = Literal["summary", "decisions", "full"]

#: Stages retained at ``level="summary"``: run lifecycle, health transitions, incidents, and the
#: drop marker. Everything else is COUNTED but not KEPT — the continuous production posture.
_SUMMARY_STAGES: frozenset[str] = frozenset({
    STAGE_RUN_STARTED, STAGE_RUN_ENDED, STAGE_SOURCE_HEALTH,
    STAGE_INCIDENT_CREATED, STAGE_TRACE_DROPPED,
})

#: Additional stages retained at ``level="decisions"``: the judge/gate decision detail. The per-frame
#: ``edge.decision`` is deliberately NOT here — that is the ``full`` level.
_DECISION_STAGES: frozenset[str] = _SUMMARY_STAGES | frozenset({
    STAGE_SOURCE_FRAME, STAGE_JUDGE_STARTED, STAGE_JUDGE_FINISHED, STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_USAGE, STAGE_JUDGE_DROPPED, STAGE_GATE_CLASSIFIED, STAGE_GATE_SUPPRESSED,
})


@dataclass(frozen=True)
class TracePolicy:
    """What a run is allowed to RETAIN. Counters are always exact; this bounds storage only.

    - ``level="summary"`` — continuous production. Lifecycle, health transitions and incidents are
      kept; per-frame and per-judge records are counted and discarded.
    - ``level="decisions"`` — judge/gate detail kept, per-frame ``edge.decision`` discarded.
    - ``level="full"`` — everything, including a record per frame. Correct for a 36s probe,
      **dangerous for 24/7**: always pair it with ``max_duration_s`` / ``max_bytes``.

    ``incident_window_s`` gives production the third posture the card asks for: full detail is
    retained for records whose ``source_ts`` falls within this many seconds of an incident or a
    health transition, even at ``summary``/``decisions`` level (see
    :meth:`TraceRecorder.mark_incident`). ``max_duration_s`` caps retention by SOURCE time from the
    first record; ``max_bytes`` caps it by approximate serialized size; ``include_media`` gates
    frames/snippets ever being embedded (default off — a trace must be safe to hand to support).
    """

    level: TraceLevel = "decisions"
    max_duration_s: float | None = None
    max_bytes: int = 8_000_000
    include_media: bool = False
    incident_window_s: float = 10.0

    def __post_init__(self) -> None:
        if self.level not in ("summary", "decisions", "full"):
            raise ValueError(f"level must be summary/decisions/full, not {self.level!r}")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive or None")
        if self.incident_window_s < 0:
            raise ValueError("incident_window_s must be >= 0")

    # -- the three documented postures ------------------------------------
    @classmethod
    def diagnostic(cls, max_duration_s: float = 300.0) -> "TracePolicy":
        """Bounded per-frame diagnostics — what ``percept doctor`` runs."""
        return cls(level="full", max_duration_s=max_duration_s, max_bytes=32_000_000)

    @classmethod
    def production(cls) -> "TracePolicy":
        """Continuous summaries + full detail around incidents/health transitions."""
        return cls(level="summary", max_duration_s=None, max_bytes=4_000_000, incident_window_s=10.0)

    @classmethod
    def evaluation(cls) -> "TracePolicy":
        """Decision-level detail for ``percept eval`` — every judge and gate call, no per-frame rows."""
        return cls(level="decisions", max_duration_s=None, max_bytes=16_000_000)

    def retains(self, stage: str) -> bool:
        """Whether ``stage`` is retained at this level, ignoring the incident window."""
        if self.level == "full":
            return True
        if self.level == "decisions":
            return stage in _DECISION_STAGES
        return stage in _SUMMARY_STAGES

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level, "max_duration_s": self.max_duration_s, "max_bytes": self.max_bytes,
            "include_media": self.include_media, "incident_window_s": self.incident_window_s,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TracePolicy":
        return cls(
            level=payload.get("level", "decisions"),  # type: ignore[arg-type]
            max_duration_s=payload.get("max_duration_s"),
            max_bytes=int(payload.get("max_bytes", 8_000_000)),
            include_media=bool(payload.get("include_media", False)),
            incident_window_s=float(payload.get("incident_window_s", 10.0)),
        )


# ---------------------------------------------------------------------------
# 4. The manifest + the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunTraceManifest:
    """WHO ran, WITH WHAT, and UNDER WHICH assumptions — the header of every trace artifact.

    Everything a reader needs to decide whether two runs are comparable: the run id (the correlation
    root every ``trace_id`` hangs off), the source, the contract/entrypoint versions, the plan
    identity, the adapter+model identities, the retention policy, and the declared savings baseline.
    ``started_source_ts`` and ``started_wall_ts`` are the two clocks' origins.
    """

    run_id: str
    source: str = "camera"
    schema_version: int = TRACE_SCHEMA_VERSION
    percept_version: str | None = None
    watch_version: str | None = None
    contract_version: str | None = None
    started_wall_ts: float = 0.0
    started_source_ts: float | None = None
    driver: str = "run"            # "run" (runtime worker) | "perceive_judged" | anything a client names
    plan: PlanIdentity = field(default_factory=PlanIdentity)
    components: tuple[ComponentIdentity, ...] = ()
    profile_hash: str | None = None
    policy: TracePolicy = field(default_factory=TracePolicy)
    baseline: BaselineAssumption = field(default_factory=BaselineAssumption)
    #: Nominal evidence-window length for an admission that does not declare its own window. Used
    #: ONLY as a fallback, and ``0.0`` means "an admission is an instant" — the union then measures
    #: exactly the declared windows and nothing is invented.
    evidence_window_s: float = 0.0
    #: The Gate's decision knobs at run start (``threshold`` / ``fresh_min`` /
    #: ``require_false_first`` / ``refractory_s``). Additive, and load-bearing for card D8b:
    #: :mod:`percept.replay` re-runs recorded verdicts through a Gate rebuilt from EXACTLY these
    #: numbers, so a cognition replay is not silently done against today's defaults. Empty = not
    #: captured, and the replayer says so instead of guessing (see
    #: :attr:`percept.replay.ReplayReport.gate_config_recorded`).
    gate_config: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source: str = "camera",
        driver: str = "run",
        goals: Iterable[Any] | None = None,
        plan: PlanIdentity | None = None,
        components: Sequence[ComponentIdentity] = (),
        policy: TracePolicy | None = None,
        baseline: BaselineAssumption | None = None,
        profile_hash: str | None = None,
        evidence_window_s: float = 0.0,
        started_source_ts: float | None = None,
        wall_clock: Callable[[], float] = time.time,
        run_id: str | None = None,
        gate_config: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> "RunTraceManifest":
        """Build a manifest, filling versions from the installed package (never hand-synced)."""
        percept_version = watch_version = contract_version = None
        try:  # local imports: tracing must stay importable from inside percept/__init__'s graph
            from . import __version__ as _pv  # noqa: PLC0415
            percept_version = _pv
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            from .watch import WATCH_VERSION as _wv  # noqa: PLC0415
            watch_version = _wv
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            from .contracts import CONTRACT_VERSION as _cv  # noqa: PLC0415
            contract_version = _cv
        except Exception:  # pragma: no cover - defensive
            pass
        return cls(
            run_id=run_id or uuid.uuid4().hex[:16],
            source=source,
            percept_version=percept_version,
            watch_version=watch_version,
            contract_version=contract_version,
            started_wall_ts=float(wall_clock()),
            started_source_ts=started_source_ts,
            driver=driver,
            plan=plan if plan is not None else (
                PlanIdentity.from_goals(goals) if goals is not None else PlanIdentity()
            ),
            components=tuple(components),
            profile_hash=profile_hash,
            policy=policy if policy is not None else TracePolicy(),
            baseline=baseline if baseline is not None else BaselineAssumption(),
            evidence_window_s=float(evidence_window_s),
            gate_config=dict(gate_config or {}),
            labels=dict(labels or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "schema_version": self.schema_version,
            "percept_version": self.percept_version,
            "watch_version": self.watch_version,
            "contract_version": self.contract_version,
            "started_wall_ts": self.started_wall_ts,
            "started_source_ts": self.started_source_ts,
            "driver": self.driver,
            "plan": self.plan.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "profile_hash": self.profile_hash,
            "policy": self.policy.to_dict(),
            "baseline": self.baseline.to_dict(),
            "evidence_window_s": self.evidence_window_s,
            "gate_config": dict(self.gate_config),
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunTraceManifest":
        return cls(
            run_id=str(payload["run_id"]),
            source=str(payload.get("source", "camera")),
            schema_version=int(payload.get("schema_version", TRACE_SCHEMA_VERSION)),
            percept_version=payload.get("percept_version"),
            watch_version=payload.get("watch_version"),
            contract_version=payload.get("contract_version"),
            started_wall_ts=float(payload.get("started_wall_ts", 0.0)),
            started_source_ts=payload.get("started_source_ts"),
            driver=str(payload.get("driver", "run")),
            plan=PlanIdentity.from_dict(payload.get("plan") or {}),
            components=tuple(ComponentIdentity.from_dict(c) for c in payload.get("components") or ()),
            profile_hash=payload.get("profile_hash"),
            policy=TracePolicy.from_dict(payload.get("policy") or {}),
            baseline=BaselineAssumption.from_dict(payload.get("baseline") or {}),
            evidence_window_s=float(payload.get("evidence_window_s", 0.0)),
            gate_config=dict(payload.get("gate_config") or {}),
            labels=dict(payload.get("labels") or {}),
        )


@dataclass(frozen=True)
class TraceRecord:
    """One structured event on the run's timeline, carrying BOTH clocks.

    - ``source_ts`` — the media/event timeline. ``26.2`` = 26.2s into the clip. **Incident event
      time is always this.**
    - ``observed_wall_ts`` — when the runtime first saw this unit of evidence (inherited from the
      first record of the same ``trace_id``, so a downstream stage measures from the frame's arrival,
      not from its own).
    - ``emitted_wall_ts`` — when THIS decision was emitted. :attr:`latency_s` is the difference:
      real processing latency, in wall time, never source time.

    ``trace_id`` correlates every stage of one unit of evidence (the runtime already shapes it as
    ``"{source}@{t}"``); ``seq`` is the recorder's monotonic ingest order, which survives ties when
    two records share a ``source_ts``.
    """

    seq: int
    trace_id: str
    stage: str
    source_ts: float | None = None
    observed_wall_ts: float | None = None
    emitted_wall_ts: float | None = None
    goal_id: str | None = None
    source: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_s(self) -> float | None:
        """Real processing latency for this decision — WALL time. ``None`` if a clock is missing."""
        if self.observed_wall_ts is None or self.emitted_wall_ts is None:
            return None
        return self.emitted_wall_ts - self.observed_wall_ts

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "trace_id": self.trace_id, "stage": self.stage,
            "source_ts": self.source_ts, "observed_wall_ts": self.observed_wall_ts,
            "emitted_wall_ts": self.emitted_wall_ts, "goal_id": self.goal_id,
            "source": self.source, "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceRecord":
        return cls(
            seq=int(payload.get("seq", 0)),
            trace_id=str(payload.get("trace_id", "")),
            stage=str(payload["stage"]),
            source_ts=payload.get("source_ts"),
            observed_wall_ts=payload.get("observed_wall_ts"),
            emitted_wall_ts=payload.get("emitted_wall_ts"),
            goal_id=payload.get("goal_id"),
            source=payload.get("source"),
            data=dict(payload.get("data") or {}),
        )


# ---------------------------------------------------------------------------
# 5. The funnel — derived ONCE, here
# ---------------------------------------------------------------------------


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Total length of the UNION of ``intervals`` — overlapping evidence windows counted once.

    This is why :attr:`TraceSummary.admitted_evidence_union_s` is not ``count x nominal_interval``:
    a 2s window admitted 10 times over a 3s span covers ~3 seconds of the source, not 20.
    """
    spans = sorted((float(a), float(b)) for a, b in intervals if b >= a)
    total = 0.0
    cur_a: float | None = None
    cur_b = 0.0
    for a, b in spans:
        if cur_a is None:
            cur_a, cur_b = a, b
            continue
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        elif b > cur_b:
            cur_b = b
    if cur_a is not None:
        total += cur_b - cur_a
    return total


@dataclass(frozen=True)
class TraceSummary:
    """The derived funnel — the ONE place ``doctor``, ``eval`` and the visualizer get their numbers.

    ``observed`` -> ``admitted`` -> ``judge_started`` -> ``judge_completed`` -> gate outcomes ->
    ``incidents``, plus cost and (only with a declared baseline) savings.

    Two definitions worth stating out loud:

    - ``admitted`` carries BOTH a count and ``admitted_evidence_union_s`` — the union of the
      evidence windows those admissions covered. The count times a nominal interval is *not* a
      duration and is not reported here.
    - ``estimated_saved_tokens`` is ``None`` — rendered ``—`` — whenever there is no declared
      baseline or zero actual judge calls. There is no mean input-token figure to multiply by, so
      there is no number to print.
    """

    run_id: str
    watch_duration_s: float = 0.0
    source_healthy_s: float = 0.0
    observed: int = 0
    observed_unit: str = "frames"
    admitted: int = 0
    admitted_evidence_union_s: float = 0.0
    judge_started: int = 0
    judge_completed: int = 0
    judge_degraded: int = 0
    judge_dropped: int = 0
    judge_superseded: int = 0
    gate_outcomes: dict[str, int] = field(default_factory=dict)
    incidents: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cost_usd: float = 0.0
    usage_reported_calls: int = 0
    records_retained: int = 0
    records_dropped: int = 0
    baseline: BaselineAssumption = field(default_factory=BaselineAssumption)
    estimated_baseline_calls: int | None = None
    estimated_saved_calls: int | None = None
    estimated_saved_tokens: float | None = None
    mean_input_tokens_per_call: float | None = None
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None

    # -- derivation -------------------------------------------------------
    @classmethod
    def derive(
        cls,
        manifest: RunTraceManifest,
        records: Sequence[TraceRecord],
        *,
        counters: Mapping[str, int] | None = None,
        usage: Mapping[str, float] | None = None,
        evidence_windows: Sequence[tuple[float, float]] = (),
        unhealthy_s: float = 0.0,
        watch_duration_s: float | None = None,
        records_dropped: int = 0,
        observed_unit: str = "frames",
    ) -> "TraceSummary":
        """Derive the funnel.

        ``counters``/``usage``/``evidence_windows``/``unhealthy_s`` are the recorder's EXACT running
        aggregates (folded for every ingested record, whatever the retention level). When they are
        omitted — deriving from a trace file, where only retained records survive — the funnel is
        recounted from ``records``, which is exact at ``level="full"`` and a floor otherwise.
        """
        if counters is None:
            counters, usage, evidence_windows, unhealthy_s, span_s = _recount(records, manifest)
            if watch_duration_s is None:
                watch_duration_s = span_s
        usage = usage or {}
        c = counters

        observed = int(c.get("observed", 0))
        admitted = int(c.get("admitted", 0))
        if observed == 0:  # a direct perceive_judged() driver never emits edge.decision
            observed = admitted
        judge_completed = int(c.get("judge_completed", 0))

        if watch_duration_s is None:
            watch_duration_s = 0.0
        healthy = max(0.0, float(watch_duration_s) - float(unhealthy_s))

        actual_calls = judge_completed or int(c.get("judge_started", 0))
        usage_calls = int(usage.get("calls", 0))
        input_tokens = float(usage.get("input_tokens", 0.0))
        mean_in: float | None = (input_tokens / usage_calls) if usage_calls > 0 else None

        baseline = manifest.baseline
        baseline_calls: int | None = baseline.opportunities
        saved_calls: int | None = None
        saved_tokens: float | None = None
        if baseline_calls is not None:
            saved_calls = max(0, int(baseline_calls) - int(actual_calls))
            # SAVINGS IS NEVER FABRICATED: with no actual call there is no measured mean input-token
            # cost to project onto the avoided calls, so the token figure stays None ("—").
            if actual_calls > 0 and mean_in is not None:
                saved_tokens = saved_calls * mean_in

        lat = sorted(x for x in (r.latency_s for r in records) if x is not None)

        return cls(
            run_id=manifest.run_id,
            watch_duration_s=float(watch_duration_s),
            source_healthy_s=healthy,
            observed=observed,
            observed_unit=observed_unit,
            admitted=admitted,
            admitted_evidence_union_s=union_seconds(evidence_windows),
            judge_started=int(c.get("judge_started", 0)),
            judge_completed=judge_completed,
            judge_degraded=int(c.get("judge_degraded", 0)),
            judge_dropped=int(c.get("judge_dropped", 0)),
            judge_superseded=int(c.get("judge_superseded", 0)),
            gate_outcomes={k: int(c.get(f"gate.{k}", 0)) for k in GATE_OUTCOMES},
            incidents=int(c.get("incidents", 0)),
            input_tokens=input_tokens,
            output_tokens=float(usage.get("output_tokens", 0.0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
            usage_reported_calls=usage_calls,
            records_retained=len(records),
            records_dropped=int(records_dropped or c.get("records_dropped", 0)),
            baseline=baseline,
            estimated_baseline_calls=baseline_calls,
            estimated_saved_calls=saved_calls,
            estimated_saved_tokens=saved_tokens,
            mean_input_tokens_per_call=mean_in,
            latency_p50_s=_percentile(lat, 0.50),
            latency_p95_s=_percentile(lat, 0.95),
        )

    def savings_text(self) -> str:
        """The renderer-safe savings line — ``"—"`` when there is nothing honest to say."""
        if self.estimated_saved_tokens is None or self.estimated_saved_calls is None:
            return "—"
        return (
            f"{self.estimated_saved_calls} calls / ~{self.estimated_saved_tokens:,.0f} input tokens "
            f"({self.baseline.description})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "watch_duration_s": self.watch_duration_s,
            "source_healthy_s": self.source_healthy_s,
            "observed": self.observed,
            "observed_unit": self.observed_unit,
            "admitted": self.admitted,
            "admitted_evidence_union_s": self.admitted_evidence_union_s,
            "judge_started": self.judge_started,
            "judge_completed": self.judge_completed,
            "judge_degraded": self.judge_degraded,
            "judge_dropped": self.judge_dropped,
            "judge_superseded": self.judge_superseded,
            "gate_outcomes": dict(self.gate_outcomes),
            "incidents": self.incidents,
            "cost": {
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "usd": self.cost_usd, "usage_reported_calls": self.usage_reported_calls,
            },
            "records": {"retained": self.records_retained, "dropped": self.records_dropped},
            "savings": {
                "baseline": self.baseline.to_dict(),
                "estimated_baseline_calls": self.estimated_baseline_calls,
                "estimated_saved_calls": self.estimated_saved_calls,
                "estimated_saved_tokens": self.estimated_saved_tokens,
                "mean_input_tokens_per_call": self.mean_input_tokens_per_call,
                "text": self.savings_text(),
            },
            "latency": {"p50_s": self.latency_p50_s, "p95_s": self.latency_p95_s},
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def _recount(
    records: Sequence[TraceRecord], manifest: RunTraceManifest
) -> tuple[dict[str, int], dict[str, float], list[tuple[float, float]], float, float]:
    """Recount the funnel from retained records alone (loading a trace file back)."""
    fold = _Aggregate(manifest)
    for r in records:
        fold.fold(r)
    fold.close(fold.last_source_ts)
    return dict(fold.counters), dict(fold.usage), list(fold.windows), fold.unhealthy_s, fold.span_s


class _Aggregate:
    """The EXACT running funnel. Folded for every ingested record, retained or not — which is what
    makes ``level="summary"`` a storage decision rather than a measurement one."""

    def __init__(self, manifest: RunTraceManifest) -> None:
        self._m = manifest
        self.counters: Counter[str] = Counter()
        self.usage: dict[str, float] = {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
        self.windows: list[tuple[float, float]] = []
        self.unhealthy_s = 0.0
        self._degraded_at: float | None = None
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._seen_observed: set[str] = set()
        self._seen_admitted: set[str] = set()

    # counters are ints; usage is float — kept apart so a token count can't land in a funnel field
    def fold(self, r: TraceRecord) -> None:
        stage, ts = r.stage, r.source_ts
        if ts is not None:
            self._first_ts = ts if self._first_ts is None else min(self._first_ts, ts)
            self._last_ts = ts if self._last_ts is None else max(self._last_ts, ts)
        if stage == OBSERVATION_STAGE:
            if r.trace_id not in self._seen_observed:  # one unit of evidence, however many records
                self._seen_observed.add(r.trace_id)
                self.counters["observed"] += 1
        elif stage == ADMISSION_STAGE:
            if r.trace_id not in self._seen_admitted:
                self._seen_admitted.add(r.trace_id)
                self.counters["admitted"] += 1
                self.windows.append(self._window_for(r))
        elif stage == STAGE_JUDGE_STARTED:
            self.counters["judge_started"] += 1
        elif stage == STAGE_JUDGE_FINISHED:
            self.counters["judge_completed"] += 1
        elif stage == STAGE_JUDGE_DEGRADED:
            self.counters["judge_degraded"] += 1
        elif stage == STAGE_JUDGE_DROPPED:
            self.counters["judge_dropped"] += 1
            if r.data.get("kind") == "judge_superseded" or str(r.data.get("reason", "")).startswith("superseded"):
                self.counters["judge_superseded"] += 1
        elif stage == STAGE_GATE_CLASSIFIED:
            action = str(r.data.get("action", "")).lower()
            if action in GATE_OUTCOMES:
                self.counters[f"gate.{action}"] += 1
        elif stage == STAGE_GATE_SUPPRESSED:
            self.counters["gate.suppressed"] += 1
        elif stage == STAGE_INCIDENT_CREATED:
            self.counters["incidents"] += 1
        elif stage == STAGE_JUDGE_USAGE:
            self.usage["calls"] += 1
            for k in ("input_tokens", "output_tokens", "cost_usd"):
                v = r.data.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.usage[k] += float(v)
        elif stage == STAGE_SOURCE_HEALTH:
            self._fold_health(r)
        elif stage == STAGE_TRACE_DROPPED:
            self.counters["records_dropped"] = int(r.data.get("dropped", 0))

    def _window_for(self, r: TraceRecord) -> tuple[float, float]:
        """The evidence window this admission covered — DECLARED if the record says so, else the
        manifest's nominal fallback (default ``0.0`` = an instant, so nothing is invented)."""
        a = r.data.get("window_start_s")
        b = r.data.get("window_end_s")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return (float(a), float(b))
        ts = float(r.source_ts or 0.0)
        w = float(r.data.get("evidence_window_s", self._m.evidence_window_s) or 0.0)
        return (ts - w, ts) if w > 0 else (ts, ts)

    def _fold_health(self, r: TraceRecord) -> None:
        state = str(r.data.get("state", ""))
        ts = float(r.source_ts or 0.0)
        if state == "degraded" and self._degraded_at is None:
            self._degraded_at = ts
        elif state in ("recovered", "ended") and self._degraded_at is not None:
            self.unhealthy_s += max(0.0, ts - self._degraded_at)
            self._degraded_at = None

    def close(self, at_source_ts: float | None) -> None:
        """A still-latched degradation at end-of-run runs to the end (degrade-loud, never forgiven)."""
        if self._degraded_at is not None and at_source_ts is not None:
            self.unhealthy_s += max(0.0, float(at_source_ts) - self._degraded_at)
            self._degraded_at = None

    @property
    def last_source_ts(self) -> float | None:
        return self._last_ts

    @property
    def span_s(self) -> float:
        if self._first_ts is None or self._last_ts is None:
            return 0.0
        return max(0.0, self._last_ts - self._first_ts)


# ---------------------------------------------------------------------------
# 6. The recorder — bounded, non-blocking, loud about its own losses
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Structure and persist the live trace stream WITHOUT ever blocking capture.

    Wire it to whichever driver you have — both are supported, because a client bypassing the
    runtime worker must not lose its trace surface::

        rec = TraceRecorder(RunTraceManifest.create(source="camera", goals=goals))

        # (a) the runtime worker
        async for incident in percept.run(core, source=..., on_trace=rec.tracer(),
                                          on_health=rec.health_sink()):
            ...

        # (b) a direct perceive_judged() driver — no worker involved
        core.on_trace = rec.tracer()
        fires = await core.perceive_judged(frame, source="camera", t=26.2)

        summary = rec.summary()        # the derived funnel
        rec.write_json(path)           # manifest + records + summary, one artifact

    Ingest is synchronous and allocation-light: fold the counters, decide retention, append. It never
    awaits, never does I/O, and never raises into the caller — :meth:`tracer` wraps everything so a
    recorder fault degrades to a dropped record instead of a stalled capture loop.

    **Bounds and the drop marker.** Retention is bounded by ``max_records``, by
    ``policy.max_bytes`` and by ``policy.max_duration_s``. When a bound is hit the record is dropped
    and a ``trace_dropped`` record is written into the retained stream by evicting the OLDEST
    retained record — so the loss is visible even if the stream is never drained, and the drop count
    rides in :attr:`dropped` and in :attr:`TraceSummary.records_dropped`. Counters are folded BEFORE
    the retention decision, so a dropped record still moves the funnel: the numbers stay exact, only
    the detail is lost.
    """

    def __init__(
        self,
        manifest: RunTraceManifest,
        *,
        max_records: int = 20_000,
        wall_clock: Callable[[], float] = time.monotonic,
        on_record: Callable[[TraceRecord], None] | None = None,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.manifest = manifest
        self.policy = manifest.policy
        self._max_records = int(max_records)
        self._wall = wall_clock
        self._on_record = on_record
        self._records: list[TraceRecord] = []
        self._agg = _Aggregate(manifest)
        self._seq = 0
        self._dropped = 0
        self._drop_reasons: Counter[str] = Counter()
        self._bytes = 0
        self._observed_wall: dict[str, float] = {}   # trace_id -> when the runtime FIRST saw it
        self._first_source_ts: float | None = None
        self._last_source_ts: float | None = None
        self._incident_ts: list[float] = []          # incident/health anchors for the retention window
        self._closed = False

    # -- properties -------------------------------------------------------
    @property
    def records(self) -> tuple[TraceRecord, ...]:
        """The RETAINED records, in ingest order."""
        return tuple(self._records)

    @property
    def dropped(self) -> int:
        """How many records this recorder dropped. Never silent — see the class docstring."""
        return self._dropped

    @property
    def drop_reasons(self) -> dict[str, int]:
        return dict(self._drop_reasons)

    @property
    def lost_records(self) -> int:
        """Records LOST to a bound (ring/bytes/duration/error) — i.e. :attr:`dropped` minus the
        intentional policy-level filtering the operator asked for. This is the number that means
        "your trace is incomplete"; it is what :attr:`TraceSummary.records_dropped` reports."""
        return self._dropped - self._drop_reasons["policy_level"] - self._drop_reasons["closed"]

    # -- ingest -----------------------------------------------------------
    def record(
        self,
        stage: str,
        *,
        trace_id: str | None = None,
        source_ts: float | None = None,
        goal_id: str | None = None,
        source: str | None = None,
        observed_wall_ts: float | None = None,
        emitted_wall_ts: float | None = None,
        **data: Any,
    ) -> TraceRecord:
        """Ingest one record. Always returns the built record; retention may still have dropped it.

        The two clocks are filled here: ``emitted_wall_ts`` defaults to *now* (this decision is being
        emitted), and ``observed_wall_ts`` defaults to the FIRST wall time seen for this
        ``trace_id`` — so ``gate.classified`` measures latency from when the frame arrived, not from
        itself. A ``trace_id`` never seen before anchors on now.
        """
        now = float(self._wall())
        tid = trace_id or f"{source or self.manifest.source}@{source_ts}"
        if observed_wall_ts is None:
            observed_wall_ts = self._observed_wall.get(tid)
            if observed_wall_ts is None:
                observed_wall_ts = now
                if len(self._observed_wall) >= 4 * self._max_records:
                    self._observed_wall.clear()  # bounded: correlation map can't leak on a 24/7 run
                self._observed_wall[tid] = now
        if emitted_wall_ts is None:
            emitted_wall_ts = now
        if not self.policy.include_media:
            data = {k: v for k, v in data.items() if k not in _MEDIA_KEYS}
        rec = TraceRecord(
            seq=self._seq, trace_id=tid, stage=stage, source_ts=source_ts,
            observed_wall_ts=observed_wall_ts, emitted_wall_ts=emitted_wall_ts,
            goal_id=goal_id, source=source or self.manifest.source, data=dict(data),
        )
        self._seq += 1
        self._agg.fold(rec)                       # EXACT counters, whatever retention decides
        if source_ts is not None:
            self._first_source_ts = source_ts if self._first_source_ts is None else min(self._first_source_ts, source_ts)
            self._last_source_ts = source_ts if self._last_source_ts is None else max(self._last_source_ts, source_ts)
        if stage in (STAGE_INCIDENT_CREATED, STAGE_SOURCE_HEALTH) and source_ts is not None:
            self.mark_incident(source_ts)
        self._maybe_retain(rec)
        if self._on_record is not None:
            try:
                self._on_record(rec)
            except Exception:  # noqa: BLE001 — a sink is observability; never break capture
                pass
        return rec

    def mark_incident(self, source_ts: float) -> None:
        """Anchor a full-detail retention window at ``source_ts`` (incident / health transition)."""
        self._incident_ts.append(float(source_ts))
        if len(self._incident_ts) > 512:
            del self._incident_ts[: len(self._incident_ts) - 512]

    def _near_incident(self, source_ts: float | None) -> bool:
        if source_ts is None or self.policy.incident_window_s <= 0:
            return False
        w = self.policy.incident_window_s
        return any(abs(source_ts - a) <= w for a in self._incident_ts)

    def _maybe_retain(self, rec: TraceRecord) -> None:
        if self._closed:
            return self._drop("closed")
        if not (self.policy.retains(rec.stage) or self._near_incident(rec.source_ts)):
            return self._drop("policy_level")
        if self.policy.max_duration_s is not None and rec.source_ts is not None:
            if self._first_source_ts is not None and rec.source_ts - self._first_source_ts > self.policy.max_duration_s:
                return self._drop("max_duration_s")
        size = _approx_bytes(rec)
        if self._bytes + size > self.policy.max_bytes:
            return self._drop("max_bytes")
        if len(self._records) >= self._max_records:
            return self._drop("max_records")
        self._records.append(rec)
        self._bytes += size

    def _drop(self, reason: str) -> None:
        """Drop a record and MAKE THE LOSS VISIBLE — the whole point of rule 2.

        The drop marker is written into the retained stream by evicting the oldest retained record
        (which is itself counted as dropped), so a full ring still reports its own truncation. A
        purely policy-level filter (a record the operator asked not to keep) is not a loss and does
        not evict anything — it is counted separately so the two are never confused.
        """
        self._dropped += 1
        self._drop_reasons[reason] += 1
        if reason in ("policy_level", "closed"):
            return  # intentional filtering, not loss: counted, but no marker and no eviction
        self._write_drop_marker()

    def _write_drop_marker(self) -> None:
        lost = self._dropped - self._drop_reasons["policy_level"] - self._drop_reasons["closed"]
        reasons = {k: v for k, v in self._drop_reasons.items() if k not in ("policy_level", "closed")}
        if self._records and self._records[-1].stage == STAGE_TRACE_DROPPED:
            last = self._records[-1]
            self._records[-1] = replace(last, data={"dropped": lost, "reasons": reasons})
            self._agg.counters["records_dropped"] = lost
            return
        marker = TraceRecord(
            seq=self._seq, trace_id=f"{self.manifest.run_id}#trace_dropped",
            stage=STAGE_TRACE_DROPPED, source_ts=self._last_source_ts,
            observed_wall_ts=float(self._wall()), emitted_wall_ts=float(self._wall()),
            source=self.manifest.source, data={"dropped": lost, "reasons": reasons},
        )
        self._seq += 1
        size = _approx_bytes(marker)
        while self._records and (
            len(self._records) >= self._max_records or self._bytes + size > self.policy.max_bytes
        ):
            evicted = self._records.pop(0)
            self._bytes -= _approx_bytes(evicted)
            if evicted.stage != STAGE_TRACE_DROPPED:
                self._dropped += 1
                self._drop_reasons["evicted_for_drop_marker"] += 1
                lost += 1
                marker = replace(marker, data={"dropped": lost, "reasons": dict(
                    (k, v) for k, v in self._drop_reasons.items() if k not in ("policy_level", "closed")
                )})
        self._records.append(marker)
        self._bytes += size
        self._agg.counters["records_dropped"] = lost

    # -- driver adapters --------------------------------------------------
    def tracer(self) -> Callable[[Any], None]:
        """An ``on_trace`` callback: pass to :func:`percept.run` or set as ``core.on_trace``.

        Consumes the EXISTING :class:`~percept.observability.TraceEvent` stream unchanged — the
        per-frame ``edge.decision``, ``source.frame``, ``judge.*``, ``gate.classified`` and
        ``incident.created`` stages the runtime already emits. Never raises and never awaits.
        """
        def _cb(ev: Any) -> None:
            try:
                self.record(
                    str(getattr(ev, "stage", "unknown")),
                    trace_id=getattr(ev, "trace_id", None),
                    source_ts=getattr(ev, "t", None),
                    goal_id=getattr(ev, "goal_id", None),
                    **dict(getattr(ev, "data", None) or {}),
                )
            except Exception:  # noqa: BLE001 — observability must not crash perception
                self._dropped += 1
                self._drop_reasons["recorder_error"] += 1
        return _cb

    def health_sink(self) -> Callable[[Any], None]:
        """An ``on_health`` callback: pass to :func:`percept.run` as ``on_health=``.

        Maps ``RuntimeHealthEvent`` kinds onto this module's vocabulary — scheduler drops become
        ``judge.dropped``, gate suppression becomes ``gate.suppressed``, and the degrade/recover
        latch (M3R) becomes ``source.health`` transitions that drive ``source_healthy_s``. Kinds
        outside those sets are recorded verbatim under ``source.health`` with ``state="info"`` so a
        new runtime event can never be silently lost.
        """
        def _cb(ev: Any) -> None:
            try:
                kind = str(getattr(ev, "kind", "unknown"))
                ts = getattr(ev, "ts", None)
                data = dict(getattr(ev, "data", None) or {})
                if kind in JUDGE_DROP_HEALTH_KINDS:
                    self.record(STAGE_JUDGE_DROPPED, source_ts=ts, kind=kind, **data)
                elif kind == "gate_suppressed":
                    self.record(STAGE_GATE_SUPPRESSED, source_ts=ts, kind=kind, **data)
                elif kind in DEGRADED_HEALTH_KINDS:
                    self.record(STAGE_SOURCE_HEALTH, source_ts=ts, kind=kind, state="degraded", **data)
                elif kind in RECOVERED_HEALTH_KINDS:
                    self.record(STAGE_SOURCE_HEALTH, source_ts=ts, kind=kind, state="recovered", **data)
                elif kind == "runtime_started":
                    self.record(STAGE_RUN_STARTED, source_ts=ts, kind=kind, **data)
                elif kind == "runtime_ended":
                    self.record(STAGE_RUN_ENDED, source_ts=ts, kind=kind, **data)
                else:
                    self.record(STAGE_SOURCE_HEALTH, source_ts=ts, kind=kind, state="info", **data)
            except Exception:  # noqa: BLE001
                self._dropped += 1
                self._drop_reasons["recorder_error"] += 1
        return _cb

    def record_usage(
        self,
        *,
        trace_id: str | None = None,
        source_ts: float | None = None,
        goal_id: str | None = None,
        input_tokens: float | None = None,
        output_tokens: float | None = None,
        cost_usd: float | None = None,
    ) -> TraceRecord | None:
        """Record one ACTUAL judge call's token/cost usage — the denominator of the savings formula.

        ``mean_input_tokens_per_actual_call`` is measured from exactly these records. With none, the
        summary reports ``estimated_saved_tokens=None`` rather than inventing a per-call cost.
        """
        payload = {k: v for k, v in (
            ("input_tokens", input_tokens), ("output_tokens", output_tokens), ("cost_usd", cost_usd),
        ) if v is not None}
        if not payload:
            return None
        return self.record(
            STAGE_JUDGE_USAGE, trace_id=trace_id, source_ts=source_ts, goal_id=goal_id, **payload
        )

    def record_verdict_usage(self, verdict: Any, **kw: Any) -> TraceRecord | None:
        """:meth:`record_usage` from a backend-annotated verdict, via
        :func:`percept.observability.judge_usage` (the frozen ``Verdict`` contract is unchanged)."""
        from .observability import judge_usage  # noqa: PLC0415 - avoid an import cycle at module load

        usage = judge_usage(verdict)
        if not usage:
            return None
        return self.record_usage(**usage, **kw)

    # -- outputs ----------------------------------------------------------
    def close(self, *, source_ts: float | None = None) -> None:
        """End the run: close any latched degradation and stop accepting records."""
        self._agg.close(source_ts if source_ts is not None else self._last_source_ts)
        self._closed = True

    def watch_duration_s(self) -> float:
        """Span of SOURCE time this run covered (last observed ``source_ts`` minus the first)."""
        if self._first_source_ts is None or self._last_source_ts is None:
            return 0.0
        return max(0.0, self._last_source_ts - self._first_source_ts)

    def summary(self, *, observed_unit: str = "frames") -> TraceSummary:
        """Derive the funnel from this recorder's EXACT aggregates (not from retained records)."""
        agg = self._agg
        return TraceSummary.derive(
            self.manifest,
            self._records,
            counters=dict(agg.counters),
            usage=dict(agg.usage),
            evidence_windows=list(agg.windows),
            unhealthy_s=agg.unhealthy_s,
            watch_duration_s=self.watch_duration_s(),
            records_dropped=self.lost_records,
            observed_unit=observed_unit,
        )

    def to_dict(self, *, observed_unit: str = "frames") -> dict[str, Any]:
        """The whole artifact: manifest + summary + retained records."""
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "summary": self.summary(observed_unit=observed_unit).to_dict(),
            "records": [r.to_dict() for r in self._records],
        }

    def to_json(self, *, indent: int | None = 2, observed_unit: str = "frames") -> str:
        return json.dumps(self.to_dict(observed_unit=observed_unit), indent=indent, default=str)

    def write_json(self, path: Any, *, observed_unit: str = "frames") -> Any:
        """Persist the artifact to ``path`` (str or ``Path``); returns the path written."""
        import pathlib  # noqa: PLC0415

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(observed_unit=observed_unit), encoding="utf-8")
        return p


#: Record-data keys that may carry media; stripped unless ``policy.include_media`` is set, so a
#: trace stays safe to hand to support by default.
_MEDIA_KEYS: frozenset[str] = frozenset({"frame", "frames", "image", "snippet", "clip", "media", "roi_image"})


def _approx_bytes(rec: TraceRecord) -> int:
    """Cheap size estimate for the byte bound — no full serialization on the hot path."""
    n = 96 + len(rec.stage) + len(rec.trace_id)
    for k, v in rec.data.items():
        n += len(k) + (len(v) if isinstance(v, str) else 24)
    return n


def load_trace(payload: Mapping[str, Any] | str) -> tuple[RunTraceManifest, list[TraceRecord], TraceSummary]:
    """Load a persisted artifact back into ``(manifest, records, summary)``.

    The summary is taken from the file when present (it was derived from EXACT counters at record
    time, including records retention discarded); otherwise it is re-derived from the records, which
    is exact only for ``level="full"``.
    """
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    manifest = RunTraceManifest.from_dict(data["manifest"])
    records = [TraceRecord.from_dict(r) for r in data.get("records") or ()]
    stored = data.get("summary")
    if stored:
        s = dict(stored)
        cost = dict(s.get("cost") or {})
        recs = dict(s.get("records") or {})
        sav = dict(s.get("savings") or {})
        lat = dict(s.get("latency") or {})
        summary = TraceSummary(
            run_id=str(s.get("run_id", manifest.run_id)),
            watch_duration_s=float(s.get("watch_duration_s", 0.0)),
            source_healthy_s=float(s.get("source_healthy_s", 0.0)),
            observed=int(s.get("observed", 0)),
            observed_unit=str(s.get("observed_unit", "frames")),
            admitted=int(s.get("admitted", 0)),
            admitted_evidence_union_s=float(s.get("admitted_evidence_union_s", 0.0)),
            judge_started=int(s.get("judge_started", 0)),
            judge_completed=int(s.get("judge_completed", 0)),
            judge_degraded=int(s.get("judge_degraded", 0)),
            judge_dropped=int(s.get("judge_dropped", 0)),
            judge_superseded=int(s.get("judge_superseded", 0)),
            gate_outcomes=dict(s.get("gate_outcomes") or {}),
            incidents=int(s.get("incidents", 0)),
            input_tokens=float(cost.get("input_tokens", 0.0)),
            output_tokens=float(cost.get("output_tokens", 0.0)),
            cost_usd=float(cost.get("usd", 0.0)),
            usage_reported_calls=int(cost.get("usage_reported_calls", 0)),
            records_retained=int(recs.get("retained", len(records))),
            records_dropped=int(recs.get("dropped", 0)),
            baseline=BaselineAssumption.from_dict(sav.get("baseline") or {}),
            estimated_baseline_calls=sav.get("estimated_baseline_calls"),
            estimated_saved_calls=sav.get("estimated_saved_calls"),
            estimated_saved_tokens=sav.get("estimated_saved_tokens"),
            mean_input_tokens_per_call=sav.get("mean_input_tokens_per_call"),
            latency_p50_s=lat.get("p50_s"),
            latency_p95_s=lat.get("p95_s"),
        )
    else:
        summary = TraceSummary.derive(manifest, records)
    return manifest, records, summary


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TRACE_STAGES",
    "GATE_OUTCOMES",
    "OBSERVATION_STAGE",
    "ADMISSION_STAGE",
    "DEGRADED_HEALTH_KINDS",
    "RECOVERED_HEALTH_KINDS",
    "JUDGE_DROP_HEALTH_KINDS",
    "STAGE_EDGE_DECISION",
    "STAGE_SOURCE_FRAME",
    "STAGE_JUDGE_STARTED",
    "STAGE_JUDGE_FINISHED",
    "STAGE_JUDGE_DEGRADED",
    "STAGE_JUDGE_USAGE",
    "STAGE_JUDGE_DROPPED",
    "STAGE_GATE_CLASSIFIED",
    "STAGE_GATE_SUPPRESSED",
    "STAGE_INCIDENT_CREATED",
    "STAGE_SOURCE_HEALTH",
    "STAGE_TRACE_DROPPED",
    "STAGE_RUN_STARTED",
    "STAGE_RUN_ENDED",
    "BaselineAssumption",
    "ComponentIdentity",
    "PlanIdentity",
    "RunTraceManifest",
    "TracePolicy",
    "TraceRecord",
    "TraceRecorder",
    "TraceSummary",
    "canonical_json",
    "content_hash",
    "short_hash",
    "load_trace",
    "union_seconds",
]
