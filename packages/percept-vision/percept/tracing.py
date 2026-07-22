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
STAGE_SHADOW_FOLD = "shadow.fold"          # ADVISORY — the subject + EpisodeFoldResult the
#: model WOULD have proposed for this observation. Emitted alongside a decision, it governs NOTHING:
#: it changes no firing, re-arms no edge, suppresses nothing. It exists so a client running the wheel
#: on THEIR data can hand back a ``.ptrace`` (media OFF) that carries what the fold proposed, gradeable
#: against their ground truth. A consumer that ignores it sees byte-for-byte today's behaviour.
STAGE_JUDGMENT_EVENT = "judgment.event"    # the frozen percept.contracts.JudgmentEvent for one decision,
#: projected into the trace stream. Emitted alongside ``gate.classified`` whenever a fire carried a
#: JudgmentEvent (the judged watch path only — episode/rhythm/boundary/transition/absence fires take no
#: judge-funnel stamps and emit nothing here). ``data["evidence_seq"]`` is stamped by the RECORDER, not
#: the caller: the ``seq`` of the ``source.frame`` TraceRecord for this SAME ``trace_id`` — i.e. the
#: literal evidence that fed this judgment, resolvable within the same ``.ptrace``. ``data["dispatch_seq"]``
#: carries the JudgmentEvent's OWN ``evidence_seq`` (the per-key ADR-001 §8 dispatch sequence, a distinct
#: number) so neither meaning is lost. Purely additive observability — governs nothing.
STAGE_EXTRACTOR_ENRICHED = "extractor.enriched"  # a percept.extraction.EvidenceExtractor ran for this
#: goal and produced an Enrichment (possibly empty) BEFORE the judge call it feeds. Emitted right
#: before ``judge.started`` when an extractor is configured; absent entirely when none is.
#: ``data["facts"]`` is the SAME dict merged into ``context["extractor"]`` for the judge — this
#: stage is what makes "did the extractor's facts actually reach the judge" answerable from the trace
#: alone, not just from reading the call site. Purely additive observability — governs nothing.

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
    STAGE_SHADOW_FOLD,
    STAGE_JUDGMENT_EVENT,
    STAGE_EXTRACTOR_ENRICHED,
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
#: ``gate.classified``; ``suppressed`` is the typed suppression (start_anchored / refractory) and
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
    provider/adapter identity (the NAME is the identity); ``model`` is
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


#: The resolved policy knobs that a plan identity is expected to carry. A threshold A/B is
#: the canonical eval shape — two arms whose GOALS are byte-identical and whose only difference is a
#: cutoff — so a plan hash blind to these knobs cannot tell the two arms apart, which defeats the whole
#: point of quoting it. These names are the RESOLVED runtime fields, not env-var spellings:
#: ``motion_gate`` / ``motion_gate_block_max`` / ``motion_gate_p95`` / ``salience_stat`` come off
#: :class:`percept.config.EdgeSettings`, ``require_false_first`` / ``refractory_s`` off
#: :class:`percept.config.GateSettings`, ``dwell`` is the goal-level ``state['hold_s']`` sustain, and
#: ``plan_mode`` is the resolved profile mode. The list is DOCUMENTATION, not a whitelist:
#: :meth:`PlanIdentity.from_goals` hashes every key it is handed, so a knob nobody thought of here still
#: moves the hash rather than hiding inside it.
PLAN_POLICY_FIELDS: tuple[str, ...] = (
    "motion_gate",
    "motion_gate_block_max",
    "motion_gate_p95",
    "salience_stat",
    "gate_threshold",
    "require_false_first",
    "refractory_s",
    "dwell",
    "plan_mode",
)


def canonical_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a resolved-policy mapping into the canonical form that enters the plan hash.

    Order-independent (the hash is taken over :func:`canonical_json`, which sorts keys) and
    ``None``-pruned: an explicitly-unset knob and an absent knob are the SAME configuration and must
    not fork the identity. Numbers are coerced to ``float`` so an int ``1`` and a float ``1.0`` cutoff
    agree; bools stay bools (they are flags, not thresholds). Nested mappings are FLATTENED to dotted
    keys (``"gate.refractory_s"``) rather than nested, so the result is always a flat mapping of scalars
    — that is what keeps :class:`PlanIdentity` hashable and its ``to_dict``/``from_dict`` round-trip
    hash-stable.
    """
    if not policy:
        return {}
    out: dict[str, Any] = {}
    for raw_key, value in policy.items():
        key = str(raw_key)
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        elif isinstance(value, Mapping):
            for sub_key, sub_value in canonical_policy(value).items():
                out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = str(value)
    return dict(sorted(out.items()))


def _goal_dwell(goal: Any) -> float | None:
    """The goal's resolved sustain (``state['hold_s']``) as a positive float, or ``None``.

    Mirrors :func:`percept.behavior._coerce_hold_s` exactly — a malformed or non-positive hint is no
    dwell at all, so a garbage value can never mint a distinct identity for an identical plan.
    """
    state = getattr(goal, "state", None)
    if not isinstance(state, Mapping):
        return None
    v = state.get("hold_s")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if v > 0 else None


@dataclass(frozen=True)
class PlanIdentity:
    """WHAT was being watched: the goal set AND the resolved policy it was watched under, content-addressed.

    ``goal_ids`` is the ordered active set; ``plan_hash`` addresses the full plan payload — conditions,
    per-goal dwell, and the resolved thresholds/policy in ``policy`` — so two runs that quote the same
    ``plan_hash`` provably watched for the same thing *under the same cutoffs*; ``revision`` is the live
    :class:`~percept.goals.GoalRegistry` revision the run started at (goal edits move it mid-run, and a
    judged frame is always pinned to the revision it was admitted under).

    The hash covers policy because the commonest eval is a threshold A/B, where the goals are
    identical by construction. A hash that could not separate those two arms was worse than no hash —
    it let a report claim two different configurations were the same one.
    """

    goal_ids: tuple[str, ...] = ()
    plan_hash: str | None = None
    revision: int | None = None
    plan_mode: str | None = None
    #: Sorted ``(key, value)`` pairs of the resolved policy that entered ``plan_hash``. Stored as a
    #: tuple, not a dict, so :class:`PlanIdentity` stays hashable and frozen; read it via
    #: :attr:`policy_dict`.
    policy: tuple[tuple[str, Any], ...] = ()

    @property
    def policy_dict(self) -> dict[str, Any]:
        """The resolved policy as a mapping — what the hash was actually taken over."""
        return dict(self.policy)

    @classmethod
    def from_goals(cls, goals: Iterable[Any], *, revision: int | None = None,
                   plan_mode: str | None = None,
                   policy: Mapping[str, Any] | None = None) -> "PlanIdentity":
        """Build from anything with ``.id`` / ``.condition`` (a :class:`~percept.adapters.Goal`).

        ``policy`` is the RESOLVED threshold/policy mapping this plan runs under (see
        :data:`PLAN_POLICY_FIELDS`). ``plan_mode``, when given, is folded into it, so the mode is part
        of the identity rather than a label beside it. Every key is hashed; key order never matters.
        """
        items = list(goals)
        goal_payload = [
            {
                "id": str(getattr(g, "id", "")),
                "condition": str(getattr(g, "condition", "")),
                "say": str(getattr(g, "say", "")),
                "modality": getattr(g, "modality", None),
                "dwell": _goal_dwell(g),
            }
            for g in items
        ]
        resolved = canonical_policy(policy)
        if plan_mode is not None:
            resolved["plan_mode"] = str(plan_mode)
        return cls(
            goal_ids=tuple(str(getattr(g, "id", "")) for g in items),
            plan_hash=content_hash({"goals": goal_payload, "policy": resolved}),
            revision=revision,
            plan_mode=plan_mode,
            policy=tuple(sorted(resolved.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_ids": list(self.goal_ids), "plan_hash": self.plan_hash,
            "revision": self.revision, "plan_mode": self.plan_mode,
            "policy": self.policy_dict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanIdentity":
        return cls(
            goal_ids=tuple(payload.get("goal_ids") or ()),
            plan_hash=payload.get("plan_hash"),
            revision=payload.get("revision"),
            plan_mode=payload.get("plan_mode"),
            policy=tuple(sorted(canonical_policy(payload.get("policy")).items())),
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
    # The shadow advisory is retained at EVERY level (it is low-volume — one record per
    # subject/fold decision, not per frame). It must survive a client's media-OFF ``.ptrace``, which
    # may be captured at production (summary) level; a fold that only lived at ``full`` would be absent
    # from exactly the trace a client can send back for offline grading.
    STAGE_SHADOW_FOLD,
})

#: Additional stages retained at ``level="decisions"``: the judge/gate decision detail. The per-frame
#: ``edge.decision`` is deliberately NOT here — that is the ``full`` level.
_DECISION_STAGES: frozenset[str] = _SUMMARY_STAGES | frozenset({
    STAGE_SOURCE_FRAME, STAGE_JUDGE_STARTED, STAGE_JUDGE_FINISHED, STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_USAGE, STAGE_JUDGE_DROPPED, STAGE_GATE_CLASSIFIED, STAGE_GATE_SUPPRESSED,
    STAGE_JUDGMENT_EVENT, STAGE_EXTRACTOR_ENRICHED,
})


@dataclass(frozen=True)
class TracePolicy:
    """What a run is allowed to RETAIN. Counters are always exact; this bounds storage only.

    - ``level="summary"`` — continuous production. Lifecycle, health transitions and incidents are
      kept; per-frame and per-judge records are counted and discarded.
    - ``level="decisions"`` — judge/gate detail kept, per-frame ``edge.decision`` discarded.
    - ``level="full"`` — everything, including a record per frame. Correct for a 36s probe,
      **dangerous for 24/7**: always pair it with ``max_duration_s`` / ``max_bytes``.

    ``incident_window_s`` gives production the third posture that's needed: full detail is
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
    #: ``require_false_first`` / ``refractory_s``). Additive, and load-bearing:
    #: :mod:`percept.replay` re-runs recorded verdicts through a Gate rebuilt from EXACTLY these
    #: numbers, so a cognition replay is not silently done against today's defaults. Empty = not
    #: captured, and the replayer says so instead of guessing (see
    #: :attr:`percept.replay.ReplayReport.gate_config_recorded`).
    gate_config: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    #: Wheel/sdist provenance of the package that produced this trace — the git commit it was
    #: built from and the UTC build timestamp (see :func:`percept.build_info`). ``"unreleased"``
    #: in a bare source checkout that never went through the packaging build step. Additive: ties
    #: a returned ``.ptrace`` back to the exact build, on top of the already-recorded
    #: ``percept_version``/``contract_version``.
    build_commit: str | None = None
    build_timestamp: str | None = None
    #: The resolved OBJECT identities behind this run, when it started from a
    #: :class:`~percept.contracts.watchplan.ResolvedWatch` (:meth:`create`'s ``resolved=`` kwarg) —
    #: composes with ``build_commit``/``build_timestamp`` above as the same kind of fact: not what
    #: ran, but exactly WHICH configuration objects it ran with. ``source_profile_id`` is
    #: ``"<name>@<content_hash[:8]>"`` (the profile has no separate version field today, so its own
    #: content hash plays that role — two profiles with the same name and knobs share an id).
    #: ``profiles`` names every registered ``WatchProfile`` the resolution consulted (empty when
    #: none did). ``component_snapshot_hash`` / ``optimization_lock_hash`` mirror the same-named
    #: :class:`~percept.contracts.watchplan.ResolvedWatch` fields. All four are ``None``/empty —
    #: byte-for-byte the previous manifest shape — when the run did not go through
    #: :class:`~percept.resolver.ConfigResolver` first.
    source_profile_id: str | None = None
    profiles: tuple[str, ...] = ()
    component_snapshot_hash: str | None = None
    optimization_lock_hash: str | None = None

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
        plan_policy: Mapping[str, Any] | None = None,
        resolved: "Any | None" = None,
        source_profile_id: str | None = None,
        profiles: Sequence[str] = (),
        component_snapshot_hash: str | None = None,
        optimization_lock_hash: str | None = None,
    ) -> "RunTraceManifest":
        """Build a manifest, filling versions from the installed package (never hand-synced).

        ``resolved`` — an optional :class:`~percept.contracts.watchplan.ResolvedWatch` (the output
        of :meth:`percept.resolver.ConfigResolver.resolve`). When given, its object identities are
        read into ``source_profile_id``/``profiles``/``component_snapshot_hash``/
        ``optimization_lock_hash`` UNLESS the caller also passed the matching kwarg explicitly (an
        explicit value always wins). Typed loosely (``Any``) so this module never imports
        ``percept.resolver``/``percept.contracts.watchplan`` at module scope — the same
        local-import discipline the version lookups just above already use, to keep
        ``percept.tracing`` importable early in the package's own import graph.
        """
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
        build_commit = build_timestamp = None
        try:
            from ._build import build_info as _bi  # noqa: PLC0415
            _build = _bi()
            build_commit = _build.git_commit
            build_timestamp = _build.build_timestamp
        except Exception:  # pragma: no cover - defensive
            pass

        # Resolved object identities (additive; see the field docstrings above) — an explicit
        # kwarg always wins over what ``resolved`` would derive.
        resolved_source_profile_id = source_profile_id
        resolved_profiles = tuple(profiles)
        resolved_component_snapshot_hash = component_snapshot_hash
        resolved_optimization_lock_hash = optimization_lock_hash
        if resolved is not None:
            if resolved_source_profile_id is None:
                sp = getattr(resolved, "source_profile", None)
                name = sp.get("name") if sp else None
                if name:
                    resolved_source_profile_id = f"{name}@{content_hash(dict(sp))[:8]}"
            if not resolved_profiles:
                # Registered WatchProfile names the resolution consulted, per-field provenance
                # names them as "profile_resolution.resolve_profile('<name>')" source ids — see
                # percept.resolver.ConfigResolver.resolve's situation-tier contribution.
                names: set[str] = set()
                for origin in getattr(resolved, "provenance", {}).values():
                    source_id = getattr(origin, "source_id", "")
                    prefix = "profile_resolution.resolve_profile("
                    if source_id.startswith(prefix):
                        names.add(source_id[len(prefix):].rstrip(")").strip("'\""))
                resolved_profiles = tuple(sorted(names))
            if resolved_component_snapshot_hash is None:
                resolved_component_snapshot_hash = getattr(resolved, "component_snapshot_hash", None)
            if resolved_optimization_lock_hash is None:
                resolved_optimization_lock_hash = getattr(resolved, "optimization_lock_hash", None)

        return cls(
            run_id=run_id or uuid.uuid4().hex[:16],
            source=source,
            percept_version=percept_version,
            watch_version=watch_version,
            contract_version=contract_version,
            build_commit=build_commit,
            build_timestamp=build_timestamp,
            source_profile_id=resolved_source_profile_id,
            profiles=resolved_profiles,
            component_snapshot_hash=resolved_component_snapshot_hash,
            optimization_lock_hash=resolved_optimization_lock_hash,
            started_wall_ts=float(wall_clock()),
            started_source_ts=started_source_ts,
            driver=driver,
            # When the caller hands us goals rather than a built PlanIdentity, the resolved
            # cutoffs it ran under (``plan_policy``) still belong in the identity — otherwise two runs
            # of the same goals at different thresholds quote the same plan_hash.
            plan=plan if plan is not None else (
                PlanIdentity.from_goals(goals, policy=plan_policy) if goals is not None
                else PlanIdentity()
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
            "build_commit": self.build_commit,
            "build_timestamp": self.build_timestamp,
            "source_profile_id": self.source_profile_id,
            "profiles": list(self.profiles),
            "component_snapshot_hash": self.component_snapshot_hash,
            "optimization_lock_hash": self.optimization_lock_hash,
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
            build_commit=payload.get("build_commit"),
            build_timestamp=payload.get("build_timestamp"),
            source_profile_id=payload.get("source_profile_id"),
            profiles=tuple(payload.get("profiles") or ()),
            component_snapshot_hash=payload.get("component_snapshot_hash"),
            optimization_lock_hash=payload.get("optimization_lock_hash"),
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
    - **cost is never a FALSE ZERO.** ``input_tokens`` / ``output_tokens`` are ``None`` when no judge
      call reported usage at all, and ``cost_usd`` is ``None`` unless at least one call could be
      priced honestly (a backend-reported ``cost_usd``, or tokens against a model in
      :data:`percept.slo.PRICE_TABLE`). ``0.0`` would be a WRONG NUMBER THAT A BAR CAN PASS — a run
      that really spent money would show green against a ``cost_usd`` ceiling. ``—`` (unknown) is
      safe; ``0.0`` is not, so the unknown case is typed ``None`` and rendered ``—``. The ONE
      licensed zero is a run that made no judge call at all (``percept doctor``, a fully gated
      stream) — there the zero is measured.
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
    input_tokens: float | None = None
    output_tokens: float | None = None
    cost_usd: float | None = None
    usage_reported_calls: int = 0
    priced_calls: int = 0
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
        priced_calls = int(usage.get("priced_calls", 0))
        # NO FALSE ZEROS — but a PROVEN zero is still a number. Exactly one case licenses 0.0: the
        # run made no judge call at all (a `percept doctor` dry run, a fully gated stream), where
        # zero spend is measured, not assumed. Once calls were made, silence about their usage is
        # UNKNOWN (None -> "—"), never free: the failure this rule exists to stop is a run that spent
        # 134,946 input tokens across 108 calls and reported usd 0.0, green against a cost bar.
        no_calls = actual_calls == 0 and usage_calls == 0
        if no_calls:
            input_tokens: float | None = 0.0
            output_tokens: float | None = 0.0
            cost_usd: float | None = 0.0
        else:
            input_tokens = float(usage.get("input_tokens", 0.0)) if usage_calls > 0 else None
            output_tokens = float(usage.get("output_tokens", 0.0)) if usage_calls > 0 else None
            cost_usd = float(usage.get("cost_usd", 0.0)) if priced_calls > 0 else None
        mean_in: float | None = (
            (input_tokens / usage_calls) if usage_calls > 0 and input_tokens is not None else None
        )

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
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            usage_reported_calls=usage_calls,
            priced_calls=priced_calls,
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

    @property
    def dollars_per_stream_hour(self) -> float | None:
        """$/stream-hour for THIS run — ``None`` (never ``0.0``) when cost or duration is unknown.

        Divides the run's real ``cost_usd`` by the SOURCE time it covered, via the already-landed
        :class:`~percept.harness.metrics.CostSummary` aggregator (one $/stream-hour definition in the
        tree, not two). An unpriceable run has no rate — reporting ``0.0`` would let a $/stream-hour
        bar pass a run that spent money.
        """
        if self.cost_usd is None or self.watch_duration_s <= 0:
            return None
        from .harness.metrics import CostSummary  # noqa: PLC0415 - keep tracing import-light

        return CostSummary(
            total_judgments=self.usage_reported_calls,
            total_cost_usd=self.cost_usd,
            total_elapsed_s=self.watch_duration_s,
            total_stream_hours=self.watch_duration_s / 3600.0,
        ).dollars_per_stream_hour

    def cost_text(self) -> str:
        """The renderer-safe cost line — ``"—"`` when no judge call could be priced honestly."""
        if self.cost_usd is None:
            return "—"
        return f"${self.cost_usd:.4f}"

    def tokens_text(self) -> str:
        """The renderer-safe token line — ``"—"`` when no judge call reported usage."""
        if self.input_tokens is None or self.output_tokens is None:
            return "—"
        return f"in {self.input_tokens:.0f} / out {self.output_tokens:.0f} tokens"

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
                "priced_calls": self.priced_calls, "text": self.cost_text(),
                "usd_per_stream_hour": self.dollars_per_stream_hour,
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


def _opt_float(v: Any) -> float | None:
    """``float(v)`` — but ``None``/unparseable stays ``None``, never becomes ``0.0``."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _verdict_model(verdict: Any) -> str | None:
    """The model id a verdict was billed against, if the backend said so — else ``None``.

    Reads ``Verdict.usage["model"]`` first (the free-form usage dict backends already fill), then a
    loose ``.model`` attribute. Never guesses: an unknown model means an unpriced call, which is the
    honest outcome (see :func:`percept.slo.price_usage_or_none`).
    """
    usage = getattr(verdict, "usage", None)
    if isinstance(usage, dict):
        m = usage.get("model")
        if isinstance(m, str) and m:
            return m
    m = getattr(verdict, "model", None)
    return m if isinstance(m, str) and m else None


class _Aggregate:
    """The EXACT running funnel. Folded for every ingested record, retained or not — which is what
    makes ``level="summary"`` a storage decision rather than a measurement one."""

    def __init__(self, manifest: RunTraceManifest) -> None:
        self._m = manifest
        self.counters: Counter[str] = Counter()
        # ``calls`` = judge calls that reported ANY usage; ``priced_calls`` = the subset that carried
        # a real dollar figure. They are counted apart so a run with tokens but no usable price
        # reports cost ``None`` ("—") instead of a $0.00 that a cost bar would pass.
        self.usage: dict[str, float] = {
            "calls": 0.0, "priced_calls": 0.0,
            "input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0,
        }
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
                    if k == "cost_usd":
                        self.usage["priced_calls"] += 1
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
        self._frame_seq: dict[str, int] = {}          # trace_id -> this trace's source.frame seq
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
        if stage == STAGE_JUDGMENT_EVENT and "evidence_seq" not in data:
            # The RECORDER, not the caller, stamps the correlation handle: the seq of THIS
            # trace_id's own source.frame record (the literal evidence that fed this judgment),
            # resolvable within the same .ptrace without needing the frame itself. A caller that
            # already supplied one (e.g. a replay driver reconstructing a record) is left alone.
            frame_seq = self._frame_seq.get(tid)
            if frame_seq is not None:
                data = dict(data)
                data["evidence_seq"] = frame_seq
        rec = TraceRecord(
            seq=self._seq, trace_id=tid, stage=stage, source_ts=source_ts,
            observed_wall_ts=observed_wall_ts, emitted_wall_ts=emitted_wall_ts,
            goal_id=goal_id, source=source or self.manifest.source, data=dict(data),
        )
        self._seq += 1
        if stage == STAGE_SOURCE_FRAME:
            # Remember THIS trace_id's admitted-frame seq so a same-trace judgment.event
            # (emitted later, same tick) can resolve back to it. Bounded exactly like
            # ``_observed_wall`` — a correlation map must not leak on a 24/7 run.
            if len(self._frame_seq) >= 4 * self._max_records:
                self._frame_seq.clear()
            self._frame_seq[tid] = rec.seq
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

        ``judge.usage`` — emitted once per REAL judge call by ``Percept._timed_judge`` — is routed
        through :meth:`record_usage` rather than :meth:`record` so the run path gets the SAME pricing
        rule as a direct driver: tokens x :data:`percept.slo.PRICE_TABLE`, or no dollar figure at all.
        This is the wire that makes ``TraceSummary.cost_usd`` a measured number on a real run instead
        of a structural zero.
        """
        def _cb(ev: Any) -> None:
            try:
                stage = str(getattr(ev, "stage", "unknown"))
                data = dict(getattr(ev, "data", None) or {})
                if stage == STAGE_JUDGE_USAGE:
                    self.record_usage(
                        trace_id=getattr(ev, "trace_id", None),
                        source_ts=getattr(ev, "t", None),
                        goal_id=getattr(ev, "goal_id", None),
                        input_tokens=data.get("input_tokens"),
                        output_tokens=data.get("output_tokens"),
                        cost_usd=data.get("cost_usd"),
                        model=data.get("model"),
                    )
                    return
                self.record(
                    stage,
                    trace_id=getattr(ev, "trace_id", None),
                    source_ts=getattr(ev, "t", None),
                    goal_id=getattr(ev, "goal_id", None),
                    **data,
                )
            except Exception:  # noqa: BLE001 — observability must not crash perception
                self._dropped += 1
                self._drop_reasons["recorder_error"] += 1
        return _cb

    def health_sink(self) -> Callable[[Any], None]:
        """An ``on_health`` callback: pass to :func:`percept.run` as ``on_health=``.

        Maps ``RuntimeHealthEvent`` kinds onto this module's vocabulary — scheduler drops become
        ``judge.dropped``, gate suppression becomes ``gate.suppressed``, and the degrade/recover
        latch becomes ``source.health`` transitions that drive ``source_healthy_s``. Kinds
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
                elif kind == "zero_admission_coverage":
                    # ADR-008 §Corrections(b): this ONE kind is degraded IFF the payload
                    # says a declared coverage floor went unmet (percept.runtime.worker.run's
                    # coverage_floor= kwarg) — never inferred from the zero itself. No floor
                    # declared (floor_unmet absent/False, the default) stays exactly the
                    # diagnostic-only reading it always was: a still, motion-gated source admitting
                    # nothing never opens an unhealthy interval.
                    state = "degraded" if data.get("floor_unmet") else "info"
                    self.record(STAGE_SOURCE_HEALTH, source_ts=ts, kind=kind, state=state, **data)
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
        model: str | None = None,
    ) -> TraceRecord | None:
        """Record one ACTUAL judge call's token/cost usage — the denominator of the savings formula.

        ``mean_input_tokens_per_actual_call`` is measured from exactly these records. With none, the
        summary reports ``estimated_saved_tokens=None`` rather than inventing a per-call cost.

        PRICING (the one place tokens become dollars on the run path). When the caller supplies no
        ``cost_usd``, this prices the tokens through :func:`percept.slo.price_usage_or_none` against
        ``model``. That function returns ``None`` — and NO ``cost_usd`` is recorded — for a model
        absent from :data:`percept.slo.PRICE_TABLE`, so an un-catalogued judge yields ``—`` rather
        than a :data:`percept.slo.DEFAULT_PRICE` guess dressed up as a real price. Tokens are still
        recorded in that case: the run reports what it knows and nothing it doesn't.
        """
        if cost_usd is None and (input_tokens is not None or output_tokens is not None):
            from .slo import price_usage_or_none  # noqa: PLC0415 - keep tracing import-light

            tokens = {k: float(v) for k, v in (
                ("input_tokens", input_tokens), ("output_tokens", output_tokens),
            ) if v is not None}
            try:
                cost_usd = price_usage_or_none(tokens, model=model)
            except Exception:  # noqa: BLE001 — pricing is observability; never crash a run over it
                cost_usd = None
        payload: dict[str, Any] = {k: v for k, v in (
            ("input_tokens", input_tokens), ("output_tokens", output_tokens), ("cost_usd", cost_usd),
        ) if v is not None}
        if not payload:
            return None
        if model:
            payload["model"] = str(model)  # non-numeric: carried for debugging, never summed
        return self.record(
            STAGE_JUDGE_USAGE, trace_id=trace_id, source_ts=source_ts, goal_id=goal_id, **payload
        )

    def record_verdict_usage(self, verdict: Any, **kw: Any) -> TraceRecord | None:
        """:meth:`record_usage` from a backend-annotated verdict, via
        :func:`percept.observability.judge_usage` (the frozen ``Verdict`` contract is unchanged).

        The direct-driver seam: a caller holding completed verdicts (a harness, an eval arm, a
        ``perceive_judged`` loop without ``run()``) meters them with one call. The RUN path does not
        come through here — it meters inside ``Percept._timed_judge``, which is the only site every
        judge call passes through (a drained runtime result carries FIRES only, so metering there
        would silently miss every judgment that didn't fire — the same false-number failure in a
        different disguise)."""
        from .observability import judge_usage  # noqa: PLC0415 - avoid an import cycle at module load

        usage = judge_usage(verdict)
        if not usage:
            return None
        model = kw.pop("model", None) or _verdict_model(verdict)
        return self.record_usage(**usage, model=model, **kw)

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
            # None round-trips as None: a reloaded trace must not turn "unknown" into $0.00.
            input_tokens=_opt_float(cost.get("input_tokens")),
            output_tokens=_opt_float(cost.get("output_tokens")),
            cost_usd=_opt_float(cost.get("usd")),
            usage_reported_calls=int(cost.get("usage_reported_calls", 0)),
            priced_calls=int(cost.get("priced_calls", 0)),
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
    "STAGE_SHADOW_FOLD",
    "STAGE_EXTRACTOR_ENRICHED",
    "STAGE_SOURCE_HEALTH",
    "STAGE_TRACE_DROPPED",
    "STAGE_RUN_STARTED",
    "STAGE_RUN_ENDED",
    "PLAN_POLICY_FIELDS",
    "BaselineAssumption",
    "ComponentIdentity",
    "PlanIdentity",
    "RunTraceManifest",
    "TracePolicy",
    "TraceRecord",
    "TraceRecorder",
    "TraceSummary",
    "canonical_json",
    "canonical_policy",
    "content_hash",
    "short_hash",
    "load_trace",
    "union_seconds",
]
