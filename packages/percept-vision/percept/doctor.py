"""``percept doctor`` — a plan-aware tier-0 dry run that spends ZERO judge calls.

Two paying client teams returned "not viable" on percept 0.7.0 — not because perception failed, but
because when the SDK stayed silent there was no way to find out why. This module is the answer to
that, and it is deliberately answerable at 2am with no API key and no access to us::

    percept-doctor --source clip.mp4 --goal "the document being edited changes"

It answers, in order:

1. **Did the source open?** and did frames actually arrive (count, first/last ``t``, gate frame size).
2. **What timebase was used?** — resolved through the runtime's OWN
   :func:`percept.runtime.worker._resolve_timebase`, so doctor cannot report a clock the run would
   not have used.
3. **What plan was compiled?** — the real :func:`percept.harness.tier0.compile_watchspec` output:
   one :class:`~percept.adapters.WatchCondition` per goal with its effective threshold, salience
   stat, cadence, gate_on channel and schedule.
4. **Which source profile was active?** — the :class:`~percept.watchprofile.WatchProfile` resolved
   from the situation axes AND the :class:`~percept.sourceprofile.SourceProfile`
   (ADR-005) naming what KIND of world is on the other end of the stream. When that
   profile declares its admission signal :data:`~percept.sourceprofile.UNTUNED` — as
   ``screen_document`` does, because percept ships no calibrated page-change signal — doctor says
   so out loud (:data:`ADMISSION_SIGNAL_UNTUNED`): *this profile is running on coast alone, supply
   a change signal*. It never invents a threshold to hide it.
5. **Which admission signals were available?** — motion is computed here; structural/semantic are
   ``0.0`` in the reference gate unless an edge adapter fills them, and audio needs an audio source.
   A goal that gates on a channel nobody feeds is **plan unobservable** (exit 4), not "quiet".
6. **How many candidates did each goal admit, and WHICH RULE suppressed each non-admit** — typed
   reasons from :data:`SUPPRESSION_REASONS`, counted per goal.
7. **Would alternative statistics have admitted different evidence?** — all three salience stats,
   each against ITS OWN effective threshold.

Two rules this module exists to enforce
---------------------------------------

**Never compare the three salience stats to one threshold.** ``mean``, ``block_max`` and ``p95``
live on different value scales and the SDK already carries separately calibrated cutoffs
(``EdgeSettings.motion_gate`` 0.10 / ``motion_gate_block_max`` 0.20 / ``motion_gate_p95`` 0.15,
reachable through a profile's ``stat_thresholds``). Every row here prints its own effective
threshold beside its own value, and the admitted counts come from running the REAL
:class:`~percept.harness.tier0.MotionGate` once per (goal, stat) — never from a re-implemented
comparison that could drift from the gate.

**Zero admits is DIAGNOSTIC, not necessarily failure.** A quiet camera under a motion-gated watch
correctly admits nothing, and that silence is exactly the economic claim percept makes. So doctor
always reports :data:`ZERO_ADMITS_ON_FIXTURE` loudly and exits ``2`` — but this module adds no
runtime health alarm and no runtime behaviour whatsoever. Runtime-side coverage degradation is only
justified when a DECLARED coverage expectation is violated, which is not something a dry run knows.

**No judge spend.** Tier-0 is pure CV: this module never constructs a
:class:`~percept.agent.Percept`, never imports a vision backend and never calls ``perceive_judged``.
The funnel it renders is derived from :class:`percept.tracing.TraceSummary` — the same one definition
``percept eval`` and the visualizer use — so ``judge_started``/``judge_completed`` are structurally
zero and the savings line reads ``—`` (no measured mean input-token cost to project).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .adapters import Goal, WatchCondition, WatchSpec
from .config import Settings, load_settings
from .envelope import (
    ACTUATION_CLASSES,
    ATTENTION_CLASS_ENFORCEMENT,
    ATTENTION_CLASS_GAP,
    DEFAULT_SUBJECT_RESOLVERS,
    OUT_OF_ENVELOPE,
    EnvelopeCheck,
    EnvelopeViolation,
    actuation_check,
)
from .harness.tier0 import (
    SALIENCE_STATS,
    Frame,
    MotionGate,
    compile_watchspec,
    salience_stats,
    schedule_active,
)
from .tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    BaselineAssumption,
    ComponentIdentity,
    PlanIdentity,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    TraceSummary,
)

# ---------------------------------------------------------------------------
# Exit codes — the machine-readable contract (`--json` carries the same in `exit_code`)
# ---------------------------------------------------------------------------

EXIT_OK = 0                  #: completed; no hard incompatibility found
EXIT_ZERO_COVERAGE = 2       #: the run admitted nothing for any goal — zero semantic coverage
EXIT_SOURCE_FAILURE = 3      #: the source did not open, or delivered no frames
EXIT_PLAN_UNOBSERVABLE = 4   #: the compiled plan cannot be observed under the selected profile
EXIT_INVALID_CONFIG = 5      #: the configuration itself is invalid (bad goal/profile/argument)
EXIT_OUT_OF_ENVELOPE = 6     #: the deployment DECLARES itself outside percept's envelope (ADR-007)

#: Findings vocabulary. Closed on purpose: a client reading ``--json`` maps a code to a sentence.
ZERO_ADMITS_ON_FIXTURE = "ZERO_ADMITS_ON_FIXTURE"
SOURCE_FAILED = "SOURCE_FAILED"
NO_FRAMES = "NO_FRAMES"
PLAN_EMPTY = "PLAN_EMPTY"
SIGNAL_UNAVAILABLE = "SIGNAL_UNAVAILABLE"
INVALID_CONFIG = "INVALID_CONFIG"
ALTERNATIVE_STAT_WOULD_ADMIT = "ALTERNATIVE_STAT_WOULD_ADMIT"
SINGLE_FRAME_RUN = "SINGLE_FRAME_RUN"
EGO_COMPENSATED_GATE = "EGO_COMPENSATED_GATE"
UNEXPLAINED_SUPPRESSION = "UNEXPLAINED_SUPPRESSION"
ADMISSION_SIGNAL_UNTUNED = "ADMISSION_SIGNAL_UNTUNED"
SOURCE_PROFILE_EXPERIMENTAL = "SOURCE_PROFILE_EXPERIMENTAL"
SOURCE_PROFILE_SELECTION_AMBIGUOUS = "SOURCE_PROFILE_SELECTION_AMBIGUOUS"
OUT_OF_ENVELOPE_ACTUATION = "OUT_OF_ENVELOPE_ACTUATION"      #: ADR-007
PERSON_IDENTIFYING_RESOLVER = "PERSON_IDENTIFYING_RESOLVER"  #: ADR-007
EVIDENCE_LINK_BROKEN = "EVIDENCE_LINK_BROKEN"                #: a supplied trace has a delivered Incident
#: that does not resolve to exactly one judgment.event (percept.replay.verify_evidence_links)

#: Why a frame did NOT become a candidate for a goal. Every non-admit lands in exactly one bucket —
#: "it was quiet" is only ONE of these, and conflating it with the others is the bug this module fixes.
SUPPRESSION_REASONS: tuple[str, ...] = (
    "no_predecessor_frame",   # the first frame: nothing to diff against, by definition not motion
    "out_of_schedule",        # the goal's Schedule was not armed at this civil time
    "cadence_not_elapsed",    # min_judge_interval since this goal's last escalation
    "signal_unavailable",     # gate_on names a channel nothing in this topology feeds (always 0.0)
    "below_threshold",        # the honest quiet case: the statistic did not clear ITS cutoff
    "unexplained",            # doctor could not attribute it — always a bug, always reported loud
)


class DoctorConfigError(ValueError):
    """An invalid doctor configuration (bad goal, bad stat, bad profile) -> exit 5."""


# ---------------------------------------------------------------------------
# Report structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One typed observation. ``severity`` is ``"info"``/``"warn"``/``"error"``; only the code and
    the numbers are contractual, the ``detail`` sentence is for the human."""

    code: str
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class SourceDiagnosis:
    """Did the source open, and did evidence actually arrive?"""

    uri: str | None
    opened: bool
    frames: int
    error: str | None = None
    first_ts: float | None = None
    last_ts: float | None = None
    requested_fps: float | None = None
    gate_frame: str | None = None
    timebase_requested: str | None = None
    timebase_resolved: str | None = None
    timebase_reason: str | None = None

    @property
    def duration_s(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def delivered_fps(self) -> float | None:
        """Frames per SOURCE second actually delivered — the number that tells you whether the
        capture rate you asked for is the rate the gate really saw."""
        d = self.duration_s
        if d <= 0 or self.frames < 2:
            return None
        return (self.frames - 1) / d

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri, "opened": self.opened, "frames": self.frames, "error": self.error,
            "first_ts": self.first_ts, "last_ts": self.last_ts, "duration_s": self.duration_s,
            "requested_fps": self.requested_fps, "delivered_fps": self.delivered_fps,
            "gate_frame": self.gate_frame,
            "timebase": {
                "requested": self.timebase_requested, "resolved": self.timebase_resolved,
                "reason": self.timebase_reason,
            },
        }


@dataclass(frozen=True)
class StatDiagnosis:
    """One salience statistic for one goal, against ITS OWN effective threshold.

    ``max_value``/``median_value``/``mean_value`` describe the statistic's distribution over the run;
    ``effective_threshold`` is what ``compile_watchspec`` actually produced for THIS goal under THIS
    stat (base cutoff + the goal's ``motion_bias``); ``admitted`` is what the REAL
    :class:`~percept.harness.tier0.MotionGate` escalated with that condition. Comparing ``max_value``
    across rows is meaningful only against each row's own threshold — the scales differ.
    """

    stat: str
    effective_threshold: float
    admitted: int
    max_value: float = 0.0
    median_value: float = 0.0
    mean_value: float = 0.0
    first_admit_ts: float | None = None
    active: bool = False

    @property
    def headroom(self) -> float:
        """``max_value - effective_threshold``: how far the loudest frame was from clearing. Negative
        means nothing in the whole run came close; a small negative means a near miss."""
        return self.max_value - self.effective_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "stat": self.stat, "effective_threshold": self.effective_threshold,
            "admitted": self.admitted, "max_value": self.max_value,
            "median_value": self.median_value, "mean_value": self.mean_value,
            "headroom": self.headroom, "first_admit_ts": self.first_admit_ts, "active": self.active,
        }


@dataclass(frozen=True)
class GoalDiagnosis:
    """Everything the dry run learned about ONE goal."""

    goal_id: str
    condition: str
    active_stat: str
    gate_on: str
    ego_compensate: bool
    cadence_s: float | None
    scheduled: bool
    region_hint: bool
    detectors: tuple[str, ...]
    admitted: int
    suppressions: dict[str, int]
    stats: tuple[StatDiagnosis, ...]
    observable: bool = True
    unobservable_reason: str | None = None

    @property
    def active(self) -> StatDiagnosis | None:
        for s in self.stats:
            if s.active:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id, "condition": self.condition, "active_stat": self.active_stat,
            "gate_on": self.gate_on, "ego_compensate": self.ego_compensate,
            "cadence_s": self.cadence_s, "scheduled": self.scheduled,
            "region_hint": self.region_hint, "detectors": list(self.detectors),
            "admitted": self.admitted, "suppressions": dict(self.suppressions),
            "stats": [s.to_dict() for s in self.stats],
            "observable": self.observable, "unobservable_reason": self.unobservable_reason,
        }


@dataclass(frozen=True)
class PlanDiagnosis:
    """What was compiled, from which profile, and which admission signals could feed it."""

    plan_hash: str | None
    plan_mode: str
    goal_ids: tuple[str, ...]
    salience_stat: str
    stat_thresholds: dict[str, float]
    profile_name: str | None
    profile_key: tuple[str, str, str] | None
    profile_source: str
    source_profile: str | None
    source_profile_note: str
    signals: dict[str, bool]
    signal_notes: dict[str, str]
    conditions: tuple[dict[str, Any], ...] = ()
    #: The full ``SourceProfile.to_dict()`` — strategies, parameters, stability. ``None``
    #: only if the source-profile registry could not be imported at all.
    source_profile_detail: dict[str, Any] | None = None
    #: How the source profile was chosen: ``"explicit"`` (the caller named it) or ``"default"``.
    source_profile_origin: str = "default"
    #: ``True`` iff :func:`percept.sourceprofile.select_source_profile` flagged this source's URI
    #: as ambiguous enough to name a RECOMMENDATION (screen://-style scheme, source_kind, or
    #: capture_type were all absent) — doctor stayed on the honest default rather than guessing,
    #: and says so both here (structured) and via a :data:`SOURCE_PROFILE_SELECTION_AMBIGUOUS`
    #: finding (prose). Mirrors :class:`~percept.sourceprofile.SourceProfileSelection.recommend`.
    source_profile_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash, "plan_mode": self.plan_mode,
            "goal_ids": list(self.goal_ids), "salience_stat": self.salience_stat,
            "stat_thresholds": dict(self.stat_thresholds),
            "profile": {
                "name": self.profile_name,
                "key": list(self.profile_key) if self.profile_key else None,
                "source": self.profile_source,
            },
            "source_profile": {
                "active": self.source_profile,
                "note": self.source_profile_note,
                "origin": self.source_profile_origin,
                "profile": self.source_profile_detail,
                "recommended_review": self.source_profile_recommended,
            },
            "signals": dict(self.signals), "signal_notes": dict(self.signal_notes),
            "conditions": [dict(c) for c in self.conditions],
        }


@dataclass(frozen=True)
class EnvelopeDiagnosis:
    """ADR-007 — where this deployment sits relative to percept's declared envelope.

    Three facts, all read from TYPED config rather than from prose:

    - the declared ``actuation`` class and whether any compiled goal uses ``modality='act'`` — the
      pair that is out of the envelope (``robotics_control_loop``);
    - the shipped subject-resolver registry and how many of its resolvers identify persons (an
      invariant zero — ``security_person_id`` / ``passive_analytics``);
    - whether "assistant-class delivery only" is enforced in config or only in docs, plus what it
      would take to change that (:data:`~percept.envelope.ATTENTION_CLASS_GAP`).
    """

    check: EnvelopeCheck
    act_goal_ids: tuple[str, ...]
    subject_resolvers: tuple[str, ...]
    person_identifying_resolvers: tuple[str, ...]
    attention_class_enforcement: str = ATTENTION_CLASS_ENFORCEMENT
    attention_class_gap: str = ATTENTION_CLASS_GAP

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "percept.envelope/1",
            "check": self.check.to_dict(),
            "act_goal_ids": list(self.act_goal_ids),
            "subject_resolvers": list(self.subject_resolvers),
            "person_identifying_resolvers": list(self.person_identifying_resolvers),
            "attention_class_enforcement": self.attention_class_enforcement,
            "attention_class_gap": self.attention_class_gap,
            "out_of_envelope": {k: v.to_dict() for k, v in OUT_OF_ENVELOPE.items()},
        }


@dataclass(frozen=True)
class DoctorReport:
    """The whole dry run: source, plan, per-goal admissions, the admission funnel, findings, exit code."""

    source: SourceDiagnosis
    plan: PlanDiagnosis
    goals: tuple[GoalDiagnosis, ...]
    summary: TraceSummary
    findings: tuple[Finding, ...] = ()
    exit_code: int = EXIT_OK
    judge_calls: int = 0
    trace_path: str | None = None
    manifest: RunTraceManifest | None = None
    #: Where the standalone HTML visualizer was written, if one was asked for.
    visualize_path: str | None = None
    #: The envelope panel. Always present; ``None`` only on a hand-built report.
    envelope: EnvelopeDiagnosis | None = None

    @property
    def admitted(self) -> int:
        """Frames admitted by AT LEAST ONE goal — the run-level candidate count."""
        return self.summary.admitted

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "percept.doctor/1",
            "exit_code": self.exit_code,
            "judge_calls": self.judge_calls,
            "source": self.source.to_dict(),
            "plan": self.plan.to_dict(),
            "goals": [g.to_dict() for g in self.goals],
            "funnel": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "trace_path": self.trace_path,
            "visualize_path": self.visualize_path,
            "envelope": self.envelope.to_dict() if self.envelope else None,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self) -> str:
        return render_report(self)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _resolve_source_profile(spec: Any, *, uri: str | None = None) -> tuple[Any, str, bool]:
    """Resolve the :class:`~percept.sourceprofile.SourceProfile` doctor should report.

    ``spec`` is a name, a ``SourceProfile``, or ``None`` (meaning: nobody said, so fall through to
    :func:`percept.sourceprofile.select_source_profile`'s selection rule: STRONG source-adapter
    metadata — here, only the ``uri``'s scheme, since doctor's own surface carries no
    ``source_kind``/``capture_type`` beyond that — infers a profile; anything ambiguous (including a
    plain file path) stays the honest ``physical_scene`` default, never a silent auto-switch).
    Returns ``(profile, origin, recommend)`` — ``recommend`` is ``True`` only for an ambiguous URI
    doctor chose not to act on, so the caller can surface a RECOMMENDATION finding. An unknown
    EXPLICIT name raises :class:`DoctorConfigError` (exit 5): a typo must not silently fall back to
    the wrong source class.
    """
    from .sourceprofile import (  # noqa: PLC0415 - keeps doctor's import graph flat
        SourceProfile,
        SourceProfileNotFoundError,
        get_source_profile,
        select_source_profile,
    )
    from .source_profiles_default import DEFAULT_SOURCE_PROFILE_NAME  # noqa: PLC0415

    if isinstance(spec, SourceProfile):
        return spec, "explicit", False
    if spec is not None and not isinstance(spec, str):
        raise DoctorConfigError(
            f"source_profile must be a name or a SourceProfile, got {type(spec).__name__}"
        )
    selection = select_source_profile(explicit=spec, uri=uri, default=DEFAULT_SOURCE_PROFILE_NAME)
    try:
        return get_source_profile(selection.name), selection.reason, selection.recommend
    except SourceProfileNotFoundError as exc:
        raise DoctorConfigError(f"unknown source profile: {exc}") from exc


def _resolve_timebase_for(source: Any, requested: str):
    """The runtime's OWN timebase resolution — imported, never re-implemented here."""
    from .runtime.worker import _resolve_timebase  # noqa: PLC0415 - runtime extra, lazy

    return _resolve_timebase(source, requested)


def _capture(
    uri: str, *, fps: float, max_frames: int | None, timebase: str
) -> tuple[list[tuple[float, Frame]], str | None]:
    """Pull ``(t, gate_frame)`` pairs through the runtime's real capture path.

    Reuses :func:`percept.runtime.source.open_source` so doctor decodes exactly what a run decodes
    (same downsample, same frame stepping, same ``t`` origin). The judged JPEG each frame carries is
    dropped immediately — a dry run never retains media. Returns ``(frames, error)``; ``error`` is a
    message when the source could not be opened or read, and is never raised at the caller.
    """
    import asyncio  # noqa: PLC0415

    frames: list[tuple[float, Frame]] = []

    async def _pull() -> None:
        from .runtime.source import open_source  # noqa: PLC0415 - runtime extra, lazy

        agen = open_source(uri, fps=fps, timebase=timebase)
        try:
            async for t, gate_frame, _judge_frame in agen:
                frames.append((float(t), gate_frame))
                if max_frames is not None and len(frames) >= max_frames:
                    break
        finally:
            await agen.aclose()

    try:
        asyncio.run(_pull())
    except Exception as exc:  # noqa: BLE001 — a source failure is a REPORT, not a traceback
        return frames, f"{type(exc).__name__}: {exc}"
    return frames, None


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------


def _coerce_goals(goals: Iterable[Any] | None) -> list[Goal]:
    """Accept ``Goal``s or plain condition strings; anything else is an invalid configuration."""
    out: list[Goal] = []
    for i, g in enumerate(goals or ()):
        if isinstance(g, Goal):
            out.append(g)
        elif isinstance(g, str) and g.strip():
            out.append(Goal(id=f"g{i}", condition=g.strip(), say=g.strip()))
        else:
            raise DoctorConfigError(f"goal #{i} must be a Goal or a non-empty condition string, got {g!r}")
    return out


def _profile_settings(
    *,
    goal_temporality: Any,
    camera_mount: Any,
    start_state: Any,
    base: Settings | None,
) -> tuple[Settings, str, str | None, tuple[str, str, str] | None, str]:
    """Resolve the situation axes into REAL settings via the shipped profile seams.

    Returns ``(settings, plan_mode, profile_name, profile_key, profile_source)``. When no axis is
    given, the deployment's own :func:`~percept.config.load_settings` values are used unchanged and
    ``profile_source`` is ``"settings"`` — doctor never invents a profile the run would not have had.
    """
    if goal_temporality is None and camera_mount is None and start_state is None:
        settings = base if base is not None else load_settings()
        return settings, "single", None, None, "settings"
    from .profile_resolution import resolve_profile  # noqa: PLC0415
    from .profile_wiring import resolve_watch_profile  # noqa: PLC0415
    from .profiles_default import seed  # noqa: PLC0415
    from .watchprofile import DEFAULT_REGISTRY, WatchProfileError  # noqa: PLC0415

    seed(DEFAULT_REGISTRY)
    try:
        profile, _overrides = resolve_profile(goal_temporality, camera_mount, start_state)
        resolved = resolve_watch_profile(profile, base=base)
    except WatchProfileError as exc:
        raise DoctorConfigError(f"profile resolution failed: {exc}") from exc
    return resolved.settings, resolved.plan_mode, profile.name, profile.key, "profile_resolution"


def _stat_thresholds(settings: Settings) -> dict[str, float]:
    """The per-stat calibrated cutoffs, read off the SAME fields the worker compiles with.

    THREE separate numbers, deliberately: ``block_max``/``p95`` sit on a higher value scale than the
    whole-frame ``mean`` and are never compared to the mean's cutoff.
    """
    return {
        "mean": float(settings.edge.motion_gate),
        "block_max": float(settings.edge.motion_gate_block_max),
        "p95": float(settings.edge.motion_gate_p95),
    }


def _resolved_policy(settings: Settings) -> dict[str, Any]:
    """The resolved threshold/policy knobs that belong in the plan identity.

    Read off the SAME :class:`~percept.config.Settings` fields the worker compiles with — see
    :data:`percept.tracing.PLAN_POLICY_FIELDS`. Two dry runs that differ only in a cutoff are two
    different configurations, and their ``plan_hash`` must say so.
    """
    return {
        "motion_gate": float(settings.edge.motion_gate),
        "motion_gate_block_max": float(settings.edge.motion_gate_block_max),
        "motion_gate_p95": float(settings.edge.motion_gate_p95),
        "salience_stat": str(settings.edge.salience_stat),
        "ego_compensate": bool(settings.edge.ego_compensate),
        "require_false_first": bool(settings.gate.require_false_first),
        "refractory_s": float(settings.gate.refractory_s),
    }


def _compile(goals: Sequence[Goal], settings: Settings, *, stat: str | None = None) -> WatchSpec:
    """The REAL :func:`compile_watchspec`, optionally forcing a deployment-wide salience stat."""
    return compile_watchspec(
        goals,
        threshold=settings.edge.motion_gate,
        salience_stat=stat or settings.edge.salience_stat,
        ego_compensate=settings.edge.ego_compensate,
        stat_thresholds=_stat_thresholds(settings),
    )


def _condition_for(spec: WatchSpec, goal_id: str) -> WatchCondition | None:
    for c in spec.conditions:
        if c.goal_id == goal_id:
            return c
    return None


def _replay(
    condition: WatchCondition, frames: Sequence[tuple[float, Frame]], settings: Settings
) -> tuple[list[bool], list[str]]:
    """Replay ONE condition through the REAL gate; return ``(escalated, reason)`` per frame.

    The admissions come from :class:`~percept.harness.tier0.MotionGate` itself (a one-condition
    spec), so a doctor number can never drift from what the runtime would have done. The REASON for
    each non-admit is attributed here, in the gate's own precedence order (schedule, then cadence,
    then the channel's value against the condition's threshold), and any frame the gate refused that
    this attribution cannot explain is reported as ``"unexplained"`` rather than silently bucketed.
    """
    gate = MotionGate(WatchSpec(conditions=[condition]), settings=settings)
    escalated: list[bool] = []
    reasons: list[str] = []
    last_escalate = -math.inf
    prev: Frame | None = None
    for i, (t, frame) in enumerate(frames):
        decision = gate.evaluate(frame, ts=t)
        escalated.append(decision.escalate)
        if decision.escalate:
            last_escalate = t
            reasons.append("admitted")
            prev = frame
            continue
        if not schedule_active(condition.schedule, None):
            reasons.append("out_of_schedule")
        elif condition.cadence_s is not None and t - last_escalate < condition.cadence_s:
            reasons.append("cadence_not_elapsed")
        elif condition.gate_on in ("structural", "semantic"):
            reasons.append("signal_unavailable")
        elif i == 0 or prev is None:
            reasons.append("no_predecessor_frame")
        elif condition.gate_on == "fixed":
            reasons.append("cadence_not_elapsed")
        else:
            reasons.append("below_threshold")
        prev = frame
    return escalated, reasons


def run_doctor(
    *,
    source: str | None = None,
    frames: Sequence[tuple[float, Frame]] | None = None,
    goals: Iterable[Any] | None = None,
    fps: float = 2.0,
    max_frames: int | None = None,
    timebase: str = "auto",
    goal_temporality: Any = None,
    camera_mount: Any = None,
    start_state: Any = None,
    source_profile: Any = None,
    actuation: str = "none",
    settings: Settings | None = None,
    trace_out: str | None = None,
    visualize_out: str | None = None,
    visualize_media: bool = False,
    golden: Any = None,
    trace_records: Any = None,
) -> DoctorReport:
    """Run the tier-0 dry run and build the :class:`DoctorReport`. Spends no judge call, ever.

    Either ``source`` (a cv2-readable URI decoded through the runtime's own capture path) or
    ``frames`` (an explicit ``[(t, Frame), ...]`` stream, which is how the tests drive it) must be
    given. ``goals`` accepts :class:`~percept.adapters.Goal` objects or plain condition strings.

    ``trace_records`` (optional) is a client's recorded :class:`~percept.tracing.TraceRecord` stream: when
    supplied, doctor runs the pure evidence-link audit (:func:`percept.replay.verify_evidence_links`) over
    it and adds an ``EVIDENCE_LINK_BROKEN`` finding if any delivered Incident fails to resolve to exactly
    one judgment record. The dry run itself spends no judge call, so this audit only ever concerns the
    trace the caller handed in — never doctor's own recorder.

    Raises :class:`DoctorConfigError` for an invalid configuration; every other failure — including
    a source that will not open — is REPORTED (with an exit code), not raised.
    """
    if (source is None) == (frames is None):
        raise DoctorConfigError("give exactly one of source= or frames=")
    goal_list = _coerce_goals(goals)
    if not goal_list:
        raise DoctorConfigError("at least one goal is required — doctor is plan-AWARE by definition")
    if timebase not in ("auto", "media", "wall"):
        raise DoctorConfigError(f"timebase must be auto/media/wall, got {timebase!r}")

    # The envelope declaration is CONFIG, so a bad value is an exit-5 config error, exactly
    # like a bad timebase — never a silently-ignored string.
    act_goal_ids = tuple(
        g.id for g in goal_list if getattr(g, "modality", "speak") == "act"
    )
    try:
        envelope_check = actuation_check(
            modality="act" if act_goal_ids else "speak", actuation=actuation
        )
    except EnvelopeViolation as exc:
        raise DoctorConfigError(str(exc)) from exc

    src_profile, src_profile_origin, src_profile_recommend = _resolve_source_profile(
        source_profile, uri=source
    )

    cfg, plan_mode, profile_name, profile_key, profile_source = _profile_settings(
        goal_temporality=goal_temporality, camera_mount=camera_mount,
        start_state=start_state, base=settings,
    )

    # -- 1. the source ----------------------------------------------------
    findings: list[Finding] = []
    error: str | None = None
    tb_requested = tb_resolved = tb_reason = None
    if source is not None:
        tb = _resolve_timebase_for(source, timebase)
        tb_requested, tb_resolved, tb_reason = tb.requested, tb.resolved, tb.reason
        captured, error = _capture(source, fps=fps, max_frames=max_frames, timebase=tb.resolved)
    else:
        captured = [(float(t), f) for t, f in (frames or ())]
        if max_frames is not None:
            captured = captured[:max_frames]
        tb_requested, tb_resolved, tb_reason = timebase, "media", "deterministic-stream"

    src = SourceDiagnosis(
        uri=source, opened=error is None, frames=len(captured), error=error,
        first_ts=captured[0][0] if captured else None,
        last_ts=captured[-1][0] if captured else None,
        requested_fps=fps if source is not None else None,
        gate_frame=f"{captured[0][1].w}x{captured[0][1].h}" if captured else None,
        timebase_requested=tb_requested, timebase_resolved=tb_resolved, timebase_reason=tb_reason,
    )
    if error is not None:
        findings.append(Finding(SOURCE_FAILED, "error", f"source did not open/read: {error}"))
    elif not captured:
        findings.append(Finding(NO_FRAMES, "error", "the source opened but delivered zero frames"))
    elif len(captured) == 1:
        findings.append(Finding(
            SINGLE_FRAME_RUN, "warn",
            "only one frame arrived; motion needs a predecessor, so nothing can admit on motion",
        ))

    # -- 2. the plan ------------------------------------------------------
    spec = _compile(goal_list, cfg)
    compiled_ids = tuple(c.goal_id for c in spec.conditions)
    if not spec.conditions:
        findings.append(Finding(
            PLAN_EMPTY, "error",
            "no goal compiled to a WatchCondition — every goal's state['type'] is outside "
            "watch/process, so the tier-0 gate has nothing to look for",
        ))

    signals = {
        "motion": True,
        "structural": False,
        "semantic": False,
        "audio": False,
    }
    signal_notes = {
        "motion": "computed here from the decoded gate frames (pure CV, no model call)",
        "structural": "0.0 — no structural adapter is wired in the reference gate (percept-edge CLIP/pose)",
        "semantic": "0.0 — no semantic adapter is wired in the reference gate (percept-edge CLIP)",
        "audio": "no audio source was opened by this dry run; energy-gated goals cannot be observed",
    }

    # -- 3. per-goal replay, per stat ------------------------------------
    thresholds = _stat_thresholds(cfg)
    specs_by_stat = {stat: _compile(goal_list, cfg, stat=stat) for stat in SALIENCE_STATS}
    # ONE pass over the pixels per frame pair produces all three statistics (they are pure selectors
    # over the same per-block score list) — the distributions below cost what one gate pass costs.
    per_frame_stats: list[dict[str, float]] = []
    prev: Frame | None = None
    for _t, f in captured:
        per_frame_stats.append(salience_stats(prev, f))
        prev = f

    goal_diags: list[GoalDiagnosis] = []
    admitted_idx: set[int] = set()   # frame INDEXES (not ts: two frames may share a timestamp)
    for goal in goal_list:
        active_condition = _condition_for(spec, goal.id)
        if active_condition is None:
            continue
        stat_rows: list[StatDiagnosis] = []
        active_reasons: dict[str, int] = {}
        active_admitted = 0
        for stat in SALIENCE_STATS:
            cond = _condition_for(specs_by_stat[stat], goal.id)
            if cond is None:  # pragma: no cover - defensive: same goals compiled with same rules
                continue
            escalated, reasons = _replay(cond, captured, cfg)
            values = [s[stat] for s in per_frame_stats]
            is_active = stat == active_condition.salience_stat
            first_ts = next((t for (t, _f), e in zip(captured, escalated) if e), None)
            stat_rows.append(StatDiagnosis(
                stat=stat,
                effective_threshold=float(cond.threshold),
                admitted=sum(1 for e in escalated if e),
                max_value=max(values) if values else 0.0,
                median_value=statistics.median(values) if values else 0.0,
                mean_value=statistics.fmean(values) if values else 0.0,
                first_admit_ts=first_ts,
                active=is_active,
            ))
            if is_active:
                active_admitted = sum(1 for e in escalated if e)
                for i, r in enumerate(reasons):
                    if r == "admitted":
                        admitted_idx.add(i)
                    else:
                        active_reasons[r] = active_reasons.get(r, 0) + 1

        observable = True
        unobservable_reason: str | None = None
        if active_condition.gate_on in ("structural", "semantic") and not signals[active_condition.gate_on]:
            observable = False
            unobservable_reason = (
                f"goal gates on '{active_condition.gate_on}', which no adapter feeds in this "
                "topology — the signal is structurally 0.0, so the plan cannot be observed"
            )
        goal_diags.append(GoalDiagnosis(
            goal_id=goal.id,
            condition=goal.condition,
            active_stat=active_condition.salience_stat,
            gate_on=active_condition.gate_on,
            ego_compensate=bool(active_condition.ego_compensate),
            cadence_s=active_condition.cadence_s,
            scheduled=active_condition.schedule is not None,
            region_hint=active_condition.region_hint is not None,
            detectors=tuple(active_condition.detectors or ()),
            admitted=active_admitted,
            suppressions=active_reasons,
            stats=tuple(stat_rows),
            observable=observable,
            unobservable_reason=unobservable_reason,
        ))

    # -- 4. the funnel, derived from the trace summary (never recounted here) -----------
    manifest = RunTraceManifest.create(
        source=source or "stream",
        driver="doctor",
        # The plan identity carries the RESOLVED cutoffs it ran under, not just the goals.
        # A doctor run at threshold 0.10 and one at 0.20 are two different configurations and must
        # never quote the same plan_hash.
        plan=PlanIdentity.from_goals(goal_list, plan_mode=plan_mode, policy=_resolved_policy(cfg)),
        components=(
            ComponentIdentity(role="vision", name="tier0-motion-gate", model=None),
            ComponentIdentity(role="profile", name=profile_name or "settings"),
        ),
        policy=TracePolicy.diagnostic(),
        baseline=BaselineAssumption.every_observation(len(captured)),
        # The per-frame salience values below are meaningless without the cutoffs they
        # were judged against, and a visualizer must never invent a threshold. Carrying them in the
        # manifest makes the trace self-describing, so the counterfactual panel ("what would
        # mean/block_max/p95 each have decided against ITS OWN threshold") is READ, not recomputed.
        labels={"stat_thresholds": dict(thresholds), "salience_stat": cfg.edge.salience_stat},
    )
    recorder = TraceRecorder(manifest)
    for i, (t, _f) in enumerate(captured):
        values = per_frame_stats[i]
        recorder.record(
            OBSERVATION_STAGE, source_ts=t, trace_id=f"{manifest.source}@{i}",
            escalate=i in admitted_idx, **values,
        )
        if i in admitted_idx:
            recorder.record(ADMISSION_STAGE, source_ts=t, trace_id=f"{manifest.source}@{i}")
    recorder.close()
    summary = recorder.summary()
    trace_path = str(recorder.write_json(trace_out)) if trace_out else None

    # The visualizer renders THIS recorder's manifest/records/summary — the same objects
    # this report prints — so the HTML and the terminal panel can never disagree.
    viz_path: str | None = None
    if visualize_out:
        import pathlib as _pathlib  # noqa: PLC0415

        from .visualize import VizOptions, render_html  # noqa: PLC0415

        page = render_html(
            manifest, list(recorder.records), summary,
            options=VizOptions(include_media=bool(visualize_media),
                               title=f"percept doctor — {source or 'stream'}"),
            golden=golden,
        )
        out = _pathlib.Path(visualize_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        viz_path = str(out)

    # -- 5. findings + exit code -----------------------------------------
    total_admitted = summary.admitted
    if captured and total_admitted == 0:
        findings.append(Finding(
            ZERO_ADMITS_ON_FIXTURE, "warn",
            f"no goal admitted a single candidate across {len(captured)} frames. This is DIAGNOSTIC, "
            "not automatically a failure: a quiet scene under a motion-gated watch correctly admits "
            "nothing and saves every model call. Check the per-stat table — if an alternative "
            "statistic WOULD have admitted, the eye is mis-selected, not the scene",
        ))
    for g in goal_diags:
        if not g.observable:
            findings.append(Finding(SIGNAL_UNAVAILABLE, "error", f"{g.goal_id}: {g.unobservable_reason}"))
        active = g.active
        better = [s for s in g.stats if not s.active and s.admitted > (active.admitted if active else 0)]
        if better:
            names = ", ".join(f"{s.stat} would admit {s.admitted}" for s in better)
            findings.append(Finding(
                ALTERNATIVE_STAT_WOULD_ADMIT, "warn",
                f"{g.goal_id}: active stat '{g.active_stat}' admitted {g.admitted}; {names} "
                "(each against its OWN calibrated threshold)",
            ))
        if g.ego_compensate:
            findings.append(Finding(
                EGO_COMPENSATED_GATE, "info",
                f"{g.goal_id}: ego-compensation is ON, so the gate escalates on the whole-frame "
                "ego RESIDUAL, not on the block statistics printed above. The admitted counts are "
                "still the real gate's; the value columns describe the uncompensated frames",
            ))
        if g.suppressions.get("unexplained"):
            findings.append(Finding(
                UNEXPLAINED_SUPPRESSION, "error",
                f"{g.goal_id}: {g.suppressions['unexplained']} non-admits could not be attributed to "
                "a rule — this is a doctor bug, please report it with this JSON",
            ))

    # An OPTIONAL evidence-link audit: only when the caller SUPPLIED a trace (a client's .ptrace). The
    # dry run itself spends no judge call and emits no judgment.event/incident.created, so this never
    # fires on doctor's own recorder — it verifies the trace a client handed over resolves every
    # delivered Incident back to EXACTLY ONE judgment record (percept.replay.verify_evidence_links). A
    # broken link is an ERROR-severity finding but not its own exit code: it is a data-integrity fact
    # about the supplied trace, surfaced loudly rather than folded into a pass/fail the run doesn't own.
    if trace_records is not None:
        from .replay import verify_evidence_links  # noqa: PLC0415 — optional, avoids an import at module load

        link_report = verify_evidence_links(trace_records)
        if not link_report.ok:
            kinds = ", ".join(sorted({v.kind for v in link_report.violations}))
            findings.append(Finding(
                EVIDENCE_LINK_BROKEN, "error",
                f"supplied trace: {len(link_report.violations)} of {link_report.checked} verifiable "
                f"incident(s) did not resolve to exactly one judgment.event ({kinds}); "
                f"{link_report.legacy_unverifiable} legacy incident(s) carried no link and were not checked",
            ))

    # The source profile is a REPORTED fact, and an untuned admission signal is a finding,
    # not a footnote. It is deliberately a WARN, never an error and never an exit code of its own:
    # coasting is a real (degraded) operating mode, and the client — not percept — owns the fix.
    if not src_profile.admission_tuned:
        findings.append(Finding(
            ADMISSION_SIGNAL_UNTUNED, "warn",
            f"source profile '{src_profile.name}' is running on COAST ALONE — supply a change "
            f"signal. {src_profile.admission_note()} percept's stock tier-0 eye is calibrated on "
            "physical_scene only; on this source class the salience table above is expected to "
            "read near-zero, and lowering a threshold to force an admit would wreck physical-scene "
            "precision without making this content observable.",
        ))
    if src_profile.stability == "experimental":
        findings.append(Finding(
            SOURCE_PROFILE_EXPERIMENTAL, "warn",
            f"source profile '{src_profile.name}' is EXPERIMENTAL — evaluation only. "
            + src_profile.notes,
        ))
    if src_profile_recommend:
        # Selection rule: the URI was ambiguous (a plain file path can be surveillance, handheld,
        # or a recorded meeting) so doctor never silently switched profiles — it stayed on the honest
        # default and says so, loudly, as a RECOMMENDATION rather than an error.
        findings.append(Finding(
            SOURCE_PROFILE_SELECTION_AMBIGUOUS, "info",
            f"source '{source}' did not carry strong-enough metadata (a screen:// scheme, "
            "source_kind='screen', or a capture_type) to infer a source profile, so this run used "
            f"the default '{src_profile.name}'. If this source is actually screen/document content, "
            "pass source_profile='screen_document' explicitly — percept never auto-switches on a "
            "plain file path.",
        ))

    # ADR-007: the envelope is read from TYPED config, and doctor is the tool that says it.
    person_id_resolvers = tuple(
        r.name for r in DEFAULT_SUBJECT_RESOLVERS.person_identifying()
    )
    envelope = EnvelopeDiagnosis(
        check=envelope_check,
        act_goal_ids=act_goal_ids,
        subject_resolvers=tuple(DEFAULT_SUBJECT_RESOLVERS.names()),
        person_identifying_resolvers=person_id_resolvers,
    )
    if not envelope_check.in_envelope:
        findings.append(Finding(OUT_OF_ENVELOPE_ACTUATION, "error", envelope_check.note))
    elif envelope_check.verdict is not None:
        # actuation='physical' declared but nothing acts — a warn, not an error: the declaration is
        # doing its job (ConsentProfile.permits() refuses every action while it stands).
        findings.append(Finding(OUT_OF_ENVELOPE_ACTUATION, "warn", envelope_check.note))
    if person_id_resolvers:
        findings.append(Finding(
            PERSON_IDENTIFYING_RESOLVER, "error",
            "person-identifying subject resolvers are registered in this process: "
            f"{sorted(person_id_resolvers)}. percept ships none and derives no person identity "
            "from pixels; this came from a third-party registration. "
            + OUT_OF_ENVELOPE["security_person_id"].sentence(),
        ))

    exit_code = EXIT_OK
    if not envelope_check.in_envelope or person_id_resolvers:
        exit_code = EXIT_OUT_OF_ENVELOPE
    elif src.error is not None or not captured:
        exit_code = EXIT_SOURCE_FAILURE
    elif not spec.conditions or any(not g.observable for g in goal_diags):
        exit_code = EXIT_PLAN_UNOBSERVABLE
    elif total_admitted == 0:
        exit_code = EXIT_ZERO_COVERAGE

    plan = PlanDiagnosis(
        plan_hash=manifest.plan.plan_hash,
        plan_mode=plan_mode,
        goal_ids=compiled_ids,
        salience_stat=cfg.edge.salience_stat,
        stat_thresholds=thresholds,
        profile_name=profile_name,
        profile_key=profile_key,
        profile_source=profile_source,
        source_profile=src_profile.name,
        source_profile_note=(
            f"{src_profile.stability}; modalities={sorted(src_profile.modalities)}, "
            f"temporal_unit={src_profile.temporal_unit}, "
            f"admission={src_profile.admission_strategy}, "
            f"evidence={src_profile.evidence_strategy}, "
            f"identity={src_profile.identity_strategy}, "
            f"timebase={src_profile.default_timebase} "
            f"[{src_profile_origin}]. {src_profile.admission_note()}"
        ),
        source_profile_detail=src_profile.to_dict(),
        source_profile_origin=src_profile_origin,
        source_profile_recommended=src_profile_recommend,
        signals=signals,
        signal_notes=signal_notes,
        conditions=tuple(
            {
                "goal_id": c.goal_id, "threshold": c.threshold, "salience_stat": c.salience_stat,
                "gate_on": c.gate_on, "cadence_s": c.cadence_s,
                "ego_compensate": c.ego_compensate,
                "scheduled": c.schedule is not None, "region_hint": c.region_hint is not None,
                "detectors": list(c.detectors or ()),
            }
            for c in spec.conditions
        ),
    )

    return DoctorReport(
        source=src, plan=plan, goals=tuple(goal_diags), summary=summary,
        findings=tuple(findings), exit_code=exit_code, judge_calls=0,
        trace_path=trace_path, manifest=manifest, visualize_path=viz_path,
        envelope=envelope,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_W = 78


def _rule(title: str = "") -> str:
    if not title:
        return "─" * _W
    head = f"── {title} "
    return head + "─" * max(0, _W - len(head))


def _fmt(v: Any, spec: str = ".4f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def render_funnel(summary: TraceSummary) -> str:
    """The primary-metrics funnel panel — every number READ off :class:`TraceSummary`."""
    lines = [_rule("FUNNEL (primary metrics, percept.tracing.TraceSummary)")]
    rows = [
        ("observed", f"{summary.observed} {summary.observed_unit}"),
        ("admitted", f"{summary.admitted}  (evidence union {summary.admitted_evidence_union_s:.2f}s)"),
        ("judge started", str(summary.judge_started)),
        ("judge completed", str(summary.judge_completed)),
        ("judge degraded/dropped", f"{summary.judge_degraded} / {summary.judge_dropped}"),
        ("gate outcomes", ", ".join(f"{k}={v}" for k, v in sorted(summary.gate_outcomes.items())) or "—"),
        ("incidents", str(summary.incidents)),
        ("watch duration", f"{summary.watch_duration_s:.2f}s (healthy {summary.source_healthy_s:.2f}s)"),
        # "—", never "$0.0000": an unpriced/unreported run must not read as a free one.
        ("cost", f"{summary.cost_text()}  {summary.tokens_text()}"),
        ("savings", summary.savings_text()),
        ("records", f"retained={summary.records_retained} dropped={summary.records_dropped}"),
    ]
    for label, value in rows:
        lines.append(f"  {label:<24}{value}")
    lines.append("  (a dry run judges nothing: judge counts are structurally 0 and savings is '—',")
    lines.append("   because with no actual call there is no measured token cost to project.)")
    return "\n".join(lines)


def render_report(report: DoctorReport) -> str:
    """The human panel set. Machine consumers should read :meth:`DoctorReport.to_dict` instead."""
    s = report.source
    out: list[str] = [_rule("PERCEPT DOCTOR — tier-0 dry run (zero judge spend)")]

    out.append(_rule("SOURCE"))
    out.append(f"  uri                     {s.uri or '<in-memory stream>'}")
    out.append(f"  opened                  {'yes' if s.opened else 'NO'}"
               + (f"   ({s.error})" if s.error else ""))
    out.append(f"  frames received         {s.frames}")
    if s.frames:
        out.append(f"  source t range          {_fmt(s.first_ts, '.3f')} … {_fmt(s.last_ts, '.3f')}  "
                   f"({s.duration_s:.2f}s)")
        out.append(f"  gate frame              {s.gate_frame}")
        out.append(f"  fps requested/delivered {_fmt(s.requested_fps, '.2f')} / {_fmt(s.delivered_fps, '.2f')}")
    out.append(f"  timebase                {s.timebase_requested} -> {s.timebase_resolved}  "
               f"({s.timebase_reason})")

    p = report.plan
    out.append(_rule("PLAN"))
    out.append(f"  plan_hash               {p.plan_hash}")
    out.append(f"  plan_mode               {p.plan_mode}")
    out.append(f"  compiled goals          {len(p.conditions)}  {list(p.goal_ids)}")
    out.append(f"  deployment salience     {p.salience_stat}")
    out.append(f"  profile                 {p.profile_name or '(none — deployment settings)'}"
               f"   [{p.profile_source}]")
    if p.profile_key:
        out.append(f"  profile key             {tuple(p.profile_key)}")
    out.append(f"  source profile          {p.source_profile or 'not available'}   "
               f"[{p.source_profile_origin}]")
    if p.source_profile_recommended:
        out.append(
            "    ** RECOMMENDATION: source metadata was ambiguous — see FINDINGS "
            f"[{SOURCE_PROFILE_SELECTION_AMBIGUOUS}] below before trusting this profile **"
        )
    d = p.source_profile_detail or {}
    if d:
        out.append(f"    stability             {d.get('stability')}")
        out.append(f"    modalities/unit       {', '.join(d.get('modalities', ()))} / "
                   f"{d.get('temporal_unit')}")
        out.append(f"    admission strategy    {d.get('admission_strategy')}"
                   f"{'' if d.get('admission_tuned') else '   ** UNTUNED **'}")
        out.append(f"    evidence / identity   {d.get('evidence_strategy')} / "
                   f"{d.get('identity_strategy')}")
        coast = (d.get("parameters") or {}).get("coast_interval_s")
        out.append(f"    coast / timebase      "
                   f"{'none' if coast is None else f'every {coast:g}s'} / "
                   f"{d.get('default_timebase')}")
    for line in _wrap(p.source_profile_note, _W - 6):
        out.append(f"      {line}")

    out.append(_rule("ADMISSION SIGNALS"))
    for name, ok in p.signals.items():
        out.append(f"  {name:<12}{'available' if ok else 'UNAVAILABLE':<14}{p.signal_notes[name]}")

    for g in report.goals:
        out.append(_rule(f"GOAL {g.goal_id}"))
        out.append(f"  condition               {g.condition}")
        out.append(f"  active stat / gate_on   {g.active_stat} / {g.gate_on}"
                   + ("   [ego-compensated]" if g.ego_compensate else ""))
        out.append(f"  cadence / schedule      {_fmt(g.cadence_s, '.2f')} / "
                   f"{'armed by schedule' if g.scheduled else 'always armed'}")
        out.append(f"  admitted                {g.admitted}")
        out.append("  SALIENCE STATS — each against ITS OWN calibrated threshold (different scales,")
        out.append("  never one cutoff): mean is whole-frame; block_max/p95 are localized blocks.")
        out.append(f"    {'stat':<11}{'max':>9}{'median':>10}{'mean':>10}"
                   f"{'threshold':>12}{'admitted':>10}   ")
        for st in g.stats:
            mark = " <- active" if st.active else ""
            out.append(f"    {st.stat:<11}{st.max_value:>9.4f}{st.median_value:>10.4f}"
                       f"{st.mean_value:>10.4f}{st.effective_threshold:>12.3f}{st.admitted:>10}{mark}")
        if g.suppressions:
            out.append("  WHICH RULE SUPPRESSED THE NON-ADMITS")
            for reason in SUPPRESSION_REASONS:
                n = g.suppressions.get(reason)
                if n:
                    out.append(f"    {reason:<24}{n}")
        if not g.observable:
            out.append(f"  ! UNOBSERVABLE           {g.unobservable_reason}")

    out.append(render_funnel(report.summary))

    e = report.envelope
    if e is not None:
        out.append(_rule("ENVELOPE (ADR-007)"))
        out.append(f"  actuation declared      {e.check.actuation}"
                   + ("" if e.check.in_envelope else "   ** OUT OF ENVELOPE **"))
        out.append(f"  goals with modality=act {len(e.act_goal_ids)}  {list(e.act_goal_ids)}")
        out.append(f"  subject resolvers       {len(e.subject_resolvers)} shipped; "
                   f"{len(e.person_identifying_resolvers)} identify persons "
                   f"(percept derives NO person identity from pixels)")
        out.append(f"  assistant-class only    enforced in {e.attention_class_enforcement.upper()} "
                   f"— AttentionPolicy clamps interrupt AUTHORITY, not consumer CLASS")
        for line in _wrap(e.check.note, _W - 6):
            out.append(f"      {line}")

    out.append(_rule("FINDINGS"))
    if not report.findings:
        out.append("  none — the run completed with no hard incompatibility.")
    for f in report.findings:
        out.append(f"  [{f.severity.upper():<5}] {f.code}")
        for line in _wrap(f.detail, _W - 4):
            out.append(f"      {line}")
    if report.trace_path:
        out.append(f"  trace artifact written: {report.trace_path}")
    if report.visualize_path:
        out.append(f"  visualizer written:     {report.visualize_path}")
    out.append(f"  judge calls spent: {report.judge_calls}   exit code: {report.exit_code}")
    out.append(_rule())
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="percept-doctor",
        description=(
            "Tier-0 dry run: did the source open, what plan compiled, what admitted, and WHICH RULE "
            "suppressed the rest. Pure CV — no API key, no judge call, no cost."
        ),
    )
    # NOT required=True: --visualize-from renders an ALREADY-CAPTURED trace/bundle and opens no
    # source at all. main() enforces "exactly one of --source / --visualize-from".
    p.add_argument("--source", default=None, help="a video path / rtsp:// URL / webcam:0")
    p.add_argument("--goal", action="append", default=[], help="a plain-language condition (repeatable)")
    p.add_argument("--fps", type=float, default=2.0, help="capture rate to pull (default 2.0)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames (default: all)")
    p.add_argument("--timebase", default="auto", choices=("auto", "media", "wall"))
    p.add_argument("--goal-temporality", default=None, choices=("stative", "dynamic"))
    p.add_argument("--camera-mount", default=None, choices=("fixed", "wearable", "glasses", "handheld"))
    p.add_argument("--start-state", default=None, choices=("start_anchored", "transition"))
    # Deliberately NOT choices=(...): the source-profile registry is OPEN (ADR-005), so a
    # third-party profile (industrial_line, medical_imaging, ...) must be selectable here the moment
    # it is registered, without editing this parser. An unknown name is a clear exit-5 error.
    p.add_argument(
        "--source-profile", default=None,
        help="source class: physical_scene | screen_document | conversation, or any "
             "registered third-party profile name (default: physical_scene)",
    )
    p.add_argument(
        "--actuation", default="none", choices=ACTUATION_CLASSES,
        help="ADR-007: what an act can physically do in this deployment. "
             "none (default) | screen (advisory software acts — the narrowed survivor) | "
             "physical (OUT of percept's envelope; ConsentProfile.permits() then refuses every "
             "action and doctor exits 6)",
    )
    p.add_argument("--trace-out", default=None, help="write the trace artifact (JSON) here")
    p.add_argument("--json", action="store_true", help="emit the machine-readable report on stdout")
    # -- the native visualizer (no second console-script family; these are flags here) --
    p.add_argument(
        "--visualize-out", default=None,
        help="write the standalone HTML visualizer here (lanes over media time, click-through "
             "decision chains, per-stat counterfactual). Opens from file:// with no network.",
    )
    p.add_argument(
        "--visualize-from", default=None,
        help="render an EXISTING trace JSON or .ptrace bundle instead of capturing a source "
             "(requires --visualize-out; opens no camera and spends nothing)",
    )
    p.add_argument(
        "--visualize-media", action="store_true",
        help="OPT IN to embedding media bytes in the HTML. Default OFF — the page never carries "
             "pixels and never fetches a source URI unless you ask for it.",
    )
    p.add_argument(
        "--golden", default=None,
        help="ground-truth episodes (JSON) for the drift lane; scored by the deterministic "
             "percept.harness.golden_eval matcher",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``percept-doctor`` entry point. Exit codes: 0 ok, 2 zero coverage, 3 source failure,
    4 plan unobservable, 5 invalid configuration, 6 declared out of envelope."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    # -- render-only mode. No source is opened, no goal is compiled, nothing is spent. --
    if args.visualize_from:
        from .visualize import VisualizeError, VizOptions, visualize_trace  # noqa: PLC0415

        if not args.visualize_out:
            print("percept-doctor: --visualize-from requires --visualize-out", file=sys.stderr)
            return EXIT_INVALID_CONFIG
        if args.source:
            print("percept-doctor: give either --source or --visualize-from, not both",
                  file=sys.stderr)
            return EXIT_INVALID_CONFIG
        try:
            written = visualize_trace(
                args.visualize_from, args.visualize_out,
                options=VizOptions(include_media=bool(args.visualize_media),
                                   title=f"percept trace — {args.visualize_from}"),
                golden=args.golden,
            )
        except VisualizeError as exc:
            print(f"percept-doctor: {exc}", file=sys.stderr)
            return EXIT_INVALID_CONFIG
        print(f"visualizer written: {written}")
        return EXIT_OK

    if not args.source:
        print("percept-doctor: --source is required (or --visualize-from to render an existing "
              "trace)", file=sys.stderr)
        return EXIT_INVALID_CONFIG
    if not args.goal:
        print("percept-doctor: at least one --goal is required (doctor is plan-aware)", file=sys.stderr)
        return EXIT_INVALID_CONFIG
    try:
        report = run_doctor(
            source=args.source,
            goals=args.goal,
            fps=args.fps,
            max_frames=args.max_frames,
            timebase=args.timebase,
            goal_temporality=args.goal_temporality,
            camera_mount=args.camera_mount,
            start_state=args.start_state,
            source_profile=args.source_profile,
            actuation=args.actuation,
            trace_out=args.trace_out,
            visualize_out=args.visualize_out,
            visualize_media=args.visualize_media,
            golden=args.golden,
        )
    except DoctorConfigError as exc:
        print(f"percept-doctor: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_INVALID_CONFIG
    print(report.to_json() if args.json else report.render())
    return report.exit_code


__all__ = [
    "EXIT_OK",
    "EXIT_ZERO_COVERAGE",
    "EXIT_SOURCE_FAILURE",
    "EXIT_PLAN_UNOBSERVABLE",
    "EXIT_INVALID_CONFIG",
    "EXIT_OUT_OF_ENVELOPE",
    "OUT_OF_ENVELOPE_ACTUATION",
    "PERSON_IDENTIFYING_RESOLVER",
    "EnvelopeDiagnosis",
    "SUPPRESSION_REASONS",
    "ZERO_ADMITS_ON_FIXTURE",
    "ALTERNATIVE_STAT_WOULD_ADMIT",
    "ADMISSION_SIGNAL_UNTUNED",
    "SOURCE_PROFILE_EXPERIMENTAL",
    "SOURCE_PROFILE_SELECTION_AMBIGUOUS",
    "SIGNAL_UNAVAILABLE",
    "DoctorConfigError",
    "DoctorReport",
    "Finding",
    "GoalDiagnosis",
    "PlanDiagnosis",
    "SourceDiagnosis",
    "StatDiagnosis",
    "build_parser",
    "main",
    "render_funnel",
    "render_report",
    "run_doctor",
]
