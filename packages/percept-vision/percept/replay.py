"""COGNITION REPLAY — recorded verdicts fed back through the Gate. Deterministic, and only that.

This module ships exactly ONE of the two things people mean by "replay", and refuses to imply the
other. ADR-004 (``docs/adr/ADR-004-delivery-flow-replay.md`` §2) is the normative text; this is its
implementation.

**COGNITION REPLAY (this module).** Take the NORMALIZED verdicts a run already recorded
(``satisfied`` / ``confidence`` / ``freshness``, with their ``source_ts`` and their
``(goal_id, entity_id)`` key) out of a :class:`~percept.tracing.TraceRecord` stream, and push them
back through a freshly constructed :class:`~percept.gate.Gate` and the incident-emission rule. Every
layer involved is pure: :func:`~percept.gate.classify` is arithmetic, ``Gate.decide`` is a state
machine over ``(key, truth, now)``, and "a delivered FIRE becomes an incident" is a mapping. So the
same trace in produces BYTE-IDENTICAL decisions out — :attr:`ReplayReport.digest` is a content hash
over the whole decision sequence, and ``test_cognition_replay.py`` asserts it.

**MEDIA REEVALUATION (NOT this module).** Sending recorded media at a live model again is not
deterministic — sampling, model drift, provider routing and timeouts all move the verdict, one layer
BELOW the gate, where no amount of gate-side purity can help. That belongs to ``percept eval``.
Nothing here calls a model, and a bundle that carries only a media URI does not "replay
the run" — it lets you re-run perception and get a possibly different answer.

What replay needs from the trace
--------------------------------
A ``gate.classified`` record carries the gate's OUTPUT (``truth``/``action``/``reason``) and now
also its INPUTS (``satisfied``/``confidence``/``freshness``/``entity_id``/``hold_s``).
Those inputs are what get re-fed. The gate's own knobs come from
:attr:`~percept.tracing.RunTraceManifest.gate_config` (``threshold`` / ``fresh_min`` /
``require_false_first`` / ``refractory_s``); if the trace never recorded them,
:attr:`ReplayReport.gate_config_recorded` is ``False`` and the report says the run was replayed
against library defaults rather than pretending otherwise.

What replay deliberately does NOT reconstruct
---------------------------------------------
Authored composition that lives on the GOAL, not in the trace: ``fire_when(n)`` accumulation,
``suppress_false`` never-goal keying, and the hard-quadrant ``FIRE``->``ASK`` escalation
(:mod:`percept.behavior`). A caller who knows those can declare them per goal with
:class:`ReplayGoalConfig`; otherwise they are treated as absent, and any resulting difference from
the recorded run shows up honestly in :attr:`ReplayReport.divergences` instead of being smoothed
over.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .gate import ASK, FIRE, SILENT, Gate, GateAction, classify
from .tracing import (
    STAGE_GATE_CLASSIFIED,
    RunTraceManifest,
    TraceRecord,
    canonical_json,
    content_hash,
)

#: Version of the replay contract (the decision tuple + digest definition). A change that could move
#: a digest for an unchanged trace bumps this.
REPLAY_VERSION = "1.0.0"

#: The gate knobs a replay rebuilds from. Anything outside this set is NOT reconstructed.
GATE_CONFIG_KEYS: tuple[str, ...] = ("threshold", "fresh_min", "require_false_first", "refractory_s")


def gate_config_of(gate: Any) -> dict[str, Any]:
    """Snapshot a live :class:`~percept.gate.Gate`'s decision knobs for the trace manifest.

    Pass the result as ``RunTraceManifest.create(gate_config=...)`` at run start; the replayer
    rebuilds the gate from exactly these numbers.
    """
    return {k: getattr(gate, k, None) for k in GATE_CONFIG_KEYS}


@dataclass(frozen=True)
class ReplayGoalConfig:
    """Authored per-goal behaviour a TRACE cannot know, declared explicitly for a replay.

    ``escalates`` is the hard-quadrant rule (a FIRE is delivered outward as an ASK and therefore
    creates no incident — :mod:`percept.behavior`); ``suppress_false`` is the never-goal keying
    (a confident-false never reaches the gate, and the key is exempt from eviction);
    ``hold_s`` overrides the dwell recorded on the record. All default to "not declared", which is
    the honest posture for a trace handed over by a client.
    """

    escalates: bool = False
    suppress_false: bool = False
    hold_s: float | None = None


@dataclass(frozen=True)
class ReplayInput:
    """One normalized verdict, ready to be re-fed to the gate. No media, no model, no I/O."""

    seq: int
    trace_id: str
    goal_id: str
    entity_id: str | None
    source_ts: float
    satisfied: bool
    confidence: float
    freshness: float = 1.0
    hold_s: float | None = None
    #: The judge failed on this tick (fail-closed): the gate must NOT see it at all — it degrades to
    #: an ASK-class abstention with the edge state left exactly as it was (``agent.py``).
    degraded: bool = False
    #: What the ORIGINAL run recorded as its outward action, when the trace has it. Used only to
    #: report divergence; never fed into the replayed decision.
    recorded_action: str | None = None

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.goal_id, self.entity_id)


@dataclass(frozen=True)
class ReplayedDecision:
    """One replayed gate decision — the unit the digest is computed over."""

    seq: int
    trace_id: str
    goal_id: str
    entity_id: str | None
    source_ts: float
    truth: str
    action: GateAction            # the GATE's action
    reason: str                   # the gate's typed WHY (GateReason), or "judge_degraded"
    outward_action: str | None    # what a subscriber would receive (None = nothing delivered)
    incident: bool                # an incident is created iff the outward action is a FIRE

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "trace_id": self.trace_id, "goal_id": self.goal_id,
            "entity_id": self.entity_id, "source_ts": self.source_ts, "truth": self.truth,
            "action": self.action, "reason": self.reason,
            "outward_action": self.outward_action, "incident": self.incident,
        }


@dataclass(frozen=True)
class Divergence:
    """The replayed outward action differed from what the run recorded. Reported, never hidden."""

    seq: int
    goal_id: str
    entity_id: str | None
    source_ts: float
    recorded_action: str
    replayed_action: str | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "goal_id": self.goal_id, "entity_id": self.entity_id,
            "source_ts": self.source_ts, "recorded_action": self.recorded_action,
            "replayed_action": self.replayed_action, "note": self.note,
        }


@dataclass(frozen=True)
class ReplayReport:
    """The result of a cognition replay: the decisions, their digest, and what was NOT reconstructed.

    :attr:`digest` is a content hash over :attr:`decisions` — the determinism claim in machine-
    checkable form. Two replays of the same trace with the same :class:`ReplayGoalConfig` map MUST
    produce the same digest; if they ever do not, that is a bug in this module, not a tolerance to
    widen.
    """

    run_id: str
    replay_version: str
    decisions: tuple[ReplayedDecision, ...]
    digest: str
    inputs: int = 0
    incidents: int = 0
    gate_outcomes: dict[str, int] = field(default_factory=dict)
    divergences: tuple[Divergence, ...] = ()
    #: ``False`` when the trace never recorded the gate's knobs — the replay then ran against
    #: library defaults, which is stated rather than silently assumed.
    gate_config_recorded: bool = True
    gate_config: dict[str, Any] = field(default_factory=dict)
    #: Records that LOOKED like gate decisions but carried no usable verdict (an older trace, or a
    #: retention level that dropped the inputs). Counted, never silently skipped.
    unreplayable: int = 0

    @property
    def deterministic(self) -> bool:
        """This replay used no live model and no wall clock. Always true here — it is a property of
        the mechanism, kept as a field so a caller can assert on it explicitly."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "replay_version": self.replay_version,
            "kind": "cognition_replay",
            "deterministic": True,
            "digest": self.digest,
            "inputs": self.inputs,
            "unreplayable": self.unreplayable,
            "incidents": self.incidents,
            "gate_outcomes": dict(self.gate_outcomes),
            "gate_config_recorded": self.gate_config_recorded,
            "gate_config": dict(self.gate_config),
            "decisions": [d.to_dict() for d in self.decisions],
            "divergences": [d.to_dict() for d in self.divergences],
        }


def replay_inputs(records: Iterable[TraceRecord]) -> list[ReplayInput]:
    """Extract the normalized verdicts from a recorded stream, in ingest order.

    Only ``gate.classified`` records carry BOTH the gate's inputs and the run's outward action, so
    that is the one stage read here. A record without ``confidence``/``satisfied`` (an older trace,
    or one whose retention dropped them) is not guessed at — it is left out and counted as
    unreplayable by :func:`replay_cognition`.
    """
    out: list[ReplayInput] = []
    for r in records:
        if r.stage != STAGE_GATE_CLASSIFIED:
            continue
        d = r.data
        goal_id = r.goal_id or d.get("goal_id")
        if not goal_id:
            continue
        degraded = bool(d.get("degraded", False))
        sat, conf = d.get("satisfied"), d.get("confidence")
        if not degraded and not isinstance(conf, (int, float)):
            continue  # no usable verdict — counted upstream, never invented
        out.append(
            ReplayInput(
                seq=r.seq,
                trace_id=r.trace_id,
                goal_id=str(goal_id),
                entity_id=(str(d["entity_id"]) if d.get("entity_id") is not None else None),
                source_ts=float(r.source_ts if r.source_ts is not None else 0.0),
                satisfied=bool(sat),
                confidence=float(conf or 0.0),
                freshness=float(d.get("freshness", 1.0) or 0.0),
                hold_s=(float(d["hold_s"]) if isinstance(d.get("hold_s"), (int, float)) else None),
                degraded=degraded,
                recorded_action=(str(d["action"]) if d.get("action") is not None else None),
            )
        )
    return out


def _unreplayable_count(records: Iterable[TraceRecord], used: int) -> int:
    total = sum(1 for r in records if r.stage == STAGE_GATE_CLASSIFIED)
    return max(0, total - used)


def replay_cognition(
    manifest: RunTraceManifest,
    records: Sequence[TraceRecord],
    *,
    goal_config: Mapping[str, ReplayGoalConfig] | None = None,
    gate_config: Mapping[str, Any] | None = None,
) -> ReplayReport:
    """Re-run a recorded run's verdicts through a fresh Gate. Pure, offline, deterministic.

    The gate is built from ``gate_config`` (explicit) or ``manifest.gate_config`` (recorded) or —
    stated in the report — library defaults. Eviction is disabled for the replay
    (``max_keys=None, key_ttl_s=None``): a replay must reproduce the DECISION logic, and letting a
    capacity bound silently re-arm an edge would make the same trace replay differently at a
    different key volume.
    """
    cfg_source = dict(gate_config) if gate_config is not None else dict(manifest.gate_config or {})
    recorded = bool(cfg_source)
    cfg = {k: cfg_source.get(k) for k in GATE_CONFIG_KEYS}
    gate = Gate(
        threshold=float(cfg["threshold"]) if cfg["threshold"] is not None else 0.7,
        fresh_min=float(cfg["fresh_min"]) if cfg["fresh_min"] is not None else 0.5,
        max_keys=None,
        key_ttl_s=None,
        require_false_first=bool(cfg["require_false_first"]),
        refractory_s=(float(cfg["refractory_s"]) if cfg["refractory_s"] else None),
    )
    goals = dict(goal_config or {})

    inputs = replay_inputs(records)
    decisions: list[ReplayedDecision] = []
    divergences: list[Divergence] = []
    outcomes: dict[str, int] = {FIRE: 0, ASK: 0, SILENT: 0}
    incidents = 0

    for item in inputs:
        gcfg = goals.get(item.goal_id, ReplayGoalConfig())
        if item.degraded:
            # Fail-closed: the gate never sees a degraded tick, so its edge state is
            # untouched; the tick degrades to an ASK-class abstention.
            action: GateAction = ASK
            reason: str = "judge_degraded"
            truth = "unknown"
            outward: str | None = ASK
        else:
            verdict = _StubVerdict(item.satisfied, item.confidence)
            truth = classify(verdict, item.freshness, gate.threshold, gate.fresh_min)
            if gcfg.suppress_false and truth == "no":
                # never-contract: a confident-false must not reach the gate at all
                decisions.append(ReplayedDecision(
                    seq=item.seq, trace_id=item.trace_id, goal_id=item.goal_id,
                    entity_id=item.entity_id, source_ts=item.source_ts, truth=truth,
                    action=SILENT, reason="not_satisfied", outward_action=None, incident=False,
                ))
                outcomes[SILENT] += 1
                _note_divergence(divergences, item, None)
                continue
            outcome = gate.decide(
                item.key, verdict, freshness=item.freshness,
                hold_s=gcfg.hold_s if gcfg.hold_s is not None else item.hold_s,
                now=item.source_ts, no_evict=gcfg.suppress_false,
            )
            action = outcome.action
            reason = str(outcome.reason)
            outward = None
            if action == FIRE:
                outward = ASK if gcfg.escalates else FIRE
            elif action == ASK:
                outward = ASK
        outcomes[action] = outcomes.get(action, 0) + 1
        is_incident = outward == FIRE
        if is_incident:
            incidents += 1
        decisions.append(ReplayedDecision(
            seq=item.seq, trace_id=item.trace_id, goal_id=item.goal_id, entity_id=item.entity_id,
            source_ts=item.source_ts, truth=truth, action=action, reason=reason,
            outward_action=outward, incident=is_incident,
        ))
        _note_divergence(divergences, item, outward)

    payload = [d.to_dict() for d in decisions]
    return ReplayReport(
        run_id=manifest.run_id,
        replay_version=REPLAY_VERSION,
        decisions=tuple(decisions),
        digest=content_hash({"replay_version": REPLAY_VERSION, "decisions": payload}),
        inputs=len(inputs),
        incidents=incidents,
        gate_outcomes=outcomes,
        divergences=tuple(divergences),
        gate_config_recorded=recorded,
        gate_config={k: cfg[k] for k in GATE_CONFIG_KEYS},
        unreplayable=_unreplayable_count(records, len(inputs)),
    )


def _note_divergence(sink: list[Divergence], item: ReplayInput, outward: str | None) -> None:
    """Compare the replayed OUTWARD action against what the run recorded, when it recorded one.

    The recorded ``action`` on a ``gate.classified`` record is the outward action the run delivered
    (``fire``/``ask``), or ``silent`` when nothing was delivered — so ``None`` here means
    ``"silent"``.
    """
    if item.recorded_action is None:
        return
    replayed = outward or SILENT
    if replayed == item.recorded_action:
        return
    sink.append(Divergence(
        seq=item.seq, goal_id=item.goal_id, entity_id=item.entity_id, source_ts=item.source_ts,
        recorded_action=item.recorded_action, replayed_action=replayed,
        note="replayed gate decision differs from the recorded run — check gate_config and "
             "ReplayGoalConfig (fire_when/suppress_false/hard-quadrant escalation are authored on "
             "the goal and are not carried by a trace)",
    ))


@dataclass(frozen=True)
class _StubVerdict:
    """The two fields :func:`~percept.gate.classify` and ``Gate.decide`` actually read.

    Deliberately NOT a :class:`~percept.adapters.Verdict`: a replay must not be able to smuggle
    evidence, media or a reason string into a gate decision. If the gate ever needs a third field,
    that is a contract change and it should fail loudly here rather than pick up a default.
    """

    satisfied: bool
    confidence: float


def replay_report_json(report: ReplayReport, *, indent: int | None = 2) -> str:
    """Serialize a report. ``indent=None`` gives the canonical, digest-stable form."""
    if indent is None:
        return canonical_json(report.to_dict())
    import json  # noqa: PLC0415

    return json.dumps(report.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# The front door: load an artifact a shipped command actually produces
# ---------------------------------------------------------------------------


class ReplayLoadError(Exception):
    """``percept-replay`` was handed something that is not a percept trace artifact."""


def load_replayable(path: Any) -> tuple[RunTraceManifest, list[TraceRecord]]:
    """Load ``(manifest, records)`` from a trace JSON **or** a portable ``.ptrace`` bundle.

    The two artifacts a client can actually hand over: the JSON written by
    ``percept-watch --trace out.json`` (:meth:`~percept.tracing.TraceRecorder.write_json`) and the
    ``run-123.ptrace`` directory/archive from :mod:`percept.trace_bundle`. Anything else raises
    :class:`ReplayLoadError` rather than half-loading.
    """
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(path)
    if not p.exists():
        raise ReplayLoadError(f"no such trace: {p}")
    if p.is_dir() or p.suffix in (".ptrace", ".zip") or p.name.endswith(".ptrace.zip"):
        from .trace_bundle import BundleError, TraceBundle  # noqa: PLC0415

        try:
            bundle = TraceBundle.read(p)
        except BundleError as exc:
            raise ReplayLoadError(str(exc)) from exc
        return bundle.manifest, list(bundle.records)
    from .tracing import load_trace  # noqa: PLC0415

    try:
        manifest, records, _summary = load_trace(p.read_text(encoding="utf-8"))
    except (ValueError, KeyError, TypeError) as exc:
        raise ReplayLoadError(f"not a percept trace artifact: {p} ({exc})") from exc
    return manifest, records


def format_report(report: ReplayReport, *, trace: str = "") -> str:
    """The human-readable replay panel — the digest first, the divergences last and unhidden."""
    lines: list[str] = []
    lines.append("─" * 68)
    lines.append(f"  PERCEPT · cognition replay   ·   run_id={report.run_id}")
    if trace:
        lines.append(f"  trace: {trace}")
    lines.append("─" * 68)
    lines.append(f"  digest          {report.digest}")
    lines.append(f"  replay_version  {report.replay_version}   ·   deterministic={report.deterministic}")
    lines.append(f"  inputs          {report.inputs} replayable"
                 + (f"   ·   {report.unreplayable} unreplayable (no recorded verdict)"
                    if report.unreplayable else ""))
    outcomes = "  ".join(f"{k}={v}" for k, v in sorted(report.gate_outcomes.items()))
    lines.append(f"  gate outcomes   {outcomes or '(none)'}")
    lines.append(f"  incidents       {report.incidents}")
    if report.gate_config_recorded:
        cfg = "  ".join(f"{k}={report.gate_config.get(k)!r}" for k in GATE_CONFIG_KEYS)
        lines.append(f"  gate config     {cfg}   (recorded)")
    else:
        lines.append("  gate config     NOT RECORDED — replayed against library defaults. Record it "
                     "with RunTraceManifest.create(gate_config=gate_config_of(core.gate)).")
    lines.append("─" * 68)
    if not report.divergences:
        lines.append("  ✓ 0 divergences — every replayed outward action matches the recorded run.")
    else:
        lines.append(f"  ✗ {len(report.divergences)} divergence(s):")
        for d in report.divergences:
            lines.append(
                f"      seq={d.seq} goal={d.goal_id}"
                + (f" entity={d.entity_id}" if d.entity_id else "")
                + f" t={d.source_ts}  recorded={d.recorded_action}  replayed={d.replayed_action}"
            )
        lines.append("    " + report.divergences[0].note)
    lines.append("─" * 68)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``percept-replay TRACE`` — re-run a recorded run's verdicts through a fresh Gate.

    Exit code is the contract: ``0`` when every replayed outward action matches what the run
    recorded, ``1`` when any diverges (so CI can assert it), ``2`` for a load failure.
    """
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        prog="percept-replay",
        description="Replay a recorded percept trace through the Gate. Deterministic, offline, "
                    "no model call. Produce the trace with `percept-watch --trace out.json`.",
    )
    p.add_argument("trace", help="a trace JSON, or a .ptrace bundle directory/archive")
    p.add_argument("--json", action="store_true", help="emit the ReplayReport as JSON instead of a panel")
    p.add_argument("--escalates", action="append", default=[], metavar="GOAL_ID",
                   help="declare a goal whose FIRE is delivered outward as an ASK (repeatable)")
    p.add_argument("--suppress-false", action="append", default=[], metavar="GOAL_ID",
                   help="declare a never-goal: a confident-false never reaches the gate (repeatable)")
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        manifest, records = load_replayable(args.trace)
    except ReplayLoadError as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
        return 2

    goal_ids = set(args.escalates) | set(args.suppress_false)
    goal_config = {
        gid: ReplayGoalConfig(
            escalates=gid in set(args.escalates),
            suppress_false=gid in set(args.suppress_false),
        )
        for gid in goal_ids
    }
    report = replay_cognition(manifest, records, goal_config=goal_config or None)
    print(replay_report_json(report) if args.json else format_report(report, trace=str(args.trace)))
    return 1 if report.divergences else 0


__all__ = [
    "GATE_CONFIG_KEYS",
    "REPLAY_VERSION",
    "Divergence",
    "ReplayGoalConfig",
    "ReplayInput",
    "ReplayLoadError",
    "ReplayReport",
    "ReplayedDecision",
    "format_report",
    "gate_config_of",
    "load_replayable",
    "main",
    "replay_cognition",
    "replay_inputs",
    "replay_report_json",
]
