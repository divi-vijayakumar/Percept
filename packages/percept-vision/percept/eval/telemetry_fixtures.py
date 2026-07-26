"""``percept eval ingest-trace`` — the telemetry -> fixtures closed loop.

A production ``.ptrace`` bundle (:mod:`percept.trace_bundle` + :mod:`percept.tracing`) already
carries every decision a run made: which frames were admitted, what the judge said, what the Gate
classified, and which incidents were actually delivered. What it does NOT carry is whether any of
that was *right*. This module is the bridge from "a client reported a bad run" to a labeled
regression case the derivation harness (:mod:`percept.eval.derivation`, via its fixture loader in
:mod:`percept.eval.differential`) can score — a real miss or false-fire becomes a fixture instead of
a Slack screenshot.

**It never guesses a verdict.** A ``.ptrace`` says WHAT happened; it cannot say whether a delivered
incident was correct, or whether a decision that stayed silent should have fired. Only a human (or an
audited downstream labeller) can say that, via a **labels sidecar** — a flat JSON object
``{incident_or_span_id: verdict}`` where ``verdict`` is one of :data:`VERDICTS`
(``"correct"`` / ``"false_fire"`` / ``"miss"``). Every ``incident.created`` record the bundle
DELIVERED must be covered by the sidecar or ingestion REFUSES outright (:class:`TelemetryFixtureError`)
— an unlabeled delivered incident is never silently dropped or silently assumed correct. A label that
names an id the trace does not contain is refused the same way: this pipeline extracts cases from
telemetry, it does not fabricate them from a sidecar typo.

The id space, and what each kind means
---------------------------------------

- **incident id** (``kind="incident"``) — the ``incident_id`` off a ``incident.created`` record: a
  fire the run actually delivered. Labelable ``"correct"`` (a true positive — a ground-truth episode
  is emitted, and the fire is preserved as the matching prediction) or ``"false_fire"`` (a false
  positive — the fire is preserved as a prediction but NO ground-truth episode is emitted, so the
  derivation harness's matcher scores it an unmatched false fire, exactly as it would a live one).
- **span id** (``kind="span"``) — the ``trace_id`` off a ``gate.classified`` record whose ``action``
  was NOT ``"fire"`` (i.e. the Gate stayed silent or asked): a decision that happened but delivered
  nothing. Labelable ONLY ``"miss"`` — a human who knows the condition WAS actually true at that
  timestamp flags it; a ground-truth episode is emitted with NO matching prediction, so the
  derivation harness's matcher scores it a miss, exactly as a live miss would score.

Every case is stamped with the ``evidence_seq`` that fed it, resolved off the SAME correlation the
recorder itself uses (:data:`~percept.tracing.STAGE_JUDGMENT_EVENT`'s recorder-stamped
``data["evidence_seq"]`` when a judgment was made on this exact ``trace_id``, falling back to the
``source.frame`` record's own ``seq`` for the same ``trace_id`` when no judgment fired) — the literal
evidence resolvable within the SAME ``.ptrace``, never invented.

Emitted fixture shape — the EXISTING format, unowned by this module
---------------------------------------------------------------------

This module writes into the fixture directory shape :func:`percept.eval.differential.load_fixtures`
already reads (and :mod:`percept.eval.derivation` scores against a ladder) — it does not define a new
one::

    <out>/telemetry-<run_id>-<goal_id>/
      ground_truth.json      # goal_shape="point" + one episode per correct/miss case, PLUS a
                              # "provenance" block (source_run_id, build_commit, ...) — an extra
                              # top-level key the existing loader already ignores gracefully
      <arm>/events.json      # one predicted JudgmentEvent per correct/false_fire case (never miss —
                              # nothing was predicted, that is what makes it a miss)
      <arm>/trace.json       # the run's full trace artifact (percept.tracing.TraceRecorder.to_dict
                              # shape), so percept.tracing.load_trace can recover its funnel/cost

One fixture per (run, goal) touched by a labeled case — never one giant fixture, so a later
derivation ladder can score, include, or drop a goal's regression cases independently.

CLI::

    percept eval ingest-trace run-123.ptrace --labels labels.json --out fixtures-dir/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..trace_bundle import BundleError, TraceBundle
from ..tracing import (
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGMENT_EVENT,
    STAGE_SOURCE_FRAME,
    TRACE_SCHEMA_VERSION,
    RunTraceManifest,
    TraceRecord,
    TraceSummary,
)
from .differential import EvalInputError

__all__ = [
    "VERDICTS",
    "DEFAULT_ARM_DIR",
    "DEFAULT_GOAL_SHAPE",
    "DEFAULT_ALLOWED_LATENESS_S",
    "ANNOTATION_VERSION",
    "TelemetryFixtureError",
    "TelemetryCase",
    "load_labels",
    "extract_cases",
    "write_fixture_cases",
    "ingest_trace",
    "build_parser",
    "main",
]

#: The three (and only three) verdicts a human labeller may write. Anything else is a typo, refused.
VERDICTS: tuple[str, ...] = ("correct", "false_fire", "miss")

#: Default arm subdirectory a case's prediction/trace lands in — see the module docstring's tree.
DEFAULT_ARM_DIR = "observed"

#: Telemetry-derived cases are always the matcher's native, fully-SUPPORTED shape (ADR-006): a
#: single fire at a single labeled instant. A `.ptrace` carries no richer episode-window semantics
#: to promote it to "episode", so this module never invents one.
DEFAULT_GOAL_SHAPE = "point"

#: Grace window (seconds) past a case's timestamp a same-key prediction may still land in and match.
#: A telemetry case's ground-truth window is a single instant (the fire/decision timestamp), so
#: without slack a fixture would be scorable only by a byte-identical timestamp; this is deliberately
#: generous (production judge latency is real) and is always overridable via ``--allowed-lateness-s``.
DEFAULT_ALLOWED_LATENESS_S = 5.0

ANNOTATION_VERSION = "telemetry-v1"


class TelemetryFixtureError(EvalInputError):
    """A ``.ptrace``, its labels sidecar, or the pairing between them cannot become a fixture case
    without guessing. Raised INSTEAD of guessing — see the module docstring."""


def _sanitize(value: str) -> str:
    """A bare, filesystem-safe path component — never empty."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class TelemetryCase:
    """One labeled decision span, resolved from a ``.ptrace`` against the labels sidecar.

    ``kind`` is ``"incident"`` (the trace DELIVERED this — an ``incident.created`` record) or
    ``"span"`` (the trace did NOT deliver this — a non-firing ``gate.classified`` record a human
    flags as a miss). ``case_id`` is the labels-sidecar key that resolved to this case.
    """

    case_id: str
    kind: str  # "incident" | "span"
    goal_id: str
    verdict: str  # "correct" | "false_fire" | "miss"
    source_ts: float
    evidence_seq: int | None
    trace_id: str
    from_boundary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "kind": self.kind, "goal_id": self.goal_id,
            "verdict": self.verdict, "source_ts": self.source_ts,
            "evidence_seq": self.evidence_seq, "trace_id": self.trace_id,
            "from_boundary": self.from_boundary,
        }


# ---------------------------------------------------------------------------
# 1. The labels sidecar
# ---------------------------------------------------------------------------


def load_labels(path: str | Path) -> dict[str, str]:
    """Load ``{incident_or_span_id: verdict}``. Refuses a malformed file or an unknown verdict word
    rather than treating a typo as "this case is unlabeled" (which would silently drop it)."""
    p = Path(path)
    if not p.exists():
        raise TelemetryFixtureError(f"labels file not found: {p}")
    try:
        data = json.loads(p.read_text() or "{}")
    except json.JSONDecodeError as exc:
        raise TelemetryFixtureError(f"{p}: not valid JSON ({exc})") from exc
    if not isinstance(data, Mapping):
        raise TelemetryFixtureError(
            f"{p}: labels file must be a JSON object of {{id: verdict}}, got {type(data).__name__}"
        )
    out: dict[str, str] = {}
    bad: list[str] = []
    for key, raw_verdict in data.items():
        verdict = str(raw_verdict)
        if verdict not in VERDICTS:
            bad.append(f"{key!r}: {raw_verdict!r}")
            continue
        out[str(key)] = verdict
    if bad:
        raise TelemetryFixtureError(
            f"{p}: every label must be one of {VERDICTS}; got unrecognized verdict(s): "
            f"{'; '.join(bad)}"
        )
    if not out:
        raise TelemetryFixtureError(f"{p}: labels file has no entries — nothing to ingest")
    return out


# ---------------------------------------------------------------------------
# 2. Extraction — trace records -> labeled cases (refuses rather than guesses)
# ---------------------------------------------------------------------------


def extract_cases(
    records: Sequence[TraceRecord], labels: Mapping[str, str],
) -> tuple[TelemetryCase, ...]:
    """Resolve every labels-sidecar entry against the trace's own decision sequence.

    Two refusals, both typed, both listing exactly what is wrong:

    1. Any ``incident.created`` (a DELIVERED fire) in ``records`` that has no entry in ``labels`` —
       this pipeline never assumes a delivered fire was correct.
    2. Any ``labels`` entry that resolves to neither a known ``incident_id`` nor a non-firing
       ``gate.classified`` record's ``trace_id`` — this pipeline never fabricates a case for an id
       the trace does not contain, nor accepts a verdict/kind mismatch (a fire labeled ``"miss"``,
       or a silent decision labeled ``"correct"``/``"false_fire"``).
    """
    frame_seq_by_trace: dict[str, int] = {}
    judgment_evidence_seq_by_trace: dict[str, int] = {}
    incidents: dict[str, TraceRecord] = {}
    spans: dict[str, TraceRecord] = {}

    for rec in records:
        if rec.stage == STAGE_SOURCE_FRAME:
            frame_seq_by_trace[rec.trace_id] = rec.seq
        elif rec.stage == STAGE_JUDGMENT_EVENT:
            seq = rec.data.get("evidence_seq")
            if isinstance(seq, int):
                judgment_evidence_seq_by_trace[rec.trace_id] = seq
        elif rec.stage == STAGE_INCIDENT_CREATED:
            incident_id = rec.data.get("incident_id")
            if incident_id:
                incidents[str(incident_id)] = rec
        elif rec.stage == STAGE_GATE_CLASSIFIED and rec.data.get("action") != "fire":
            spans.setdefault(rec.trace_id, rec)

    missing = sorted(set(incidents) - set(labels))
    if missing:
        raise TelemetryFixtureError(
            "REFUSING TO GUESS: this .ptrace delivered incident(s) with no verdict in the labels "
            f"sidecar: {missing}. Label every delivered incident 'correct' or 'false_fire' (or omit "
            "it from --labels only if it must be excluded from this ingest) before ingesting."
        )

    def _evidence_seq(trace_id: str) -> int | None:
        if trace_id in judgment_evidence_seq_by_trace:
            return judgment_evidence_seq_by_trace[trace_id]
        return frame_seq_by_trace.get(trace_id)

    cases: list[TelemetryCase] = []
    for case_id, verdict in labels.items():
        if case_id in incidents:
            rec = incidents[case_id]
            if verdict not in ("correct", "false_fire"):
                raise TelemetryFixtureError(
                    f"label {case_id!r} names a DELIVERED incident but verdict is {verdict!r}; a "
                    "delivered fire can only be labeled 'correct' or 'false_fire' — it already fired, "
                    "so 'miss' cannot apply."
                )
            goal_id = str(rec.goal_id or rec.data.get("goal_id") or "")
            if not goal_id:
                raise TelemetryFixtureError(
                    f"label {case_id!r} resolves to an incident with no goal_id — cannot build a "
                    "fixture without knowing which goal it belongs to"
                )
            cases.append(TelemetryCase(
                case_id=case_id, kind="incident", goal_id=goal_id, verdict=verdict,
                source_ts=float(rec.source_ts if rec.source_ts is not None else 0.0),
                evidence_seq=_evidence_seq(rec.trace_id), trace_id=rec.trace_id,
                from_boundary=bool(rec.data.get("from_boundary", False)),
            ))
        elif case_id in spans:
            rec = spans[case_id]
            if verdict != "miss":
                raise TelemetryFixtureError(
                    f"label {case_id!r} names a non-firing decision (action="
                    f"{rec.data.get('action')!r}) but verdict is {verdict!r}; a decision that never "
                    "fired can only be labeled 'miss'."
                )
            if not rec.goal_id:
                raise TelemetryFixtureError(
                    f"label {case_id!r} resolves to a span with no goal_id — cannot build a fixture "
                    "without knowing which goal it belongs to"
                )
            cases.append(TelemetryCase(
                case_id=case_id, kind="span", goal_id=str(rec.goal_id), verdict=verdict,
                source_ts=float(rec.source_ts if rec.source_ts is not None else 0.0),
                evidence_seq=_evidence_seq(rec.trace_id), trace_id=rec.trace_id,
            ))
        else:
            raise TelemetryFixtureError(
                f"label {case_id!r} matches no delivered incident_id and no non-firing decision's "
                "id (trace_id) in this .ptrace — refusing to fabricate a case for an id the trace "
                "does not contain."
            )

    if not cases:
        raise TelemetryFixtureError("no labeled cases resolved from this .ptrace — nothing to ingest")
    return tuple(sorted(cases, key=lambda c: (c.goal_id, c.source_ts, c.case_id)))


# ---------------------------------------------------------------------------
# 3. Writing — cases -> the derivation harness's existing fixture format
# ---------------------------------------------------------------------------


def write_fixture_cases(
    manifest: RunTraceManifest,
    cases: Sequence[TelemetryCase],
    records: Sequence[TraceRecord],
    summary: TraceSummary | None,
    out_dir: str | Path,
    *,
    arm_dir: str = DEFAULT_ARM_DIR,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
) -> tuple[str, ...]:
    """Write one fixture directory per goal touched by ``cases`` under ``out_dir``.

    Layout matches :func:`percept.eval.differential.load_fixtures` exactly (see the module
    docstring). Returns the ``fixture_id``\\ s written, in the order the fixtures loader will read
    them back (sorted).
    """
    if not cases:
        raise TelemetryFixtureError("write_fixture_cases: no cases to write")
    out = Path(out_dir)
    by_goal: dict[str, list[TelemetryCase]] = {}
    for case in cases:
        by_goal.setdefault(case.goal_id, []).append(case)

    trace_payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "manifest": manifest.to_dict(),
        "summary": summary.to_dict() if summary is not None else None,
        "records": [r.to_dict() for r in records],
    }
    trace_text = json.dumps(trace_payload, indent=2, default=str)

    written: list[str] = []
    for goal_id in sorted(by_goal):
        goal_cases = by_goal[goal_id]
        fixture_id = f"telemetry-{_sanitize(manifest.run_id)}-{_sanitize(goal_id)}"
        fixture_dir = out / fixture_id

        # A ground-truth episode exists for every case EXCEPT a false_fire — by definition nothing
        # was supposed to happen there, so no episode is emitted (see the module docstring).
        episodes = [
            {
                "episode_id": case.case_id,
                "event_start_ts": case.source_ts,
                "decision_eligible_ts": case.source_ts,
                "episode_end_ts": case.source_ts,
                "allowed_lateness_s": allowed_lateness_s,
                "annotation_version": ANNOTATION_VERSION,
                "annotator_or_fixture_id": manifest.run_id,
            }
            for case in goal_cases if case.verdict != "false_fire"
        ]
        if not episodes:
            raise TelemetryFixtureError(
                f"goal {goal_id!r} has only false_fire case(s) — a fixture needs at least one "
                "ground-truth episode ('correct' or 'miss') to be scorable; label at least one "
                "correct/miss case for this goal, or drop its false_fire case(s) from this ingest."
            )

        fixture_dir.mkdir(parents=True, exist_ok=True)
        (fixture_dir / "ground_truth.json").write_text(
            json.dumps(
                {
                    "goal_shape": DEFAULT_GOAL_SHAPE,
                    "episodes": episodes,
                    # Extra top-level key — percept.eval.differential.load_fixtures reads only
                    # "goal_shape"/"episodes" and ignores the rest, so this rides along harmlessly
                    # and is exactly the provenance a regression case must carry (the implementation §1).
                    "provenance": {
                        "source": "telemetry",
                        "source_run_id": manifest.run_id,
                        "build_commit": manifest.build_commit,
                        "build_timestamp": manifest.build_timestamp,
                        "percept_version": manifest.percept_version,
                        "goal_id": goal_id,
                        "case_ids": [case.case_id for case in goal_cases],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # A prediction exists for every case EXCEPT a miss — nothing was predicted there, which is
        # exactly what makes it score as a miss against the episode emitted above.
        events = [
            {
                "key": case.case_id,
                "evidence_seq": case.evidence_seq if case.evidence_seq is not None else 0,
                "action": "fire",
                "timing": {
                    "ingest_observed_ts": case.source_ts,
                    "admitted_ts": case.source_ts,
                    "judge_started_ts": case.source_ts,
                    "judge_completed_ts": case.source_ts,
                    "decision_emitted_ts": case.source_ts,
                    "source_media_ts": case.source_ts,
                },
            }
            for case in goal_cases if case.verdict != "miss"
        ]
        arm_path = fixture_dir / arm_dir
        arm_path.mkdir(parents=True, exist_ok=True)
        (arm_path / "events.json").write_text(
            json.dumps({"events": events}, indent=2), encoding="utf-8"
        )
        (arm_path / "trace.json").write_text(trace_text, encoding="utf-8")
        written.append(fixture_id)
    return tuple(written)


# ---------------------------------------------------------------------------
# 4. The whole job
# ---------------------------------------------------------------------------


def ingest_trace(
    bundle_path: str | Path,
    labels_path: str | Path,
    out_dir: str | Path,
    *,
    arm_dir: str = DEFAULT_ARM_DIR,
    allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
) -> tuple[str, ...]:
    """``.ptrace`` + labels sidecar -> written fixture case directories. The whole pipeline."""
    try:
        bundle = TraceBundle.read(bundle_path)
    except BundleError as exc:
        raise TelemetryFixtureError(f"could not read {bundle_path}: {exc}") from exc
    labels = load_labels(labels_path)
    cases = extract_cases(bundle.records, labels)
    return write_fixture_cases(
        bundle.manifest, cases, bundle.records, bundle.summary, out_dir,
        arm_dir=arm_dir, allowed_lateness_s=allowed_lateness_s,
    )


# ---------------------------------------------------------------------------
# 5. CLI — ``percept eval ingest-trace`` (routed from percept.cli)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="percept eval ingest-trace",
        description=(
            "Convert a production .ptrace bundle + a human labels sidecar into labeled derivation "
            "fixture cases (the SAME format percept.eval.derivation scores a ladder against). "
            "Refuses to guess a verdict for any delivered incident, or any labeled id, the trace "
            "does not account for."
        ),
    )
    p.add_argument("bundle", help="path to a .ptrace directory or .ptrace.zip archive")
    p.add_argument(
        "--labels", required=True,
        help="labels sidecar: JSON object of {incident_or_span_id: 'correct'|'false_fire'|'miss'}",
    )
    p.add_argument("--out", required=True, help="fixtures root directory to write into")
    p.add_argument(
        "--arm", default=DEFAULT_ARM_DIR,
        help=f"arm subdirectory each fixture's events.json/trace.json is written under "
             f"(default: {DEFAULT_ARM_DIR!r})",
    )
    p.add_argument(
        "--allowed-lateness-s", type=float, default=DEFAULT_ALLOWED_LATENESS_S,
        help="grace window (seconds) past a case's timestamp a same-key prediction may still land "
             "in and match (default: %(default)s)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``percept eval ingest-trace``. 0 = fixtures written, 2 = refused (nothing written)."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        written = ingest_trace(
            args.bundle, args.labels, args.out,
            arm_dir=args.arm, allowed_lateness_s=args.allowed_lateness_s,
        )
    except (TelemetryFixtureError, BundleError) as exc:
        print(f"percept eval ingest-trace: REFUSED — {exc}", file=sys.stderr)
        return 2
    print(f"percept eval ingest-trace: wrote {len(written)} fixture case dir(s) under {args.out}:")
    for fixture_id in written:
        print(f"  {fixture_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
