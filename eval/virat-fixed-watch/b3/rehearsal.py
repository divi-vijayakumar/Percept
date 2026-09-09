#!/usr/bin/env python3
"""Zero-paid production capture/runtime rehearsal on exposed VIRAT media."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import pathlib
import tempfile
import time

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.harness.webhook_probe import LoopbackWebhookProbe
from percept.persistence import SqlitePersistence
from percept.runtime import RuntimeStats, run
from percept.runtime.source import open_source
from percept.source_profiles_default import PHYSICAL_SCENE
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder


class SequenceVision:
    """Local deterministic judge used only to prove wiring, never perception quality."""

    model = "local/sequence-rehearsal"

    def __init__(self) -> None:
        self.calls = 0
        self._states = (False, True, True, False, True, False)

    async def judge(self, _condition, _frame, **_kwargs) -> Verdict:
        state = self._states[self.calls] if self.calls < len(self._states) else False
        self.calls += 1
        return Verdict(
            satisfied=state,
            confidence=1.0,
            evidence=f"scripted rehearsal state={state}",
            usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )

    async def detect(self, _frame, _goals):
        return []


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def percentile(value: float) -> float:
        position = (len(ordered) - 1) * value
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(ordered),
    }


async def rehearse(media: pathlib.Path) -> dict[str, object]:
    if "calibration-exposed" not in str(media.parent):
        raise ValueError("B3 rehearsal accepts only explicitly exposed calibration media")
    vision = SequenceVision()
    core = Percept.create(
        vision=vision,
        discover_plugins=False,
        source_profile=PHYSICAL_SCENE,
    )
    core.add_goal(
        Goal(
            id="f2-z1-rehearsal",
            condition="an eligible person enters the monitored area",
            say="Eligible person entered the monitored area.",
            state={
                "type": "watch",
                "identity_policy": "scene",
                "cadence": {"gate_on": "fixed", "min_judge_interval": 1.0},
            },
        )
    )
    recorder = TraceRecorder(
        RunTraceManifest.create(source=str(media), policy=TracePolicy(level="full"))
    )
    health_events = []
    trace_health = recorder.health_sink()

    def health(event) -> None:
        health_events.append(event)
        trace_health(event)

    stats = RuntimeStats()
    sdk_yield_to_arrival: list[float] = []
    sdk_yield_to_ack: list[float] = []
    arrival_to_ack: list[float] = []
    yielded = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        store = SqlitePersistence(str(pathlib.Path(directory) / "rehearsal.sqlite3"))
        with LoopbackWebhookProbe(hold_acks=True, sink_timeout_s=5.0) as receiver:
            source = open_source(
                str(media),
                fps=2.0,
                queue_policy="block",
                timebase="media",
            )
            async for incident in run(
                core,
                source=source,
                fps=2.0,
                on_incident=receiver.sink,
                on_health=health,
                on_trace=recorder.tracer(),
                stats=stats,
                persistence=store,
                judge_dispatch="inline",
                timebase="media",
            ):
                yielded_at = time.monotonic()
                arrived = await asyncio.to_thread(
                    receiver.wait_for_arrival, incident.id, timeout_s=5.0
                )
                if arrived is None:
                    raise RuntimeError("receiver did not observe incident")
                receiver.release_ack(incident.id)
                acknowledged = await asyncio.to_thread(
                    receiver.wait_for_ack, incident.id, timeout_s=5.0
                )
                if acknowledged is None or acknowledged.ack_at is None:
                    raise RuntimeError("receiver did not ACK incident")
                yielded.append(incident)
                sdk_yield_to_arrival.append(arrived.arrival_at - yielded_at)
                sdk_yield_to_ack.append(acknowledged.ack_at - yielded_at)
                arrival_to_ack.append(acknowledged.ack_at - arrived.arrival_at)

            pending = store.load_pending_deliveries()
            delivery_ledger = store.load("percept_delivery_ledger") or []
            receiver_rows = [receiver.lookup(incident.id) for incident in yielded]
        persisted = store.load("percept")
        store.close()

    recorder.close()
    trace = recorder.summary().to_dict()
    episode_count = sum(
        len(getattr(log, "_episodes", ())) for log in core.episodes.values()
    )
    action_log_count = len(core.action_log.entries())
    receiver_attempts = sum(row.attempts for row in receiver_rows if row is not None)
    receiver_acks = sum(
        row is not None and row.ack_at is not None for row in receiver_rows
    )
    exact = all(
        (
            stats.fires == len(yielded),
            stats.incidents == len(yielded),
            episode_count == len(yielded),
            action_log_count == len(yielded),
            len(pending) == 0,
            len(delivery_ledger) == len(yielded),
            receiver_attempts == len(yielded),
            receiver_acks == len(yielded),
        )
    )
    return {
        "schema": "percept.benchmark.b3-rehearsal.v1",
        "source": {
            "name": media.name,
            "sha256": sha256(media),
            "partition": "calibration_exposed",
        },
        "runtime": {
            "fps_requested": 2.0,
            "queue_policy": "block",
            "timebase": "media",
            "judge_dispatch": "inline",
            "wall_seconds": time.monotonic() - started,
            "stats": {
                name: getattr(stats, name)
                for name in (
                    "frames",
                    "frames_dropped",
                    "escalations",
                    "judge_calls",
                    "fires",
                    "asks",
                    "incidents",
                    "sink_errors",
                    "deliveries_dropped",
                    "signal_admits",
                    "fixed_admits",
                    "coast_admits",
                    "goal_admissions",
                    "perceive_dispatches",
                )
            },
            "health_events": dict(Counter(event.kind for event in health_events)),
        },
        "local_judge": {
            "identity": vision.model,
            "calls": vision.calls,
            "remote_provider_calls": 0,
            "provider_cost_usd": 0.0,
            "all_in_cost_usd": None,
            "all_in_cost_note": "not measured: no hardware/energy rate was frozen for B3",
            "trace_provider_calls_are_local_judge_seam_dispatches": True,
            "quality_claim": False,
        },
        "latency_seconds": {
            "sdk_yield_to_receiver_arrival": summary(sdk_yield_to_arrival),
            "sdk_yield_to_receiver_ack": summary(sdk_yield_to_ack),
            "receiver_arrival_to_ack": summary(arrival_to_ack),
            "source_onset_to_receiver_ack": None,
            "source_onset_note": "not measured: scripted rehearsal states are not GT occurrences",
        },
        "reconciliation": {
            "yielded_incidents": len(yielded),
            "episodes": episode_count,
            "context_action_log_entries": action_log_count,
            "outbox_pending_after_drain": len(pending),
            "delivery_ledger_entries": len(delivery_ledger),
            "receiver_attempts": receiver_attempts,
            "receiver_acks": receiver_acks,
            "persisted_snapshot_present": persisted is not None,
            "exact": exact,
        },
        "trace_summary": trace,
        "qualification_labels_opened": False,
        "paid_activity_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    result = asyncio.run(rehearse(arguments.media))
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
