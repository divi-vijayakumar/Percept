"""Stage-3 JUDGE POOL validation — SV-3.3 / SV-3.4 / SV-3.5 / SV-3.6.

Boundedness and overload validation for ``percept.backends.judge_pool.JudgePool``.

These tests need **no provider credentials and make no network calls**.  Every
judge is a locally constructed double with an injected delay.  The runtime under
test is the installed ``percept`` wheel, never the repo working tree.

What is being falsified
-----------------------
``judge_pool.py`` claims (module docstring, lines 8-19) that the pool is a
*bounded* router: capacity exhaustion and deadline expiry fail closed to an
UNKNOWN-shaped verdict with an inspectable receipt, and a cancellation-hostile
adapter keeps occupying a bounded route slot instead of leaking.  The claim
under test is therefore **admission-time rejection plus fail-closed shedding**,
NOT unbounded queueing.

* **SV-3.3 Overload allocation** — drive input far faster than the pool can
  serve.  ``max_inflight`` must never be exceeded (measured at the adapter, not
  sampled from the pool's own counters), excess must shed with a ``capacity``
  receipt, the pool's implicit queue must not grow without bound, and nothing
  may be left in flight at shutdown.  Run in two regimes: *sustained* (open-loop
  arrivals at a fixed rate) and *burst* (repeated instantaneous bursts), because
  a real fast-packet stream is bursty and the two regimes can differ.
* **SV-3.4 Deadline authority** — the pool's outer deadline is authoritative
  over any route-declared SLO and over the adapter's own retry mechanics, and a
  provider reply that lands *after* expiry must not resurrect or mutate an
  already-timed-out decision.
* **SV-3.5 Reflex route rejection** — a route declaring
  ``latency_slo_s > deadline_s`` must be rejected AT SELECTION in reflex mode
  (``no_route``, adapter never invoked), not attempted and timed out
  (``deadline``).  The receipt must distinguish the two.
* **SV-3.6 Specialist/audio parity** — every guarantee above must hold on the
  ``audio``, ``fused`` and ``context_stream`` (specialist) evidence shapes, not
  only on ``single_image``.

Run standalone::

    /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python \
        eval/stage-validation/sv3_pool.py

Writes ``results/sv3_pool.json`` next to this file and prints a PASS/FAIL table.
Importable: ``from sv3_pool import run_all`` returns the same result payload.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------------------
# runtime guard — the wheel is the system under test, never the repo working tree
# --------------------------------------------------------------------------------------

_REPO_SRC_MARKER = os.path.join("Percept-sdk", "packages", "percept-vision")


def assert_runtime() -> dict[str, Any]:
    """Refuse to run against anything but an installed ``percept`` distribution."""
    for entry in sys.path:
        if _REPO_SRC_MARKER in os.path.abspath(entry):
            raise RuntimeError(
                "refusing to run: the repo's packages/percept-vision is on sys.path "
                f"({entry!r}); SV-3.x must exercise the installed wheel"
            )
    import percept
    import percept.backends.judge_pool as jp

    path = os.path.abspath(jp.__file__)
    if _REPO_SRC_MARKER in path:
        raise RuntimeError(f"refusing to run: judge_pool imported from repo tree {path!r}")
    return {
        "percept_version": getattr(percept, "__version__", "unknown"),
        "judge_pool_module": path,
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


RUNTIME = assert_runtime()

from percept.audio_input import prepare_audio_judge_binding  # noqa: E402
from percept.backends import (  # noqa: E402
    FakeVision,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.contracts import AudioFrame, Verdict  # noqa: E402

# --------------------------------------------------------------------------------------
# tolerances (explicit, so a marginal result is visible rather than hidden)
# --------------------------------------------------------------------------------------

#: Wall-clock slack allowed above ``deadline_s`` before we call the deadline non-authoritative.
#: Covers event-loop wakeup granularity on a loaded host; measured values are always reported.
DEADLINE_SLACK_S = 0.120

#: An "immediate" admission-time rejection must cost far less than the deadline.
ADMISSION_FAST_FRACTION = 0.25

#: Queue growth tolerance: last-quartile mean outstanding vs second-quartile mean.
QUEUE_GROWTH_FACTOR = 1.5
QUEUE_GROWTH_ABSOLUTE = 10.0

AUDIO_FRAME = AudioFrame(pcm=b"\x00\x01" * 160, sample_rate=16000, channels=1)
EVIDENCE_SHAPE_KEY = "_percept_evidence_shape"


# --------------------------------------------------------------------------------------
# instrumented judge doubles (local, deterministic, zero network)
# --------------------------------------------------------------------------------------


class TrackedJudge:
    """A fixed-latency judge that measures its own true concurrency.

    ``peak_live`` is incremented/decremented around the awaited body with a
    ``finally``, so it counts real provider occupancy including calls killed by
    cancellation.  This is the ground truth for "``max_inflight`` was never
    exceeded" — it does not trust the pool's own bookkeeping.
    """

    def __init__(
        self,
        *,
        name: str = "tracked",
        latency_s: float = 0.02,
        confidence: float = 0.95,
        satisfied: bool = True,
    ) -> None:
        self.name = name
        self.latency_s = float(latency_s)
        self.confidence = float(confidence)
        self.satisfied = bool(satisfied)
        self.calls = 0
        self.completions = 0
        self.cancellations = 0
        self.live = 0
        self.peak_live = 0

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        self.calls += 1
        self.live += 1
        if self.live > self.peak_live:
            self.peak_live = self.live
        try:
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            self.completions += 1
            return Verdict(
                satisfied=self.satisfied,
                confidence=self.confidence,
                evidence=f"{self.name}: scripted",
            )
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        finally:
            self.live -= 1


class HostileJudge:
    """A cancellation-HOSTILE judge: swallows cancellation and replies late.

    Models the adapter class the pool docstring (lines 16-19) says it defends
    against — a provider client whose retry/socket layer does not honour
    cancellation.  The late reply is deliberately *decisive and satisfied* so
    that if the pool could be resurrected by it, SV-3.4 would see a fired
    verdict instead of an UNKNOWN one.
    """

    def __init__(self, *, name: str = "hostile", delay_s: float = 0.9) -> None:
        self.name = name
        self.delay_s = float(delay_s)
        self.calls = 0
        self.cancel_seen = 0
        self.late_returns = 0
        self.late_verdict: Verdict | None = None
        self.finished = asyncio.Event()

    async def _work(self) -> Verdict:
        await asyncio.sleep(self.delay_s)
        verdict = Verdict(
            satisfied=True,
            confidence=0.99,
            evidence=f"{self.name}: LATE reply that must not resurrect a decision",
        )
        self.late_verdict = verdict
        return verdict

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        self.calls += 1
        self.finished.clear()
        inner = asyncio.ensure_future(self._work())
        try:
            try:
                return await asyncio.shield(inner)
            except asyncio.CancelledError:
                self.cancel_seen += 1
                # Refuse to die.  Bounded spin so a wedged test can never hang.
                for _ in range(20_000):
                    if inner.done():
                        break
                    try:
                        await asyncio.sleep(0.005)
                    except asyncio.CancelledError:  # pragma: no cover - re-cancel
                        continue
                self.late_returns += 1
                return inner.result()
        finally:
            self.finished.set()


class MutatingJudge:
    """Returns a Verdict object it retains a reference to, then mutates it later.

    ``Verdict`` is a non-frozen dataclass, so a provider adapter that keeps a
    handle on the object it returned can in principle rewrite a decision the
    caller already acted on.  SV-3.4 asks whether the pool defends the returned
    decision against exactly this.
    """

    def __init__(self, *, name: str = "mutating", latency_s: float = 0.02, confidence: float = 0.4) -> None:
        self.name = name
        self.latency_s = float(latency_s)
        self.confidence = float(confidence)
        self.calls = 0
        self.issued: list[Verdict] = []

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        self.calls += 1
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        verdict = Verdict(
            satisfied=False,
            confidence=self.confidence,
            evidence=f"{self.name}: provider-owned object",
        )
        self.issued.append(verdict)
        return verdict

    def mutate_all(self) -> None:
        for verdict in self.issued:
            verdict.satisfied = True
            verdict.confidence = 0.99
            verdict.evidence = f"{self.name}: MUTATED after the caller returned"


class NeverJudge:
    """Never completes inside any realistic deadline.  Counts invocations."""

    def __init__(self, *, name: str = "never", delay_s: float = 30.0) -> None:
        self.name = name
        self.delay_s = float(delay_s)
        self.calls = 0
        self.cancellations = 0

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        self.calls += 1
        try:
            await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        return Verdict(satisfied=True, confidence=0.99, evidence="never")


# --------------------------------------------------------------------------------------
# receipt accounting + verdict classification
# --------------------------------------------------------------------------------------


class ReceiptTally:
    """Counts EVERY receipt via ``on_receipt``.

    ``pool.receipts()`` is a bounded ``deque`` (by design), so under a
    thousands-of-units overload it cannot be used for accounting.  The callback
    sees every receipt; the bounded deque is separately asserted to stay bounded.
    """

    def __init__(self) -> None:
        self.by_outcome: dict[str, int] = {}
        self.by_route_outcome: dict[str, dict[str, int]] = {}
        self.by_shape_outcome: dict[str, dict[str, int]] = {}
        self.total = 0

    def __call__(self, receipt: Any) -> None:
        self.total += 1
        outcome = receipt.outcome
        self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + 1
        route = receipt.route or "<none>"
        self.by_route_outcome.setdefault(route, {})
        self.by_route_outcome[route][outcome] = self.by_route_outcome[route].get(outcome, 0) + 1
        shape = receipt.evidence_shape
        self.by_shape_outcome.setdefault(shape, {})
        self.by_shape_outcome[shape][outcome] = self.by_shape_outcome[shape].get(outcome, 0) + 1

    def count(self, outcome: str) -> int:
        return self.by_outcome.get(outcome, 0)


def classify_verdict(verdict: Any) -> str:
    """Map a returned Verdict onto the pool's fail-closed vocabulary."""
    evidence = getattr(verdict, "evidence", "") or ""
    if "judge pool unavailable" not in evidence:
        return "served"
    if "all eligible routes were unavailable" in evidence:
        return "shed"
    if "deadline expired" in evidence:
        return "deadline"
    if "no eligible route" in evidence:
        return "no_route"
    return "unavailable_other"


def unknown_shaped(verdict: Any) -> bool:
    return (
        getattr(verdict, "satisfied", None) is False
        and float(getattr(verdict, "confidence", 1.0)) == 0.0
        and "judge pool unavailable" in (getattr(verdict, "evidence", "") or "")
    )


def pool_task_count() -> int:
    """Live asyncio tasks that belong to ``JudgePool._call_route``."""
    count = 0
    for task in asyncio.all_tasks():
        coro = task.get_coro()
        qualname = getattr(coro, "__qualname__", "") or ""
        if "JudgePool._call_route" in qualname and not task.done():
            count += 1
    return count


# --------------------------------------------------------------------------------------
# result plumbing
# --------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    test_id: str
    title: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks) and bool(self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["failed_checks"] = [c.name for c in self.checks if not c.passed]
        return payload


def route(
    name: str,
    adapter: Any,
    *,
    shapes: Iterable[str],
    slo_s: float,
    modes: Iterable[str] = ("reflex", "deliberate"),
    max_inflight: int = 1,
    priority: int = 0,
) -> JudgeRoute:
    return JudgeRoute(
        name=name,
        adapter=adapter,
        capability=JudgeCapability(
            evidence_shapes=frozenset(shapes),
            latency_slo_s=slo_s,
            modes=frozenset(modes),
            max_inflight=max_inflight,
        ),
        priority=priority,
    )


def shape_call_kwargs(shape: str) -> tuple[Any, dict[str, Any]]:
    """Return ``(frame, kwargs)`` that make ``_evidence_shape`` resolve to ``shape``."""
    if shape == "single_image":
        return "frame-0", {}
    if shape == "ordered_images":
        return "frame-0", {"frames": ["frame-0", "frame-1"]}
    if shape == "audio":
        return None, {"audio": AUDIO_FRAME}
    if shape == "fused":
        return "frame-0", {"audio": AUDIO_FRAME}
    if shape in ("context_stream", "video_clip"):
        return "frame-0", {"context": {EVIDENCE_SHAPE_KEY: shape}}
    raise ValueError(f"unsupported shape {shape!r}")


def linear_slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against their index (units per sample)."""
    n = len(values)
    if n < 3:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = statistics.fmean(values)
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def quartile_means(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    size = max(1, len(values) // 4)
    return [
        statistics.fmean(values[i * size : (i + 1) * size] or [0.0])
        for i in range(4)
    ]


# --------------------------------------------------------------------------------------
# overload driver
# --------------------------------------------------------------------------------------


@dataclass
class LoadReport:
    dispatched: int = 0
    completed: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    peak_outstanding: int = 0
    outstanding_samples: list[int] = field(default_factory=list)
    peak_pool_tasks: int = 0
    pool_task_samples: list[int] = field(default_factory=list)
    peak_snapshot_inflight: int = 0
    wall_s: float = 0.0
    exceptions: dict[str, int] = field(default_factory=dict)


async def drive_open_loop(
    pool: JudgePool,
    *,
    frame: Any,
    call_kwargs: dict[str, Any],
    total_units: int,
    batch: int,
    tick_s: float,
    task_sample_every: int = 5,
) -> LoadReport:
    """Open-loop arrivals: fire ``batch`` decisions every ``tick_s`` regardless of drain.

    A closed loop (N workers looping) can never demonstrate unbounded queueing —
    its outstanding count is pinned to N by construction.  Only an open loop can,
    so this is the regime the boundedness claim is actually tested in.
    """
    report = LoadReport()
    outstanding = 0
    pending: set[asyncio.Task] = set()

    async def one_call() -> None:
        nonlocal outstanding
        try:
            verdict = await pool.judge(condition="sv3 overload", frame=frame, **call_kwargs)
            kind = classify_verdict(verdict)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
            kind = "exception"
            report.exceptions[type(exc).__name__] = report.exceptions.get(type(exc).__name__, 0) + 1
        finally:
            outstanding -= 1
            report.completed += 1
        report.outcomes[kind] = report.outcomes.get(kind, 0) + 1

    started = time.monotonic()
    tick = 0
    while report.dispatched < total_units:
        for _ in range(min(batch, total_units - report.dispatched)):
            outstanding += 1
            report.dispatched += 1
            task = asyncio.ensure_future(one_call())
            pending.add(task)
            task.add_done_callback(pending.discard)
        report.peak_outstanding = max(report.peak_outstanding, outstanding)
        report.outstanding_samples.append(outstanding)
        if tick % task_sample_every == 0:
            live = pool_task_count()
            report.peak_pool_tasks = max(report.peak_pool_tasks, live)
            report.pool_task_samples.append(live)
            snap = pool.snapshot()["routes"]
            report.peak_snapshot_inflight = max(
                report.peak_snapshot_inflight,
                sum(r["inflight"] for r in snap.values()),
            )
        tick += 1
        await asyncio.sleep(tick_s)

    while pending:
        report.peak_outstanding = max(report.peak_outstanding, outstanding)
        await asyncio.wait(set(pending), timeout=5.0)
    report.wall_s = time.monotonic() - started
    return report


async def drive_bursts(
    pool: JudgePool,
    *,
    frame: Any,
    call_kwargs: dict[str, Any],
    bursts: int,
    burst_size: int,
    gap_s: float,
) -> tuple[LoadReport, list[int]]:
    """Repeated instantaneous bursts — the shape a real fast-packet stream has.

    Returns the aggregate report plus the residual outstanding count measured at
    the end of each inter-burst gap.  A residual that climbs burst over burst is
    the signature of unbounded accumulation.
    """
    report = LoadReport()
    outstanding = 0
    residuals: list[int] = []
    pending: set[asyncio.Task] = set()

    async def one_call() -> None:
        nonlocal outstanding
        try:
            verdict = await pool.judge(condition="sv3 burst", frame=frame, **call_kwargs)
            kind = classify_verdict(verdict)
        except Exception as exc:  # noqa: BLE001
            kind = "exception"
            report.exceptions[type(exc).__name__] = report.exceptions.get(type(exc).__name__, 0) + 1
        finally:
            outstanding -= 1
            report.completed += 1
        report.outcomes[kind] = report.outcomes.get(kind, 0) + 1

    started = time.monotonic()
    for _ in range(bursts):
        for _ in range(burst_size):
            outstanding += 1
            report.dispatched += 1
            task = asyncio.ensure_future(one_call())
            pending.add(task)
            task.add_done_callback(pending.discard)
        report.peak_outstanding = max(report.peak_outstanding, outstanding)
        report.peak_pool_tasks = max(report.peak_pool_tasks, pool_task_count())
        deadline = time.monotonic() + gap_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.005)
            report.outstanding_samples.append(outstanding)
            snap = pool.snapshot()["routes"]
            report.peak_snapshot_inflight = max(
                report.peak_snapshot_inflight,
                sum(r["inflight"] for r in snap.values()),
            )
        residuals.append(outstanding)

    while pending:
        await asyncio.wait(set(pending), timeout=10.0)
    report.wall_s = time.monotonic() - started
    return report, residuals


async def settle(seconds: float = 0.15) -> None:
    """Let cancellations and release callbacks run before a leak assertion."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------------------
# SV-3.3 — overload allocation
# --------------------------------------------------------------------------------------

SUSTAINED_UNITS = 4000
SUSTAINED_BATCH = 4
SUSTAINED_TICK_S = 0.002
# The gap is deliberately SHORTER than one served provider call (0.02s), so bursts overlap and
# any per-burst residual would compound visibly across the run.
BURST_COUNT = 10
BURST_SIZE = 800
BURST_GAP_S = 0.015


def _overload_pool(tally: ReceiptTally, *, latency_s: float = 0.02) -> tuple[JudgePool, dict[str, TrackedJudge]]:
    """Two routes with different capacities, zero hedge delay, tiny receipt ring."""
    a = TrackedJudge(name="route_a", latency_s=latency_s)
    b = TrackedJudge(name="route_b", latency_s=latency_s)
    pool = JudgePool(
        [
            route("route_a", a, shapes={"single_image"}, slo_s=0.10, max_inflight=4, priority=0),
            route("route_b", b, shapes={"single_image"}, slo_s=0.10, max_inflight=2, priority=1),
        ],
        mode="reflex",
        deadline_s=0.25,
        hedge_after_s=0.0,
        max_parallel=2,
        min_confidence=0.7,
        receipt_limit=256,
        on_receipt=tally,
    )
    return pool, {"route_a": a, "route_b": b}


async def sv3_3() -> TestResult:
    result = TestResult(
        test_id="SV-3.3",
        title="Overload allocation — bounded in-flight, capacity shedding, no unbounded queue, no leak",
    )

    # ---------------- sustained (open-loop, constant arrival rate) ----------------
    tally = ReceiptTally()
    pool, judges = _overload_pool(tally)
    frame, call_kwargs = shape_call_kwargs("single_image")
    sustained = await drive_open_loop(
        pool,
        frame=frame,
        call_kwargs=call_kwargs,
        total_units=SUSTAINED_UNITS,
        batch=SUSTAINED_BATCH,
        tick_s=SUSTAINED_TICK_S,
    )
    await settle(0.30)

    caps = {"route_a": 4, "route_b": 2}
    peak_live = {name: j.peak_live for name, j in judges.items()}
    total_cap = sum(caps.values())
    sustained_leak_live = sum(j.live for j in judges.values())
    sustained_snapshot = pool.snapshot()["routes"]
    sustained_leak_snapshot = sum(r["inflight"] for r in sustained_snapshot.values())
    sustained_leak_tasks = pool_task_count()
    sustained_quartiles = quartile_means([float(v) for v in sustained.outstanding_samples])
    sustained_slope = linear_slope([float(v) for v in sustained.outstanding_samples])
    sustained_offered_rate = sustained.dispatched / max(sustained.wall_s, 1e-9)
    sustained_shed_frac = (
        sustained.outcomes.get("shed", 0) / max(sustained.completed, 1)
    )

    result.metrics["sustained"] = {
        "units_dispatched": sustained.dispatched,
        "units_completed": sustained.completed,
        "wall_s": round(sustained.wall_s, 3),
        "offered_decisions_per_s": round(sustained_offered_rate, 1),
        "served_capacity_decisions_per_s": round(caps["route_a"] / 0.02, 1),
        "outcomes": dict(sorted(sustained.outcomes.items())),
        "shed_fraction": round(sustained_shed_frac, 4),
        "peak_concurrent_provider_calls_per_route": peak_live,
        "declared_max_inflight_per_route": caps,
        "peak_concurrent_provider_calls_total": sum(peak_live.values()),
        "declared_max_inflight_total": total_cap,
        "peak_pool_snapshot_inflight": sustained.peak_snapshot_inflight,
        "peak_queue_depth_outstanding_decisions": sustained.peak_outstanding,
        "peak_live_call_route_tasks": sustained.peak_pool_tasks,
        "outstanding_quartile_means": [round(v, 2) for v in sustained_quartiles],
        "outstanding_slope_per_tick": round(sustained_slope, 6),
        "receipts_total_observed": tally.total,
        "receipts_by_outcome": dict(sorted(tally.by_outcome.items())),
        "capacity_receipts": tally.count("capacity"),
        "retained_receipt_ring_len": len(pool.receipts()),
        "receipt_limit": 256,
        "leak_live_provider_calls": sustained_leak_live,
        "leak_pool_snapshot_inflight": sustained_leak_snapshot,
        "leak_call_route_tasks": sustained_leak_tasks,
        "exceptions": sustained.exceptions,
    }

    for name, cap in caps.items():
        result.check(
            f"sustained: {name} never exceeded max_inflight={cap}",
            peak_live[name] <= cap,
            f"peak concurrent provider calls = {peak_live[name]}",
        )
    result.check(
        "sustained: pool-wide concurrency never exceeded sum(max_inflight)",
        sum(peak_live.values()) <= total_cap,
        f"peak total = {sum(peak_live.values())} vs cap {total_cap}",
    )
    result.check(
        "sustained: offered load genuinely exceeded served capacity",
        sustained_offered_rate > 3 * (caps["route_a"] / 0.02),
        f"offered {sustained_offered_rate:.0f}/s vs served capacity ~{caps['route_a'] / 0.02:.0f}/s",
    )
    result.check(
        "sustained: excess shed with a `capacity` receipt",
        tally.count("capacity") > 0 and sustained.outcomes.get("shed", 0) > 0,
        f"capacity receipts={tally.count('capacity')}, fully-shed decisions={sustained.outcomes.get('shed', 0)}",
    )
    result.check(
        "sustained: queue depth did not grow without bound",
        sustained_quartiles[3]
        <= max(sustained_quartiles[1] * QUEUE_GROWTH_FACTOR, sustained_quartiles[1] + QUEUE_GROWTH_ABSOLUTE),
        f"quartile means {[round(v, 2) for v in sustained_quartiles]}, slope {sustained_slope:.5f}/tick",
    )
    result.check(
        "sustained: peak queue depth stayed far below units offered",
        sustained.peak_outstanding < SUSTAINED_UNITS // 8,
        f"peak outstanding={sustained.peak_outstanding} of {SUSTAINED_UNITS} offered",
    )
    result.check(
        "sustained: zero in-flight leak at shutdown",
        sustained_leak_live == 0 and sustained_leak_snapshot == 0 and sustained_leak_tasks == 0,
        f"provider live={sustained_leak_live}, snapshot inflight={sustained_leak_snapshot}, "
        f"live _call_route tasks={sustained_leak_tasks}",
    )
    result.check(
        "sustained: every decision terminated (no hang, no dropped unit)",
        sustained.completed == sustained.dispatched,
        f"{sustained.completed}/{sustained.dispatched}",
    )
    result.check(
        "sustained: retained receipt ring stayed bounded by receipt_limit",
        len(pool.receipts()) <= 256 and tally.total > 256,
        f"ring={len(pool.receipts())} while {tally.total} receipts were emitted",
    )
    result.check(
        "sustained: no verdict was fabricated for a shed decision",
        all(
            kind in ("served", "shed", "deadline", "no_route")
            for kind in sustained.outcomes
        )
        and not sustained.exceptions,
        f"outcome kinds={sorted(sustained.outcomes)}, exceptions={sustained.exceptions}",
    )

    # ---------------- burst (instantaneous arrivals, repeated) ----------------
    burst_tally = ReceiptTally()
    burst_pool, burst_judges = _overload_pool(burst_tally)
    burst, residuals = await drive_bursts(
        burst_pool,
        frame=frame,
        call_kwargs=call_kwargs,
        bursts=BURST_COUNT,
        burst_size=BURST_SIZE,
        gap_s=BURST_GAP_S,
    )
    await settle(0.30)

    burst_peak_live = {name: j.peak_live for name, j in burst_judges.items()}
    burst_leak_live = sum(j.live for j in burst_judges.values())
    burst_snapshot = burst_pool.snapshot()["routes"]
    burst_leak_snapshot = sum(r["inflight"] for r in burst_snapshot.values())
    burst_leak_tasks = pool_task_count()

    result.metrics["burst"] = {
        "bursts": BURST_COUNT,
        "burst_size": BURST_SIZE,
        "gap_s": BURST_GAP_S,
        "units_dispatched": burst.dispatched,
        "units_completed": burst.completed,
        "wall_s": round(burst.wall_s, 3),
        "outcomes": dict(sorted(burst.outcomes.items())),
        "shed_fraction": round(burst.outcomes.get("shed", 0) / max(burst.completed, 1), 4),
        "peak_concurrent_provider_calls_per_route": burst_peak_live,
        "declared_max_inflight_per_route": caps,
        "peak_pool_snapshot_inflight": burst.peak_snapshot_inflight,
        "peak_queue_depth_outstanding_decisions": burst.peak_outstanding,
        "peak_live_call_route_tasks": burst.peak_pool_tasks,
        "residual_outstanding_after_each_gap": residuals,
        "residual_slope_per_burst": round(linear_slope([float(v) for v in residuals]), 4),
        "offered_decisions_per_s": round(burst.dispatched / max(burst.wall_s, 1e-9), 1),
        "completed_decisions_per_s": round(burst.completed / max(burst.wall_s, 1e-9), 1),
        "unbounded_queue_would_have_reached": BURST_COUNT * BURST_SIZE,
        "receipts_by_outcome": dict(sorted(burst_tally.by_outcome.items())),
        "capacity_receipts": burst_tally.count("capacity"),
        "leak_live_provider_calls": burst_leak_live,
        "leak_pool_snapshot_inflight": burst_leak_snapshot,
        "leak_call_route_tasks": burst_leak_tasks,
        "exceptions": burst.exceptions,
    }

    for name, cap in caps.items():
        result.check(
            f"burst: {name} never exceeded max_inflight={cap}",
            burst_peak_live[name] <= cap,
            f"peak concurrent provider calls = {burst_peak_live[name]}",
        )
    result.check(
        "burst: excess shed with a `capacity` receipt",
        burst_tally.count("capacity") > 0 and burst.outcomes.get("shed", 0) > 0,
        f"capacity receipts={burst_tally.count('capacity')}, fully-shed={burst.outcomes.get('shed', 0)}",
    )
    # The driver offers one burst per gap, so the STEADY-STATE residual is expected to sit at
    # roughly one burst.  What would prove unbounded queueing is residual GROWTH across bursts
    # (it would climb toward BURST_COUNT * BURST_SIZE), not its equilibrium level.
    residual_slope = linear_slope([float(v) for v in residuals])
    result.check(
        "burst: residual queue depth did not accumulate burst over burst",
        residuals[-1]
        <= max(float(residuals[0]), 4.0) * QUEUE_GROWTH_FACTOR + QUEUE_GROWTH_ABSOLUTE
        and abs(residual_slope) < BURST_SIZE * 0.05
        and max(residuals) <= 2 * BURST_SIZE,
        f"residual outstanding after each gap = {residuals}, "
        f"slope={residual_slope:.3f}/burst (flat), equilibrium ~1 burst of {BURST_SIZE}; "
        f"unbounded queueing would have reached ~{BURST_COUNT * BURST_SIZE}",
    )
    result.check(
        "burst: peak queue depth stayed within one burst — bursts did not compound",
        burst.peak_outstanding <= BURST_SIZE + max(4, residuals[0]) * 2,
        f"peak outstanding={burst.peak_outstanding} vs one burst={BURST_SIZE} "
        f"of {BURST_COUNT * BURST_SIZE} total offered",
    )
    result.check(
        "burst: every decision terminated",
        burst.completed == burst.dispatched,
        f"{burst.completed}/{burst.dispatched}",
    )
    result.check(
        "burst: zero in-flight leak at shutdown",
        burst_leak_live == 0 and burst_leak_snapshot == 0 and burst_leak_tasks == 0,
        f"provider live={burst_leak_live}, snapshot inflight={burst_leak_snapshot}, "
        f"live _call_route tasks={burst_leak_tasks}",
    )

    result.note(
        "The pool holds NO queue of its own: `judge()` is a per-decision call that "
        "immediately fans out one asyncio task per eligible route, and a task that "
        "cannot reserve a route slot returns `capacity` at once (judge_pool.py:430-431). "
        "'Queue depth' here is therefore the caller-side outstanding-decision count, "
        "which is the only place work can pile up."
    )
    result.note(
        "`max_parallel` is a PER-DECISION fan-out semaphore, constructed fresh inside "
        "every `judge()` call (judge_pool.py:568). It does not bound pool-wide "
        "concurrency. The only pool-wide bound is sum(JudgeCapability.max_inflight)."
    )
    return result


# --------------------------------------------------------------------------------------
# SV-3.4 — deadline authority
# --------------------------------------------------------------------------------------


async def sv3_4() -> TestResult:
    result = TestResult(
        test_id="SV-3.4",
        title="Deadline authority — outer deadline wins; a late reply cannot resurrect or mutate a decision",
    )
    deadline_s = 0.30

    # ---- A: a slow route is cut at the deadline, not at its declared SLO ----
    tally_a = ReceiptTally()
    slow = TrackedJudge(name="slow", latency_s=deadline_s * 5)
    pool_a = JudgePool(
        [route("slow", slow, shapes={"single_image"}, slo_s=0.05, max_inflight=1)],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_a,
    )
    started = time.monotonic()
    verdict_a = await pool_a.judge(condition="sv3.4 slow", frame="frame-0")
    elapsed_a = time.monotonic() - started
    await settle(0.10)

    result.metrics["A_slow_route"] = {
        "deadline_s": deadline_s,
        "declared_latency_slo_s": 0.05,
        "actual_provider_latency_s": deadline_s * 5,
        "elapsed_s": round(elapsed_a, 4),
        "verdict_class": classify_verdict(verdict_a),
        "verdict_satisfied": getattr(verdict_a, "satisfied", None),
        "verdict_confidence": getattr(verdict_a, "confidence", None),
        "receipts_by_outcome": dict(sorted(tally_a.by_outcome.items())),
        "provider_calls": slow.calls,
        "provider_cancellations": slow.cancellations,
    }
    result.check(
        "A: judge() returned within the deadline",
        elapsed_a <= deadline_s + DEADLINE_SLACK_S,
        f"elapsed={elapsed_a:.4f}s vs deadline={deadline_s}s (+{DEADLINE_SLACK_S}s slack)",
    )
    result.check(
        "A: returned an UNKNOWN-shaped fail-closed verdict",
        unknown_shaped(verdict_a) and classify_verdict(verdict_a) == "deadline",
        f"satisfied={getattr(verdict_a, 'satisfied', None)}, "
        f"confidence={getattr(verdict_a, 'confidence', None)}",
    )
    result.check(
        "A: emitted a `deadline` receipt and cancelled the in-flight route",
        tally_a.count("deadline") == 1 and tally_a.count("cancelled") == 1,
        f"receipts={dict(sorted(tally_a.by_outcome.items()))}",
    )
    result.check(
        "A: the declared latency_slo_s did NOT truncate the call early",
        slow.calls == 1 and elapsed_a > 0.05 * 2,
        f"declared SLO 0.05s but the pool ran the route for {elapsed_a:.4f}s "
        "(SLO is a routing ceiling, not a timeout)",
    )
    result.check(
        "A: the pool actually cancelled the provider call",
        slow.cancellations == 1 and slow.live == 0,
        f"cancellations={slow.cancellations}, live={slow.live}",
    )

    # ---- B: a cancellation-hostile late reply must not resurrect the decision ----
    tally_b = ReceiptTally()
    hostile = HostileJudge(name="hostile", delay_s=deadline_s * 3)
    pool_b = JudgePool(
        [route("hostile", hostile, shapes={"single_image"}, slo_s=0.10, max_inflight=1)],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_b,
    )
    started = time.monotonic()
    verdict_b = await pool_b.judge(condition="sv3.4 hostile", frame="frame-0")
    elapsed_b = time.monotonic() - started
    verdict_b_snapshot = (
        getattr(verdict_b, "satisfied", None),
        getattr(verdict_b, "confidence", None),
        getattr(verdict_b, "evidence", None),
    )
    receipts_at_return = pool_b.receipts()
    outcomes_at_return = dict(sorted(tally_b.by_outcome.items()))
    inflight_at_return = pool_b.snapshot()["routes"]["hostile"]["inflight"]

    # While the hostile adapter still occupies its only slot, a second decision must shed.
    shed_started = time.monotonic()
    verdict_b2 = await pool_b.judge(condition="sv3.4 hostile while occupied", frame="frame-0")
    shed_elapsed = time.monotonic() - shed_started
    occupied_shed_class = classify_verdict(verdict_b2)
    capacity_receipts_while_occupied = tally_b.count("capacity")

    # Control snapshot: everything this test itself caused is already recorded here, so any
    # later delta is attributable ONLY to the late provider reply.
    receipts_before_late = pool_b.receipts()
    outcomes_before_late = dict(sorted(tally_b.by_outcome.items()))

    # Now let the late reply land.
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(hostile.finished.wait(), timeout=5.0)
    await settle(0.25)

    receipts_after_late = pool_b.receipts()
    outcomes_after_late = dict(sorted(tally_b.by_outcome.items()))
    inflight_after_late = pool_b.snapshot()["routes"]["hostile"]["inflight"]
    verdict_b_after = (
        getattr(verdict_b, "satisfied", None),
        getattr(verdict_b, "confidence", None),
        getattr(verdict_b, "evidence", None),
    )

    result.metrics["B_hostile_late_reply"] = {
        "elapsed_s": round(elapsed_b, 4),
        "provider_delay_s": deadline_s * 3,
        "cancellation_swallowed_by_adapter": hostile.cancel_seen,
        "adapter_returned_late": hostile.late_returns,
        "late_verdict_would_have_been": (
            {
                "satisfied": hostile.late_verdict.satisfied,
                "confidence": hostile.late_verdict.confidence,
            }
            if hostile.late_verdict is not None
            else None
        ),
        "verdict_class": classify_verdict(verdict_b),
        "receipt_outcomes_at_return": outcomes_at_return,
        "receipt_outcomes_after_late_reply": outcomes_after_late,
        "route_inflight_at_return": inflight_at_return,
        "route_inflight_after_late_reply": inflight_after_late,
        "second_decision_while_occupied": occupied_shed_class,
        "second_decision_elapsed_s": round(shed_elapsed, 4),
        "capacity_receipts_while_occupied": capacity_receipts_while_occupied,
        "receipt_outcomes_before_late_reply": outcomes_before_late,
    }
    result.check(
        "B: hostile adapter really did swallow cancellation and reply late",
        hostile.cancel_seen >= 1 and hostile.late_returns >= 1,
        f"cancel_seen={hostile.cancel_seen}, late_returns={hostile.late_returns}",
    )
    result.check(
        "B: judge() still returned at the deadline",
        elapsed_b <= deadline_s + DEADLINE_SLACK_S,
        f"elapsed={elapsed_b:.4f}s",
    )
    result.check(
        "B: the late high-confidence reply did NOT resurrect the decision",
        unknown_shaped(verdict_b) and classify_verdict(verdict_b) == "deadline",
        f"returned {classify_verdict(verdict_b)}; late reply was satisfied=True confidence=0.99",
    )
    result.check(
        "B: the returned verdict object was not mutated by the late reply",
        verdict_b_snapshot == verdict_b_after,
        f"before={verdict_b_snapshot!r} after={verdict_b_after!r}",
    )
    result.check(
        "B: the late reply injected NO new receipt",
        receipts_before_late == receipts_after_late
        and outcomes_before_late == outcomes_after_late,
        f"immediately before the late reply={outcomes_before_late}, "
        f"after the late reply={outcomes_after_late} "
        f"(at judge() return it was {outcomes_at_return}; the `capacity` entry is the "
        "occupancy probe this test issued, not the late reply)",
    )
    result.check(
        "B: no `decisive` receipt was ever recorded for the hostile route",
        tally_b.count("decisive") == 0 and "decisive" not in outcomes_at_return,
        f"at return={outcomes_at_return}, final={outcomes_after_late}, "
        f"terminal receipts at return={[r.outcome for r in receipts_at_return]}",
    )
    result.check(
        "B: the hostile route kept occupying its bounded slot (documented, not a leak)",
        inflight_at_return == 1,
        f"inflight immediately after judge() returned = {inflight_at_return}",
    )
    result.check(
        "B: a decision issued while the slot was occupied shed fast instead of queueing",
        occupied_shed_class == "shed"
        and shed_elapsed < deadline_s * ADMISSION_FAST_FRACTION
        and capacity_receipts_while_occupied >= 1,
        f"class={occupied_shed_class}, elapsed={shed_elapsed:.4f}s, "
        f"capacity receipts={capacity_receipts_while_occupied}",
    )
    result.check(
        "B: the slot was released once the hostile adapter actually terminated",
        inflight_after_late == 0,
        f"inflight after late reply = {inflight_after_late}",
    )

    # ---- C: can a provider that retains its Verdict rewrite an already-returned decision? ----
    tally_c = ReceiptTally()
    mutator = MutatingJudge(name="mutating", latency_s=0.02, confidence=0.40)
    never = NeverJudge(name="never", delay_s=30.0)
    pool_c = JudgePool(
        [
            route("mutating", mutator, shapes={"single_image"}, slo_s=0.10, max_inflight=1, priority=0),
            route("never", never, shapes={"single_image"}, slo_s=0.10, max_inflight=1, priority=1),
        ],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        min_confidence=0.7,
        on_receipt=tally_c,
    )
    started = time.monotonic()
    verdict_c = await pool_c.judge(condition="sv3.4 mutation trap", frame="frame-0")
    elapsed_c = time.monotonic() - started
    before_c = (
        getattr(verdict_c, "satisfied", None),
        getattr(verdict_c, "confidence", None),
    )
    returns_provider_object = any(verdict_c is issued for issued in mutator.issued)
    mutator.mutate_all()
    after_c = (
        getattr(verdict_c, "satisfied", None),
        getattr(verdict_c, "confidence", None),
    )
    await settle(0.10)

    # Control: the same aliasing question on the ordinary decisive path (no deadline involved).
    decisive_mutator = MutatingJudge(name="decisive_mutator", latency_s=0.0, confidence=0.95)
    pool_c2 = JudgePool(
        [route("decisive_mutator", decisive_mutator, shapes={"single_image"}, slo_s=0.10)],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        min_confidence=0.7,
    )
    verdict_c2 = await pool_c2.judge(condition="sv3.4 decisive aliasing", frame="frame-0")
    before_c2 = (verdict_c2.satisfied, verdict_c2.confidence)
    decisive_mutator.mutate_all()
    after_c2 = (verdict_c2.satisfied, verdict_c2.confidence)

    # Control: does the SDK's own scripted adapter alias too?  Scopes the finding as
    # "inherited SDK-wide property" vs "introduced by JudgePool".
    scripted = Verdict(satisfied=False, confidence=0.4, evidence="scripted")
    fake = FakeVision(script={"frame-0": scripted})
    fake_returned = await fake.judge("control", "frame-0")
    fake_aliases = fake_returned is scripted

    result.metrics["C_returned_object_aliasing"] = {
        "scenario": "route A replies low-confidence before expiry; route B never replies; the "
        "pool concludes the decision by DEADLINE EXPIRY and hands back route A's object",
        "elapsed_s": round(elapsed_c, 4),
        "verdict_class": classify_verdict(verdict_c),
        "receipts_by_outcome": dict(sorted(tally_c.by_outcome.items())),
        "decision_concluded_by_deadline": tally_c.count("deadline") == 1,
        "pool_returned_the_provider_owned_object": returns_provider_object,
        "verdict_before_provider_mutation": before_c,
        "verdict_after_provider_mutation": after_c,
        "verdict_dataclass_is_frozen": Verdict.__dataclass_params__.frozen,
        "control_decisive_path_before": before_c2,
        "control_decisive_path_after": after_c2,
        "control_decisive_path_aliased": before_c2 != after_c2,
        "control_sdk_FakeVision_aliases_its_script": fake_aliases,
    }
    result.check(
        "C: the decision WAS concluded by deadline expiry, returning the retained uncertain reply",
        elapsed_c <= deadline_s + DEADLINE_SLACK_S
        and classify_verdict(verdict_c) == "served"
        and tally_c.count("deadline") == 1,
        f"elapsed={elapsed_c:.4f}s, class={classify_verdict(verdict_c)}, "
        f"receipts={dict(sorted(tally_c.by_outcome.items()))} "
        "(judge_pool.py:695 prefers a retained `uncertain` over the synthesised UNKNOWN)",
    )
    aliasing_safe = before_c == after_c
    result.check(
        "C: a provider cannot rewrite a decision the pool already concluded at the deadline",
        aliasing_safe,
        f"pool returned the provider's own object={returns_provider_object}; "
        f"before={before_c} after={after_c}; Verdict frozen={Verdict.__dataclass_params__.frozen}",
    )
    if not aliasing_safe:
        result.note(
            "DEFECT (post-decision aliasing) — judge_pool.py:667 retains and judge_pool.py:695 "
            "returns the provider's own Verdict instance with no defensive copy, and "
            "percept/contracts/__init__.py:472 declares Verdict as a NON-frozen dataclass. "
            "A route whose reply was accepted (or merely retained as `uncertain`) can therefore "
            "rewrite satisfied/confidence after the pool has already concluded the decision and "
            "the caller has acted on it. Measured here: the pool concluded by deadline expiry and "
            f"returned {before_c}; the provider then rewrote the same object to {after_c}. "
            f"The ordinary decisive path is equally aliased (control: {before_c2} -> {after_c2}). "
            "SCOPE: this is an SDK-wide ownership convention, not something JudgePool introduces "
            "— the SDK's own FakeVision.judge (backends/fake.py:61-95) also hands back its "
            f"scripted object by identity (control: aliased={fake_aliases}). It is nevertheless a "
            "real gap in the pool's 'a late provider reply cannot mutate a concluded decision' "
            "guarantee: only the synthesised-UNKNOWN paths (no_route / capacity / deadline with "
            "no retained reply) are actually immune. Fix would be a copy at judge_pool.py:659/667 "
            "or freezing Verdict."
        )

    # ---- D: terminal_outcomes='deadline' raises and latches instead of silently degrading ----
    tally_d = ReceiptTally()
    never_d = NeverJudge(name="never_d", delay_s=30.0)
    pool_d = JudgePool(
        [route("never_d", never_d, shapes={"single_image"}, slo_s=0.10, max_inflight=1)],
        mode="reflex",
        deadline_s=0.15,
        hedge_after_s=0.0,
        on_receipt=tally_d,
        terminal_outcomes=("deadline",),
    )
    raised: str | None = None
    started = time.monotonic()
    try:
        await pool_d.judge(condition="sv3.4 terminal", frame="frame-0")
    except BaseException as exc:  # noqa: BLE001
        raised = type(exc).__name__
    elapsed_d = time.monotonic() - started
    latched = pool_d.take_terminal_error()
    await settle(0.10)

    result.metrics["D_terminal_deadline"] = {
        "raised": raised,
        "elapsed_s": round(elapsed_d, 4),
        "latched_error": type(latched).__name__ if latched is not None else None,
        "receipts_by_outcome": dict(sorted(tally_d.by_outcome.items())),
        "route_inflight_after": pool_d.snapshot()["routes"]["never_d"]["inflight"],
    }
    result.check(
        "D: terminal deadline raised TimeoutError within the deadline and latched the error",
        raised == "TimeoutError"
        and elapsed_d <= 0.15 + DEADLINE_SLACK_S
        and isinstance(latched, TimeoutError),
        f"raised={raised}, elapsed={elapsed_d:.4f}s, latched={type(latched).__name__ if latched else None}",
    )
    result.check(
        "D: the raising path still released the route slot",
        pool_d.snapshot()["routes"]["never_d"]["inflight"] == 0,
        f"inflight={pool_d.snapshot()['routes']['never_d']['inflight']}",
    )

    result.note(
        "The deadline is enforced by the outer `asyncio.wait(timeout=remaining)` loop "
        "(judge_pool.py:591-599) and the `finally` block that cancels every unfinished "
        "route task (judge_pool.py:696-716). Because `judge()` has already returned by "
        "then, a late reply is structurally unable to add a receipt or change the verdict "
        "— the only post-return state it touches is the route's inflight/failure counters."
    )
    return result


# --------------------------------------------------------------------------------------
# SV-3.5 — reflex route rejection at selection
# --------------------------------------------------------------------------------------


async def sv3_5(shape: str = "single_image") -> TestResult:
    result = TestResult(
        test_id="SV-3.5",
        title="Reflex route rejection — over-SLO routes are refused at selection, never attempted",
    )
    frame, call_kwargs = shape_call_kwargs(shape)
    deadline_s = 0.60

    # ---- 1: reflex + latency_slo_s > deadline_s -> rejected at selection ----
    tally_1 = ReceiptTally()
    over = TrackedJudge(name="over_slo", latency_s=0.0)
    pool_1 = JudgePool(
        [route("over_slo", over, shapes={shape}, slo_s=deadline_s * 8, max_inflight=1)],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_1,
    )
    started = time.monotonic()
    v1 = await pool_1.judge(condition="sv3.5 over-slo", frame=frame, **call_kwargs)
    elapsed_1 = time.monotonic() - started
    receipts_1 = pool_1.receipts()

    result.metrics["case_1_over_slo_reflex"] = {
        "evidence_shape": shape,
        "deadline_s": deadline_s,
        "declared_latency_slo_s": deadline_s * 8,
        "elapsed_s": round(elapsed_1, 5),
        "verdict_class": classify_verdict(v1),
        "adapter_invocations": over.calls,
        "receipts_by_outcome": dict(sorted(tally_1.by_outcome.items())),
        "receipt_route_field": [r.route for r in receipts_1],
    }
    result.check(
        "1: rejected AT SELECTION — the adapter was never invoked",
        over.calls == 0,
        f"adapter invocations={over.calls}",
    )
    result.check(
        "1: receipt reason is `no_route` (not `deadline`) with route=None",
        tally_1.count("no_route") == 1
        and tally_1.count("deadline") == 0
        and len(receipts_1) == 1
        and receipts_1[0].route is None,
        f"receipts={dict(sorted(tally_1.by_outcome.items()))}, route field={receipts_1[0].route!r}",
    )
    result.check(
        "1: rejection was immediate, not a deadline burn",
        elapsed_1 < deadline_s * ADMISSION_FAST_FRACTION,
        f"elapsed={elapsed_1:.5f}s vs deadline={deadline_s}s",
    )
    result.check(
        "1: fail-closed UNKNOWN verdict, no fabrication",
        unknown_shaped(v1) and classify_verdict(v1) == "no_route",
        f"class={classify_verdict(v1)}, satisfied={getattr(v1, 'satisfied', None)}",
    )

    # ---- 2: in-SLO route that is actually slow -> attempted then `deadline` ----
    tally_2 = ReceiptTally()
    liar = TrackedJudge(name="in_slo_but_slow", latency_s=deadline_s * 5)
    pool_2 = JudgePool(
        [route("in_slo_but_slow", liar, shapes={shape}, slo_s=deadline_s / 2, max_inflight=1)],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_2,
    )
    started = time.monotonic()
    v2 = await pool_2.judge(condition="sv3.5 in-slo but slow", frame=frame, **call_kwargs)
    elapsed_2 = time.monotonic() - started
    await settle(0.10)

    result.metrics["case_2_in_slo_but_slow"] = {
        "declared_latency_slo_s": deadline_s / 2,
        "actual_provider_latency_s": deadline_s * 5,
        "elapsed_s": round(elapsed_2, 4),
        "verdict_class": classify_verdict(v2),
        "adapter_invocations": liar.calls,
        "receipts_by_outcome": dict(sorted(tally_2.by_outcome.items())),
    }
    result.check(
        "2: an in-SLO route IS attempted",
        liar.calls == 1,
        f"adapter invocations={liar.calls}",
    )
    result.check(
        "2: receipt reason is `deadline` (+`cancelled`), distinct from case 1's `no_route`",
        tally_2.count("deadline") == 1
        and tally_2.count("cancelled") == 1
        and tally_2.count("no_route") == 0,
        f"receipts={dict(sorted(tally_2.by_outcome.items()))}",
    )
    result.check(
        "2: the two rejection reasons are genuinely distinguishable",
        set(tally_1.by_outcome) == {"no_route"}
        and set(tally_2.by_outcome) == {"deadline", "cancelled"},
        f"case1={sorted(tally_1.by_outcome)} vs case2={sorted(tally_2.by_outcome)}",
    )

    # ---- 3: deliberate mode admits the same over-SLO route ----
    tally_3 = ReceiptTally()
    over_3 = TrackedJudge(name="over_slo_deliberate", latency_s=0.01, confidence=0.95)
    pool_3 = JudgePool(
        [
            route(
                "over_slo_deliberate",
                over_3,
                shapes={shape},
                slo_s=deadline_s * 8,
                modes=("deliberate",),
                max_inflight=1,
            )
        ],
        mode="deliberate",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_3,
    )
    v3 = await pool_3.judge(condition="sv3.5 deliberate", frame=frame, **call_kwargs)

    result.metrics["case_3_deliberate_admits_over_slo"] = {
        "mode": "deliberate",
        "declared_latency_slo_s": deadline_s * 8,
        "adapter_invocations": over_3.calls,
        "verdict_class": classify_verdict(v3),
        "receipts_by_outcome": dict(sorted(tally_3.by_outcome.items())),
    }
    result.check(
        "3: deliberate mode admits an over-SLO route (rejection is reflex-specific)",
        over_3.calls == 1 and tally_3.count("decisive") == 1,
        f"invocations={over_3.calls}, receipts={dict(sorted(tally_3.by_outcome.items()))}",
    )

    # ---- 4: mixed pool — the over-SLO route must be skipped, not merely deprioritised ----
    tally_4 = ReceiptTally()
    bad = TrackedJudge(name="bad_over_slo", latency_s=0.0, confidence=0.99)
    good = TrackedJudge(name="good_in_slo", latency_s=0.01, confidence=0.95)
    pool_4 = JudgePool(
        [
            route("bad_over_slo", bad, shapes={shape}, slo_s=deadline_s * 8, max_inflight=1, priority=0),
            route("good_in_slo", good, shapes={shape}, slo_s=deadline_s / 4, max_inflight=1, priority=1),
        ],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_4,
    )
    v4 = await pool_4.judge(condition="sv3.5 mixed", frame=frame, **call_kwargs)

    result.metrics["case_4_mixed_pool"] = {
        "over_slo_adapter_invocations": bad.calls,
        "in_slo_adapter_invocations": good.calls,
        "verdict_class": classify_verdict(v4),
        "receipts_by_route_outcome": tally_4.by_route_outcome,
    }
    result.check(
        "4: the over-SLO route in a mixed pool was never invoked",
        bad.calls == 0 and good.calls == 1,
        f"over_slo={bad.calls}, in_slo={good.calls}",
    )
    result.check(
        "4: no receipt was emitted for the never-selected route",
        "bad_over_slo" not in tally_4.by_route_outcome,
        f"receipt routes={sorted(tally_4.by_route_outcome)}",
    )

    # ---- 5: a deliberate-only route is refused in reflex mode regardless of SLO ----
    tally_5 = ReceiptTally()
    deliberate_only = TrackedJudge(name="deliberate_only", latency_s=0.0)
    pool_5 = JudgePool(
        [
            route(
                "deliberate_only",
                deliberate_only,
                shapes={shape},
                slo_s=0.001,
                modes=("deliberate",),
                max_inflight=1,
            )
        ],
        mode="reflex",
        deadline_s=deadline_s,
        hedge_after_s=0.0,
        on_receipt=tally_5,
    )
    v5 = await pool_5.judge(condition="sv3.5 deliberate-only in reflex", frame=frame, **call_kwargs)

    result.metrics["case_5_mode_gate"] = {
        "declared_latency_slo_s": 0.001,
        "adapter_invocations": deliberate_only.calls,
        "verdict_class": classify_verdict(v5),
        "receipts_by_outcome": dict(sorted(tally_5.by_outcome.items())),
    }
    result.check(
        "5: a deliberate-only route is refused in reflex mode even with a trivial SLO",
        deliberate_only.calls == 0 and tally_5.count("no_route") == 1,
        f"invocations={deliberate_only.calls}, receipts={dict(sorted(tally_5.by_outcome.items()))}",
    )

    result.note(
        "The reflex SLO filter is `_eligible()` at judge_pool.py:367-377; it runs before any "
        "task is created (judge_pool.py:518), which is why the adapter invocation count is "
        "the decisive evidence that rejection happened at selection rather than at timeout."
    )
    return result


# --------------------------------------------------------------------------------------
# SV-3.6 — specialist / audio parity
# --------------------------------------------------------------------------------------

PARITY_SHAPES = ("single_image", "audio", "fused", "context_stream")
PARITY_UNITS = 900
PARITY_BATCH = 3
PARITY_TICK_S = 0.002


async def _parity_for_shape(shape: str) -> dict[str, Any]:
    """Re-run the SV-3.3/3.4/3.5 invariants on one evidence shape."""
    frame, call_kwargs = shape_call_kwargs(shape)
    out: dict[str, Any] = {"evidence_shape": shape}

    # capacity / overload
    tally = ReceiptTally()
    a = TrackedJudge(name=f"{shape}_a", latency_s=0.02)
    pool = JudgePool(
        [route(f"{shape}_a", a, shapes={shape}, slo_s=0.10, max_inflight=3)],
        mode="reflex",
        deadline_s=0.25,
        hedge_after_s=0.0,
        receipt_limit=256,
        on_receipt=tally,
    )
    load = await drive_open_loop(
        pool,
        frame=frame,
        call_kwargs=call_kwargs,
        total_units=PARITY_UNITS,
        batch=PARITY_BATCH,
        tick_s=PARITY_TICK_S,
    )
    await settle(0.25)
    quartiles = quartile_means([float(v) for v in load.outstanding_samples])
    out["overload"] = {
        "declared_max_inflight": 3,
        "peak_concurrent_provider_calls": a.peak_live,
        "peak_queue_depth_outstanding_decisions": load.peak_outstanding,
        "outstanding_quartile_means": [round(v, 2) for v in quartiles],
        "units": load.dispatched,
        "completed": load.completed,
        "outcomes": dict(sorted(load.outcomes.items())),
        "capacity_receipts": tally.count("capacity"),
        "leak_live_provider_calls": a.live,
        "leak_snapshot_inflight": pool.snapshot()["routes"][f"{shape}_a"]["inflight"],
        "leak_call_route_tasks": pool_task_count(),
        "bounded": a.peak_live <= 3,
        "shed": tally.count("capacity") > 0 and load.outcomes.get("shed", 0) > 0,
        "queue_bounded": quartiles[3]
        <= max(quartiles[1] * QUEUE_GROWTH_FACTOR, quartiles[1] + QUEUE_GROWTH_ABSOLUTE),
        "no_leak": a.live == 0
        and pool.snapshot()["routes"][f"{shape}_a"]["inflight"] == 0
        and pool_task_count() == 0,
        "all_terminated": load.completed == load.dispatched,
    }

    # deadline authority
    d_tally = ReceiptTally()
    slow = TrackedJudge(name=f"{shape}_slow", latency_s=1.2)
    d_pool = JudgePool(
        [route(f"{shape}_slow", slow, shapes={shape}, slo_s=0.10, max_inflight=1)],
        mode="reflex",
        deadline_s=0.25,
        hedge_after_s=0.0,
        on_receipt=d_tally,
    )
    started = time.monotonic()
    dv = await d_pool.judge(condition="sv3.6 deadline", frame=frame, **call_kwargs)
    d_elapsed = time.monotonic() - started
    await settle(0.10)
    out["deadline"] = {
        "deadline_s": 0.25,
        "elapsed_s": round(d_elapsed, 4),
        "within_deadline": d_elapsed <= 0.25 + DEADLINE_SLACK_S,
        "verdict_class": classify_verdict(dv),
        "unknown_shaped": unknown_shaped(dv),
        "receipts_by_outcome": dict(sorted(d_tally.by_outcome.items())),
        "receipt_shape": sorted(d_tally.by_shape_outcome),
        "ok": (
            d_elapsed <= 0.25 + DEADLINE_SLACK_S
            and unknown_shaped(dv)
            and classify_verdict(dv) == "deadline"
            and d_tally.count("deadline") == 1
            and list(d_tally.by_shape_outcome) == [shape]
        ),
    }

    # reflex rejection at selection
    r_tally = ReceiptTally()
    over = TrackedJudge(name=f"{shape}_over", latency_s=0.0)
    r_pool = JudgePool(
        [route(f"{shape}_over", over, shapes={shape}, slo_s=4.0, max_inflight=1)],
        mode="reflex",
        deadline_s=0.5,
        hedge_after_s=0.0,
        on_receipt=r_tally,
    )
    started = time.monotonic()
    rv = await r_pool.judge(condition="sv3.6 reflex", frame=frame, **call_kwargs)
    r_elapsed = time.monotonic() - started
    out["reflex_rejection"] = {
        "adapter_invocations": over.calls,
        "elapsed_s": round(r_elapsed, 5),
        "verdict_class": classify_verdict(rv),
        "receipts_by_outcome": dict(sorted(r_tally.by_outcome.items())),
        "ok": over.calls == 0
        and r_tally.count("no_route") == 1
        and r_tally.count("deadline") == 0
        and classify_verdict(rv) == "no_route",
    }

    # evidence-shape exactness: a single_image-only route must never serve this shape
    x_tally = ReceiptTally()
    wrong = TrackedJudge(name=f"{shape}_wrong_shape", latency_s=0.0, confidence=0.99)
    x_pool = JudgePool(
        [route(f"{shape}_wrong_shape", wrong, shapes={"single_image"}, slo_s=0.05, max_inflight=1)],
        mode="reflex",
        deadline_s=0.5,
        hedge_after_s=0.0,
        on_receipt=x_tally,
    )
    xv = await x_pool.judge(condition="sv3.6 shape exactness", frame=frame, **call_kwargs)
    expect_served = shape == "single_image"
    out["shape_exactness"] = {
        "single_image_only_route_invocations": wrong.calls,
        "verdict_class": classify_verdict(xv),
        "expected_served": expect_served,
        "ok": (wrong.calls == 1 and classify_verdict(xv) == "served")
        if expect_served
        else (wrong.calls == 0 and classify_verdict(xv) == "no_route"),
    }
    return out


async def sv3_6() -> TestResult:
    result = TestResult(
        test_id="SV-3.6",
        title="Specialist/audio parity — the same bounds hold on audio, fused and context_stream routes",
    )
    per_shape: dict[str, Any] = {}
    for shape in PARITY_SHAPES:
        per_shape[shape] = await _parity_for_shape(shape)
    result.metrics["per_shape"] = per_shape

    for shape, data in per_shape.items():
        overload = data["overload"]
        result.check(
            f"{shape}: max_inflight never exceeded under overload",
            overload["bounded"],
            f"peak concurrent provider calls={overload['peak_concurrent_provider_calls']} "
            f"vs max_inflight={overload['declared_max_inflight']}",
        )
        result.check(
            f"{shape}: excess sheds with a `capacity` receipt",
            overload["shed"],
            f"capacity receipts={overload['capacity_receipts']}, "
            f"fully-shed={overload['outcomes'].get('shed', 0)}",
        )
        result.check(
            f"{shape}: queue depth bounded under overload",
            overload["queue_bounded"],
            f"peak outstanding={overload['peak_queue_depth_outstanding_decisions']}, "
            f"quartile means={overload['outstanding_quartile_means']}",
        )
        result.check(
            f"{shape}: zero in-flight leak at shutdown",
            overload["no_leak"] and overload["all_terminated"],
            f"live={overload['leak_live_provider_calls']}, "
            f"snapshot={overload['leak_snapshot_inflight']}, "
            f"tasks={overload['leak_call_route_tasks']}, "
            f"completed={overload['completed']}/{overload['units']}",
        )
        result.check(
            f"{shape}: outer deadline authoritative + correctly shaped receipt",
            data["deadline"]["ok"],
            f"elapsed={data['deadline']['elapsed_s']}s, "
            f"receipts={data['deadline']['receipts_by_outcome']}",
        )
        result.check(
            f"{shape}: reflex over-SLO route rejected at selection (adapter never called)",
            data["reflex_rejection"]["ok"],
            f"invocations={data['reflex_rejection']['adapter_invocations']}, "
            f"receipts={data['reflex_rejection']['receipts_by_outcome']}",
        )
        result.check(
            f"{shape}: evidence shape is exact — no silent fallback to an image route",
            data["shape_exactness"]["ok"],
            f"single_image-only route invocations="
            f"{data['shape_exactness']['single_image_only_route_invocations']}, "
            f"class={data['shape_exactness']['verdict_class']}",
        )

    # Cross-shape parity: the invariant verdict must be identical for every shape.
    invariant_matrix = {
        shape: [
            data["overload"]["bounded"],
            data["overload"]["shed"],
            data["overload"]["queue_bounded"],
            data["overload"]["no_leak"],
            data["deadline"]["ok"],
            data["reflex_rejection"]["ok"],
        ]
        for shape, data in per_shape.items()
    }
    result.metrics["invariant_matrix"] = {
        shape: dict(
            zip(
                ("bounded", "sheds", "queue_bounded", "no_leak", "deadline", "reflex_reject"),
                values,
            )
        )
        for shape, values in invariant_matrix.items()
    }
    baseline = invariant_matrix["single_image"]
    result.check(
        "parity: audio / fused / context_stream match the vision baseline on every invariant",
        all(values == baseline for values in invariant_matrix.values()),
        f"matrix={result.metrics['invariant_matrix']}",
    )

    # Architecture probe: does the compiled-audio judge binding go through JudgePool at all?
    binding_pool = JudgePool(
        [route("audio_only", TrackedJudge(name="audio_only"), shapes={"audio"}, slo_s=0.1)],
        mode="reflex",
        deadline_s=0.5,
    )
    pool_binding_error: str | None = None
    try:
        prepare_audio_judge_binding(binding_pool, mode="audio_only", application_key="sv3")
    except Exception as exc:  # noqa: BLE001
        pool_binding_error = f"{type(exc).__name__}: {exc}"
    fake_binding_ok = False
    fake_binding_error: str | None = None
    try:
        binding = prepare_audio_judge_binding(FakeVision(), mode="audio_only", application_key="sv3")
        fake_binding_ok = callable(binding.dispatch)
    except Exception as exc:  # noqa: BLE001
        fake_binding_error = f"{type(exc).__name__}: {exc}"

    result.metrics["audio_judge_binding"] = {
        "prepare_audio_judge_binding(JudgePool)": pool_binding_error or "accepted",
        "prepare_audio_judge_binding(FakeVision)": "accepted" if fake_binding_ok else fake_binding_error,
        "trusted_adapters": ["percept.backends.fake.FakeVision", "percept.backends.gemini.GeminiVision"],
    }
    result.check(
        "audio binding: the compiled-audio path is a closed allowlist (probe succeeded)",
        pool_binding_error is not None and fake_binding_ok,
        f"JudgePool -> {pool_binding_error}; FakeVision -> {'accepted' if fake_binding_ok else fake_binding_error}",
    )
    result.note(
        "ARCHITECTURE: `prepare_audio_judge_binding` (audio_input.py:1028-1062) accepts ONLY an "
        "exact SDK FakeVision or GeminiVision instance, so a JudgePool cannot be installed as a "
        "compiled AudioJudgeBinding. The compiled-audio dispatch therefore does NOT inherit the "
        "pool's capacity/deadline bounds — it is a frozen single-adapter dispatch. Pool parity on "
        "the `audio`/`fused` evidence shapes (proved above) applies when a JudgePool is used as "
        "the Vision adapter, not when the compiled-audio binding path is used."
    )
    result.note(
        "`FakeSpecialist` (backends/fake.py:124) is a token streamer with no `judge()`, so it is "
        "not routable through JudgePool by construction (JudgeRoute.__post_init__, "
        "judge_pool.py:119-122). The specialist path exercised here is the `context_stream` "
        "evidence shape — the context-holding route the module docstring calls out at line 10-11."
    )
    return result


# --------------------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------------------

SUITE: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("SV-3.3", sv3_3),
    ("SV-3.4", sv3_4),
    ("SV-3.5", sv3_5),
    ("SV-3.6", sv3_6),
)


async def run_all(only: list[str] | None = None) -> dict[str, Any]:
    results: list[TestResult] = []
    for test_id, fn in SUITE:
        if only and test_id not in only:
            continue
        try:
            results.append(await fn())
        except Exception as exc:  # noqa: BLE001
            failed = TestResult(test_id=test_id, title=f"{test_id} crashed")
            failed.error = f"{type(exc).__name__}: {exc}"
            failed.check("suite executed", False, failed.error)
            results.append(failed)
    return {
        "suite": "SV-3.x judge pool boundedness",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime": RUNTIME,
        "network_calls": 0,
        "provider_credentials_used": False,
        "tolerances": {
            "deadline_slack_s": DEADLINE_SLACK_S,
            "admission_fast_fraction": ADMISSION_FAST_FRACTION,
            "queue_growth_factor": QUEUE_GROWTH_FACTOR,
            "queue_growth_absolute": QUEUE_GROWTH_ABSOLUTE,
        },
        "results": [r.to_dict() for r in results],
        "passed": all(r.passed for r in results),
    }


def _print_table(payload: dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print("SV-3.x  JUDGE POOL BOUNDEDNESS / OVERLOAD VALIDATION")
    print(f"runtime: percept {payload['runtime']['percept_version']} @ {payload['runtime']['judge_pool_module']}")
    print("=" * 100)
    for result in payload["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['test_id']} — {result['title']}")
        if result["error"]:
            print(f"       ERROR: {result['error']}")
        for check in result["checks"]:
            mark = "  ok  " if check["passed"] else " FAIL "
            print(f"   [{mark}] {check['name']}")
            if check["detail"]:
                print(f"            {check['detail']}")
        for note in result["notes"]:
            print(f"   note: {note}")
    print()
    print("-" * 100)
    for result in payload["results"]:
        total = len(result["checks"])
        ok = sum(1 for c in result["checks"] if c["passed"])
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {result['test_id']:<8} {status:<5} {ok}/{total} checks")
    print(f"  {'OVERALL':<8} {'PASS' if payload['passed'] else 'FAIL'}")
    print("-" * 100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SV-3.x judge pool boundedness validation")
    parser.add_argument("--only", nargs="*", help="restrict to specific test ids, e.g. --only SV-3.3")
    parser.add_argument(
        "--json",
        default=str(Path(__file__).resolve().parent / "results" / "sv3_pool.json"),
        help="where to write the machine-readable result payload",
    )
    args = parser.parse_args(argv)

    payload = asyncio.run(run_all(args.only))
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    _print_table(payload)
    print(f"\njson: {out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
