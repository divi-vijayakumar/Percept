"""Deadline-bounded, capability-aware routing across constructed Vision judges.

``JudgePool`` is a Vision adapter, so it composes with the existing SDK instead
of creating a second judgment path.  The cognition core still consumes exactly
one :class:`~percept.contracts.Verdict`; this adapter only decides which
deployment-provided judge may produce it.

The pool has three load-bearing rules:

* evidence shape is exact -- a native-video or context-holding request never
  silently falls back to a sprite or a stateless ordered-image prompt;
* the deadline covers route selection, hedge delay, and provider execution;
* capacity exhaustion and deadline expiry fail closed to an UNKNOWN-shaped
  verdict while retaining bounded route occupancy and an inspectable receipt.

Provider adapters can still own their retry/circuit-breaker mechanics.  The
pool's outer deadline is authoritative for the Watch: retries that cannot
finish within it are cancelled, and a cancellation-hostile adapter continues
to occupy its bounded route slot instead of allowing an unbounded task leak.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ..contracts import Verdict

EvidenceShape = Literal[
    "single_image",
    "ordered_images",
    "video_clip",
    "context_stream",
    "audio",
    "fused",
]
LatencyMode = Literal["reflex", "deliberate"]

_EVIDENCE_SHAPES = frozenset(
    {
        "single_image",
        "ordered_images",
        "video_clip",
        "context_stream",
        "audio",
        "fused",
    }
)
_LATENCY_MODES = frozenset({"reflex", "deliberate"})
_EVIDENCE_SHAPE_CONTEXT_KEY = "_percept_evidence_shape"


@dataclass(frozen=True)
class JudgeCapability:
    """Deployment-reviewed eligibility for one concrete judge adapter.

    ``latency_slo_s`` is a declared/measured routing ceiling, not a timeout.
    Reflex routing considers a route only when this value fits the pool's hard
    deadline.  The hard deadline is enforced independently at runtime.
    """

    evidence_shapes: frozenset[EvidenceShape]
    latency_slo_s: float
    modes: frozenset[LatencyMode] = frozenset({"deliberate"})
    max_inflight: int = 1

    def __post_init__(self) -> None:
        shapes = frozenset(self.evidence_shapes)
        modes = frozenset(self.modes)
        if not shapes or not shapes.issubset(_EVIDENCE_SHAPES):
            raise ValueError(
                "JudgeCapability.evidence_shapes must contain only supported "
                f"shapes, got {sorted(shapes)!r}"
            )
        if not modes or not modes.issubset(_LATENCY_MODES):
            raise ValueError(
                "JudgeCapability.modes must contain only 'reflex'/'deliberate'"
            )
        if (
            isinstance(self.latency_slo_s, bool)
            or not isinstance(self.latency_slo_s, (int, float))
            or not math.isfinite(float(self.latency_slo_s))
            or float(self.latency_slo_s) <= 0
        ):
            raise ValueError(
                "JudgeCapability.latency_slo_s must be a positive finite number"
            )
        if (
            isinstance(self.max_inflight, bool)
            or not isinstance(self.max_inflight, int)
            or self.max_inflight < 1
        ):
            raise ValueError(
                "JudgeCapability.max_inflight must be a positive integer"
            )
        object.__setattr__(self, "evidence_shapes", shapes)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "latency_slo_s", float(self.latency_slo_s))


@dataclass(frozen=True)
class JudgeRoute:
    """One named, already-constructed judge and its reviewed capability."""

    name: str
    adapter: Any
    capability: JudgeCapability
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("JudgeRoute.name must be a non-empty string")
        if not callable(getattr(self.adapter, "judge", None)):
            raise ValueError(
                f"JudgeRoute {self.name!r} adapter has no callable judge()"
            )
        if not isinstance(self.capability, JudgeCapability):
            raise ValueError(
                f"JudgeRoute {self.name!r} requires an exact JudgeCapability"
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("JudgeRoute.priority must be an integer")


@dataclass(frozen=True)
class JudgeRouteReceipt:
    """One bounded routing outcome, safe to retain in traces/benchmarks."""

    route: str | None
    evidence_shape: EvidenceShape
    mode: LatencyMode
    outcome: Literal[
        "decisive",
        "uncertain",
        "provider_error",
        "malformed",
        "deadline",
        "capacity",
        "no_route",
        "cancelled",
    ]
    latency_s: float
    deadline_s: float
    confidence: float | None = None
    exception_type: str | None = None


@dataclass
class _RouteState:
    route: JudgeRoute
    inflight: int = 0
    successes: int = 0
    failures: int = 0
    latency_ewma_s: float | None = None


@dataclass(frozen=True)
class _RouteResult:
    route: JudgeRoute
    verdict: Any = None
    outcome: str = "provider_error"
    latency_s: float = 0.0
    exception_type: str | None = None
    exception: BaseException | None = None


def _valid_verdict(value: Any) -> bool:
    satisfied = getattr(value, "satisfied", None)
    confidence = getattr(value, "confidence", None)
    evidence = getattr(value, "evidence", None)
    return (
        isinstance(satisfied, bool)
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
        and isinstance(evidence, str)
    )


def _evidence_shape(
    *,
    frame: Any,
    frames: list | None,
    audio: Any,
    context: dict | None,
) -> EvidenceShape:
    requested = (
        context.get(_EVIDENCE_SHAPE_CONTEXT_KEY)
        if isinstance(context, dict)
        else None
    )
    if requested is not None:
        if requested not in _EVIDENCE_SHAPES:
            raise ValueError(
                f"unsupported requested evidence shape {requested!r}"
            )
        return requested
    if (
        isinstance(context, dict)
        and context.get("event_chain")
        and (frame is not None or frames)
        and audio is None
    ):
        # Motion-sensitive event-chain watches require a declared temporal
        # route. Its physical transport may be native video or an explicitly
        # configured timestamped image burst, but its logical evidence is a
        # bounded temporal clip.
        return "video_clip"
    if audio is not None and (frame is not None or frames):
        return "fused"
    if audio is not None:
        return "audio"
    if frames:
        return "ordered_images"
    return "single_image"


class JudgePool:
    """A bounded Vision judge router with one authoritative decision deadline.

    ``mode='reflex'`` rejects routes whose declared latency SLO exceeds
    ``deadline_s``.  ``mode='deliberate'`` may use any route that explicitly
    declares deliberate support, but the same configured hard deadline still
    bounds this pool instance.
    """

    def __init__(
        self,
        routes: Iterable[JudgeRoute],
        *,
        mode: LatencyMode = "reflex",
        deadline_s: float = 1.8,
        hedge_after_s: float = 0.2,
        max_parallel: int = 2,
        min_confidence: float = 0.7,
        receipt_limit: int = 256,
        detector: Any = None,
        on_receipt: Callable[[JudgeRouteReceipt], Any] | None = None,
        terminal_outcomes: Iterable[str] = (),
    ) -> None:
        route_list = sorted(
            tuple(routes), key=lambda item: (item.priority, item.name)
        )
        if not route_list:
            raise ValueError("JudgePool requires at least one JudgeRoute")
        names = [route.name for route in route_list]
        if len(set(names)) != len(names):
            raise ValueError("JudgePool route names must be unique")
        if mode not in _LATENCY_MODES:
            raise ValueError("JudgePool.mode must be 'reflex' or 'deliberate'")
        for name, value, allow_zero in (
            ("deadline_s", deadline_s, False),
            ("hedge_after_s", hedge_after_s, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < (0.0 if allow_zero else 1e-12)
            ):
                raise ValueError(f"JudgePool.{name} has an invalid value")
        if (
            isinstance(max_parallel, bool)
            or not isinstance(max_parallel, int)
            or max_parallel < 1
        ):
            raise ValueError("JudgePool.max_parallel must be positive")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("JudgePool.min_confidence must be in [0, 1]")
        if (
            isinstance(receipt_limit, bool)
            or not isinstance(receipt_limit, int)
            or receipt_limit < 1
        ):
            raise ValueError("JudgePool.receipt_limit must be positive")

        self._routes = route_list
        self._states = {route.name: _RouteState(route) for route in route_list}
        self._mode = mode
        self._deadline_s = float(deadline_s)
        self._hedge_after_s = float(hedge_after_s)
        self._max_parallel = max_parallel
        self._min_confidence = float(min_confidence)
        self._detector = detector
        self._on_receipt = on_receipt
        terminal = frozenset(terminal_outcomes)
        supported_outcomes = {
            "provider_error",
            "malformed",
            "deadline",
            "capacity",
            "no_route",
        }
        if not terminal.issubset(supported_outcomes):
            raise ValueError(
                "JudgePool.terminal_outcomes contains unsupported outcomes"
            )
        self._terminal_outcomes = terminal
        self._receipts: deque[JudgeRouteReceipt] = deque(maxlen=receipt_limit)
        self._state_lock = threading.Lock()
        self._terminal_errors: deque[BaseException] = deque(maxlen=256)

    @property
    def mode(self) -> LatencyMode:
        return self._mode

    @property
    def deadline_s(self) -> float:
        return self._deadline_s

    def receipts(self) -> tuple[JudgeRouteReceipt, ...]:
        with self._state_lock:
            return tuple(self._receipts)

    def snapshot(self) -> dict[str, Any]:
        """Return bounded operational state without adapter objects or secrets."""
        with self._state_lock:
            routes = {
                name: {
                    "inflight": state.inflight,
                    "max_inflight": state.route.capability.max_inflight,
                    "successes": state.successes,
                    "failures": state.failures,
                    "latency_ewma_s": state.latency_ewma_s,
                    "evidence_shapes": sorted(
                        state.route.capability.evidence_shapes
                    ),
                    "modes": sorted(state.route.capability.modes),
                    "latency_slo_s": state.route.capability.latency_slo_s,
                }
                for name, state in self._states.items()
            }
        return {
            "mode": self._mode,
            "deadline_s": self._deadline_s,
            "max_parallel": self._max_parallel,
            "routes": routes,
        }

    def take_terminal_error(self) -> BaseException | None:
        """Consume a configured terminal routing failure after cognition degrades it.

        Percept deliberately converts provider exceptions to an UNKNOWN verdict.
        An async runtime or paid qualification lane may still need to stop
        already-accepted pending work. This opt-in latch preserves that signal
        without changing the default fail-closed cognition contract.
        """
        with self._state_lock:
            error = (
                self._terminal_errors.popleft()
                if self._terminal_errors
                else None
            )
        return error

    def _set_terminal_error(self, error: BaseException) -> None:
        with self._state_lock:
            self._terminal_errors.append(error)

    def _eligible(self, shape: EvidenceShape) -> tuple[JudgeRoute, ...]:
        return tuple(
            route
            for route in self._routes
            if shape in route.capability.evidence_shapes
            and self._mode in route.capability.modes
            and (
                self._mode != "reflex"
                or route.capability.latency_slo_s <= self._deadline_s
            )
        )

    def _reserve(self, route: JudgeRoute) -> bool:
        with self._state_lock:
            state = self._states[route.name]
            if state.inflight >= route.capability.max_inflight:
                return False
            state.inflight += 1
            return True

    def _release(
        self,
        route: JudgeRoute,
        *,
        success: bool,
        latency_s: float,
    ) -> None:
        with self._state_lock:
            state = self._states[route.name]
            state.inflight = max(0, state.inflight - 1)
            if success:
                state.successes += 1
            else:
                state.failures += 1
            if latency_s >= 0:
                state.latency_ewma_s = (
                    latency_s
                    if state.latency_ewma_s is None
                    else 0.2 * latency_s + 0.8 * state.latency_ewma_s
                )

    def _record(self, receipt: JudgeRouteReceipt) -> None:
        with self._state_lock:
            self._receipts.append(receipt)
        if self._on_receipt is not None:
            try:
                self._on_receipt(receipt)
            except Exception:
                # Observability must never become the judgment failure.
                pass

    async def _call_route(
        self,
        route: JudgeRoute,
        delay_s: float,
        condition: str,
        frame: Any,
        kwargs: dict[str, Any],
        dispatch_slots: asyncio.Semaphore,
    ) -> _RouteResult:
        if delay_s:
            await asyncio.sleep(delay_s)
        async with dispatch_slots:
            if not self._reserve(route):
                return _RouteResult(route=route, outcome="capacity")
            started = time.monotonic()
            success = False
            try:
                try:
                    verdict = await route.adapter.judge(
                        condition,
                        frame,
                        **kwargs,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - route failure is fail-closed
                    return _RouteResult(
                        route=route,
                        outcome="provider_error",
                        latency_s=time.monotonic() - started,
                        exception_type=type(exc).__name__,
                        exception=exc,
                    )
                if not _valid_verdict(verdict):
                    return _RouteResult(
                        route=route,
                        verdict=verdict,
                        outcome="malformed",
                        latency_s=time.monotonic() - started,
                    )
                success = True
                outcome = (
                    "decisive"
                    if float(verdict.confidence) >= self._min_confidence
                    else "uncertain"
                )
                return _RouteResult(
                    route=route,
                    verdict=verdict,
                    outcome=outcome,
                    latency_s=time.monotonic() - started,
                )
            finally:
                self._release(
                    route,
                    success=success,
                    latency_s=time.monotonic() - started,
                )

    @staticmethod
    def _consume_task(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _unknown(
        self,
        shape: EvidenceShape,
        reason: str,
    ) -> Verdict:
        return Verdict(
            satisfied=False,
            confidence=0.0,
            evidence=(
                f"judge pool unavailable ({reason}); mode={self._mode}; "
                f"evidence_shape={shape}; deadline_s={self._deadline_s:g}"
            ),
        )

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        shape = _evidence_shape(
            frame=frame,
            frames=frames,
            audio=audio,
            context=context,
        )
        eligible = self._eligible(shape)
        if not eligible:
            self._record(
                JudgeRouteReceipt(
                    route=None,
                    evidence_shape=shape,
                    mode=self._mode,
                    outcome="no_route",
                    latency_s=0.0,
                    deadline_s=self._deadline_s,
                )
            )
            if "no_route" in self._terminal_outcomes:
                error = RuntimeError(
                    f"judge pool has no eligible {shape} route"
                )
                self._set_terminal_error(error)
                raise error
            return self._unknown(shape, "no eligible route")

        kwargs = {
            "roi": roi,
            "exemplar": exemplar,
            "corrections": corrections,
            "frames": frames,
            "audio": audio,
            "context": context,
        }
        # Schedule every eligible route so capacity exhaustion on a preferred
        # route can fall through to a healthy lower-priority route. The
        # semaphore bounds physical provider execution for this decision;
        # route-level max_inflight bounds occupancy across concurrent
        # decisions.
        selected = eligible
        if shape == "video_clip":
            # If a deployment publishes both exact temporal transports, prefer
            # native video before the timestamped-image alternative. Both are
            # explicit video_clip routes; this only resolves the transport
            # tie, without inventing a fallback for ordinary image judges.
            selected = sorted(
                eligible,
                key=lambda route: (
                    0
                    if getattr(route.adapter, "TEMPORAL_TRANSPORT", None)
                    == "native_video"
                    else 1,
                    route.priority,
                    route.name,
                ),
            )
        dispatch_slots = asyncio.Semaphore(self._max_parallel)
        tasks: set[asyncio.Task] = set()
        task_routes: dict[asyncio.Task, JudgeRoute] = {}
        for index, route in enumerate(selected):
            task = asyncio.create_task(
                self._call_route(
                    route,
                    min(index * self._hedge_after_s, self._deadline_s),
                    condition,
                    frame,
                    kwargs,
                    dispatch_slots,
                )
            )
            task.add_done_callback(self._consume_task)
            tasks.add(task)
            task_routes[task] = route

        cancelled_receipts: set[asyncio.Task] = set()
        try:
            loop = asyncio.get_running_loop()
            expires_at = loop.time() + self._deadline_s
            uncertain: Verdict | None = None
            while tasks:
                remaining = expires_at - loop.time()
                if remaining <= 0:
                    break
                done, tasks = await asyncio.wait(
                    tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                # ``asyncio.wait`` returns a set. Resolve a simultaneous hedge
                # batch in declared route priority order, record every
                # physical completion, and let any configured terminal outcome
                # win before accepting a decisive result.
                results = [
                    task.result()
                    for task in task_routes
                    if task in done
                ]
                for result in results:
                    confidence = (
                        float(result.verdict.confidence)
                        if _valid_verdict(result.verdict)
                        else None
                    )
                    self._record(
                        JudgeRouteReceipt(
                            route=result.route.name,
                            evidence_shape=shape,
                            mode=self._mode,
                            outcome=result.outcome,
                            latency_s=result.latency_s,
                            deadline_s=self._deadline_s,
                            confidence=confidence,
                            exception_type=result.exception_type,
                        )
                    )
                terminal_result = next(
                    (
                        result
                        for result in results
                        if result.outcome in self._terminal_outcomes
                    ),
                    None,
                )
                if terminal_result is not None:
                    if terminal_result.exception is not None:
                        self._set_terminal_error(
                            terminal_result.exception
                        )
                        raise terminal_result.exception
                    error = RuntimeError(
                        "judge pool terminal route outcome: "
                        f"{terminal_result.outcome} "
                        f"({terminal_result.route.name})"
                    )
                    self._set_terminal_error(error)
                    raise error
                decisive = next(
                    (
                        result
                        for result in results
                        if result.outcome == "decisive"
                    ),
                    None,
                )
                if decisive is not None:
                    return decisive.verdict
                for result in results:
                    if result.outcome == "uncertain":
                        if (
                            uncertain is None
                            or float(result.verdict.confidence)
                            > float(uncertain.confidence)
                        ):
                            uncertain = result.verdict

            if not tasks:
                return uncertain or self._unknown(
                    shape,
                    "all eligible routes were unavailable",
                )
            self._record(
                JudgeRouteReceipt(
                    route=None,
                    evidence_shape=shape,
                    mode=self._mode,
                    outcome="deadline",
                    latency_s=self._deadline_s,
                    deadline_s=self._deadline_s,
                    confidence=(
                        float(uncertain.confidence)
                        if uncertain is not None
                        else None
                    ),
                )
            )
            if "deadline" in self._terminal_outcomes:
                error = TimeoutError(
                    f"judge pool deadline expired after {self._deadline_s:g}s"
                )
                self._set_terminal_error(error)
                raise error
            return uncertain or self._unknown(shape, "deadline expired")
        finally:
            for task in task_routes:
                if task.done():
                    continue
                task.cancel()
                cancelled_receipts.add(task)
                self._record(
                    JudgeRouteReceipt(
                        route=task_routes[task].name,
                        evidence_shape=shape,
                        mode=self._mode,
                        outcome="cancelled",
                        latency_s=0.0,
                        deadline_s=self._deadline_s,
                    )
                )
            if cancelled_receipts:
                # One non-blocking turn lets cancellation-friendly adapters
                # release route occupancy. Cancellation-hostile adapters stay
                # visible as inflight/capacity until they really terminate.
                await asyncio.sleep(0)

    async def detect(self, frame: Any, goals: list[Any]) -> list[Any]:
        if self._detector is None:
            return []
        try:
            result = await asyncio.wait_for(
                self._detector.detect(frame, goals),
                timeout=self._deadline_s,
            )
        except Exception:
            return []
        return result if isinstance(result, list) else []

    async def detect_events(
        self,
        frames: list,
        goals: list[Any],
    ) -> list[Any]:
        if self._detector is None:
            return []
        method = getattr(self._detector, "detect_events", None)
        if not callable(method):
            return []
        try:
            result = await asyncio.wait_for(
                method(frames, goals),
                timeout=self._deadline_s,
            )
        except Exception:
            return []
        return result if isinstance(result, list) else []


__all__ = [
    "EvidenceShape",
    "JudgeCapability",
    "JudgePool",
    "JudgeRoute",
    "JudgeRouteReceipt",
    "LatencyMode",
]
