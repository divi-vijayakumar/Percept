from __future__ import annotations

import asyncio
import inspect
import math
import time

import pytest

from percept import Verdict
from percept.backends.judge_pool import (
    REFLEX_PROVIDER_DEADLINE_S,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
    _valid_verdict,
)
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime.worker import RuntimeStats
from percept.runtime.worker import run as worker_run


class _Judge:
    def __init__(
        self,
        verdict: Verdict,
        *,
        delay_s: float = 0.0,
        failure: Exception | None = None,
    ) -> None:
        self.verdict = verdict
        self.delay_s = delay_s
        self.failure = failure
        self.calls = 0
        self.contexts: list[dict | None] = []

    async def judge(self, _condition, _frame, **kwargs):
        self.calls += 1
        self.contexts.append(kwargs.get("context"))
        await asyncio.sleep(self.delay_s)
        if self.failure is not None:
            raise self.failure
        return self.verdict


def _cap(
    *shapes: str,
    latency_slo_s: float = 0.05,
    max_inflight: int = 1,
):
    return JudgeCapability(
        evidence_shapes=frozenset(shapes),
        latency_slo_s=latency_slo_s,
        modes=frozenset({"reflex", "deliberate"}),
        max_inflight=max_inflight,
    )


def _record_dispatch_delays(pool: JudgePool) -> list[tuple[str, float]]:
    """Capture the hedge delay the pool actually schedules for each route.

    The stagger is ``min(index * hedge_after_s, deadline_s)`` and it is otherwise
    invisible: a route delayed past the deadline and a route delayed *to* the
    deadline both end up cancelled, so only the scheduled number distinguishes a
    clamped stagger from an unclamped one.
    """
    observed: list[tuple[str, float]] = []
    original = pool._call_route

    async def _spy(route, delay_s, *args, **kwargs):
        observed.append((route.name, delay_s))
        return await original(route, delay_s, *args, **kwargs)

    pool._call_route = _spy  # type: ignore[method-assign]
    return observed


@pytest.mark.asyncio
async def test_reflex_pool_hedges_and_first_confident_judge_wins():
    slow = _Judge(
        Verdict(False, 0.95, "slow no"),
        delay_s=0.2,
    )
    fast = _Judge(
        Verdict(True, 0.91, "fast yes"),
        delay_s=0.01,
    )
    pool = JudgePool(
        [
            JudgeRoute("slow", slow, _cap("single_image"), priority=0),
            JudgeRoute("fast", fast, _cap("single_image"), priority=1),
        ],
        deadline_s=0.1,
        hedge_after_s=0.005,
    )

    started = time.monotonic()
    verdict = await pool.judge("condition", b"frame")
    elapsed = time.monotonic() - started

    assert verdict.satisfied is True
    assert elapsed < 0.1
    assert fast.calls == 1
    assert any(
        receipt.route == "fast" and receipt.outcome == "decisive"
        for receipt in pool.receipts()
    )


@pytest.mark.asyncio
async def test_deadline_is_an_outcome_bound_not_a_health_only_threshold():
    slow = _Judge(
        Verdict(True, 0.99, "too late"),
        delay_s=0.2,
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "slow",
                slow,
                _cap("single_image", latency_slo_s=0.01),
            )
        ],
        deadline_s=0.03,
    )

    started = time.monotonic()
    verdict = await pool.judge("condition", b"frame")
    elapsed = time.monotonic() - started

    assert verdict.satisfied is False
    assert verdict.confidence == 0.0
    assert "deadline expired" in verdict.evidence
    assert elapsed < 0.1
    assert [receipt.outcome for receipt in pool.receipts()][-2:] == [
        "deadline",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_parent_cancellation_cancels_friendly_route_and_releases_capacity():
    entered = asyncio.Event()

    class Friendly:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    pool = JudgePool(
        [JudgeRoute("friendly", Friendly(), _cap("single_image"))],
        deadline_s=1.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert pool.snapshot()["routes"]["friendly"]["inflight"] == 0
    assert pool.receipts()[-1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_parent_cancellation_keeps_hostile_route_physically_occupied():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Hostile:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return Verdict(False, 0.9, "released")

    pool = JudgePool(
        [JudgeRoute("hostile", Hostile(), _cap("single_image"))],
        deadline_s=1.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert pool.snapshot()["routes"]["hostile"]["inflight"] == 1
    unavailable = await pool.judge("condition", b"next")
    assert unavailable.confidence == 0.0
    assert pool.receipts()[-1].outcome == "capacity"

    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if pool.snapshot()["routes"]["hostile"]["inflight"] == 0:
            break
    assert pool.snapshot()["routes"]["hostile"]["inflight"] == 0


@pytest.mark.asyncio
async def test_simultaneous_hedges_record_every_physical_completion():
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class BarrierJudge:
        def __init__(self, evidence):
            self.evidence = evidence

        async def judge(self, *_args, **_kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return Verdict(True, 0.99, self.evidence)

    pool = JudgePool(
        [
            JudgeRoute("first", BarrierJudge("first"), _cap("single_image")),
            JudgeRoute("second", BarrierJudge("second"), _cap("single_image")),
        ],
        deadline_s=0.2,
        max_parallel=2,
        hedge_after_s=0.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await both_entered.wait()
    release.set()
    verdict = await task

    assert verdict.evidence == "first"
    assert [
        (receipt.route, receipt.outcome)
        for receipt in pool.receipts()
    ] == [("first", "decisive"), ("second", "decisive")]


@pytest.mark.asyncio
async def test_simultaneous_terminal_failure_wins_over_decisive_hedge():
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class BarrierJudge:
        def __init__(self, *, fail=False):
            self.fail = fail

        async def judge(self, *_args, **_kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            if self.fail:
                raise RuntimeError("quota")
            return Verdict(True, 0.99, "grounded")

    pool = JudgePool(
        [
            JudgeRoute("winner", BarrierJudge(), _cap("single_image")),
            JudgeRoute(
                "terminal",
                BarrierJudge(fail=True),
                _cap("single_image"),
            ),
        ],
        deadline_s=0.2,
        max_parallel=2,
        hedge_after_s=0.0,
        terminal_outcomes={"provider_error"},
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await both_entered.wait()
    release.set()
    with pytest.raises(RuntimeError, match="quota"):
        await task

    assert {
        (receipt.route, receipt.outcome)
        for receipt in pool.receipts()
    } == {
        ("winner", "decisive"),
        ("terminal", "provider_error"),
    }
    terminal = pool.take_terminal_error()
    assert isinstance(terminal, RuntimeError)
    assert str(terminal) == "quota"


@pytest.mark.asyncio
async def test_reflex_rejects_route_whose_declared_slo_exceeds_deadline():
    judge = _Judge(Verdict(True, 0.99, "would be yes"))
    pool = JudgePool(
        [
            JudgeRoute(
                "deliberate-only-speed",
                judge,
                _cap("single_image", latency_slo_s=4.0),
            )
        ],
        mode="reflex",
        deadline_s=1.8,
    )

    verdict = await pool.judge("condition", b"frame")

    assert verdict.satisfied is False
    assert judge.calls == 0
    assert pool.receipts()[-1].outcome == "no_route"


@pytest.mark.asyncio
async def test_native_clip_request_never_silently_flattens_to_ordered_images():
    strip = _Judge(Verdict(True, 0.99, "strip"))
    clip = _Judge(Verdict(True, 0.95, "native clip"))
    pool = JudgePool(
        [
            JudgeRoute("strip", strip, _cap("ordered_images")),
            JudgeRoute("clip", clip, _cap("video_clip")),
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "condition",
        b"clip-payload",
        context={"_percept_evidence_shape": "video_clip"},
    )

    assert verdict.evidence == "native clip"
    assert strip.calls == 0
    assert clip.calls == 1


@pytest.mark.asyncio
async def test_context_stream_request_does_not_fall_back_to_stateless_strip():
    strip = _Judge(Verdict(True, 0.99, "stateless"))
    observer = _Judge(Verdict(True, 0.91, "context held"))
    pool = JudgePool(
        [
            JudgeRoute("strip", strip, _cap("ordered_images")),
            JudgeRoute("observer", observer, _cap("context_stream")),
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "condition",
        b"latest",
        frames=[b"earlier"],
        context={"_percept_evidence_shape": "context_stream"},
    )

    assert verdict.evidence == "context held"
    assert observer.calls == 1
    assert strip.calls == 0


@pytest.mark.asyncio
async def test_uncertain_primary_does_not_preempt_confident_hedge():
    uncertain = _Judge(
        Verdict(False, 0.2, "unknown"),
        delay_s=0.001,
    )
    decisive = _Judge(
        Verdict(True, 0.9, "grounded"),
        delay_s=0.01,
    )
    pool = JudgePool(
        [
            JudgeRoute("uncertain", uncertain, _cap("single_image")),
            JudgeRoute("decisive", decisive, _cap("single_image")),
        ],
        deadline_s=0.1,
        hedge_after_s=0.0,
    )

    verdict = await pool.judge("condition", b"frame")

    assert verdict.satisfied is True
    assert verdict.evidence == "grounded"


@pytest.mark.asyncio
async def test_route_capacity_is_bounded_while_a_prior_call_is_running():
    entered = asyncio.Event()

    class HeldJudge:
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, *_args, **_kwargs):
            self.calls += 1
            entered.set()
            await asyncio.sleep(0.05)
            return Verdict(True, 0.99, "late")

    held = HeldJudge()
    pool = JudgePool(
        [
            JudgeRoute(
                "held",
                held,
                _cap(
                    "single_image",
                    latency_slo_s=0.1,
                    max_inflight=1,
                ),
            )
        ],
        deadline_s=0.2,
    )

    first_task = asyncio.create_task(
        pool.judge("condition", b"frame-1")
    )
    await entered.wait()
    second = await pool.judge("condition", b"frame-2")
    first = await first_task

    assert first.satisfied is True
    assert second.confidence == 0.0
    assert held.calls == 1
    assert any(
        receipt.route == "held" and receipt.outcome == "capacity"
        for receipt in pool.receipts()
    )
    assert pool.snapshot()["routes"]["held"]["inflight"] == 0


@pytest.mark.asyncio
async def test_occupied_primary_falls_through_to_lower_priority_route():
    entered = asyncio.Event()
    release = asyncio.Event()

    class HeldPrimary:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            await release.wait()
            return Verdict(True, 0.99, "primary")

    fallback = _Judge(Verdict(True, 0.95, "fallback"))
    pool = JudgePool(
        [
            JudgeRoute(
                "primary",
                HeldPrimary(),
                _cap("single_image", max_inflight=1),
                priority=0,
            ),
            JudgeRoute(
                "fallback",
                fallback,
                _cap("single_image", max_inflight=1),
                priority=1,
            ),
        ],
        deadline_s=0.2,
        hedge_after_s=0.0,
        max_parallel=1,
    )

    first = asyncio.create_task(pool.judge("condition", b"first"))
    await entered.wait()
    second = await pool.judge("condition", b"second")

    assert second.evidence == "fallback"
    assert fallback.calls == 1
    assert any(
        receipt.route == "primary" and receipt.outcome == "capacity"
        for receipt in pool.receipts()
    )

    release.set()
    assert (await first).evidence == "primary"


def test_capability_and_pool_configuration_fail_loudly():
    judge = _Judge(Verdict(False, 0.0, "unknown"))
    with pytest.raises(ValueError):
        JudgeCapability(frozenset(), latency_slo_s=1.0)
    with pytest.raises(ValueError):
        JudgeCapability(
            frozenset({"sprite"}),  # type: ignore[arg-type]
            latency_slo_s=1.0,
        )
    with pytest.raises(ValueError):
        JudgePool([])
    with pytest.raises(ValueError):
        JudgePool(
            [
                JudgeRoute("same", judge, _cap("single_image")),
                JudgeRoute("same", judge, _cap("single_image")),
            ]
        )


@pytest.mark.asyncio
async def test_pool_is_on_the_real_async_runtime_watch_path():
    width = height = 8
    still = gray_frame(width, height, 0)
    pixels = [0] * (width * height)
    for index in range(40):
        pixels[index] = 255
    motion = Frame(width, height, tuple(pixels))

    slow = _Judge(Verdict(False, 0.95, "slow"), delay_s=0.2)
    fast = _Judge(Verdict(True, 0.95, "fast"), delay_s=0.005)
    pool = JudgePool(
        [
            JudgeRoute("slow", slow, _cap("single_image")),
            JudgeRoute("fast", fast, _cap("single_image")),
        ],
        deadline_s=0.05,
        hedge_after_s=0.001,
    )
    from percept import Goal, Percept

    core = Percept.create(vision=pool, discover_plugins=False)
    core.add_goal(
        Goal(
            id="arrival",
            condition="someone arrived",
            say="arrival",
        )
    )
    stats = RuntimeStats()

    async def source():
        yield 0.0, still, b"still"
        await asyncio.sleep(0)
        yield 1.0, motion, b"motion"
        await asyncio.sleep(0)

    incidents = [
        incident
        async for incident in worker_run(
            core,
            source=source(),
            stats=stats,
            judge_dispatch="async",
        )
    ]

    assert len(incidents) == 1
    assert incidents[0].concern_id == "arrival"
    assert stats.judge_calls == 1
    assert any(
        receipt.route == "fast" and receipt.outcome == "decisive"
        for receipt in pool.receipts()
    )


# --- degenerate hedge stagger: everything at once, or nothing --------------------------------------


@pytest.mark.asyncio
async def test_degenerate_hedge_stagger_is_all_at_once_or_nothing():
    """Both ends of the stagger knob, pinned to the number the pool schedules.

    ``hedge_after_s=0.0`` is legal (the constructor allows zero for this knob and
    only this knob) and means *no* stagger: every eligible route is dispatched
    with delay 0 and the only thing left bounding physical concurrency is
    ``max_parallel``.  A ``hedge_after_s`` so large that ``index *
    hedge_after_s`` runs past the deadline is clamped to exactly ``deadline_s``,
    so the late route can never contribute — it lands a ``cancelled`` receipt
    without the adapter being entered once.
    """
    entered: list[str] = []
    two_entered = asyncio.Event()
    release = asyncio.Event()

    class _Barrier:
        def __init__(self, name: str) -> None:
            self.name = name

        async def judge(self, *_args, **_kwargs):
            entered.append(self.name)
            if len(entered) == 2:
                two_entered.set()
            await release.wait()
            return Verdict(True, 0.99, self.name)

    at_once = JudgePool(
        [
            JudgeRoute("a", _Barrier("a"), _cap("single_image"), priority=0),
            JudgeRoute("b", _Barrier("b"), _cap("single_image"), priority=1),
            JudgeRoute("c", _Barrier("c"), _cap("single_image"), priority=2),
        ],
        deadline_s=0.4,
        hedge_after_s=0.0,
        max_parallel=2,
    )
    at_once_delays = _record_dispatch_delays(at_once)

    task = asyncio.create_task(at_once.judge("condition", b"frame"))
    await two_entered.wait()

    # Every route was scheduled with zero delay -- nothing is staggered.
    assert at_once_delays == [("a", 0.0), ("b", 0.0), ("c", 0.0)]
    # ... and `max_parallel` is the ONLY thing holding the third route back.
    assert entered == ["a", "b"]

    release.set()
    verdict = await task
    assert verdict.evidence == "a"

    # The other end: a stagger far past the deadline.
    primary = _Judge(Verdict(True, 0.99, "primary"), delay_s=0.005)
    late = _Judge(Verdict(True, 0.99, "late"))
    nothing = JudgePool(
        [
            JudgeRoute("primary", primary, _cap("single_image"), priority=0),
            JudgeRoute("late", late, _cap("single_image"), priority=1),
        ],
        deadline_s=0.05,
        hedge_after_s=10.0,
        max_parallel=2,
    )
    nothing_delays = _record_dispatch_delays(nothing)

    late_verdict = await nothing.judge("condition", b"frame")

    assert late_verdict.evidence == "primary"
    # 1 * 10.0 clamped to the deadline, not scheduled 10 s out.
    assert nothing_delays == [("primary", 0.0), ("late", nothing.deadline_s)]
    assert nothing_delays[1][1] == 0.05
    assert late.calls == 0
    assert [
        (receipt.route, receipt.outcome) for receipt in nothing.receipts()
    ] == [("primary", "decisive"), ("late", "cancelled")]


@pytest.mark.asyncio
async def test_min_confidence_boundary_is_inclusive():
    """``confidence >= min_confidence`` -- exactly at the bar is DECISIVE.

    One representable step below is ``uncertain``, which never preempts and only
    survives as the fallback answer.  An exclusive comparison here would silently
    demote every judge that reports exactly the configured bar.
    """
    at_bar = _Judge(Verdict(True, 0.7, "exactly at the bar"))
    inclusive = JudgePool(
        [JudgeRoute("edge", at_bar, _cap("single_image"))],
        deadline_s=0.2,
        min_confidence=0.7,
    )

    verdict = await inclusive.judge("condition", b"frame")

    assert verdict.evidence == "exactly at the bar"
    assert [
        (receipt.route, receipt.outcome) for receipt in inclusive.receipts()
    ] == [("edge", "decisive")]

    one_step_below = math.nextafter(0.7, 0.0)
    assert one_step_below < 0.7
    under_bar = _Judge(Verdict(True, one_step_below, "one epsilon below the bar"))
    exclusive = JudgePool(
        [JudgeRoute("edge", under_bar, _cap("single_image"))],
        deadline_s=0.2,
        min_confidence=0.7,
    )

    demoted = await exclusive.judge("condition", b"frame")

    assert demoted.evidence == "one epsilon below the bar"
    assert [
        (receipt.route, receipt.outcome) for receipt in exclusive.receipts()
    ] == [("edge", "uncertain")]


class _VerdictSubclass(Verdict):
    """A deployment's own Verdict subclass -- the check is duck-typed, not exact-type."""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_verdict,valid,pool_outcome",
    [
        # `isinstance(True, int)` is True in Python, so a bool would sail
        # through a naive numeric check and be read as confidence 1.0.
        pytest.param(
            lambda: Verdict(True, True, "a bool is not a confidence"),
            False,
            "malformed",
            id="confidence_is_a_bool",
        ),
        pytest.param(
            lambda: Verdict(True, float("nan"), "not a number"),
            False,
            "malformed",
            id="confidence_is_nan",
        ),
        pytest.param(
            lambda: Verdict(True, float("inf"), "infinite"),
            False,
            "malformed",
            id="confidence_is_inf",
        ),
        pytest.param(
            lambda: Verdict(True, -0.0, "negative zero"),
            True,
            "uncertain",
            id="confidence_is_negative_zero",
        ),
        pytest.param(
            lambda: Verdict(True, 0.0, "floor"),
            True,
            "uncertain",
            id="confidence_is_zero",
        ),
        pytest.param(
            lambda: Verdict(True, 1.0, "ceiling"),
            True,
            "decisive",
            id="confidence_is_one",
        ),
        pytest.param(
            lambda: Verdict(True, 1.5, "over the ceiling"),
            False,
            "malformed",
            id="confidence_above_one",
        ),
        pytest.param(
            lambda: Verdict(True, -0.1, "under the floor"),
            False,
            "malformed",
            id="confidence_below_zero",
        ),
        pytest.param(
            lambda: Verdict(True, 0.9, None),
            False,
            "malformed",
            id="evidence_is_none",
        ),
        pytest.param(
            lambda: _VerdictSubclass(True, 0.9, "a subclass is still a verdict"),
            True,
            "decisive",
            id="verdict_subclass",
        ),
    ],
)
async def test_verdict_validity_table_at_the_pool_boundary(
    make_verdict,
    valid,
    pool_outcome,
):
    """The direct validity table, asserted on the predicate AND on the pool.

    ``_valid_verdict`` is the pool's whole malformed-detection surface, and the
    classification it produces is what the resolution loop turns into a receipt.
    Asserting only the predicate would leave the wiring untested; asserting only
    the receipt would not say which rule refused.
    """
    verdict = make_verdict()
    assert _valid_verdict(verdict) is valid

    judge = _Judge(verdict)
    pool = JudgePool(
        [JudgeRoute("only", judge, _cap("single_image"))],
        deadline_s=0.2,
    )

    returned = await pool.judge("condition", b"frame")

    receipts = pool.receipts()
    assert [
        (receipt.route, receipt.outcome) for receipt in receipts
    ] == [("only", pool_outcome)]
    if pool_outcome == "malformed":
        # A malformed reply never reaches the gate as an answer, and never
        # lands a confidence in the receipt either.
        assert receipts[0].confidence is None
        assert returned.satisfied is False
        assert returned.confidence == 0.0
        assert "judge pool unavailable" in returned.evidence
    else:
        assert returned is verdict
        assert receipts[0].confidence == float(verdict.confidence)


def test_the_pool_deadline_stays_below_the_stall_and_transport_guards():
    """The three hang guards must keep their order, as an executable check.

    Pool deadline (8 s) < runtime ``stall_after_s`` (20 s) < transport
    ``_REQUEST_TIMEOUT_MS`` (120 s).  Invert any pair and the pool stops being
    the thing that cancels first: the "stall" health event stops reporting on
    something the pool failed to bound, and the socket guard starts deciding
    outcomes.  The stall default is read out of the shipped signatures rather
    than hardcoded, so moving it is what fails this test.
    """
    from percept.backends import gemini as gemini_backend
    from percept.runtime import worker as worker_module

    stall_defaults = {
        "_LegacyJudgeScheduler.stall_after_s": inspect.signature(
            worker_module._LegacyJudgeScheduler.__init__
        ).parameters["stall_after_s"].default,
        "_JudgeScheduler.stall_after_s": inspect.signature(
            worker_module._JudgeScheduler.__init__
        ).parameters["stall_after_s"].default,
        "run.judge_stall_after_s": inspect.signature(
            worker_module.run
        ).parameters["judge_stall_after_s"].default,
    }
    assert len(set(stall_defaults.values())) == 1, stall_defaults
    stall_after_s = float(next(iter(stall_defaults.values())))
    transport_timeout_s = gemini_backend._REQUEST_TIMEOUT_MS / 1000.0

    assert REFLEX_PROVIDER_DEADLINE_S < stall_after_s < transport_timeout_s

    # The invariant is about the SHIPPED default, so pin that the pool built
    # without an explicit deadline really is the constant compared above.
    shipped = JudgePool(
        [
            JudgeRoute(
                "only",
                _Judge(Verdict(True, 0.99, "yes")),
                _cap("single_image"),
            )
        ]
    )
    assert shipped.deadline_s == REFLEX_PROVIDER_DEADLINE_S


@pytest.mark.asyncio
async def test_the_deadline_boundary_accepts_nothing_that_lands_after_it():
    """Both sides of the loop bound, with the documented side pinned.

    A route whose only completion lands past the deadline is not accepted late:
    the pool records ``deadline`` then ``cancelled`` and returns the UNKNOWN
    verdict.  And a pool whose budget is already spent when the loop is entered
    (``remaining <= 0`` on the first pass) breaks out *before* accepting any
    route result at all -- the adapter is never entered.
    """
    late = _Judge(Verdict(True, 0.99, "answered after the bell"), delay_s=0.2)
    expired = JudgePool(
        [JudgeRoute("late", late, _cap("single_image", latency_slo_s=0.01))],
        deadline_s=0.03,
    )

    verdict = await expired.judge("condition", b"frame")

    assert verdict.satisfied is False
    assert verdict.confidence == 0.0
    assert "deadline expired" in verdict.evidence
    receipts = expired.receipts()
    assert [
        (receipt.route, receipt.outcome) for receipt in receipts
    ] == [(None, "deadline"), ("late", "cancelled")]
    # The deadline receipt is charged the full budget and carries no confidence,
    # because nothing -- not even an uncertain answer -- was in hand.
    assert receipts[0].latency_s == expired.deadline_s
    assert receipts[0].confidence is None

    # `remaining <= 0` on entry: the smallest deadline the constructor accepts
    # is already spent by the time the first `loop.time()` comparison runs.
    never_asked = _Judge(Verdict(True, 0.99, "never asked"))
    spent = JudgePool(
        [
            JudgeRoute(
                "spent",
                never_asked,
                _cap("single_image", latency_slo_s=1e-12),
            )
        ],
        deadline_s=1e-12,
    )

    unknown = await spent.judge("condition", b"frame")

    assert never_asked.calls == 0
    assert unknown.satisfied is False
    assert unknown.confidence == 0.0
    assert "deadline expired" in unknown.evidence
    assert [
        (receipt.route, receipt.outcome) for receipt in spent.receipts()
    ] == [(None, "deadline"), ("spent", "cancelled")]


@pytest.mark.asyncio
async def test_a_terminal_outcome_with_no_exception_synthesizes_and_latches_one():
    """A terminal `malformed` carries no exception, so the pool must make one.

    ``provider_error`` arrives with the adapter's own exception to re-raise.
    ``malformed`` does not -- the adapter returned successfully, it just
    returned rubbish -- so a deployment that made ``malformed`` terminal would
    otherwise get a silent continue instead of the stop it configured.
    """

    class _Garbled:
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, *_args, **_kwargs):
            self.calls += 1
            return {"unexpected_shape": True, "raw": "<garbled provider payload>"}

    garbled = _Garbled()
    pool = JudgePool(
        [JudgeRoute("garbled", garbled, _cap("single_image"))],
        deadline_s=0.2,
        terminal_outcomes={"malformed"},
    )

    with pytest.raises(RuntimeError) as excinfo:
        await pool.judge("condition", b"frame")

    assert "judge pool terminal route outcome: malformed" in str(excinfo.value)
    assert "(garbled)" in str(excinfo.value)
    assert garbled.calls == 1
    assert [
        (receipt.route, receipt.outcome) for receipt in pool.receipts()
    ] == [("garbled", "malformed")]
    # Latched, and it is the SAME object -- a runtime consuming the latch to
    # abort the rest of a split batch must see the failure it actually raised.
    assert pool.take_terminal_error() is excinfo.value
    assert pool.take_terminal_error() is None

    routes = [
        JudgeRoute(
            "only",
            _Judge(Verdict(True, 0.99, "yes")),
            _cap("single_image"),
        )
    ]
    for supported in (
        "provider_error",
        "malformed",
        "deadline",
        "capacity",
        "no_route",
    ):
        assert JudgePool(routes, terminal_outcomes={supported}) is not None
    for unsupported in ("cancelled", "decisive", "uncertain", "timeout"):
        with pytest.raises(ValueError):
            JudgePool(routes, terminal_outcomes={unsupported})
