"""``vision.detect.gated`` — the hosted BYOK route that attributes an incident to a MOMENT.

Contract 1.0 exposed semantic judgment only: a hosted run yielded exactly one verdict for a whole
clip, with no *when* and no *how many*. This module covers the route that closes that gap, and the
properties a hosted security review depends on:

* the capability is DECLARED per provider and refused at preflight, not at call time;
* proposals carry temporal identity (frame index / offset / wall clock) and are bounded by an
  observable refractory, so a 600 s clip cannot become 1,200 provider calls;
* the route contacts no provider and can read no provider credential from the environment;
* cancellation is genuine — no thread pool, no child process, no orphan;
* a run that admits NOTHING raises a diagnostic instead of returning an empty list.

Everything here is offline and deterministic: gate frames are synthesized luma grids and any
provider seam is a fake that fails the test if it is ever touched.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import percept
from percept import (
    HOSTED_BYOK_CAPABILITIES,
    HOSTED_BYOK_CONTRACT_VERSION,
    HOSTED_BYOK_DEFAULT_REFRACTORY_S,
    HOSTED_BYOK_MAX_GATE_FRAME_PIXELS,
    HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS,
    HostedByokAdapterFactory,
    HostedByokContractError,
    HostedByokCancelled,
    HostedByokDetectionGoal,
    HostedByokDetectionProposal,
    HostedByokDetectionStarved,
    HostedByokGateFrame,
    HostedByokRequest,
    HostedByokUnsupportedError,
)
from percept import hosted_byok as _hosted_byok
from percept.config import Settings
from percept.runtime.source import GATE_H, GATE_W

# The gate frame this route is defined over is exactly the tier-0 downsample the capture path
# already produces, so the transport budget below is the real one, not an assumed one.
W, H = GATE_W, GATE_H
GATE_FRAME_BYTES = W * H
FPS = 2.0


def _request(
    *,
    provider: str = "gemini",
    model: str = "gemini-3.5-flash",
    capability: str = "vision.detect.gated",
    contract_version: str = HOSTED_BYOK_CONTRACT_VERSION,
    cancellation: asyncio.Event | None = None,
    timeout_s: float | None = None,
) -> HostedByokRequest:
    return HostedByokRequest(
        provider=provider,
        model=model,
        credential="opaque-lease-secret",
        credential_lease_id="lease-1",
        principal_id="principal-1",
        capability=capability,
        contract_version=contract_version,
        cancellation=cancellation,
        timeout_s=timeout_s,
    )


def _flat(fill: int) -> tuple[int, ...]:
    return tuple([fill]) * (W * H)


def _patch(fill: int, background: int, *, rows: int = 12, cols: int = 16) -> tuple[int, ...]:
    """A background grid with a bright rectangle — a LOCALIZED mover, not a whole-frame flip."""

    pixels = [background] * (W * H)
    for y in range(rows):
        for x in range(cols):
            pixels[y * W + x] = fill
    return tuple(pixels)


def _frame(index: int, pixels: tuple[int, ...], *, t0_unix: float = 1_700_000_000.0):
    offset_s = index / FPS
    return HostedByokGateFrame(
        width=W,
        height=H,
        pixels=pixels,
        frame_index=index,
        offset_s=offset_s,
        timestamp_unix_s=t0_unix + offset_s,
    )


def _still_clip(frames: int, *, fill: int = 40) -> list[HostedByokGateFrame]:
    pixels = _flat(fill)
    return [_frame(index, pixels) for index in range(frames)]


def _clip_with_occurrences(
    seconds: float,
    occurrence_times_s: tuple[float, ...],
    *,
    background: int = 40,
) -> list[HostedByokGateFrame]:
    """A still scene with a localized bright patch appearing for one frame at each named time."""

    still = _flat(background)
    moved = _patch(255, background)
    occurrence_indices = {round(t * FPS) for t in occurrence_times_s}
    total = round(seconds * FPS)
    return [
        _frame(index, moved if index in occurrence_indices else still)
        for index in range(total)
    ]


def _child_pids() -> set[int]:
    """Every live direct child of this process, by PID census — not by trusting a return.

    The census itself forks ``ps``, so that one PID is excluded; everything else attributed to
    this process is a child something under test spawned.
    """

    process = subprocess.Popen(
        ["ps", "-eo", "pid=,ppid="],
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout, _ = process.communicate()
    mine = os.getpid()
    pids: set[int] = set()
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if ppid == mine and pid != process.pid:
            pids.add(pid)
    return pids


def _bind(**kwargs):
    return HostedByokAdapterFactory().bind(_request(**kwargs))


# ---------------------------------------------------------------------------
# Declared capability + contract version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(HOSTED_BYOK_CAPABILITIES))
def test_every_supported_provider_declares_the_gated_detection_capability(provider):
    assert "vision.detect.gated" in HOSTED_BYOK_CAPABILITIES[provider]


def test_unsupported_provider_is_refused_at_preflight_before_any_frame_is_read():
    # A real Percept backend that is NOT a hosted BYOK provider. The refusal must happen while
    # constructing the route, so no clip is ever walked on an unentitled lease.
    request = _request(provider="anthropic", model="claude-opus-4-8")
    factory = HostedByokAdapterFactory()
    scanned: list[int] = []

    def counting_adapter(frame):  # pragma: no cover - must never run
        scanned.append(1)
        return 0.0

    with pytest.raises(HostedByokUnsupportedError, match="unsupported hosted BYOK provider"):
        factory.stream_gated_detections(
            request,
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _still_clip(4),
            semantic_adapter=counting_adapter,
        )
    with pytest.raises(HostedByokUnsupportedError, match="unsupported hosted BYOK provider"):
        factory.bind(request)
    assert scanned == []


def test_provider_without_the_declared_capability_is_refused_at_preflight(monkeypatch):
    # Narrow gemini's declared capability set and prove the check reads the TABLE, not a constant:
    # an entitled-looking request against a provider that no longer declares detection is refused
    # before a single gate frame is touched.
    narrowed = dict(HOSTED_BYOK_CAPABILITIES)
    narrowed["gemini"] = frozenset({"vision.judge"})
    monkeypatch.setattr(_hosted_byok, "HOSTED_BYOK_CAPABILITIES", narrowed)

    factory = HostedByokAdapterFactory()
    request = HostedByokRequest(
        provider="gemini",
        model="gemini-3.5-flash",
        credential="opaque",
        credential_lease_id="lease-1",
        principal_id="principal-1",
        capability="vision.judge",
    )
    scanned: list[int] = []

    def counting_adapter(frame):
        scanned.append(1)
        return 0.0

    with pytest.raises(HostedByokUnsupportedError, match="vision.detect.gated"):
        factory.stream_gated_detections(
            request,
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _still_clip(4),
            semantic_adapter=counting_adapter,
        )
    assert scanned == []


def test_capability_is_refused_for_a_lease_pinned_to_the_older_contract():
    # Cloud pins 1.0.0 today. That lease keeps judging unchanged, and is refused for a capability
    # its contract version never carried — rather than silently upgraded into a new surface.
    request = _request(capability="vision.judge", contract_version="1.0.0")
    assert request.contract_version == "1.0.0"
    factory = HostedByokAdapterFactory()
    with pytest.raises(HostedByokUnsupportedError, match="introduced in hosted BYOK contract"):
        factory.stream_gated_detections(
            request,
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _still_clip(4),
        )
    with pytest.raises(HostedByokUnsupportedError, match="introduced in hosted BYOK contract"):
        factory.bind(_request(capability="vision.detect.gated", contract_version="1.0.0"))
    # ...while the same 1.0.0 lease still binds and judges exactly as before.
    assert factory.bind(_request(capability="vision.judge", contract_version="1.0.0")) is not None


def test_contract_version_is_bumped_and_reported_by_build_info():
    assert HOSTED_BYOK_CONTRACT_VERSION == "1.1.0"
    assert HOSTED_BYOK_CONTRACT_VERSION != "1.0.0"
    # The version Cloud pins today is still executable: bumping is additive, never a mutation.
    assert "1.0.0" in HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS
    assert HOSTED_BYOK_CONTRACT_VERSION in HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS

    info = HostedByokAdapterFactory().build_info()
    assert info["contract_version"] == HOSTED_BYOK_CONTRACT_VERSION
    assert info["supported_contract_versions"] == list(HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS)
    assert info["capabilities"]["gemini"] == sorted(HOSTED_BYOK_CAPABILITIES["gemini"])
    assert info["detection"]["capability"] == "vision.detect.gated"
    assert info["detection"]["spends_provider_calls"] is False
    assert info["detection"]["default_refractory_s"] == HOSTED_BYOK_DEFAULT_REFRACTORY_S
    assert info["detection"]["max_gate_frame_pixels"] == HOSTED_BYOK_MAX_GATE_FRAME_PIXELS

    bound = _bind().build_info()
    assert bound["request"]["capability"] == "vision.detect.gated"
    assert bound["request"]["contract_version"] == HOSTED_BYOK_CONTRACT_VERSION
    # The credential never appears in a descriptor a hosted service will log.
    assert "opaque-lease-secret" not in repr(bound)

    # The package accessor reports it too, without widening the frozen four-field build receipt
    # that published benchmark receipts seal.
    assert percept.build_info().hosted_byok_contract_version == HOSTED_BYOK_CONTRACT_VERSION
    assert set(percept.build_info().to_dict()) == {
        "package_version",
        "git_commit",
        "contract_version",
        "build_timestamp",
    }


# ---------------------------------------------------------------------------
# Multiple proposals, each with temporal identity
# ---------------------------------------------------------------------------


def test_n_well_separated_occurrences_yield_n_proposals_with_temporal_identity():
    occurrences = (20.0, 80.0, 140.0)
    frames = _clip_with_occurrences(180.0, occurrences)
    vision = _bind()

    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="loading-dock", condition="a person enters")],
            frames,
        )
    )

    # Not one verdict for the whole clip, and not one proposal per frame.
    assert len(proposals) == len(occurrences)
    assert len(frames) == 360
    assert [p.offset_s for p in proposals] == list(occurrences)
    by_index = {frame.frame_index: frame for frame in frames}
    for proposal, occurrence in zip(proposals, occurrences):
        assert isinstance(proposal, HostedByokDetectionProposal)
        assert proposal.goal_id == "loading-dock"
        # Temporal identity is copied off the admitting frame, not re-derived.
        source = by_index[proposal.frame_index]
        assert proposal.offset_s == source.offset_s == occurrence
        assert proposal.timestamp_unix_s == source.timestamp_unix_s
        assert proposal.salience >= proposal.threshold > 0.0
        assert proposal.cause == "motion"
        assert proposal.detector_id == "motion_score.block_max"
        assert proposal.contract_version == HOSTED_BYOK_CONTRACT_VERSION
        payload = proposal.to_dict()
        assert payload["frame_index"] == proposal.frame_index
        assert payload["timestamp_unix_s"] == proposal.timestamp_unix_s


def test_streaming_form_yields_incrementally_and_can_stop_early():
    frames = _clip_with_occurrences(180.0, (20.0, 80.0, 140.0))
    vision = _bind()
    seen: list[int] = []

    def counting_adapter(frame):
        seen.append(1)
        return 0.0

    async def scenario():
        stream = vision.stream_gated_detections(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            frames,
            semantic_adapter=counting_adapter,
        )
        first = await stream.__anext__()
        scanned_at_first = len(seen)
        await stream.aclose()
        return first, scanned_at_first

    first, scanned_at_first = asyncio.run(scenario())
    assert first.offset_s == 20.0
    # A streaming caller stops paying for the rest of the clip.
    assert scanned_at_first < len(frames)
    assert len(seen) == scanned_at_first


def test_per_goal_refractory_is_independent_and_shared_with_the_gate_cadence():
    frames = _clip_with_occurrences(120.0, (10.0, 18.0))
    vision = _bind()

    proposals = asyncio.run(
        vision.detect_gated(
            [
                HostedByokDetectionGoal(goal_id="slow", condition="a person enters"),
                HostedByokDetectionGoal(
                    goal_id="fast", condition="a person enters", refractory_s=3.0
                ),
            ],
            frames,
        )
    )
    by_goal: dict[str, list[float]] = {}
    for proposal in proposals:
        by_goal.setdefault(proposal.goal_id, []).append(proposal.offset_s)
    assert by_goal["slow"] == [10.0]
    assert by_goal["fast"] == [10.0, 18.0]


# ---------------------------------------------------------------------------
# The refractory is OBSERVABLE
# ---------------------------------------------------------------------------


def test_two_occurrences_inside_the_refractory_yield_one_proposal_outside_it_yield_two():
    goal = HostedByokDetectionGoal(goal_id="g", condition="a person enters", refractory_s=15.0)
    vision = _bind()

    inside = _clip_with_occurrences(120.0, (10.0, 20.0))  # 10 s apart, inside the 15 s window
    outside = _clip_with_occurrences(120.0, (10.0, 40.0))  # 30 s apart, outside it

    inside_proposals = asyncio.run(vision.detect_gated([goal], inside))
    outside_proposals = asyncio.run(vision.detect_gated([goal], outside))

    assert [p.offset_s for p in inside_proposals] == [10.0]
    assert [p.offset_s for p in outside_proposals] == [10.0, 40.0]


def test_refractory_must_be_a_finite_positive_number():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(HostedByokContractError, match="refractory_s"):
            HostedByokDetectionGoal(goal_id="g", condition="c", refractory_s=bad)


def test_bounded_worst_case_for_a_600s_clip_at_the_default_refractory():
    # The stated bound Cloud budgets against: a 600 s clip whose EVERY frame is maximally salient.
    frames = [
        _frame(index, _flat(0 if index % 2 else 255))
        for index in range(round(600.0 * FPS))
    ]
    assert len(frames) == 1200
    # Transport budget: gate frames, not full frames.
    assert len(frames) * GATE_FRAME_BYTES == 2_073_600
    assert len(frames) * GATE_FRAME_BYTES < _hosted_byok._MAX_PROCESS_IPC_BYTES

    vision = _bind()
    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            frames,
        )
    )
    # 600 s / 15 s refractory. The very first frame has no predecessor, so it can never admit;
    # admissions land at 0.5, 15.5, ... 585.5.
    assert len(proposals) == 40
    assert proposals[0].offset_s == 0.5
    assert proposals[-1].offset_s == 585.5
    separations = {
        round(b.offset_s - a.offset_s, 6) for a, b in zip(proposals, proposals[1:])
    }
    assert separations == {HOSTED_BYOK_DEFAULT_REFRACTORY_S}
    # Provider calls spent by detection: zero. This is why the bound matters — it caps the
    # judgments the CALLER will then spend, one per proposal at most.
    assert len(proposals) <= int(600.0 / HOSTED_BYOK_DEFAULT_REFRACTORY_S)


# ---------------------------------------------------------------------------
# No provider is contacted and no provider credential is readable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_key",
    [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_AISTUDIO_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    ],
)
def test_no_provider_credential_can_be_read_from_the_environment_during_detection(
    monkeypatch, env_key
):
    # A caller-supplied tier-0 channel runs INSIDE the detection scope; it is the one place
    # arbitrary code executes on this path, so it is exactly where the guard must hold. Deleting
    # the firewall makes this adapter return the sentinel and the run succeed — this test fails.
    monkeypatch.setenv(env_key, "super-secret-value")
    observed: list[object] = []

    def leaking_adapter(frame):
        observed.append(os.environ.get(env_key))
        return 0.0

    vision = _bind()
    with pytest.raises(HostedByokContractError, match="must not read provider"):
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
                _clip_with_occurrences(60.0, (10.0,)),
                semantic_adapter=leaking_adapter,
            )
        )
    assert observed == []
    # Outside the scope the process environment is untouched and os.environ is the real object.
    assert os.environ.get(env_key) == "super-secret-value"
    assert not isinstance(os.environ, _hosted_byok._CredentialFirewallEnviron)


def test_credential_firewall_restores_the_exact_environ_object_and_spares_other_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("PERCEPT_UNRELATED_KEY", "visible")
    original_environ = os.environ
    unrelated: list[object] = []

    def probing_adapter(frame):
        # Non-provider keys stay readable: the firewall is targeted, not a blanket environment ban.
        unrelated.append(os.environ.get("PERCEPT_UNRELATED_KEY"))
        return 0.0

    vision = _bind()
    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _clip_with_occurrences(60.0, (10.0,)),
            semantic_adapter=probing_adapter,
        )
    )
    assert len(proposals) == 1
    assert set(unrelated) == {"visible"}
    assert os.environ is original_environ
    assert os.environ.get("GEMINI_API_KEY") == "secret"


def test_detection_constructs_no_provider_client_or_backend():
    def exploding_client_factory(request, timeout_s):  # pragma: no cover - must never run
        raise AssertionError("gated detection must not construct a provider client")

    def exploding_backend_factory(request, timeout_s):  # pragma: no cover - must never run
        raise AssertionError("gated detection must not construct a provider backend")

    factory = HostedByokAdapterFactory(
        gemini_client_factory=exploding_client_factory,
        openrouter_client_factory=exploding_client_factory,
        backend_factories={"gemini": exploding_backend_factory},
    )
    vision = factory.bind(_request())
    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _clip_with_occurrences(60.0, (10.0,)),
        )
    )
    assert len(proposals) == 1


# ---------------------------------------------------------------------------
# Cancellation is genuine: no executor, no thread, no orphaned child
# ---------------------------------------------------------------------------


def test_detection_never_offloads_to_the_loops_default_executor():
    # Cloud installs a private ThreadPoolExecutor as the loop default and joins it; anything this
    # SDK offloads would be caught there. MotionGate is synchronous arithmetic over a tiny buffer
    # and must stay on the loop.
    class _RecordingExecutor(ThreadPoolExecutor):
        def __init__(self) -> None:
            super().__init__(max_workers=1)
            self.submissions: list[object] = []

        def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
            self.submissions.append(fn)
            return super().submit(fn, *args, **kwargs)

    pool = _RecordingExecutor()
    vision = _bind()

    async def scenario():
        asyncio.get_running_loop().set_default_executor(pool)
        return await vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _clip_with_occurrences(120.0, (10.0, 40.0)),
        )

    try:
        proposals = asyncio.run(scenario())
    finally:
        pool.shutdown(wait=True)
    assert len(proposals) == 2
    assert pool.submissions == []
    assert not pool._threads


def test_cancellation_signal_stops_mid_clip_and_leaves_no_orphaned_child_process():
    frames = _clip_with_occurrences(600.0, (10.0,))
    cancellation = asyncio.Event()
    scanned: list[int] = []

    def counting_adapter(frame):
        scanned.append(1)
        if len(scanned) == 25:
            cancellation.set()
        return 0.0

    vision = HostedByokAdapterFactory().bind(_request(cancellation=cancellation))
    before = _child_pids()

    async def scenario():
        with pytest.raises(HostedByokCancelled):
            await vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
                frames,
                semantic_adapter=counting_adapter,
            )

    asyncio.run(scenario())
    # Stopped mid-clip, not after quietly finishing it.
    assert 25 <= len(scanned) < len(frames)
    # PID census, not "the call returned": no child process was spawned, so none can be orphaned.
    assert _child_pids() <= before


def test_task_cancellation_mid_detection_leaves_no_orphaned_child_process():
    frames = _clip_with_occurrences(600.0, (10.0,))
    scanned: list[int] = []

    def counting_adapter(frame):
        scanned.append(1)
        return 0.0

    vision = _bind()
    before = _child_pids()

    async def scenario():
        task = asyncio.ensure_future(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
                frames,
                semantic_adapter=counting_adapter,
            )
        )
        while len(scanned) < 25:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(scenario())
    assert task.cancelled()
    assert 25 <= len(scanned) < len(frames)
    assert _child_pids() <= before


def test_an_already_cancelled_signal_is_refused_before_any_frame_is_read():
    cancellation = asyncio.Event()
    scanned: list[int] = []

    def counting_adapter(frame):  # pragma: no cover - must never run
        scanned.append(1)
        return 0.0

    async def scenario():
        cancellation.set()
        vision = HostedByokAdapterFactory().bind(_request(cancellation=cancellation))
        with pytest.raises(HostedByokCancelled):
            await vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
                _still_clip(8),
                semantic_adapter=counting_adapter,
            )

    asyncio.run(scenario())
    assert scanned == []


# ---------------------------------------------------------------------------
# Zero admissions is a DIAGNOSTIC, never an empty list
# ---------------------------------------------------------------------------


def test_a_still_clip_raises_the_zero_admission_diagnostic_instead_of_returning_empty():
    vision = _bind()
    with pytest.raises(HostedByokDetectionStarved) as excinfo:
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
                _still_clip(120),
            )
        )
    diagnostic = excinfo.value
    assert diagnostic.frames == 120
    assert diagnostic.goal_ids == ("g",)
    assert diagnostic.max_salience == 0.0
    # No salience at all: the clip really was still.
    assert diagnostic.misconfigured is False
    payload = diagnostic.to_dict()
    assert payload["code"] == "hosted_byok_detection_admitted_nothing"
    assert payload["thresholds"] == {"g": Settings().edge.motion_gate_block_max}
    assert payload["salience_stat"] == "block_max"


def test_a_gate_that_sees_real_salience_but_never_admits_reports_a_misconfiguration():
    # This is the failure both shipped gate defaults actually had at some point: real signal, a
    # cutoff it can never clear, and an empty result that looked exactly like a quiet clip.
    vision = _bind()
    with pytest.raises(HostedByokDetectionStarved) as excinfo:
        asyncio.run(
            vision.detect_gated(
                [
                    HostedByokDetectionGoal(
                        goal_id="g", condition="a person enters", threshold=1.0
                    )
                ],
                _clip_with_occurrences(120.0, (10.0, 40.0, 70.0)),
            )
        )
    diagnostic = excinfo.value
    assert diagnostic.max_salience > 0.0
    assert diagnostic.mean_salience > 0.0
    assert diagnostic.misconfigured is True
    assert "never cleared its cutoff" in str(diagnostic)
    assert diagnostic.thresholds == {"g": 1.0}


def test_the_streaming_form_also_raises_rather_than_ending_silently():
    vision = _bind()

    async def scenario():
        stream = vision.stream_gated_detections(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _still_clip(40),
        )
        with pytest.raises(HostedByokDetectionStarved):
            async for _ in stream:  # pragma: no cover - the stream yields nothing
                raise AssertionError("a still clip cannot admit")

    asyncio.run(scenario())


def test_the_diagnostic_is_a_hosted_error_a_cloud_handler_can_catch_narrowly():
    assert issubclass(HostedByokDetectionStarved, percept.HostedByokError)
    # Deliberately NOT a contract error: the request was well-formed; the run admitted nothing.
    assert not issubclass(HostedByokDetectionStarved, HostedByokContractError)


# ---------------------------------------------------------------------------
# The transport contract: gate frames only
# ---------------------------------------------------------------------------


def test_a_full_resolution_frame_is_refused_at_the_boundary():
    with pytest.raises(HostedByokContractError, match="exceeds the tier-0 gate resolution"):
        HostedByokGateFrame(
            width=640,
            height=480,
            pixels=tuple([0]) * (640 * 480),
            frame_index=0,
            offset_s=0.0,
        )
    assert HOSTED_BYOK_MAX_GATE_FRAME_PIXELS == GATE_W * GATE_H == 1728


def test_gate_frames_must_be_well_formed_and_consistently_shaped():
    with pytest.raises(HostedByokContractError, match="pixel count"):
        HostedByokGateFrame(width=4, height=4, pixels=(0, 1, 2), frame_index=0, offset_s=0.0)
    with pytest.raises(HostedByokContractError, match="0..255"):
        HostedByokGateFrame(
            width=2, height=1, pixels=(0, 256), frame_index=0, offset_s=0.0
        )
    with pytest.raises(HostedByokContractError, match="offset_s must not be negative"):
        HostedByokGateFrame(width=2, height=1, pixels=(0, 1), frame_index=0, offset_s=-1.0)

    vision = _bind()
    mixed = [
        HostedByokGateFrame(width=4, height=4, pixels=_flat(0)[:16], frame_index=0, offset_s=0.0),
        HostedByokGateFrame(width=8, height=2, pixels=_flat(0)[:16], frame_index=1, offset_s=0.5),
    ]
    with pytest.raises(HostedByokContractError, match="one exact gate resolution"):
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="c")], mixed
            )
        )


def test_frames_must_be_ordered_forward_in_time():
    vision = _bind()
    frames = _still_clip(4)
    pixels = _flat(40)
    # Monotonic frame_index, but the clock runs backwards: a gate cannot run time backwards.
    backwards = [
        HostedByokGateFrame(
            width=W, height=H, pixels=pixels, frame_index=0, offset_s=5.0
        ),
        HostedByokGateFrame(
            width=W, height=H, pixels=pixels, frame_index=1, offset_s=1.0
        ),
    ]
    with pytest.raises(HostedByokContractError, match="non-decreasing offset_s"):
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="c")], backwards
            )
        )
    duplicated = [frames[0], frames[0]]
    with pytest.raises(HostedByokContractError, match="strictly increasing frame_index"):
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="c")], duplicated
            )
        )


def test_a_payload_over_the_boundary_ceiling_is_refused_before_the_scan(monkeypatch):
    monkeypatch.setattr(_hosted_byok, "_MAX_PROCESS_IPC_BYTES", GATE_FRAME_BYTES * 4)
    vision = _bind()
    with pytest.raises(HostedByokContractError, match="boundary ceiling"):
        asyncio.run(
            vision.detect_gated(
                [HostedByokDetectionGoal(goal_id="g", condition="c")], _still_clip(8)
            )
        )


def test_empty_goals_or_frames_are_refused_rather_than_admitting_nothing():
    vision = _bind()
    with pytest.raises(HostedByokContractError, match="at least one goal"):
        asyncio.run(vision.detect_gated([], _still_clip(4)))
    with pytest.raises(HostedByokContractError, match="at least one gate frame"):
        asyncio.run(
            vision.detect_gated([HostedByokDetectionGoal(goal_id="g", condition="c")], [])
        )
    with pytest.raises(HostedByokContractError, match="more than once"):
        asyncio.run(
            vision.detect_gated(
                [
                    HostedByokDetectionGoal(goal_id="g", condition="c"),
                    HostedByokDetectionGoal(goal_id="g", condition="d"),
                ],
                _still_clip(4),
            )
        )


# ---------------------------------------------------------------------------
# Tier-0 configuration is honoured, not re-invented
# ---------------------------------------------------------------------------


def test_the_deployment_tier0_default_eye_and_its_calibrated_cutoff_are_applied():
    settings = Settings()
    assert settings.edge.salience_stat == "block_max"
    vision = _bind()
    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _clip_with_occurrences(60.0, (10.0,)),
        )
    )
    # block_max sits on a higher value scale than the whole-frame mean and carries its own cutoff;
    # naming no threshold must land on the calibrated pair, never on the mean's 0.10.
    assert proposals[0].salience_stat == "block_max"
    assert proposals[0].threshold == settings.edge.motion_gate_block_max == 0.20


def test_naming_the_mean_statistic_uses_the_means_own_calibrated_cutoff():
    settings = Settings()
    vision = _bind()
    proposals = asyncio.run(
        vision.detect_gated(
            [
                HostedByokDetectionGoal(
                    goal_id="g", condition="a person enters", salience_stat="mean"
                )
            ],
            _clip_with_occurrences(60.0, (10.0,), background=0),
            settings=settings,
        )
    )
    assert proposals
    assert proposals[0].threshold == settings.edge.motion_gate == 0.10
    assert proposals[0].detector_id == "motion_score.mean"


def test_each_proposal_reports_its_own_goals_salience_statistic():
    # A call may mix statistics across goals; a proposal must never report an aggregate label.
    settings = Settings()
    vision = _bind()
    proposals = asyncio.run(
        vision.detect_gated(
            [
                HostedByokDetectionGoal(goal_id="localized", condition="a person enters"),
                HostedByokDetectionGoal(
                    goal_id="whole-frame", condition="a person enters", salience_stat="mean"
                ),
            ],
            _clip_with_occurrences(60.0, (10.0,), background=0),
        )
    )
    by_goal = {p.goal_id: p for p in proposals}
    assert by_goal["localized"].salience_stat == "block_max"
    assert by_goal["localized"].threshold == settings.edge.motion_gate_block_max
    assert by_goal["whole-frame"].salience_stat == "mean"
    assert by_goal["whole-frame"].threshold == settings.edge.motion_gate
    assert "mixed" not in {p.salience_stat for p in proposals}


def test_the_route_does_not_interpret_the_goal_condition_text():
    # Goal-semantics authoring is deliberately not this seam's job: the text rides through as
    # provenance and changes nothing about admission.
    frames = _clip_with_occurrences(60.0, (10.0,))
    vision = _bind()
    first = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")], frames
        )
    )
    second = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="the kettle is boiling")], frames
        )
    )
    assert [p.offset_s for p in first] == [p.offset_s for p in second]
    assert [p.salience for p in first] == [p.salience for p in second]


def test_judge_is_untouched_by_the_detection_route():
    # Cloud keeps calling judge(); its signature and default capability must not have moved.
    import inspect

    signature = inspect.signature(percept.HostedByokVision.judge)
    assert list(signature.parameters) == [
        "self",
        "condition",
        "frame",
        "roi",
        "exemplar",
        "corrections",
        "frames",
        "audio",
        "context",
    ]
    assert HostedByokRequest.__dataclass_fields__["capability"].default == "vision.judge"
    # The object-detection surface of the Vision protocol still answers honestly.
    vision = _bind()
    assert asyncio.run(vision.detect(object(), [])) == []
    assert asyncio.run(vision.detect_events([], [])) == []


def test_detection_runs_on_the_openrouter_route_too():
    vision = HostedByokAdapterFactory().bind(
        _request(provider="openrouter", model="qwen/qwen3-vl-32b-instruct")
    )
    proposals = asyncio.run(
        vision.detect_gated(
            [HostedByokDetectionGoal(goal_id="g", condition="a person enters")],
            _clip_with_occurrences(60.0, (10.0, 40.0)),
        )
    )
    assert [p.offset_s for p in proposals] == [10.0, 40.0]


def test_public_exports_are_cloud_usable():
    for name in (
        "HOSTED_BYOK_DEFAULT_REFRACTORY_S",
        "HOSTED_BYOK_MAX_GATE_FRAME_PIXELS",
        "HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS",
        "HostedByokDetectionGoal",
        "HostedByokDetectionProposal",
        "HostedByokDetectionStarved",
        "HostedByokGateFrame",
    ):
        assert name in percept.__all__
        assert getattr(percept, name) is getattr(_hosted_byok, name)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
