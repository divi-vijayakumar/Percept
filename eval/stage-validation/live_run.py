"""Per-stage metrics and end-to-end latency for the REAL runtime against a LIVE provider.

``stage_metrics.py`` answers "where does the pipeline lose?" with scripted, network-free judges: its
Stages 3-4 measure PLUMBING against a ground-truth oracle, and it says so.  That leaves the question
a provider claim actually turns on unanswered — what happens when the judge is a real model on a real
network, under a real deadline.  This module answers it.

It drives the same public path (``percept.run(...)``) over the same 100-case corpus, but the judge is
a live provider behind a :class:`~percept.backends.judge_pool.JudgePool`, run TWICE:

``reflex``      ``JudgePool(mode='reflex')``.  The pool stamps ``_percept_latency_mode`` on the
                adapter context, and the shared verbosity policy in ``percept/backends/anthropic.py``
                turns that into a BOUNDED ask — 320 output tokens for the event-chain shape these
                watches send, and no free-text ``evidence`` field.
``deliberate``  ``JudgePool(mode='deliberate')``.  The key is absent, so the adapter keeps the
                historical 1024-token verbose contract byte-for-byte.

Everything else is held identical between the two arms — same corpus, same WatchPlan, same tier-0
operating point, same paced replay — EXCEPT the pool deadline, and that exception is deliberate and
declared:

    reflex      deadline_s = 1.5 (ACTION_PROVIDER_DEADLINE_S), decision deadline 1.9
    deliberate  deadline_s = 30.0, decision deadline 40.0

A deliberate arm held at 1.5 s would not measure deliberate QUALITY; it would measure how often a
1024-token generation misses a 1.5 s bus, which is already known.  The question this file exists to
answer is what the latency bound COSTS in quality, so the deliberate arm is given room to finish and
its latency is reported separately — including the counterfactual "what fraction of deliberate calls
would have breached 1.5 s", which the per-call route receipts give for free.

Why the replay is PACED
-----------------------
``run(source="clip.mp4")`` decodes as fast as the machine allows — there is no realtime/pace kwarg
anywhere in the runtime.  Wall-clock latency measured over an unpaced replay is meaningless: frames
arrive in a burst, the latest-frame queue policy evicts, and provider calls pile on each other.  The
source is therefore wrapped in the repo's own real-time pacer (``eval/bench/track_perception_test_e2e.py``
``_paced_fixture_source``), with a lossless bounded queue, and a media->monotonic anchor is recorded
per case so an action ONSET on the source clock can be converted into a wall instant.  Without that
anchor there is no such thing as "action-onset -> Incident".

Metric definitions are NOT re-chosen here
-----------------------------------------
Every per-stage definition — the two-population recall rule, the unit-time tiling, the
suppressed-with-receipt vs dropped-without-receipt distinction, the interval algebra, the pooling
rule — is imported from ``stage_metrics`` and used unmodified.  This module supplies a different
ARM (live provider instead of scripted oracle) and a latency instrument; it does not supply a second
opinion about what recall means.

What is measured per mode
-------------------------
1. Per-stage recall (stage-local AND cumulative) / precision / economy / drop rate.
2. End-to-end latency decomposed action-onset -> ingest -> admission -> judge dispatch -> verdict ->
   decision -> Incident, p50/p95/max, against the < 2 s product gate.
3. Provider accounting: logical calls, physical attempts, terminal outcome classes from the route
   receipts (``deadline``/``capacity``/``no_route``/``provider_error``/``malformed``/``cancelled``),
   tokens, estimated cost.
4. Quality: occurrence precision/recall and timely recall at 2 s, per ``goal_kind``.

Run it::

    set -a; . .env; set +a
    PERCEPT_VISION_BACKEND=gemini PERCEPT_VISION_MODEL=gemini-3.1-flash-lite \
      <venv>/bin/python eval/stage-validation/live_run.py --limit 10 --concurrency 4

Writes ``results/live_run.json`` and ``results/live_run.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------------------------
# metric code is REUSED, never reimplemented -- see the module docstring
# ---------------------------------------------------------------------------------------------
import stage_metrics as SM  # noqa: E402

from percept import Percept  # noqa: E402
from percept._build import build_info  # noqa: E402
from percept.backends.judge_pool import (  # noqa: E402
    LATENCY_MODE_CONTEXT_KEY,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.config import load_settings  # noqa: E402
from percept.contracts.watchplan import WatchPlan  # noqa: E402
from percept.incidents import incident_judgment  # noqa: E402
from percept.resolver import ConfigResolver  # noqa: E402
from percept.runtime import (  # noqa: E402
    NativeVideoVision,
    RuntimeStats,
    TemporalImageVision,
    prepare_temporal_images,
)
from percept.runtime.source import open_source  # noqa: E402
from percept.runtime.worker import run as worker_run  # noqa: E402
from percept.sensorplan import sensor_plan_for  # noqa: E402
from percept.source_profiles_default import PHYSICAL_SCENE  # noqa: E402
from percept.watchplan_apply import apply_watch_plan  # noqa: E402

# ---------------------------------------------------------------------------------------------
# pinned constants -- cited from the runtime/bench, or DECLARED here before the run
# ---------------------------------------------------------------------------------------------

#: The action lane's split budget, copied verbatim from the qualification bench
#: (``packages/percept-vision/eval/bench/track_perception_test_e2e.py:91-92``).  These are the
#: numbers the product gate is written against; they are not re-chosen here.
#: OVERRIDABLE so the deadline itself can be a DECLARED ARM rather than a fixed assumption.
#: The defaults are the qualification bench's numbers, unchanged, so a run that sets neither
#: variable reproduces every earlier result byte-for-byte.
#:
#: Why an arm at all: measured on this corpus, the reflex deadline sits BELOW the median call.
#: The uncensored distribution (same corpus, same evidence shape) is p25 1.424 / p50 1.653 /
#: p95 2.283 / p99 3.129 s, so 66.5% of calls cannot physically finish inside 1.5 s. A reflex run
#: at 1.5 s therefore measures the timer, not the pipeline: it completed 1,058 of 7,193 calls
#: (14.7%) and its latency "distribution" tops out at 1.4974 s -- the deadline's shadow, not a
#: measurement. Raising the ceiling is not tuning a metric until it passes; it changes WHICH
#: question the run answers, and the latency cost must be reported next to any recall gain.
#:
#: NOTE this is orthogonal to ``mode``: the pool still stamps ``_percept_latency_mode='reflex'``,
#: so the bounded 320-token reply, the absent free-text ``evidence`` field and (on a build that has
#: it) the 3-frame evidence bound all stay in force. A raised-deadline reflex arm is NOT the
#: deliberate arm, which additionally relaxes the output contract to 1024 verbose tokens.
ACTION_PROVIDER_DEADLINE_S = float(os.environ.get("PERCEPT_SV_PROVIDER_DEADLINE_S", "1.5"))
ACTION_DECISION_DEADLINE_S = float(os.environ.get("PERCEPT_SV_DECISION_DEADLINE_S", "1.9"))

#: The user-visible product gate: action onset -> Incident.
ACTION_TO_INCIDENT_GATE_S = 2.0

#: The deliberate arm's budget.  Generous BY DESIGN so deliberate quality is measured un-truncated
#: (see the module docstring).  A deliberate arm at 1.5 s measures the bus, not the model.
DELIBERATE_PROVIDER_DEADLINE_S = 30.0
DELIBERATE_DECISION_DEADLINE_S = 40.0

#: Evidence shape held identical across arms -- the event-chain occurrence watch, which is what a
#: real occurrence watch sends.  Copied from the bench (lines 65-80 of the same file).
ACTION_BRIEF_WINDOW_S = 1.0
ACTION_TEMPORAL_WINDOW_S = 2.0
ACTION_TEMPORAL_FRAMES = int(os.environ.get("PERCEPT_SV_FRAMES", "5"))
# Overridable so the frame-count/latency trade can be measured without editing the harness: fewer
# frames cut provider latency (5 frames ~1.19s vs 3 ~0.73-0.91s measured) and therefore cut deadline
# expiries, but cost temporal context per judgment. Which effect dominates recall is empirical.
ACTION_TEMPORAL_SAMPLING = "uniform"

#: JPEG long-edge the temporal frames are downscaled to before upload.  ``0`` keeps the runtime
#: default (``prepare_temporal_images`` max_dimension=768) untouched.
#:
#: MEASURED 2026-07-27 and kept only as an instrument: on Gemini this knob does NOT reduce the
#: prompt.  The provider bills a FIXED 1100 image tokens per frame whether the JPEG is 768x432 or
#: 256x144 (3-frame prompt = 3434 tokens at every setting from 768 down to 256), so shrinking the
#: pixels shrinks the UPLOAD and nothing else, and TTFT did not move across a 5.7x payload-byte
#: range.  Lowering it therefore buys no latency and no cost while discarding detail -- strictly
#: dominated.  It stays overridable so the null is re-testable on another provider, not because
#: turning it down is expected to help.
ACTION_MAX_DIMENSION = int(os.environ.get("PERCEPT_SV_MAX_DIM", "0"))

#: Gemini 3's own vision-token budget, which IS the knob ``PERCEPT_SV_MAX_DIM`` is mistaken for.
#: On the identical 3-frame payload: HIGH (the SDK default) = 3434 prompt tokens, MEDIUM = 1715,
#: LOW = 926.  Empty string leaves the provider default in place.
ACTION_MEDIA_RESOLUTION = os.environ.get("PERCEPT_SV_MEDIA_RESOLUTION", "").strip()

#: Path to a FROZEN ``percept.conductor`` authoring artifact.  When set, each case's goal text is
#: looked up in the pin and the authored ``Goal.state.*`` paths are merged UNDER this harness's own
#: explicit paths, so the goal arms on the primitive the conductor selected rather than on the fixed
#: occurrence watch.  A frozen artifact raises on a miss, so no provider call is ever made here and
#: the selected primitives are exactly the pinned ones.  Unset = the fixed watch, unchanged.
ACTION_CONDUCTOR_PIN = os.environ.get("PERCEPT_SV_CONDUCTOR", "").strip()

_CAPTURE_HEADROOM_FPS = 8.0
_FIXTURE_QUEUE_SIZE = 8
_LEGACY_CAPTURE_FLOOR_FPS = 2.0

MODES = ("reflex", "deliberate")

#: Receipt outcomes that mean the pool produced no usable provider answer.  NOTE: unlike the
#: qualification bench, this harness does NOT pass these as ``terminal_outcomes`` -- a measurement
#: run must COUNT a deadline, not raise on it.
FAILED_OUTCOMES = frozenset({"provider_error", "malformed", "deadline", "capacity", "no_route"})

#: Prices are an INPUT to this report, never a measurement.  Tokens are measured exactly; cost is
#: tokens x a declared price.  Overridable from the command line so the number can be re-derived
#: without re-running the corpus.  ``None`` means "not priced" and the report says so rather than
#: inventing a figure.
DEFAULT_PRICES: dict[str, dict[str, Any]] = {}


def capture_fps(cadence: dict) -> float:
    """Decode rate: dense enough that the plan's sensitive cadence isn't starved by a coarse floor.

    Copied from ``eval/bench/track_perception_test_e2e.py:113`` so both harnesses decode identically.
    """
    want = cadence.get("max_fps") or cadence.get("base_fps") or 0.0
    return max(_LEGACY_CAPTURE_FLOOR_FPS, min(_CAPTURE_HEADROOM_FPS, float(want)))


@functools.lru_cache(maxsize=1)
def _authoring_pin() -> Any:
    """Load the FROZEN conductor artifact once.  Imported lazily so a build without
    ``percept.conductor`` (the pre-launch baseline) never touches this path at all."""
    from percept.conductor import AuthoringCache

    return AuthoringCache.load(ACTION_CONDUCTOR_PIN, frozen=True)


async def authored_for(goal_text: str) -> Any:
    """The pinned :class:`~percept.conductor.AuthoredPrimitive` for one goal, or ``None``.

    A frozen cache raises :class:`AuthoringCacheMiss` rather than authoring live, so a run bound to
    the pin provably uses the recorded decisions and spends nothing.  The miss is allowed to
    propagate: silently falling back to the fixed watch would make the conductor arm untestable.
    """
    if not ACTION_CONDUCTOR_PIN:
        return None
    from percept.conductor import PlanConductor

    return await PlanConductor(cache=_authoring_pin()).author(goal_text)


def watch_plan_for(goal_id: str, condition: str, authored: Any = None) -> WatchPlan:
    """The fixed, corpus-independent occurrence WatchPlan every case is armed with.

    Identical in shape to the qualification bench's action plan.  Nothing in it is derived from the
    case's annotation — no window, no duration, no positive count — so no scoring data leaks into
    runtime admission.

    When ``authored`` is supplied (the conductor arm), the conductor's ``Goal.state.*`` paths form
    the BASE and the harness's own sense/perception paths are layered on top, so the two arms differ
    in the selected PRIMITIVE and in nothing else about the evidence shape.  The conductor is
    structurally barred from writing outside ``Goal.state.`` , so it can never reach the
    ``WatchProfile.salience_stat`` held constant across both arms.
    """
    explicit: dict[str, Any] = dict(getattr(authored, "explicit", None) or {})
    explicit.update(
        {
            "Goal.state.sense.brief_window_s": ACTION_BRIEF_WINDOW_S,
            "Goal.state.perception.temporal": "strip",
            "Goal.state.perception.n_frames": ACTION_TEMPORAL_FRAMES,
            "Goal.state.perception.sampling": ACTION_TEMPORAL_SAMPLING,
            "Goal.state.perception.window_s": ACTION_TEMPORAL_WINDOW_S,
            "WatchProfile.salience_stat": "block_max",
        }
    )
    return WatchPlan(
        id=goal_id,
        condition=condition,
        sense="video",
        rearm="each_event",
        explicit=explicit,
    )


# ---------------------------------------------------------------------------------------------
# the live provider, wrapped so physical attempts are counted at the ADAPTER boundary
# ---------------------------------------------------------------------------------------------


def install_media_resolution_override(value: str) -> str | None:
    """Force a ``media_resolution`` on every Gemini request, WITHOUT touching ``percept/``.

    ``percept.backends.gemini`` builds its ``GenerateContentConfig`` inline and re-imports
    ``google.genai.types`` per call (``_require_genai``), so rebinding the SDK's attribute in this
    process is enough to reach it — and it reaches only this process, which is the point: the
    runtime default must not move until a measurement says it should.

    Subclass rather than a factory function so ``isinstance`` checks inside the SDK still hold.
    """
    if not value:
        return None
    from google.genai import types as gtypes

    base = gtypes.GenerateContentConfig
    if getattr(base, "_sv_media_resolution", None):  # idempotent under --resume re-entry
        return str(base._sv_media_resolution)

    class _PinnedMediaResolution(base):  # type: ignore[misc, valid-type]
        _sv_media_resolution = value

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("media_resolution", value)
            super().__init__(**kwargs)

    gtypes.GenerateContentConfig = _PinnedMediaResolution
    return value


class RecordingVision:
    """Counts and times every PHYSICAL call that reaches the provider adapter.

    The pool's receipts count ROUTE attempts; this counts what actually crossed into the backend.
    With one route at ``max_inflight=1`` the two should agree, and a disagreement is reported rather
    than reconciled away — it would mean the adapter retried underneath the route.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.SUPPORTS_NATIVE_VIDEO_TRANSPORT = bool(
            getattr(inner, "SUPPORTS_NATIVE_VIDEO_TRANSPORT", False)
        )
        self.PARSES_CLAIM_CHANNEL = bool(getattr(inner, "PARSES_CLAIM_CHANNEL", False))
        self.calls = 0
        self.errors = 0
        self.cancelled = 0
        #: Elapsed for calls that RETURNED. A cancelled call's elapsed is a lower bound on what the
        #: provider would have taken, so the two are never pooled into one distribution.
        self.latencies_completed: list[float] = []
        self.latencies_censored: list[float] = []
        self.usages: list[dict] = []
        self.reflex_stamped = 0
        self.evidence_nonempty = 0
        self.input_bytes = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.error_types: Counter = Counter()

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any):
        self.calls += 1
        context = dict(kwargs.get("context") or {})
        if context.get(LATENCY_MODE_CONTEXT_KEY) == "reflex":
            self.reflex_stamped += 1
        payload = frame[1] if isinstance(frame, tuple) and len(frame) == 2 else frame
        if isinstance(payload, (bytes, bytearray)):
            self.input_bytes += len(payload)
        for f in kwargs.get("frames") or []:
            fp = f[1] if isinstance(f, tuple) and len(f) == 2 else f
            if isinstance(fp, (bytes, bytearray)):
                self.input_bytes += len(fp)
        t0 = time.monotonic()
        try:
            verdict = await self.inner.judge(condition, frame, **kwargs)
        except asyncio.CancelledError:
            # The pool cancels a hedge or a deadline-breaching call. This is the ONLY place the
            # true provider elapsed for such a call is observable, and `except Exception` would
            # miss it entirely (CancelledError is a BaseException) -- which is how the first
            # smoke run reported "0 provider calls" while burning 18 of them.
            self.cancelled += 1
            self.latencies_censored.append(time.monotonic() - t0)
            raise
        except Exception as exc:
            self.errors += 1
            self.error_types[type(exc).__name__] += 1
            self.latencies_censored.append(time.monotonic() - t0)
            raise
        self.latencies_completed.append(time.monotonic() - t0)
        usage = dict(getattr(verdict, "usage", None) or {})
        if usage:
            self.usages.append(usage)
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
        if str(getattr(verdict, "evidence", "") or "").strip():
            self.evidence_nonempty += 1
        return verdict


# ---------------------------------------------------------------------------------------------
# real-time pacing -- the only thing that makes a wall-clock latency number mean anything
# ---------------------------------------------------------------------------------------------


@dataclass
class ReplayAnchor:
    """The one bridge between the source clock and the wall clock.

    ``media_ts`` is the first decoded frame's source timestamp; ``monotonic_ts`` is the instant it
    was released into the runtime.  Every "action onset in wall time" in this report is
    ``monotonic_ts + (onset_media - media_ts)`` and nothing else.
    """

    media_ts: float
    monotonic_ts: float


@dataclass
class CaseChannels:
    frames_yielded: int = 0
    anchor: ReplayAnchor | None = None
    pace_lag_s: list[float] = field(default_factory=list)


async def paced_source(path: str, *, fps: float, ch: CaseChannels, dropped: list[int]):
    """Replay a finite file at its media cadence, preserving envelopes.

    Structurally the bench's ``_paced_fixture_source``: a LOSSLESS bounded queue
    (``queue_policy='block'``) so the measurement cannot silently lose coverage, plus a sleep to the
    frame's media instant.  The per-frame pacing LAG is recorded — if the runtime ever falls behind
    the source clock, every downstream latency is understated and the report must say so.
    """
    started_wall: float | None = None
    first_media: float | None = None
    async for envelope in open_source(
        path,
        fps=fps,
        maxsize=_FIXTURE_QUEUE_SIZE,
        queue_policy="block",
        timebase="media",
        dropped=dropped,
    ):
        if envelope.source_time is None:
            raise ValueError("paced replay requires decoder source timestamps")
        media_t = float(envelope.source_time)
        if first_media is None:
            first_media = media_t
            started_wall = time.monotonic()
            ch.anchor = ReplayAnchor(media_ts=media_t, monotonic_ts=started_wall)
        assert started_wall is not None
        target = started_wall + (media_t - first_media)
        remaining = target - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        else:
            ch.pace_lag_s.append(-remaining)
        ch.frames_yielded += 1
        yield envelope


# ---------------------------------------------------------------------------------------------
# one case, one mode
# ---------------------------------------------------------------------------------------------


def _goal_id(case_id: str) -> str:
    return "lr-" + re.sub(r"[^A-Za-z0-9_-]+", "-", case_id).strip("-")


def _pct(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ys[int(k)])
    return float(ys[lo] + (ys[hi] - ys[lo]) * (k - lo))


def dist(xs: Sequence[float]) -> dict[str, Any]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None, "min": None}
    return {
        "n": len(xs),
        "min": round(min(xs), 4),
        "p50": round(_pct(xs, 0.50) or 0.0, 4),
        "p95": round(_pct(xs, 0.95) or 0.0, 4),
        "max": round(max(xs), 4),
        "mean": round(statistics.fmean(xs), 4),
    }


async def run_case(case: dict, mode: str, *, backend: str, transport: str = "images") -> dict[str, Any]:
    """Drive ONE corpus case through ``percept.run`` with a live judge in ``mode``."""
    case_id = case["case_id"]
    path = case["video_path"]
    out: dict[str, Any] = {
        "case_id": case_id,
        "dataset": case["dataset"],
        "goal_kind": case["goal_kind"],
        "mode": mode,
        "backend": backend,
        "notes": [],
        "error": None,
        "measurable": False,
    }

    src_fps, n_native = SM.probe_media(path)
    out["native_fps"] = src_fps
    out["native_frames"] = n_native

    settings = load_settings()
    if backend == "gemini":
        from percept.backends.gemini import GeminiVision

        inner = GeminiVision(settings)
        model = getattr(inner, "_model", None) or getattr(inner, "model", None)
    elif backend == "openrouter":
        from percept.backends.openrouter import OpenRouterVision

        inner = OpenRouterVision(settings)
        model = getattr(inner, "_model", None) or getattr(inner, "model", None)
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown backend {backend!r}")
    out["model"] = model

    provider = RecordingVision(inner)
    # `auto` reproduces the qualification bench's route selection -- native video whenever the
    # adapter advertises it. That selection is measured here, not assumed: see the transport
    # comparison in the report.
    native_transport = (
        provider.SUPPORTS_NATIVE_VIDEO_TRANSPORT if transport == "auto" else transport == "native"
    )
    if native_transport and not provider.SUPPORTS_NATIVE_VIDEO_TRANSPORT:
        out["error"] = f"backend {backend!r} does not support native video transport"
        return out
    temporal = (
        NativeVideoVision(
            provider,
            fps=_CAPTURE_HEADROOM_FPS,
            max_frames=ACTION_TEMPORAL_FRAMES,
            require_native_transport=True,
        )
        if native_transport
        else TemporalImageVision(
            provider,
            max_frames=min(ACTION_TEMPORAL_FRAMES, 5),
            # The transport's own injection point, so the resolution sweep never edits `percept/`.
            preprocessor=(
                functools.partial(prepare_temporal_images, max_dimension=ACTION_MAX_DIMENSION)
                if ACTION_MAX_DIMENSION
                else prepare_temporal_images
            ),
        )
    )
    route_name = "native-video-primary" if native_transport else "timestamped-images-primary"
    out["transport"] = "native_video" if native_transport else "ordered_timestamped_images"

    provider_deadline = (
        ACTION_PROVIDER_DEADLINE_S if mode == "reflex" else DELIBERATE_PROVIDER_DEADLINE_S
    )
    decision_deadline = (
        ACTION_DECISION_DEADLINE_S if mode == "reflex" else DELIBERATE_DECISION_DEADLINE_S
    )
    out["provider_deadline_s"] = provider_deadline
    out["decision_deadline_s"] = decision_deadline

    receipts: list[dict] = []

    def on_receipt(r: Any) -> None:
        receipts.append(
            {
                "route": r.route,
                "evidence_shape": r.evidence_shape,
                "mode": r.mode,
                "outcome": r.outcome,
                "latency_s": round(float(r.latency_s), 6),
                "deadline_s": float(r.deadline_s),
                "confidence": r.confidence,
                "exception_type": r.exception_type,
            }
        )

    pool = JudgePool(
        [
            JudgeRoute(
                route_name,
                temporal,
                JudgeCapability(
                    # `video_clip` is the only shape a plain occurrence watch ever asks for, so the
                    # extra two are INERT for the fixed-watch arm (verified: every baseline receipt
                    # carries `video_clip`). They exist because an authored `transition` primitive
                    # asks for `ordered_images`/`single_image`, and a route that cannot serve the
                    # shape returns `no_route` — zero provider calls and zero incidents, which would
                    # score the conductor arm as a total miss on every transition goal for a reason
                    # that is this harness's routing table, not the runtime's behaviour.
                    evidence_shapes=frozenset({"video_clip", "ordered_images", "single_image"}),
                    latency_slo_s=provider_deadline,
                    modes=frozenset({mode}),
                    max_inflight=1,
                ),
            )
        ],
        mode=mode,
        deadline_s=provider_deadline,
        max_parallel=1,
        on_receipt=on_receipt,
        # DELIBERATELY EMPTY: a measurement run counts a deadline, it does not raise on one.
        terminal_outcomes=(),
    )

    core = Percept.create(
        vision=pool,
        specialist="scripted",
        conductor="scripted-conductor",
        stt="fake",
        tts="fake",
        settings=settings,
        discover_plugins=False,
    )
    goal_id = _goal_id(case_id)
    # The conductor arm selects the PRIMITIVE from the goal's own words (replayed from the frozen
    # pin, zero provider calls); the baseline arm keeps the corpus-declared fixed watch. Recorded
    # per case either way so the report can name which goals were re-routed and to what.
    authored = await authored_for(case["goal_text"])
    out["authored_type"] = getattr(authored, "behavior_type", None)
    out["authored_origin"] = getattr(authored, "origin", None)
    out["authored_explicit"] = dict(getattr(authored, "explicit", None) or {})
    receipt = apply_watch_plan(
        core, watch_plan_for(goal_id, case["goal_text"], authored), resolver=ConfigResolver(),
        source_profile=PHYSICAL_SCENE,
    )
    if receipt.status != "applied" or receipt.watch_ids != (goal_id,):
        out["error"] = (
            f"WatchPlan did not arm: status={receipt.status} "
            f"warnings={[w.code for w in receipt.warnings]!r}"
        )
        return out
    goal = core.goals.get(goal_id)
    cadence = sensor_plan_for(goal).cadence
    fps = capture_fps(cadence)
    out["capture_fps"] = fps
    out["watch_warnings"] = [w.code for w in receipt.warnings]

    stats = RuntimeStats()
    ch = CaseChannels()
    dropped = [0]
    trace: list[tuple[str, float | None, dict]] = []
    health: list[dict] = []
    incidents: list[dict] = []
    asks: list[dict] = []

    def on_trace(ev: Any) -> None:
        trace.append((ev.stage, ev.t, dict(ev.data)))

    def on_health(ev: Any) -> None:
        health.append({"kind": ev.kind, "ts": ev.ts, "data": dict(ev.data or {})})

    def on_ask(fire: Any) -> None:
        asks.append({"observed_at": time.monotonic(), "goal_id": getattr(fire, "goal_id", None)})

    wall0 = time.monotonic()
    try:
        runtime = worker_run(
            core,
            source=paced_source(path, fps=fps, ch=ch, dropped=dropped),
            fps=fps,
            stats=stats,
            gate_source="camera",
            judge_dispatch="async",
            judge_stall_after_s=decision_deadline,
            judge_decision_deadline_s=decision_deadline,
            timebase="media",
            on_trace=on_trace,
            on_health=on_health,
            on_ask=on_ask,
            sink_errors="ignore",
        )
        try:
            async for inc in runtime:
                observed = time.monotonic()
                judgment = incident_judgment(inc)
                timing = getattr(judgment, "timing", None)
                incidents.append(
                    {
                        "id": getattr(inc, "id", None),
                        "ts": float(getattr(inc, "ts", 0.0) or 0.0),
                        "observed_monotonic": observed,
                        "confidence": float(getattr(getattr(inc, "verdict", None), "confidence", 0.0) or 0.0),
                        "evidence_len": len(str(getattr(getattr(inc, "verdict", None), "evidence", "") or "")),
                        "timing": None
                        if timing is None
                        else {
                            "ingest_observed_ts": timing.ingest_observed_ts,
                            "admitted_ts": timing.admitted_ts,
                            "judge_started_ts": timing.judge_started_ts,
                            "judge_completed_ts": timing.judge_completed_ts,
                            "decision_emitted_ts": timing.decision_emitted_ts,
                            "source_media_ts": timing.source_media_ts,
                        },
                    }
                )
        finally:
            await runtime.aclose()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["wall_s"] = round(time.monotonic() - wall0, 2)

    # ------------------------------------------------------------------ channels -> stages ----
    ed = [(t, d) for (stage, t, d) in trace if stage == "edge.decision" and t is not None]
    out["frames_yielded"] = ch.frames_yielded
    out["decoder_dropped"] = dropped[0]
    out["pace_lag"] = dist(ch.pace_lag_s)
    out["anchor"] = None if ch.anchor is None else {"media_ts": ch.anchor.media_ts,
                                                    "monotonic_ts": ch.anchor.monotonic_ts}
    if not ed:
        out["notes"].append("no edge.decision trace -- case not measurable")
        return out

    ts = [t for t, _ in ed]
    esc = [bool(d.get("escalate") or d.get("coast_admit")) for _, d in ed]
    media_end = max(ts[-1], float(case.get("duration_s") or 0.0),
                    (n_native / src_fps) if src_fps else 0.0)
    gt, gt_events, gt_kind, gt_occ = SM.ground_truth(case["positives"], media_end)
    tiles = SM.tile(ts, media_end)
    received_all = SM.union(tiles)
    admitted_time = SM.spans_of(tiles, esc)
    index = {round(t, 6): i for i, t in enumerate(ts)}

    def spans_for(stage: str, predicate=None) -> list[tuple[float, float]]:
        keep = [False] * len(ts)
        for st, t, d in trace:
            if st != stage or t is None:
                continue
            if predicate is not None and not predicate(d):
                continue
            i = index.get(round(float(t), 6))
            if i is not None:
                keep[i] = True
        return SM.spans_of(tiles, keep)

    def n_stage(stage: str) -> int:
        return sum(1 for st, _t, _d in trace if st == stage)

    ended = next((h["data"] for h in health if h["kind"] == "runtime_ended"), {})
    cov = (ended.get("source_coverage") or {}).get("video", {})

    out.update(
        gt_kind=gt_kind, gt_occurrences=gt_occ, gt_events=len(gt_events),
        gt_time_s=SM.measure(gt), media_end_s=media_end, measurable=True,
    )

    # ---- STAGE 0 -- ingest -------------------------------------------------------------------
    step = max(1, round(n_native / len(ed))) if ed else 1
    selected = math.ceil(n_native / step) if step else n_native
    emitted = int(stats.frames)
    frames_dropped = int(stats.frames_dropped)
    excluded = int(cov.get("excluded", 0))
    unaccounted0 = max(0, selected - emitted - frames_dropped - excluded)
    s0 = SM.stage_metrics(
        passed_time=received_all, received_time=[(0.0, media_end)], gt=gt, gt_events=gt_events,
        units_received=n_native, units_passed=emitted, units_dropped_no_receipt=unaccounted0,
    )
    s0.update(deliberate_decimation_step=step, units_selected_after_decimation=selected,
              frames_dropped_with_receipt=frames_dropped,
              envelopes_excluded_with_receipt=excluded,
              decoder_dropped_no_receipt=dropped[0],
              missing_sequences=int(cov.get("missing_sequences", 0)),
              discontinuities=int(cov.get("discontinuities", 0)))

    # ---- STAGE 1 -- admission ----------------------------------------------------------------
    n_adm = sum(esc)
    s1 = SM.stage_metrics(
        passed_time=admitted_time, received_time=received_all, gt=gt, gt_events=gt_events,
        units_received=len(ed), units_passed=n_adm, units_dropped_no_receipt=0,
    )
    s1.update(reduction=1.0 - (n_adm / len(ed)) if ed else None,
              admission_by_cause=dict(stats.admission_by_cause),
              escalations=int(stats.escalations), signal_admits=int(stats.signal_admits),
              coast_admits=int(stats.coast_admits), fixed_admits=int(stats.fixed_admits),
              suppressed_with_receipt=len(ed) - n_adm)

    # ---- STAGE 2 -- local perception + judge scheduling -----------------------------------------
    # A live judge makes this boundary REAL in a way the scripted-oracle arm cannot: with
    # `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while
    # a ~1 s provider call is out, every newly admitted frame but the freshest is SUPERSEDED. That
    # is designed backpressure and it carries a receipt on both channels -- a named `RuntimeStats`
    # counter and a `judge_superseded` health event -- so it is economy, NOT drop. Counting it as a
    # drop (which this harness did on its first pass) invents an 0.86 drop rate out of the runtime
    # working exactly as specified.
    s2_time = spans_for("source.frame")
    n_adm2 = int(stats.escalations)
    n_disp = int(stats.perceive_dispatches)
    scheduler_receipted = (int(stats.judge_superseded) + int(stats.judge_revision_dropped)
                           + int(stats.judge_deadline_dropped)
                           + int(stats.judge_source_discontinuity_dropped))
    s2 = SM.stage_metrics(
        passed_time=s2_time, received_time=admitted_time, gt=gt, gt_events=gt_events,
        units_received=n_adm2, units_passed=n_disp,
        units_dropped_no_receipt=max(0, n_adm2 - n_disp - scheduler_receipted),
    )
    s2.update(source_frame_units=n_stage("source.frame"),
              extractor_enriched=n_stage("extractor.enriched"),
              superseded_with_receipt=int(stats.judge_superseded),
              scheduler_dropped_with_receipt=scheduler_receipted,
              backpressure_events=sum(1 for h in health
                                      if h["kind"] == "judge_scheduler_backpressure"),
              identity_passthrough=n_disp == n_adm2)

    # ---- STAGE 3 -- judge dispatch (LIVE) ------------------------------------------------------
    s3_time = spans_for("judge.finished")
    n_started = n_stage("judge.started")
    n_finished = n_stage("judge.finished")
    n_degraded = n_stage("judge.degraded")
    judge_receipted = int(stats.judge_deadline_expired)
    outcome_counts = Counter(r["outcome"] for r in receipts)
    failed = sum(outcome_counts[o] for o in FAILED_OUTCOMES)
    completed_outcomes = {"decisive", "uncertain"}
    s3 = SM.stage_metrics(
        passed_time=s3_time, received_time=s2_time, gt=gt, gt_events=gt_events,
        units_received=n_disp, units_passed=n_finished,
        units_dropped_no_receipt=max(0, n_disp - n_finished - n_degraded - judge_receipted),
    )
    usage_rows = [d for st, _t, d in trace if st == "judge.usage"]
    fn = dict(core.metrics.snapshot())
    s3.update(
        logical_calls=n_started,
        physical_attempts=provider.calls,
        route_attempts=sum(1 for r in receipts if r["route"] is not None),
        judge_calls_batches=int(stats.judge_calls),
        finished=n_finished, degraded=n_degraded,
        dropped_with_receipt=judge_receipted,
        adapter_errors=provider.errors,
        adapter_cancelled=provider.cancelled,
        adapter_error_types=dict(provider.error_types),
        reflex_context_stamped=provider.reflex_stamped,
        verdicts_with_evidence_text=provider.evidence_nonempty,
        # Completed vs censored are never pooled: a cancelled call's elapsed is a LOWER BOUND on
        # what the provider would have taken, not a measurement of it.
        provider_latency=dist(provider.latencies_completed),
        provider_latency_censored=dist(provider.latencies_censored),
        # The runtime's OWN timing of the pool call, as an independent cross-check.
        runtime_judge_latency_mean_s=(
            (fn.get("judge.latency.total_s") or 0.0) / fn["judge.latency.count"]
            if fn.get("judge.latency.count") else None
        ),
        route_latency=dist([r["latency_s"] for r in receipts
                            if r["route"] is not None and r["outcome"] in completed_outcomes]),
        terminal_outcomes=dict(sorted(outcome_counts.items())),
        failed_route_calls=failed,
        input_tokens=provider.input_tokens,
        output_tokens=provider.output_tokens,
        usage_rows=len(usage_rows),
        input_bytes=provider.input_bytes,
        runtime_stats={
            "judge_superseded": int(stats.judge_superseded),
            "judge_revision_dropped": int(stats.judge_revision_dropped),
            "judge_deadline_dropped": int(stats.judge_deadline_dropped),
            "judge_deadline_expired": int(stats.judge_deadline_expired),
        },
    )

    # ---- STAGE 4 -- cognition + disposition ----------------------------------------------------
    s4_time = spans_for("gate.classified", lambda d: bool(d.get("satisfied")))
    n_gate = n_stage("gate.classified")
    n_true = len({round(float(t), 6) for st, t, d in trace
                  if st == "gate.classified" and d.get("satisfied") and t is not None})
    inc_ts = [i["ts"] for i in incidents]
    matched = sum(1 for a, b in gt_events if any(a <= t < b for t in inc_ts))
    s4 = SM.stage_metrics(
        passed_time=s4_time, received_time=s3_time, gt=gt, gt_events=gt_events,
        units_received=n_finished, units_passed=n_true,
        units_dropped_no_receipt=max(0, n_finished - n_gate),
    )
    s4.update(gate_classified=n_gate, gate_true_units=n_true, incidents=len(incidents),
              gt_occurrences=len(gt_events), occurrences_matched=matched,
              occurrence_recall=(matched / len(gt_events)) if gt_events else None,
              incidents_outside_gt=sum(1 for t in inc_ts if not any(a <= t < b for a, b in gt)),
              asks=int(stats.asks), fires=int(stats.fires),
              gate_actions=dict(Counter(str(d.get("action")) for st, _t, d in trace
                                        if st == "gate.classified")))

    # ---- LATENCY --------------------------------------------------------------------------------
    # Every wall figure below is anchored through ONE conversion: onset_wall = anchor.monotonic_ts +
    # (onset_media - anchor.media_ts).  No other clock crossing happens anywhere in this file.
    lat_rows: list[dict] = []
    if ch.anchor is not None:
        a = ch.anchor
        for inc in incidents:
            tm = inc["timing"]
            occ = next(((lo, hi) for lo, hi in gt_events if lo <= inc["ts"] < hi), None)
            row: dict[str, Any] = {
                "case_id": case_id, "goal_kind": case["goal_kind"], "dataset": case["dataset"],
                "incident_media_ts": inc["ts"], "matched_occurrence": occ,
                "confidence": inc["confidence"], "evidence_len": inc["evidence_len"],
            }
            if occ is not None:
                onset_wall = a.monotonic_ts + (occ[0] - a.media_ts)
                row["onset_to_incident_s"] = inc["observed_monotonic"] - onset_wall
                if tm is not None:
                    row["onset_to_ingest_s"] = tm["ingest_observed_ts"] - onset_wall
                    row["onset_to_admission_s"] = tm["admitted_ts"] - onset_wall
            if tm is not None:
                row.update(
                    ingest_to_admission_s=tm["admitted_ts"] - tm["ingest_observed_ts"],
                    admission_to_dispatch_s=tm["judge_started_ts"] - tm["admitted_ts"],
                    dispatch_to_verdict_s=tm["judge_completed_ts"] - tm["judge_started_ts"],
                    verdict_to_decision_s=tm["decision_emitted_ts"] - tm["judge_completed_ts"],
                    decision_to_incident_s=inc["observed_monotonic"] - tm["decision_emitted_ts"],
                    ingest_to_incident_s=inc["observed_monotonic"] - tm["ingest_observed_ts"],
                    admission_to_incident_s=inc["observed_monotonic"] - tm["admitted_ts"],
                )
            lat_rows.append(row)
    else:
        out["notes"].append("no replay anchor -- latency not measurable for this case")

    out["latency_rows"] = lat_rows
    out["receipts"] = receipts
    out["_provider_latency_raw"] = {
        "completed": [round(x, 4) for x in provider.latencies_completed],
        "censored": [round(x, 4) for x in provider.latencies_censored],
    }
    # The funnel's own accounting, exercised on a workload that actually produces ASKs -- which the
    # scripted-oracle suite explicitly could NOT do (its judge always answered with high confidence,
    # so the gate resolved to fire/silent and almost never to ASK). A disagreement here is a
    # live-run finding, not noise.
    _fn = dict(core.metrics.snapshot())
    _g = lambda k: int(_fn.get(k) or 0)  # noqa: E731
    out["funnel_balance"] = {
        "judged": _g("funnel.judged"),
        "fired": _g("funnel.fired"),
        "refused": _g("funnel.refused"),
        "ask": _g("gate.ask"),
        # ONE refusal per ASK is the intended accounting. Two is the double-increment.
        "refused_equals_ask": _g("funnel.refused") == _g("gate.ask"),
        "refused_is_double_ask": _g("gate.ask") > 0 and _g("funnel.refused") == 2 * _g("gate.ask"),
        # Independent token cross-check: the funnel's own meter against the adapter's.
        "funnel_input_tokens": int(_fn.get("judge.tokens.input") or 0),
        "funnel_output_tokens": int(_fn.get("judge.tokens.output") or 0),
        "tokens_agree_with_adapter": (
            int(_fn.get("judge.tokens.input") or 0) == provider.input_tokens
            and int(_fn.get("judge.tokens.output") or 0) == provider.output_tokens
        ),
    }
    out["stages"] = {"stage0_ingest": s0, "stage1_admission": s1, "stage2_perception": s2,
                     "stage3_judge": s3, "stage4_cognition": s4}
    out["funnel"] = dict(core.metrics.snapshot())
    out["health_kinds"] = dict(Counter(h["kind"] for h in health))
    out["incident_ts"] = inc_ts[:200]
    return out


# ---------------------------------------------------------------------------------------------
# corpus selection
# ---------------------------------------------------------------------------------------------


def stratified(cases: list[dict], n: int) -> list[dict]:
    """A deterministic sample that keeps every (dataset, goal_kind) cell represented.

    Round-robin over the cells in a fixed order, taking cases in corpus order.  A truncated prefix of
    the corpus would be a sample of `charades` and nothing else; this is the honest alternative when
    the full corpus does not fit the budget.
    """
    if n <= 0 or n >= len(cases):
        return list(cases)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cases:
        cells[(c["dataset"], c["goal_kind"])].append(c)
    keys = sorted(cells)
    picked: list[dict] = []
    i = 0
    while len(picked) < n:
        progressed = False
        for k in keys:
            if i < len(cells[k]):
                picked.append(cells[k][i])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        i += 1
    order = {c["case_id"]: j for j, c in enumerate(cases)}
    return sorted(picked, key=lambda c: order[c["case_id"]])


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------


def quality(rows: list[dict], lat: list[dict]) -> dict[str, Any]:
    """Occurrence precision/recall and TIMELY recall at the product gate.

    Timely recall is not recall with a filter bolted on: an Incident that lands inside the occurrence
    but 6 s after its onset is a correct answer that arrived too late to be a notification, and the
    pair (recall, timely recall) is the only way to see the difference.
    """
    gt_events = sum(r["stages"]["stage4_cognition"]["gt_occurrences"] for r in rows)
    matched = sum(r["stages"]["stage4_cognition"]["occurrences_matched"] for r in rows)
    incidents = sum(r["stages"]["stage4_cognition"]["incidents"] for r in rows)
    outside = sum(r["stages"]["stage4_cognition"]["incidents_outside_gt"] for r in rows)
    timely_occ = set()
    for row in lat:
        if row.get("matched_occurrence") and row.get("onset_to_incident_s") is not None:
            if row["onset_to_incident_s"] <= ACTION_TO_INCIDENT_GATE_S:
                timely_occ.add((row["case_id"], tuple(row["matched_occurrence"])))
    return {
        "gt_occurrences": gt_events,
        "occurrences_matched": matched,
        "occurrence_recall": (matched / gt_events) if gt_events else None,
        "incidents": incidents,
        "incidents_inside_gt": incidents - outside,
        "occurrence_precision": ((incidents - outside) / incidents) if incidents else None,
        "occurrences_timely_2s": len(timely_occ),
        "timely_recall_2s": (len(timely_occ) / gt_events) if gt_events else None,
    }


def accounting(rows: list[dict], prices: dict) -> dict[str, Any]:
    s3 = [r["stages"]["stage3_judge"] for r in rows]
    outcomes: Counter = Counter()
    for r in rows:
        outcomes.update(r["stages"]["stage3_judge"]["terminal_outcomes"])
    tin = sum(x["input_tokens"] for x in s3)
    tout = sum(x["output_tokens"] for x in s3)
    model = rows[0]["model"] if rows else None
    price = prices.get(str(model)) if model else None
    cost = None
    if price and price.get("input_usd_per_million") is not None:
        cost = round(tin / 1e6 * float(price["input_usd_per_million"])
                     + tout / 1e6 * float(price["output_usd_per_million"]), 6)
    return {
        "model": model,
        "logical_calls": sum(x["logical_calls"] for x in s3),
        "physical_attempts_adapter": sum(x["physical_attempts"] for x in s3),
        "route_attempts_receipted": sum(x["route_attempts"] for x in s3),
        "scheduler_batches": sum(x["judge_calls_batches"] for x in s3),
        "finished": sum(x["finished"] for x in s3),
        "degraded": sum(x["degraded"] for x in s3),
        "adapter_errors": sum(x["adapter_errors"] for x in s3),
        "adapter_cancelled": sum(x["adapter_cancelled"] for x in s3),
        "terminal_outcomes": dict(sorted(outcomes.items())),
        "failed_route_calls": sum(x["failed_route_calls"] for x in s3),
        "input_tokens": tin,
        "output_tokens": tout,
        "usage_rows": sum(x["usage_rows"] for x in s3),
        "input_bytes": sum(x["input_bytes"] for x in s3),
        "price_snapshot": price,
        "estimated_cost_usd": cost,
        "reflex_context_stamped": sum(x["reflex_context_stamped"] for x in s3),
        "verdicts_with_evidence_text": sum(x["verdicts_with_evidence_text"] for x in s3),
        "provider_latency": dist([v for r in rows for v in _flat(r, "completed")]),
        "provider_latency_censored": dist([v for r in rows for v in _flat(r, "censored")]),
        "mean_output_tokens_per_completed_call": (
            tout / sum(x["usage_rows"] for x in s3) if sum(x["usage_rows"] for x in s3) else None
        ),
    }


def _flat(row: dict, which: str) -> list[float]:
    """Per-call provider elapsed, reconstructed from the per-case distribution's raw list.

    Distributions are summarized per case, so the corpus-level percentile is assembled from the
    per-case raw values carried on the case record rather than by averaging per-case percentiles
    (which would be a percentile of percentiles and is not a percentile of anything).
    """
    return list(row.get("_provider_latency_raw", {}).get(which, []))


LATENCY_LEGS = (
    ("onset_to_incident_s", "action onset -> Incident (THE PRODUCT GATE)"),
    ("onset_to_ingest_s", "action onset -> frame ingested"),
    ("onset_to_admission_s", "action onset -> admitted by tier-0"),
    ("ingest_to_admission_s", "ingest -> admission"),
    ("admission_to_dispatch_s", "admission -> judge dispatch"),
    ("dispatch_to_verdict_s", "judge dispatch -> verdict"),
    ("verdict_to_decision_s", "verdict -> decision emitted"),
    ("decision_to_incident_s", "decision -> Incident observed"),
    ("ingest_to_incident_s", "ingest -> Incident"),
    ("admission_to_incident_s", "admission -> Incident"),
)


#: Legs anchored on the ACTION ONSET.  These answer "how soon after the action did X happen?", so
#: their unit is one GT OCCURRENCE -- not one Incident.  A long GT interval can emit dozens of
#: correct Incidents, and averaging over emissions weights that single occurrence dozens of times,
#: answering "how late were all the repeated messages?" instead of "how soon did we first alert?".
#: Measured on the n=100 v10_r1 artifact, that inflates the reported median by 4.4x-9.5x per leg.
ONSET_ANCHORED_LEGS = ("onset_to_incident_s", "onset_to_ingest_s", "onset_to_admission_s")

#: Everything else is genuinely per-emission or per-call (ingest->admission, dispatch->verdict, ...).
#: One judge call is one sample of provider latency, so those distributions are already correct and
#: are deliberately left weighted per emission.
PER_EMISSION_LEGS = tuple(k for k, _ in LATENCY_LEGS if k not in ONSET_ANCHORED_LEGS)

#: Schema marker so a downstream consumer can tell WHICH aggregation unit produced a number.
LATENCY_SCHEMA = "percept.latency/2-occurrence-anchored"


def occurrence_key(row: dict) -> tuple[str, tuple] | None:
    """The dedup identity ``quality()`` already uses -- ``(case_id, matched_occurrence)``.

    Sharing one key function is what keeps the latency denominator and the timely-recall numerator
    describing the same population.  Identical intervals in DIFFERENT cases stay distinct because
    ``case_id`` is part of the key.
    """
    occ = row.get("matched_occurrence")
    if not occ:
        return None
    return (row["case_id"], tuple(occ) if isinstance(occ, (list, tuple)) else (occ,))


def earliest_rows_per_occurrence(lat: Sequence[dict]) -> tuple[list[dict], int]:
    """One row per matched occurrence: the emission with the SMALLEST ``onset_to_incident_s``.

    Returns ``(rows, n_pre_onset_dropped)``.  Rows are never mutated -- the winners are the caller's
    own dict objects, returned by reference.

    The whole row wins, not each leg independently: ``onset_to_ingest`` and ``onset_to_admission``
    are read off the SAME emission that produced the earliest Incident, so the legs still decompose
    one coherent delivery.  Taking a per-leg minimum would mix emissions and produce a breakdown
    that does not describe any delivery that actually happened.

    A negative ``onset_to_incident_s`` means an Incident matched an occurrence it PRECEDES.  That is
    not an early alert, it is a matcher artifact, and taking ``min`` would silently promote it to
    best-case.  Such rows are dropped from the latency population and COUNTED, never silently kept.
    """
    best: dict[tuple, dict] = {}
    pre_onset = 0
    for row in lat:
        key = occurrence_key(row)
        if key is None:
            continue  # unmatched Incident: precision evidence, not latency evidence
        value = row.get("onset_to_incident_s")
        if value is None or not math.isfinite(float(value)):
            continue
        if float(value) < 0.0:
            pre_onset += 1
            continue
        incumbent = best.get(key)
        if incumbent is None or float(value) < float(incumbent["onset_to_incident_s"]):
            best[key] = row
    return list(best.values()), pre_onset


def latency_summary(lat: list[dict], rows: Sequence[dict] | None = None) -> dict[str, Any]:
    """Action-onset latency, aggregated PER OCCURRENCE; per-call legs left per emission.

    ``rows`` supplies the ground-truth denominator so a reader can see that this distribution is
    CONDITIONAL ON DETECTION -- occurrences we never found contribute no latency sample at all, and
    quoting a median without the miss count beside it overstates the product.

    Percentiles use ``_pct``, this module's LINEAR-INTERPOLATION convention.  ``percept.slo`` uses
    nearest-rank, so a figure here and a figure there are not directly comparable at the same
    quantile.  Stated rather than reconciled: changing either convention silently would break the
    other report's history.
    """
    earliest, pre_onset = earliest_rows_per_occurrence(lat)
    matched_keys = {occurrence_key(r) for r in lat if occurrence_key(r) is not None}

    gt_occurrences = None
    if rows is not None:
        gt_occurrences = sum(
            r["stages"]["stage4_cognition"]["gt_occurrences"]
            for r in rows if r.get("stages")
        )
    matched_occurrences = len(matched_keys)
    missed = (gt_occurrences - matched_occurrences) if gt_occurrences is not None else None

    onset = [float(r["onset_to_incident_s"]) for r in earliest]
    timely = sum(1 for x in onset if x <= ACTION_TO_INCIDENT_GATE_S)

    legs: dict[str, Any] = {}
    for key, _ in LATENCY_LEGS:
        source = earliest if key in ONSET_ANCHORED_LEGS else lat
        legs[key] = dist([r[key] for r in source if r.get(key) is not None])

    def by(field: str) -> dict[str, Any]:
        return {
            k: dist([r["onset_to_incident_s"] for r in earliest
                     if r.get(field) == k and r.get("onset_to_incident_s") is not None])
            for k in sorted({r.get(field) for r in earliest if r.get(field) is not None})
        }

    return {
        "schema": LATENCY_SCHEMA,
        "aggregation_unit": {
            leg: ("occurrence:earliest_incident" if leg in ONSET_ANCHORED_LEGS else "incident_emission")
            for leg, _ in LATENCY_LEGS
        },
        "methodology": (
            "Onset-anchored legs are aggregated per GT occurrence, taking the EARLIEST Incident that "
            "matched it -- 'how soon did we first alert?'. Legs downstream of ingest stay per "
            "emission because one judge call is one genuine sample. The distribution is conditional "
            "on detection: missed_occurrences contribute no sample, so read p50 next to "
            "timely_recall_all_gt, never alone. Percentiles are linearly interpolated (_pct)."
        ),
        "legs": legs,
        "gate_s": ACTION_TO_INCIDENT_GATE_S,
        "gt_occurrences": gt_occurrences,
        "matched_occurrences": matched_occurrences,
        "missed_occurrences": missed,
        "timely_occurrences": timely,
        "timely_recall_all_gt": (timely / gt_occurrences) if gt_occurrences else None,
        "timely_rate_matched_only": (timely / matched_occurrences) if matched_occurrences else None,
        "pre_onset_rows_dropped": pre_onset,
        "n_gated_incidents": len(onset),
        "n_under_gate": timely,
        "fraction_under_gate": (timely / len(onset)) if onset else None,
        "by_goal_kind": by("goal_kind"),
        "by_dataset": by("dataset"),
        #: AUDIT ONLY -- the per-emission distribution this report used to publish as product
        #: latency.  It answers "how late were all repeated messages for an occurrence?", which is a
        #: real diagnostic (delivery volume, merge-window behaviour) but is NOT the product gate.
        "raw_incident_emissions": {
            "aggregation_unit": "incident_emission",
            "note": "AUDIT ONLY. Not product latency -- weights one occurrence once per Incident.",
            "legs": {k: dist([r[k] for r in lat if r.get(k) is not None])
                     for k in ONSET_ANCHORED_LEGS},
            "n_matched_emissions": sum(1 for r in lat if occurrence_key(r) is not None),
        },
    }


def mode_report(rows: list[dict], prices: dict) -> dict[str, Any]:
    ok = [r for r in rows if r.get("measurable") and r.get("stages")]
    lat = [x for r in ok for x in r.get("latency_rows", [])]
    return {
        "n_cases_attempted": len(rows),
        "n_cases_measured": len(ok),
        "n_cases_errored": sum(1 for r in rows if r.get("error")),
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in rows if r.get("error")],
        "media_s": round(sum(r.get("media_end_s") or 0.0 for r in ok), 1),
        "wall_s": round(sum(r.get("wall_s") or 0.0 for r in rows), 1),
        "overall": {st: SM.pool(ok, st) for st in SM.STAGES},
        "by_goal_kind": {st: SM.breakdown(ok, st, "goal_kind") for st in SM.STAGES},
        "by_dataset": {st: SM.breakdown(ok, st, "dataset") for st in SM.STAGES},
        "quality": quality(ok, lat),
        "quality_by_goal_kind": {
            k: quality([r for r in ok if r["goal_kind"] == k],
                       [x for x in lat if x["goal_kind"] == k])
            for k in sorted({r["goal_kind"] for r in ok})
        },
        "quality_by_dataset": {
            k: quality([r for r in ok if r["dataset"] == k],
                       [x for x in lat if x["dataset"] == k])
            for k in sorted({r["dataset"] for r in ok})
        },
        "latency": latency_summary(lat, ok),
        "accounting": accounting(ok, prices),
        "pace_lag_p95_s": dist([r["pace_lag"]["p95"] for r in ok
                                if r.get("pace_lag", {}).get("p95") is not None]),
        "decoder_dropped_total": sum(int(r.get("decoder_dropped") or 0) for r in ok),
        "funnel_balance": {
            "cases_checked": sum(1 for r in ok if r.get("funnel_balance")),
            "cases_refused_equals_ask": sum(
                1 for r in ok if (r.get("funnel_balance") or {}).get("refused_equals_ask")),
            "cases_refused_is_double_ask": sum(
                1 for r in ok if (r.get("funnel_balance") or {}).get("refused_is_double_ask")),
            "cases_tokens_agree": sum(
                1 for r in ok if (r.get("funnel_balance") or {}).get("tokens_agree_with_adapter")),
            "judged": sum((r.get("funnel_balance") or {}).get("judged", 0) for r in ok),
            "fired": sum((r.get("funnel_balance") or {}).get("fired", 0) for r in ok),
            "refused": sum((r.get("funnel_balance") or {}).get("refused", 0) for r in ok),
            "ask": sum((r.get("funnel_balance") or {}).get("ask", 0) for r in ok),
        },
    }


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------

_f = SM._f


def _d(d: dict | None, k: str = "p50") -> str:
    if not d or d.get(k) is None:
        return "n/a"
    return f"{d[k]:.3f}"


def stage_table(payload_mode: dict) -> list[str]:
    L = ["| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | "
         "Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |",
         "|---|---|---|---|---|---|---|---|---|"]
    for st in SM.STAGES:
        p = payload_mode["overall"][st]
        if not p.get("n"):
            L.append(f"| {SM.STAGE_TITLES[st]} | - | - | - | - | - | - | - | - |")
            continue
        L.append(
            f"| {SM.STAGE_TITLES[st]} | {_f(p['recall_stage_local'])} | {_f(p['recall_cumulative'])} | "
            f"{_f(p.get('recall_interval_stage_local'))} | {_f(p.get('recall_interval_cumulative'))} | "
            f"{_f(p['precision'])} | {_f(p['economy'])} | {_f(p['drop_rate'])} | "
            f"{p['units_received']:,} -> {p['units_passed']:,} |"
        )
    return L


def _breakdown_table(rows: dict, key_label: str) -> list[str]:
    L = [f"| {key_label} | Recall (stage-local) | Recall (cumulative) | Event recall (local) | "
         "Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        if not r.get("n"):
            continue
        L.append(
            f"| {k} | {_f(r['recall_stage_local'])} | {_f(r['recall_cumulative'])} | "
            f"{_f(r.get('recall_interval_stage_local'))} | {_f(r.get('recall_interval_cumulative'))} | "
            f"{_f(r['precision'])} | {_f(r['economy'])} | {_f(r['drop_rate'])} | {r['n']} | "
            f"{r.get('gt_intervals', 0)} |"
        )
    return L


def render_md(payload: dict) -> str:
    L: list[str] = []
    A = L.append
    meta = payload["meta"]
    A("# Live-provider run: per-stage metrics and end-to-end latency")
    A("")
    A(f"Runtime under test: `percept` {meta['package_version']} built from `{meta['build_commit']}` "
      f"(contract {meta['contract_version']}), interpreter `{meta['interpreter']}`.")
    A(f"Provider: **{meta['backend']} / `{meta['model']}`**, transport `{meta['transport']}`, "
      f"evidence shape `video_clip` ({meta['temporal_frames']} frames over "
      f"{meta['temporal_window_s']} s).  Generated {meta['generated']}.")
    A("")
    A(f"**What was actually run:** {meta['n_selected']} of {meta['n_corpus']} corpus cases, "
      f"each in BOTH modes ({meta['selection']}).  "
      f"{meta['media_s']:.0f} s of media per mode, replayed in REAL TIME.  "
      f"Case concurrency {meta['concurrency']}.  Total wall {meta['total_wall_s']:.0f} s.")
    A("")
    A("Every case is driven through the real public path — `percept.run(...)` over a paced file "
      "source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a "
      "`JudgePool`. No stage is stubbed and no verdict is scripted.")
    A("")
    A("The two arms differ in exactly two declared ways: the pool's `mode` (which stamps "
      "`_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text "
      "`evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline "
      f"({ACTION_PROVIDER_DEADLINE_S} s reflex / {DELIBERATE_PROVIDER_DEADLINE_S} s deliberate). "
      "The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how "
      "often a 1024-token generation misses a bus, not what the bound costs in quality. The "
      "counterfactual is reported instead, from the per-call route receipts.")
    A("")

    for mode in payload["modes_run"]:
        m = payload["modes"][mode]
        A(f"## Per-stage metrics — `{mode}`")
        A("")
        A(f"{m['n_cases_measured']} of {m['n_cases_attempted']} cases measured; "
          f"{m['n_cases_errored']} errored (errors are counted, never dropped from a denominator).")
        A("")
        A("Recall appears TWICE at every stage. **stage-local** is versus what that stage actually "
          "received; **cumulative** is versus the original source ground truth. A stage at 1.000 "
          "stage-local lost nothing itself however low its cumulative figure is — the pair is what "
          "localises the loss, and a single number per stage cannot.")
        A("")
        L.extend(stage_table(m))
        A("")
        A("Two boundaries read differently here than they do in the scripted-oracle suite, and both "
          "differences are caused by the judge being REAL:")
        A("")
        A("- **Stage 2** carries the judge scheduler. With `judge_dispatch=\"async\"` the scheduler is "
          "single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every "
          "newly admitted frame but the freshest is superseded. Its economy is therefore the ratio "
          "of provider throughput to admission rate, not a perception loss — and every superseded "
          "unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is "
          "economy and not drop.")
        A("- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates "
          "it. The column that localises a judge regression is EVENT recall: did at least one unit "
          "inside each ground-truth occurrence survive to a completed verdict?")
        A("")

    A("## Per-stage breakdown by goal_kind")
    A("")
    A("`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.")
    A("")
    for mode in payload["modes_run"]:
        for st in SM.STAGES:
            rows = payload["modes"][mode]["by_goal_kind"].get(st) or {}
            if not any(r.get("n") for r in rows.values()):
                continue
            A(f"### `{mode}` — {SM.STAGE_TITLES[st]}")
            A("")
            L.extend(_breakdown_table(rows, "goal_kind"))
            A("")

    A("## Per-stage breakdown by dataset")
    A("")
    for mode in payload["modes_run"]:
        for st in SM.STAGES:
            rows = payload["modes"][mode]["by_dataset"].get(st) or {}
            if not any(r.get("n") for r in rows.values()):
                continue
            A(f"### `{mode}` — {SM.STAGE_TITLES[st]}")
            A("")
            L.extend(_breakdown_table(rows, "dataset"))
            A("")

    A("## Reflex vs deliberate — what the latency bound costs")
    A("")
    if len(payload["modes_run"]) == 2:
        r, d = payload["modes"]["reflex"], payload["modes"]["deliberate"]
        A("| Measure | reflex | deliberate | delta (reflex - deliberate) |")
        A("|---|---|---|---|")

        def cmp_row(label: str, a: Any, b: Any, nd: int = 3) -> None:
            delta = (f"{a - b:+.{nd}f}" if isinstance(a, (int, float)) and isinstance(b, (int, float))
                     else "n/a")
            A(f"| {label} | {_f(a, nd)} | {_f(b, nd)} | {delta} |")

        cmp_row("Occurrence recall", r["quality"]["occurrence_recall"], d["quality"]["occurrence_recall"])
        cmp_row("Occurrence precision", r["quality"]["occurrence_precision"],
                d["quality"]["occurrence_precision"])
        cmp_row("Timely recall @ 2 s", r["quality"]["timely_recall_2s"], d["quality"]["timely_recall_2s"])
        cmp_row("Incidents emitted", r["quality"]["incidents"], d["quality"]["incidents"], 0)
        cmp_row("Stage-4 event recall (cumulative)",
                r["overall"]["stage4_cognition"].get("recall_interval_cumulative"),
                d["overall"]["stage4_cognition"].get("recall_interval_cumulative"))
        cmp_row("Stage-3 event recall (stage-local)",
                r["overall"]["stage3_judge"].get("recall_interval_stage_local"),
                d["overall"]["stage3_judge"].get("recall_interval_stage_local"))
        cmp_row("action -> Incident p50 (s)", r["latency"]["legs"]["onset_to_incident_s"]["p50"],
                d["latency"]["legs"]["onset_to_incident_s"]["p50"])
        cmp_row("action -> Incident p95 (s)", r["latency"]["legs"]["onset_to_incident_s"]["p95"],
                d["latency"]["legs"]["onset_to_incident_s"]["p95"])
        cmp_row("provider call p50 (s)", r["accounting"]["provider_latency"]["p50"],
                d["accounting"]["provider_latency"]["p50"])
        cmp_row("provider call p95 (s)", r["accounting"]["provider_latency"]["p95"],
                d["accounting"]["provider_latency"]["p95"])
        cmp_row("logical judge calls", r["accounting"]["logical_calls"],
                d["accounting"]["logical_calls"], 0)
        cmp_row("output tokens", r["accounting"]["output_tokens"], d["accounting"]["output_tokens"], 0)
        cmp_row("input tokens", r["accounting"]["input_tokens"], d["accounting"]["input_tokens"], 0)
        A("")
        cf = payload.get("counterfactual") or {}
        if cf:
            A(f"**Counterfactual on the deliberate arm:** {cf['breaching']:,} of {cf['n']:,} "
              f"deliberate provider calls took longer than the reflex budget of "
              f"{ACTION_PROVIDER_DEADLINE_S} s ({_f(cf['fraction_breaching'])}). Those calls would "
              f"have returned a `deadline` receipt and an UNKNOWN verdict under the reflex "
              f"configuration, so the deliberate quality numbers above are NOT reachable at the "
              f"reflex deadline.")
            A("")
    else:
        A("Only one mode was run; no comparison is available.")
        A("")

    A("## End-to-end latency")
    A("")
    A("Every wall figure crosses the media clock exactly once, through the replay anchor recorded at "
      "the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. "
      "Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus "
      "the instant the `Incident` was observed by the consumer.")
    A("")
    for mode in payload["modes_run"]:
        lat = payload["modes"][mode]["latency"]
        A(f"### `{mode}`")
        A("")
        A("| Leg | n | p50 | p95 | max | mean |")
        A("|---|---|---|---|---|---|")
        for key, label in LATENCY_LEGS:
            dd = lat["legs"][key]
            A(f"| {label} | {dd['n']} | {_d(dd,'p50')} | {_d(dd,'p95')} | {_d(dd,'max')} | "
              f"{_d(dd,'mean')} |")
        A("")
        g = lat["fraction_under_gate"]
        A(f"**Product gate (action -> Incident < {lat['gate_s']} s): "
          f"{lat['n_under_gate']}/{lat['n_gated_incidents']} matched Incidents clear it "
          f"({_f(g)}).**")
        A("")
        if lat["by_goal_kind"]:
            A("| goal_kind | n | action->Incident p50 | p95 | max |")
            A("|---|---|---|---|---|")
            for k, dd in lat["by_goal_kind"].items():
                A(f"| {k} | {dd['n']} | {_d(dd,'p50')} | {_d(dd,'p95')} | {_d(dd,'max')} |")
            A("")

    A("## Quality by goal_kind")
    A("")
    A("`state` has n=4 cases in the full corpus and is reported, never gated.")
    A("")
    for mode in payload["modes_run"]:
        A(f"### `{mode}`")
        A("")
        A("| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |")
        A("|---|---|---|---|---|---|")
        for k, q in payload["modes"][mode]["quality_by_goal_kind"].items():
            A(f"| {k} | {q['gt_occurrences']} | {_f(q['occurrence_recall'])} | "
              f"{_f(q['occurrence_precision'])} | {_f(q['timely_recall_2s'])} | {q['incidents']} |")
        q = payload["modes"][mode]["quality"]
        A(f"| **all** | {q['gt_occurrences']} | {_f(q['occurrence_recall'])} | "
          f"{_f(q['occurrence_precision'])} | {_f(q['timely_recall_2s'])} | {q['incidents']} |")
        A("")

    A("## Coverage — every (dataset, goal_kind) cell that was actually measured")
    A("")
    A("An aggregate hides which category the pipeline fails on. This table is the receipt that every "
      "cell was reached; a cell short of its selected count is a gap, not a rounding error.")
    A("")
    A("| dataset / goal_kind | selected | " + " | ".join(f"`{m}` measured" for m in payload["modes_run"]) + " |")
    A("|---|---" + "|---" * len(payload["modes_run"]) + "|")
    for cell, row in payload["coverage"].items():
        A(f"| {cell} | {row['selected']} | "
          + " | ".join(str(row.get(m, 0)) for m in payload["modes_run"]) + " |")
    A("")

    A("## Quality by dataset")
    A("")
    for mode in payload["modes_run"]:
        A(f"### `{mode}`")
        A("")
        A("| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |")
        A("|---|---|---|---|---|---|")
        for k, q in payload["modes"][mode]["quality_by_dataset"].items():
            A(f"| {k} | {q['gt_occurrences']} | {_f(q['occurrence_recall'])} | "
              f"{_f(q['occurrence_precision'])} | {_f(q['timely_recall_2s'])} | {q['incidents']} |")
        A("")

    A("## Provider accounting")
    A("")
    A("| Measure | " + " | ".join(f"`{m}`" for m in payload["modes_run"]) + " |")
    A("|---" * (len(payload["modes_run"]) + 1) + "|")
    acc = {m: payload["modes"][m]["accounting"] for m in payload["modes_run"]}
    for label, key in (
        ("Logical judge calls (`judge.started`)", "logical_calls"),
        ("Physical attempts (adapter boundary)", "physical_attempts_adapter"),
        ("Route attempts (receipted)", "route_attempts_receipted"),
        ("Scheduler batches (`RuntimeStats.judge_calls`)", "scheduler_batches"),
        ("Verdicts completed", "finished"),
        ("Degraded (adapter raised)", "degraded"),
        ("Adapter-level exceptions", "adapter_errors"),
        ("Failed route calls (no usable answer)", "failed_route_calls"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Input bytes to provider", "input_bytes"),
        ("Reflex context stamped on call", "reflex_context_stamped"),
        ("Verdicts carrying evidence prose", "verdicts_with_evidence_text"),
        ("Estimated cost (USD)", "estimated_cost_usd"),
    ):
        cells = " | ".join(
            (f"{acc[m][key]:,}" if isinstance(acc[m][key], int)
             else (f"{acc[m][key]:.4f}" if isinstance(acc[m][key], float) else "not priced"))
            for m in payload["modes_run"]
        )
        A(f"| {label} | {cells} |")
    A("")
    A("Terminal outcome classes from the pool's own route receipts (this harness passes NO "
      "`terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the "
      "qualification bench raises on these, which is right for a gate and wrong for a measurement):")
    A("")
    all_out = sorted({o for m in payload["modes_run"] for o in acc[m]["terminal_outcomes"]})
    A("| Outcome | " + " | ".join(f"`{m}`" for m in payload["modes_run"]) + " |")
    A("|---" * (len(payload["modes_run"]) + 1) + "|")
    for o in all_out:
        A(f"| `{o}` | " + " | ".join(str(acc[m]["terminal_outcomes"].get(o, 0))
                                     for m in payload["modes_run"]) + " |")
    A("")
    price = next((acc[m]["price_snapshot"] for m in payload["modes_run"] if acc[m]["price_snapshot"]), None)
    if price:
        A(f"Cost is tokens x a DECLARED price, not a measurement: "
          f"${price['input_usd_per_million']}/M input, ${price['output_usd_per_million']}/M output "
          f"({price.get('source', 'source not recorded')}). Re-derive with `--price-in/--price-out`.")
    else:
        A("**No price was declared, so no cost is reported.** Token counts above are measured "
          "exactly; multiply them by a price you trust rather than trusting one invented here.")
    A("")

    A("## Provider latency, completed vs censored")
    A("")
    A("A call the pool CANCELS at its deadline has no measured provider latency — only a lower "
      "bound (the elapsed at cancellation). The two are reported separately; pooling them would "
      "manufacture a fast-looking distribution out of calls that never finished.")
    A("")
    A("| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |")
    A("|---|---|---|---|---|---|---|")
    for m in payload["modes_run"]:
        a = payload["modes"][m]["accounting"]
        c, z = a["provider_latency"], a["provider_latency_censored"]
        A(f"| `{m}` | {c['n']:,} | {_d(c,'p50')} | {_d(c,'p95')} | {_d(c,'max')} | "
          f"{a['adapter_cancelled']:,} | {_d(z,'p50')} |")
    A("")

    A("## Runtime bookkeeping under a workload that produces ASKs")
    A("")
    A("The scripted-oracle suite could not exercise this: its judge always answered with high "
      "confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite "
      "explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live "
      "judge produces ASKs in bulk, so the counter is exercised here for the first time.")
    A("")
    A("| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with "
      "`refused == 2 x ask` | cases where funnel tokens == adapter tokens |")
    A("|---|---|---|---|---|---|---|---|")
    for m in payload["modes_run"]:
        b = payload["modes"][m]["funnel_balance"]
        A(f"| `{m}` | {b['judged']:,} | {b['fired']:,} | {b['refused']:,} | {b['ask']:,} | "
          f"{b['cases_refused_equals_ask']}/{b['cases_checked']} | "
          f"{b['cases_refused_is_double_ask']}/{b['cases_checked']} | "
          f"{b['cases_tokens_agree']}/{b['cases_checked']} |")
    A("")
    A("`funnel.refused` is intended to count one refusal per refused judgment. Where it reads "
      "exactly twice the ASK count, it is double-incrementing on the ASK path — a bookkeeping "
      "defect in the runtime's own mirror, not a data-path defect: every per-stage number in this "
      "report is derived from the trace stream and `RuntimeStats`, both of which reconcile.")
    A("")

    tp = payload.get("transport_probe")
    if tp:
        A("## Control: which provider transport can clear the reflex budget")
        A("")
        A("The qualification bench selects the `native_video` route whenever the adapter advertises "
          "it, and this adapter does. That selection is a LATENCY decision, so it is measured here "
          "rather than inherited: the same clip, the same 5-frame event-chain ask, the same model, "
          "differing only in how the frames reach the provider. Isolated calls, no runtime "
          "attached.")
        A("")
        A("| Shape | n | p50 | p95 | max | input tokens/call |")
        A("|---|---|---|---|---|---|")
        for row in tp.get("rows", []):
            dd = row["latency"]
            A(f"| {row['label']} | {dd['n']} | {_d(dd,'p50')} | {_d(dd,'p95')} | {_d(dd,'max')} | "
              f"{row.get('input_tokens', 'n/a')} |")
        A("")
        A(tp.get("verdict", ""))
        A("")

    cc = payload.get("concurrency_control")
    if cc:
        A("## Control: how much of the measured latency is this harness's own concurrency")
        A("")
        A("Cases run concurrently, so some of the provider latency above could be contention rather "
          "than the provider. The same cases were re-run at concurrency 1 to bound that.")
        A("")
        A("| Arm | Completed calls | p50 | p95 | max | calls cancelled at deadline |")
        A("|---|---|---|---|---|---|")
        for row in cc.get("rows", []):
            dd = row["latency"]
            A(f"| {row['label']} | {dd['n']} | {_d(dd,'p50')} | {_d(dd,'p95')} | {_d(dd,'max')} | "
              f"{row.get('cancelled', 'n/a')} |")
        A("")
        A(cc.get("verdict", ""))
        A("")

    A("## Measurement integrity")
    A("")
    for item in payload["integrity"]:
        A(f"- {item}")
    A("")

    A("## What this run does NOT establish")
    A("")
    for item in payload["limits"]:
        A(f"- {item}")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------


async def run_mode(cases: list[dict], mode: str, *, backend: str, concurrency: int,
                   transport: str = "images", on_done=None, progress: bool = True) -> list[dict]:
    """Run every case in one mode at bounded concurrency, checkpointing each as it lands.

    ``on_done`` is invoked with the finished record BEFORE the next case is admitted to the
    semaphore, so a crash costs the case in flight and nothing already measured.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    done = [0]
    t0 = time.monotonic()
    # Longest-first: a real-time replay costs its own media duration, so submitting a 360 s clip
    # last would leave one worker running alone for six minutes at the end of the mode. Results are
    # re-ordered to corpus order by the caller, so this changes only the schedule.
    cases = sorted(cases, key=lambda c: -(c.get("duration_s") or 0.0))

    async def one(case: dict) -> dict:
        async with sem:
            try:
                r = await run_case(case, mode, backend=backend, transport=transport)
            except BaseException as exc:  # a crashed case is DATA, never a silent hole
                r = {"case_id": case["case_id"], "dataset": case["dataset"],
                     "goal_kind": case["goal_kind"], "mode": mode, "measurable": False,
                     "error": f"{type(exc).__name__}: {exc}", "notes": [], "latency_rows": [],
                     "receipts": []}
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            done[0] += 1
            if on_done is not None:
                on_done(mode, r)
            if progress:
                print(f"[{mode}] {done[0]}/{len(cases)} {case['case_id']} "
                      f"{'ERR ' + str(r.get('error'))[:70] if r.get('error') else 'ok'} "
                      f"({time.monotonic() - t0:.0f}s)", file=sys.stderr, flush=True)
            return r

    return list(await asyncio.gather(*(one(c) for c in cases)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live-provider per-stage + latency run")
    ap.add_argument("--limit", type=int, default=0, help="stratified sample size (0 = full corpus)")
    ap.add_argument("--cases", default="", help="comma-separated case_ids to run instead")
    ap.add_argument("--modes", default="reflex,deliberate")
    ap.add_argument("--backend", default=os.environ.get("PERCEPT_VISION_BACKEND", "gemini"))
    ap.add_argument("--transport", default="images", choices=("auto", "images", "native"),
                    help="'auto' = the qualification bench's own route selection (native video "
                         "whenever the adapter advertises it); 'images' = ordered timestamped "
                         "images; 'native' = force the mp4 clip route")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--price-source", default="")
    ap.add_argument("--out-stem", default="live_run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--expect-commit", default="",
                    help="abort unless build_info().git_commit starts with this — the arm's own "
                         "proof it is the build it claims to be")
    ap.add_argument("--expect-conductor", action="store_true",
                    help="abort unless a frozen conductor pin is configured")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="reuse cases already in the checkpoint (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--sidecar", default="",
                    help="JSON file of control measurements to fold into the report "
                         "(keys: transport_probe, concurrency_control). Re-render a finished run "
                         "with --resume and no new provider spend.")
    args = ap.parse_args(argv)

    # ---- ARM SELF-CHECK ----------------------------------------------------------------------
    # An arm that cannot prove it is configured correctly must ABORT, not run: a silently
    # unconfigured arm would look identical to the other one and manufacture a null result.
    if args.expect_commit:
        actual = str(build_info().git_commit or "")
        if not actual.startswith(args.expect_commit):
            raise SystemExit(
                f"[live_run] ARM ABORTED: expected build {args.expect_commit}..., got "
                f"{actual or '<none>'} from {sys.executable}"
            )
        print(f"[live_run] build verified: {actual}", file=sys.stderr)
    if args.expect_conductor and not ACTION_CONDUCTOR_PIN:
        raise SystemExit("[live_run] ARM ABORTED: --expect-conductor but PERCEPT_SV_CONDUCTOR unset")
    if ACTION_CONDUCTOR_PIN:
        pin = _authoring_pin()
        print(f"[live_run] conductor pin: {len(pin)} frozen decisions from {ACTION_CONDUCTOR_PIN}",
              file=sys.stderr)

    if ACTION_MEDIA_RESOLUTION:
        applied = install_media_resolution_override(ACTION_MEDIA_RESOLUTION)
        print(f"[live_run] media_resolution pinned to {applied} (harness-local monkeypatch; "
              f"percept/ untouched)", flush=True)
    if ACTION_MAX_DIMENSION:
        print(f"[live_run] temporal frames downscaled to max_dimension={ACTION_MAX_DIMENSION}",
              flush=True)

    corpus = SM.load_corpus()
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected = [c for c in corpus if c["case_id"] in want]
        selection = f"explicit case list, n={len(selected)}"
    elif args.limit:
        selected = stratified(corpus, args.limit)
        selection = (f"deterministic stratified sample over all (dataset, goal_kind) cells, "
                     f"n={len(selected)}")
    else:
        selected = list(corpus)
        selection = "full corpus"

    settings = load_settings()
    model = settings.models.vision_model
    prices: dict[str, dict] = {}
    if args.price_in is not None and args.price_out is not None:
        prices[str(model)] = {
            "input_usd_per_million": args.price_in,
            "output_usd_per_million": args.price_out,
            "source": args.price_source or "declared on the command line",
        }

    modes_run = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes_run:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}")

    print(f"[live_run] backend={args.backend} model={model} transport={args.transport} "
          f"cases={len(selected)} modes={modes_run} concurrency={args.concurrency}",
          file=sys.stderr)

    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / f"{args.out_stem}.checkpoint.jsonl"

    # ---- resume ---------------------------------------------------------------------------
    # A case already measured is NOT re-run: it costs money and it would change nothing. The
    # checkpoint is append-only and keyed on (mode, case_id); a later line wins, so a partially
    # written tail line is survivable.
    raw: dict[str, dict[str, dict]] = {m: {} for m in modes_run}
    if args.resume and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = rec.get("mode")
            if m in raw and rec.get("case_id"):
                raw[m][rec["case_id"]] = rec
        print(f"[live_run] resumed: "
              + ", ".join(f"{m}={len(raw[m])}" for m in modes_run), file=sys.stderr)

    ck_fh = ckpt.open("a")
    t_all = time.monotonic()
    written = [0]

    def flush_report() -> None:
        try:
            partial = build_payload(
                {m: [raw[m][c["case_id"]] for c in selected if c["case_id"] in raw[m]]
                 for m in modes_run},
                corpus=corpus, selected=selected, selection=selection, modes_run=modes_run,
                prices=prices, args=args, model=model,
                total_wall=time.monotonic() - t_all, complete=False,
            )
            (RESULTS / f"{args.out_stem}.json").write_text(json.dumps(partial, indent=1, default=str))
            (RESULTS / f"{args.out_stem}.md").write_text(render_md(partial))
        except Exception as exc:  # a partial-report failure must never kill the measurement
            print(f"[live_run] partial report failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def on_done(mode: str, rec: dict) -> None:
        raw[mode][rec["case_id"]] = rec
        ck_fh.write(json.dumps(rec, default=str) + "\n")
        ck_fh.flush()
        os.fsync(ck_fh.fileno())
        written[0] += 1
        if written[0] % 5 == 0:
            flush_report()

    try:
        for m in modes_run:
            todo = [c for c in selected if c["case_id"] not in raw[m]]
            if not todo:
                print(f"[live_run] {m}: all {len(selected)} cases already checkpointed",
                      file=sys.stderr)
                continue
            asyncio.run(run_mode(todo, m, backend=args.backend, transport=args.transport,
                                 concurrency=args.concurrency, on_done=on_done))
    finally:
        ck_fh.close()
    total_wall = time.monotonic() - t_all

    ordered = {m: [raw[m][c["case_id"]] for c in selected if c["case_id"] in raw[m]]
               for m in modes_run}
    payload = build_payload(ordered, corpus=corpus, selected=selected, selection=selection,
                            modes_run=modes_run, prices=prices, args=args, model=model,
                            total_wall=total_wall, complete=True)
    (RESULTS / f"{args.out_stem}.json").write_text(json.dumps(payload, indent=1, default=str))
    (RESULTS / f"{args.out_stem}.md").write_text(render_md(payload))
    print(f"[live_run] wrote {RESULTS / (args.out_stem + '.json')}", file=sys.stderr)
    print(f"[live_run] wrote {RESULTS / (args.out_stem + '.md')}", file=sys.stderr)
    return 0


def build_payload(raw: dict[str, list[dict]], *, corpus: list[dict], selected: list[dict],
                  selection: str, modes_run: list[str], prices: dict, args: Any, model: str,
                  total_wall: float, complete: bool) -> dict[str, Any]:
    modes = {m: mode_report(raw.get(m) or [], prices) for m in modes_run}

    counterfactual = None
    if "deliberate" in modes:
        lats = [x for row in (raw.get("deliberate") or [])
                for x in (row.get("_provider_latency_raw") or {}).get("completed", [])]
        if lats:
            breach = sum(1 for x in lats if x > ACTION_PROVIDER_DEADLINE_S)
            counterfactual = {"n": len(lats), "breaching": breach,
                              "fraction_breaching": breach / len(lats),
                              "budget_s": ACTION_PROVIDER_DEADLINE_S}

    bi = build_info()
    first = next((r for m in modes_run for r in (raw.get(m) or []) if r.get("model")), {})
    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tag": args.tag,
            "package_version": bi.package_version,
            "build_commit": bi.git_commit,
            "contract_version": bi.contract_version,
            "interpreter": sys.executable,
            "backend": args.backend,
            "model": first.get("model", model),
            "transport": first.get("transport"),
            "transport_selection": args.transport,
            "temporal_frames": ACTION_TEMPORAL_FRAMES,
            "temporal_window_s": ACTION_TEMPORAL_WINDOW_S,
            # Declared, never inferred: a run whose evidence was shrunk must say so in its own meta.
            "max_dimension": ACTION_MAX_DIMENSION or "runtime default (768)",
            "media_resolution": ACTION_MEDIA_RESOLUTION or "provider default (HIGH)",
            "conductor_pin": ACTION_CONDUCTOR_PIN or "none (corpus-declared fixed watch)",
            "authored_types": dict(Counter(
                r.get("authored_type") or "none"
                for m in modes_run for r in (raw.get(m) or []))),
            "n_corpus": len(corpus),
            "n_selected": len(selected),
            "selection": selection,
            "selected_case_ids": [c["case_id"] for c in selected],
            "concurrency": args.concurrency,
            "modes": modes_run,
            "media_s": sum(modes[m]["media_s"] for m in modes_run) / max(1, len(modes_run)),
            "total_wall_s": round(total_wall, 1),
            "provider_deadline_s": {"reflex": ACTION_PROVIDER_DEADLINE_S,
                                    "deliberate": DELIBERATE_PROVIDER_DEADLINE_S},
            "decision_deadline_s": {"reflex": ACTION_DECISION_DEADLINE_S,
                                    "deliberate": DELIBERATE_DECISION_DEADLINE_S},
            "action_to_incident_gate_s": ACTION_TO_INCIDENT_GATE_S,
        },
        "modes_run": modes_run,
        "modes": modes,
        "counterfactual": counterfactual,
        "coverage": _coverage(selected, raw, modes_run),
        "metric_definitions": {
            "source": "eval/stage-validation/stage_metrics.py (imported, not reimplemented)",
            "recall_stage_local": "positive source time passed / positive source time received",
            "recall_cumulative": "positive source time passed / positive source time in the ORIGINAL source",
            "recall_interval_*": "same pair computed on ground-truth OCCURRENCES rather than time",
            "precision": "positive source time passed / total time passed",
            "economy": "units_passed / units_received",
            "drop_rate": "units lost WITHOUT a receipt / units received; deliberate suppression is NOT a drop",
            "occurrence_recall": "GT occurrences containing >=1 Incident / GT occurrences",
            "timely_recall_2s": "GT occurrences whose Incident arrived <2 s after ONSET / GT occurrences",
        },
        "integrity": [
            f"Runtime verified at import: `build_info().git_commit == {bi.git_commit}` — this is the "
            f"wheel that carries the reflex output budget, not the pre-change build.",
            "Reflex bounding verified at the ADAPTER boundary, not assumed: the harness counts how "
            "many provider calls arrived with `_percept_latency_mode='reflex'` stamped on the "
            "context, and how many verdicts carried evidence prose. See the accounting table.",
            "Replay is PACED to the media clock with a lossless bounded queue "
            "(`queue_policy='block'`); the per-frame pacing lag is measured and reported, because a "
            "runtime that falls behind the source clock would understate every latency below.",
            "Exactly one clock crossing exists in this file: the replay anchor recorded at the first "
            "decoded frame. All other latency arithmetic is monotonic-to-monotonic.",
            "A case that errors is reported and counted; it is never dropped from a denominator.",
            "Suppression is not loss: a tier-0 rejection emits its own `edge.decision` receipt, and a "
            "superseded judge job increments a named `RuntimeStats` counter. Only units that vanish "
            "with NO receipt enter `drop_rate`.",
            "The pool is given NO `terminal_outcomes`, so `deadline`/`capacity`/`provider_error` are "
            "recorded as receipts and counted rather than raised.",
        ],
        "limits": [
            "A 0.95 recall bar is not statistically ratifiable on this corpus. An exact one-sided "
            "Clopper-Pearson lower bound clearing 0.95 with zero misses needs n=59 positives; the "
            "duration and transition cells carry 38-40 occurrences in the FULL corpus and fewer in "
            "any subset. No 0.95 claim is made or implied here.",
            "`state` carries n=4 cases corpus-wide. It is reported and never gated.",
            "Stage 2 is an identity pass-through with no `EvidenceExtractor` armed: its economy of "
            "1.000 cannot distinguish 'lossless' from 'absent'.",
            "Judge quality here is one provider, one model, one prompt shape, on one day. It is a "
            "measurement of THIS route, not a general statement about the runtime's ceiling.",
            "Cost is tokens x a declared price. The tokens are measured; the price is an input.",
            "Cases run concurrently share one event loop and one machine. The concurrency level is "
            "recorded with every number; a latency figure measured at concurrency N is not "
            "automatically the figure at concurrency 1.",
        ],
        "cases": {m: [{k: v for k, v in r.items() if k not in {"receipts"}}
                      for r in (raw.get(m) or [])] for m in modes_run},
    }
    sidecar_path = getattr(args, "sidecar", "")
    if sidecar_path and os.path.exists(sidecar_path):
        payload.update(json.loads(Path(sidecar_path).read_text()))
    payload["meta"]["complete"] = complete
    payload["meta"]["cases_completed"] = {m: len(raw.get(m) or []) for m in modes_run}
    if not complete:
        payload["meta"]["selection"] = (
            f"{selection} — PARTIAL SNAPSHOT, run still in flight ("
            + ", ".join(f"{m}: {len(raw.get(m) or [])}/{len(selected)}" for m in modes_run) + ")"
        )
    return payload


def _coverage(selected: list[dict], raw: dict[str, list[dict]], modes_run: list[str]) -> dict:
    want = Counter((c["dataset"], c["goal_kind"]) for c in selected)
    got = {m: Counter((r["dataset"], r["goal_kind"]) for r in (raw.get(m) or [])
                      if r.get("measurable")) for m in modes_run}
    return {f"{ds}/{gk}": {"selected": n, **{m: got[m].get((ds, gk), 0) for m in modes_run}}
            for (ds, gk), n in sorted(want.items())}


if __name__ == "__main__":
    raise SystemExit(main())
